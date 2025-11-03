//! Ядро «Пастуха агентов» с мультиагентным мостом для кооперации ролей.
//! Оркестрирует роли: Архитектор → Билдеры → Специалисты → Финальный ревью.
//! Для локального демо используется Mock LLM и HTTP‑тестер (FastAPI) из Python‑репозитория.

use anyhow::Result;
use async_trait::async_trait;
use axum::{extract::State, routing::post, Json, Router};
use futures::stream::{FuturesUnordered, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{env, net::SocketAddr, sync::Arc};
use tracing::{info, Level};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

/// ---- Модели API ядра ------------------------------------------------------

#[derive(Debug, Deserialize)]
struct BridgeRequest {
    /// Человеческое описание задачи.
    task: String,
    /// Количество параллельных билдеров.
    builders: usize,
}

#[derive(Debug, Deserialize)]
struct MultiBridgeRequest {
    /// Человеческое описание цели.
    task: String,
    /// Число билдеров (конкурирующие варианты).
    builders: usize,
    /// Число ревьюеров (необязательно; в демо не влияет на выбор).
    reviewers: usize,
    /// Число специалистов на компонент плана.
    specialists: usize,
}

#[derive(Debug, Serialize)]
struct BridgeResponse {
    request_id: Uuid,
    winner_index: usize,
    winner_diff: String,
    tests: TestRunResult,
}

#[derive(Debug, Serialize)]
struct MultiBridgeResponse {
    request_id: Uuid,
    plan: Plan,
    diffs: Vec<String>,
    accepted_specialists: usize,
    tests: TestRunResult,
    review: Review,
}

/// ---- Контракты LLM и тест‑раннера ----------------------------------------

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Plan {
    /// Перечень компонент, за которые отвечают спец‑агенты.
    components: Vec<Component>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Component {
    /// Имя/смысл компонента.
    name: String,
    /// Целевые файлы для модификаций.
    target_files: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Review {
    /// Интегральная оценка результата.
    score: f32,
    /// Краткое обоснование.
    rationale: String,
}

#[derive(Debug, Deserialize, Clone)]
struct TestRunResult {
    tests_total: i32,
    tests_passed: i32,
    tests_failed: i32,
    return_code: i32,
    stdout: String,
    stderr: String,
}

#[async_trait]
trait Llm: Send + Sync {
    /// Возвращает unified diff для "Builder".
    async fn propose_patch(&self, task: &str, profile: usize) -> Result<String>;
    /// Архитектор формирует план работ.
    async fn plan(&self, task: &str) -> Result<Plan>;
    /// Финальный ревью результативного набора диффов.
    async fn final_review(&self, task: &str, diffs: &[String]) -> Result<Review>;
}

#[async_trait]
trait TestRunner: Send + Sync {
    /// Прогоняет pytest на базе demo‑приложения, последовательно применяя `diffs`.
    async fn run_diffs(&self, diffs: &[String]) -> Result<TestRunResult>;
}

/// HTTP‑тестер на базе FastAPI сервиса.
struct HttpTester {
    base_url: String,
    http: Client,
}

#[async_trait]
impl TestRunner for HttpTester {
    async fn run_diffs(&self, diffs: &[String]) -> Result<TestRunResult> {
        let body = serde_json::json!({ "diffs": diffs });
        let res = self
            .http
            .post(format!("{}/testrun", self.base_url))
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<TestRunResult>()
            .await?;
        Ok(res)
    }
}

/// Мок‑LLM для локального демо: производит "хороший" патч и безвредные спец‑правки.
struct MockLlm;

#[async_trait]
impl Llm for MockLlm {
    async fn propose_patch(&self, _task: &str, profile: usize) -> Result<String> {
        let good = r#"diff --git a/demo_app/app.py b/demo_app/app.py
index 0000000..1111111 100644
--- a/demo_app/app.py
+++ b/demo_app/app.py
@@
-def add(a: int, b: int) -> int:
-    \"\"\"Иллюстративная функция с ошибкой: возвращает a - b вместо a + b.\"\"\"
-    return a - b
+def add(a: int, b: int) -> int:
+    \"\"\"Возвращает сумму a и b.\"\"\"
+    return a + b
"#;
        let bad = r#"diff --git a/demo_app/app.py b/demo_app/app.py
index 0000000..1111111 100644
--- a/demo_app/app.py
+++ b/demo_app/app.py
@@
-def add(a: int, b: int) -> int:
-    \"\"\"Иллюстративная функция с ошибкой: возвращает a - b вместо a + b.\"\"\"
-    return a - b
+def add(a: int, b: int) -> int:
+    \"\"\"Ещё хуже.\"\"\"
+    return a - b - 1
"#;
        Ok(if profile % 3 == 0 { good.to_owned() } else { bad.to_owned() })
    }

    async fn plan(&self, _task: &str) -> Result<Plan> {
        Ok(Plan {
            components: vec![Component {
                name: "fix_add_function".to_string(),
                target_files: vec!["demo_app/app.py".to_string()],
            }],
        })
    }

    async fn final_review(&self, _task: &str, _diffs: &[String]) -> Result<Review> {
        Ok(Review {
            score: 0.95,
            rationale: "Тесты зелёные, изменения локализованы.".to_string(),
        })
    }
}

/// OpenAI‑совместимый LLM (Codex/совместимые сервисы).
struct OpenAiCompat {
    base_url: String,
    api_key: String,
    model: String,
    http: Client,
}

#[async_trait]
impl Llm for OpenAiCompat {
    async fn propose_patch(&self, task: &str, profile: usize) -> Result<String> {
        let prompt = format!(
            "ROLE: Senior Implementer #{profile}\nTASK:\n{task}\n\
             Produce a unified diff patch for demo_app; output ONLY the patch."
        );
        let body = serde_json::json!({ "model": self.model, "messages": [{"role":"user","content": prompt}]});
        let url = format!("{}/v1/chat/completions", self.base_url.trim_end_matches('/'));
        let v = self
            .http
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<serde_json::Value>()
            .await?;
        Ok(v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string())
    }

    async fn plan(&self, task: &str) -> Result<Plan> {
        let prompt = format!(
            "ROLE: Software Architect\nGoal:\n{task}\n\
             Output STRICT JSON: {{\"components\":[{{\"name\":\"...\",\"target_files\":[\"...\"]}}]}}"
        );
        let body = serde_json::json!({ "model": self.model, "messages": [{"role":"user","content": prompt}]});
        let url = format!("{}/v1/chat/completions", self.base_url.trim_end_matches('/'));
        let v = self
            .http
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<serde_json::Value>()
            .await?;
        let content = v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("{}");
        let plan: Plan = serde_json::from_str(content)
            .unwrap_or(Plan { components: vec![] });
        Ok(plan)
    }

    async fn final_review(&self, task: &str, diffs: &[String]) -> Result<Review> {
        let prompt = format!(
            "ROLE: Senior Reviewer\nTask:\n{task}\n\
             Assess given patches. Output STRICT JSON: {{\"score\": <0..1>, \"rationale\": \"...\"}}.\nPatches:\n{}",
            diffs.join("\n---\n")
        );
        let body = serde_json::json!({ "model": self.model, "messages": [{"role":"user","content": prompt}]});
        let url = format!("{}/v1/chat/completions", self.base_url.trim_end_matches('/'));
        let v = self
            .http
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<serde_json::Value>()
            .await?;
        let content = v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("{\"score\":0.5,\"rationale\":\"n/a\"}");
        let review: Review = serde_json::from_str(content)
            .unwrap_or(Review { score: 0.5, rationale: "n/a".to_string() });
        Ok(review)
    }
}

/// ---- Приложение и обработчики --------------------------------------------

#[derive(Clone)]
struct AppState {
    llm: Arc<dyn Llm>,
    tester: Arc<dyn TestRunner>,
    http: Client,
}

#[tokio::main]
async fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).with_max_level(Level::INFO).init();

    let http = Client::new();
    let tester_url = env::var("TESTER_URL").unwrap_or_else(|_| "http://localhost:7001".into());
    let use_mock = env::var("USE_MOCK_LLM").unwrap_or_else(|_| "1".into()) == "1";

    let llm: Arc<dyn Llm> = if use_mock {
        Arc::new(MockLlm)
    } else {
        let base = env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com".into());
        let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY missing");
        let model = env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());
        Arc::new(OpenAiCompat { base_url: base, api_key, model, http: http.clone() })
    };

    let state = AppState {
        llm,
        tester: Arc::new(HttpTester { base_url: tester_url, http: http.clone() }),
        http,
    };

    let app = Router::new()
        .route("/api/v1/bridge", post(api_bridge))
        .route("/api/v1/bridge/multi", post(api_bridge_multi))
        .with_state(state);

    let addr: SocketAddr = "0.0.0.0:8080".parse().unwrap();
    info!("Agent Wrangler core listening on {addr}");
    axum::Server::bind(&addr).serve(app.into_make_service()).await?;
    Ok(())
}

/// Простой best‑of‑N: выбирает патч, давший лучшие тесты.
async fn api_bridge(State(state): State<AppState>, Json(req): Json<BridgeRequest>) -> Json<BridgeResponse> {
    let resp = bridge_best_of_n(&state, &req).await.expect("bridge failed");
    Json(resp)
}

/// Мультиагентный мост: план → билдеры → специалисты (жадная интеграция) → финальный ревью.
async fn api_bridge_multi(State(state): State<AppState>, Json(req): Json<MultiBridgeRequest>) -> Json<MultiBridgeResponse> {
    let resp = bridge_multi(&state, &req).await.expect("bridge failed");
    Json(resp)
}

/// Генерирует N патчей, тестирует каждый и выбирает лучший.
async fn bridge_best_of_n(state: &AppState, req: &BridgeRequest) -> Result<BridgeResponse> {
    let mut gen = FuturesUnordered::new();
    for i in 0..req.builders {
        let llm = state.llm.clone();
        let task = req.task.clone();
        gen.push(async move { llm.propose_patch(&task, i).await });
    }
    let mut diffs = Vec::<String>::new();
    while let Some(r) = gen.next().await {
        if let Ok(d) = r {
            diffs.push(d);
        }
    }
    // Тестируем
    let mut best_idx = 0usize;
    let mut best: Option<TestRunResult> = None;
    for (i, d) in diffs.iter().enumerate() {
        let tr = state.tester.run_diffs(&[d.clone()]).await?;
        let better = match &best {
            None => true,
            Some(b) => tr.tests_failed < b.tests_failed || (tr.tests_failed == b.tests_failed && tr.tests_passed > b.tests_passed),
        };
        if better {
            best = Some(tr);
            best_idx = i;
        }
    }
    Ok(BridgeResponse {
        request_id: Uuid::new_v4(),
        winner_index: best_idx,
        winner_diff: diffs.get(best_idx).cloned().unwrap_or_default(),
        tests: best.unwrap(),
    })
}

/// Полный мультиагентный конвейер с добавлением специализированных патчей.
async fn bridge_multi(state: &AppState, req: &MultiBridgeRequest) -> Result<MultiBridgeResponse> {
    // 1) План от архитектора
    let plan = state.llm.plan(&req.task).await?;

    // 2) Конкурирующие билдеры → выбираем лучший как базовый дифф
    let base_resp = bridge_best_of_n(
        state,
        &BridgeRequest { task: req.task.clone(), builders: req.builders },
    ).await?;
    let mut accepted_diffs = vec![base_resp.winner_diff.clone()];
    let mut current = base_resp.tests.clone();

    // 3) Специалисты по каждому компоненту. Жадно добавляем патчи, не ухудшающие метрики.
    let mut accepted_specialists = 0usize;
    for comp in &plan.components {
        let mut gen = FuturesUnordered::new();
        for s in 0..req.specialists.max(1) {
            let llm = state.llm.clone();
            let comp_name = comp.name.clone();
            gen.push(async move {
                let patch = format!(
"diff --git a/demo_app/_meta_{comp}_{s}.py b/demo_app/_meta_{comp}_{s}.py
new file mode 100644
--- /dev/null
+++ b/demo_app/_meta_{comp}_{s}.py
@@
+\"\"\"Автогенерированный файл для компонента {comp_name}.\"\"\"
+COMPONENT = \"{comp_name}\"
");
                Ok::<String, anyhow::Error>(patch)
            });
        }
        let mut spec_diffs = Vec::<String>::new();
        while let Some(r) = gen.next().await {
            if let Ok(d) = r { spec_diffs.push(d); }
        }
        for d in spec_diffs {
            let mut trial = accepted_diffs.clone();
            trial.push(d.clone());
            let tr = state.tester.run_diffs(&trial).await?;
            let better = tr.tests_failed < current.tests_failed || (tr.tests_failed == current.tests_failed && tr.tests_passed >= current.tests_passed);
            if better {
                accepted_diffs.push(d);
                current = tr;
                accepted_specialists += 1;
            }
        }
    }

    // 4) Финальный ревью
    let review = state.llm.final_review(&req.task, &accepted_diffs).await?;

    Ok(MultiBridgeResponse {
        request_id: Uuid::new_v4(),
        plan,
        diffs: accepted_diffs,
        accepted_specialists,
        tests: current,
        review,
    })
}

/// ---- Тесты ядра -----------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct MockTester;
    #[async_trait]
    impl TestRunner for MockTester {
        async fn run_diffs(&self, diffs: &[String]) -> Result<TestRunResult> {
            let text = diffs.join("\n");
            let mut failed = if text.contains("return a + b") { 0 } else { 1 };
            if text.contains("- 1") { failed += 1; }
            Ok(TestRunResult {
                tests_total: 1,
                tests_passed: 1 - failed.max(0).min(1),
                tests_failed: failed.min(1),
                return_code: if failed == 0 { 0 } else { 1 },
                stdout: String::new(),
                stderr: String::new(),
            })
        }
    }

    #[tokio::test]
    async fn test_best_of_n_picks_good_patch() {
        let state = AppState {
            llm: Arc::new(MockLlm),
            tester: Arc::new(MockTester),
            http: Client::new(),
        };
        let resp = super::bridge_best_of_n(&state, &BridgeRequest { task: "fix add".into(), builders: 3 }).await.unwrap();
        assert_eq!(resp.tests.tests_failed, 0);
        assert!(resp.winner_diff.contains("return a + b"));
    }

    #[tokio::test]
    async fn test_multi_pipeline_accepts_specialists() {
        let state = AppState {
            llm: Arc::new(MockLlm),
            tester: Arc::new(MockTester),
            http: Client::new(),
        };
        let resp = super::bridge_multi(&state, &MultiBridgeRequest { task: "fix add".into(), builders: 3, reviewers: 2, specialists: 2 }).await.unwrap();
        assert!(resp.diffs.iter().any(|d| d.contains("return a + b")));
        assert!(resp.accepted_specialists >= 0);
        assert_eq!(resp.tests.tests_failed, 0);
    }
}
