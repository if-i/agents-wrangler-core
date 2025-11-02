//! HTTP‑ядро «Пастуха агентов»: мост между N билдер‑агентами Codex и тест‑сервисом.
//! Для локального запуска по умолчанию используется Mock LLM, не требующий ключей.
//! При желании можно переключиться на OpenAI‑совместимый API, выставив переменные
//! OPENAI_API_KEY/OPENAI_BASE_URL/OPENAI_MODEL и USE_MOCK_LLM=0.

use anyhow::Result;
use async_trait::async_trait;
use axum::{
    extract::State,
    routing::post,
    Json, Router,
};
use futures::stream::{FuturesUnordered, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{env, net::SocketAddr, sync::Arc};
use tracing::{info, Level};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

/// Запрос на запуск моста: человеческая формулировка задачи и число параллельных билдеров.
#[derive(Debug, Deserialize)]
struct BridgeRequest {
    task: String,
    builders: usize,
}

/// Результаты тестового прогона от tester‑service.
#[derive(Debug, Deserialize)]
struct TestRunResult {
    tests_total: i32,
    tests_passed: i32,
    tests_failed: i32,
    return_code: i32,
    stdout: String,
    stderr: String,
}

/// Ответ ядра с агрегированным результатом и победившим патчем.
#[derive(Debug, Serialize)]
struct BridgeResponse {
    request_id: Uuid,
    winner_index: usize,
    winner_diff: String,
    tests: TestRunResult,
}

/// Абстракция для LLM, чтобы можно было подменять мок/реальную модель.
#[async_trait]
trait Llm: Send + Sync {
    /// Возвращает unified diff‑патч как строку (в стиле `git diff`), отвечая за роль "Builder".
    async fn propose_patch(&self, task: &str, profile: usize) -> Result<String>;
}

/// Мок‑LLM: генерирует один «хороший» и несколько «плохих» патчей для демо‑прогона.
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
}

/// OpenAI‑совместимый клиент (Codex/любой совместимый API).
struct OpenAiCompat {
    base_url: String,
    api_key: String,
    model: String,
    http: Client,
}

#[async_trait]
impl Llm for OpenAiCompat {
    async fn propose_patch(&self, task: &str, profile: usize) -> Result<String> {
        // Подсказка имитирует роль "Senior Implementer #profile" и просит только unified diff.
        let prompt = format!(
            "ROLE: Senior Implementer #{}\nTASK:\n{}\nProduce a *unified diff* patch and nothing else. Target file demo_app/app.py.",
            profile, task
        );
        let body = serde_json::json!({
          "model": self.model,
          "messages": [{"role":"user","content": prompt}],
        });
        let url = format!("{}/v1/chat/completions", self.base_url.trim_end_matches('/'));
        let resp = self.http
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<serde_json::Value>()
            .await?;
        let content = resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        Ok(content)
    }
}

/// Приложение ядра: содержит LLM и URL тест‑сервиса.
#[derive(Clone)]
struct AppState {
    llm: Arc<dyn Llm>,
    tester_url: String,
    http: Client,
}

/// Реализация моста: генерирует N патчей, прогоняет каждый через tester‑service
/// и выбирает лучший по числу пройденных тестов.
async fn bridge_once(state: &AppState, req: &BridgeRequest) -> Result<BridgeResponse> {
    let mut gen = FuturesUnordered::new();
    for i in 0..req.builders {
        let llm = state.llm.clone();
        let task = req.task.clone();
        gen.push(async move { llm.propose_patch(&task, i).await });
    }

    // Собираем кандидатов
    let mut candidates: Vec<String> = Vec::new();
    while let Some(r) = gen.next().await {
        if let Ok(diff) = r {
            candidates.push(diff);
        }
    }

    // Тестируем каждого кандидата параллельно
    let mut tests = FuturesUnordered::new();
    for diff in &candidates {
        let http = state.http.clone();
        let tester_url = state.tester_url.clone();
        let body = serde_json::json!({ "diff": diff });
        tests.push(async move {
            http.post(format!("{}/testrun", tester_url))
                .json(&body)
                .send()
                .await?
                .error_for_status()?
                .json::<TestRunResult>()
                .await
        });
    }

    let mut best_idx = 0usize;
    let mut best: Option<TestRunResult> = None;
    for (idx, res) in tests.enumerate().collect::<Vec<_>>().await.into_iter() {
        let tr = res?;
        let better = match &best {
            None => true,
            Some(b) => tr.tests_failed < b.tests_failed || (tr.tests_failed == b.tests_failed && tr.tests_passed > b.tests_passed),
        };
        if better {
            best = Some(tr);
            best_idx = idx;
        }
    }

    let tests = best.expect("no candidates produced any result");
    Ok(BridgeResponse {
        request_id: Uuid::new_v4(),
        winner_index: best_idx,
        winner_diff: candidates.get(best_idx).cloned().unwrap_or_default(),
        tests,
    })
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

    let state = AppState { llm, tester_url, http };
    let app = Router::new()
        .route("/api/v1/bridge", post(api_bridge))
        .with_state(state);

    let addr: SocketAddr = "0.0.0.0:8080".parse().unwrap();
    info!("Agent Wrangler core listening on {}", addr);
    axum::Server::bind(&addr).serve(app.into_make_service()).await?;
    Ok(())
}

async fn api_bridge(State(state): State<AppState>, Json(req): Json<BridgeRequest>) -> Json<BridgeResponse> {
    let resp = bridge_once(&state, &req).await.expect("bridge failed");
    Json(resp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct TestLlm;
    #[async_trait]
    impl Llm for TestLlm {
        async fn propose_patch(&self, _task: &str, profile: usize) -> Result<String> {
            Ok(if profile == 0 {
                "diff --git a/demo_app/app.py b/demo_app/app.py\n--- a/demo_app/app.py\n+++ b/demo_app/app.py\n@@\n- return a - b\n+ return a + b\n".into()
            } else {
                "diff --git a/demo_app/app.py b/demo_app/app.py\n--- a/demo_app/app.py\n+++ b/demo_app/app.py\n@@\n- return a - b\n+ return a - b - 1\n".into()
            })
        }
    }

    #[tokio::test]
    async fn test_llm_mock() {
        let _ = TestLlm.propose_patch("x", 0).await.unwrap();
    }

    // Интеграционный тест мост/агрегатор опустим: он зависит от внешнего tester-service.
}
