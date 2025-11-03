//! Ядро «Пастуха агентов» с поддержкой локального Codex через codex-runner HTTP.
//! Провайдеры:
//! - mock      — демо без реальной модели
//! - openai    — OpenAI-совместимый чат-комплишн
//! - codex     — локальный Codex CLI через HTTP-обёртку (генерация патчей)

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

/// --- API модели ---

#[derive(Debug, Deserialize)]
struct BridgeRequest {
    task: String,
    builders: usize,
}

#[derive(Debug, Serialize)]
struct BridgeResponse {
    request_id: Uuid,
    winner_index: usize,
    winner_diff: String,
    tests: TestRunResult,
}

#[derive(Debug, Deserialize)]
struct TestRunResult {
    tests_total: i32,
    tests_passed: i32,
    tests_failed: i32,
    return_code: i32,
    stdout: String,
    stderr: String,
}

/// --- Контракты провайдеров ---

#[async_trait]
trait Llm: Send + Sync {
    /// Возвращает unified diff для "Builder".
    async fn propose_patch(&self, task: &str, profile: usize) -> Result<String>;
}

/// HTTP-тестраннер (наш FastAPI-сервис)
#[async_trait]
trait TestRunner: Send + Sync {
    async fn run_diffs(&self, diffs: &[String]) -> Result<TestRunResult>;
}

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

/// Мок-LLM: даёт "хороший" и "плохой" патч.
struct MockLlm;

#[async_trait]
impl Llm for MockLlm {
    async fn propose_patch(&self, _task: &str, profile: usize) -> Result<String> {
        let good = r#"diff --git a/demo_app/app.py b/demo_app/app.py
--- a/demo_app/app.py
+++ b/demo_app/app.py
@@
-    return a - b
+    return a + b
"#;
        let bad = r#"diff --git a/demo_app/app.py b/demo_app/app.py
--- a/demo_app/app.py
+++ b/demo_app/app.py
@@
-    return a - b
+    return a - b - 1
"#;
        Ok(if profile % 3 == 0 { good.into() } else { bad.into() })
    }
}

/// OpenAI-совместимый провайдер.
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
}

/// Codex-CLI как HTTP: вызывает наш codex-runner и возвращает diff.
struct CodexHttp {
    base_url: String,
    http: Client,
}

#[derive(Debug, Serialize)]
struct CodexPatchReq<'a> {
    task: &'a str,
    model: Option<&'a str>,
}

#[derive(Debug, Deserialize)]
struct CodexPatchResp {
    diff: String,
    stdout: String,
    stderr: String,
}

#[async_trait]
impl Llm for CodexHttp {
    async fn propose_patch(&self, task: &str, _profile: usize) -> Result<String> {
        let body = CodexPatchReq { task, model: None };
        let v = self
            .http
            .post(format!("{}/codex/patch", self.base_url))
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<CodexPatchResp>()
            .await?;
        Ok(v.diff)
    }
}

/// --- Приложение ---

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
    let provider = env::var("LLM_PROVIDER").unwrap_or_else(|_| "mock".into());

    let llm: Arc<dyn Llm> = match provider.as_str() {
        "openai" => {
            let base = env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com".into());
            let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY missing");
            let model = env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());
            Arc::new(OpenAiCompat { base_url: base, api_key, model, http: http.clone() })
        }
        "codex" => {
            let base = env::var("CODEX_RUNNER_URL").unwrap_or_else(|_| "http://localhost:7002".into());
            Arc::new(CodexHttp { base_url: base, http: http.clone() })
        }
        _ => Arc::new(MockLlm),
    };

    let state = AppState {
        llm,
        tester: Arc::new(HttpTester { base_url: tester_url, http: http.clone() }),
        http,
    };

    let app = Router::new()
        .route("/api/v1/bridge", post(api_bridge))
        .with_state(state);

    let addr: SocketAddr = "0.0.0.0:8080".parse().unwrap();
    info!("Agent Wrangler core listening on {addr}");
    axum::Server::bind(&addr).serve(app.into_make_service()).await?;
    Ok(())
}

async fn api_bridge(State(state): State<AppState>, Json(req): Json<BridgeRequest>) -> Json<BridgeResponse> {
    let resp = bridge_best_of_n(&state, &req).await.expect("bridge failed");
    Json(resp)
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{routing::post, Router};
    use std::net::SocketAddr;

    #[tokio::test]
    async fn test_codexhttp_provider_ok() {
        // Локальный мок codex-runner
        let app = Router::new().route("/codex/patch", post(|Json::<super::CodexPatchReq> { .. }| async move {
            axum::Json(super::CodexPatchResp {
                diff: "diff --git a/demo_app/app.py b/demo_app/app.py\n--- a/demo_app/app.py\n+++ b/demo_app/app.py\n@@\n-    return a - b\n+    return a + b\n".into(),
                stdout: String::new(),
                stderr: String::new(),
            })
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let state = AppState {
            llm: Arc::new(CodexHttp { base_url: format!("http://{}", addr), http: Client::new() }),
            tester: Arc::new(HttpTester { base_url: "http://localhost:7001".into(), http: Client::new() }),
            http: Client::new(),
        };

        // Мокаем тест-раннер вызовом самого CodexHttp? Нет — просто проверим получение диффа.
        let diff = state.llm.propose_patch("x", 0).await.unwrap();
        assert!(diff.contains("return a + b"));
    }
}
