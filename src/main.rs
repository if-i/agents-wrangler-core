//! Ядро «Пастуха агентов» с мульти-инстансными мостами поверх локального Codex CLI.
//! Провайдеры:
//! - mock      — демо без реальной модели
//! - openai    — OpenAI-совместимый чат-комплишн
//! - codex     — локальный Codex через HTTP-обёртку, с пулами инстансов для ролей

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

/// ---- API моделей ----

#[derive(Debug, Deserialize)]
struct BridgeRequest {
    task: String,
    builders: usize,
}

#[derive(Debug, Deserialize)]
struct MultiBridgeRequest {
    task: String,
    builders: usize,
    reviewers: usize,
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

#[derive(Debug, Deserialize, Clone)]
struct TestRunResult {
    tests_total: i32,
    tests_passed: i32,
    tests_failed: i32,
    return_code: i32,
    stdout: String,
    stderr: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Plan {
    components: Vec<Component>,
}
#[derive(Debug, Deserialize, Serialize, Clone)]
struct Component {
    name: String,
    target_files: Vec<String>,
}
#[derive(Debug, Deserialize, Serialize, Clone)]
struct Review {
    score: f32,
    rationale: String,
}

/// ---- Контракты провайдеров ----

#[async_trait]
trait Llm: Send + Sync {
    async fn propose_patch(&self, task: &str, profile: usize) -> Result<String>;
    async fn plan(&self, task: &str) -> Result<Plan>;
    async fn final_review(&self, task: &str, diffs: &[String]) -> Result<Review>;
}

#[async_trait]
trait TestRunner: Send + Sync {
    async fn run_diffs(&self, diffs: &[String]) -> Result<TestRunResult>;
}

/// ---- Тест-раннер (HTTP) ----

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

/// ---- Mock LLM ----

struct MockLlm;
#[async_trait]
impl Llm for MockLlm {
    async fn propose_patch(&self, _task: &str, profile: usize) -> Result<String> {
        let good = "diff --git a/demo_app/app.py b/demo_app/app.py\n--- a/demo_app/app.py\n+++ b/demo_app/app.py\n@@\n-    return a - b\n+    return a + b\n";
        let bad = "diff --git a/demo_app/app.py b/demo_app/app.py\n--- a/demo_app/app.py\n+++ b/demo_app/app.py\n@@\n-    return a - b\n+    return a - b - 1\n";
        Ok(if profile % 3 == 0 { good.into() } else { bad.into() })
    }
    async fn plan(&self, _task: &str) -> Result<Plan> {
        Ok(Plan { components: vec![Component { name: "fix_add_function".into(), target_files: vec!["demo_app/app.py".into()] }] })
    }
    async fn final_review(&self, _task: &str, _diffs: &[String]) -> Result<Review> {
        Ok(Review { score: 0.9, rationale: "mock ok".into() })
    }
}

/// ---- OpenAI-совместимый провайдер ----

struct OpenAiCompat {
    base_url: String,
    api_key: String,
    model: String,
    http: Client,
}
#[async_trait]
impl Llm for OpenAiCompat {
    async fn propose_patch(&self, task: &str, profile: usize) -> Result<String> {
        let prompt = format!("ROLE: Senior Implementer #{profile}\nTASK:\n{task}\nProduce a unified diff, only the patch.");
        let body = serde_json::json!({ "model": self.model, "messages": [{"role":"user","content": prompt}]});
        let url = format!("{}/v1/chat/completions", self.base_url.trim_end_matches('/'));
        let v = self.http.post(url).bearer_auth(&self.api_key).json(&body).send().await?.error_for_status()?.json::<serde_json::Value>().await?;
        Ok(v["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string())
    }
    async fn plan(&self, task: &str) -> Result<Plan> {
        let prompt = format!("ROLE: Architect\nGoal:\n{task}\nReturn STRICT JSON {{\"components\":[{{\"name\":\"...\",\"target_files\":[\"...\"]}}]}}");
        let body = serde_json::json!({ "model": self.model, "messages": [{"role":"user","content": prompt}]});
        let url = format!("{}/v1/chat/completions", self.base_url.trim_end_matches('/'));
        let v = self.http.post(url).bearer_auth(&self.api_key).json(&body).send().await?.error_for_status()?.json::<serde_json::Value>().await?;
        let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("{}");
        let plan: Plan = serde_json::from_str(content).unwrap_or(Plan { components: vec![] });
        Ok(plan)
    }
    async fn final_review(&self, task: &str, diffs: &[String]) -> Result<Review> {
        let prompt = format!("ROLE: Reviewer\nTask:\n{task}\nPatches:\n{}\nReturn STRICT JSON {{\"score\":<0..1>,\"rationale\":\"...\"}}", diffs.join("\n---\n"));
        let body = serde_json::json!({ "model": self.model, "messages": [{"role":"user","content": prompt}]});
        let url = format!("{}/v1/chat/completions", self.base_url.trim_end_matches('/'));
        let v = self.http.post(url).bearer_auth(&self.api_key).json(&body).send().await?.error_for_status()?.json::<serde_json::Value>().await?;
        let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("{\"score\":0.5,\"rationale\":\"n/a\"}");
        let review: Review = serde_json::from_str(content).unwrap_or(Review { score: 0.5, rationale: "n/a".into() });
        Ok(review)
    }
}

/// ---- Codex провайдер с пулами инстансов ----

#[derive(Clone)]
struct CodexPools {
    plan: Vec<String>,
    build: Vec<String>,
    review: Vec<String>,
}
impl CodexPools {
    fn from_env() -> Self {
        fn split(var: &str) -> Vec<String> {
            env::var(var).ok().map(|s| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()).unwrap_or_default()
        }
        let single = env::var("CODEX_RUNNER_URL").ok();
        let mut plan = split("CODEX_PLAN_URLS");
        let mut build = split("CODEX_BUILDER_URLS");
        let mut review = split("CODEX_REVIEW_URLS");
        if plan.is_empty() { if let Some(s) = single.clone() { plan.push(s); } }
        if build.is_empty() { if let Some(s) = single.clone() { build.push(s); } }
        if review.is_empty() { if let Some(s) = single.clone() { review.push(s); } }
        Self { plan, build, review }
    }
    fn pick_plan(&self) -> &str { &self.plan[0] }
    fn pick_build(&self, idx: usize) -> &str { &self.build[idx % self.build.len()] }
    fn pick_review(&self) -> &str { &self.review[0] }
}

struct CodexHttp {
    http: Client,
    pools: CodexPools,
}
#[derive(Serialize)]
struct CodexPlanReq<'a> { task: &'a str }
#[derive(Deserialize)]
struct CodexPlanResp { components: Vec<Component> }
#[derive(Serialize)]
struct CodexImplReq<'a> { task: &'a str }
#[derive(Deserialize)]
struct CodexImplResp { diff: String, stdout: String, stderr: String }
#[derive(Serialize)]
struct CodexRevReq<'a> { task: &'a str, diffs: &'a [String] }
#[derive(Deserialize)]
struct CodexRevResp { score: f32, rationale: String }

#[async_trait]
impl Llm for CodexHttp {
    async fn propose_patch(&self, task: &str, profile: usize) -> Result<String> {
        let url = self.pools.pick_build(profile).to_string() + "/codex/implement";
        let v = self.http.post(url).json(&CodexImplReq { task }).send().await?.error_for_status()?.json::<CodexImplResp>().await?;
        Ok(v.diff)
    }
    async fn plan(&self, task: &str) -> Result<Plan> {
        let url = self.pools.pick_plan().to_string() + "/codex/plan";
        let v = self.http.post(url).json(&CodexPlanReq { task }).send().await?.error_for_status()?.json::<CodexPlanResp>().await?;
        Ok(Plan { components: v.components })
    }
    async fn final_review(&self, task: &str, diffs: &[String]) -> Result<Review> {
        let url = self.pools.pick_review().to_string() + "/codex/review";
        let v = self.http.post(url).json(&CodexRevReq { task, diffs }).send().await?.error_for_status()?.json::<CodexRevResp>().await?;
        Ok(Review { score: v.score, rationale: v.rationale })
    }
}

/// ---- Приложение ----

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
        "codex" => Arc::new(CodexHttp { http: http.clone(), pools: CodexPools::from_env() }),
        _ => Arc::new(MockLlm),
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

/// best-of-N для билдеров
async fn api_bridge(State(state): State<AppState>, Json(req): Json<BridgeRequest>) -> Json<BridgeResponse> {
    let resp = bridge_best_of_n(&state, &req).await.expect("bridge failed");
    Json(resp)
}

/// мульти-мост: план → билдеры → специалисты → финальный ревью
async fn api_bridge_multi(State(state): State<AppState>, Json(req): Json<MultiBridgeRequest>) -> Json<MultiBridgeResponse> {
    let resp = bridge_multi(&state, &req).await.expect("bridge failed");
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
        if let Ok(d) = r { diffs.push(d); }
    }
    let mut best_idx = 0usize;
    let mut best: Option<TestRunResult> = None;
    for (i, d) in diffs.iter().enumerate() {
        let tr = state.tester.run_diffs(&[d.clone()]).await?;
        let better = match &best {
            None => true,
            Some(b) => tr.tests_failed < b.tests_failed || (tr.tests_failed == b.tests_failed && tr.tests_passed > b.tests_passed),
        };
        if better { best = Some(tr); best_idx = i; }
    }
    Ok(BridgeResponse {
        request_id: Uuid::new_v4(),
        winner_index: best_idx,
        winner_diff: diffs.get(best_idx).cloned().unwrap_or_default(),
        tests: best.unwrap(),
    })
}

async fn bridge_multi(state: &AppState, req: &MultiBridgeRequest) -> Result<MultiBridgeResponse> {
    // План
    let plan = state.llm.plan(&req.task).await?;

    // Билдеры → базовый патч
    let base = bridge_best_of_n(state, &BridgeRequest { task: req.task.clone(), builders: req.builders }).await?;
    let mut accepted_diffs = vec![base.winner_diff.clone()];
    let mut current = base.tests.clone();

    // Специалисты: жадно, пока метрики не хуже
    let mut accepted_specialists = 0usize;
    for comp in &plan.components {
        for _ in 0..req.specialists.max(1) {
            // используем propose_patch ещё раз с модифицированной формулировкой
            let prompt = format!("Implement specialized improvements for component '{}'", comp.name);
            let patch = state.llm.propose_patch(&prompt, accepted_specialists).await?;
            let mut trial = accepted_diffs.clone();
            trial.push(patch.clone());
            let tr = state.tester.run_diffs(&trial).await?;
            let better = tr.tests_failed < current.tests_failed || (tr.tests_failed == current.tests_failed && tr.tests_passed >= current.tests_passed);
            if better {
                accepted_diffs.push(patch);
                current = tr;
                accepted_specialists += 1;
            }
        }
    }

    // Финальный ревью
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

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTester;
    #[async_trait]
    impl TestRunner for MockTester {
        async fn run_diffs(&self, diffs: &[String]) -> Result<TestRunResult> {
            let text = diffs.join("\n");
            let failed = if text.contains("return a + b") { 0 } else { 1 };
            Ok(TestRunResult {
                tests_total: 1,
                tests_passed: 1 - failed,
                tests_failed: failed,
                return_code: if failed == 0 { 0 } else { 1 },
                stdout: String::new(),
                stderr: String::new(),
            })
        }
    }

    #[tokio::test]
    async fn test_best_of_n() {
        let state = crate::AppState {
            llm: Arc::new(MockLlm),
            tester: Arc::new(MockTester),
            http: Client::new(),
        };
        let resp = super::bridge_best_of_n(&state, &BridgeRequest { task: "fix".into(), builders: 3 }).await.unwrap();
        assert_eq!(resp.tests.tests_failed, 0);
    }

    #[tokio::test]
    async fn test_multi_ok() {
        let state = crate::AppState {
            llm: Arc::new(MockLlm),
            tester: Arc::new(MockTester),
            http: Client::new(),
        };
        let resp = super::bridge_multi(&state, &MultiBridgeRequest { task: "fix", builders: 3, reviewers: 1, specialists: 2 }).await.unwrap();
        assert!(resp.diffs.iter().any(|d| d.contains("return a + b")));
        assert_eq!(resp.tests.tests_failed, 0);
    }
}
