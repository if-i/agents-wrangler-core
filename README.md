# agents-wrangler-core (Rust)

Fast core orchestrator for Agent Wrangler.

## Run (standalone)

The core expects a tester service at `TESTER_URL` (FastAPI app from the Python repo).
By default it uses a Mock LLM (no API key required).

```bash
# build & run locally
cargo run
# or via Docker
docker build -t aw-core .
docker run --rm -p 8080:8080 -e TESTER_URL=http://host.docker.internal:7001 aw-core
```

Switch to real LLM (OpenAI-compatible)

```
USE_MOCK_LLM=0
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com
OPENAI_MODEL=gpt-4o-mini
```

API
	•	POST /api/v1/bridge – best-of-N bridge
	•	POST /api/v1/bridge/multi – multi-agent pipeline (architect → builders → specialists → final review)

For end-to-end local setup (tester + core via Docker Compose), see the main Quick Start in
agent-wrangler: https://github.com/if-i/agent-wrangler

