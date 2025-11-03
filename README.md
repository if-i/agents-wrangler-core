## Getting Started (Docker Compose)

> Clone two repos side by side:
>
> ```
> ~/work/agent-wrangler        # Python (this repo)
> ~/work/agents-wrangler-core  # Rust core
> ```

### 1) Build & run services
```bash
docker compose up --build
# core -> http://localhost:8080
# tester -> http://localhost:7001
