# Hybrid Quantum-Classical Control Room

Docker-ready web demo for a **Quantum Digital Twin** prototype built with Streamlit and a modular hybrid quantum-classical engine.

## Repository structure

```text
q-infratwin-webapp/
├── app.py
├── requirements.txt
├── runtime.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── render.yaml
├── .gitignore
├── .streamlit/
│   └── config.toml
└── src/
    └── q_infratwin/
        ├── __init__.py
        └── engine.py
```

## Run locally without Docker

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Docker deployment

### Option A — plain Docker

```bash
docker build -t hybrid-control-room .
docker run --rm -p 8501:8501 hybrid-control-room
```

Open:

```text
http://localhost:8501
```

### Option B — Docker Compose

```bash
docker compose up --build
```

To stop it:

```bash
docker compose down
```

## Recommended Docker workflow for your repo

1. Replace the current repository files with this package.
2. Commit and push to GitHub.
3. On the target machine:
   - clone the repo,
   - run `docker compose up --build -d`,
   - expose port `8501` or map it behind your reverse proxy.

## Notes

- The container uses Python 3.11.
- `PYTHONPATH=/app/src` is already configured, so imports from `src/q_infratwin` work directly.
- The app starts with `python -m streamlit run app.py`.
- The container runs as a non-root user.

## Public demo flow

1. Start with **Balanced demo**.
2. Show objective, latency and routing behaviour in **Overview**.
3. Open **Twin drill-down** to inspect a single twin.
4. Open **Audit inspector** to show the quantum request and result envelopes.
5. Export the run summary for reporting.
