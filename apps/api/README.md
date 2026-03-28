# Gomoku Platform API

FastAPI gateway bridging the React frontend and the C++ engine via JSON subprocess calls.

## Status

🔲 **Phase 4** — Scaffold complete. Engine adapter subprocess stub ready. Full implementation pending.

## Architecture

```
src/gomoku_api/
├── main.py              # FastAPI app, CORS, lifespan
├── routers/
│   ├── engine.py        # /analyze, /best-move, /suggest, /engine/info
│   ├── training.py      # /train/jobs CRUD
│   └── health.py        # /health liveness probe
└── services/
    └── engine_adapter.py  # Subprocess call → C++ CLI JSON protocol
```

The engine adapter launches `gomoku-engine` as a subprocess, sends JSON via stdin, reads response from stdout. Binary path configured via `GOMOKU_ENGINE_BINARY` env var.

## Quick start

```bash
cd apps/api
pip install -e ".[dev]"
uvicorn gomoku_api.main:app --reload --port 8000

# Or via Makefile from repo root:
make api
```

## Tests

```bash
pytest
```

## Endpoints

| Method | Path              | Description                        |
|--------|-------------------|------------------------------------|
| GET    | /health           | Liveness probe                     |
| GET    | /engine/info      | Engine version and capabilities    |
| POST   | /analyze          | Full position analysis + PV line   |
| POST   | /best-move        | Single best move (fastest)         |
| POST   | /suggest          | Top-K move hints for UI            |
| POST   | /train/jobs       | Create training job                |
| GET    | /train/jobs/{id}  | Get training job status            |

Interactive docs available at `http://localhost:8000/docs` when running.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOMOKU_ENGINE_BINARY` | `gomoku-engine` | Path to C++ engine binary |
| `GOMOKU_HOST` | `0.0.0.0` | Listen host |
| `GOMOKU_PORT` | `8000` | Listen port |
