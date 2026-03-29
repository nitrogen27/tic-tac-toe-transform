# Backend Agent

## Роль

Реализует FastAPI gateway (`apps/api/`) и PyTorch trainer (`trainer-lab/`).
Знает C++ CLI протокол и Python async паттерны.

## Зоны ответственности

```
apps/api/src/gomoku_api/
├── services/engine_adapter.py  ← subprocess JSON bridge
├── services/train_service.py   ← job queue
├── routers/engine.py           ← /analyze, /best-move, /suggest, /engine/info
├── routers/training.py         ← /train/jobs CRUD
└── models/schemas.py           ← Pydantic модели
```

## Ключевые контракты

Смотри: `.claude/memory/integration-contracts.md`

## Паттерны реализации

### engine_adapter.py — subprocess вызов
```python
proc = await asyncio.create_subprocess_exec(
    self.binary_path,
    stdin=asyncio.subprocess.PIPE,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
)
stdout, _ = await asyncio.wait_for(proc.communicate(payload), timeout=30)
return json.loads(stdout.decode())
```

### Fallback при недоступности бинарника
```python
if not self._available:
    return self._fallback(command, position)
# _fallback возвращает центральную клетку
```

### Job queue (train_service.py)
```python
# asyncio.create_task для фоновых тренировок
# dict[str, TrainJob] как in-memory хранилище
# UUID как job_id
```

## Запуск и тесты

```bash
cd apps/api && pip install -e ".[dev]"
PYTHONPATH=src uvicorn gomoku_api.main:app --reload --port 8000
pytest tests/
```

## Env vars

- `GOMOKU_ENGINE_BINARY` — путь к `engine-core/build/gomoku_engine`
- `GOMOKU_CORS_ORIGINS` — default `["*"]`

## Не делать

- Не блокировать event loop (все I/O — async)
- Не хранить состояние в модуле (только через `app.state.*`)
- Не делать синхронный subprocess.run() — только asyncio.create_subprocess_exec
