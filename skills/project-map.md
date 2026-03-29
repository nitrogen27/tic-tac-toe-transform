# Skill: project-map

## Описание

Карта проекта для быстрого поиска файла или функции.
Используй когда нужно найти где что-то находится, не делая grep по всему репозиторию.

---

## Где что живёт

### "Где добавить новый API endpoint?"
→ `apps/api/src/gomoku_api/routers/` (engine.py или training.py)
→ Схемы: `apps/api/src/gomoku_api/models/schemas.py`
→ Логика: `apps/api/src/gomoku_api/services/`
→ Шаблон: `.claude/templates/new-api-endpoint.md`

### "Где добавить новый UI компонент?"
→ `apps/web/src/components/<FeatureName>/`
→ Подключить в: `apps/web/src/components/Layout/MainLayout.tsx`
→ Стиль: `.claude/rules/frontend-style-guide.md`
→ Шаблон: `.claude/templates/new-page.md`

### "Где определены TypeScript типы?"
→ `apps/web/src/api/types.ts` — все API интерфейсы
→ `packages/shared/generated/ts/index.ts` — сгенерированные из JSON Schema

### "Где вызывается C++ движок?"
→ `apps/api/src/gomoku_api/services/engine_adapter.py` — subprocess вызов
→ Протокол: `.claude/memory/integration-contracts.md`

### "Где конфиги сервисов?"
→ FastAPI: `apps/api/src/gomoku_api/config.py` (env prefix: GOMOKU_)
→ Trainer: `trainer-lab/src/trainer_lab/config.py` (ModelConfig, TrainConfig)
→ C++ Engine: `engine-core/include/gomoku/config.hpp`

### "Где находятся тесты?"
→ C++ engine: `engine-core/tests/test_board.cpp`
→ FastAPI: `apps/api/tests/`
→ Trainer: `trainer-lab/tests/`
→ Web: `apps/web/tests/` (vitest)

### "Где состояние игры на фронте?"
→ `apps/web/src/store/gameStore.tsx` — GameState + reducer + Context
→ Actions: PLACE_STONE, UNDO, NEW_GAME, SET_BOARD_SIZE, SET_MODE, SET_ANALYSIS

### "Где конвертация SVG клик → индекс?"
→ `apps/web/src/hooks/useBoard.ts` — eventToIndex()
→ `apps/web/src/utils/boardUtils.ts` — flatToRowCol, rowColToFlat

### "Где GPU настройки?"
→ `docker-compose.gpu.yml` — TF_FORCE_GPU_ALLOW_GROWTH, MAX_SAFE_BATCH
→ `docker-entrypoint-server.sh` — nvidia-smi -ac (clock lock)
→ `server/src/train_ttt5_service.mjs` — MAX_SAFE_BATCH=256, setTimeout(4ms)

### "Где задокументированы оставшиеся задачи?"
→ `docs/gomoku-platform-v3/MVP_ROADMAP.md`
→ `.claude/memory/tasks-completed.md`

### "Где архитектурные решения?"
→ `.claude/memory/architecture-decisions.md` — AD-001 через AD-010

### "Где JSON Schema контракты?"
→ `packages/shared/schemas/*.schema.json`
→ Codegen: `packages/shared/codegen.sh`
→ Сгенерированный Python: `packages/shared/generated/python/schema_registry.py`

---

## Дерево файлов по слоям

```
Контракты
  packages/shared/schemas/     → источник истины типов
  packages/shared/generated/   → сгенерированные (не редактировать вручную)

C++ Engine
  engine-core/include/gomoku/  → заголовки (types, board, search, engine, ...)
  engine-core/src/             → реализации
  engine-core/tests/           → GoogleTest

Python FastAPI
  apps/api/src/gomoku_api/
    models/schemas.py          → Pydantic модели
    services/engine_adapter.py → subprocess ↔ C++ CLI
    routers/                   → HTTP эндпоинты

Python Trainer
  trainer-lab/src/trainer_lab/
    models/resnet.py            → PolicyValueResNet
    training/trainer.py         → train() функция
    data/encoder.py             → board → tensor
    self_play/                  → Phase-6 stubs

React Web
  apps/web/src/
    api/types.ts               → TypeScript интерфейсы
    api/client.ts              → fetch функции
    store/gameStore.tsx        → GameState
    hooks/                     → useGame, useBoard, useAnalysis
    components/                → Board, Analysis, Game, Layout

Инфраструктура
  Makefile                     → make build-engine, api, web, train, dev-up
  docker-compose.dev.yml       → новая платформа
  docker-compose.gpu.yml       → legacy GPU
  .claude/                     → агенты, шаблоны, скрипты
```
