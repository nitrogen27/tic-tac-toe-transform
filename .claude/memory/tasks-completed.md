# Завершённые задачи

---

## Phase 1 — Monorepo Scaffolding ✅

**Коммит:** в `feature/gomoku-platform-v3`

- Структура папок: `engine-core/`, `trainer-lab/`, `apps/api/`, `apps/web/`, `packages/shared/`
- JSON Schema контракты: position, analyze, train_job, model_artifact, enums
- `packages/shared/codegen.sh` — JSON Schema → TS types + Python Pydantic models
- `docker-compose.dev.yml` — оркестрация 4 сервисов новой платформы
- `Makefile` — единая точка входа: codegen, build-engine, api, web, train, dev-up/down, legacy-up/down
- `docker-compose.gpu.yml` — GPU-оптимизации legacy стека

---

## Phase 2 — C++17 Engine Core ✅

**Результаты:** 36/36 тестов | CLI верифицирован | build: gomoku-engine.exe

**Файлы (26 шт., ~4000 LOC):**

| Модуль | Файлы |
|--------|-------|
| Типы | `types.hpp` |
| Доска | `board.hpp/cpp` — uint64_t Zobrist hash, make/undo move |
| Конфиг | `config.hpp` — FIVE=1M, OPEN_FOUR=100K, depth=12 |
| Паттерны | `patterns.hpp/cpp` — scan + gapped, VCF |
| Оценка | `evaluate.hpp/cpp` — static eval, tanh normalize |
| TT | `transposition.hpp/cpp` — EXACT/LOWER/UPPER, depth-based |
| Упорядочивание | `move_ordering.hpp/cpp` — killer (2/depth) + history |
| Кандидаты | `candidates.hpp/cpp` — radius-based + policy merge |
| Угрозы | `threats.hpp/cpp` — VCF/VCT, four-creating moves |
| Поиск | `search.hpp/cpp` — IDAB + PVS + LMR + aspiration |
| Симметрия | `symmetry.hpp/cpp` — D4 group, thread-safe кэш |
| Движок | `engine.hpp/cpp` — 7-layer facade |
| CLI | `cli_main.cpp` — JSON stdin/stdout |

**Бенчмарки:** MakeMove=10нс, LegalMoves(15×15)=392нс

**CLI верификация:** пустая 15×15 → bestMove=112 (центр) ✓

---

## Phase 3 — Trainer Lab ✅

**Результаты:** 21/21 тестов | scaffold complete

**Файлы:**

| Модуль | Файл | Тесты |
|--------|------|-------|
| Кодировщик | `data/encoder.py` | 9 |
| Аугментация | `data/augmentation.py` | 6 |
| ResNet | `models/resnet.py` | 6 |
| Лосс | `training/loss.py` | — |
| Метрики | `training/metrics.py` | — |
| Шедулер | `training/scheduler.py` | — |
| Тренер | `training/trainer.py` | — |
| ONNX | `export/onnx_export.py` | — |
| Eval | `evaluation/eval_script.py` | — |
| Self-play | `self_play/` (skeleton) | — |

---

## GPU Оптимизации Legacy стека ✅

**Проблема:** GPU загрузка 90% по утилизации, но ~20Вт вместо ожидаемых 80–95Вт

**Root causes и fix:**
1. Microtask flush → setTimeout(4ms) — +40W, главная причина
2. BATCH_SIZE: 16 → 128 — накопление батча
3. batchParallel: 96 вместо 32
4. inferenceSimulations: 200, trainingSimulations: 800

**OOM fix:**
- `TF_FORCE_GPU_ALLOW_GROWTH=true` (отменён false)
- Убран `TF_GPU_ALLOCATOR=cuda_malloc_async`
- `MAX_SAFE_BATCH=256` — кап в коде сервера
- Причина: WDDM + Windows GUI = ~2.5GB VRAM занято системой

---

## Документация ✅

- `docs/gomoku-platform-v3/README.md` — прогресс фаз 1–6
- `docs/gomoku-platform-v3/MVP_ROADMAP.md` — детальный план фаз 4–6
- `engine-core/README.md` — build, CLI, тесты, бенчмарки
- `trainer-lab/README.md` — архитектура, setup, формат данных
- `apps/api/README.md` — эндпоинты, env vars, архитектура
- `README.md` (root) — таблица статусов всех фаз

---

## Pending

| Фаза | Файлы | Статус |
|------|-------|--------|
| Phase 4 — FastAPI | `apps/api/services/engine_adapter.py` (реализация), роутеры | 🔲 |
| Phase 5 — Web | `apps/web/src/api/client.ts`, Board logic, Analysis sidebar | 🔲 |
| Phase 6 — Self-Play | `trainer-lab/self_play/player.py` (MCTS), pipeline loop | 🔲 |
