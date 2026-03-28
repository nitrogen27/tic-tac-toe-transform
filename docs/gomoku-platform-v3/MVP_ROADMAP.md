# Gomoku Platform V3 — MVP Roadmap

> Документ описывает что осталось реализовать для полного рабочего MVP.
> Фазы 1–3 завершены. Ниже — детальный список задач для фаз 4–6.

---

## ✅ Завершено

| Фаза | Описание | Статус |
|------|----------|--------|
| Phase 1 | Monorepo scaffolding, JSON schemas, Makefile, docker-compose | ✅ DONE |
| Phase 2 | C++17 engine — 36/36 тестов, CLI, 10 нс/ход | ✅ DONE |
| Phase 3 | PyTorch ResNet trainer — 21/21 тестов, ONNX export, D4 augmentation | ✅ DONE |

---

## 🔲 Phase 4 — FastAPI Gateway

**Файлы:** `apps/api/src/gomoku_api/`

Scaffold уже есть (роутеры, модели, engine_adapter stub). Нужно реализовать:

### 4.1 Engine Adapter (`services/engine_adapter.py`)
- [ ] Реализовать `call_engine(command, position)` — запуск `gomoku-engine` как subprocess
- [ ] Передача JSON в stdin, чтение JSON из stdout
- [ ] Парсинг ответа в Pydantic-модели (`AnalyzeResponse`, `MoveCandidate`)
- [ ] Timeout handling (обрыв subprocess по таймауту)
- [ ] Путь к бинарнику из `GOMOKU_ENGINE_BINARY` env var

### 4.2 Engine Router (`routers/engine.py`)
- [ ] `POST /analyze` — полный анализ позиции (depth, PV line, topMoves)
- [ ] `POST /best-move` — быстрый лучший ход
- [ ] `POST /suggest` — top-K подсказок для UI
- [ ] `GET /engine/info` — версия движка и поддерживаемые размеры доски

### 4.3 Training Router (`routers/training.py`)
- [ ] `POST /train/jobs` — создание job (variant, config)
- [ ] `GET /train/jobs/{id}` — статус job (queued/running/completed/failed)
- [ ] In-memory job queue с фоновым запуском через `asyncio.create_task`
- [ ] Интеграция с `trainer_lab.training.trainer`

### 4.4 Тесты
- [ ] `tests/test_engine_router.py` — mock subprocess, проверить 200/422/500
- [ ] `tests/test_training_router.py` — CRUD job lifecycle
- [ ] Integration test: реальный вызов `gomoku-engine` в CI

### 4.5 Docker
- [ ] Убедиться что `GOMOKU_ENGINE_BINARY` корректно маппится через volume `engine-bin`
- [ ] Healthcheck в `docker-compose.dev.yml`

**Ориентир LOC:** ~350 строк Python

---

## 🔲 Phase 5 — Web MVP (React + TypeScript)

**Файлы:** `apps/web/src/`

Scaffold уже есть (компоненты, хуки, store). Нужно реализовать:

### 5.1 API Client (`api/client.ts`)
- [ ] `analyzePosition(position)` → `POST /analyze`
- [ ] `getBestMove(position)` → `POST /best-move`
- [ ] `getSuggestions(position, n)` → `POST /suggest`
- [ ] Error handling (сеть недоступна, движок не запущен)
- [ ] AbortController для отмены in-flight запросов

### 5.2 Board Component (`components/Board/Board.tsx`)
- [ ] SVG рендер для размеров 7×7 → 16×16
- [ ] Клик по пересечению → размещение камня
- [ ] Подсветка лучшего хода (цветной маркер)
- [ ] Подсветка PV line (первые 3 хода)
- [ ] Последний ход — специальный маркер

### 5.3 Game State (`hooks/useGame.ts`, `store/gameStore.tsx`)
- [ ] Play mode: чередование ходов (человек vs движок)
- [ ] Review mode: пошаговый просмотр истории
- [ ] Смена размера доски — сброс игры
- [ ] Определение победителя на клиенте

### 5.4 Analysis Sidebar (`components/Analysis/`)
- [ ] `ValueBar.tsx` — прогресс-бар value ∈ [-1, 1]
- [ ] `TopMovesTable.tsx` — таблица топ-5 ходов с вероятностями
- [ ] `PVLine.tsx` — отображение principal variation (цепочка ходов)
- [ ] `BestMoveDisplay.tsx` — крупный лучший ход с source (alpha-beta / vcf-win / …)
- [ ] Кнопка «Analyze» — триггер вызова API

### 5.5 Layout
- [ ] Адаптивный layout (доска + сайдбар)
- [ ] Селектор размера доски (7, 9, 11, 13, 15, 16)
- [ ] Индикатор загрузки (spinner пока идёт анализ)
- [ ] Toast при ошибке API

### 5.6 Тесты
- [ ] Vitest unit-тесты для `boardUtils.ts` (перевод клик → индекс)
- [ ] Vitest тесты для `useGame` hook (ходы, смена режима)
- [ ] Playwright/Cypress E2E: открыть, сделать ход, получить анализ (опционально)

**Ориентир LOC:** ~900 строк TypeScript/TSX

---

## 🔲 Phase 6 — Self-Play Pipeline

**Файлы:** `trainer-lab/src/trainer_lab/self_play/`

Skeleton уже есть. Нужно реализовать полный цикл:

### 6.1 MCTS Player (`self_play/player.py`)
- [ ] Класс `MCTSNode` — дерево поиска (visit count, value sum, prior)
- [ ] `mcts_search(position, model, simulations)` — PUCT selection + expansion
- [ ] Батчинг инференса: накапливать листья, один forward pass на батч
- [ ] Temperature τ: τ=1 в начале игры, τ→0 после хода N
- [ ] Dirichlet noise на корне для exploration
- [ ] `SelfPlayPlayer.play_game()` → список `(position, policy, outcome)`

### 6.2 Replay Buffer (`self_play/replay_buffer.py`)
- [ ] Приоритизированная выборка (опционально — uniform достаточно для MVP)
- [ ] Сохранение буфера на диск (`torch.save`) для восстановления после падения
- [ ] `sample(n)` → батч `(planes, policy, value)`

### 6.3 Pipeline (`self_play/pipeline.py`)
- [ ] `run(generations)` — цикл: генерация → обучение → оценка → ONNX export
- [ ] Evaluation: модель vs движок (минимум 20 игр, win rate > 55% для промоции)
- [ ] ONNX экспорт лучшей модели после каждого поколения
- [ ] Логирование метрик (TensorBoard / stdout)
- [ ] Checkpoint сравнение: новая модель заменяет старую только при улучшении

### 6.4 Интеграция с движком
- [ ] Использовать `engine_adapter` для генерации training positions (engine-played games)
- [ ] Bootstrap фаза: первые N игр — движок vs движок для наполнения буфера

### 6.5 Тесты
- [ ] Тест `play_game()` с random policy (smoke test — не падает)
- [ ] Тест replay buffer: add/sample/overflow
- [ ] Тест ONNX export из pipeline

**Ориентир LOC:** ~650 строк Python

---

## MVP Definition of Done

Минимальный рабочий продукт считается достигнутым когда:

- [ ] **Engine**: `curl -X POST http://localhost:8000/best-move -d '{...}'` возвращает ход за < 200 мс
- [ ] **Web**: открыть `http://localhost:5174`, выбрать размер 15×15, сделать ход, нажать Analyze — видна оценка позиции
- [ ] **Training**: `make train` не падает на GPU (хотя бы 1 epoch на синтетических данных)
- [ ] **Docker**: `make dev-up` поднимает все 4 сервиса (engine-core, api, web, trainer-lab)
- [ ] **Legacy**: `docker compose -f docker-compose.gpu.yml up` всё ещё работает

---

## Порядок выполнения (рекомендуемый)

```
Phase 4 (FastAPI) → Phase 5 (Web) → интеграционный тест API+Web → Phase 6 (Self-Play)
```

Phase 4 и 5 можно вести параллельно: Phase 5 использует mock API пока Phase 4 не готова.

---

## Оценка оставшейся работы

| Фаза | LOC | Приоритет |
|------|-----|-----------|
| Phase 4 — FastAPI | ~350 | 🔴 Высокий (блокирует Web) |
| Phase 5 — Web MVP | ~900 | 🔴 Высокий (видимый результат) |
| Phase 6 — Self-Play | ~650 | 🟡 Средний (после MVP) |
| **Итого** | **~1900** | |
