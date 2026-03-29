# Technical Reference — Gomoku Platform V3

## Repo & Branch

- **Repo:** https://github.com/nitrogen27/tic-tac-toe-transform.git
- **Active branch:** `feature/gomoku-platform-v3`
- **Base branch:** `main` (legacy TensorFlow.js stack)

---

## Monorepo Layout

```
tic-tac-toe-transform/
├── engine-core/          C++17 alpha-beta engine (Phase 2 ✅)
├── trainer-lab/          PyTorch ResNet trainer (Phase 3 ✅)
├── apps/
│   ├── api/              FastAPI gateway (Phase 4 🔲)
│   └── web/              React + TypeScript UI (Phase 5 🔲)
├── packages/
│   └── shared/           JSON Schema contracts + codegen
├── docs/gomoku-platform-v3/
│   ├── README.md
│   └── MVP_ROADMAP.md    ← задачи по фазам 4–6
├── server/               Legacy Node.js + TensorFlow.js (работает)
├── client/               Legacy Vue.js (работает)
├── docker-compose.dev.yml    Новая платформа
├── docker-compose.gpu.yml    Legacy GPU стек
└── Makefile              Единая точка входа
```

---

## Key Commands

```bash
make build-engine     # cmake -B engine-core/build -S engine-core ...
make test-engine      # ctest --output-on-failure
make bench-engine     # gomoku_bench
make api              # uvicorn gomoku_api.main:app --reload --port 8000
make web              # cd apps/web && npm run dev  (порт 5174)
make train            # python -m trainer_lab.training.trainer
make dev-up           # docker compose -f docker-compose.dev.yml up -d --build
make legacy-up        # docker compose -f docker-compose.gpu.yml up -d --build
make codegen          # JSON Schema → TS + Python
```

---

## Service Ports

| Сервис | Порт | Описание |
|--------|------|----------|
| FastAPI API | 8000 | REST gateway → C++ engine |
| React Web | 5174 | Vite dev server |
| Legacy Server | 8080 | Node.js WS + TF.js |
| Legacy Client | 5173 | Vue.js dev |

---

## C++ Engine (`engine-core/`)

**Build:** CMake 3.20+, C++17, FetchContent (nlohmann-json, GTest, benchmark)

**CLI Protocol:** JSON stdin → JSON stdout, newline-delimited

```json
// Запрос
{"command":"best-move","position":{"boardSize":15,"winLength":5,"cells":[],"sideToMove":1,"moveCount":0,"lastMove":-1,"moveHistory":[]}}
// Ответ
{"bestMove":112,"source":"alpha-beta","value":0.0,"depth":6,"nodesSearched":15420,"timeMs":12,"topMoves":[...],"pvLine":[112,...],"policy":[]}
```

**Commands:** `best-move` | `analyze` | `suggest` | `info`

**Engine binary path (API default):**
`engine-core/build/gomoku_engine` (overridden via `GOMOKU_ENGINE_BINARY`)

**7-layer decision facade:**
SafetyWin → SafetyBlock → SafetyMultiBlock → VcfWin → VcfDefense → Fork → AlphaBeta

**Test results:** 36/36 pass | **Benchmarks:** MakeMove=10ns, LegalMoves=392ns

---

## FastAPI (`apps/api/`)

```
src/gomoku_api/
├── main.py                # create_app(), lifespan, CORS
├── config.py              # Settings(BaseSettings), env prefix GOMOKU_
├── models/schemas.py      # Pydantic models (Position, AnalyzeRequest/Response, ...)
├── routers/engine.py      # POST /analyze, /best-move, /suggest, GET /engine/info
├── routers/training.py    # POST /train/jobs, GET /train/jobs/{id}
├── routers/health.py      # GET /health
├── services/engine_adapter.py  # asyncio.create_subprocess_exec → CLI JSON
└── services/train_service.py   # Job queue stub
```

**Env vars:**
- `GOMOKU_ENGINE_BINARY` — путь к бинарнику (default: авто-определяется)
- `GOMOKU_HOST` — 0.0.0.0
- `GOMOKU_PORT` — 8000
- `GOMOKU_CORS_ORIGINS` — ["*"]

**Install:** `cd apps/api && pip install -e ".[dev]"` | Python 3.11+

---

## React Web (`apps/web/`)

```
src/
├── App.tsx                      # <GameProvider><MainLayout/>
├── api/
│   ├── client.ts                # fetch wrappers (stub — нужна реализация)
│   └── types.ts                 # TS-интерфейсы (Position, AnalyzeResponse, ...)
├── components/
│   ├── Board/Board.tsx          # SVG рендер 7–16, клики, ghost stone
│   ├── Board/Stone.tsx          # SVG circle, black/white/ghost
│   ├── Analysis/                # ValueBar, TopMovesTable, PVLine, BestMoveDisplay
│   ├── Game/                    # GameControls, MoveHistory
│   └── Layout/MainLayout.tsx
├── hooks/
│   ├── useGame.ts               # dispatch-обёртка над gameStore
│   ├── useBoard.ts              # SVG клик → индекс, hover
│   └── useAnalysis.ts           # вызов API (stub — нужна реализация)
├── store/gameStore.tsx          # GameState + useReducer + Context
└── utils/
    ├── boardUtils.ts            # flatToRowCol, rowColToFlat, emptyCells
    └── constants.ts             # BOARD_SIZES, COLORS, WIN_LENGTH=5
```

**Stack:** React 18 + TypeScript + Tailwind CSS + Vite (порт 5174)

**GameState структура:**
```ts
{ boardSize: 15, cells: CellValue[], currentPlayer: 1|-1,
  moveHistory: number[], lastMove: number,
  mode: "play"|"analyze"|"review",
  analysis: AnalyzeResponse|null, isAnalyzing: boolean }
```

---

## PyTorch Trainer (`trainer-lab/`)

```
src/trainer_lab/
├── config.py           # ModelConfig(res_blocks=8, res_filters=128), TrainConfig, SelfPlayConfig
├── data/encoder.py     # board → [6,16,16] float32 tensor
├── data/augmentation.py # D4 group, 8 transforms
├── models/resnet.py    # PolicyValueResNet: dual head policy[256]+value[1]
├── training/trainer.py # train(loader) → AMP + checkpoints + TensorBoard
├── export/onnx_export.py # opset 17, input "input" [1,6,16,16]
└── self_play/          # Phase-6 stubs
```

**Test results:** 21/21 pass | **Python:** 3.11+ | **Install:** `pip install -e ".[dev]"`

**ONNX outputs:** `"policy_logits"` [B,256] + `"value"` [B,1]

---

## Shared Schemas (`packages/shared/`)

JSON Schema → codegen в `generated/ts/index.ts` и `generated/python/schema_registry.py`

**Схемы:** `position`, `analyze`, `train_job`, `model_artifact`, `enums`

Запуск кодогена: `make codegen` или `bash packages/shared/codegen.sh`

---

## GPU / Infrastructure

- **GPU:** RTX 3060 Laptop 6GB VRAM | CUDA 12.0 | WDDM mode (Windows 11)
- **TF.js:** `TF_FORCE_GPU_ALLOW_GROWTH=true`, MAX_SAFE_BATCH=256
- **Нельзя:** `nvidia-smi -pl` из контейнера в WDDM режиме
- **Можно:** `nvidia-smi -ac` внутри контейнера для блокировки тактов

---

## Legacy Stack (server/ + client/)

- **Node.js** + WebSocket (`ws`) + TensorFlow.js GPU backend
- **Vue.js 3** frontend
- **Запуск:** `make legacy-up` → docker-compose.gpu.yml
- **Порты:** server:8080, client:5173
- Не трогать без явного запроса
