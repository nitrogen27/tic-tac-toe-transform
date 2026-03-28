# Gomoku Platform V3

New production-grade Gomoku platform living alongside the legacy TensorFlow.js stack during migration. Supports board sizes 7×7 to 16×16.

## Architecture

```
packages/shared/    JSON Schema contracts → TS + Python codegen
engine-core/        C++17 alpha-beta engine + CLI (JSON stdin/stdout)
trainer-lab/        PyTorch ResNet policy-value trainer
apps/api/           FastAPI gateway (subprocess → engine CLI)
apps/web/           React + TypeScript + Tailwind board UI
```

Legacy `server/` and `client/` remain fully operational.

---

## Progress

### ✅ Phase 1 — Monorepo Scaffolding

**Status: COMPLETE**

- Full folder scaffold for all new modules
- JSON Schema contracts: `position`, `analyze`, `train_job`, `model_artifact`, `enums`
- Codegen script: JSON Schema → TypeScript types + Python Pydantic models
- `docker-compose.dev.yml` — orchestrates engine-core, api, trainer-lab, web
- `Makefile` — top-level developer commands (codegen, build-engine, test-engine, api, web, train, dev-up/down, legacy-up/down)

### ✅ Phase 2 — C++17 Engine Core

**Status: COMPLETE**

Full port of the JS Gomoku engine (2420 LOC) to C++17 (~4000 LOC across 26 files).

**Modules:**

| File | Description |
|------|-------------|
| `types.hpp` | Core types: Position, EngineResult, EngineSource enum |
| `board.hpp/cpp` | GomokuBoard with uint64_t Zobrist hashing, make/undo move |
| `config.hpp` | Constants matching JS config (FIVE=1M, OPEN_FOUR=100K, etc.) |
| `patterns.hpp/cpp` | Pattern scan + gapped detection, win/threat counting |
| `evaluate.hpp/cpp` | Static eval, quick_move_eval, tanh normalisation |
| `transposition.hpp/cpp` | Fixed-size TT with depth-based replacement (EXACT/LOWER/UPPER) |
| `move_ordering.hpp/cpp` | Killer table (2 slots/depth) + history heuristic |
| `candidates.hpp/cpp` | Radius-based candidate generation + policy merge |
| `threats.hpp/cpp` | VCF/VCT threat-space search |
| `search.hpp/cpp` | IDAB + PVS + LMR + aspiration windows + time management |
| `symmetry.hpp/cpp` | D4 group (8 transforms), thread-safe caching |
| `engine.hpp/cpp` | 7-layer decision facade: SafetyWin→Block→MultiBlock→VcfWin→VcfDef→Fork→AB |
| `cli_main.cpp` | JSON stdin/stdout CLI (best-move, analyze, suggest, info) |

**Build:** CMake + FetchContent (nlohmann-json, GoogleTest, Google Benchmark)

**Test results: 36/36 passed**

**Benchmarks:**
- `MakeMove`: ~10 ns
- `LegalMoves` (15×15): ~392 ns

**CLI verification (empty 15×15):**
```json
{"bestMove":112,"source":"alpha-beta","value":0.0,"depth":1,"nodesSearched":1,"timeMs":0,"topMoves":[],"pvLine":[112],"policy":[]}
```
Move 112 = center (row 7, col 7 on 15×15 board) ✓

### ✅ Phase 3 — Trainer Lab (Python + PyTorch)

**Status: SCAFFOLD COMPLETE**

Full PyTorch training pipeline scaffolded and tested.

**Modules:**

| Path | Description |
|------|-------------|
| `config.py` | ModelConfig, TrainConfig, SelfPlayConfig (Pydantic) |
| `data/encoder.py` | board → [6, 16, 16] tensor (6 planes, padded to 16×16) |
| `data/augmentation.py` | D4 symmetry augmentation (8 transforms) |
| `data/dataset.py` | PositionDataset loading JSON position files |
| `models/blocks.py` | ResBlock + SEBlock (channel attention stub) |
| `models/resnet.py` | PolicyValueResNet: 8 ResBlocks, policy + value dual heads |
| `training/loss.py` | GomokuLoss: masked CE (policy) + MSE (value) |
| `training/metrics.py` | Top-1 policy accuracy + value MAE |
| `training/scheduler.py` | Cosine warmup LR scheduler |
| `training/trainer.py` | Full train loop: AMP, checkpointing, TensorBoard |
| `export/onnx_export.py` | ONNX export, fixed [B, 6, 16, 16] input, opset 17 |
| `evaluation/eval_script.py` | Policy accuracy + value calibration evaluation |
| `self_play/` | Phase-6 skeleton: player, replay buffer, pipeline |

**Tests: encoder (9), augmentation (6), resnet (6) — all pass**

### 🔲 Phase 4 — FastAPI Gateway

**Status: SCAFFOLDED** — engine_adapter subprocess stub ready, routes defined.

### 🔲 Phase 5 — Web MVP (React + TypeScript)

**Status: SCAFFOLDED** — Vite + React 18 + Tailwind project, SVG board component ready.

### 🔲 Phase 6 — Self-Play Pipeline

**Status: SKELETON** — player.py, replay_buffer.py, pipeline.py stubs in place.

---

## Quick Start

```bash
# Build C++ engine
make build-engine

# Run engine tests
make test-engine

# Start all services (Docker)
make dev-up

# Start FastAPI dev server
make api

# Start React dev server
make web
```

## Key Design Decisions

1. **Plain Makefile** — no nx/turborepo; polyglot stack (C++, Python, TS)
2. **JSON Schema → codegen** — single source of truth for all API contracts
3. **CMake + FetchContent** — no vcpkg dependency required
4. **CLI subprocess bridge** — C++ engine ↔ Python API via JSON stdin/stdout
5. **16×16 padded tensors** — all board sizes (7–16) padded to 16×16 at NN input
6. **D4 augmentation** — 8× data expansion via dihedral symmetry group
