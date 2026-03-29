# Integration Contracts

Контракты между сервисами — единый источник истины для API-границ.

---

## C++ Engine CLI ↔ FastAPI

**Протокол:** JSON stdin → JSON stdout, newline-delimited

### Запрос (общая структура)
```json
{
  "command": "best-move|analyze|suggest|info",
  "position": {
    "boardSize": 15,
    "winLength": 5,
    "cells": [],
    "sideToMove": 1,
    "moveCount": 0,
    "lastMove": -1,
    "moveHistory": []
  },
  "options": {
    "topK": 5,
    "timeLimitMs": 3000,
    "includePv": true
  }
}
```

### Ответ (EngineResult)
```json
{
  "bestMove": 112,
  "source": "alpha-beta",
  "value": 0.0,
  "confidence": 0.0,
  "depth": 6,
  "nodesSearched": 15420,
  "timeMs": 12,
  "topMoves": [{"move": 112, "score": 0.02, "row": 7, "col": 7}],
  "pvLine": [112, 113, 97],
  "policy": []
}
```

### EngineSource enum
```
"safety_win" | "safety_block" | "safety_multi_block" |
"vcf_win" | "vcf_defense" | "fork" | "alpha_beta"
```

**Fallback:** если бинарник не найден → центральная клетка `(boardSize*boardSize)//2`

---

## FastAPI REST ↔ React Web

**Base URL:** `http://localhost:8000`

### POST /analyze
```json
// Request
{"position": {...}, "topK": 5, "timeLimitMs": 3000, "includePv": true}
// Response
{"bestMove": 112, "value": 0.0, "confidence": 0.0, "source": "alpha_beta",
 "depth": 6, "nodesSearched": 15420, "timeMs": 12,
 "topMoves": [...], "pvLine": [...], "policy": [...], "engineMeta": {...}}
```

### POST /best-move
```json
// Request
{"position": {...}, "timeLimitMs": 1000}
// Response
{"move": 112, "row": 7, "col": 7, "value": 0.0, "source": "alpha_beta"}
```

### POST /suggest
```json
// Request
{"position": {...}, "topK": 5}
// Response
{"suggestions": [{"move": 112, "score": 0.02, "confidence": 0.8, "row": 7, "col": 7}]}
```

### GET /engine/info
```json
{"version": "0.1.0", "supportedBoardSizes": [7,8,9,10,11,12,13,14,15,16], "capabilities": [...]}
```

### POST /train/jobs
```json
// Request
{"variant": "15x15", "batchSize": 256, "epochs": 30, "lr": 0.001,
 "selfPlayGames": 200, "selfPlaySimulations": 400}
// Response
{"jobId": "uuid", "variant": "15x15", "status": "queued",
 "createdAt": "ISO8601", "updatedAt": null}
```

### GET /train/jobs/{id}
```json
{"jobId": "uuid", "status": "running",
 "progress": {"phase": "training", "epoch": 5, "totalEpochs": 30,
               "loss": 1.2, "policyAccuracy": 0.45, "valueMae": 0.3,
               "gamesGenerated": 150, "positionsCollected": 12000, "elapsedSec": 120}}
```

### GET /health
```json
{"status": "ok", "engine_available": true}
```

---

## Position (общий формат)

```ts
interface Position {
  boardSize: number;       // 7–16
  winLength: number;       // 5 (константа)
  currentPlayer: 1 | -1;  // 1=черные, -1=белые
  cells: (0|1|-1)[];       // flat массив N*N
  lastMove: number;        // flat index или -1
  variant?: string;        // опционально "15x15"
}
```

**Mapping cells↔sideToMove:** `currentPlayer` (TS) = `sideToMove` (C++ CLI, значение ±1)

---

## PyTorch Trainer ↔ ONNX

**Входной тензор:** `[B, 6, 16, 16]` float32

| Плоскость | Содержимое |
|-----------|------------|
| 0 | Камни текущего игрока |
| 1 | Камни противника |
| 2 | Маска легальных ходов |
| 3 | Последний ход (бинарно) |
| 4 | Сторона хода (константа ±1) |
| 5 | Маска размера доски (1 внутри boardSize×boardSize) |

**Выходные тензоры:**
- `"policy_logits"` — `[B, 256]` (не softmax, сырые логиты)
- `"value"` — `[B, 1]` ∈ [-1, 1] после tanh

**Формат данных для обучения:**
```json
[{"board_size": 15, "board": [[...]], "current_player": 1,
  "last_move": [7, 7], "policy": [0.0, ..., 0.95, ...], "value": 0.12}]
```

---

## Схемы JSON (packages/shared/)

Файлы: `position.schema.json`, `analyze.schema.json`, `train_job.schema.json`,
`model_artifact.schema.json`, `enums.schema.json`

Генерация: `make codegen` → `packages/shared/generated/ts/index.ts` + `python/schema_registry.py`
