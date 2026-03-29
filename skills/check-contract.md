# Skill: check-contract

## Описание

Проверяет что фронтенд и бэкенд используют одинаковые контракты.
Сравнивает TypeScript типы в `apps/web/src/api/types.ts` с Pydantic моделями
в `apps/api/src/gomoku_api/models/schemas.py` и JSON Schema в `packages/shared/schemas/`.

## Когда использовать

- После изменений в API схемах
- Перед коммитом с изменениями в `schemas.py` или `types.ts`
- При ошибках типов на фронте (422, unexpected shape)

## Алгоритм проверки

### 1. Сравнить Position

**TypeScript** (`api/types.ts`):
```ts
interface Position {
  boardSize: number;
  winLength: number;
  currentPlayer: 1 | -1;
  cells: (0|1|-1)[];
  lastMove: number;
  variant?: string;
}
```

**Python** (`models/schemas.py`): должен быть `Position(BaseModel)` с теми же полями.

**C++ CLI** (`types.hpp`): структура `Position` с `boardSize`, `winLength`, `cells`, `sideToMove` (=currentPlayer).

Проверить: `sideToMove` в C++ = `currentPlayer` в TS/Python.

### 2. Сравнить AnalyzeResponse

```ts
// TS:
interface AnalyzeResponse {
  bestMove: number; value: number; confidence: number; source: EngineSource;
  depth: number; nodesSearched: number; timeMs: number;
  topMoves: MoveCandidate[]; pvLine: number[]; policy?: number[];
}
```

Python должен иметь `AnalyzeResponse(BaseModel)` с теми же полями (snake_case → camelCase через alias).

### 3. Сравнить EngineSource

```ts
// TS:
type EngineSource = "safety_win"|"safety_block"|"safety_multi_block"|"vcf_win"|"vcf_defense"|"fork"|"alpha_beta";
```

Python должен иметь `EngineSource(str, Enum)` с теми же значениями.
C++ должен маппить `EngineSource::ALPHA_BETA` → строку `"alpha_beta"`.

### 4. Сравнить TrainJob

TS `TrainJob` ↔ Python `TrainJobResponse` — все поля, все enum-значения.

## Команды проверки

```bash
# Проверить что схемы не разошлись с codegen
make codegen
git diff packages/shared/generated/

# Запустить API тесты
cd apps/api && pytest tests/ -q

# Проверить TypeScript типы
cd apps/web && npm run type-check 2>/dev/null || npx tsc --noEmit
```

## Частые расхождения

| Проблема | Где смотреть |
|----------|-------------|
| `sideToMove` vs `currentPlayer` | `types.hpp` vs `schemas.py` |
| camelCase vs snake_case | Pydantic `model_config = {"populate_by_name": True}` |
| `alpha-beta` vs `alpha_beta` | EngineSource в C++ cli_main.cpp |
| `bestMove` vs `best_move` | Pydantic Field alias |
| Policy индексация | boardSize×boardSize vs всегда 256 |

## Шаблон исправления snake_case ↔ camelCase (Python)

```python
class AnalyzeResponse(BaseModel):
    best_move: int = Field(alias="bestMove")
    nodes_searched: int = Field(alias="nodesSearched")
    time_ms: int = Field(alias="timeMs")
    top_moves: list[MoveCandidate] = Field(alias="topMoves")
    pv_line: list[int] = Field(alias="pvLine")

    model_config = ConfigDict(populate_by_name=True)
```
