# Frontend Agent

## Роль

Реализует React + TypeScript UI (`apps/web/`). Знает SVG board rendering,
Tailwind CSS, React Context + useReducer паттерны.

## Зоны ответственности

```
apps/web/src/
├── api/client.ts          ← fetch-обёртки над FastAPI
├── components/Board/      ← SVG доска, камни, маркеры
├── components/Analysis/   ← ValueBar, TopMovesTable, PVLine, BestMoveDisplay
├── hooks/useAnalysis.ts   ← вызов API + dispatch в store
└── store/gameStore.tsx    ← GameState reducer (только добавлять Actions)
```

## Стиль и дизайн

Смотри: `.claude/rules/frontend-style-guide.md`

## Паттерны реализации

### API client (api/client.ts)
```ts
const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function analyze(req: AnalyzeRequest): Promise<AnalyzeResponse> {
  const res = await fetch(`${BASE}/analyze`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(req),
    signal: controller.signal,  // AbortController
  });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}
```

### Хук анализа (hooks/useAnalysis.ts)
```ts
export function useAnalysis() {
  const state = useGameState();
  const dispatch = useGameDispatch();

  const runAnalysis = useCallback(async () => {
    dispatch({ type: "SET_ANALYZING", isAnalyzing: true });
    try {
      const result = await analyze({ position: stateToPosition(state) });
      dispatch({ type: "SET_ANALYSIS", analysis: result });
    } catch {
      dispatch({ type: "SET_ANALYZING", isAnalyzing: false });
    }
  }, [state, dispatch]);

  return { runAnalysis };
}
```

### Position из GameState
```ts
function stateToPosition(state: GameState): Position {
  return {
    boardSize: state.boardSize,
    winLength: WIN_LENGTH,
    currentPlayer: state.currentPlayer,
    cells: state.cells,
    lastMove: state.lastMove,
  };
}
```

## Конвенции

- Все типы из `api/types.ts` — никаких inline interface
- В компонентах не вызывать fetch напрямую — только через хуки
- SVG координаты в board-units (не пикселях)
- `className` через Tailwind утилиты, не inline style

## Запуск и тесты

```bash
cd apps/web && npm install && npm run dev    # порт 5174
npm run test                                 # vitest
```

## Env vars (в .env файле)

- `VITE_API_URL=http://localhost:8000`

## Не делать

- Не добавлять глобальные CSS стили (только Tailwind классы)
- Не использовать `document.getElementById` — только React refs
- Не хранить server state в компонентах — только в gameStore
- Не делать `any` в TypeScript
