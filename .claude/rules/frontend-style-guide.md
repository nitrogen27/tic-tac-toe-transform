# Frontend Style Guide — Gomoku Platform V3

## Stack

- **Framework:** React 18 + TypeScript
- **Styling:** Tailwind CSS v3
- **Build:** Vite (порт 5174)
- **Board rendering:** SVG (не canvas, не DOM-элементы)

---

## Цвета

### Tailwind custom tokens (tailwind.config.ts)

```ts
colors: {
  board: {
    bg:   "#DCB35C",   // фон доски (светло-коричневый)
    line: "#8B6914",   // линии сетки
  },
  stone: {
    black: "#1a1a1a",  // чёрный камень
    white: "#f5f5f5",  // белый камень
  },
}
```

### JS-константы (`src/utils/constants.ts`)

```ts
COLORS = {
  boardBg:         "#DCB35C",  // фон SVG доски
  boardLine:       "#8B6914",  // линии и звёздные точки
  stoneBlack:      "#1a1a1a",  // камень игрока 1
  stoneWhite:      "#f5f5f5",  // камень игрока -1
  lastMoveMarker:  "#e74c3c",  // маркер последнего хода (красный)
  hintGood:        "#27ae60",  // хороший ход (зелёный)
  hintNeutral:     "#f39c12",  // нейтральный ход (жёлтый)
  hintBad:         "#e74c3c",  // плохой ход (красный)
}
```

---

## SVG Board

### Единицы измерения

- Координатная система: одна ячейка = 1 SVG unit
- boardSize=15 → maxCoord=14 → viewBox от -pad до 14+pad
- `BOARD_PADDING = 0.8` (отступ от края до крайней линии)
- `viewBox="${-pad} ${-pad} ${viewSize} ${viewSize}"`

### Камни

- Радиус: `0.46` от ячейки
- Черный: `fill="#1a1a1a"`, `stroke="#000"`, `strokeWidth=0.03`
- Белый: `fill="#f5f5f5"`, `stroke="#999"`, `strokeWidth=0.03`
- Ghost (hover): `opacity=0.35`

### Линии сетки

- `strokeWidth=0.03` для линий
- Звёздные точки: `r=0.09`, `fill=COLORS.boardLine`
- Координаты (A–P, 1–16): `fontSize=0.4`, `fill="#666"`

### Маркер последнего хода

- Маленький красный кружок: `r=0.15`, `fill="#e74c3c"`

---

## Размеры доски

```ts
BOARD_SIZES = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
DEFAULT_BOARD_SIZE = 15
WIN_LENGTH = 5
```

---

## Компонентная иерархия

```
App
└── GameProvider (Context)
    └── MainLayout
        ├── Header
        ├── Board (SVG)
        │   ├── Stone[]
        │   ├── Coordinates
        │   └── LastMoveMarker
        ├── AnalysisSidebar
        │   ├── BestMoveDisplay
        │   ├── ValueBar         ← value ∈ [-1,1], прогресс-бар
        │   ├── TopMovesTable    ← топ-5 ходов с score
        │   └── PVLine           ← цепочка principal variation
        └── GameControls
            └── BoardSizeSelector
```

---

## Game State

```ts
mode: "play" | "analyze" | "review"
currentPlayer: 1 | -1   // 1 = черные, -1 = белые
cells: CellValue[]       // 0=пусто, 1=черный, -1=белый
```

### Actions

```ts
PLACE_STONE    // idx → меняет cells, переключает currentPlayer
UNDO           // убирает последний ход
NEW_GAME       // сброс state
SET_BOARD_SIZE // сброс + новый размер
SET_MODE       // play|analyze|review
SET_ANALYSIS   // сохранить AnalyzeResponse
SET_ANALYZING  // spinner флаг
```

---

## Типографика

- Монospace шрифт для координат на доске
- System UI для UI-элементов (кнопки, таблицы)
- Координаты колонок: A-P (пропускается I)
- Координаты строк: 1–16 (снизу вверх)

---

## Tailwind-конвенции

- Layout: `flex`, `grid`, `gap-*`
- Адаптивность: `md:`, `lg:` breakpoints
- Максимальная ширина доски: `max-w-[600px]`
- Board SVG: `w-full max-w-[600px] select-none`
- Кнопки: `px-4 py-2 rounded-md bg-board-line text-white hover:opacity-80`

---

## API-интерфейсы (для компонентов)

```ts
// apps/web/src/api/types.ts
EngineSource = "safety_win"|"safety_block"|"vcf_win"|"vcf_defense"|"fork"|"alpha_beta"

AnalyzeResponse = {
  bestMove: number,     // flat index
  value: number,        // ∈ [-1, 1]
  confidence: number,
  source: EngineSource,
  topMoves: MoveCandidate[],  // [{move, score, row, col}]
  pvLine: number[],           // flat indices
}
```

---

## Правила написания компонентов

1. Все компоненты — functional, никаких class components
2. Состояние — только через `gameStore` (useGameState, useGameDispatch)
3. Локальный UI-state (hover, loading) — `useState` внутри компонента
4. SVG пиксели в board-units (не px), конвертация через `eventToIndex`
5. Не использовать `any` — все типы из `api/types.ts`
6. Не импортировать напрямую из `store/` в `components/` — только через `hooks/`
