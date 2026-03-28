import { BOARD_SIZES } from "../../utils/constants";
import { useGame } from "../../hooks/useGame";

export function BoardSizeSelector() {
  const { boardSize, setBoardSize } = useGame();

  return (
    <div className="flex items-center gap-2">
      <label htmlFor="board-size" className="text-sm font-medium text-neutral-600">
        Board
      </label>
      <select
        id="board-size"
        value={boardSize}
        onChange={(e) => setBoardSize(Number(e.target.value))}
        className="rounded border border-neutral-300 bg-white px-2 py-1 text-sm"
      >
        {BOARD_SIZES.map((s) => (
          <option key={s} value={s}>
            {s}x{s}
          </option>
        ))}
      </select>
    </div>
  );
}
