/**
 * Move list display.
 */

import { useGameState } from "../../store/gameStore";
import { moveToString } from "../../utils/boardUtils";

export function MoveHistory() {
  const { moveHistory, boardSize } = useGameState();

  if (moveHistory.length === 0) {
    return (
      <p className="text-xs text-neutral-400">No moves yet.</p>
    );
  }

  return (
    <div className="space-y-1">
      <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-500">
        Moves
      </h4>
      <div className="flex flex-wrap gap-1 font-mono text-xs">
        {moveHistory.map((m, i) => (
          <span
            key={i}
            className={`rounded px-1.5 py-0.5 ${
              i % 2 === 0
                ? "bg-neutral-800 text-white"
                : "bg-neutral-200 text-neutral-700"
            }`}
          >
            {i + 1}. {moveToString(m, boardSize)}
          </span>
        ))}
      </div>
    </div>
  );
}
