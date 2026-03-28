/**
 * New game, undo, pass, and analyze action buttons.
 */

import { useGame } from "../../hooks/useGame";
import { useAnalysis } from "../../hooks/useAnalysis";
import { BoardSizeSelector } from "../Board/BoardSizeSelector";

export function GameControls() {
  const { undo, newGame, moveHistory, isAnalyzing } = useGame();
  const { runAnalysis } = useAnalysis();

  return (
    <div className="flex flex-wrap items-center gap-2">
      <BoardSizeSelector />

      <button
        onClick={() => newGame()}
        className="rounded border border-neutral-300 bg-white px-3 py-1 text-sm font-medium transition hover:bg-neutral-50"
      >
        New Game
      </button>

      <button
        onClick={undo}
        disabled={moveHistory.length === 0}
        className="rounded border border-neutral-300 bg-white px-3 py-1 text-sm font-medium transition hover:bg-neutral-50 disabled:opacity-40"
      >
        Undo
      </button>

      <button
        onClick={runAnalysis}
        disabled={isAnalyzing}
        className="rounded bg-blue-600 px-3 py-1 text-sm font-medium text-white transition hover:bg-blue-500 disabled:opacity-50"
      >
        {isAnalyzing ? "Analyzing..." : "Analyze"}
      </button>
    </div>
  );
}
