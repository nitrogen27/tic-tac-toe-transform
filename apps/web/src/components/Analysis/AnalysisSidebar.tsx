/**
 * Sidebar wrapper showing analysis results.
 */

import { useGameState } from "../../store/gameStore";
import { useAnalysis } from "../../hooks/useAnalysis";
import { BestMoveDisplay } from "./BestMoveDisplay";
import { ValueBar } from "./ValueBar";
import { TopMovesTable } from "./TopMovesTable";
import { PVLine } from "./PVLine";

export function AnalysisSidebar() {
  const { boardSize, analysis, isAnalyzing } = useGameState();
  const { runAnalysis } = useAnalysis();

  return (
    <aside className="flex w-full flex-col gap-4 rounded-xl bg-white p-4 shadow-sm lg:w-80">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-neutral-500">
          Analysis
        </h3>
        <button
          onClick={runAnalysis}
          disabled={isAnalyzing}
          className="rounded bg-neutral-800 px-3 py-1 text-xs font-medium text-white transition hover:bg-neutral-700 disabled:opacity-50"
        >
          {isAnalyzing ? "Analyzing..." : "Analyze"}
        </button>
      </div>

      {analysis ? (
        <div className="flex flex-col gap-4">
          <BestMoveDisplay analysis={analysis} boardSize={boardSize} />
          <ValueBar value={analysis.value} confidence={analysis.confidence} />
          <TopMovesTable moves={analysis.topMoves} boardSize={boardSize} />
          <PVLine pvLine={analysis.pvLine} boardSize={boardSize} />
        </div>
      ) : (
        <p className="text-sm text-neutral-400">
          {isAnalyzing ? "Running engine analysis..." : "Click Analyze to evaluate the position."}
        </p>
      )}
    </aside>
  );
}
