/**
 * Displays the best move from the latest analysis.
 */

import type { AnalyzeResponse } from "../../api/types";
import { moveToString } from "../../utils/boardUtils";

interface BestMoveDisplayProps {
  analysis: AnalyzeResponse;
  boardSize: number;
}

export function BestMoveDisplay({ analysis, boardSize }: BestMoveDisplayProps) {
  return (
    <div className="flex items-baseline gap-3">
      <span className="text-2xl font-bold font-mono">
        {moveToString(analysis.bestMove, boardSize)}
      </span>
      <span className="text-sm text-neutral-500">
        depth {analysis.depth} | {analysis.nodesSearched.toLocaleString()} nodes |{" "}
        {analysis.timeMs}ms | {analysis.source}
      </span>
    </div>
  );
}
