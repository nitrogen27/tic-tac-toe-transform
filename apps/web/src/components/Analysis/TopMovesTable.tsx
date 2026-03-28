/**
 * Table of top move candidates from the engine analysis.
 */

import type { MoveCandidate } from "../../api/types";
import { moveToString } from "../../utils/boardUtils";

interface TopMovesTableProps {
  moves: MoveCandidate[];
  boardSize: number;
}

export function TopMovesTable({ moves, boardSize }: TopMovesTableProps) {
  if (moves.length === 0) return null;

  return (
    <div className="space-y-1">
      <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-500">
        Top Moves
      </h4>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-neutral-200 text-left text-xs text-neutral-500">
            <th className="py-1 pr-2">#</th>
            <th className="py-1 pr-2">Move</th>
            <th className="py-1 pr-2">Score</th>
            <th className="py-1">Conf</th>
          </tr>
        </thead>
        <tbody>
          {moves.map((m, i) => (
            <tr key={i} className="border-b border-neutral-100">
              <td className="py-1 pr-2 text-neutral-400">{i + 1}</td>
              <td className="py-1 pr-2 font-mono font-medium">
                {moveToString(m.move, boardSize)}
              </td>
              <td className="py-1 pr-2 font-mono">
                {m.score >= 0 ? "+" : ""}
                {m.score.toFixed(3)}
              </td>
              <td className="py-1 font-mono text-neutral-500">
                {m.confidence != null ? `${(m.confidence * 100).toFixed(1)}%` : "--"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
