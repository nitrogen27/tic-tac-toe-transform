/**
 * Displays the principal variation (best move sequence).
 */

import { moveToString } from "../../utils/boardUtils";

interface PVLineProps {
  pvLine: number[];
  boardSize: number;
}

export function PVLine({ pvLine, boardSize }: PVLineProps) {
  if (pvLine.length === 0) return null;

  return (
    <div className="space-y-1">
      <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-500">
        Principal Variation
      </h4>
      <p className="font-mono text-sm text-neutral-700">
        {pvLine.map((m, i) => (
          <span key={i}>
            {i > 0 && " "}
            <span className={i === 0 ? "font-bold" : ""}>{moveToString(m, boardSize)}</span>
          </span>
        ))}
      </p>
    </div>
  );
}
