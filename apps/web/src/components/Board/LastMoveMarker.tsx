import { COLORS } from "../../utils/constants";
import { flatToRowCol } from "../../utils/boardUtils";
import type { CellValue } from "../../api/types";

interface LastMoveMarkerProps {
  lastMove: number;
  boardSize: number;
  cells: CellValue[];
}

export function LastMoveMarker({ lastMove, boardSize, cells }: LastMoveMarkerProps) {
  if (lastMove < 0) return null;
  const [r, c] = flatToRowCol(lastMove, boardSize);
  const stoneColor = cells[lastMove];
  const markerColor = stoneColor === 1 ? COLORS.stoneWhite : COLORS.stoneBlack;

  return (
    <circle
      cx={c}
      cy={r}
      r={0.12}
      fill={markerColor}
      opacity={0.85}
      pointerEvents="none"
    />
  );
}
