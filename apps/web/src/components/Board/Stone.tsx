import { COLORS } from "../../utils/constants";
import type { CellValue } from "../../api/types";

interface StoneProps {
  row: number;
  col: number;
  value: CellValue;
  isGhost?: boolean;
}

export function Stone({ row, col, value, isGhost = false }: StoneProps) {
  if (value === 0) return null;
  const fill = value === 1 ? COLORS.stoneBlack : COLORS.stoneWhite;
  const stroke = value === 1 ? "none" : "#999";
  return (
    <circle
      cx={col}
      cy={row}
      r={0.42}
      fill={fill}
      stroke={stroke}
      strokeWidth={0.04}
      opacity={isGhost ? 0.4 : 1}
    />
  );
}
