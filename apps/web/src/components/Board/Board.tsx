/**
 * SVG Gomoku board renderer.
 * Supports board sizes from 7 to 16.
 */

import { useGameState } from "../../store/gameStore";
import { useBoard } from "../../hooks/useBoard";
import { flatToRowCol } from "../../utils/boardUtils";
import { COLORS, BOARD_PADDING } from "../../utils/constants";
import { Stone } from "./Stone";
import { Coordinates } from "./Coordinates";
import { LastMoveMarker } from "./LastMoveMarker";

/** Star-point positions for common board sizes. */
const STAR_POINTS: Record<number, [number, number][]> = {
  15: [
    [3, 3], [3, 7], [3, 11],
    [7, 3], [7, 7], [7, 11],
    [11, 3], [11, 7], [11, 11],
  ],
  13: [
    [3, 3], [3, 6], [3, 9],
    [6, 3], [6, 6], [6, 9],
    [9, 3], [9, 6], [9, 9],
  ],
  9: [
    [2, 2], [2, 6],
    [4, 4],
    [6, 2], [6, 6],
  ],
};

export function Board() {
  const { boardSize, cells, lastMove, currentPlayer } = useGameState();
  const { hoverIndex, handleClick, handleMouseMove, handleMouseLeave } = useBoard();

  const pad = BOARD_PADDING;
  const maxCoord = boardSize - 1;
  const viewSize = maxCoord + pad * 2;

  // Grid lines
  const lines: JSX.Element[] = [];
  for (let i = 0; i <= maxCoord; i++) {
    lines.push(
      <line key={`h${i}`} x1={0} y1={i} x2={maxCoord} y2={i} stroke={COLORS.boardLine} strokeWidth={0.03} />,
      <line key={`v${i}`} x1={i} y1={0} x2={i} y2={maxCoord} stroke={COLORS.boardLine} strokeWidth={0.03} />
    );
  }

  // Star points
  const stars = STAR_POINTS[boardSize] ?? [];

  // Stones
  const stones: JSX.Element[] = [];
  for (let idx = 0; idx < cells.length; idx++) {
    if (cells[idx] !== 0) {
      const [r, c] = flatToRowCol(idx, boardSize);
      stones.push(<Stone key={idx} row={r} col={c} value={cells[idx]} />);
    }
  }

  // Ghost stone for hover
  let ghost: JSX.Element | null = null;
  if (hoverIndex !== null && hoverIndex >= 0 && cells[hoverIndex] === 0) {
    const [gr, gc] = flatToRowCol(hoverIndex, boardSize);
    ghost = <Stone row={gr} col={gc} value={currentPlayer} isGhost />;
  }

  return (
    <svg
      viewBox={`${-pad} ${-pad} ${viewSize} ${viewSize}`}
      className="w-full max-w-[600px] select-none"
      onClick={handleClick}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      {/* Board background */}
      <rect
        x={-pad}
        y={-pad}
        width={viewSize}
        height={viewSize}
        fill={COLORS.boardBg}
        rx={0.3}
      />

      {/* Grid */}
      {lines}

      {/* Star points */}
      {stars.map(([r, c]) => (
        <circle key={`star-${r}-${c}`} cx={c} cy={r} r={0.09} fill={COLORS.boardLine} />
      ))}

      {/* Coordinates */}
      <Coordinates boardSize={boardSize} />

      {/* Stones */}
      {stones}

      {/* Ghost */}
      {ghost}

      {/* Last move marker */}
      <LastMoveMarker lastMove={lastMove} boardSize={boardSize} cells={cells} />
    </svg>
  );
}
