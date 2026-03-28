/**
 * Board interaction handlers (click, hover).
 */

import { useState, useCallback, type MouseEvent } from "react";
import { useGame } from "./useGame";
import { BOARD_PADDING } from "../utils/constants";

export function useBoard() {
  const { boardSize, cells, placeStone, mode } = useGame();
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  /**
   * Convert an SVG mouse event to a board (row, col) index.
   * Returns -1 if outside the grid.
   */
  const eventToIndex = useCallback(
    (e: MouseEvent<SVGSVGElement>): number => {
      const svg = e.currentTarget;
      const rect = svg.getBoundingClientRect();
      const totalSize = boardSize - 1 + BOARD_PADDING * 2;
      const cellPx = rect.width / totalSize;

      const col = Math.round((e.clientX - rect.left) / cellPx - BOARD_PADDING);
      const row = Math.round((e.clientY - rect.top) / cellPx - BOARD_PADDING);

      if (row < 0 || row >= boardSize || col < 0 || col >= boardSize) return -1;
      return row * boardSize + col;
    },
    [boardSize]
  );

  const handleClick = useCallback(
    (e: MouseEvent<SVGSVGElement>) => {
      if (mode !== "play") return;
      const idx = eventToIndex(e);
      if (idx >= 0 && cells[idx] === 0) {
        placeStone(idx);
      }
    },
    [mode, eventToIndex, cells, placeStone]
  );

  const handleMouseMove = useCallback(
    (e: MouseEvent<SVGSVGElement>) => {
      const idx = eventToIndex(e);
      setHoverIndex(idx >= 0 && cells[idx] === 0 ? idx : null);
    },
    [eventToIndex, cells]
  );

  const handleMouseLeave = useCallback(() => {
    setHoverIndex(null);
  }, []);

  return { hoverIndex, handleClick, handleMouseMove, handleMouseLeave };
}
