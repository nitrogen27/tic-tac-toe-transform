/** Application-wide constants. */

export const BOARD_SIZES = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16] as const;

export const DEFAULT_BOARD_SIZE = 15;
export const WIN_LENGTH = 5;

export const COLORS = {
  boardBg: "#DCB35C",
  boardLine: "#8B6914",
  stoneBlack: "#1a1a1a",
  stoneWhite: "#f5f5f5",
  lastMoveMarker: "#e74c3c",
  hintGood: "#27ae60",
  hintNeutral: "#f39c12",
  hintBad: "#e74c3c",
} as const;

/** Padding around the SVG board (in board-cell units). */
export const BOARD_PADDING = 0.8;
