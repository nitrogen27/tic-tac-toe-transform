/** Board coordinate conversion helpers. */

/**
 * Convert a flat cell index to (row, col).
 */
export function flatToRowCol(index: number, boardSize: number): [number, number] {
  return [Math.floor(index / boardSize), index % boardSize];
}

/**
 * Convert (row, col) to a flat cell index.
 */
export function rowColToFlat(row: number, col: number, boardSize: number): number {
  return row * boardSize + col;
}

/**
 * Return a column label (A, B, ... excluding I).
 * Standard gomoku notation skips the letter I to avoid confusion with 1.
 */
export function colLabel(col: number): string {
  const code = col < 8 ? 65 + col : 65 + col + 1; // skip 'I'
  return String.fromCharCode(code);
}

/**
 * Return a row label (1-based, from bottom).
 */
export function rowLabel(row: number, boardSize: number): string {
  return String(boardSize - row);
}

/**
 * Human-readable move string, e.g. "H8".
 */
export function moveToString(index: number, boardSize: number): string {
  const [r, c] = flatToRowCol(index, boardSize);
  return `${colLabel(c)}${rowLabel(r, boardSize)}`;
}

/**
 * Create an empty cells array.
 */
export function emptyCells(boardSize: number): (0 | 1 | -1)[] {
  return new Array(boardSize * boardSize).fill(0);
}
