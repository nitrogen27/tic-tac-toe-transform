// Pattern recognition for gomoku (5-in-a-row)
// Scans all lines in 4 directions: horizontal, vertical, diagonal, anti-diagonal
// Classifies patterns: FIVE, OPEN_FOUR, HALF_FOUR, OPEN_THREE, HALF_THREE, OPEN_TWO, etc.

import { PATTERN_SCORES } from './config.mjs';

// Pattern type constants
export const P = {
  NONE:        0,
  OPEN_ONE:    1,
  HALF_TWO:    2,
  OPEN_TWO:    3,
  HALF_THREE:  4,
  OPEN_THREE:  5,
  HALF_FOUR:   6,
  OPEN_FOUR:   7,
  FIVE:        8,
};

const DIRS = [[0, 1], [1, 0], [1, 1], [1, -1]]; // H, V, D, AD

/**
 * Scan a single line segment and classify the pattern for `player`.
 *
 * Given a line of cells (from board), find all contiguous groups of `player`
 * stones and classify based on length and open/closed ends.
 *
 * Strategy: For each cell containing `player`'s stone, extend in both directions
 * of the line to find the maximal consecutive group, then check endpoints.
 *
 * @param {Int8Array} cells - board cells
 * @param {number} N - board size
 * @param {number} winLen - win length (typically 5)
 * @param {number} player - +1 or -1
 * @returns {PatternCounts} counts of each pattern type for this player
 */
export function scanAllPatterns(cells, N, winLen, player) {
  const counts = new Int32Array(9); // indexed by P.*
  const visited = new Uint8Array(N * N * 4); // visited per direction

  for (let dir = 0; dir < 4; dir++) {
    const [dr, dc] = DIRS[dir];
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const pos = r * N + c;
        if (cells[pos] !== player) continue;
        if (visited[pos * 4 + dir]) continue;

        // Extend forward from (r, c) in direction (dr, dc)
        let len = 1;
        let rr = r + dr, cc = c + dc;
        while (rr >= 0 && cc >= 0 && rr < N && cc < N && cells[rr * N + cc] === player) {
          visited[(rr * N + cc) * 4 + dir] = 1;
          len++;
          rr += dr;
          cc += dc;
        }
        visited[pos * 4 + dir] = 1;

        // Check if already a win
        if (len >= winLen) {
          counts[P.FIVE]++;
          continue;
        }

        // Check endpoints
        // Forward end: (rr, cc) is first cell after the group
        const fwdR = r + len * dr, fwdC = c + len * dc;
        const fwdOpen = fwdR >= 0 && fwdC >= 0 && fwdR < N && fwdC < N && cells[fwdR * N + fwdC] === 0;

        // Backward end: (r - dr, c - dc)
        const bwdR = r - dr, bwdC = c - dc;
        const bwdOpen = bwdR >= 0 && bwdC >= 0 && bwdR < N && bwdC < N && cells[bwdR * N + bwdC] === 0;

        const openEnds = (fwdOpen ? 1 : 0) + (bwdOpen ? 1 : 0);

        if (openEnds === 0) continue; // completely blocked — no threat

        // Check if the line segment has enough space to form winLen
        // Count available space (consecutive empty + own stones) in both directions
        if (!hasEnoughSpace(cells, N, winLen, player, r, c, dr, dc, len)) continue;

        classifyPattern(counts, len, openEnds);
      }
    }
  }

  // Also scan for gapped patterns (e.g., XX_XX, X_XXX, XXX_X)
  scanGappedPatterns(cells, N, winLen, player, counts);

  return counts;
}

/**
 * Check if there's enough space around a group to potentially form winLen.
 */
function hasEnoughSpace(cells, N, winLen, player, r, c, dr, dc, len) {
  let space = len;

  // Extend forward through empty cells
  let rr = r + len * dr, cc = c + len * dc;
  while (space < winLen && rr >= 0 && cc >= 0 && rr < N && cc < N) {
    const v = cells[rr * N + cc];
    if (v === player || v === 0) {
      space++;
      rr += dr;
      cc += dc;
    } else break;
  }

  // Extend backward through empty cells
  rr = r - dr;
  cc = c - dc;
  while (space < winLen && rr >= 0 && cc >= 0 && rr < N && cc < N) {
    const v = cells[rr * N + cc];
    if (v === player || v === 0) {
      space++;
      rr -= dr;
      cc -= dc;
    } else break;
  }

  return space >= winLen;
}

/**
 * Classify a contiguous group of `len` stones with `openEnds` open endpoints.
 */
function classifyPattern(counts, len, openEnds) {
  if (len >= 4) {
    if (openEnds === 2) counts[P.OPEN_FOUR]++;
    else counts[P.HALF_FOUR]++;
  } else if (len === 3) {
    if (openEnds === 2) counts[P.OPEN_THREE]++;
    else counts[P.HALF_THREE]++;
  } else if (len === 2) {
    if (openEnds === 2) counts[P.OPEN_TWO]++;
    else counts[P.HALF_TWO]++;
  } else if (len === 1) {
    if (openEnds === 2) counts[P.OPEN_ONE]++;
  }
}

/**
 * Scan for gapped patterns like X_XXX, XX_XX, XXX_X (one internal gap).
 * These are effectively HALF_FOUR since filling the gap creates FIVE.
 */
function scanGappedPatterns(cells, N, winLen, player, counts) {
  for (let dir = 0; dir < 4; dir++) {
    const [dr, dc] = DIRS[dir];

    // Iterate all possible lines in this direction
    const starts = getLineStarts(N, dr, dc);

    for (const [sr, sc] of starts) {
      // Collect the line
      const line = [];
      let rr = sr, cc = sc;
      while (rr >= 0 && cc >= 0 && rr < N && cc < N) {
        line.push({ r: rr, c: cc, v: cells[rr * N + cc] });
        rr += dr;
        cc += dc;
      }

      if (line.length < winLen) continue;

      // Sliding window of size winLen
      for (let i = 0; i <= line.length - winLen; i++) {
        const window = line.slice(i, i + winLen);
        let playerCount = 0;
        let emptyCount = 0;
        let opponentCount = 0;

        for (const cell of window) {
          if (cell.v === player) playerCount++;
          else if (cell.v === 0) emptyCount++;
          else opponentCount++;
        }

        if (opponentCount > 0) continue; // blocked by opponent

        // Pattern with exactly 1 gap (winLen-1 stones + 1 empty in window)
        if (playerCount === winLen - 1 && emptyCount === 1) {
          // This is effectively a four (one move to complete)
          // Check if it's already counted as a contiguous four
          // Only count if there's an internal gap (not just extending)
          let gapIdx = -1;
          for (let j = 0; j < window.length; j++) {
            if (window[j].v === 0) { gapIdx = j; break; }
          }
          // If gap is at the edge, it's already counted as HALF_FOUR from contiguous scan
          if (gapIdx > 0 && gapIdx < window.length - 1) {
            counts[P.HALF_FOUR]++;
          }
        }

        // Pattern with 2 gaps (winLen-2 stones + 2 empty)
        // These are open/half threes depending on endpoints
        if (playerCount === winLen - 2 && emptyCount === 2 && winLen >= 5) {
          // Check if gaps are internal (not at edges)
          let internalGaps = 0;
          for (let j = 1; j < window.length - 1; j++) {
            if (window[j].v === 0) internalGaps++;
          }
          if (internalGaps >= 1 && playerCount >= 3) {
            // Three with gap — treat as half-three (e.g., X_X_X or XX__X)
            // Only if filling any gap creates a four
            const beforeOpen = i > 0 && line[i - 1].v === 0;
            const afterOpen = i + winLen < line.length && line[i + winLen].v === 0;
            if (internalGaps === 1) {
              // e.g., XX_XX — filling gap creates FIVE directly
              // Already counted if winLen=5 and playerCount=4 above
              // For winLen=5, playerCount=3 with 1 internal gap:
              // e.g., X_XX_ — this creates four patterns
              if (beforeOpen || afterOpen) {
                counts[P.HALF_THREE]++;
              }
            }
          }
        }
      }
    }
  }
}

/**
 * Get all line starting positions for a given direction.
 */
function getLineStarts(N, dr, dc) {
  const starts = [];
  if (dr === 0 && dc === 1) {
    // Horizontal: start from leftmost column of each row
    for (let r = 0; r < N; r++) starts.push([r, 0]);
  } else if (dr === 1 && dc === 0) {
    // Vertical: start from top row of each column
    for (let c = 0; c < N; c++) starts.push([0, c]);
  } else if (dr === 1 && dc === 1) {
    // Diagonal ↘: top row + left column
    for (let c = 0; c < N; c++) starts.push([0, c]);
    for (let r = 1; r < N; r++) starts.push([r, 0]);
  } else if (dr === 1 && dc === -1) {
    // Anti-diagonal ↙: top row + right column
    for (let c = 0; c < N; c++) starts.push([0, c]);
    for (let r = 1; r < N; r++) starts.push([r, N - 1]);
  }
  return starts;
}

/**
 * Fast per-move pattern evaluation: count patterns for a player around a specific move.
 * Used for move ordering — evaluates the tactical impact of placing a stone at `pos`.
 *
 * @param {GomokuBoard} board - board state (move NOT yet made)
 * @param {number} pos - position to evaluate
 * @param {number} player - player making the move
 * @returns {number} tactical score for this move
 */
export function evaluateMovePatterns(board, pos, player) {
  const N = board.N;
  const winLen = board.winLen;
  const cells = board.cells;
  let score = 0;

  // Temporarily place the stone
  cells[pos] = player;

  const r = (pos / N) | 0;
  const c = pos % N;

  for (const [dr, dc] of DIRS) {
    // Count consecutive in both directions from pos
    let fwd = 0, bwd = 0;
    let rr, cc;

    // Forward
    rr = r + dr; cc = c + dc;
    while (rr >= 0 && cc >= 0 && rr < N && cc < N && cells[rr * N + cc] === player) {
      fwd++;
      rr += dr;
      cc += dc;
    }
    const fwdOpen = rr >= 0 && cc >= 0 && rr < N && cc < N && cells[rr * N + cc] === 0;

    // Backward
    rr = r - dr; cc = c - dc;
    while (rr >= 0 && cc >= 0 && rr < N && cc < N && cells[rr * N + cc] === player) {
      bwd++;
      rr -= dr;
      cc -= dc;
    }
    const bwdOpen = rr >= 0 && cc >= 0 && rr < N && cc < N && cells[rr * N + cc] === 0;

    const len = 1 + fwd + bwd;
    const openEnds = (fwdOpen ? 1 : 0) + (bwdOpen ? 1 : 0);

    if (len >= winLen) {
      score += PATTERN_SCORES.FIVE;
    } else if (len === winLen - 1) {
      score += openEnds === 2 ? PATTERN_SCORES.OPEN_FOUR : (openEnds === 1 ? PATTERN_SCORES.HALF_FOUR : 0);
    } else if (len === winLen - 2) {
      score += openEnds === 2 ? PATTERN_SCORES.OPEN_THREE : (openEnds === 1 ? PATTERN_SCORES.HALF_THREE : 0);
    } else if (len === winLen - 3) {
      score += openEnds === 2 ? PATTERN_SCORES.OPEN_TWO : (openEnds === 1 ? PATTERN_SCORES.HALF_TWO : 0);
    }
  }

  // Also check for gapped fours through this position
  score += evaluateGappedFours(cells, N, winLen, player, r, c);

  // Remove the stone
  cells[pos] = 0;

  return score;
}

/**
 * Check if placing at (r,c) creates gapped four patterns
 * (e.g., XX_XX where pos fills a gap creating 5, or XXX_X)
 */
function evaluateGappedFours(cells, N, winLen, player, r, c) {
  let score = 0;

  for (const [dr, dc] of DIRS) {
    // Check windows of size winLen that include (r, c)
    for (let offset = 0; offset < winLen; offset++) {
      const startR = r - offset * dr;
      const startC = c - offset * dc;

      // Check all winLen cells in this window
      let playerCount = 0;
      let emptyCount = 0;
      let valid = true;

      for (let k = 0; k < winLen; k++) {
        const kr = startR + k * dr;
        const kc = startC + k * dc;
        if (kr < 0 || kc < 0 || kr >= N || kc >= N) { valid = false; break; }
        const v = cells[kr * N + kc];
        if (v === player) playerCount++;
        else if (v === 0) emptyCount++;
        else { valid = false; break; } // opponent blocks
      }

      if (!valid) continue;

      // We already have the stone placed, so if all winLen cells are player, it's five
      // If winLen-1 are player and 1 is empty, it's a four
      // But we've already counted contiguous fours above
      // We want to find cases where the four is gapped
      // Since the stone IS placed, we look for winLen cells with exactly 1 gap
      // that aren't all contiguous
      if (playerCount === winLen) {
        // Already counted as FIVE above
      } else if (playerCount === winLen - 1 && emptyCount === 1) {
        // This is a four — check if the gap is internal (not counted by contiguous scan)
        // Find gap position
        for (let k = 0; k < winLen; k++) {
          const kr = startR + k * dr;
          const kc = startC + k * dc;
          if (cells[kr * N + kc] === 0) {
            // Gap at position k in window
            // It's "gapped" if there are player stones on both sides
            if (k > 0 && k < winLen - 1) {
              score += PATTERN_SCORES.HALF_FOUR * 0.5; // partial credit to avoid double-counting
            }
            break;
          }
        }
      }
    }
  }

  return score;
}

/**
 * Count immediate threats for a player (moves that create FIVE).
 * Used for VCF/VCT and safety checks.
 */
export function countThreats(board, player) {
  const N = board.N;
  const winLen = board.winLen;
  const cells = board.cells;
  let threats = 0;

  for (let pos = 0; pos < board.size; pos++) {
    if (cells[pos] !== 0) continue;

    cells[pos] = player;
    if (board.isWinningMove(pos, player)) {
      threats++;
    }
    cells[pos] = 0;
  }

  return threats;
}

/**
 * Find all moves that create an immediate win (four-in-a-row completion).
 */
export function findWinningMoves(board, player) {
  const cells = board.cells;
  const moves = [];

  for (let pos = 0; pos < board.size; pos++) {
    if (cells[pos] !== 0) continue;
    cells[pos] = player;
    if (board.isWinningMove(pos, player)) {
      moves.push(pos);
    }
    cells[pos] = 0;
  }

  return moves;
}

/**
 * Find all moves that create a four (one move away from win).
 * Returns moves that, after placement, leave exactly 1 empty cell to win.
 */
export function findFourCreatingMoves(board, player) {
  const N = board.N;
  const winLen = board.winLen;
  const cells = board.cells;
  const moves = [];

  for (let pos = 0; pos < board.size; pos++) {
    if (cells[pos] !== 0) continue;

    cells[pos] = player;

    // Check if this creates any winning threats
    let createsWinningThreat = false;
    for (let pos2 = 0; pos2 < board.size; pos2++) {
      if (cells[pos2] !== 0) continue;
      cells[pos2] = player;
      if (board.isWinningMove(pos2, player)) {
        createsWinningThreat = true;
        cells[pos2] = 0;
        break;
      }
      cells[pos2] = 0;
    }

    cells[pos] = 0;

    if (createsWinningThreat) {
      moves.push(pos);
    }
  }

  return moves;
}

/**
 * Fast: count how many winning moves opponent would have after we make a move.
 */
export function countOpponentThreatsAfterMove(board, pos, player) {
  const cells = board.cells;
  cells[pos] = player;
  const threats = findWinningMoves(board, -player);
  cells[pos] = 0;
  return threats.length;
}

/**
 * Evaluate pattern score for a position (both players).
 * Returns { myScore, oppScore, myCounts, oppCounts }
 */
export function evaluatePatterns(board, player) {
  const cells = board.cells;
  const N = board.N;
  const winLen = board.winLen;

  const myCounts = scanAllPatterns(cells, N, winLen, player);
  const oppCounts = scanAllPatterns(cells, N, winLen, -player);

  const myScore = patternCountsToScore(myCounts);
  const oppScore = patternCountsToScore(oppCounts);

  return { myScore, oppScore, myCounts, oppCounts };
}

/**
 * Convert pattern counts to numeric score.
 */
export function patternCountsToScore(counts) {
  let score = 0;
  score += counts[P.FIVE]       * PATTERN_SCORES.FIVE;
  score += counts[P.OPEN_FOUR]  * PATTERN_SCORES.OPEN_FOUR;
  score += counts[P.HALF_FOUR]  * PATTERN_SCORES.HALF_FOUR;
  score += counts[P.OPEN_THREE] * PATTERN_SCORES.OPEN_THREE;
  score += counts[P.HALF_THREE] * PATTERN_SCORES.HALF_THREE;
  score += counts[P.OPEN_TWO]   * PATTERN_SCORES.OPEN_TWO;
  score += counts[P.HALF_TWO]   * PATTERN_SCORES.HALF_TWO;
  score += counts[P.OPEN_ONE]   * PATTERN_SCORES.OPEN_ONE;
  return score;
}
