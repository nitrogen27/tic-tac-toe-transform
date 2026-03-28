// Static evaluation function for gomoku positions
// Aggregates pattern scores with combination bonuses

import { scanAllPatterns, P, patternCountsToScore } from './patterns.mjs';
import { COMBO_BONUSES } from './config.mjs';

// Score normalization: map raw score to [-1, +1] range for value estimation
const SCORE_SCALE = 200_000; // scores beyond this are essentially won/lost

/**
 * Evaluate a board position from `player`'s perspective.
 * Returns a score where positive = good for player, negative = bad.
 *
 * @param {GomokuBoard} board
 * @param {number} player - +1 or -1
 * @returns {number} evaluation score (raw, not normalized)
 */
export function evaluate(board, player) {
  const cells = board.cells;
  const N = board.N;
  const winLen = board.winLen;

  const myCounts = scanAllPatterns(cells, N, winLen, player);
  const oppCounts = scanAllPatterns(cells, N, winLen, -player);

  // Check for immediate win/loss
  if (myCounts[P.FIVE] > 0) return 1_000_000;
  if (oppCounts[P.FIVE] > 0) return -1_000_000;

  let myScore = patternCountsToScore(myCounts);
  let oppScore = patternCountsToScore(oppCounts);

  // Combination bonuses — multiple simultaneous threats
  myScore += comboBonuses(myCounts);
  oppScore += comboBonuses(oppCounts);

  // Open four is unstoppable — equivalent to a win
  if (myCounts[P.OPEN_FOUR] > 0) return 900_000;
  if (oppCounts[P.OPEN_FOUR] > 0) return -900_000;

  // Attacker advantage: it's player's turn, so their threats are more valuable
  // Multiply own threats slightly and discount opponent threats
  const score = myScore * 1.05 - oppScore;

  // Center control bonus (minor)
  const centerBonus = centerControlScore(board, player);

  return score + centerBonus;
}

/**
 * Combination bonuses for multiple simultaneous threats.
 */
function comboBonuses(counts) {
  let bonus = 0;

  // Double half-four = effectively open four (opponent can only block one)
  if (counts[P.HALF_FOUR] >= 2) {
    bonus += COMBO_BONUSES.DOUBLE_HALF_FOUR;
  }

  // Open three + half four = unstoppable (blocking four allows open-four from three)
  if (counts[P.OPEN_THREE] >= 1 && counts[P.HALF_FOUR] >= 1) {
    bonus += COMBO_BONUSES.OPEN_THREE_HALF_FOUR;
  }

  // Double open three = very strong (blocking one allows the other to become open four)
  if (counts[P.OPEN_THREE] >= 2) {
    bonus += COMBO_BONUSES.DOUBLE_OPEN_THREE;
  }

  // Triple half three = complex multi-threat
  if (counts[P.HALF_THREE] >= 3) {
    bonus += COMBO_BONUSES.TRIPLE_HALF_THREE;
  }

  return bonus;
}

/**
 * Minor bonus for center control.
 * Stones closer to center are slightly more valuable.
 */
function centerControlScore(board, player) {
  const N = board.N;
  const mid = (N - 1) / 2;
  const cells = board.cells;
  let score = 0;

  for (let i = 0; i < board.size; i++) {
    if (cells[i] === 0) continue;
    const r = (i / N) | 0;
    const c = i % N;
    const dist = Math.abs(r - mid) + Math.abs(c - mid);
    const centerValue = Math.max(0, N - dist) * 2;
    if (cells[i] === player) score += centerValue;
    else score -= centerValue;
  }

  return score;
}

/**
 * Normalize raw evaluation score to [-1, +1] range.
 * Used for value output to match NN value format.
 */
export function normalizeScore(score) {
  if (score >= 1_000_000) return 1.0;
  if (score <= -1_000_000) return -1.0;
  return Math.tanh(score / SCORE_SCALE);
}

/**
 * Quick evaluation for move ordering — how good is a move without deep search.
 * Evaluates the position AFTER the move is made.
 */
export function quickMoveEval(board, pos, player) {
  const cells = board.cells;
  const N = board.N;
  const winLen = board.winLen;

  // Place stone temporarily
  cells[pos] = player;

  // Quick pattern scan around this move (not full board)
  let score = 0;
  const r = (pos / N) | 0;
  const c = pos % N;
  const DIRS = [[0, 1], [1, 0], [1, 1], [1, -1]];

  for (const [dr, dc] of DIRS) {
    // Count consecutive own stones in this direction
    let fwd = 0, bwd = 0;
    let rr, cc;

    rr = r + dr; cc = c + dc;
    while (rr >= 0 && cc >= 0 && rr < N && cc < N && cells[rr * N + cc] === player) {
      fwd++; rr += dr; cc += dc;
    }
    const fwdEmpty = rr >= 0 && cc >= 0 && rr < N && cc < N && cells[rr * N + cc] === 0;

    rr = r - dr; cc = c - dc;
    while (rr >= 0 && cc >= 0 && rr < N && cc < N && cells[rr * N + cc] === player) {
      bwd++; rr -= dr; cc -= dc;
    }
    const bwdEmpty = rr >= 0 && cc >= 0 && rr < N && cc < N && cells[rr * N + cc] === 0;

    const len = 1 + fwd + bwd;
    const openEnds = (fwdEmpty ? 1 : 0) + (bwdEmpty ? 1 : 0);

    if (len >= winLen) score += 1_000_000;
    else if (len === winLen - 1 && openEnds >= 1) score += openEnds === 2 ? 100_000 : 10_000;
    else if (len === winLen - 2 && openEnds >= 1) score += openEnds === 2 ? 5_000 : 1_000;
    else if (len === winLen - 3 && openEnds >= 1) score += openEnds === 2 ? 500 : 100;
  }

  // Also evaluate defensive value: count opponent threats this blocks
  cells[pos] = -player;
  let defScore = 0;
  for (const [dr, dc] of DIRS) {
    let fwd = 0, bwd = 0;
    let rr, cc;

    rr = r + dr; cc = c + dc;
    while (rr >= 0 && cc >= 0 && rr < N && cc < N && cells[rr * N + cc] === -player) {
      fwd++; rr += dr; cc += dc;
    }

    rr = r - dr; cc = c - dc;
    while (rr >= 0 && cc >= 0 && rr < N && cc < N && cells[rr * N + cc] === -player) {
      bwd++; rr -= dr; cc -= dc;
    }

    const oppLen = 1 + fwd + bwd;
    if (oppLen >= winLen) defScore += 500_000;
    else if (oppLen === winLen - 1) defScore += 50_000;
    else if (oppLen === winLen - 2) defScore += 2_500;
  }

  cells[pos] = 0; // restore

  // Center proximity bonus
  const mid = (N - 1) / 2;
  const centerDist = Math.abs(r - mid) + Math.abs(c - mid);
  const centerBonus = Math.max(0, N - centerDist) * 5;

  return score + defScore * 0.8 + centerBonus;
}
