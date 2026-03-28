// Move ordering for alpha-beta search
// Good move ordering is critical for pruning efficiency

import { quickMoveEval } from './evaluate.mjs';
import { findWinningMoves } from './patterns.mjs';

/**
 * Order candidate moves for alpha-beta search.
 * Priority: TT best move > winning moves > killer moves > NN policy > pattern eval > proximity
 *
 * @param {GomokuBoard} board
 * @param {number} player
 * @param {number[]} candidates - candidate move positions
 * @param {object} opts
 * @param {number} [opts.ttBestMove] - transposition table best move
 * @param {number[]} [opts.killerMoves] - killer moves for this depth
 * @param {Float32Array} [opts.nnPolicy] - NN policy distribution
 * @param {Uint32Array} [opts.historyTable] - history heuristic scores
 * @returns {number[]} ordered move positions (best first)
 */
export function orderMoves(board, player, candidates, opts = {}) {
  const {
    ttBestMove = -1,
    killerMoves = [],
    nnPolicy = null,
    historyTable = null,
  } = opts;

  const scored = new Array(candidates.length);

  for (let i = 0; i < candidates.length; i++) {
    const move = candidates[i];
    let score = 0;

    // 1. TT best move — highest priority
    if (move === ttBestMove) {
      score += 10_000_000;
    }

    // 2. Killer moves
    if (killerMoves[0] === move) score += 500_000;
    else if (killerMoves[1] === move) score += 400_000;

    // 3. NN policy score
    if (nnPolicy) {
      score += Math.floor((nnPolicy[move] || 0) * 200_000);
    }

    // 4. Quick pattern evaluation (attack + defense)
    score += quickMoveEval(board, move, player);

    // 5. History heuristic
    if (historyTable) {
      score += Math.min(100_000, historyTable[move] || 0);
    }

    // 6. Proximity to last move (minor)
    if (board.lastMove >= 0) {
      const N = board.N;
      const lr = (board.lastMove / N) | 0, lc = board.lastMove % N;
      const mr = (move / N) | 0, mc = move % N;
      const dist = Math.max(Math.abs(mr - lr), Math.abs(mc - lc));
      if (dist <= 1) score += 3000;
      else if (dist <= 2) score += 1500;
    }

    scored[i] = { move, score };
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.map(s => s.move);
}

/**
 * Lightweight ordering for quiescence or shallow search.
 * Only considers winning moves and basic pattern scores.
 */
export function orderMovesLight(board, player, candidates) {
  // First: check for immediate winning moves (always play them first)
  const winMoves = findWinningMoves(board, player);
  if (winMoves.length > 0) {
    // Put winning moves first, then the rest
    const winSet = new Set(winMoves);
    const rest = candidates.filter(m => !winSet.has(m));
    return [...winMoves, ...rest];
  }

  // Check for blocking moves
  const blockMoves = findWinningMoves(board, -player);
  if (blockMoves.length > 0) {
    const blockSet = new Set(blockMoves);
    const rest = candidates.filter(m => !blockSet.has(m));
    return [...blockMoves, ...rest];
  }

  return candidates; // no ordering for shallow search
}

/**
 * Killer move table — stores moves that caused beta cutoffs at each depth.
 * Two slots per depth (standard technique).
 */
export class KillerTable {
  constructor(maxDepth = 32) {
    this.table = new Array(maxDepth);
    for (let i = 0; i < maxDepth; i++) {
      this.table[i] = [-1, -1];
    }
  }

  /** Record a killer move at the given depth */
  store(depth, move) {
    if (depth >= this.table.length) return;
    if (this.table[depth][0] === move) return;
    this.table[depth][1] = this.table[depth][0];
    this.table[depth][0] = move;
  }

  /** Get killer moves for a depth */
  get(depth) {
    if (depth >= this.table.length) return [-1, -1];
    return this.table[depth];
  }

  clear() {
    for (let i = 0; i < this.table.length; i++) {
      this.table[i] = [-1, -1];
    }
  }
}

/**
 * History heuristic table — records how often a move caused a cutoff.
 * Indexed by [playerIndex][position].
 */
export class HistoryTable {
  constructor(boardSize = 256) {
    this.scores = [
      new Uint32Array(boardSize),
      new Uint32Array(boardSize),
    ];
  }

  /** Record a successful cutoff move */
  record(player, move, depth) {
    const idx = player === 1 ? 0 : 1;
    this.scores[idx][move] += depth * depth; // deeper cutoffs are more valuable
  }

  /** Get history score for a move */
  get(player, move) {
    const idx = player === 1 ? 0 : 1;
    return this.scores[idx][move];
  }

  /** Get the full table for a player (for use in orderMoves) */
  getTable(player) {
    return this.scores[player === 1 ? 0 : 1];
  }

  /** Age scores (decay) to prevent stale history from dominating */
  age() {
    for (let p = 0; p < 2; p++) {
      for (let i = 0; i < this.scores[p].length; i++) {
        this.scores[p][i] = (this.scores[p][i] / 2) | 0;
      }
    }
  }

  clear() {
    this.scores[0].fill(0);
    this.scores[1].fill(0);
  }
}
