// Gomoku Engine V2 — Public API Facade
// Combines all engine components into a unified interface

import { GomokuBoard } from './board.mjs';
import { evaluate, normalizeScore } from './evaluate.mjs';
import { iterativeDeepeningSearch } from './search.mjs';
import { vcfSearch, findBestVCFDefense, hasImmediateWin, countFourThreats } from './threats.mjs';
import { findWinningMoves, evaluatePatterns } from './patterns.mjs';
import { generateCandidates, mergePolicyCandidates } from './candidates.mjs';
import { TranspositionTable } from './transposition.mjs';
import { SEARCH_CFG, THREAT_CFG, CANDIDATE_CFG } from './config.mjs';

/**
 * Create a gomoku engine instance.
 *
 * @param {object} opts
 * @param {number} opts.N - board size (7-16)
 * @param {number} [opts.winLen=5] - stones in a row to win
 * @param {object} [opts.searchConfig] - override search configuration
 * @param {object} [opts.threatConfig] - override threat configuration
 * @returns {GomokuEngine}
 */
export function createGomokuEngine(opts = {}) {
  const N = opts.N || 15;
  const winLen = opts.winLen ?? 5;
  const searchCfg = { ...SEARCH_CFG, ...opts.searchConfig };
  const threatCfg = { ...THREAT_CFG, ...opts.threatConfig };
  const candidateCfg = { ...CANDIDATE_CFG, ...opts.candidateConfig };

  const transTable = new TranspositionTable(searchCfg.ttSize);

  // Adjust search params based on board size
  const adjustedCfg = adjustForBoardSize(N, searchCfg);

  return {
    N,
    winLen,

    /**
     * Find the best move for the current position.
     *
     * @param {Int8Array|number[]} boardArray - flat board in {0, +1, -1} format
     * @param {number} player - current player (+1 or -1)
     * @param {object} [moveOpts]
     * @param {Float32Array} [moveOpts.nnPolicy] - NN policy for move ordering
     * @param {number} [moveOpts.timeLimitMs] - override time limit
     * @param {number} [moveOpts.maxDepth] - override max depth
     * @returns {EngineResult}
     */
    bestMove(boardArray, player, moveOpts = {}) {
      const board = GomokuBoard.fromArray(boardArray, N, winLen);
      const nnPolicy = moveOpts.nnPolicy || null;
      const timeLimitMs = moveOpts.timeLimitMs || adjustedCfg.timeLimitMs;
      const maxDepth = moveOpts.maxDepth || adjustedCfg.maxDepth;

      // Layer 1: Immediate win
      const winMoves = findWinningMoves(board, player);
      if (winMoves.length > 0) {
        return makeResult(winMoves[0], 1.0, 'safety_win', 1, 1, buildPolicy(winMoves[0], board.size));
      }

      // Layer 2: Immediate block (opponent has winning move)
      const oppWinMoves = findWinningMoves(board, -player);
      if (oppWinMoves.length === 1) {
        return makeResult(oppWinMoves[0], -0.3, 'safety_block', 1, 1, buildPolicy(oppWinMoves[0], board.size));
      }
      if (oppWinMoves.length >= 2) {
        // Multiple threats — find the best defense
        const candidates = generateCandidates(board, { radius: candidateCfg.radius });
        const bestDef = findBestDefenseMove(board, player, oppWinMoves, candidates);
        return makeResult(bestDef, -0.8, 'safety_multi_block', 1, 1, buildPolicy(bestDef, board.size));
      }

      // Layer 3: VCF search (forced win through continuous fours)
      const vcfResult = vcfSearch(board, player, threatCfg.vcfMaxDepth);
      if (vcfResult) {
        return makeResult(vcfResult.move, 1.0, 'vcf_win', vcfResult.sequence.length, 1,
          buildPolicy(vcfResult.move, board.size));
      }

      // Layer 4: Defend opponent's VCF
      const oppVcf = vcfSearch(board, -player, threatCfg.vcfMaxDepth);
      if (oppVcf) {
        const candidates = generateCandidates(board, { radius: candidateCfg.radius });
        const defMove = findBestVCFDefense(board, player, -player, candidates);
        if (defMove >= 0) {
          return makeResult(defMove, -0.5, 'vcf_defense', threatCfg.vcfMaxDepth, candidates.length,
            buildPolicy(defMove, board.size));
        }
        // If no defense found, fall through to alpha-beta (try anyway)
      }

      // Layer 5: Check for double-threat (fork) creation
      const forkMove = findForkMove(board, player);
      if (forkMove >= 0) {
        return makeResult(forkMove, 0.9, 'fork', 2, 1, buildPolicy(forkMove, board.size));
      }

      // Layer 6: Alpha-beta search
      const searchResult = iterativeDeepeningSearch(board, player, {
        maxDepth,
        timeLimitMs,
        nnPolicy,
        transTable,
      });

      const policy = buildSearchPolicy(searchResult, board.size, nnPolicy);

      return makeResult(
        searchResult.move,
        searchResult.normalizedScore,
        'alpha_beta',
        searchResult.depth,
        searchResult.nodesSearched,
        policy,
        {
          timeMs: searchResult.timeMs,
          ttStats: searchResult.ttStats,
          rawScore: searchResult.score,
        }
      );
    },

    /**
     * Evaluate a position without searching (static evaluation).
     */
    evaluate(boardArray, player) {
      const board = GomokuBoard.fromArray(boardArray, N, winLen);
      return {
        score: evaluate(board, player),
        normalized: normalizeScore(evaluate(board, player)),
        patterns: evaluatePatterns(board, player),
      };
    },

    /**
     * Get hint/suggestion moves with scores.
     */
    getHints(boardArray, player, topK = 5) {
      const board = GomokuBoard.fromArray(boardArray, N, winLen);
      const candidates = generateCandidates(board, {
        radius: candidateCfg.radius,
        maxCandidates: candidateCfg.maxCandidates,
      });

      // Quick evaluate each candidate
      const scored = candidates.map(move => {
        board.makeMove(move, player);
        const score = evaluate(board, player);
        board.undoMove();
        return { move, score };
      });

      scored.sort((a, b) => b.score - a.score);
      return scored.slice(0, topK);
    },

    /** Check for VCF for a player */
    findVCF(boardArray, player) {
      const board = GomokuBoard.fromArray(boardArray, N, winLen);
      return vcfSearch(board, player, threatCfg.vcfMaxDepth);
    },

    /** Clear transposition table */
    clearCache() {
      transTable.clear();
    },

    /** Get engine stats */
    stats() {
      return {
        ttStats: transTable.stats(),
        boardSize: N,
        winLen,
        searchConfig: adjustedCfg,
      };
    },
  };
}

/**
 * Adjust search parameters based on board size.
 * Larger boards need more time but less depth (wider search).
 */
function adjustForBoardSize(N, baseCfg) {
  const cfg = { ...baseCfg };

  if (N <= 9) {
    cfg.maxDepth = Math.min(cfg.maxDepth, 14);
    cfg.timeLimitMs = Math.min(cfg.timeLimitMs, 2000);
  } else if (N <= 11) {
    cfg.maxDepth = Math.min(cfg.maxDepth, 12);
    cfg.timeLimitMs = Math.max(cfg.timeLimitMs, 3000);
  } else if (N <= 13) {
    cfg.maxDepth = Math.min(cfg.maxDepth, 10);
    cfg.timeLimitMs = Math.max(cfg.timeLimitMs, 4000);
  } else {
    // 15x15, 16x16
    cfg.maxDepth = Math.min(cfg.maxDepth, 8);
    cfg.timeLimitMs = Math.max(cfg.timeLimitMs, 5000);
  }

  return cfg;
}

/**
 * Find a fork move (creates 2+ simultaneous four-threats).
 */
function findForkMove(board, player) {
  const candidates = generateCandidates(board, { radius: 2, maxCandidates: 30 });
  const cells = board.cells;

  for (const move of candidates) {
    cells[move] = player;

    // Count winning threats after this move
    const threats = countFourThreats(board, player);

    cells[move] = 0;

    if (threats >= 2) return move;
  }

  return -1;
}

/**
 * Find best defense when opponent has multiple winning threats.
 */
function findBestDefenseMove(board, player, oppWinMoves, candidates) {
  const cells = board.cells;
  let bestMove = oppWinMoves[0]; // default: block first threat
  let bestScore = -Infinity;

  for (const move of candidates) {
    if (cells[move] !== 0) continue;

    cells[move] = player;

    // Count remaining opponent threats
    let remainingThreats = 0;
    for (const wm of oppWinMoves) {
      if (cells[wm] !== 0) continue; // blocked by our move
      cells[wm] = -player;
      if (board.isWinningMove(wm, -player)) {
        remainingThreats++;
      }
      cells[wm] = 0;
    }

    // Also count own threats (prefer moves that create counter-threats)
    const ownThreats = countFourThreats(board, player);

    const score = -remainingThreats * 10000 + ownThreats * 5000;

    cells[move] = 0;

    if (score > bestScore) {
      bestScore = score;
      bestMove = move;
    }
  }

  return bestMove;
}

/**
 * Build a one-hot-ish policy from a single move.
 */
function buildPolicy(move, boardSize) {
  const policy = new Float32Array(boardSize);
  policy[move] = 1.0;
  return policy;
}

/**
 * Build a policy distribution from search results.
 * Blends search best move with NN policy if available.
 */
function buildSearchPolicy(searchResult, boardSize, nnPolicy) {
  const policy = new Float32Array(boardSize);

  if (nnPolicy) {
    // Blend: 70% search best move + 30% NN policy
    for (let i = 0; i < boardSize; i++) {
      policy[i] = (nnPolicy[i] || 0) * 0.3;
    }
    policy[searchResult.move] += 0.7;
  } else {
    policy[searchResult.move] = 1.0;
  }

  return policy;
}

/**
 * Standard result format.
 */
function makeResult(move, value, source, depth, nodes, policy, extra = {}) {
  return {
    move,
    value,
    source,
    depth,
    nodesSearched: nodes,
    policy,
    ...extra,
  };
}

// Re-export key types for convenience
export { GomokuBoard } from './board.mjs';
export { TranspositionTable } from './transposition.mjs';
export { GOMOKU_VARIANTS, SEARCH_CFG, THREAT_CFG } from './config.mjs';
export { getSymmetryMaps, transformBoard, inverseTransformPolicy, transformPlanes } from './symmetry_nxn.mjs';
