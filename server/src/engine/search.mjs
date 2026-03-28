// Alpha-Beta search with iterative deepening for gomoku
// Core search engine inspired by Rapfi architecture

import { evaluate, normalizeScore } from './evaluate.mjs';
import { generateCandidates, mergePolicyCandidates } from './candidates.mjs';
import { orderMoves, KillerTable, HistoryTable } from './move_ordering.mjs';
import { TranspositionTable, TT_EXACT, TT_LOWER, TT_UPPER } from './transposition.mjs';
import { findWinningMoves } from './patterns.mjs';
import { SEARCH_CFG, CANDIDATE_CFG } from './config.mjs';

const INF = 9_999_999;
const WIN_SCORE = 1_000_000;

/**
 * Iterative deepening alpha-beta search.
 *
 * @param {GomokuBoard} board
 * @param {number} player - current player (+1 or -1)
 * @param {object} opts
 * @param {number} [opts.maxDepth] - maximum search depth
 * @param {number} [opts.timeLimitMs] - time budget in ms
 * @param {Float32Array} [opts.nnPolicy] - NN policy for move ordering
 * @param {TranspositionTable} [opts.transTable]
 * @returns {{ move: number, score: number, depth: number, nodesSearched: number, pv: number[] }}
 */
export function iterativeDeepeningSearch(board, player, opts = {}) {
  const maxDepth = opts.maxDepth ?? SEARCH_CFG.maxDepth;
  const timeLimitMs = opts.timeLimitMs ?? SEARCH_CFG.timeLimitMs;
  const tt = opts.transTable ?? new TranspositionTable(SEARCH_CFG.ttSize);
  const nnPolicy = opts.nnPolicy ?? null;

  const killers = new KillerTable(maxDepth + 2);
  const history = new HistoryTable(board.size);

  const startTime = Date.now();
  let nodesSearched = 0;
  let bestMove = -1;
  let bestScore = 0;
  let completedDepth = 0;
  let aborted = false;

  // Check for immediate wins FIRST (before candidate generation)
  const winMoves = findWinningMoves(board, player);
  if (winMoves.length > 0) {
    return { move: winMoves[0], score: WIN_SCORE, normalizedScore: 1.0, depth: 1, nodesSearched: 1, timeMs: 0, pv: [winMoves[0]] };
  }

  // Check if we must block
  const oppWinMoves = findWinningMoves(board, -player);
  if (oppWinMoves.length === 1) {
    return { move: oppWinMoves[0], score: -WIN_SCORE + 100, normalizedScore: -0.9, depth: 1, nodesSearched: 1, timeMs: 0, pv: [oppWinMoves[0]] };
  }

  // Generate root candidates
  let rootCandidates = generateCandidates(board, { radius: CANDIDATE_CFG.radius });
  if (nnPolicy) {
    rootCandidates = mergePolicyCandidates(rootCandidates, nnPolicy, board, CANDIDATE_CFG.nnTopK);
  }

  // Ensure blocking moves are in candidates
  if (oppWinMoves.length > 1) {
    const candSet = new Set(rootCandidates);
    for (const m of oppWinMoves) {
      if (!candSet.has(m)) rootCandidates.push(m);
    }
  }

  if (rootCandidates.length === 0) {
    return { move: -1, score: 0, normalizedScore: 0, depth: 0, nodesSearched: 0, timeMs: 0, pv: [] };
  }

  if (rootCandidates.length === 1) {
    return { move: rootCandidates[0], score: 0, normalizedScore: 0, depth: 1, nodesSearched: 1, timeMs: 0, pv: [rootCandidates[0]] };
  }

  /** Alpha-beta with negamax convention */
  function alphaBeta(depth, alpha, beta, currentPlayer, ply) {
    nodesSearched++;

    // Time check every 4096 nodes
    if ((nodesSearched & 4095) === 0) {
      if (Date.now() - startTime > timeLimitMs) {
        aborted = true;
        return 0;
      }
    }

    // Terminal check
    const winner = board.winner();
    if (winner !== null) {
      if (winner === 0) return 0; // draw
      return winner === currentPlayer ? WIN_SCORE - ply : -(WIN_SCORE - ply);
    }

    // Transposition table probe
    const hashKey = board.hashKey();
    const ttResult = tt.tryUseEntry(hashKey, depth, alpha, beta);
    if (ttResult && ttResult.usable) {
      return ttResult.score;
    }

    // Depth 0: static evaluation
    if (depth <= 0) {
      return evaluate(board, currentPlayer);
    }

    // Generate and order moves
    const candidates = generateCandidates(board, {
      radius: CANDIDATE_CFG.radius,
      maxCandidates: CANDIDATE_CFG.maxCandidates,
    });

    if (candidates.length === 0) {
      return 0; // no moves — draw
    }

    const ttBestMove = tt.getBestMove(hashKey);
    const orderedMoves = orderMoves(board, currentPlayer, candidates, {
      ttBestMove,
      killerMoves: killers.get(ply),
      nnPolicy: ply === 0 ? nnPolicy : null, // NN policy only at root
      historyTable: history.getTable(currentPlayer),
    });

    let bestMoveLocal = orderedMoves[0];
    let bestScoreLocal = -INF;
    let moveIndex = 0;

    for (const move of orderedMoves) {
      if (aborted) return 0;

      board.makeMove(move, currentPlayer);

      let score;

      if (moveIndex === 0) {
        // Full window search for first (best) move
        score = -alphaBeta(depth - 1, -beta, -alpha, -currentPlayer, ply + 1);
      } else {
        // Late Move Reductions
        let reduction = 0;
        if (moveIndex >= SEARCH_CFG.lmrThreshold && depth >= SEARCH_CFG.lmrDepthMin) {
          reduction = 1;
          if (moveIndex >= 8) reduction = 2;
        }

        // Null window search with possible reduction
        score = -alphaBeta(depth - 1 - reduction, -alpha - 1, -alpha, -currentPlayer, ply + 1);

        // Re-search if failed high
        if (score > alpha && (reduction > 0 || score < beta)) {
          score = -alphaBeta(depth - 1, -beta, -alpha, -currentPlayer, ply + 1);
        }
      }

      board.undoMove();

      if (aborted) return 0;

      if (score > bestScoreLocal) {
        bestScoreLocal = score;
        bestMoveLocal = move;
      }

      if (score > alpha) {
        alpha = score;
      }

      if (alpha >= beta) {
        // Beta cutoff
        killers.store(ply, move);
        history.record(currentPlayer, move, depth);
        break;
      }

      moveIndex++;
    }

    // Store in transposition table
    let flag;
    if (bestScoreLocal <= alpha - 1) flag = TT_UPPER; // Was alpha not improved? Use original alpha check
    else if (bestScoreLocal >= beta) flag = TT_LOWER;
    else flag = TT_EXACT;

    // Fix: use the original alpha value for flag determination
    // Actually the standard approach is simpler:
    if (bestScoreLocal >= beta) flag = TT_LOWER;
    else if (moveIndex === 0 || bestScoreLocal <= alpha) flag = TT_UPPER;
    else flag = TT_EXACT;

    tt.store(hashKey, bestScoreLocal, bestMoveLocal, depth, flag);

    return bestScoreLocal;
  }

  // Iterative deepening loop
  for (let depth = 1; depth <= maxDepth; depth++) {
    aborted = false;

    // Aspiration window
    let alpha, beta;
    if (depth > 2 && Math.abs(bestScore) < WIN_SCORE - 100) {
      alpha = bestScore - SEARCH_CFG.aspirationWindow;
      beta = bestScore + SEARCH_CFG.aspirationWindow;
    } else {
      alpha = -INF;
      beta = INF;
    }

    // Root search — manually iterate moves to track best move
    const rootHashKey = board.hashKey();
    const ttBestMove = tt.getBestMove(rootHashKey);
    const orderedMoves = orderMoves(board, player, rootCandidates, {
      ttBestMove: bestMove >= 0 ? bestMove : ttBestMove,
      killerMoves: killers.get(0),
      nnPolicy,
      historyTable: history.getTable(player),
    });

    let rootBestMove = orderedMoves[0];
    let rootBestScore = -INF;
    let rootAlpha = alpha;

    for (let i = 0; i < orderedMoves.length; i++) {
      const move = orderedMoves[i];
      board.makeMove(move, player);

      let score;
      if (i === 0) {
        score = -alphaBeta(depth - 1, -beta, -rootAlpha, -player, 1);
      } else {
        // Null window
        score = -alphaBeta(depth - 1, -rootAlpha - 1, -rootAlpha, -player, 1);
        if (score > rootAlpha && score < beta) {
          score = -alphaBeta(depth - 1, -beta, -rootAlpha, -player, 1);
        }
      }

      board.undoMove();

      if (aborted) break;

      if (score > rootBestScore) {
        rootBestScore = score;
        rootBestMove = move;
      }
      if (score > rootAlpha) rootAlpha = score;
      if (rootAlpha >= beta) break;
    }

    if (aborted && depth > 1) {
      // Time ran out — use result from previous complete iteration
      break;
    }

    // Aspiration fail — re-search with full window
    if (!aborted && (rootBestScore <= alpha || rootBestScore >= beta)) {
      aborted = false;
      rootBestScore = -INF;
      rootAlpha = -INF;

      for (let i = 0; i < orderedMoves.length; i++) {
        const move = orderedMoves[i];
        board.makeMove(move, player);
        const score = -alphaBeta(depth - 1, -INF, -rootAlpha, -player, 1);
        board.undoMove();

        if (aborted) break;
        if (score > rootBestScore) {
          rootBestScore = score;
          rootBestMove = move;
        }
        if (score > rootAlpha) rootAlpha = score;
      }

      if (aborted && depth > 1) break;
    }

    if (!aborted) {
      bestMove = rootBestMove;
      bestScore = rootBestScore;
      completedDepth = depth;
    }

    // Early termination on guaranteed win/loss
    if (Math.abs(bestScore) >= WIN_SCORE - 100) break;

    // Time check before next iteration
    const elapsed = Date.now() - startTime;
    if (elapsed > timeLimitMs * 0.6) break; // likely won't finish next depth

    // Age history between iterations
    history.age();
  }

  return {
    move: bestMove,
    score: bestScore,
    normalizedScore: normalizeScore(bestScore),
    depth: completedDepth,
    nodesSearched,
    timeMs: Date.now() - startTime,
    ttStats: tt.stats(),
  };
}
