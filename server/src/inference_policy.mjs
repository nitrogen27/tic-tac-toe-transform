// InferencePolicy — safety layer + optional MCTS search for move selection.
// Flow: immediate win check → immediate block check → MCTS (if enabled) → raw NN argmax.

import { mctsPUCT } from './mcts_puct.mjs';

/**
 * @param {object} opts
 * @param {GameAdapter} opts.adapter — from game_adapter.mjs
 * @param {Function} opts.nnEval — async (board, player) => {policy: Float32Array, value: number}
 * @param {Int8Array} opts.board
 * @param {number} opts.player — +1 or -1
 * @param {boolean} [opts.useMCTS=false]
 * @param {number} [opts.mctsSimulations=100]
 * @param {number} [opts.mctsCpuct=1.5]
 * @param {number} [opts.temperature=0.1]
 * @returns {Promise<{move: number, policy: number[], value: number, source: string}>}
 */
export async function inferencePolicy({
  adapter,
  nnEval,
  board,
  player,
  useMCTS = false,
  mctsSimulations = 100,
  mctsCpuct = 1.5,
  temperature = 0.1,
}) {
  const seqLen = adapter.seqLen;
  const isLargeBoard = adapter.N >= 5;
  const candidateOpts = {
    radius: isLargeBoard ? 2 : 1,
    maxMoves: isLargeBoard ? Math.min(18, seqLen) : seqLen,
  };

  // 1. Safety: immediate win
  const winMove = adapter.findImmediateWin(board, player);
  if (winMove >= 0) {
    const policy = new Float32Array(seqLen);
    policy[winMove] = 1.0;
    return { move: winMove, policy: Array.from(policy), value: 1.0, source: 'safety_win' };
  }

  // 2. Safety: immediate block
  const blockMove = adapter.findImmediateBlock(board, player);
  if (blockMove >= 0) {
    const policy = new Float32Array(seqLen);
    policy[blockMove] = 1.0;
    return { move: blockMove, policy: Array.from(policy), value: 0.0, source: 'safety_block' };
  }

  const { policy, value } = await nnEval(board, player);
  const candidateMoves = adapter.candidateMoves(board, { ...candidateOpts, policyProbs: policy });

  // 3. Safety: reduce multi-threat positions before search.
  const opponentImmediateWins = adapter.collectImmediateWins(board, -player, candidateMoves);
  if (opponentImmediateWins.length > 1) {
    const defenseMove = adapter.findBestDefensiveMove(board, player, candidateMoves, policy);
    if (defenseMove >= 0) {
      const forcedPolicy = new Float32Array(seqLen);
      forcedPolicy[defenseMove] = 1.0;
      return {
        move: defenseMove,
        policy: Array.from(forcedPolicy),
        value,
        source: 'safety_multi_block',
      };
    }
  }

  // 4. Tactical fork/double-threat creation.
  const doubleThreatMove = adapter.findDoubleThreatMove(board, player, candidateMoves);
  if (doubleThreatMove >= 0) {
    const forcedPolicy = new Float32Array(seqLen);
    forcedPolicy[doubleThreatMove] = 1.0;
    return {
      move: doubleThreatMove,
      policy: Array.from(forcedPolicy),
      value,
      source: 'safety_double_threat',
    };
  }

  // 5. MCTS search (if enabled)
  if (useMCTS && mctsSimulations > 0) {
    const mcts = mctsPUCT({
      N: adapter.N,
      nnEval,
      C_puct: mctsCpuct,
      sims: mctsSimulations,
      temperature,
      batchParallel: 32,
      winnerFn: (b) => adapter.winner(b),
      legalMovesFn: (b) => adapter.legalMoves(b),
      candidateMovesFn: (b, p, legal, policyProbs) => {
        const opts = {
          ...candidateOpts,
          maxMoves: isLargeBoard ? Math.min(16, legal.length || seqLen) : legal.length,
          policyProbs,
          includePolicyTopK: isLargeBoard ? 6 : 4,
        };
        return adapter.candidateMoves(b, opts);
      },
    });

    const { pi } = await mcts.run(adapter.cloneBoard(board), player);

    // Pick best move from MCTS policy
    const moves = adapter.legalMoves(board);
    let bestMove = moves[0], bestPi = pi[moves[0]];
    for (const m of moves) {
      if (pi[m] > bestPi) { bestPi = pi[m]; bestMove = m; }
    }

    return { move: bestMove, policy: Array.from(pi), value, source: 'mcts' };
  }

  // 6. Raw NN policy argmax
  const moves = adapter.legalMoves(board);
  if (moves.length === 0) {
    return { move: -1, policy: Array.from(policy), value, source: 'no_moves' };
  }

  let bestMove = moves[0], bestProb = policy[moves[0]];
  for (const m of moves) {
    if (policy[m] > bestProb) { bestProb = policy[m]; bestMove = m; }
  }

  return { move: bestMove, policy: Array.from(policy), value, source: 'nn_raw' };
}
