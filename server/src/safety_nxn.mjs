// Generic N×N safety layer — works with any GameAdapter.
// Priority: immediate win → immediate block → policy argmax among legal moves.

/**
 * @param {GameAdapter} adapter — from game_adapter.mjs
 * @param {Int8Array} board
 * @param {number} player — +1 or -1
 * @param {Float32Array|number[]} policyProbs — probabilities per cell
 * @returns {number} chosen move index
 */
export function safePick(adapter, board, player, policyProbs) {
  // 1. Immediate win
  const winMove = adapter.findImmediateWin(board, player);
  if (winMove >= 0) return winMove;

  // 2. Immediate block
  const blockMove = adapter.findImmediateBlock(board, player);
  if (blockMove >= 0) return blockMove;

  // 3. Policy argmax among legal moves
  const moves = adapter.legalMoves(board);
  if (moves.length === 0) return -1;

  let bestMove = moves[0];
  let bestProb = policyProbs[moves[0]];
  for (const m of moves) {
    if (policyProbs[m] > bestProb) {
      bestProb = policyProbs[m];
      bestMove = m;
    }
  }
  return bestMove;
}
