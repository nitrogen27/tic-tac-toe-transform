// Unified GameAdapter — factory returning a common game interface for any variant.
import {
  emptyBoard, cloneBoard, legalMoves, encodePlanes, maskFromBoard,
  getWinnerWithLen, findImmediateWinWithLen, findImmediateBlockWithLen,
  candidateMovesWithLen, collectImmediateWinsWithLen, countImmediateWinsWithLen,
  findDoubleThreatMoveWithLen, findBestDefensiveMoveWithLen, centerIndex,
} from './game_nxn.mjs';

/**
 * @param {object} opts
 * @param {string} opts.variant — 'ttt3', 'ttt5', or 'gomokuN' (e.g. 'gomoku9', 'gomoku15')
 * @param {number} [opts.winLen] — override win length (default: 3 for ttt3, 4 for ttt5, 5 for gomoku)
 * @param {number} [opts.boardSize] — explicit board size (used for gomoku if not parsed from variant)
 * @returns {GameAdapter}
 */
export function createGameAdapter({ variant = 'ttt3', winLen, boardSize } = {}) {
  let N, effectiveWinLen;

  if (variant.startsWith('gomoku')) {
    N = boardSize || parseInt(variant.replace('gomoku', ''), 10) || 15;
    effectiveWinLen = winLen ?? 5;
  } else if (variant === 'ttt5') {
    N = 5;
    effectiveWinLen = winLen ?? 4;
  } else {
    N = 3;
    effectiveWinLen = winLen ?? 3;
  }

  const seqLen = N * N;

  return {
    variant,
    N,
    seqLen,
    winLen: effectiveWinLen,

    emptyBoard() {
      return emptyBoard(N);
    },

    centerMove() {
      return centerIndex(N);
    },

    cloneBoard(b) {
      return cloneBoard(b);
    },

    legalMoves(b) {
      return legalMoves(b);
    },

    applyMove(b, mv, player) {
      const c = cloneBoard(b);
      c[mv] = player;
      return c;
    },

    winner(board) {
      return getWinnerWithLen(board, N, effectiveWinLen);
    },

    isTerminal(board) {
      return this.winner(board) !== null;
    },

    encodePlanes(b, player) {
      return encodePlanes(b, player);
    },

    maskLegalMoves(b) {
      return maskFromBoard(b);
    },

    candidateMoves(b, opts = {}) {
      return candidateMovesWithLen(b, N, effectiveWinLen, opts);
    },

    findImmediateWin(b, player) {
      return findImmediateWinWithLen(b, N, effectiveWinLen, player);
    },

    findImmediateBlock(b, player) {
      return findImmediateBlockWithLen(b, N, effectiveWinLen, player);
    },

    collectImmediateWins(b, player, moves) {
      return collectImmediateWinsWithLen(b, N, effectiveWinLen, player, moves);
    },

    countImmediateWins(b, player, moves) {
      return countImmediateWinsWithLen(b, N, effectiveWinLen, player, moves);
    },

    findDoubleThreatMove(b, player, moves) {
      return findDoubleThreatMoveWithLen(b, N, effectiveWinLen, player, moves);
    },

    findBestDefensiveMove(b, player, moves, policyProbs = null) {
      return findBestDefensiveMoveWithLen(b, N, effectiveWinLen, player, moves, policyProbs);
    },
  };
}
