// Threat Space Search: VCF (Victory by Continuous Fours)
// Finds forced winning sequences through continuous four-threats
// The defender has exactly one response to each four (must block)

import { findWinningMoves } from './patterns.mjs';
import { THREAT_CFG } from './config.mjs';

/**
 * VCF Search — find a forced win by playing only four-threats.
 *
 * A "four" is a move that creates an immediate win on the NEXT move.
 * The opponent MUST respond by blocking (exactly one blocking move per four).
 *
 * The search alternates:
 *   1. Attacker plays a four-creating move
 *   2. Defender is forced to block
 *   3. Repeat until attacker achieves FIVE or depth limit
 *
 * @param {GomokuBoard} board
 * @param {number} attacker - +1 or -1
 * @param {number} [maxDepth] - max depth in half-moves (attack + defense pairs)
 * @returns {{ move: number, sequence: number[] } | null} winning first move or null
 */
export function vcfSearch(board, attacker, maxDepth) {
  maxDepth = maxDepth ?? THREAT_CFG.vcfMaxDepth;
  let nodesSearched = 0;
  const maxNodes = THREAT_CFG.vcfMaxNodes;

  /**
   * Recursive VCF search.
   * @param {number} depth - remaining depth
   * @param {boolean} isAttacker - true = attacker's turn
   * @param {number[]} sequence - move sequence so far
   * @returns {number[]|null} winning sequence or null
   */
  function vcfRecurse(depth, isAttacker, sequence) {
    if (nodesSearched++ > maxNodes) return null;
    if (depth <= 0) return null;

    const cells = board.cells;

    if (isAttacker) {
      // Find moves that create an immediate winning threat (a "four")
      // A four-creating move is one where, after placement, there exists
      // at least one cell where placing another stone wins
      const fourMoves = findFourCreatingMovesOptimized(board, attacker);

      for (const attackMove of fourMoves) {
        board.makeMove(attackMove, attacker);

        // Check if this move itself is a win (five in a row)
        if (board.isWinningMove(attackMove, attacker)) {
          board.undoMove();
          return [...sequence, attackMove];
        }

        // Find the threat(s) this creates
        const threats = findWinningMoves(board, attacker);

        if (threats.length >= 2) {
          // Double threat = immediate win (opponent can block only one)
          board.undoMove();
          return [...sequence, attackMove];
        }

        if (threats.length === 1) {
          // Single threat — opponent must block at threats[0]
          const blockMove = threats[0];

          board.makeMove(blockMove, -attacker);

          // After block, check if game continues
          const winner = board.winner();
          if (winner === null) {
            // Continue: attacker plays next four
            const result = vcfRecurse(depth - 2, true, [...sequence, attackMove, blockMove]);
            if (result) {
              board.undoMove(); // undo block
              board.undoMove(); // undo attack
              return result;
            }
          }

          board.undoMove(); // undo block
        }

        board.undoMove(); // undo attack
      }

      return null;
    }

    return null; // Should not reach here
  }

  // Quick check: if attacker already has a winning move, return it
  const immediateWins = findWinningMoves(board, attacker);
  if (immediateWins.length >= 1) {
    return { move: immediateWins[0], sequence: [immediateWins[0]] };
  }

  const result = vcfRecurse(maxDepth, true, []);
  if (result && result.length > 0) {
    return { move: result[0], sequence: result };
  }

  return null;
}

/**
 * Optimized four-creating move finder.
 * A four-creating move is one where, after placing a stone:
 * - The move itself forms part of a line of (winLen-1) with at least one open end, OR
 * - The move creates a new winning threat on the next move
 *
 * We use a fast line-scan approach instead of brute-force.
 */
function findFourCreatingMovesOptimized(board, player) {
  const N = board.N;
  const winLen = board.winLen;
  const cells = board.cells;
  const moves = [];
  const seen = new Uint8Array(board.size);

  const DIRS = [[0, 1], [1, 0], [1, 1], [1, -1]];

  for (let pos = 0; pos < board.size; pos++) {
    if (cells[pos] !== 0 || seen[pos]) continue;

    // Temporarily place
    cells[pos] = player;

    // Check each direction: does this create a line of winLen-1 with open end?
    let createsFour = false;
    const r = (pos / N) | 0;
    const c = pos % N;

    for (const [dr, dc] of DIRS) {
      let fwd = 0, bwd = 0;
      let rr, cc;

      // Forward
      rr = r + dr; cc = c + dc;
      while (rr >= 0 && cc >= 0 && rr < N && cc < N && cells[rr * N + cc] === player) {
        fwd++; rr += dr; cc += dc;
      }
      const fwdEmpty = rr >= 0 && cc >= 0 && rr < N && cc < N && cells[rr * N + cc] === 0;

      // Backward
      rr = r - dr; cc = c - dc;
      while (rr >= 0 && cc >= 0 && rr < N && cc < N && cells[rr * N + cc] === player) {
        bwd++; rr -= dr; cc -= dc;
      }
      const bwdEmpty = rr >= 0 && cc >= 0 && rr < N && cc < N && cells[rr * N + cc] === 0;

      const len = 1 + fwd + bwd;

      // Creates five directly
      if (len >= winLen) {
        createsFour = true;
        break;
      }

      // Creates a four (winLen-1 in a row with at least one open end)
      if (len === winLen - 1 && (fwdEmpty || bwdEmpty)) {
        createsFour = true;
        break;
      }

      // Check for gapped fours: e.g., XX_XX where filling pos creates X_XXXX or XXXX_X
      // Check windows of size winLen that include pos
      for (let offset = 0; offset < winLen; offset++) {
        const startR = r - offset * dr;
        const startC = c - offset * dc;
        let pCount = 0, eCount = 0, valid = true;

        for (let k = 0; k < winLen; k++) {
          const kr = startR + k * dr;
          const kc = startC + k * dc;
          if (kr < 0 || kc < 0 || kr >= N || kc >= N) { valid = false; break; }
          const v = cells[kr * N + kc];
          if (v === player) pCount++;
          else if (v === 0) eCount++;
          else { valid = false; break; }
        }

        if (valid && pCount === winLen - 1 && eCount === 1) {
          createsFour = true;
          break;
        }
      }

      if (createsFour) break;
    }

    cells[pos] = 0; // restore

    if (createsFour) {
      moves.push(pos);
      seen[pos] = 1;
    }
  }

  return moves;
}

/**
 * Find the best defensive move against an opponent's VCF.
 *
 * Strategy: try each candidate move for the defender, and check if
 * the attacker's VCF still works. Return the move that breaks the VCF.
 *
 * @param {GomokuBoard} board
 * @param {number} defender - defending player
 * @param {number} attacker - attacking player
 * @param {number[]} candidates - candidate defensive moves
 * @returns {number} best defensive move, or -1 if none found
 */
export function findBestVCFDefense(board, defender, attacker, candidates) {
  // First, try all candidate moves
  for (const defMove of candidates) {
    if (board.cells[defMove] !== 0) continue;

    board.makeMove(defMove, defender);

    // Check if attacker still has VCF with reduced depth
    const attackVcf = vcfSearch(board, attacker, THREAT_CFG.vcfMaxDepth - 2);

    board.undoMove();

    if (!attackVcf) {
      // This move breaks the VCF!
      return defMove;
    }
  }

  return -1; // no defense found
}

/**
 * Quick check: does a player have any immediate winning move?
 */
export function hasImmediateWin(board, player) {
  return findWinningMoves(board, player).length > 0;
}

/**
 * Count how many separate four-threats exist for a player.
 * Useful for detecting double-threat (fork) situations.
 */
export function countFourThreats(board, player) {
  const cells = board.cells;
  let count = 0;

  for (let pos = 0; pos < board.size; pos++) {
    if (cells[pos] !== 0) continue;
    cells[pos] = player;
    if (board.isWinningMove(pos, player)) {
      count++;
    }
    cells[pos] = 0;
  }

  return count;
}
