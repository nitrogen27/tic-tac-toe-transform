// Candidate move generation for gomoku
// On large boards (15x15, 16x16), only consider moves near existing stones

import { CANDIDATE_CFG } from './config.mjs';

/**
 * Generate candidate moves within a radius of existing stones.
 * Critical optimization for large boards: reduces 200+ legal moves to ~25-30.
 *
 * @param {GomokuBoard} board
 * @param {object} [opts]
 * @param {number} [opts.radius=2] - distance from existing stones
 * @param {number} [opts.maxCandidates=30] - max candidates to return
 * @returns {number[]} array of candidate move positions
 */
export function generateCandidates(board, opts = {}) {
  const radius = opts.radius ?? CANDIDATE_CFG.radius;
  const maxCandidates = opts.maxCandidates ?? CANDIDATE_CFG.maxCandidates;
  const N = board.N;
  const cells = board.cells;

  // Count actual stones on board (handles boards created via fromArray with moveCount=0)
  let stoneCount = board.moveCount;
  if (stoneCount === 0) {
    for (let i = 0; i < board.size; i++) {
      if (cells[i] !== 0) stoneCount++;
    }
  }

  // First move: play center
  if (stoneCount === 0) {
    const mid = ((N / 2) | 0) * N + ((N / 2) | 0);
    return [mid];
  }

  // Second move: play adjacent to center or center if empty
  if (stoneCount === 1) {
    const mid = ((N / 2) | 0) * N + ((N / 2) | 0);
    if (cells[mid] === 0) return [mid];
    // Play adjacent to opponent's first stone
    const candidates = [];
    const lr = board.lastMove;
    const r = (lr / N) | 0, c = lr % N;
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        if (dr === 0 && dc === 0) continue;
        const nr = r + dr, nc = c + dc;
        if (nr >= 0 && nc >= 0 && nr < N && nc < N && cells[nr * N + nc] === 0) {
          candidates.push(nr * N + nc);
        }
      }
    }
    return candidates;
  }

  const marked = new Uint8Array(board.size);

  // Mark empty cells within radius of any stone
  for (let i = 0; i < board.size; i++) {
    if (cells[i] === 0) continue;
    const r = (i / N) | 0;
    const c = i % N;
    for (let dr = -radius; dr <= radius; dr++) {
      for (let dc = -radius; dc <= radius; dc++) {
        const nr = r + dr, nc = c + dc;
        if (nr >= 0 && nc >= 0 && nr < N && nc < N) {
          const idx = nr * N + nc;
          if (cells[idx] === 0) marked[idx] = 1;
        }
      }
    }
  }

  const candidates = [];
  for (let i = 0; i < board.size; i++) {
    if (marked[i]) candidates.push(i);
  }

  // If very few candidates, expand radius
  if (candidates.length < 5 && board.moveCount > 2) {
    const expandRadius = radius + 1;
    for (let i = 0; i < board.size; i++) {
      if (cells[i] === 0) continue;
      const r = (i / N) | 0;
      const c = i % N;
      for (let dr = -expandRadius; dr <= expandRadius; dr++) {
        for (let dc = -expandRadius; dc <= expandRadius; dc++) {
          const nr = r + dr, nc = c + dc;
          if (nr >= 0 && nc >= 0 && nr < N && nc < N) {
            const idx = nr * N + nc;
            if (cells[idx] === 0 && !marked[idx]) {
              marked[idx] = 1;
              candidates.push(idx);
            }
          }
        }
      }
    }
  }

  if (candidates.length <= maxCandidates) return candidates;

  // Too many candidates — no scoring here, let move_ordering handle it
  // Just return all within radius and let the ordering module truncate
  return candidates;
}

/**
 * Merge NN top-K moves into candidate set (ensures they're always considered).
 */
export function mergePolicyCandidates(candidates, nnPolicy, board, topK) {
  if (!nnPolicy || topK <= 0) return candidates;

  const cells = board.cells;
  // Find top-K legal moves from policy
  const legalProbs = [];
  for (let i = 0; i < board.size; i++) {
    if (cells[i] === 0) {
      legalProbs.push({ pos: i, prob: nnPolicy[i] || 0 });
    }
  }
  legalProbs.sort((a, b) => b.prob - a.prob);

  const result = new Set(candidates);
  for (let i = 0; i < Math.min(topK, legalProbs.length); i++) {
    result.add(legalProbs[i].pos);
  }

  return Array.from(result);
}
