// D4 dihedral group symmetry for any NxN board
// 8 symmetries: identity, 3 rotations, 2 mirrors, 2 diagonal mirrors

/**
 * Build symmetry maps for an NxN board.
 * Each map[newIdx] = oldIdx (maps destination to source).
 *
 * @param {number} N - board size
 * @returns {Array<{name: string, map: number[]}>}
 */
export function buildSymmetryMaps(N) {
  const last = N - 1;

  function buildMap(fn) {
    const size = N * N;
    const map = new Array(size);
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const [nr, nc] = fn(r, c);
        map[nr * N + nc] = r * N + c;
      }
    }
    return map;
  }

  return [
    { name: 'identity',  map: buildMap((r, c) => [r, c]) },
    { name: 'rot90',     map: buildMap((r, c) => [c, last - r]) },
    { name: 'rot180',    map: buildMap((r, c) => [last - r, last - c]) },
    { name: 'rot270',    map: buildMap((r, c) => [last - c, r]) },
    { name: 'mirrorV',   map: buildMap((r, c) => [r, last - c]) },
    { name: 'mirrorH',   map: buildMap((r, c) => [last - r, c]) },
    { name: 'diagMain',  map: buildMap((r, c) => [c, r]) },
    { name: 'diagAnti',  map: buildMap((r, c) => [last - c, last - r]) },
  ];
}

// Cache symmetry maps per board size
const _cache = new Map();

/**
 * Get cached symmetry maps for board size N.
 */
export function getSymmetryMaps(N) {
  if (!_cache.has(N)) {
    _cache.set(N, buildSymmetryMaps(N));
  }
  return _cache.get(N);
}

/**
 * Transform a board (Int8Array or flat array) by a symmetry map.
 * map[newIdx] = oldIdx
 */
export function transformBoard(board, map) {
  const size = board.length;
  const out = new Int8Array(size);
  for (let i = 0; i < size; i++) {
    out[i] = board[map[i]];
  }
  return out;
}

/**
 * Transform a policy distribution by a symmetry map.
 */
export function transformPolicy(policy, map) {
  const size = policy.length;
  const out = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    out[i] = policy[map[i]];
  }
  return out;
}

/**
 * Inverse-transform a policy distribution (undo a symmetry).
 */
export function inverseTransformPolicy(policy, map) {
  const size = policy.length;
  const out = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    out[map[i]] = policy[i];
  }
  return out;
}

/**
 * Transform 3-plane encoding by a symmetry map.
 * planes = [cell0_p0, cell0_p1, cell0_p2, cell1_p0, ...]
 * Each cell has 3 values (my/opponent/empty).
 */
export function transformPlanes(planes, map, planesPerCell = 3) {
  const numCells = planes.length / planesPerCell;
  const out = new Float32Array(planes.length);
  for (let i = 0; i < numCells; i++) {
    const src = map[i] * planesPerCell;
    const dst = i * planesPerCell;
    for (let p = 0; p < planesPerCell; p++) {
      out[dst + p] = planes[src + p];
    }
  }
  return out;
}

/**
 * Transform a single move position by a symmetry map.
 * Returns the position in the transformed board where `pos` maps to.
 */
export function transformMove(pos, map) {
  // map[newIdx] = oldIdx, so we need inverse: find i where map[i] === pos
  for (let i = 0; i < map.length; i++) {
    if (map[i] === pos) return i;
  }
  return pos; // shouldn't happen
}

/**
 * Inverse-transform a move position.
 */
export function inverseTransformMove(pos, map) {
  return map[pos];
}
