// Симметрии доски 3x3.
// map[newIdx] = oldIdx, то есть элемент в новой позиции берется из oldIdx.

export const SYMMETRY_MAPS = [
  { name: 'identity', map: [0, 1, 2, 3, 4, 5, 6, 7, 8] },
  { name: 'rot90', map: [6, 3, 0, 7, 4, 1, 8, 5, 2] },
  { name: 'rot180', map: [8, 7, 6, 5, 4, 3, 2, 1, 0] },
  { name: 'rot270', map: [2, 5, 8, 1, 4, 7, 0, 3, 6] },
  { name: 'mirrorV', map: [2, 1, 0, 5, 4, 3, 8, 7, 6] },
  { name: 'mirrorH', map: [6, 7, 8, 3, 4, 5, 0, 1, 2] },
  { name: 'diagMain', map: [0, 3, 6, 1, 4, 7, 2, 5, 8] },
  { name: 'diagAnti', map: [8, 5, 2, 7, 4, 1, 6, 3, 0] },
];

export function transformBoard(board, map) {
  const out = new Int8Array(9);
  for (let i = 0; i < 9; i++) {
    out[i] = board[map[i]];
  }
  return out;
}

export function transformPolicy(policy, map) {
  const out = new Float32Array(9);
  for (let i = 0; i < 9; i++) {
    out[i] = policy[map[i]];
  }
  return out;
}

export function inverseTransformPolicy(policy, map) {
  const out = new Float32Array(9);
  for (let i = 0; i < 9; i++) {
    out[map[i]] = policy[i];
  }
  return out;
}

export function transformPlanes(planes, map) {
  const out = new Float32Array(27);
  for (let i = 0; i < 9; i++) {
    const src = map[i] * 3;
    const dst = i * 3;
    out[dst + 0] = planes[src + 0];
    out[dst + 1] = planes[src + 1];
    out[dst + 2] = planes[src + 2];
  }
  return out;
}
