// Symmetries for 5x5 board (D4 dihedral group: 4 rotations + 4 reflections).
// map[newIdx] = oldIdx — element at new position is taken from oldIdx.
// For 5x5: index = row * 5 + col

function buildMap(fn) {
  const map = new Array(25);
  for (let r = 0; r < 5; r++) {
    for (let c = 0; c < 5; c++) {
      const [nr, nc] = fn(r, c);
      map[nr * 5 + nc] = r * 5 + c;
    }
  }
  return map;
}

export const SYMMETRY_MAPS_5 = [
  { name: 'identity',  map: buildMap((r, c) => [r, c]) },
  { name: 'rot90',     map: buildMap((r, c) => [c, 4 - r]) },
  { name: 'rot180',    map: buildMap((r, c) => [4 - r, 4 - c]) },
  { name: 'rot270',    map: buildMap((r, c) => [4 - c, r]) },
  { name: 'mirrorV',   map: buildMap((r, c) => [r, 4 - c]) },
  { name: 'mirrorH',   map: buildMap((r, c) => [4 - r, c]) },
  { name: 'diagMain',  map: buildMap((r, c) => [c, r]) },
  { name: 'diagAnti',  map: buildMap((r, c) => [4 - c, 4 - r]) },
];

export function transformBoard(board, map) {
  const out = new Int8Array(25);
  for (let i = 0; i < 25; i++) out[i] = board[map[i]];
  return out;
}

export function transformPolicy(policy, map) {
  const out = new Float32Array(25);
  for (let i = 0; i < 25; i++) out[i] = policy[map[i]];
  return out;
}

export function inverseTransformPolicy(policy, map) {
  const out = new Float32Array(25);
  for (let i = 0; i < 25; i++) out[map[i]] = policy[i];
  return out;
}

export function transformPlanes(planes, map) {
  const out = new Float32Array(75);
  for (let i = 0; i < 25; i++) {
    const src = map[i] * 3;
    const dst = i * 3;
    out[dst + 0] = planes[src + 0];
    out[dst + 1] = planes[src + 1];
    out[dst + 2] = planes[src + 2];
  }
  return out;
}
