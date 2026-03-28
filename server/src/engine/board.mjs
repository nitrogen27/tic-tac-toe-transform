// GomokuBoard — efficient board representation with Zobrist hashing for search

// Pre-compute Zobrist table for up to 16x16 = 256 positions, 2 players
// Using pairs of 32-bit values combined as a single number (avoiding BigInt for perf)
const ZOBRIST_TABLE = new Array(256);
const ZOBRIST_SIDE = [0, 0]; // side-to-move hash

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = t + Math.imul(t ^ (t >>> 7), 61 | t) ^ t;
    return ((t ^ (t >>> 14)) >>> 0);
  };
}

// Initialize with deterministic PRNG
const rng = mulberry32(0xDEADBEEF);
for (let i = 0; i < 256; i++) {
  ZOBRIST_TABLE[i] = [
    [rng(), rng()], // player +1 (hi, lo)
    [rng(), rng()], // player -1 (hi, lo)
  ];
}
ZOBRIST_SIDE[0] = rng();
ZOBRIST_SIDE[1] = rng();

function xorHash(h, z) {
  h[0] ^= z[0];
  h[1] ^= z[1];
}

export class GomokuBoard {
  /**
   * @param {number} N - board size (7-16)
   * @param {number} winLen - stones in a row to win (typically 5)
   */
  constructor(N, winLen = 5) {
    this.N = N;
    this.size = N * N;
    this.winLen = winLen;
    this.cells = new Int8Array(this.size);
    this.hashHi = 0;
    this.hashLo = 0;
    this.moveCount = 0;
    this.lastMove = -1;
    this.history = []; // stack: {pos, player, prevHashHi, prevHashLo, prevLastMove}
  }

  /** Create from flat Int8Array (compatibility with game_nxn.mjs) */
  static fromArray(arr, N, winLen = 5) {
    const b = new GomokuBoard(N, winLen);
    let count = 0;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] !== 0) {
        b.cells[i] = arr[i];
        const pIdx = arr[i] === 1 ? 0 : 1;
        xorHash([b.hashHi, b.hashLo], ZOBRIST_TABLE[i][pIdx]);
        // Direct mutation since xorHash creates new array — fix:
        b.hashHi ^= ZOBRIST_TABLE[i][pIdx][0];
        b.hashLo ^= ZOBRIST_TABLE[i][pIdx][1];
        count++;
      }
    }
    // Undo the xorHash calls above (they operated on temp arrays)
    // Actually let me rewrite more clearly:
    b.hashHi = 0;
    b.hashLo = 0;
    b.moveCount = count;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] !== 0) {
        b.cells[i] = arr[i];
        const pIdx = arr[i] === 1 ? 0 : 1;
        b.hashHi ^= ZOBRIST_TABLE[i][pIdx][0];
        b.hashLo ^= ZOBRIST_TABLE[i][pIdx][1];
      }
    }
    // Determine side to move from move count parity
    if (count % 2 === 1) {
      b.hashHi ^= ZOBRIST_SIDE[0];
      b.hashLo ^= ZOBRIST_SIDE[1];
    }
    // Find last move (approximate: rightmost non-zero — not critical)
    for (let i = arr.length - 1; i >= 0; i--) {
      if (arr[i] !== 0) { b.lastMove = i; break; }
    }
    return b;
  }

  /** Make a move (mutating, with undo support) */
  makeMove(pos, player) {
    this.history.push({
      pos,
      player,
      prevHashHi: this.hashHi,
      prevHashLo: this.hashLo,
      prevLastMove: this.lastMove,
    });
    this.cells[pos] = player;
    const pIdx = player === 1 ? 0 : 1;
    this.hashHi ^= ZOBRIST_TABLE[pos][pIdx][0];
    this.hashLo ^= ZOBRIST_TABLE[pos][pIdx][1];
    this.hashHi ^= ZOBRIST_SIDE[0];
    this.hashLo ^= ZOBRIST_SIDE[1];
    this.moveCount++;
    this.lastMove = pos;
  }

  /** Undo the last move */
  undoMove() {
    const entry = this.history.pop();
    this.cells[entry.pos] = 0;
    this.hashHi = entry.prevHashHi;
    this.hashLo = entry.prevHashLo;
    this.lastMove = entry.prevLastMove;
    this.moveCount--;
  }

  /** Compact hash key for transposition table */
  hashKey() {
    // Combine hi and lo into a single number-safe key
    // Use string key for Map (JS numbers lose precision > 2^53)
    return this.hashHi + '|' + this.hashLo;
  }

  /** Clone for independent use */
  clone() {
    const b = new GomokuBoard(this.N, this.winLen);
    b.cells.set(this.cells);
    b.hashHi = this.hashHi;
    b.hashLo = this.hashLo;
    b.moveCount = this.moveCount;
    b.lastMove = this.lastMove;
    // Don't clone history — clone is for read-only branches
    return b;
  }

  /** Get flat Int8Array (compatibility) */
  toArray() {
    return this.cells;
  }

  /** Row/col from linear index */
  row(pos) { return (pos / this.N) | 0; }
  col(pos) { return pos % this.N; }
  idx(r, c) { return r * this.N + c; }
  inBounds(r, c) { return r >= 0 && c >= 0 && r < this.N && c < this.N; }

  /** Check if cell is empty */
  isEmpty(pos) { return this.cells[pos] === 0; }

  /** Get legal moves */
  legalMoves() {
    const moves = [];
    for (let i = 0; i < this.size; i++) {
      if (this.cells[i] === 0) moves.push(i);
    }
    return moves;
  }

  /** Check winner (returns +1, -1, 0 for draw, null for ongoing) */
  winner() {
    const N = this.N;
    const winLen = this.winLen;
    const cells = this.cells;
    const DIRS = [[1, 0], [0, 1], [1, 1], [1, -1]];

    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const who = cells[r * N + c];
        if (!who) continue;
        for (const [dr, dc] of DIRS) {
          let k = 1;
          let rr = r + dr, cc = c + dc;
          while (rr >= 0 && cc >= 0 && rr < N && cc < N && cells[rr * N + cc] === who) {
            k++;
            if (k >= winLen) return who;
            rr += dr;
            cc += dc;
          }
        }
      }
    }
    // Draw check
    for (let i = 0; i < this.size; i++) {
      if (cells[i] === 0) return null;
    }
    return 0;
  }

  /** Quick check if move at pos wins for player (without full board scan) */
  isWinningMove(pos, player) {
    const N = this.N;
    const winLen = this.winLen;
    const cells = this.cells;
    const r = (pos / N) | 0;
    const c = pos % N;
    const DIRS = [[1, 0], [0, 1], [1, 1], [1, -1]];

    for (const [dr, dc] of DIRS) {
      let count = 1;
      // Forward
      let rr = r + dr, cc = c + dc;
      while (rr >= 0 && cc >= 0 && rr < N && cc < N && cells[rr * N + cc] === player) {
        count++;
        rr += dr;
        cc += dc;
      }
      // Backward
      rr = r - dr;
      cc = c - dc;
      while (rr >= 0 && cc >= 0 && rr < N && cc < N && cells[rr * N + cc] === player) {
        count++;
        rr -= dr;
        cc -= dc;
      }
      if (count >= winLen) return true;
    }
    return false;
  }
}
