// Transposition table for alpha-beta search
// Uses string hash keys (from GomokuBoard.hashKey()) in a JS Map

export const TT_EXACT = 0;
export const TT_LOWER = 1; // beta cutoff (score >= beta)
export const TT_UPPER = 2; // all-node (score <= alpha)

/**
 * @typedef {Object} TTEntry
 * @property {number} score
 * @property {number} move - best move found
 * @property {number} depth - search depth
 * @property {number} flag - TT_EXACT, TT_LOWER, or TT_UPPER
 */

export class TranspositionTable {
  /**
   * @param {number} maxEntries - maximum entries before eviction
   */
  constructor(maxEntries = 1 << 20) {
    this.maxEntries = maxEntries;
    this.table = new Map();
    this.hits = 0;
    this.misses = 0;
    this.stores = 0;
  }

  /**
   * Probe the table for a position.
   * @param {string} key - hash key from board.hashKey()
   * @returns {TTEntry|null}
   */
  probe(key) {
    const entry = this.table.get(key);
    if (entry) {
      this.hits++;
      return entry;
    }
    this.misses++;
    return null;
  }

  /**
   * Store a position in the table.
   * Replacement policy: always replace if new depth >= existing depth.
   */
  store(key, score, move, depth, flag) {
    const existing = this.table.get(key);
    if (existing && existing.depth > depth) {
      // Don't replace deeper entries
      return;
    }

    // Eviction if at capacity
    if (!existing && this.table.size >= this.maxEntries) {
      this._evict();
    }

    this.table.set(key, { score, move, depth, flag });
    this.stores++;
  }

  /**
   * Get the best move from TT for a position (for move ordering).
   */
  getBestMove(key) {
    const entry = this.table.get(key);
    return entry ? entry.move : -1;
  }

  /**
   * Try to get an exact score or usable bound from TT.
   * @returns {{ score: number, usable: boolean } | null}
   */
  tryUseEntry(key, depth, alpha, beta) {
    const entry = this.table.get(key);
    if (!entry || entry.depth < depth) return null;

    this.hits++;

    if (entry.flag === TT_EXACT) {
      return { score: entry.score, usable: true };
    }
    if (entry.flag === TT_LOWER && entry.score >= beta) {
      return { score: entry.score, usable: true };
    }
    if (entry.flag === TT_UPPER && entry.score <= alpha) {
      return { score: entry.score, usable: true };
    }

    return null;
  }

  /** Remove ~25% of entries when full */
  _evict() {
    const keys = this.table.keys();
    let count = (this.maxEntries * 0.25) | 0;
    for (const key of keys) {
      if (count-- <= 0) break;
      this.table.delete(key);
    }
  }

  clear() {
    this.table.clear();
    this.hits = 0;
    this.misses = 0;
    this.stores = 0;
  }

  get size() { return this.table.size; }

  stats() {
    const total = this.hits + this.misses;
    return {
      size: this.table.size,
      hits: this.hits,
      misses: this.misses,
      hitRate: total > 0 ? (this.hits / total * 100).toFixed(1) + '%' : 'N/A',
      stores: this.stores,
    };
  }
}
