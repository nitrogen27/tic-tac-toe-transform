// Gomoku Engine V2 Configuration

export const GOMOKU_VARIANTS = {
  gomoku7:  { N: 7,  winLen: 5, seqLen: 49 },
  gomoku9:  { N: 9,  winLen: 5, seqLen: 81 },
  gomoku11: { N: 11, winLen: 5, seqLen: 121 },
  gomoku13: { N: 13, winLen: 5, seqLen: 169 },
  gomoku15: { N: 15, winLen: 5, seqLen: 225 },
  gomoku16: { N: 16, winLen: 5, seqLen: 256 },
};

// Pattern scores — tuned for 5-in-a-row gomoku
export const PATTERN_SCORES = {
  FIVE:        1_000_000,
  OPEN_FOUR:     100_000,
  HALF_FOUR:      10_000,
  OPEN_THREE:      5_000,
  HALF_THREE:      1_000,
  OPEN_TWO:          500,
  HALF_TWO:          100,
  OPEN_ONE:           10,
};

// Combination bonuses (when multiple threats exist simultaneously)
export const COMBO_BONUSES = {
  DOUBLE_HALF_FOUR:       80_000,   // 2x half-four = unstoppable
  OPEN_THREE_HALF_FOUR:   60_000,   // open-three + half-four = unstoppable
  DOUBLE_OPEN_THREE:      40_000,   // 2x open-three = very strong
  TRIPLE_HALF_THREE:      20_000,   // 3x half-three = complex
};

// Alpha-beta search configuration
export const SEARCH_CFG = {
  maxDepth:        12,      // iterative deepening max depth
  timeLimitMs:     3000,    // per-move time budget
  ttSize:          1 << 20, // ~1M transposition table entries
  aspirationWindow: 50,     // aspiration window half-width
  nullMoveReduction: 3,     // null-move pruning depth reduction
  lmrThreshold:    4,       // late move reduction starts after move #4
  lmrDepthMin:     3,       // LMR only at depth >= 3
};

// VCF/VCT threat search configuration
export const THREAT_CFG = {
  vcfMaxDepth: 14,    // max depth for VCF search (plies)
  vctMaxDepth: 8,     // max depth for VCT search (plies)
  vcfMaxNodes: 50000, // node budget for VCF
  vctMaxNodes: 20000, // node budget for VCT
};

// Candidate move generation
export const CANDIDATE_CFG = {
  radius:       2,    // only consider empty cells within radius of stones
  maxCandidates: 30,  // max candidates per position
  nnTopK:       8,    // top-K NN policy moves to always include
};

// Transformer model configuration for gomoku
export const GOMOKU_TRANSFORMER_CFG = {
  dModel:    192,
  numLayers: 6,
  heads:     6,
  dropout:   0.08,
};

// Training configuration — optimized for GPU saturation (RTX 3060)
export const GOMOKU_TRAIN_CFG = {
  batchSize:         256,       // Увеличено с 64 — GPU batch saturation
  epochs:            30,
  lr:                1e-3,      // Увеличено с 5e-4 — linear scaling для больших батчей
  weightValue:       0.5,
  selfPlayGames:     200,
  selfPlaySimulations: 400,     // Увеличено с 150 — глубже MCTS для качества
  replayBufferMax:   20000,     // Увеличено с 10000 — больше данных
  hardBufferMax:     5000,      // Увеличено с 3000
  batchParallel:     32,        // Параллельных MCTS leaf expansions
};
