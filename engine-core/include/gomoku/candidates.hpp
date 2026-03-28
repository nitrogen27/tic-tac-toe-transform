#pragma once

#include "gomoku/board.hpp"
#include "gomoku/config.hpp"
#include "gomoku/types.hpp"

#include <vector>

namespace gomoku {

// Generate candidate moves within a radius of existing stones.
// Critical optimization for large boards: reduces 200+ legal moves to ~25-30.
//
// Special cases:
//   - First move: center
//   - Second move: adjacent to opponent's stone (or center if empty)
//   - General: all empty cells within `radius` of any stone
//   - Expands radius if too few candidates found
std::vector<int> generate_candidates(const GomokuBoard& board,
                                     int radius = 2,
                                     int max_candidates = 30);

// Merge NN top-K moves into candidate set (ensures they are always considered).
// Returns a new vector with all original candidates + top-K policy moves.
std::vector<int> merge_policy_candidates(const std::vector<int>& candidates,
                                         const float* nn_policy,
                                         const GomokuBoard& board,
                                         int top_k);

} // namespace gomoku
