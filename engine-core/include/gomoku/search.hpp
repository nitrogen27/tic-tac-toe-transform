#pragma once

#include "gomoku/board.hpp"
#include "gomoku/config.hpp"
#include "gomoku/transposition.hpp"
#include "gomoku/types.hpp"

#include <vector>

namespace gomoku {

// Result of an iterative deepening search
struct SearchResult {
    int    best_move      = -1;
    int    score          = 0;
    double normalized_score = 0.0;
    int    depth          = 0;
    uint64_t nodes_searched = 0;
    double time_ms        = 0.0;
    std::vector<int> pv;        // principal variation
    TTStats tt_stats;
};

// Options controlling the search
struct SearchOptions {
    int    max_depth     = 12;
    int    time_limit_ms = 3000;
    int    aspiration_window = 50;
    int    lmr_threshold = 4;
    int    lmr_depth_min = 3;
    const float* nn_policy = nullptr;  // NN policy for move ordering (may be null)
    int    candidate_radius = 2;
    int    max_candidates   = 30;
    int    nn_top_k         = 8;
};

// Iterative deepening alpha-beta search.
// Port of search.mjs iterativeDeepeningSearch.
//
// Features:
//   - Negamax alpha-beta with principal variation search (PVS)
//   - Late Move Reductions (LMR)
//   - Aspiration windows with fallback to full window
//   - Transposition table probing and storing
//   - Killer move + history heuristic move ordering
//   - Time management (60% heuristic before next iteration)
//   - Early termination on guaranteed win/loss
SearchResult iterative_deepening_search(GomokuBoard& board, int8_t player,
                                        const SearchOptions& opts,
                                        TranspositionTable& tt);

// Convenience: full engine search using EngineConfig
// (wraps iterative_deepening_search with config-derived options)
EngineResult alpha_beta_search(GomokuBoard& board, const EngineConfig& cfg);

} // namespace gomoku
