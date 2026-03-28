#pragma once

#include "gomoku/board.hpp"
#include "gomoku/config.hpp"
#include "gomoku/types.hpp"

#include <string>
#include <vector>

namespace gomoku {

// Public facade for the Gomoku engine.
// Coordinates patterns, evaluation, search, and threat analysis.
class GomokuEngine {
public:
    explicit GomokuEngine(EngineConfig cfg = default_config());

    // Find the best move for the given position
    EngineResult best_move(const Position& pos);

    // Evaluate the current position (heuristic score)
    double evaluate(const Position& pos);

    // Get top-N move suggestions with scores
    std::vector<MoveCandidate> get_hints(const Position& pos, int top_n = 5);

    // Search for a Victory by Continuous Four (VCF) sequence
    std::vector<int> find_vcf(const Position& pos);

    // Clear all cached data (TT, killer moves, etc.)
    void clear_cache();

    // Engine statistics as a human-readable string
    std::string stats() const;

private:
    EngineConfig config_;
};

} // namespace gomoku
