#pragma once

#include "gomoku/board.hpp"
#include "gomoku/config.hpp"

#include <array>
#include <vector>

namespace gomoku {

// Pattern type enumeration -- matches JS P.* constants
enum Pattern : int {
    PAT_NONE       = 0,
    PAT_OPEN_ONE   = 1,
    PAT_HALF_TWO   = 2,
    PAT_OPEN_TWO   = 3,
    PAT_HALF_THREE = 4,
    PAT_OPEN_THREE = 5,
    PAT_HALF_FOUR  = 6,
    PAT_OPEN_FOUR  = 7,
    PAT_FIVE       = 8,
    PAT_COUNT      = 9,
};

// Counts of each pattern type for one player
struct PatternCounts {
    std::array<int, PAT_COUNT> data{};

    int& five()       { return data[PAT_FIVE]; }
    int& open_four()  { return data[PAT_OPEN_FOUR]; }
    int& half_four()  { return data[PAT_HALF_FOUR]; }
    int& open_three() { return data[PAT_OPEN_THREE]; }
    int& half_three() { return data[PAT_HALF_THREE]; }
    int& open_two()   { return data[PAT_OPEN_TWO]; }
    int& half_two()   { return data[PAT_HALF_TWO]; }
    int& open_one()   { return data[PAT_OPEN_ONE]; }

    int five()       const { return data[PAT_FIVE]; }
    int open_four()  const { return data[PAT_OPEN_FOUR]; }
    int half_four()  const { return data[PAT_HALF_FOUR]; }
    int open_three() const { return data[PAT_OPEN_THREE]; }
    int half_three() const { return data[PAT_HALF_THREE]; }
    int open_two()   const { return data[PAT_OPEN_TWO]; }
    int half_two()   const { return data[PAT_HALF_TWO]; }
    int open_one()   const { return data[PAT_OPEN_ONE]; }

    int& operator[](int idx) { return data[idx]; }
    int  operator[](int idx) const { return data[idx]; }
};

// ---- Full-board pattern scanning ----

// Scan all patterns on the board for `player` across all 4 directions.
// Includes contiguous groups and gapped patterns (X_XXX, XX_XX, XXX_X).
PatternCounts scan_all_patterns(const GomokuBoard& board, int8_t player);

// Convert pattern counts to a numeric score using the given score weights.
int pattern_counts_to_score(const PatternCounts& counts, const PatternScores& scores);

// Compute pattern score including combo bonuses.
int score_patterns(const PatternCounts& counts, const PatternScores& scores,
                   const ComboBonuses& combo);

// ---- Per-move pattern evaluation ----

// Evaluate the tactical impact of placing a stone at `pos` for `player`.
// The board should NOT have the stone placed yet; this function temporarily
// places and removes it.
int evaluate_move_patterns(GomokuBoard& board, int pos, int8_t player,
                           const PatternScores& scores);

// ---- Threat queries ----

// Find all moves that create an immediate win (five-in-a-row) for `player`.
std::vector<int> find_winning_moves(GomokuBoard& board, int8_t player);

// Find all moves that, after placement, create at least one winning threat.
std::vector<int> find_four_creating_moves(GomokuBoard& board, int8_t player);

// Count the number of immediate winning moves for `player`.
int count_threats(GomokuBoard& board, int8_t player);

// Count opponent threats that would exist after `player` places at `pos`.
int count_opponent_threats_after_move(GomokuBoard& board, int pos, int8_t player);

// Evaluate both players' patterns and return scores + counts.
struct PatternEvalResult {
    int my_score;
    int opp_score;
    PatternCounts my_counts;
    PatternCounts opp_counts;
};

PatternEvalResult evaluate_patterns(const GomokuBoard& board, int8_t player,
                                    const PatternScores& scores);

} // namespace gomoku
