#pragma once

#include "gomoku/board.hpp"
#include "gomoku/config.hpp"
#include "gomoku/patterns.hpp"

namespace gomoku {

// Static evaluation function for a board position.
// Returns a raw score from `player`'s perspective.
// Positive = favourable for player, negative = unfavourable.
int evaluate(const GomokuBoard& board, Cell player, const EngineConfig& cfg);

// Convenience overload: evaluates from the side-to-move's perspective.
// Returns double for backward compatibility with the engine interface.
double static_evaluate(const GomokuBoard& board, const EngineConfig& cfg);

// Quick move evaluation for move ordering.
// Scores a move without deep search — evaluates the tactical impact of
// placing a stone at `pos` for `player`, plus defensive value.
// Board should NOT have the stone placed; function temporarily places/removes it.
int quick_move_eval(GomokuBoard& board, int pos, Cell player,
                    const EngineConfig& cfg);

// Normalize a raw evaluation score to [-1, +1] range using tanh mapping.
// Used for value output to match neural-network value format.
double normalize_score(int raw_score);

// Compute combination bonus for a set of pattern counts.
int compute_combo_bonuses(const PatternCounts& counts, const ComboBonuses& combo);

} // namespace gomoku
