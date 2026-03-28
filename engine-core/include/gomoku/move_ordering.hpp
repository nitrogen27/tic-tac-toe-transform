#pragma once

#include "gomoku/board.hpp"
#include "gomoku/types.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace gomoku {

// -------------------------------------------------------------------------
// KillerTable — stores 2 moves per ply that caused beta cutoffs
// -------------------------------------------------------------------------
class KillerTable {
public:
    explicit KillerTable(int max_depth = 32);

    void store(int ply, int move);
    std::array<int, 2> get(int ply) const;
    void clear();

private:
    std::vector<std::array<int, 2>> table_;
};

// -------------------------------------------------------------------------
// HistoryTable — per-player cutoff tracking indexed by board position
// -------------------------------------------------------------------------
class HistoryTable {
public:
    explicit HistoryTable(int board_size = MAX_CELLS);

    // Record a successful cutoff move (bonus = depth * depth)
    void record(int8_t player, int move, int depth);

    // Get history score for a move
    uint32_t get_score(int8_t player, int move) const;

    // Get the raw score array for a player (for use in order_moves)
    const uint32_t* get_table(int8_t player) const;

    // Age scores (halve all entries) to prevent stale history
    void age();

    void clear();

private:
    int size_;
    std::vector<uint32_t> scores_[2];  // index 0 = BLACK(+1), index 1 = WHITE(-1)

    int player_index(int8_t player) const { return player == BLACK ? 0 : 1; }
};

// -------------------------------------------------------------------------
// Move ordering functions
// -------------------------------------------------------------------------

struct MoveOrderingOpts {
    int              tt_best_move = -1;
    std::array<int,2> killer_moves = {-1, -1};
    const float*     nn_policy    = nullptr;  // may be null
    const uint32_t*  history_table = nullptr; // may be null
};

// Full move ordering for alpha-beta search
// Scoring: TT best +10M, killers +500K/400K, NN policy 0-200K,
//          quick eval, history 0-100K, proximity to last move
std::vector<int> order_moves(const GomokuBoard& board, int8_t player,
                             const std::vector<int>& candidates,
                             const MoveOrderingOpts& opts);

// Lightweight ordering: winning moves first, then blocking moves, then rest
std::vector<int> order_moves_light(GomokuBoard& board, int8_t player,
                                   const std::vector<int>& candidates);

} // namespace gomoku
