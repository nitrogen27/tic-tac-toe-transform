#pragma once

#include "gomoku/types.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace gomoku {

// ---------------------------------------------------------------------------
// Undo-history entry — stores everything needed to reverse a make_move call.
// Matches the JS: {pos, player, prevHashHi, prevHashLo, prevLastMove}
// ---------------------------------------------------------------------------
struct UndoEntry {
    int      pos;
    Cell     player;
    uint64_t prev_hash;
    int      prev_last_move;
};

// ---------------------------------------------------------------------------
// GomokuBoard — mutable game state with Zobrist hashing and undo support.
//
// Faithfully ports the JavaScript GomokuBoard class from board.mjs.
// Key design choices carried over from JS:
//   - make_move takes an explicit player argument (not auto-alternating)
//   - winner() performs a FULL board scan (not just last-move check)
//   - is_winning_move() checks a specific cell for both directions
//   - cells are int8_t: BLACK=+1, WHITE=-1, EMPTY=0
// ---------------------------------------------------------------------------
class GomokuBoard {
public:
    GomokuBoard(int board_size = 15, int win_length = 5);

    // --- Move operations ---------------------------------------------------
    // Place a stone for `player` at flat index `pos`.
    // Pushes an UndoEntry onto the history stack.
    void make_move(int pos, Cell player);

    // Undo the most recent move (pops history stack).
    void undo_move();

    // --- Queries -----------------------------------------------------------
    // Full-board scan for a winner.  Returns:
    //   BLACK (+1)  if black has win_length in a row
    //   WHITE (-1)  if white has win_length in a row
    //   0           if the board is full (draw)
    //   2           if the game is still ongoing (sentinel; not a player value)
    static constexpr Cell ONGOING = 2;
    Cell winner() const;

    // Check whether the stone at `pos` forms a winning line for `player`.
    // The cell at `pos` must already contain `player`.
    bool is_winning_move(int pos, Cell player) const;

    // All empty cell indices.
    std::vector<int> legal_moves() const;

    bool is_empty(int pos) const;

    // Zobrist hash of the current position.
    uint64_t hash_key() const;

    // Deep copy (history is NOT cloned, matching JS behaviour).
    GomokuBoard clone() const;

    // Factory: reconstruct a board from a Position struct.
    static GomokuBoard from_position(const Position& pos);

    // --- Accessors ---------------------------------------------------------
    int    board_size()    const { return N_; }
    int    win_len()       const { return win_len_; }
    int    total_cells()   const { return size_; }
    int    move_count()    const { return move_count_; }
    int    last_move()     const { return last_move_; }
    Cell   cell_at(int i)  const { return cells_[i]; }

    const std::array<Cell, MAX_CELLS>& cells() const { return cells_; }
    const std::vector<UndoEntry>& history() const { return history_; }

    // Coordinate helpers (instance versions for convenience)
    int row_of(int pos) const { return pos / N_; }
    int col_of(int pos) const { return pos % N_; }
    int idx_of(int r, int c) const { return r * N_ + c; }
    bool in_bounds(int r, int c) const {
        return r >= 0 && c >= 0 && r < N_ && c < N_;
    }

private:
    int N_;          // board dimension (e.g. 15)
    int size_;       // N_ * N_
    int win_len_;    // stones in a row to win (e.g. 5)

    std::array<Cell, MAX_CELLS> cells_;
    uint64_t hash_      = 0;
    int move_count_     = 0;
    int last_move_      = -1;

    std::vector<UndoEntry> history_;
};

} // namespace gomoku
