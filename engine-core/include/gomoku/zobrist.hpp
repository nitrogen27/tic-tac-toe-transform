#pragma once

#include <array>
#include <cstdint>

namespace gomoku {

// ---------------------------------------------------------------------------
// Zobrist hashing — provides incremental hash updates for the transposition
// table.  Pre-computes random bit-strings for each (position, player-index)
// pair using the same deterministic mulberry32 PRNG as the JavaScript engine.
//
// Player index mapping:  BLACK (+1) -> 0,  WHITE (-1) -> 1
// ---------------------------------------------------------------------------
class ZobristTable {
public:
    static constexpr int MAX_CELLS  = 256;
    static constexpr int NUM_PLAYERS = 2; // index 0 = BLACK, index 1 = WHITE

    ZobristTable();

    // Hash for placing a stone of `player_index` (0 or 1) at `cell`.
    uint64_t piece_hash(int cell, int player_index) const;

    // Side-to-move hash (XORed every time a move is made).
    uint64_t side_hash() const;

private:
    // keys_[cell][player_index]
    std::array<std::array<uint64_t, NUM_PLAYERS>, MAX_CELLS> keys_{};
    uint64_t side_key_ = 0;
};

// Global shared Zobrist table (initialised once on first call).
const ZobristTable& global_zobrist();

} // namespace gomoku
