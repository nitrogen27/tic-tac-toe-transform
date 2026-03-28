#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace gomoku {

// Transposition table entry flags
enum class TTFlag : uint8_t {
    Exact      = 0,  // Exact score
    LowerBound = 1,  // Score is a lower bound (beta cutoff)
    UpperBound = 2   // Score is an upper bound (alpha cutoff)
};

struct TTEntry {
    uint64_t key   = 0;
    int      score = 0;
    int      move  = -1;
    int      depth = 0;
    TTFlag   flag  = TTFlag::Exact;
};

struct TTStats {
    size_t   size   = 0;
    uint64_t hits   = 0;
    uint64_t misses = 0;
    uint64_t stores = 0;
    double   hit_rate = 0.0;  // percentage
};

// Fixed-size hash table for storing previously evaluated positions.
// Uses Zobrist keys with depth-based replacement.
// Size is rounded up to next power of 2 for fast indexing via bitmask.
class TranspositionTable {
public:
    explicit TranspositionTable(uint64_t size = 1 << 20);

    // Probe the table for a position
    std::optional<TTEntry> probe(uint64_t key) const;

    // Store a position in the table (replace if new depth >= existing depth)
    void store(uint64_t key, int score, int move, int depth, TTFlag flag);

    // Try to use a TT entry for cutoff: returns usable score or nullopt
    std::optional<int> try_use_entry(uint64_t key, int depth, int alpha, int beta);

    // Get the best move from TT for move ordering (-1 if none)
    int get_best_move(uint64_t key) const;

    // Clear all entries and reset stats
    void clear();

    // Get table statistics
    TTStats stats() const;

    size_t table_size() const { return table_.size(); }

private:
    std::vector<TTEntry> table_;
    uint64_t mask_ = 0;  // size - 1, for fast indexing
    mutable uint64_t hits_   = 0;
    mutable uint64_t misses_ = 0;
    uint64_t stores_ = 0;

    // Round up to next power of 2
    static uint64_t next_power_of_2(uint64_t v);
};

} // namespace gomoku
