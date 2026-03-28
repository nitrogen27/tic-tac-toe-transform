#include "gomoku/zobrist.hpp"

namespace gomoku {

// ---------------------------------------------------------------------------
// mulberry32 — exact port of the JS PRNG used in board.mjs.
// Returns an unsigned 32-bit value each call.
// ---------------------------------------------------------------------------
namespace {

class Mulberry32 {
public:
    explicit Mulberry32(uint32_t seed) : state_(seed) {}

    uint32_t operator()() {
        // Matches the JS implementation exactly:
        //   seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        //   let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        //   t = t + Math.imul(t ^ (t >>> 7), 61 | t) ^ t;
        //   return ((t ^ (t >>> 14)) >>> 0);
        state_ += 0x6D2B79F5u;
        uint32_t t = state_;
        t = (t ^ (t >> 15)) * (1u | state_);
        // The multiply: Math.imul(t ^ (t >>> 7), 61 | t) — both operands
        // are interpreted as signed 32-bit in JS but the bit pattern is the same.
        t = (t + ((t ^ (t >> 7)) * (61u | t))) ^ t;
        return t ^ (t >> 14);
    }

private:
    uint32_t state_;
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// ZobristTable constructor — seeds with 0xDEADBEEF (matching JS code).
//
// For each of the 256 positions and 2 player indices the JS code generates
// *two* 32-bit random numbers (hi, lo) to form a 64-bit hash pair because
// JavaScript lacks native 64-bit integers.  In C++ we combine them into a
// single uint64_t:  key = (uint64_t(hi) << 32) | lo
// ---------------------------------------------------------------------------
ZobristTable::ZobristTable() {
    Mulberry32 rng(0xDEADBEEF);

    for (int i = 0; i < MAX_CELLS; i++) {
        // Player index 0 (BLACK / +1): two rng calls -> hi, lo
        uint32_t hi0 = rng();
        uint32_t lo0 = rng();
        keys_[i][0] = (static_cast<uint64_t>(hi0) << 32) | lo0;

        // Player index 1 (WHITE / -1): two rng calls -> hi, lo
        uint32_t hi1 = rng();
        uint32_t lo1 = rng();
        keys_[i][1] = (static_cast<uint64_t>(hi1) << 32) | lo1;
    }

    // Side-to-move key: JS generates ZOBRIST_SIDE[0] and ZOBRIST_SIDE[1]
    uint32_t shi = rng();
    uint32_t slo = rng();
    side_key_ = (static_cast<uint64_t>(shi) << 32) | slo;
}

uint64_t ZobristTable::piece_hash(int cell, int player_index) const {
    return keys_[cell][player_index];
}

uint64_t ZobristTable::side_hash() const {
    return side_key_;
}

const ZobristTable& global_zobrist() {
    static ZobristTable instance;
    return instance;
}

} // namespace gomoku
