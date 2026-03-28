#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace gomoku {

// ---------------------------------------------------------------------------
// Cell constants — matching the JS convention: BLACK = +1, WHITE = -1, EMPTY = 0
// ---------------------------------------------------------------------------
using Cell = int8_t;

constexpr Cell EMPTY =  0;
constexpr Cell BLACK =  1;
constexpr Cell WHITE = -1;

constexpr int MAX_BOARD_SIZE = 16;
constexpr int MAX_CELLS      = MAX_BOARD_SIZE * MAX_BOARD_SIZE; // 256

// ---------------------------------------------------------------------------
// Inline helpers
// ---------------------------------------------------------------------------
inline constexpr int8_t opponent(int8_t player) { return static_cast<int8_t>(-player); }
inline constexpr int    idx(int r, int c, int N) { return r * N + c; }
inline constexpr int    row(int pos, int N)       { return pos / N; }
inline constexpr int    col(int pos, int N)       { return pos % N; }

// ---------------------------------------------------------------------------
// Source of the engine's chosen move
// ---------------------------------------------------------------------------
enum class EngineSource {
    SafetyWin,       // Immediate winning move detected
    SafetyBlock,     // Must block opponent's single winning threat
    SafetyMultiBlock,// Must block multiple opponent winning threats
    VcfWin,          // Victory by Continuous Four (forcing sequence)
    VcfDefense,      // Defensive response to opponent's VCF
    Fork,            // Double-threat / fork move
    AlphaBeta        // Alpha-beta search result
};

inline const char* engine_source_to_string(EngineSource src) {
    switch (src) {
        case EngineSource::SafetyWin:        return "safety-win";
        case EngineSource::SafetyBlock:      return "safety-block";
        case EngineSource::SafetyMultiBlock: return "safety-multi-block";
        case EngineSource::VcfWin:           return "vcf-win";
        case EngineSource::VcfDefense:       return "vcf-defense";
        case EngineSource::Fork:             return "fork";
        case EngineSource::AlphaBeta:        return "alpha-beta";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// A candidate move with its evaluation score
// ---------------------------------------------------------------------------
struct MoveCandidate {
    int    move  = -1;   // flat index into cells[]
    double score = 0.0;
    int    row   = -1;
    int    col   = -1;
};

// ---------------------------------------------------------------------------
// Represents a board position (compact, serialisable — matches JSON schema)
// ---------------------------------------------------------------------------
struct Position {
    int board_size  = 15;
    int win_length  = 5;
    std::array<Cell, MAX_CELLS> cells{};
    Cell side_to_move = BLACK;
    int    move_count    = 0;
    int    last_move     = -1;
    std::vector<int> move_history;

    Position() { cells.fill(EMPTY); }
};

// ---------------------------------------------------------------------------
// Result returned by the engine after analysis
// ---------------------------------------------------------------------------
struct EngineResult {
    int      best_move        = -1;
    double   value            = 0.0;
    std::string source;                  // human-readable source string
    int      depth            = 0;
    uint64_t nodes_searched   = 0;
    double   time_ms          = 0.0;
    std::vector<MoveCandidate> top_moves;
    std::vector<int>    pv_line;         // principal variation
    std::vector<double> policy;          // NN policy distribution (if available)
};

} // namespace gomoku
