#pragma once

#include <cstdint>

namespace gomoku {

// ---------------------------------------------------------------------------
// Pattern evaluation scores — tuned for 5-in-a-row gomoku.
// Ported from server/src/engine/config.mjs PATTERN_SCORES.
// ---------------------------------------------------------------------------
namespace pattern_scores {
    inline constexpr int FIVE        = 1'000'000;
    inline constexpr int OPEN_FOUR   =   100'000;
    inline constexpr int HALF_FOUR   =    10'000;
    inline constexpr int OPEN_THREE  =     5'000;
    inline constexpr int HALF_THREE  =     1'000;
    inline constexpr int OPEN_TWO    =       500;
    inline constexpr int HALF_TWO    =       100;
    inline constexpr int OPEN_ONE    =        10;
} // namespace pattern_scores

struct PatternScores {
    int five        = pattern_scores::FIVE;
    int open_four   = pattern_scores::OPEN_FOUR;
    int half_four   = pattern_scores::HALF_FOUR;
    int open_three  = pattern_scores::OPEN_THREE;
    int half_three  = pattern_scores::HALF_THREE;
    int open_two    = pattern_scores::OPEN_TWO;
    int half_two    = pattern_scores::HALF_TWO;
    int open_one    = pattern_scores::OPEN_ONE;
};

// ---------------------------------------------------------------------------
// Combination bonuses when multiple threats coexist.
// Ported from config.mjs COMBO_BONUSES.
// ---------------------------------------------------------------------------
namespace combo_bonuses {
    inline constexpr int DOUBLE_HALF_FOUR      = 80'000;
    inline constexpr int OPEN_THREE_HALF_FOUR  = 60'000;
    inline constexpr int DOUBLE_OPEN_THREE     = 40'000;
    inline constexpr int TRIPLE_HALF_THREE     = 20'000;
} // namespace combo_bonuses

struct ComboBonuses {
    int double_half_four      = combo_bonuses::DOUBLE_HALF_FOUR;
    int open_three_half_four  = combo_bonuses::OPEN_THREE_HALF_FOUR;
    int double_open_three     = combo_bonuses::DOUBLE_OPEN_THREE;
    int triple_half_three     = combo_bonuses::TRIPLE_HALF_THREE;
};

// ---------------------------------------------------------------------------
// Alpha-beta search parameters.
// Ported from config.mjs SEARCH_CFG.
// ---------------------------------------------------------------------------
namespace search_defaults {
    inline constexpr int      MAX_DEPTH           = 12;
    inline constexpr int      TIME_LIMIT_MS       = 3000;
    inline constexpr uint64_t TT_SIZE             = 1u << 20;
    inline constexpr int      ASPIRATION_WINDOW   = 50;
    inline constexpr int      NULL_MOVE_REDUCTION = 3;
    inline constexpr int      LMR_THRESHOLD       = 4;
    inline constexpr int      LMR_DEPTH_MIN       = 3;
} // namespace search_defaults

struct SearchConfig {
    int      max_depth           = search_defaults::MAX_DEPTH;
    int      time_limit_ms       = search_defaults::TIME_LIMIT_MS;
    uint64_t tt_size             = search_defaults::TT_SIZE;
    int      aspiration_window   = search_defaults::ASPIRATION_WINDOW;
    int      null_move_reduction = search_defaults::NULL_MOVE_REDUCTION;
    int      lmr_threshold       = search_defaults::LMR_THRESHOLD;
    int      lmr_depth_min       = search_defaults::LMR_DEPTH_MIN;
};

// ---------------------------------------------------------------------------
// VCF / VCT threat search parameters.
// Ported from config.mjs THREAT_CFG.
// ---------------------------------------------------------------------------
namespace threat_defaults {
    inline constexpr int VCF_MAX_DEPTH = 14;
    inline constexpr int VCT_MAX_DEPTH = 8;
    inline constexpr int VCF_MAX_NODES = 50'000;
    inline constexpr int VCT_MAX_NODES = 20'000;
} // namespace threat_defaults

struct ThreatConfig {
    int vcf_max_depth = threat_defaults::VCF_MAX_DEPTH;
    int vct_max_depth = threat_defaults::VCT_MAX_DEPTH;
    int vcf_max_nodes = threat_defaults::VCF_MAX_NODES;
    int vct_max_nodes = threat_defaults::VCT_MAX_NODES;
};

// ---------------------------------------------------------------------------
// Candidate move generation parameters.
// Ported from config.mjs CANDIDATE_CFG.
// ---------------------------------------------------------------------------
namespace candidate_defaults {
    inline constexpr int RADIUS         = 2;
    inline constexpr int MAX_CANDIDATES = 30;
    inline constexpr int NN_TOP_K       = 8;
} // namespace candidate_defaults

struct CandidateConfig {
    int radius         = candidate_defaults::RADIUS;
    int max_candidates = candidate_defaults::MAX_CANDIDATES;
    int nn_top_k       = candidate_defaults::NN_TOP_K;
};

// ---------------------------------------------------------------------------
// Aggregate engine configuration — groups all sub-configs for convenience.
// ---------------------------------------------------------------------------
struct EngineConfig {
    PatternScores   pattern_scores;
    ComboBonuses    combo_bonuses;
    SearchConfig    search;
    ThreatConfig    threat;
    CandidateConfig candidate;
};

// Returns a default configuration matching server/src/engine/config.mjs
inline EngineConfig default_config() {
    return EngineConfig{};
}

} // namespace gomoku
