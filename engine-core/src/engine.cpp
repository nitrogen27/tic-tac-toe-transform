#include "gomoku/engine.hpp"
#include "gomoku/board.hpp"
#include "gomoku/search.hpp"
#include "gomoku/threats.hpp"
#include "gomoku/evaluate.hpp"
#include "gomoku/candidates.hpp"
#include "gomoku/patterns.hpp"

#include <algorithm>
#include <sstream>

namespace gomoku {

// ---------------------------------------------------------------------------
// Helper: build a one-hot policy vector from a single move
// ---------------------------------------------------------------------------
static std::vector<double> build_one_hot_policy(int move, int board_cells) {
    std::vector<double> policy(board_cells, 0.0);
    if (move >= 0 && move < board_cells)
        policy[move] = 1.0;
    return policy;
}

// ---------------------------------------------------------------------------
// Helper: find the best defense move when opponent has multiple winning threats
// ---------------------------------------------------------------------------
static int find_best_defense_move(GomokuBoard& board, Cell side,
                                  const std::vector<int>& opp_win_moves,
                                  const std::vector<int>& candidates) {
    int best_move = opp_win_moves.empty() ? -1 : opp_win_moves[0];
    int best_score = -1000000;
    Cell opp = opponent(side);

    for (int mv : candidates) {
        if (!board.is_empty(mv)) continue;

        board.make_move(mv, side);

        // Count remaining opponent threats after our move
        int remaining = 0;
        for (int wm : opp_win_moves) {
            if (!board.is_empty(wm)) continue; // blocked by our move
            board.make_move(wm, opp);
            if (board.is_winning_move(wm, opp)) {
                remaining++;
            }
            board.undo_move();
        }

        // Also count own four-threats (prefer counter-threats)
        int own_threats = count_four_threats(board, side);

        int score = -remaining * 10000 + own_threats * 5000;

        board.undo_move();

        if (score > best_score) {
            best_score = score;
            best_move = mv;
        }
    }

    return best_move;
}

// ---------------------------------------------------------------------------
// Helper: find a fork move (creates 2+ simultaneous four-threats)
// ---------------------------------------------------------------------------
static int find_fork_move(GomokuBoard& board, Cell side,
                          const std::vector<int>& candidates) {
    for (int mv : candidates) {
        if (!board.is_empty(mv)) continue;

        board.make_move(mv, side);
        int threats = count_four_threats(board, side);
        board.undo_move();

        if (threats >= 2) return mv;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
GomokuEngine::GomokuEngine(EngineConfig cfg)
    : config_(std::move(cfg))
{}

// ---------------------------------------------------------------------------
// best_move -- 7-layer decision hierarchy
// ---------------------------------------------------------------------------
EngineResult GomokuEngine::best_move(const Position& pos) {
    auto board = GomokuBoard::from_position(pos);
    Cell side = pos.side_to_move;
    Cell opp  = opponent(side);
    int  cells = board.total_cells();

    // --- Layer 1: Immediate win ---
    auto win_moves = find_winning_moves(board, side);
    if (!win_moves.empty()) {
        EngineResult r;
        r.best_move      = win_moves[0];
        r.value           = 1.0;
        r.source          = engine_source_to_string(EngineSource::SafetyWin);
        r.depth           = 1;
        r.nodes_searched  = 1;
        r.policy          = build_one_hot_policy(win_moves[0], cells);
        return r;
    }

    // --- Layer 2: Single block required ---
    auto opp_win_moves = find_winning_moves(board, opp);
    if (opp_win_moves.size() == 1) {
        EngineResult r;
        r.best_move      = opp_win_moves[0];
        r.value           = -0.3;
        r.source          = engine_source_to_string(EngineSource::SafetyBlock);
        r.depth           = 1;
        r.nodes_searched  = 1;
        r.policy          = build_one_hot_policy(opp_win_moves[0], cells);
        return r;
    }

    // --- Layer 3: Multi-block defense ---
    if (opp_win_moves.size() >= 2) {
        auto candidates = generate_candidates(board,
                                              config_.candidate.radius,
                                              config_.candidate.max_candidates);
        int def_move = find_best_defense_move(board, side, opp_win_moves, candidates);
        EngineResult r;
        r.best_move      = def_move;
        r.value           = -0.8;
        r.source          = engine_source_to_string(EngineSource::SafetyMultiBlock);
        r.depth           = 1;
        r.nodes_searched  = 1;
        r.policy          = build_one_hot_policy(def_move, cells);
        return r;
    }

    // --- Layer 4: VCF search (forced win) ---
    auto vcf = vcf_search(board, side,
                           config_.threat.vcf_max_depth,
                           config_.threat.vcf_max_nodes);
    if (vcf.found && !vcf.sequence.empty()) {
        EngineResult r;
        r.best_move      = vcf.sequence[0];
        r.value           = 1.0;
        r.source          = engine_source_to_string(EngineSource::VcfWin);
        r.depth           = static_cast<int>(vcf.sequence.size());
        r.nodes_searched  = static_cast<uint64_t>(vcf.nodes_searched);
        r.pv_line         = vcf.sequence;
        r.policy          = build_one_hot_policy(vcf.sequence[0], cells);
        return r;
    }

    // --- Layer 5: VCF defense ---
    auto opp_vcf = vcf_search(board, opp,
                               config_.threat.vcf_max_depth,
                               config_.threat.vcf_max_nodes);
    if (opp_vcf.found) {
        auto candidates = generate_candidates(board,
                                              config_.candidate.radius,
                                              config_.candidate.max_candidates);
        int def_move = find_best_vcf_defense(board, side, opp,
                                              candidates,
                                              config_.threat.vcf_max_depth);
        if (def_move >= 0) {
            EngineResult r;
            r.best_move      = def_move;
            r.value           = -0.5;
            r.source          = engine_source_to_string(EngineSource::VcfDefense);
            r.depth           = config_.threat.vcf_max_depth;
            r.nodes_searched  = static_cast<uint64_t>(candidates.size());
            r.policy          = build_one_hot_policy(def_move, cells);
            return r;
        }
        // No defense found -- fall through to alpha-beta
    }

    // --- Layer 6: Fork detection (double-threat) ---
    {
        auto candidates = generate_candidates(board,
                                              config_.candidate.radius,
                                              config_.candidate.max_candidates);
        int fork_mv = find_fork_move(board, side, candidates);
        if (fork_mv >= 0) {
            EngineResult r;
            r.best_move      = fork_mv;
            r.value           = 0.9;
            r.source          = engine_source_to_string(EngineSource::Fork);
            r.depth           = 2;
            r.nodes_searched  = 1;
            r.policy          = build_one_hot_policy(fork_mv, cells);
            return r;
        }
    }

    // --- Layer 7: Alpha-beta search ---
    return alpha_beta_search(board, config_);
}

// ---------------------------------------------------------------------------
// evaluate
// ---------------------------------------------------------------------------
double GomokuEngine::evaluate(const Position& pos) {
    auto board = GomokuBoard::from_position(pos);
    return static_evaluate(board, config_);
}

// ---------------------------------------------------------------------------
// get_hints
// ---------------------------------------------------------------------------
std::vector<MoveCandidate> GomokuEngine::get_hints(const Position& pos, int top_n) {
    auto board = GomokuBoard::from_position(pos);
    Cell side  = pos.side_to_move;

    auto candidates = generate_candidates(board,
                                          config_.candidate.radius,
                                          config_.candidate.max_candidates);

    std::vector<MoveCandidate> hints;
    hints.reserve(candidates.size());

    for (int mv : candidates) {
        MoveCandidate mc;
        mc.move  = mv;
        mc.row   = mv / pos.board_size;
        mc.col   = mv % pos.board_size;
        mc.score = static_cast<double>(
            quick_move_eval(board, mv, side, config_));
        hints.push_back(mc);
    }

    // Sort descending by score
    std::sort(hints.begin(), hints.end(),
              [](const MoveCandidate& a, const MoveCandidate& b) {
                  return a.score > b.score;
              });

    if (static_cast<int>(hints.size()) > top_n)
        hints.resize(top_n);

    return hints;
}

// ---------------------------------------------------------------------------
// find_vcf
// ---------------------------------------------------------------------------
std::vector<int> GomokuEngine::find_vcf(const Position& pos) {
    auto board = GomokuBoard::from_position(pos);
    Cell side = pos.side_to_move;
    auto result = vcf_search(board, side,
                              config_.threat.vcf_max_depth,
                              config_.threat.vcf_max_nodes);
    return result.sequence;
}

// ---------------------------------------------------------------------------
// clear_cache
// ---------------------------------------------------------------------------
void GomokuEngine::clear_cache() {
    // TODO: clear transposition table, killer moves, history table
}

// ---------------------------------------------------------------------------
// stats
// ---------------------------------------------------------------------------
std::string GomokuEngine::stats() const {
    std::ostringstream oss;
    oss << "GomokuEngine v1.0.0\n"
        << "  max_depth: " << config_.search.max_depth << "\n"
        << "  time_limit: " << config_.search.time_limit_ms << " ms\n"
        << "  tt_size: " << config_.search.tt_size << " entries\n";
    return oss.str();
}

} // namespace gomoku
