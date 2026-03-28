#include "gomoku/evaluate.hpp"
#include "gomoku/patterns.hpp"

#include <cmath>
#include <algorithm>

namespace gomoku {

// Score scale for tanh normalization -- scores beyond this are essentially won/lost
static constexpr int SCORE_SCALE = 200'000;

// Direction vectors (same as in patterns.cpp)
static constexpr int DIRS[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};

static inline bool in_bounds(int r, int c, int N) {
    return r >= 0 && c >= 0 && r < N && c < N;
}

// ---------------------------------------------------------------------------
// Combination bonuses
// ---------------------------------------------------------------------------

int compute_combo_bonuses(const PatternCounts& counts, const ComboBonuses& combo) {
    int bonus = 0;

    // Double half-four = effectively open four (opponent can only block one)
    if (counts.half_four() >= 2)
        bonus += combo.double_half_four;

    // Open three + half four = unstoppable
    if (counts.open_three() >= 1 && counts.half_four() >= 1)
        bonus += combo.open_three_half_four;

    // Double open three = very strong
    if (counts.open_three() >= 2)
        bonus += combo.double_open_three;

    // Triple half three = complex multi-threat
    if (counts.half_three() >= 3)
        bonus += combo.triple_half_three;

    return bonus;
}

// ---------------------------------------------------------------------------
// Center control
// ---------------------------------------------------------------------------

static int center_control_score(const GomokuBoard& board, int8_t player) {
    const int N = board.board_size();
    const double mid = (N - 1) / 2.0;
    int score = 0;

    const int total = board.total_cells();
    for (int i = 0; i < total; i++) {
        int8_t v = board.cell_at(i);
        if (v == EMPTY) continue;

        const int r = i / N;
        const int c = i % N;
        const double dist = std::abs(r - mid) + std::abs(c - mid);
        const int center_value = std::max(0, static_cast<int>((N - dist) * 2));

        if (v == player)
            score += center_value;
        else
            score -= center_value;
    }

    return score;
}

// ---------------------------------------------------------------------------
// Full static evaluation
// ---------------------------------------------------------------------------

int evaluate(const GomokuBoard& board, int8_t player, const EngineConfig& cfg) {
    const int8_t opp = opponent(player);
    const auto& ps = cfg.pattern_scores;
    const auto& cb = cfg.combo_bonuses;

    PatternCounts my_counts  = scan_all_patterns(board, player);
    PatternCounts opp_counts = scan_all_patterns(board, opp);

    // Immediate win/loss
    if (my_counts.five() > 0) return 1'000'000;
    if (opp_counts.five() > 0) return -1'000'000;

    int my_score  = pattern_counts_to_score(my_counts, ps);
    int opp_score = pattern_counts_to_score(opp_counts, ps);

    // Combination bonuses -- multiple simultaneous threats
    my_score  += compute_combo_bonuses(my_counts, cb);
    opp_score += compute_combo_bonuses(opp_counts, cb);

    // Open four is unstoppable -- equivalent to a win
    if (my_counts.open_four() > 0) return 900'000;
    if (opp_counts.open_four() > 0) return -900'000;

    // Attacker advantage: it's player's turn, so their threats are more valuable
    const int score = static_cast<int>(my_score * 1.05) - opp_score;

    // Center control bonus (minor)
    const int center_bonus = center_control_score(board, player);

    return score + center_bonus;
}

double static_evaluate(const GomokuBoard& board, const EngineConfig& cfg) {
    // Evaluate from BLACK's perspective (convention: positive = good for current player).
    // The engine.cpp calls this for the side-to-move interpretation.
    // We use the last move's opponent (i.e., the player who just moved is not "player"),
    // so we evaluate for the player whose turn it is next.
    // Since the board doesn't track side_to_move internally, we infer from move_count:
    // even move_count = BLACK's turn, odd = WHITE's turn.
    int8_t player = (board.move_count() % 2 == 0) ? BLACK : WHITE;
    int raw = evaluate(board, player, cfg);
    return static_cast<double>(raw);
}

// ---------------------------------------------------------------------------
// Quick move evaluation (for move ordering)
// ---------------------------------------------------------------------------

int quick_move_eval(GomokuBoard& board, int pos, int8_t player,
                    const EngineConfig& cfg) {
    const int N = board.board_size();
    const int win_len = board.win_len();
    const int8_t opp = opponent(player);

    // Place stone temporarily
    board.make_move(pos, player);

    const int r = pos / N;
    const int c = pos % N;

    int score = 0;

    // Offensive: count patterns created by this move
    for (int dir = 0; dir < 4; dir++) {
        const int dr = DIRS[dir][0];
        const int dc = DIRS[dir][1];

        int fwd = 0;
        int rr = r + dr, cc = c + dc;
        while (in_bounds(rr, cc, N) && board.cell_at(rr * N + cc) == player) {
            fwd++; rr += dr; cc += dc;
        }
        bool fwd_empty = in_bounds(rr, cc, N) && board.cell_at(rr * N + cc) == EMPTY;

        int bwd = 0;
        rr = r - dr; cc = c - dc;
        while (in_bounds(rr, cc, N) && board.cell_at(rr * N + cc) == player) {
            bwd++; rr -= dr; cc -= dc;
        }
        bool bwd_empty = in_bounds(rr, cc, N) && board.cell_at(rr * N + cc) == EMPTY;

        const int len = 1 + fwd + bwd;
        const int open_ends = (fwd_empty ? 1 : 0) + (bwd_empty ? 1 : 0);

        if (len >= win_len)                             score += 1'000'000;
        else if (len == win_len - 1 && open_ends >= 1)  score += (open_ends == 2) ? 100'000 : 10'000;
        else if (len == win_len - 2 && open_ends >= 1)  score += (open_ends == 2) ? 5'000   : 1'000;
        else if (len == win_len - 3 && open_ends >= 1)  score += (open_ends == 2) ? 500     : 100;
    }

    // Defensive: evaluate how many opponent lines this move blocks.
    // Check what opponent would have had through this position.
    int def_score = 0;
    for (int dir = 0; dir < 4; dir++) {
        const int dr = DIRS[dir][0];
        const int dc = DIRS[dir][1];

        int fwd = 0;
        int rr = r + dr, cc = c + dc;
        while (in_bounds(rr, cc, N) && board.cell_at(rr * N + cc) == opp) {
            fwd++; rr += dr; cc += dc;
        }

        int bwd = 0;
        rr = r - dr; cc = c - dc;
        while (in_bounds(rr, cc, N) && board.cell_at(rr * N + cc) == opp) {
            bwd++; rr -= dr; cc -= dc;
        }

        const int opp_len = 1 + fwd + bwd; // +1 for the blocking stone itself
        if (opp_len >= win_len)          def_score += 500'000;
        else if (opp_len == win_len - 1) def_score += 50'000;
        else if (opp_len == win_len - 2) def_score += 2'500;
    }

    board.undo_move(); // restore

    // Center proximity bonus
    const double mid = (N - 1) / 2.0;
    const double center_dist = std::abs(r - mid) + std::abs(c - mid);
    const int center_bonus = std::max(0, static_cast<int>((N - center_dist) * 5));

    return score + static_cast<int>(def_score * 0.8) + center_bonus;
}

// ---------------------------------------------------------------------------
// Score normalization
// ---------------------------------------------------------------------------

double normalize_score(int raw_score) {
    if (raw_score >= 1'000'000) return 1.0;
    if (raw_score <= -1'000'000) return -1.0;
    return std::tanh(static_cast<double>(raw_score) / SCORE_SCALE);
}

} // namespace gomoku
