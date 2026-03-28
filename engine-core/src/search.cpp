#include "gomoku/search.hpp"
#include "gomoku/candidates.hpp"
#include "gomoku/evaluate.hpp"
#include "gomoku/move_ordering.hpp"
#include "gomoku/threats.hpp"
#include "gomoku/transposition.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>

namespace gomoku {

static constexpr int INF       = 9'999'999;
static constexpr int WIN_SCORE = 1'000'000;

// Normalize raw score to [-1, +1] range (matching JS normalizeScore)
static double normalize_score(int score) {
    if (score >= WIN_SCORE) return 1.0;
    if (score <= -WIN_SCORE) return -1.0;
    constexpr double SCORE_SCALE = 200'000.0;
    return std::tanh(score / SCORE_SCALE);
}

// Find all moves that create an immediate win for `player`
static std::vector<int> find_winning_moves(const GomokuBoard& board, int8_t player) {
    const int N = board.board_size();
    const int total = N * N;
    const int win_len = board.win_len();
    std::vector<int> moves;

    static constexpr int dr[4] = {0, 1, 1, 1};
    static constexpr int dc[4] = {1, 0, 1, -1};

    for (int pos = 0; pos < total; pos++) {
        if (board.cell_at(pos) != EMPTY) continue;
        int r0 = pos / N, c0 = pos % N;
        bool wins = false;
        for (int d = 0; d < 4 && !wins; d++) {
            int count = 1;
            for (int sign = -1; sign <= 1; sign += 2) {
                for (int step = 1; step < win_len; step++) {
                    int rr = r0 + sign * step * dr[d];
                    int cc = c0 + sign * step * dc[d];
                    if (rr < 0 || rr >= N || cc < 0 || cc >= N) break;
                    if (board.cell_at(rr * N + cc) != player) break;
                    count++;
                }
            }
            if (count >= win_len) wins = true;
        }
        if (wins) moves.push_back(pos);
    }
    return moves;
}

// =========================================================================
// Internal search state — captured by the alpha-beta closure
// =========================================================================

struct SearchState {
    GomokuBoard& board;
    TranspositionTable& tt;
    KillerTable killers;
    HistoryTable history;

    const SearchOptions& opts;
    const float* nn_policy;
    int8_t root_player;

    uint64_t nodes_searched = 0;
    bool aborted = false;

    std::chrono::steady_clock::time_point start_time;
    int time_limit_ms;

    SearchState(GomokuBoard& b, TranspositionTable& t, const SearchOptions& o, int8_t player)
        : board(b), tt(t)
        , killers(o.max_depth + 2)
        , history(b.board_size() * b.board_size())
        , opts(o)
        , nn_policy(o.nn_policy)
        , root_player(player)
        , time_limit_ms(o.time_limit_ms)
    {
        start_time = std::chrono::steady_clock::now();
    }

    double elapsed_ms() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_time).count();
    }
};

// =========================================================================
// Alpha-beta with negamax convention
// =========================================================================

static int alpha_beta(SearchState& state, int depth, int alpha, int beta,
                      int8_t current_player, int ply) {
    state.nodes_searched++;

    // Time check every 4096 nodes
    if ((state.nodes_searched & 4095) == 0) {
        if (state.elapsed_ms() > state.time_limit_ms) {
            state.aborted = true;
            return 0;
        }
    }

    // Terminal check
    Cell w = state.board.winner();
    if (w == EMPTY) {
        return 0;  // draw
    }
    if (w != GomokuBoard::ONGOING) {
        // w is BLACK or WHITE — someone won
        if (w == current_player)
            return WIN_SCORE - ply;
        else
            return -(WIN_SCORE - ply);
    }

    // Transposition table probe
    uint64_t hash_key = state.board.hash_key();
    auto tt_score = state.tt.try_use_entry(hash_key, depth, alpha, beta);
    if (tt_score.has_value()) {
        return tt_score.value();
    }

    // Depth 0: static evaluation
    if (depth <= 0) {
        return static_cast<int>(static_evaluate(state.board, EngineConfig{}));
    }

    // Generate and order moves
    auto candidates = generate_candidates(state.board,
                                           state.opts.candidate_radius,
                                           state.opts.max_candidates);

    if (candidates.empty()) {
        return 0;  // no moves — draw
    }

    int tt_best_move = state.tt.get_best_move(hash_key);
    MoveOrderingOpts mo_opts;
    mo_opts.tt_best_move = tt_best_move;
    mo_opts.killer_moves = state.killers.get(ply);
    mo_opts.nn_policy    = (ply == 0) ? state.nn_policy : nullptr;  // NN policy only at root
    mo_opts.history_table = state.history.get_table(current_player);

    auto ordered_moves = order_moves(state.board, current_player, candidates, mo_opts);

    int best_move_local = ordered_moves[0];
    int best_score_local = -INF;
    int original_alpha = alpha;
    int move_index = 0;

    for (int move : ordered_moves) {
        if (state.aborted) return 0;

        state.board.make_move(move, current_player);

        int score;

        if (move_index == 0) {
            // Full window search for first (best) move
            score = -alpha_beta(state, depth - 1, -beta, -alpha,
                                opponent(current_player), ply + 1);
        } else {
            // Late Move Reductions
            int reduction = 0;
            if (move_index >= state.opts.lmr_threshold &&
                depth >= state.opts.lmr_depth_min) {
                reduction = 1;
                if (move_index >= 8) reduction = 2;
            }

            // Null window search with possible reduction
            score = -alpha_beta(state, depth - 1 - reduction, -alpha - 1, -alpha,
                                opponent(current_player), ply + 1);

            // Re-search if failed high
            if (score > alpha && (reduction > 0 || score < beta)) {
                score = -alpha_beta(state, depth - 1, -beta, -alpha,
                                    opponent(current_player), ply + 1);
            }
        }

        state.board.undo_move();

        if (state.aborted) return 0;

        if (score > best_score_local) {
            best_score_local = score;
            best_move_local = move;
        }

        if (score > alpha) {
            alpha = score;
        }

        if (alpha >= beta) {
            // Beta cutoff
            state.killers.store(ply, move);
            state.history.record(current_player, move, depth);
            break;
        }

        move_index++;
    }

    // Store in transposition table
    TTFlag flag;
    if (best_score_local >= beta)
        flag = TTFlag::LowerBound;
    else if (best_score_local <= original_alpha)
        flag = TTFlag::UpperBound;
    else
        flag = TTFlag::Exact;

    state.tt.store(hash_key, best_score_local, best_move_local, depth, flag);

    return best_score_local;
}

// =========================================================================
// Iterative deepening search — main entry point
// =========================================================================

SearchResult iterative_deepening_search(GomokuBoard& board, int8_t player,
                                        const SearchOptions& opts,
                                        TranspositionTable& tt) {
    SearchState state(board, tt, opts, player);

    int best_move = -1;
    int best_score = 0;
    int completed_depth = 0;

    // Check for immediate wins FIRST
    auto win_moves = find_winning_moves(board, player);
    if (!win_moves.empty()) {
        SearchResult r;
        r.best_move = win_moves[0];
        r.score = WIN_SCORE;
        r.normalized_score = 1.0;
        r.depth = 1;
        r.nodes_searched = 1;
        r.time_ms = 0.0;
        r.pv = {win_moves[0]};
        return r;
    }

    // Check if we must block
    auto opp_win_moves = find_winning_moves(board, opponent(player));
    if (opp_win_moves.size() == 1) {
        SearchResult r;
        r.best_move = opp_win_moves[0];
        r.score = -WIN_SCORE + 100;
        r.normalized_score = -0.9;
        r.depth = 1;
        r.nodes_searched = 1;
        r.time_ms = 0.0;
        r.pv = {opp_win_moves[0]};
        return r;
    }

    // Generate root candidates
    auto root_candidates = generate_candidates(board,
                                                opts.candidate_radius,
                                                opts.max_candidates);
    if (opts.nn_policy) {
        root_candidates = merge_policy_candidates(root_candidates, opts.nn_policy,
                                                   board, opts.nn_top_k);
    }

    // Ensure blocking moves are in candidates
    if (opp_win_moves.size() > 1) {
        for (int m : opp_win_moves) {
            bool found = false;
            for (int c : root_candidates) {
                if (c == m) { found = true; break; }
            }
            if (!found) root_candidates.push_back(m);
        }
    }

    if (root_candidates.empty()) {
        SearchResult r;
        r.best_move = -1;
        r.score = 0;
        r.normalized_score = 0.0;
        r.depth = 0;
        r.nodes_searched = 0;
        r.time_ms = 0.0;
        return r;
    }

    if (root_candidates.size() == 1) {
        SearchResult r;
        r.best_move = root_candidates[0];
        r.score = 0;
        r.normalized_score = 0.0;
        r.depth = 1;
        r.nodes_searched = 1;
        r.time_ms = 0.0;
        r.pv = {root_candidates[0]};
        return r;
    }

    // Iterative deepening loop
    for (int depth = 1; depth <= opts.max_depth; depth++) {
        state.aborted = false;

        // Aspiration window
        int alpha, beta;
        if (depth > 2 && std::abs(best_score) < WIN_SCORE - 100) {
            alpha = best_score - opts.aspiration_window;
            beta  = best_score + opts.aspiration_window;
        } else {
            alpha = -INF;
            beta  = INF;
        }

        // Root search — manually iterate moves to track best move
        uint64_t root_hash = board.hash_key();
        int tt_best = tt.get_best_move(root_hash);

        MoveOrderingOpts mo_opts;
        mo_opts.tt_best_move = (best_move >= 0) ? best_move : tt_best;
        mo_opts.killer_moves = state.killers.get(0);
        mo_opts.nn_policy    = opts.nn_policy;
        mo_opts.history_table = state.history.get_table(player);

        auto ordered_moves = order_moves(board, player, root_candidates, mo_opts);

        int root_best_move = ordered_moves[0];
        int root_best_score = -INF;
        int root_alpha = alpha;

        for (int i = 0; i < static_cast<int>(ordered_moves.size()); i++) {
            int move = ordered_moves[i];
            board.make_move(move, player);

            int score;
            if (i == 0) {
                score = -alpha_beta(state, depth - 1, -beta, -root_alpha,
                                    opponent(player), 1);
            } else {
                // Null window
                score = -alpha_beta(state, depth - 1, -root_alpha - 1, -root_alpha,
                                    opponent(player), 1);
                if (score > root_alpha && score < beta) {
                    score = -alpha_beta(state, depth - 1, -beta, -root_alpha,
                                        opponent(player), 1);
                }
            }

            board.undo_move();

            if (state.aborted) break;

            if (score > root_best_score) {
                root_best_score = score;
                root_best_move = move;
            }
            if (score > root_alpha) root_alpha = score;
            if (root_alpha >= beta) break;
        }

        if (state.aborted && depth > 1) {
            // Time ran out — use result from previous complete iteration
            break;
        }

        // Aspiration fail — re-search with full window
        if (!state.aborted &&
            (root_best_score <= alpha || root_best_score >= beta)) {
            state.aborted = false;
            root_best_score = -INF;
            root_alpha = -INF;

            for (int i = 0; i < static_cast<int>(ordered_moves.size()); i++) {
                int move = ordered_moves[i];
                board.make_move(move, player);
                int score = -alpha_beta(state, depth - 1, -INF, -root_alpha,
                                        opponent(player), 1);
                board.undo_move();

                if (state.aborted) break;
                if (score > root_best_score) {
                    root_best_score = score;
                    root_best_move = move;
                }
                if (score > root_alpha) root_alpha = score;
            }

            if (state.aborted && depth > 1) break;
        }

        if (!state.aborted) {
            best_move = root_best_move;
            best_score = root_best_score;
            completed_depth = depth;
        }

        // Early termination on guaranteed win/loss
        if (std::abs(best_score) >= WIN_SCORE - 100) break;

        // Time check before next iteration
        double elapsed = state.elapsed_ms();
        if (elapsed > opts.time_limit_ms * 0.6) break;  // likely won't finish next depth

        // Age history between iterations
        state.history.age();
    }

    SearchResult result;
    result.best_move = best_move;
    result.score = best_score;
    result.normalized_score = normalize_score(best_score);
    result.depth = completed_depth;
    result.nodes_searched = state.nodes_searched;
    result.time_ms = state.elapsed_ms();
    result.tt_stats = tt.stats();
    return result;
}

// =========================================================================
// Convenience wrapper: alpha_beta_search using EngineConfig
// =========================================================================

EngineResult alpha_beta_search(GomokuBoard& board, const EngineConfig& cfg) {
    // Infer side to move from move count parity
    int8_t player = (board.move_count() % 2 == 0) ? BLACK : WHITE;

    SearchOptions opts;
    opts.max_depth        = cfg.search.max_depth;
    opts.time_limit_ms    = cfg.search.time_limit_ms;
    opts.aspiration_window = cfg.search.aspiration_window;
    opts.lmr_threshold    = cfg.search.lmr_threshold;
    opts.lmr_depth_min    = cfg.search.lmr_depth_min;
    opts.candidate_radius = cfg.candidate.radius;
    opts.max_candidates   = cfg.candidate.max_candidates;
    opts.nn_top_k         = cfg.candidate.nn_top_k;
    opts.nn_policy        = nullptr;

    TranspositionTable tt(cfg.search.tt_size);

    SearchResult sr = iterative_deepening_search(board, player, opts, tt);

    EngineResult result;
    result.best_move      = sr.best_move;
    result.value          = sr.normalized_score;
    result.source         = engine_source_to_string(EngineSource::AlphaBeta);
    result.depth          = sr.depth;
    result.nodes_searched = sr.nodes_searched;
    result.time_ms        = sr.time_ms;
    result.pv_line        = sr.pv;
    return result;
}

} // namespace gomoku
