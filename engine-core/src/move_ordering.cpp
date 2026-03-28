#include "gomoku/move_ordering.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>

namespace gomoku {

// =========================================================================
// KillerTable
// =========================================================================

KillerTable::KillerTable(int max_depth) : table_(max_depth, {-1, -1}) {}

void KillerTable::store(int ply, int move) {
    if (ply < 0 || ply >= static_cast<int>(table_.size())) return;
    if (table_[ply][0] == move) return;  // already primary killer
    table_[ply][1] = table_[ply][0];
    table_[ply][0] = move;
}

std::array<int, 2> KillerTable::get(int ply) const {
    if (ply < 0 || ply >= static_cast<int>(table_.size())) return {-1, -1};
    return table_[ply];
}

void KillerTable::clear() {
    for (auto& entry : table_) {
        entry = {-1, -1};
    }
}

// =========================================================================
// HistoryTable
// =========================================================================

HistoryTable::HistoryTable(int board_size) : size_(board_size) {
    scores_[0].resize(board_size, 0);
    scores_[1].resize(board_size, 0);
}

void HistoryTable::record(int8_t player, int move, int depth) {
    int idx = player_index(player);
    if (move >= 0 && move < size_) {
        scores_[idx][move] += static_cast<uint32_t>(depth * depth);
    }
}

uint32_t HistoryTable::get_score(int8_t player, int move) const {
    int idx = player_index(player);
    if (move >= 0 && move < size_) {
        return scores_[idx][move];
    }
    return 0;
}

const uint32_t* HistoryTable::get_table(int8_t player) const {
    return scores_[player_index(player)].data();
}

void HistoryTable::age() {
    for (int p = 0; p < 2; p++) {
        for (int i = 0; i < size_; i++) {
            scores_[p][i] >>= 1;  // halve (equivalent to (x/2)|0 in JS)
        }
    }
}

void HistoryTable::clear() {
    std::fill(scores_[0].begin(), scores_[0].end(), 0u);
    std::fill(scores_[1].begin(), scores_[1].end(), 0u);
}

// =========================================================================
// Quick move evaluation — ported from evaluate.mjs quickMoveEval
// =========================================================================

// Inline quick pattern eval around a single move (attack + defense value).
// Temporarily places stones to measure line lengths.
static int quick_move_eval(const GomokuBoard& board, int pos, int8_t player) {
    const int N = board.board_size();
    const int win_len = board.win_len();
    const int r = pos / N;
    const int c = pos % N;

    // Directions: horizontal, vertical, main diagonal, anti-diagonal
    static constexpr int dr[4] = {0, 1, 1, 1};
    static constexpr int dc[4] = {1, 0, 1, -1};

    int score = 0;

    // --- Offensive value: how good is this move for `player`? ---
    for (int d = 0; d < 4; d++) {
        int fwd = 0, bwd = 0;
        int rr, cc;

        // Forward
        rr = r + dr[d]; cc = c + dc[d];
        while (rr >= 0 && cc >= 0 && rr < N && cc < N &&
               board.cell_at(rr * N + cc) == player) {
            fwd++; rr += dr[d]; cc += dc[d];
        }
        bool fwd_empty = (rr >= 0 && cc >= 0 && rr < N && cc < N &&
                          board.cell_at(rr * N + cc) == EMPTY);

        // Backward
        rr = r - dr[d]; cc = c - dc[d];
        while (rr >= 0 && cc >= 0 && rr < N && cc < N &&
               board.cell_at(rr * N + cc) == player) {
            bwd++; rr -= dr[d]; cc -= dc[d];
        }
        bool bwd_empty = (rr >= 0 && cc >= 0 && rr < N && cc < N &&
                          board.cell_at(rr * N + cc) == EMPTY);

        int len = 1 + fwd + bwd;
        int open_ends = (fwd_empty ? 1 : 0) + (bwd_empty ? 1 : 0);

        if (len >= win_len)                         score += 1'000'000;
        else if (len == win_len - 1 && open_ends >= 1) score += (open_ends == 2 ? 100'000 : 10'000);
        else if (len == win_len - 2 && open_ends >= 1) score += (open_ends == 2 ? 5'000 : 1'000);
        else if (len == win_len - 3 && open_ends >= 1) score += (open_ends == 2 ? 500 : 100);
    }

    // --- Defensive value: how much does this block the opponent? ---
    int8_t opp = opponent(player);
    int def_score = 0;
    for (int d = 0; d < 4; d++) {
        int fwd = 0, bwd = 0;
        int rr, cc;

        rr = r + dr[d]; cc = c + dc[d];
        while (rr >= 0 && cc >= 0 && rr < N && cc < N &&
               board.cell_at(rr * N + cc) == opp) {
            fwd++; rr += dr[d]; cc += dc[d];
        }

        rr = r - dr[d]; cc = c - dc[d];
        while (rr >= 0 && cc >= 0 && rr < N && cc < N &&
               board.cell_at(rr * N + cc) == opp) {
            bwd++; rr -= dr[d]; cc -= dc[d];
        }

        int opp_len = 1 + fwd + bwd;
        if (opp_len >= win_len)        def_score += 500'000;
        else if (opp_len == win_len - 1) def_score += 50'000;
        else if (opp_len == win_len - 2) def_score += 2'500;
    }

    // Center proximity bonus
    double mid = (N - 1) / 2.0;
    int center_dist = std::abs(r - static_cast<int>(mid)) + std::abs(c - static_cast<int>(mid));
    int center_bonus = std::max(0, N - center_dist) * 5;

    return score + static_cast<int>(def_score * 0.8) + center_bonus;
}

// =========================================================================
// find_winning_moves helper (used by order_moves_light)
// =========================================================================

static std::vector<int> find_winning_moves(GomokuBoard& board, int8_t player) {
    std::vector<int> moves;
    int total = board.board_size() * board.board_size();
    for (int pos = 0; pos < total; pos++) {
        if (board.cell_at(pos) != EMPTY) continue;
        // Temporarily place, check, undo
        // We need to use make_move which uses side_to_move_ so we
        // can't do that simply. Instead we check using is_winning_move
        // pattern: the JS version directly sets cells[pos] = player and checks.
        // In C++ we don't have direct cell access, so we use a different approach:
        // We check if placing at pos would complete a line for player.
        // Since is_winning_move checks the cell's current occupant, we need
        // the board to have the stone there. Use make_move / undo_move.

        // Note: make_move uses side_to_move_, but we need to place for `player`.
        // The board's side_to_move might not match `player`.
        // Workaround: check lines manually.

        // Actually, let's just check manually:
        const int N = board.board_size();
        int r0 = pos / N, c0 = pos % N;
        bool wins = false;
        static constexpr int dr[4] = {0, 1, 1, 1};
        static constexpr int dc[4] = {1, 0, 1, -1};
        for (int d = 0; d < 4 && !wins; d++) {
            int count = 1;
            for (int sign = -1; sign <= 1; sign += 2) {
                for (int step = 1; step < board.win_len(); step++) {
                    int rr = r0 + sign * step * dr[d];
                    int cc = c0 + sign * step * dc[d];
                    if (rr < 0 || rr >= N || cc < 0 || cc >= N) break;
                    if (board.cell_at(rr * N + cc) != player) break;
                    count++;
                }
            }
            if (count >= board.win_len()) wins = true;
        }
        if (wins) moves.push_back(pos);
    }
    return moves;
}

// =========================================================================
// order_moves — full ordering for alpha-beta search
// =========================================================================

std::vector<int> order_moves(const GomokuBoard& board, int8_t player,
                             const std::vector<int>& candidates,
                             const MoveOrderingOpts& opts) {
    struct ScoredMove {
        int move;
        int score;
    };

    const int n = static_cast<int>(candidates.size());
    std::vector<ScoredMove> scored(n);

    for (int i = 0; i < n; i++) {
        int move = candidates[i];
        int s = 0;

        // 1. TT best move — highest priority
        if (move == opts.tt_best_move) {
            s += 10'000'000;
        }

        // 2. Killer moves
        if (opts.killer_moves[0] == move) s += 500'000;
        else if (opts.killer_moves[1] == move) s += 400'000;

        // 3. NN policy score
        if (opts.nn_policy != nullptr) {
            s += static_cast<int>(opts.nn_policy[move] * 200'000.0f);
        }

        // 4. Quick pattern evaluation (attack + defense)
        s += quick_move_eval(board, move, player);

        // 5. History heuristic
        if (opts.history_table != nullptr) {
            s += std::min(100'000, static_cast<int>(opts.history_table[move]));
        }

        // 6. Proximity to last move (minor bonus)
        int last = board.last_move();
        if (last >= 0) {
            int N = board.board_size();
            int lr = last / N, lc = last % N;
            int mr = move / N, mc = move % N;
            int dist = std::max(std::abs(mr - lr), std::abs(mc - lc));
            if (dist <= 1) s += 3000;
            else if (dist <= 2) s += 1500;
        }

        scored[i] = {move, s};
    }

    // Sort descending by score
    std::sort(scored.begin(), scored.end(),
              [](const ScoredMove& a, const ScoredMove& b) {
                  return a.score > b.score;
              });

    std::vector<int> result(n);
    for (int i = 0; i < n; i++) {
        result[i] = scored[i].move;
    }
    return result;
}

// =========================================================================
// order_moves_light — lightweight ordering for shallow/quiescence search
// =========================================================================

std::vector<int> order_moves_light(GomokuBoard& board, int8_t player,
                                   const std::vector<int>& candidates) {
    // Check for immediate winning moves
    auto win_moves = find_winning_moves(board, player);
    if (!win_moves.empty()) {
        // Put winning moves first, then the rest
        std::vector<int> result;
        result.reserve(candidates.size());
        // Add winning moves that are in candidates
        for (int m : win_moves) {
            result.push_back(m);
        }
        // Add remaining candidates (exclude wins)
        for (int m : candidates) {
            bool is_win = false;
            for (int w : win_moves) {
                if (w == m) { is_win = true; break; }
            }
            if (!is_win) result.push_back(m);
        }
        return result;
    }

    // Check for blocking moves
    auto block_moves = find_winning_moves(board, opponent(player));
    if (!block_moves.empty()) {
        std::vector<int> result;
        result.reserve(candidates.size());
        for (int m : block_moves) {
            result.push_back(m);
        }
        for (int m : candidates) {
            bool is_block = false;
            for (int b : block_moves) {
                if (b == m) { is_block = true; break; }
            }
            if (!is_block) result.push_back(m);
        }
        return result;
    }

    // No special ordering
    return candidates;
}

} // namespace gomoku
