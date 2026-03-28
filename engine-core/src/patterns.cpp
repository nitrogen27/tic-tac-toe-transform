#include "gomoku/patterns.hpp"

#include <cmath>
#include <vector>

namespace gomoku {

// Direction vectors: horizontal, vertical, diagonal, anti-diagonal
static constexpr int DIRS[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static inline bool in_bounds(int r, int c, int N) {
    return r >= 0 && c >= 0 && r < N && c < N;
}

// Check if there is enough space (own stones + empty cells) around a
// contiguous group of `len` stones starting at (r,c) in direction (dr,dc)
// to potentially form a line of `win_len`.
static bool has_enough_space(const GomokuBoard& board, int win_len, int8_t player,
                             int r, int c, int dr, int dc, int len) {
    const int N = board.board_size();
    int space = len;

    // Extend forward through player or empty cells
    int rr = r + len * dr;
    int cc = c + len * dc;
    while (space < win_len && in_bounds(rr, cc, N)) {
        int8_t v = board.cell_at(rr * N + cc);
        if (v == player || v == EMPTY) {
            space++;
            rr += dr;
            cc += dc;
        } else {
            break;
        }
    }

    // Extend backward
    rr = r - dr;
    cc = c - dc;
    while (space < win_len && in_bounds(rr, cc, N)) {
        int8_t v = board.cell_at(rr * N + cc);
        if (v == player || v == EMPTY) {
            space++;
            rr -= dr;
            cc -= dc;
        } else {
            break;
        }
    }

    return space >= win_len;
}

// Classify a contiguous group of `len` stones with `open_ends` open endpoints.
static void classify_pattern(PatternCounts& counts, int len, int open_ends) {
    if (len >= 4) {
        if (open_ends == 2) counts.open_four()++;
        else                counts.half_four()++;
    } else if (len == 3) {
        if (open_ends == 2) counts.open_three()++;
        else                counts.half_three()++;
    } else if (len == 2) {
        if (open_ends == 2) counts.open_two()++;
        else                counts.half_two()++;
    } else if (len == 1) {
        if (open_ends == 2) counts.open_one()++;
        // single stone with only 1 open end is PAT_NONE (no threat)
    }
}

// Collect starting positions for every line in a given direction.
struct Coord { int r, c; };

static std::vector<Coord> get_line_starts(int N, int dr, int dc) {
    std::vector<Coord> starts;
    starts.reserve(static_cast<size_t>(2 * N));

    if (dr == 0 && dc == 1) {
        // Horizontal: leftmost column of each row
        for (int r = 0; r < N; r++) starts.push_back({r, 0});
    } else if (dr == 1 && dc == 0) {
        // Vertical: top row of each column
        for (int c = 0; c < N; c++) starts.push_back({0, c});
    } else if (dr == 1 && dc == 1) {
        // Diagonal: top row + left column (skip (0,0) duplicate)
        for (int c = 0; c < N; c++) starts.push_back({0, c});
        for (int r = 1; r < N; r++) starts.push_back({r, 0});
    } else if (dr == 1 && dc == -1) {
        // Anti-diagonal: top row + right column
        for (int c = 0; c < N; c++) starts.push_back({0, c});
        for (int r = 1; r < N; r++) starts.push_back({r, N - 1});
    }
    return starts;
}

// Scan for gapped patterns (one internal gap within a window of win_len).
// Examples: X_XXX, XX_XX, XXX_X, and two-gap patterns like X_X_X.
static void scan_gapped_patterns(const GomokuBoard& board, int8_t player,
                                 PatternCounts& counts) {
    const int N = board.board_size();
    const int win_len = board.win_len();

    struct LineCell { int r; int c; int8_t v; };

    for (int dir = 0; dir < 4; dir++) {
        const int dr = DIRS[dir][0];
        const int dc = DIRS[dir][1];
        auto starts = get_line_starts(N, dr, dc);

        for (const auto& start : starts) {
            // Collect the line
            std::vector<LineCell> line;
            int rr = start.r, cc = start.c;
            while (in_bounds(rr, cc, N)) {
                line.push_back({rr, cc, board.cell_at(rr * N + cc)});
                rr += dr;
                cc += dc;
            }

            if (static_cast<int>(line.size()) < win_len) continue;

            // Sliding window of size win_len
            for (int i = 0; i <= static_cast<int>(line.size()) - win_len; i++) {
                int player_count = 0;
                int empty_count  = 0;
                bool has_opp = false;

                for (int j = i; j < i + win_len; j++) {
                    if (line[j].v == player)     player_count++;
                    else if (line[j].v == EMPTY) empty_count++;
                    else { has_opp = true; break; }
                }

                if (has_opp) continue;

                // Pattern with exactly 1 gap (win_len-1 stones + 1 empty)
                if (player_count == win_len - 1 && empty_count == 1) {
                    // Find gap position within the window
                    int gap_idx = -1;
                    for (int j = 0; j < win_len; j++) {
                        if (line[i + j].v == EMPTY) { gap_idx = j; break; }
                    }
                    // Internal gap only (edges already counted by contiguous scan)
                    if (gap_idx > 0 && gap_idx < win_len - 1) {
                        counts.half_four()++;
                    }
                }

                // Pattern with 2 gaps (win_len-2 stones + 2 empty)
                // These are gapped threes depending on endpoints
                if (player_count == win_len - 2 && empty_count == 2 && win_len >= 5) {
                    int internal_gaps = 0;
                    for (int j = 1; j < win_len - 1; j++) {
                        if (line[i + j].v == EMPTY) internal_gaps++;
                    }
                    if (internal_gaps >= 1 && player_count >= 3) {
                        bool before_open = (i > 0 && line[i - 1].v == EMPTY);
                        bool after_open  = (i + win_len < static_cast<int>(line.size())
                                            && line[i + win_len].v == EMPTY);
                        if (internal_gaps == 1) {
                            if (before_open || after_open) {
                                counts.half_three()++;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

PatternCounts scan_all_patterns(const GomokuBoard& board, int8_t player) {
    PatternCounts counts{};
    const int N = board.board_size();
    const int win_len = board.win_len();
    const int total = N * N;

    // visited[pos * 4 + dir] -- avoids double-counting contiguous groups
    std::vector<uint8_t> visited(static_cast<size_t>(total * 4), 0);

    for (int dir = 0; dir < 4; dir++) {
        const int dr = DIRS[dir][0];
        const int dc = DIRS[dir][1];

        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                const int pos = r * N + c;
                if (board.cell_at(pos) != player) continue;
                if (visited[pos * 4 + dir]) continue;

                // Extend forward from (r, c) in direction (dr, dc)
                int len = 1;
                int rr = r + dr, cc = c + dc;
                while (in_bounds(rr, cc, N) && board.cell_at(rr * N + cc) == player) {
                    visited[(rr * N + cc) * 4 + dir] = 1;
                    len++;
                    rr += dr;
                    cc += dc;
                }
                visited[pos * 4 + dir] = 1;

                // Already a win
                if (len >= win_len) {
                    counts.five()++;
                    continue;
                }

                // Check endpoints
                const int fwd_r = r + len * dr;
                const int fwd_c = c + len * dc;
                bool fwd_open = in_bounds(fwd_r, fwd_c, N)
                                && board.cell_at(fwd_r * N + fwd_c) == EMPTY;

                const int bwd_r = r - dr;
                const int bwd_c = c - dc;
                bool bwd_open = in_bounds(bwd_r, bwd_c, N)
                                && board.cell_at(bwd_r * N + bwd_c) == EMPTY;

                int open_ends = (fwd_open ? 1 : 0) + (bwd_open ? 1 : 0);

                if (open_ends == 0) continue; // completely blocked

                // Verify enough room to form win_len
                if (!has_enough_space(board, win_len, player, r, c, dr, dc, len))
                    continue;

                classify_pattern(counts, len, open_ends);
            }
        }
    }

    // Scan for gapped patterns (X_XXX, XX_XX, XXX_X, etc.)
    scan_gapped_patterns(board, player, counts);

    return counts;
}

int pattern_counts_to_score(const PatternCounts& counts, const PatternScores& scores) {
    int s = 0;
    s += counts.five()       * scores.five;
    s += counts.open_four()  * scores.open_four;
    s += counts.half_four()  * scores.half_four;
    s += counts.open_three() * scores.open_three;
    s += counts.half_three() * scores.half_three;
    s += counts.open_two()   * scores.open_two;
    s += counts.half_two()   * scores.half_two;
    s += counts.open_one()   * scores.open_one;
    return s;
}

int score_patterns(const PatternCounts& counts, const PatternScores& scores,
                   const ComboBonuses& combo) {
    int total = pattern_counts_to_score(counts, scores);

    // Combo bonuses
    if (counts.half_four() >= 2)
        total += combo.double_half_four;
    if (counts.open_three() >= 1 && counts.half_four() >= 1)
        total += combo.open_three_half_four;
    if (counts.open_three() >= 2)
        total += combo.double_open_three;
    if (counts.half_three() >= 3)
        total += combo.triple_half_three;

    return total;
}

// ---------------------------------------------------------------------------
// Per-move evaluation helpers (gapped fours through a specific position)
// ---------------------------------------------------------------------------

// Check if placing at (r,c) creates gapped four patterns.
// The stone must already be placed on the board.
static int evaluate_gapped_fours(const GomokuBoard& board, int8_t player,
                                 int r, int c, const PatternScores& scores) {
    const int N = board.board_size();
    const int win_len = board.win_len();
    int total = 0;

    for (int dir = 0; dir < 4; dir++) {
        const int dr = DIRS[dir][0];
        const int dc = DIRS[dir][1];

        // Check windows of size win_len that include (r, c)
        for (int offset = 0; offset < win_len; offset++) {
            const int start_r = r - offset * dr;
            const int start_c = c - offset * dc;

            int player_count = 0;
            int empty_count  = 0;
            bool valid = true;

            for (int k = 0; k < win_len; k++) {
                int kr = start_r + k * dr;
                int kc = start_c + k * dc;
                if (!in_bounds(kr, kc, N)) { valid = false; break; }
                int8_t v = board.cell_at(kr * N + kc);
                if (v == player)     player_count++;
                else if (v == EMPTY) empty_count++;
                else { valid = false; break; }
            }

            if (!valid) continue;

            if (player_count == win_len - 1 && empty_count == 1) {
                // Find gap position
                for (int k = 0; k < win_len; k++) {
                    int kr = start_r + k * dr;
                    int kc = start_c + k * dc;
                    if (board.cell_at(kr * N + kc) == EMPTY) {
                        // Internal gap -- partial credit to avoid double-counting
                        if (k > 0 && k < win_len - 1) {
                            total += static_cast<int>(scores.half_four * 0.5);
                        }
                        break;
                    }
                }
            }
        }
    }

    return total;
}

int evaluate_move_patterns(GomokuBoard& board, int pos, int8_t player,
                           const PatternScores& scores) {
    const int N = board.board_size();
    const int win_len = board.win_len();
    int score = 0;

    // Temporarily place the stone
    board.make_move(pos, player);

    const int r = pos / N;
    const int c = pos % N;

    for (int dir = 0; dir < 4; dir++) {
        const int dr = DIRS[dir][0];
        const int dc = DIRS[dir][1];

        // Count consecutive in both directions from pos
        int fwd = 0;
        int rr = r + dr, cc = c + dc;
        while (in_bounds(rr, cc, N) && board.cell_at(rr * N + cc) == player) {
            fwd++;
            rr += dr;
            cc += dc;
        }
        bool fwd_open = in_bounds(rr, cc, N) && board.cell_at(rr * N + cc) == EMPTY;

        int bwd = 0;
        rr = r - dr; cc = c - dc;
        while (in_bounds(rr, cc, N) && board.cell_at(rr * N + cc) == player) {
            bwd++;
            rr -= dr;
            cc -= dc;
        }
        bool bwd_open = in_bounds(rr, cc, N) && board.cell_at(rr * N + cc) == EMPTY;

        const int len = 1 + fwd + bwd;
        const int open_ends = (fwd_open ? 1 : 0) + (bwd_open ? 1 : 0);

        if (len >= win_len) {
            score += scores.five;
        } else if (len == win_len - 1) {
            score += (open_ends == 2) ? scores.open_four
                   : (open_ends == 1) ? scores.half_four
                   : 0;
        } else if (len == win_len - 2) {
            score += (open_ends == 2) ? scores.open_three
                   : (open_ends == 1) ? scores.half_three
                   : 0;
        } else if (len == win_len - 3) {
            score += (open_ends == 2) ? scores.open_two
                   : (open_ends == 1) ? scores.half_two
                   : 0;
        }
    }

    // Also check for gapped fours through this position
    score += evaluate_gapped_fours(board, player, r, c, scores);

    // Remove the stone
    board.undo_move();

    return score;
}

// ---------------------------------------------------------------------------
// Threat queries
// ---------------------------------------------------------------------------

std::vector<int> find_winning_moves(GomokuBoard& board, int8_t player) {
    std::vector<int> moves;
    const int total = board.total_cells();

    for (int pos = 0; pos < total; pos++) {
        if (board.cell_at(pos) != EMPTY) continue;

        board.make_move(pos, player);
        if (board.is_winning_move(pos, player)) {
            moves.push_back(pos);
        }
        board.undo_move();
    }

    return moves;
}

std::vector<int> find_four_creating_moves(GomokuBoard& board, int8_t player) {
    std::vector<int> moves;
    const int total = board.total_cells();
    const int N = board.board_size();
    const int win_len = board.win_len();

    for (int pos = 0; pos < total; pos++) {
        if (board.cell_at(pos) != EMPTY) continue;

        board.make_move(pos, player);

        // Check if this creates any winning threat for player:
        // scan all empty cells and see if placing player there would complete win_len.
        bool creates_threat = false;
        for (int pos2 = 0; pos2 < total && !creates_threat; pos2++) {
            if (board.cell_at(pos2) != EMPTY) continue;

            // Check if placing player at pos2 would create five-in-a-row
            // by scanning lines through pos2 (without actually placing).
            const int r2 = pos2 / N;
            const int c2 = pos2 % N;

            for (int dir = 0; dir < 4; dir++) {
                const int dr = DIRS[dir][0];
                const int dc = DIRS[dir][1];
                int count = 1; // counting pos2 itself as player

                // Forward
                int rr = r2 + dr, cc = c2 + dc;
                while (in_bounds(rr, cc, N) && board.cell_at(rr * N + cc) == player) {
                    count++;
                    rr += dr;
                    cc += dc;
                }

                // Backward
                rr = r2 - dr; cc = c2 - dc;
                while (in_bounds(rr, cc, N) && board.cell_at(rr * N + cc) == player) {
                    count++;
                    rr -= dr;
                    cc -= dc;
                }

                if (count >= win_len) {
                    creates_threat = true;
                    break;
                }
            }
        }

        board.undo_move();

        if (creates_threat) {
            moves.push_back(pos);
        }
    }

    return moves;
}

int count_threats(GomokuBoard& board, int8_t player) {
    return static_cast<int>(find_winning_moves(board, player).size());
}

int count_opponent_threats_after_move(GomokuBoard& board, int pos, int8_t player) {
    const int8_t opp = opponent(player);
    board.make_move(pos, player);
    auto threats = find_winning_moves(board, opp);
    board.undo_move();
    return static_cast<int>(threats.size());
}

PatternEvalResult evaluate_patterns(const GomokuBoard& board, int8_t player,
                                    const PatternScores& scores) {
    const int8_t opp = opponent(player);
    PatternEvalResult result{};
    result.my_counts  = scan_all_patterns(board, player);
    result.opp_counts = scan_all_patterns(board, opp);
    result.my_score   = pattern_counts_to_score(result.my_counts, scores);
    result.opp_score  = pattern_counts_to_score(result.opp_counts, scores);
    return result;
}

} // namespace gomoku
