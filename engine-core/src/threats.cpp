#include "gomoku/threats.hpp"

#include <cstdint>

namespace gomoku {

// =========================================================================
// Internal helpers
// =========================================================================

// Find all moves that create an immediate win (five in a row) for `player`.
// This is equivalent to JS findWinningMoves: try each empty cell, check if
// placing there completes a line.
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
            int count = 1;  // the stone we'd place
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

// Check if a move is winning for the player whose stone is at that cell.
// Used after make_move to verify a win.
static bool is_winning_at(const GomokuBoard& board, int pos, int8_t player) {
    const int N = board.board_size();
    const int win_len = board.win_len();
    int r0 = pos / N, c0 = pos % N;

    static constexpr int dr[4] = {0, 1, 1, 1};
    static constexpr int dc[4] = {1, 0, 1, -1};

    for (int d = 0; d < 4; d++) {
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
        if (count >= win_len) return true;
    }
    return false;
}

// =========================================================================
// find_four_creating_moves_optimized
// =========================================================================

std::vector<int> find_four_creating_moves_optimized(GomokuBoard& board, int8_t player) {
    const int N = board.board_size();
    const int win_len = board.win_len();
    const int total = N * N;
    std::vector<int> moves;
    std::vector<uint8_t> seen(total, 0);

    static constexpr int dr[4] = {0, 1, 1, 1};
    static constexpr int dc[4] = {1, 0, 1, -1};

    for (int pos = 0; pos < total; pos++) {
        if (board.cell_at(pos) != EMPTY || seen[pos]) continue;

        // Temporarily place the stone using make_move.
        // IMPORTANT: make_move uses side_to_move_. We need to place for `player`.
        // Since the board alternates sides, we must work at cell level.
        // But we don't have direct cell write access. Instead, we'll do the
        // analysis using only cell_at reads and manual line scanning.

        // Approach: simulate placement by reading neighbors. If pos were `player`:
        int r = pos / N, c = pos % N;
        bool creates_four = false;

        for (int d = 0; d < 4 && !creates_four; d++) {
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

            // Creates five directly
            if (len >= win_len) {
                creates_four = true;
                break;
            }

            // Creates a four (winLen-1 in a row with at least one open end)
            if (len == win_len - 1 && (fwd_empty || bwd_empty)) {
                creates_four = true;
                break;
            }

            // Check for gapped fours: windows of size winLen that include pos
            for (int offset = 0; offset < win_len && !creates_four; offset++) {
                int start_r = r - offset * dr[d];
                int start_c = c - offset * dc[d];
                int p_count = 0, e_count = 0;
                bool valid = true;

                for (int k = 0; k < win_len; k++) {
                    int kr = start_r + k * dr[d];
                    int kc = start_c + k * dc[d];
                    if (kr < 0 || kc < 0 || kr >= N || kc >= N) {
                        valid = false; break;
                    }
                    int idx = kr * N + kc;
                    int8_t v;
                    if (idx == pos) {
                        v = player;  // simulated placement
                    } else {
                        v = board.cell_at(idx);
                    }
                    if (v == player) p_count++;
                    else if (v == EMPTY) e_count++;
                    else { valid = false; break; }
                }

                if (valid && p_count == win_len - 1 && e_count == 1) {
                    creates_four = true;
                }
            }
        }

        if (creates_four) {
            moves.push_back(pos);
            seen[pos] = 1;
        }
    }

    return moves;
}

// =========================================================================
// VCF search (recursive)
// =========================================================================

struct VcfState {
    int nodes_searched = 0;
    int max_nodes = 50000;
};

// Recursive VCF search. Returns the winning sequence or empty vector.
static std::vector<int> vcf_recurse(GomokuBoard& board, int8_t attacker,
                                     int depth, std::vector<int>& sequence,
                                     VcfState& state) {
    if (state.nodes_searched++ > state.max_nodes) return {};
    if (depth <= 0) return {};

    // Attacker's turn: find four-creating moves
    auto four_moves = find_four_creating_moves_optimized(board, attacker);

    for (int attack_move : four_moves) {
        // Make the attack move. Note: board.make_move uses side_to_move_.
        // We need to ensure side_to_move_ matches attacker.
        // The VCF alternates attacker -> defender -> attacker, and the
        // board side also alternates. As long as we start with correct side,
        // it stays in sync.
        const int8_t defender = opponent(attacker);
        board.make_move(attack_move, attacker);

        // Check if this move itself is a win (five in a row)
        if (is_winning_at(board, attack_move, attacker)) {
            board.undo_move();
            sequence.push_back(attack_move);
            return sequence;
        }

        // Find the threat(s) this creates
        auto threats = find_winning_moves(board, attacker);

        if (static_cast<int>(threats.size()) >= 2) {
            // Double threat = immediate win (opponent can block only one)
            board.undo_move();
            sequence.push_back(attack_move);
            return sequence;
        }

        if (threats.size() == 1) {
            // Single threat — opponent must block
            int block_move = threats[0];
            board.make_move(block_move, defender);

            // After block, check if game continues
            Cell w = board.winner();
            if (w == GomokuBoard::ONGOING) {  // game still in progress
                // Continue: attacker plays next four
                sequence.push_back(attack_move);
                sequence.push_back(block_move);
                auto result = vcf_recurse(board, attacker, depth - 2, sequence, state);
                if (!result.empty()) {
                    board.undo_move();  // undo block
                    board.undo_move();  // undo attack
                    return result;
                }
                // Remove the moves we added
                sequence.pop_back();
                sequence.pop_back();
            }

            board.undo_move();  // undo block
        }

        board.undo_move();  // undo attack
    }

    return {};  // no winning sequence found
}

// =========================================================================
// Public API
// =========================================================================

ThreatResult vcf_search(GomokuBoard& board, int8_t attacker,
                        int max_depth, int max_nodes) {
    ThreatResult result;

    // Quick check: if attacker already has a winning move, return it
    auto immediate_wins = find_winning_moves(board, attacker);
    if (!immediate_wins.empty()) {
        result.found = true;
        result.move = immediate_wins[0];
        result.sequence = {immediate_wins[0]};
        result.nodes_searched = 1;
        return result;
    }

    // With the new API, make_move takes an explicit player argument,
    // so no side_to_move check is needed.

    VcfState state;
    state.max_nodes = max_nodes;

    std::vector<int> sequence;
    auto seq = vcf_recurse(board, attacker, max_depth, sequence, state);
    result.nodes_searched = state.nodes_searched;

    if (!seq.empty()) {
        result.found = true;
        result.move = seq[0];
        result.sequence = std::move(seq);
    }

    return result;
}

int find_best_vcf_defense(GomokuBoard& board, int8_t defender, int8_t attacker,
                          const std::vector<int>& candidates, int vcf_max_depth) {
    for (int def_move : candidates) {
        if (board.cell_at(def_move) != EMPTY) continue;

        board.make_move(def_move, defender);

        // Check if attacker still has VCF with reduced depth
        auto attack_vcf = vcf_search(board, attacker, vcf_max_depth - 2, 50000);

        board.undo_move();

        if (!attack_vcf.found) {
            // This move breaks the VCF
            return def_move;
        }
    }
    return -1;  // no defense found
}

bool has_immediate_win(GomokuBoard& board, int8_t player) {
    auto wins = find_winning_moves(board, player);
    return !wins.empty();
}

int count_four_threats(GomokuBoard& board, int8_t player) {
    // Count empty cells where placing player's stone would create five
    auto wins = find_winning_moves(board, player);
    return static_cast<int>(wins.size());
}

} // namespace gomoku
