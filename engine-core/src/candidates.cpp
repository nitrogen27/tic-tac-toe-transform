#include "gomoku/candidates.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace gomoku {

std::vector<int> generate_candidates(const GomokuBoard& board,
                                     int radius,
                                     int max_candidates) {
    const int N = board.board_size();
    const int total = N * N;

    // Count actual stones on board
    int stone_count = board.move_count();
    if (stone_count == 0) {
        // Board might have been built via from_position with move_count=0
        for (int i = 0; i < total; i++) {
            if (board.cell_at(i) != EMPTY) stone_count++;
        }
    }

    // First move: play center
    if (stone_count == 0) {
        int mid = (N / 2) * N + (N / 2);
        return {mid};
    }

    // Second move: play adjacent to center or center if empty
    if (stone_count == 1) {
        int mid = (N / 2) * N + (N / 2);
        if (board.cell_at(mid) == EMPTY) return {mid};

        // Play adjacent to opponent's first stone
        std::vector<int> cands;
        int last = board.last_move();
        int lr = last / N, lc = last % N;
        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                if (dr == 0 && dc == 0) continue;
                int nr = lr + dr, nc = lc + dc;
                if (nr >= 0 && nc >= 0 && nr < N && nc < N &&
                    board.cell_at(nr * N + nc) == EMPTY) {
                    cands.push_back(nr * N + nc);
                }
            }
        }
        return cands;
    }

    // General case: all empty cells within radius of any stone
    std::vector<uint8_t> marked(total, 0);

    for (int i = 0; i < total; i++) {
        if (board.cell_at(i) == EMPTY) continue;
        int r = i / N;
        int c = i % N;
        for (int dr = -radius; dr <= radius; dr++) {
            for (int dc = -radius; dc <= radius; dc++) {
                int nr = r + dr, nc = c + dc;
                if (nr >= 0 && nc >= 0 && nr < N && nc < N) {
                    int idx = nr * N + nc;
                    if (board.cell_at(idx) == EMPTY) {
                        marked[idx] = 1;
                    }
                }
            }
        }
    }

    std::vector<int> candidates;
    candidates.reserve(64);
    for (int i = 0; i < total; i++) {
        if (marked[i]) candidates.push_back(i);
    }

    // If very few candidates, expand radius
    if (static_cast<int>(candidates.size()) < 5 && board.move_count() > 2) {
        int expand_radius = radius + 1;
        for (int i = 0; i < total; i++) {
            if (board.cell_at(i) == EMPTY) continue;
            int r = i / N;
            int c = i % N;
            for (int dr = -expand_radius; dr <= expand_radius; dr++) {
                for (int dc = -expand_radius; dc <= expand_radius; dc++) {
                    int nr = r + dr, nc = c + dc;
                    if (nr >= 0 && nc >= 0 && nr < N && nc < N) {
                        int idx = nr * N + nc;
                        if (board.cell_at(idx) == EMPTY && !marked[idx]) {
                            marked[idx] = 1;
                            candidates.push_back(idx);
                        }
                    }
                }
            }
        }
    }

    // Return all candidates (ordering handles truncation)
    return candidates;
}

std::vector<int> merge_policy_candidates(const std::vector<int>& candidates,
                                         const float* nn_policy,
                                         const GomokuBoard& board,
                                         int top_k) {
    if (nn_policy == nullptr || top_k <= 0) return candidates;

    const int total = board.board_size() * board.board_size();

    // Find top-K legal moves from policy
    struct PosProb {
        int pos;
        float prob;
    };
    std::vector<PosProb> legal_probs;
    legal_probs.reserve(total);
    for (int i = 0; i < total; i++) {
        if (board.cell_at(i) == EMPTY) {
            legal_probs.push_back({i, nn_policy[i]});
        }
    }

    std::sort(legal_probs.begin(), legal_probs.end(),
              [](const PosProb& a, const PosProb& b) {
                  return a.prob > b.prob;
              });

    // Build result: original candidates + top-K from policy (deduplicated)
    // Use a flat scan for dedup since sets are overkill for ~30 elements
    std::vector<int> result = candidates;
    int limit = std::min(top_k, static_cast<int>(legal_probs.size()));
    for (int i = 0; i < limit; i++) {
        int pos = legal_probs[i].pos;
        bool found = false;
        for (int c : result) {
            if (c == pos) { found = true; break; }
        }
        if (!found) result.push_back(pos);
    }

    return result;
}

} // namespace gomoku
