#pragma once

#include "gomoku/board.hpp"
#include "gomoku/config.hpp"
#include "gomoku/types.hpp"

#include <vector>

namespace gomoku {

// Result of a VCF / threat search
struct ThreatResult {
    bool found = false;
    int  move  = -1;              // first move of the forcing sequence
    std::vector<int> sequence;    // full forcing move sequence
    int nodes_searched = 0;
};

// VCF Search — find a forced win by playing only four-threats.
// A "four" is a move that creates an immediate win on the NEXT move.
// The opponent MUST respond by blocking (exactly one blocking move per four).
// Search alternates: attacker plays four -> defender blocks -> repeat.
//
// Parameters:
//   board     — current board state (will be modified and restored)
//   attacker  — the attacking player (+1 or -1)
//   max_depth — max search depth in half-moves (default from config)
//   max_nodes — node budget (default from config)
ThreatResult vcf_search(GomokuBoard& board, int8_t attacker,
                        int max_depth = 14, int max_nodes = 50000);

// Find all moves for `player` that create a four (one move away from five).
// Uses line-scanning: checks consecutive stones + open ends + gapped windows.
std::vector<int> find_four_creating_moves_optimized(GomokuBoard& board, int8_t player);

// Find the best defensive move against an opponent's VCF.
// Tries each candidate move for the defender and checks if the attacker's
// VCF is broken. Returns the move that breaks the VCF, or -1 if none found.
int find_best_vcf_defense(GomokuBoard& board, int8_t defender, int8_t attacker,
                          const std::vector<int>& candidates, int vcf_max_depth = 14);

// Count how many separate four-threats exist for a player.
// Useful for detecting double-threat (fork) situations.
int count_four_threats(GomokuBoard& board, int8_t player);

// Quick check: does a player have any immediate winning move?
bool has_immediate_win(GomokuBoard& board, int8_t player);

} // namespace gomoku
