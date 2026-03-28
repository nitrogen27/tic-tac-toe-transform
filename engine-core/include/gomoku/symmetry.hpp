#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace gomoku {

// A symmetry map: map[new_idx] = old_idx (maps destination to source)
struct SymmetryMap {
    std::string name;
    std::vector<int> map;
};

// Build all 8 D4 symmetry maps for an NxN board.
// 8 transforms: identity, rot90, rot180, rot270, mirrorV, mirrorH, diagMain, diagAnti
std::vector<SymmetryMap> build_symmetry_maps(int N);

// Get cached symmetry maps for board size N (builds once per size).
const std::vector<SymmetryMap>& get_symmetry_maps(int N);

// Transform a board (flat cell array) by a symmetry map.
// map[new_idx] = old_idx, so out[i] = board[map[i]]
std::vector<int8_t> transform_board(const int8_t* cells, const std::vector<int>& map, int N);

// Transform a policy distribution by a symmetry map.
// out[i] = policy[map[i]]
std::vector<float> transform_policy(const float* policy, const std::vector<int>& map, int N);

// Inverse-transform a policy distribution (undo a symmetry).
// out[map[i]] = policy[i]
std::vector<float> inverse_transform_policy(const float* policy, const std::vector<int>& map, int N);

// Transform a single move position by a symmetry map.
// Returns the position in the transformed board where `pos` maps to.
// (finds i such that map[i] == pos)
int transform_move(int pos, const std::vector<int>& map);

// Inverse-transform a move position: map[pos] gives the original position.
int inverse_transform_move(int pos, const std::vector<int>& map);

// Legacy API (single-transform by symmetry_id 0..7)
int transform(int idx, int board_size, int symmetry_id);

// Return the canonical (smallest hash) rotation of a position
std::vector<int> canonical_form(const std::vector<int>& cells, int board_size);

} // namespace gomoku
