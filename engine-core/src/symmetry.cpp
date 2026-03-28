#include "gomoku/symmetry.hpp"

#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace gomoku {

// =========================================================================
// Build symmetry maps (ported from symmetry_nxn.mjs buildSymmetryMaps)
// =========================================================================

// Helper: build a map using a coordinate transform function.
// fn(r, c) -> {nr, nc} where (nr, nc) is the destination of (r, c).
// map[nr*N + nc] = r*N + c  (map[dest] = source)
static std::vector<int> build_map(int N,
    std::pair<int,int> (*fn)(int r, int c, int last)) {
    int size = N * N;
    std::vector<int> map(size);
    int last = N - 1;
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            auto [nr, nc] = fn(r, c, last);
            map[nr * N + nc] = r * N + c;
        }
    }
    return map;
}

// D4 transform functions
static std::pair<int,int> identity(int r, int c, int /*last*/) { return {r, c}; }
static std::pair<int,int> rot90(int r, int c, int last)    { return {c, last - r}; }
static std::pair<int,int> rot180(int r, int c, int last)   { return {last - r, last - c}; }
static std::pair<int,int> rot270(int r, int c, int last)   { return {last - c, r}; }
static std::pair<int,int> mirror_v(int r, int c, int last) { return {r, last - c}; }
static std::pair<int,int> mirror_h(int r, int c, int last) { return {last - r, c}; }
static std::pair<int,int> diag_main(int r, int c, int /*last*/) { return {c, r}; }
static std::pair<int,int> diag_anti(int r, int c, int last) { return {last - c, last - r}; }

std::vector<SymmetryMap> build_symmetry_maps(int N) {
    using Fn = std::pair<int,int>(*)(int, int, int);
    struct Entry { const char* name; Fn fn; };
    static const Entry entries[8] = {
        {"identity",  identity},
        {"rot90",     rot90},
        {"rot180",    rot180},
        {"rot270",    rot270},
        {"mirrorV",   mirror_v},
        {"mirrorH",   mirror_h},
        {"diagMain",  diag_main},
        {"diagAnti",  diag_anti},
    };

    std::vector<SymmetryMap> maps;
    maps.reserve(8);
    for (const auto& e : entries) {
        maps.push_back({e.name, build_map(N, e.fn)});
    }
    return maps;
}

// =========================================================================
// Cached symmetry maps per board size
// =========================================================================

static std::mutex cache_mutex;
static std::unordered_map<int, std::vector<SymmetryMap>> cache;

const std::vector<SymmetryMap>& get_symmetry_maps(int N) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(N);
    if (it == cache.end()) {
        cache[N] = build_symmetry_maps(N);
        return cache[N];
    }
    return it->second;
}

// =========================================================================
// Transform functions
// =========================================================================

std::vector<int8_t> transform_board(const int8_t* cells, const std::vector<int>& map, int N) {
    int size = N * N;
    std::vector<int8_t> out(size);
    for (int i = 0; i < size; i++) {
        out[i] = cells[map[i]];
    }
    return out;
}

std::vector<float> transform_policy(const float* policy, const std::vector<int>& map, int N) {
    int size = N * N;
    std::vector<float> out(size);
    for (int i = 0; i < size; i++) {
        out[i] = policy[map[i]];
    }
    return out;
}

std::vector<float> inverse_transform_policy(const float* policy, const std::vector<int>& map, int N) {
    int size = N * N;
    std::vector<float> out(size);
    for (int i = 0; i < size; i++) {
        out[map[i]] = policy[i];
    }
    return out;
}

int transform_move(int pos, const std::vector<int>& map) {
    // map[new_idx] = old_idx, so find new_idx where map[new_idx] == pos
    int size = static_cast<int>(map.size());
    for (int i = 0; i < size; i++) {
        if (map[i] == pos) return i;
    }
    return pos;  // shouldn't happen
}

int inverse_transform_move(int pos, const std::vector<int>& map) {
    return map[pos];
}

// =========================================================================
// Legacy API
// =========================================================================

int transform(int idx, int board_size, int symmetry_id) {
    int row = idx / board_size;
    int col = idx % board_size;
    int N = board_size - 1;
    int r2, c2;

    switch (symmetry_id) {
        case 0: r2 = row;     c2 = col;     break; // identity
        case 1: r2 = col;     c2 = N - row; break; // 90 CW
        case 2: r2 = N - row; c2 = N - col; break; // 180
        case 3: r2 = N - col; c2 = row;     break; // 270 CW
        case 4: r2 = row;     c2 = N - col; break; // horizontal flip
        case 5: r2 = N - row; c2 = col;     break; // vertical flip
        case 6: r2 = col;     c2 = row;     break; // diagonal flip
        case 7: r2 = N - col; c2 = N - row; break; // anti-diagonal flip
        default: r2 = row; c2 = col; break;
    }
    return r2 * board_size + c2;
}

std::vector<int> canonical_form(const std::vector<int>& cells, int board_size) {
    // Compute the lexicographically smallest rotation/reflection
    const auto& maps = get_symmetry_maps(board_size);
    std::vector<int> best = cells;

    for (const auto& sym : maps) {
        std::vector<int> transformed(cells.size());
        for (int i = 0; i < static_cast<int>(cells.size()); i++) {
            transformed[i] = cells[sym.map[i]];
        }
        if (transformed < best) {
            best = transformed;
        }
    }
    return best;
}

} // namespace gomoku
