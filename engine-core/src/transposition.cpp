#include "gomoku/transposition.hpp"

namespace gomoku {

uint64_t TranspositionTable::next_power_of_2(uint64_t v) {
    if (v == 0) return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
}

TranspositionTable::TranspositionTable(uint64_t size) {
    uint64_t actual = next_power_of_2(size);
    table_.resize(actual);
    mask_ = actual - 1;
}

std::optional<TTEntry> TranspositionTable::probe(uint64_t key) const {
    const auto& entry = table_[key & mask_];
    if (entry.key == key) {
        hits_++;
        return entry;
    }
    misses_++;
    return std::nullopt;
}

void TranspositionTable::store(uint64_t key, int score, int move, int depth, TTFlag flag) {
    auto& entry = table_[key & mask_];
    // Replacement policy: don't replace deeper entries for the same slot
    if (entry.key != 0 && entry.key != key && entry.depth > depth) {
        return;
    }
    entry.key   = key;
    entry.score = score;
    entry.move  = move;
    entry.depth = depth;
    entry.flag  = flag;
    stores_++;
}

std::optional<int> TranspositionTable::try_use_entry(uint64_t key, int depth, int alpha, int beta) {
    const auto& entry = table_[key & mask_];
    if (entry.key != key || entry.depth < depth) {
        return std::nullopt;
    }
    hits_++;

    if (entry.flag == TTFlag::Exact) {
        return entry.score;
    }
    if (entry.flag == TTFlag::LowerBound && entry.score >= beta) {
        return entry.score;
    }
    if (entry.flag == TTFlag::UpperBound && entry.score <= alpha) {
        return entry.score;
    }
    return std::nullopt;
}

int TranspositionTable::get_best_move(uint64_t key) const {
    const auto& entry = table_[key & mask_];
    if (entry.key == key) {
        return entry.move;
    }
    return -1;
}

void TranspositionTable::clear() {
    std::fill(table_.begin(), table_.end(), TTEntry{});
    hits_   = 0;
    misses_ = 0;
    stores_ = 0;
}

TTStats TranspositionTable::stats() const {
    TTStats s;
    s.hits   = hits_;
    s.misses = misses_;
    s.stores = stores_;
    // Count occupied entries
    size_t occupied = 0;
    for (const auto& e : table_) {
        if (e.key != 0) occupied++;
    }
    s.size = occupied;
    uint64_t total = hits_ + misses_;
    s.hit_rate = total > 0 ? (static_cast<double>(hits_) / total * 100.0) : 0.0;
    return s;
}

} // namespace gomoku
