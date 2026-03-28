#include <benchmark/benchmark.h>
#include "gomoku/board.hpp"
#include "gomoku/search.hpp"
#include "gomoku/config.hpp"

static void BM_LegalMoves(benchmark::State& state) {
    gomoku::GomokuBoard board(15, 5);
    board.make_move(7 * 15 + 7, gomoku::BLACK);
    board.make_move(7 * 15 + 8, gomoku::WHITE);
    board.make_move(8 * 15 + 7, gomoku::BLACK);

    for (auto _ : state) {
        auto moves = board.legal_moves();
        benchmark::DoNotOptimize(moves);
    }
}
BENCHMARK(BM_LegalMoves);

static void BM_MakeUndoMove(benchmark::State& state) {
    gomoku::GomokuBoard board(15, 5);
    for (auto _ : state) {
        board.make_move(0, gomoku::BLACK);
        board.undo_move();
    }
}
BENCHMARK(BM_MakeUndoMove);

static void BM_SearchStub(benchmark::State& state) {
    gomoku::GomokuBoard board(15, 5);
    board.make_move(7 * 15 + 7, gomoku::BLACK);
    auto cfg = gomoku::default_config();

    for (auto _ : state) {
        auto result = gomoku::alpha_beta_search(board, cfg);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_SearchStub);

BENCHMARK_MAIN();
