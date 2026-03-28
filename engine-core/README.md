# Gomoku Engine (C++17)

Native C++17 engine ported from the JS reference implementation (2420 LOC → ~4000 LOC C++).
Supports board sizes 7×7 to 16×16 with alpha-beta search, threat-space search (VCF/VCT), pattern evaluation, and transposition tables.

## Status

✅ **Phase 2 COMPLETE** — 36/36 tests pass, CLI verified, benchmarks measured.

## Architecture

| Module | Description |
|--------|-------------|
| `types.hpp` | Position, EngineResult, EngineSource enum, inline helpers |
| `board.hpp/cpp` | GomokuBoard — uint64_t Zobrist hash, make/undo move, winner detection |
| `config.hpp` | Constants: FIVE=1M, OPEN_FOUR=100K, search depth=12, TT size=1M |
| `patterns.hpp/cpp` | Pattern scan + gapped patterns, find_winning_moves, count_threats |
| `evaluate.hpp/cpp` | Static eval, quick_move_eval, tanh value normalisation |
| `transposition.hpp/cpp` | Fixed-size TT, depth-based replacement, EXACT/LOWER/UPPER flags |
| `move_ordering.hpp/cpp` | Killer table (2 slots/depth) + history heuristic per player |
| `candidates.hpp/cpp` | Radius-based candidate generation, policy merge |
| `threats.hpp/cpp` | VCF threat-space search, four-creating moves |
| `search.hpp/cpp` | IDAB + PVS + LMR + aspiration windows + std::chrono time management |
| `symmetry.hpp/cpp` | D4 group (8 dihedral transforms), thread-safe transform cache |
| `engine.hpp/cpp` | 7-layer facade: SafetyWin→Block→MultiBlock→VcfWin→VcfDef→Fork→AlphaBeta |
| `cli_main.cpp` | JSON stdin/stdout CLI — newline-delimited protocol |

## Prerequisites

- CMake >= 3.20
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- Internet access for FetchContent (nlohmann-json, GoogleTest, Google Benchmark)
  - _Or_ vcpkg: set `VCPKG_ROOT` and pass `-DCMAKE_TOOLCHAIN_FILE`

## Build

```bash
# Simple build (FetchContent downloads dependencies automatically)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4

# Or via Makefile from repo root
make build-engine
```

## CLI Usage

The engine reads newline-delimited JSON from stdin and writes JSON to stdout.

```bash
# Engine info
echo '{"command":"info"}' | ./build/gomoku-engine

# Best move on empty 15x15
echo '{"command":"best-move","position":{"boardSize":15,"winLength":5,"cells":[],"sideToMove":1,"moveCount":0,"lastMove":-1,"moveHistory":[]}}' \
  | ./build/gomoku-engine
# Output: {"bestMove":112,"source":"alpha-beta","value":0.0,"depth":1,...}
# Move 112 = center (row 7, col 7) ✓

# Top-5 suggestions
echo '{"command":"suggest","n":5,"position":{...}}' | ./build/gomoku-engine
```

### Commands

| Command     | Description |
|-------------|-------------|
| `best-move` | Best move for position (returns EngineResult JSON) |
| `analyze`   | Full position analysis with PV line and top moves |
| `suggest`   | Top-N move candidates with scores |
| `info`      | Engine version, board size range, compile flags |

### Response format

```json
{
  "bestMove": 112,
  "source": "alpha-beta",
  "value": 0.0,
  "depth": 6,
  "nodesSearched": 15420,
  "timeMs": 12,
  "topMoves": [{"move":112,"score":0.02}, ...],
  "pvLine": [112, 113, 97, ...],
  "policy": []
}
```

## Tests

```bash
# Via ctest
cd build && ctest --output-on-failure

# Via Makefile
make test-engine
```

**Result: 36/36 tests pass**

Tests cover: board ops, Zobrist hash, winner detection, candidate generation,
immediate tactics (win-in-1, block-in-1), VCF search, pattern detection, integration.

## Benchmarks

```bash
./build/gomoku_benchmarks

# Or via Makefile
make bench-engine
```

**Measured results:**
- `BM_MakeMove` — **~10 ns** per move
- `BM_LegalMoves` (15×15) — **~392 ns**

## Docker

```bash
docker build -t gomoku-engine .
echo '{"command":"info"}' | docker run -i gomoku-engine
```
