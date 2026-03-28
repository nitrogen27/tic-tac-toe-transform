// config.cpp — intentionally minimal.
//
// All configuration constants are declared as static constexpr in config.hpp
// and require no translation-unit definitions in C++17 (inline by default).
//
// This file exists so that build systems that glob src/*.cpp continue to work.

#include "gomoku/config.hpp"

namespace gomoku {
// No runtime definitions needed — all values are constexpr in the header.
} // namespace gomoku
