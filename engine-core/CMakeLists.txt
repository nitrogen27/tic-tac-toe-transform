cmake_minimum_required(VERSION 3.20)
project(gomoku-engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# --- nlohmann_json: try vcpkg first, then FetchContent ---
find_package(nlohmann_json CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found via find_package, using FetchContent")
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# --- GTest: try vcpkg first, then FetchContent ---
find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
    message(STATUS "GTest not found via find_package, using FetchContent")
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
        GIT_SHALLOW    TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# --- benchmark: try vcpkg first, then FetchContent ---
find_package(benchmark CONFIG QUIET)
if(NOT benchmark_FOUND)
    message(STATUS "benchmark not found via find_package, using FetchContent")
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.8.3
        GIT_SHALLOW    TRUE
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
endif()

# --- Collect all library sources ---
file(GLOB GOMOKU_CORE_SOURCES src/*.cpp)
# Remove cli_main.cpp from the library sources
list(FILTER GOMOKU_CORE_SOURCES EXCLUDE REGEX ".*cli_main\.cpp$")

# --- Static library ---
add_library(gomoku_core STATIC ${GOMOKU_CORE_SOURCES})
target_include_directories(gomoku_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# --- CLI executable ---
add_executable(gomoku-engine src/cli_main.cpp)
target_link_libraries(gomoku-engine PRIVATE
    gomoku_core
    nlohmann_json::nlohmann_json
)

# --- Tests ---
enable_testing()
file(GLOB TEST_SOURCES tests/*.cpp)
if(TEST_SOURCES)
    add_executable(gomoku_tests ${TEST_SOURCES})
    target_link_libraries(gomoku_tests PRIVATE
        gomoku_core
        GTest::gtest
        GTest::gtest_main
    )
    include(GoogleTest)
    gtest_discover_tests(gomoku_tests)
endif()

# --- Benchmarks ---
file(GLOB BENCH_SOURCES benchmarks/*.cpp)
if(BENCH_SOURCES)
    add_executable(gomoku_benchmarks ${BENCH_SOURCES})
    target_link_libraries(gomoku_benchmarks PRIVATE
        gomoku_core
        benchmark::benchmark
        benchmark::benchmark_main
    )
endif()
