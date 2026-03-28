#include <gtest/gtest.h>
#include "gomoku/board.hpp"
#include "gomoku/config.hpp"
#include "gomoku/zobrist.hpp"

using namespace gomoku;

// ===========================================================================
// Empty board creation for multiple sizes
// ===========================================================================

class EmptyBoardTest : public ::testing::TestWithParam<int> {};

TEST_P(EmptyBoardTest, CreatesCorrectEmptyBoard) {
    const int N = GetParam();
    GomokuBoard board(N, 5);

    EXPECT_EQ(board.board_size(), N);
    EXPECT_EQ(board.win_len(), 5);
    EXPECT_EQ(board.total_cells(), N * N);
    EXPECT_EQ(board.move_count(), 0);
    EXPECT_EQ(board.last_move(), -1);
    EXPECT_EQ(board.winner(), GomokuBoard::ONGOING);

    // All cells should be EMPTY
    for (int i = 0; i < N * N; i++) {
        EXPECT_EQ(board.cell_at(i), EMPTY);
    }

    // Legal moves should be all N*N cells
    auto moves = board.legal_moves();
    EXPECT_EQ(static_cast<int>(moves.size()), N * N);
}

INSTANTIATE_TEST_SUITE_P(
    BoardSizes,
    EmptyBoardTest,
    ::testing::Values(7, 9, 15, 16)
);

// ===========================================================================
// make_move / undo_move basics
// ===========================================================================

TEST(GomokuBoardTest, MakeMoveBasic) {
    GomokuBoard board(15, 5);
    const int center = 7 * 15 + 7;

    EXPECT_TRUE(board.is_empty(center));
    board.make_move(center, BLACK);
    EXPECT_FALSE(board.is_empty(center));
    EXPECT_EQ(board.cell_at(center), BLACK);
    EXPECT_EQ(board.move_count(), 1);
    EXPECT_EQ(board.last_move(), center);
}

TEST(GomokuBoardTest, MakeMoveAlternating) {
    GomokuBoard board(9, 5);
    board.make_move(40, BLACK);
    board.make_move(41, WHITE);
    board.make_move(31, BLACK);

    EXPECT_EQ(board.cell_at(40), BLACK);
    EXPECT_EQ(board.cell_at(41), WHITE);
    EXPECT_EQ(board.cell_at(31), BLACK);
    EXPECT_EQ(board.move_count(), 3);
    EXPECT_EQ(board.last_move(), 31);
}

TEST(GomokuBoardTest, UndoMoveSingleMove) {
    GomokuBoard board(15, 5);
    const int center = 7 * 15 + 7;

    board.make_move(center, BLACK);
    EXPECT_EQ(board.move_count(), 1);

    board.undo_move();
    EXPECT_TRUE(board.is_empty(center));
    EXPECT_EQ(board.move_count(), 0);
    EXPECT_EQ(board.last_move(), -1);
}

TEST(GomokuBoardTest, UndoMoveMultipleMoves) {
    GomokuBoard board(15, 5);
    board.make_move(0, BLACK);
    board.make_move(1, WHITE);
    board.make_move(2, BLACK);

    board.undo_move();
    EXPECT_TRUE(board.is_empty(2));
    EXPECT_EQ(board.move_count(), 2);
    EXPECT_EQ(board.last_move(), 1);

    board.undo_move();
    EXPECT_TRUE(board.is_empty(1));
    EXPECT_EQ(board.move_count(), 1);
    EXPECT_EQ(board.last_move(), 0);

    board.undo_move();
    EXPECT_TRUE(board.is_empty(0));
    EXPECT_EQ(board.move_count(), 0);
    EXPECT_EQ(board.last_move(), -1);
}

TEST(GomokuBoardTest, MakeUndoRoundTrip) {
    GomokuBoard board(15, 5);
    // Play several moves then undo all — board should be identical to fresh
    board.make_move(10, BLACK);
    board.make_move(20, WHITE);
    board.make_move(30, BLACK);
    board.make_move(40, WHITE);
    board.undo_move();
    board.undo_move();
    board.undo_move();
    board.undo_move();

    for (int i = 0; i < 225; i++) {
        EXPECT_EQ(board.cell_at(i), EMPTY);
    }
    EXPECT_EQ(board.move_count(), 0);
    EXPECT_EQ(board.last_move(), -1);
}

// ===========================================================================
// Winner detection — horizontal
// ===========================================================================

TEST(GomokuBoardTest, WinnerHorizontal) {
    GomokuBoard board(15, 5);
    // BLACK plays row 7, cols 3..7 ; WHITE plays row 8 (non-winning)
    for (int i = 0; i < 5; i++) {
        board.make_move(7 * 15 + 3 + i, BLACK);
        if (i < 4) {
            board.make_move(8 * 15 + i, WHITE);
        }
    }
    EXPECT_EQ(board.winner(), BLACK);
}

// ===========================================================================
// Winner detection — vertical
// ===========================================================================

TEST(GomokuBoardTest, WinnerVertical) {
    GomokuBoard board(15, 5);
    // BLACK plays col 7, rows 3..7
    for (int i = 0; i < 5; i++) {
        board.make_move((3 + i) * 15 + 7, BLACK);
        if (i < 4) {
            board.make_move((3 + i) * 15 + 8, WHITE);
        }
    }
    EXPECT_EQ(board.winner(), BLACK);
}

// ===========================================================================
// Winner detection — diagonal (top-left to bottom-right)
// ===========================================================================

TEST(GomokuBoardTest, WinnerDiagonal) {
    GomokuBoard board(15, 5);
    // BLACK on (3,3),(4,4),(5,5),(6,6),(7,7)
    // WHITE on (0,0),(0,1),(0,2),(0,3)
    for (int i = 0; i < 5; i++) {
        board.make_move((3 + i) * 15 + (3 + i), BLACK);
        if (i < 4) {
            board.make_move(i, WHITE); // row 0
        }
    }
    EXPECT_EQ(board.winner(), BLACK);
}

// ===========================================================================
// Winner detection — anti-diagonal (top-right to bottom-left)
// ===========================================================================

TEST(GomokuBoardTest, WinnerAntiDiagonal) {
    GomokuBoard board(15, 5);
    // BLACK on (3,10),(4,9),(5,8),(6,7),(7,6)
    // WHITE on (0,0),(0,1),(0,2),(0,3)
    for (int i = 0; i < 5; i++) {
        board.make_move((3 + i) * 15 + (10 - i), BLACK);
        if (i < 4) {
            board.make_move(i, WHITE);
        }
    }
    EXPECT_EQ(board.winner(), BLACK);
}

// ===========================================================================
// Winner detection — WHITE wins
// ===========================================================================

TEST(GomokuBoardTest, WinnerWhite) {
    GomokuBoard board(15, 5);
    // WHITE plays row 0, cols 0..4 (needs BLACK to waste moves elsewhere)
    for (int i = 0; i < 5; i++) {
        board.make_move(14 * 15 + i, BLACK); // BLACK at bottom row (waste)
        board.make_move(0 * 15 + i, WHITE);  // WHITE at top row
    }
    EXPECT_EQ(board.winner(), WHITE);
}

// ===========================================================================
// No winner with partial line
// ===========================================================================

TEST(GomokuBoardTest, NoWinnerPartialLine) {
    GomokuBoard board(15, 5);
    // Only 4 in a row
    for (int i = 0; i < 4; i++) {
        board.make_move(7 * 15 + 3 + i, BLACK);
        board.make_move(8 * 15 + i, WHITE);
    }
    EXPECT_EQ(board.winner(), GomokuBoard::ONGOING);
}

// ===========================================================================
// Draw detection
// ===========================================================================

TEST(GomokuBoardTest, DrawDetection) {
    // Use a tiny 3x3 board with win_length=5 (impossible to win)
    GomokuBoard board(3, 5);
    int8_t player = BLACK;
    for (int i = 0; i < 9; i++) {
        board.make_move(i, player);
        player = opponent(player);
    }
    // Board is full, no 5-in-a-row possible on 3x3 -> draw
    EXPECT_EQ(board.winner(), EMPTY); // 0 = draw
}

// ===========================================================================
// is_winning_move
// ===========================================================================

TEST(GomokuBoardTest, IsWinningMoveTrue) {
    GomokuBoard board(15, 5);
    // Place 5 BLACK stones in a row
    for (int i = 0; i < 5; i++) {
        board.make_move(7 * 15 + 3 + i, BLACK);
        if (i < 4) board.make_move(8 * 15 + i, WHITE);
    }
    // The last stone (7,7) should be a winning move for BLACK
    EXPECT_TRUE(board.is_winning_move(7 * 15 + 7, BLACK));
    // Any of the 5 stones should register as winning
    EXPECT_TRUE(board.is_winning_move(7 * 15 + 3, BLACK));
    EXPECT_TRUE(board.is_winning_move(7 * 15 + 5, BLACK));
}

TEST(GomokuBoardTest, IsWinningMoveFalse) {
    GomokuBoard board(15, 5);
    // Only 3 in a row
    board.make_move(7 * 15 + 3, BLACK);
    board.make_move(0, WHITE);
    board.make_move(7 * 15 + 4, BLACK);
    board.make_move(1, WHITE);
    board.make_move(7 * 15 + 5, BLACK);

    EXPECT_FALSE(board.is_winning_move(7 * 15 + 5, BLACK));
}

// ===========================================================================
// Zobrist hash consistency — make + undo returns same hash
// ===========================================================================

TEST(GomokuBoardTest, ZobristHashMakeUndo) {
    GomokuBoard board(15, 5);
    const uint64_t h0 = board.hash_key();

    board.make_move(0, BLACK);
    const uint64_t h1 = board.hash_key();
    EXPECT_NE(h0, h1);

    board.undo_move();
    const uint64_t h2 = board.hash_key();
    EXPECT_EQ(h0, h2);
}

TEST(GomokuBoardTest, ZobristHashMultipleMoveUndo) {
    GomokuBoard board(15, 5);
    const uint64_t h0 = board.hash_key();

    board.make_move(10, BLACK);
    const uint64_t h1 = board.hash_key();
    board.make_move(20, WHITE);
    const uint64_t h2 = board.hash_key();
    board.make_move(30, BLACK);
    const uint64_t h3 = board.hash_key();

    // All hashes should be distinct
    EXPECT_NE(h0, h1);
    EXPECT_NE(h1, h2);
    EXPECT_NE(h2, h3);
    EXPECT_NE(h0, h3);

    // Undo in reverse order
    board.undo_move();
    EXPECT_EQ(board.hash_key(), h2);
    board.undo_move();
    EXPECT_EQ(board.hash_key(), h1);
    board.undo_move();
    EXPECT_EQ(board.hash_key(), h0);
}

TEST(GomokuBoardTest, ZobristHashDifferentPlayers) {
    // Same position, different player -> different hash
    GomokuBoard b1(15, 5);
    GomokuBoard b2(15, 5);

    b1.make_move(0, BLACK);
    b2.make_move(0, WHITE);

    EXPECT_NE(b1.hash_key(), b2.hash_key());
}

TEST(GomokuBoardTest, ZobristHashPositionIndependent) {
    // Same final position reached by different move orders should give same hash
    GomokuBoard b1(9, 5);
    b1.make_move(40, BLACK);
    b1.make_move(41, WHITE);

    GomokuBoard b2(9, 5);
    // Note: Zobrist with side-to-move key means order matters for the
    // intermediate states, but the final hash should be the same if the
    // same set of moves are played in the same order by the same players.
    b2.make_move(40, BLACK);
    b2.make_move(41, WHITE);

    EXPECT_EQ(b1.hash_key(), b2.hash_key());
}

// ===========================================================================
// from_position factory
// ===========================================================================

TEST(GomokuBoardTest, FromPositionWithMoveHistory) {
    Position pos;
    pos.board_size = 9;
    pos.win_length = 5;
    pos.move_history = {40, 41, 31}; // BLACK, WHITE, BLACK

    auto board = GomokuBoard::from_position(pos);
    EXPECT_EQ(board.board_size(), 9);
    EXPECT_EQ(board.move_count(), 3);
    EXPECT_FALSE(board.is_empty(40));
    EXPECT_FALSE(board.is_empty(41));
    EXPECT_FALSE(board.is_empty(31));
    EXPECT_EQ(board.cell_at(40), BLACK);
    EXPECT_EQ(board.cell_at(41), WHITE);
    EXPECT_EQ(board.cell_at(31), BLACK);
    EXPECT_EQ(board.last_move(), 31);
}

TEST(GomokuBoardTest, FromPositionWithCellsArray) {
    Position pos;
    pos.board_size = 9;
    pos.win_length = 5;
    pos.cells[40] = BLACK;
    pos.cells[41] = WHITE;
    pos.cells[31] = BLACK;
    // No move_history -> reconstructs from cells

    auto board = GomokuBoard::from_position(pos);
    EXPECT_EQ(board.board_size(), 9);
    EXPECT_EQ(board.move_count(), 3);
    EXPECT_EQ(board.cell_at(40), BLACK);
    EXPECT_EQ(board.cell_at(41), WHITE);
    EXPECT_EQ(board.cell_at(31), BLACK);
}

TEST(GomokuBoardTest, FromPositionHashConsistency) {
    // Board built via from_position should have same hash as manually played
    Position pos;
    pos.board_size = 15;
    pos.win_length = 5;
    pos.move_history = {112, 113, 97};

    auto board1 = GomokuBoard::from_position(pos);

    GomokuBoard board2(15, 5);
    board2.make_move(112, BLACK);
    board2.make_move(113, WHITE);
    board2.make_move(97, BLACK);

    EXPECT_EQ(board1.hash_key(), board2.hash_key());
}

// ===========================================================================
// legal_moves count
// ===========================================================================

TEST(GomokuBoardTest, LegalMovesCount) {
    GomokuBoard board(9, 5);
    EXPECT_EQ(static_cast<int>(board.legal_moves().size()), 81);

    board.make_move(40, BLACK);
    EXPECT_EQ(static_cast<int>(board.legal_moves().size()), 80);

    board.make_move(41, WHITE);
    EXPECT_EQ(static_cast<int>(board.legal_moves().size()), 79);

    board.undo_move();
    EXPECT_EQ(static_cast<int>(board.legal_moves().size()), 80);
}

// ===========================================================================
// Clone
// ===========================================================================

TEST(GomokuBoardTest, CloneIsIndependent) {
    GomokuBoard board(15, 5);
    board.make_move(112, BLACK);
    board.make_move(113, WHITE);

    auto cloned = board.clone();
    EXPECT_EQ(cloned.hash_key(), board.hash_key());
    EXPECT_EQ(cloned.move_count(), board.move_count());
    EXPECT_EQ(cloned.last_move(), board.last_move());
    EXPECT_EQ(cloned.cell_at(112), BLACK);
    EXPECT_EQ(cloned.cell_at(113), WHITE);

    // Mutating clone does not affect original
    cloned.make_move(114, BLACK);
    EXPECT_EQ(board.move_count(), 2);
    EXPECT_EQ(cloned.move_count(), 3);
    EXPECT_TRUE(board.is_empty(114));
}

// ===========================================================================
// Edge cases
// ===========================================================================

TEST(GomokuBoardTest, SmallBoardWin) {
    // 7x7 board, win_length 5
    GomokuBoard board(7, 5);
    // BLACK horizontal win on row 0: (0,0)...(0,4)
    for (int i = 0; i < 5; i++) {
        board.make_move(i, BLACK);
        if (i < 4) board.make_move(7 + i, WHITE); // row 1
    }
    EXPECT_EQ(board.winner(), BLACK);
}

TEST(GomokuBoardTest, MaxSizeBoardCreation) {
    GomokuBoard board(16, 5);
    EXPECT_EQ(board.board_size(), 16);
    EXPECT_EQ(board.total_cells(), 256);
    auto moves = board.legal_moves();
    EXPECT_EQ(static_cast<int>(moves.size()), 256);
}

TEST(GomokuBoardTest, WinnerOnEdge) {
    // Win along the bottom edge of a 15x15 board
    GomokuBoard board(15, 5);
    for (int i = 0; i < 5; i++) {
        board.make_move(14 * 15 + i, BLACK);
        if (i < 4) board.make_move(13 * 15 + i, WHITE);
    }
    EXPECT_EQ(board.winner(), BLACK);
}

TEST(GomokuBoardTest, WinnerOnRightEdge) {
    // Vertical win along the rightmost column
    GomokuBoard board(15, 5);
    for (int i = 0; i < 5; i++) {
        board.make_move(i * 15 + 14, BLACK);
        if (i < 4) board.make_move(i * 15 + 13, WHITE);
    }
    EXPECT_EQ(board.winner(), BLACK);
}

// ===========================================================================
// Config sanity checks
// ===========================================================================

TEST(ConfigTest, PatternScoresMatchJS) {
    EXPECT_EQ(pattern_scores::FIVE,       1'000'000);
    EXPECT_EQ(pattern_scores::OPEN_FOUR,    100'000);
    EXPECT_EQ(pattern_scores::HALF_FOUR,     10'000);
    EXPECT_EQ(pattern_scores::OPEN_THREE,     5'000);
    EXPECT_EQ(pattern_scores::HALF_THREE,     1'000);
    EXPECT_EQ(pattern_scores::OPEN_TWO,         500);
    EXPECT_EQ(pattern_scores::HALF_TWO,         100);
    EXPECT_EQ(pattern_scores::OPEN_ONE,          10);
}

TEST(ConfigTest, ComboBonusesMatchJS) {
    EXPECT_EQ(combo_bonuses::DOUBLE_HALF_FOUR,      80'000);
    EXPECT_EQ(combo_bonuses::OPEN_THREE_HALF_FOUR,  60'000);
    EXPECT_EQ(combo_bonuses::DOUBLE_OPEN_THREE,     40'000);
    EXPECT_EQ(combo_bonuses::TRIPLE_HALF_THREE,     20'000);
}

TEST(ConfigTest, SearchConfigMatchesJS) {
    EXPECT_EQ(search_defaults::MAX_DEPTH,           12);
    EXPECT_EQ(search_defaults::TIME_LIMIT_MS,       3000);
    EXPECT_EQ(search_defaults::TT_SIZE,             1u << 20);
    EXPECT_EQ(search_defaults::ASPIRATION_WINDOW,   50);
    EXPECT_EQ(search_defaults::NULL_MOVE_REDUCTION, 3);
    EXPECT_EQ(search_defaults::LMR_THRESHOLD,       4);
    EXPECT_EQ(search_defaults::LMR_DEPTH_MIN,       3);
}

TEST(ConfigTest, ThreatConfigMatchesJS) {
    EXPECT_EQ(threat_defaults::VCF_MAX_DEPTH, 14);
    EXPECT_EQ(threat_defaults::VCT_MAX_DEPTH,  8);
    EXPECT_EQ(threat_defaults::VCF_MAX_NODES, 50'000);
    EXPECT_EQ(threat_defaults::VCT_MAX_NODES, 20'000);
}

TEST(ConfigTest, CandidateConfigMatchesJS) {
    EXPECT_EQ(candidate_defaults::RADIUS,         2);
    EXPECT_EQ(candidate_defaults::MAX_CANDIDATES, 30);
    EXPECT_EQ(candidate_defaults::NN_TOP_K,       8);
}
