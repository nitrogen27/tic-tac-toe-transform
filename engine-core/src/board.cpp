#include "gomoku/board.hpp"
#include "gomoku/zobrist.hpp"

namespace gomoku {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
GomokuBoard::GomokuBoard(int board_size, int win_length)
    : N_(board_size)
    , size_(board_size * board_size)
    , win_len_(win_length)
{
    cells_.fill(EMPTY);
}

// ---------------------------------------------------------------------------
// make_move  —  exact port of JS GomokuBoard.makeMove(pos, player)
// ---------------------------------------------------------------------------
void GomokuBoard::make_move(int pos, Cell player) {
    // Save undo state
    history_.push_back(UndoEntry{pos, player, hash_, last_move_});

    cells_[pos] = player;

    // Player-index: BLACK(+1)->0, WHITE(-1)->1
    const int pIdx = (player == BLACK) ? 0 : 1;
    hash_ ^= global_zobrist().piece_hash(pos, pIdx);
    hash_ ^= global_zobrist().side_hash();

    move_count_++;
    last_move_ = pos;
}

// ---------------------------------------------------------------------------
// undo_move  —  exact port of JS GomokuBoard.undoMove()
// ---------------------------------------------------------------------------
void GomokuBoard::undo_move() {
    const UndoEntry& entry = history_.back();
    cells_[entry.pos] = EMPTY;
    hash_      = entry.prev_hash;
    last_move_ = entry.prev_last_move;
    move_count_--;
    history_.pop_back();
}

// ---------------------------------------------------------------------------
// winner  —  full board scan, exact port of JS GomokuBoard.winner()
//
// Returns BLACK, WHITE, EMPTY (draw), or ONGOING.
// The JS version returns +1, -1, 0 (draw), null (ongoing).
// We use the sentinel ONGOING = 2 instead of null.
// ---------------------------------------------------------------------------
Cell GomokuBoard::winner() const {
    // Direction vectors: right, down, down-right, down-left
    static constexpr int DR[] = {1, 0, 1, 1};
    static constexpr int DC[] = {0, 1, 1, -1};

    for (int r = 0; r < N_; r++) {
        for (int c = 0; c < N_; c++) {
            const Cell who = cells_[r * N_ + c];
            if (who == EMPTY) continue;

            for (int d = 0; d < 4; d++) {
                int k  = 1;
                int rr = r + DR[d];
                int cc = c + DC[d];
                while (rr >= 0 && cc >= 0 && rr < N_ && cc < N_ &&
                       cells_[rr * N_ + cc] == who) {
                    k++;
                    if (k >= win_len_) return who;
                    rr += DR[d];
                    cc += DC[d];
                }
            }
        }
    }

    // Draw check — any empty cell means still ongoing
    for (int i = 0; i < size_; i++) {
        if (cells_[i] == EMPTY) return ONGOING;
    }
    return EMPTY; // draw (0)
}

// ---------------------------------------------------------------------------
// is_winning_move  —  exact port of JS GomokuBoard.isWinningMove(pos, player)
//
// Counts consecutive stones of `player` through `pos` in all 4 directions.
// The cell at `pos` must already contain `player`.
// ---------------------------------------------------------------------------
bool GomokuBoard::is_winning_move(int pos, Cell player) const {
    static constexpr int DR[] = {1, 0, 1, 1};
    static constexpr int DC[] = {0, 1, 1, -1};

    const int r0 = pos / N_;
    const int c0 = pos % N_;

    for (int d = 0; d < 4; d++) {
        int count = 1;

        // Forward
        int rr = r0 + DR[d], cc = c0 + DC[d];
        while (rr >= 0 && cc >= 0 && rr < N_ && cc < N_ &&
               cells_[rr * N_ + cc] == player) {
            count++;
            rr += DR[d];
            cc += DC[d];
        }

        // Backward
        rr = r0 - DR[d];
        cc = c0 - DC[d];
        while (rr >= 0 && cc >= 0 && rr < N_ && cc < N_ &&
               cells_[rr * N_ + cc] == player) {
            count++;
            rr -= DR[d];
            cc -= DC[d];
        }

        if (count >= win_len_) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// legal_moves  —  all empty cells
// ---------------------------------------------------------------------------
std::vector<int> GomokuBoard::legal_moves() const {
    std::vector<int> moves;
    moves.reserve(size_);
    for (int i = 0; i < size_; i++) {
        if (cells_[i] == EMPTY) moves.push_back(i);
    }
    return moves;
}

bool GomokuBoard::is_empty(int pos) const {
    return pos >= 0 && pos < size_ && cells_[pos] == EMPTY;
}

uint64_t GomokuBoard::hash_key() const {
    return hash_;
}

// ---------------------------------------------------------------------------
// clone  —  deep copy WITHOUT history (matching JS behaviour)
// ---------------------------------------------------------------------------
GomokuBoard GomokuBoard::clone() const {
    GomokuBoard b(N_, win_len_);
    b.cells_      = cells_;
    b.hash_       = hash_;
    b.move_count_ = move_count_;
    b.last_move_  = last_move_;
    // history NOT cloned — clone is for read-only branches (same as JS)
    return b;
}

// ---------------------------------------------------------------------------
// from_position  —  factory matching JS GomokuBoard.fromArray()
//
// Rebuilds the Zobrist hash from scratch by scanning the cells array.
// If the position has a move_history, replays moves instead for full fidelity.
// ---------------------------------------------------------------------------
GomokuBoard GomokuBoard::from_position(const Position& pos) {
    GomokuBoard board(pos.board_size, pos.win_length);

    if (!pos.move_history.empty()) {
        // Replay move history — alternating BLACK / WHITE
        Cell player = BLACK;
        for (int mv : pos.move_history) {
            board.make_move(mv, player);
            player = opponent(player);
        }
        return board;
    }

    // No move history — reconstruct from cells array (like JS fromArray)
    board.hash_ = 0;
    int count = 0;
    int last  = -1;

    const int total = pos.board_size * pos.board_size;
    for (int i = 0; i < total; i++) {
        if (pos.cells[i] != EMPTY) {
            board.cells_[i] = pos.cells[i];
            const int pIdx = (pos.cells[i] == BLACK) ? 0 : 1;
            board.hash_ ^= global_zobrist().piece_hash(i, pIdx);
            count++;
            last = i; // approximate last move (rightmost non-empty)
        }
    }

    board.move_count_ = count;
    board.last_move_  = last;

    // Side-to-move hash: if odd number of moves, XOR in side key
    if (count % 2 == 1) {
        board.hash_ ^= global_zobrist().side_hash();
    }

    return board;
}

} // namespace gomoku
