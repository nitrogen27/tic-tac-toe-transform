"""Tests for board_to_tensor encoder."""

import torch
from trainer_lab.data.encoder import board_to_tensor


def _make_position(board_size: int = 9) -> dict:
    """Create a small test position."""
    board = [[0] * board_size for _ in range(board_size)]
    board[4][4] = 1  # player 1 stone at center
    board[3][4] = 2  # player 2 stone above center
    return {
        "board_size": board_size,
        "board": board,
        "current_player": 1,
        "last_move": [3, 4],
    }


def test_output_shape():
    pos = _make_position(9)
    tensor = board_to_tensor(pos)
    assert tensor.shape == (6, 16, 16), f"Expected [6,16,16], got {tensor.shape}"


def test_output_dtype():
    pos = _make_position(15)
    tensor = board_to_tensor(pos)
    assert tensor.dtype == torch.float32


def test_my_stones_plane():
    pos = _make_position(9)
    tensor = board_to_tensor(pos)
    # current_player=1, stone at (4,4)
    assert tensor[0, 4, 4] == 1.0
    # opponent stone at (3,4) should NOT be in plane 0
    assert tensor[0, 3, 4] == 0.0


def test_opponent_stones_plane():
    pos = _make_position(9)
    tensor = board_to_tensor(pos)
    # opponent (player 2) stone at (3,4)
    assert tensor[1, 3, 4] == 1.0
    assert tensor[1, 4, 4] == 0.0


def test_legal_mask_plane():
    pos = _make_position(9)
    tensor = board_to_tensor(pos)
    # occupied cells should be 0
    assert tensor[2, 4, 4] == 0.0
    assert tensor[2, 3, 4] == 0.0
    # empty cell inside board should be 1
    assert tensor[2, 0, 0] == 1.0
    # outside NxN area (padding) should be 0
    assert tensor[2, 9, 0] == 0.0


def test_last_move_plane():
    pos = _make_position(9)
    tensor = board_to_tensor(pos)
    assert tensor[3, 3, 4] == 1.0
    assert tensor[3, 0, 0] == 0.0


def test_board_size_mask():
    pos = _make_position(9)
    tensor = board_to_tensor(pos)
    # inside 9x9 -> 1
    assert tensor[5, 0, 0] == 1.0
    assert tensor[5, 8, 8] == 1.0
    # outside 9x9 -> 0
    assert tensor[5, 9, 0] == 0.0
    assert tensor[5, 0, 9] == 0.0


def test_padding_is_zero():
    """All planes should be zero in the padded region."""
    pos = _make_position(9)
    tensor = board_to_tensor(pos)
    # row 10, any col — all planes zero
    assert tensor[:, 10, 5].sum().item() == 0.0
