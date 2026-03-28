"""Encode a board position dict into a tensor suitable for the ResNet."""

from __future__ import annotations

import torch


def board_to_tensor(position: dict) -> torch.Tensor:
    """Convert a position dictionary to a ``[6, 16, 16]`` float tensor.

    Expected *position* keys:

    * ``board_size`` — int, the side length N of the NxN board
    * ``board`` — list[list[int]], NxN grid where 0=empty, 1=player-1, 2=player-2
    * ``current_player`` — int, 1 or 2
    * ``last_move`` — tuple/list ``(row, col)`` or ``None``

    The six planes are:

    0. **my_stones** — 1 where current player has a stone
    1. **opp_stones** — 1 where opponent has a stone
    2. **legal_mask** — 1 where the cell is empty *and* inside the real board
    3. **last_move** — 1 at the last-move coordinate (all-zero if first move)
    4. **side_to_move** — constant plane: 1 if current_player == 1, else 0
    5. **board_size_mask** — 1 for every cell inside the real NxN area
    """

    board_size: int = position["board_size"]
    board: list[list[int]] = position["board"]
    current_player: int = position["current_player"]
    opponent: int = 3 - current_player
    last_move = position.get("last_move")

    planes = torch.zeros(6, 16, 16, dtype=torch.float32)

    for r in range(board_size):
        for c in range(board_size):
            cell = board[r][c]
            if cell == current_player:
                planes[0, r, c] = 1.0
            elif cell == opponent:
                planes[1, r, c] = 1.0

            # legal = empty and within board
            if cell == 0:
                planes[2, r, c] = 1.0

            # board-size mask
            planes[5, r, c] = 1.0

    # last-move plane
    if last_move is not None:
        lr, lc = last_move
        if 0 <= lr < board_size and 0 <= lc < board_size:
            planes[3, lr, lc] = 1.0

    # side-to-move plane (constant fill)
    if current_player == 1:
        planes[4, :board_size, :board_size] = 1.0

    return planes
