"""Helpers for converting variable-size policy targets to the padded 16x16 format."""

from __future__ import annotations

from typing import Sequence

import torch


def pad_policy_target(
    policy: Sequence[float] | torch.Tensor,
    board_size: int,
    board_max: int = 16,
) -> torch.Tensor:
    """Pad a flat ``board_size x board_size`` policy into a flat ``board_max x board_max`` tensor.

    If *policy* is already of length ``board_max * board_max`` it is returned unchanged
    as a float tensor. This makes the helper safe to use across both fixed-size and
    variable-size datasets.
    """
    target_cells = board_max * board_max
    if isinstance(policy, torch.Tensor):
        tensor = policy.detach().clone().to(dtype=torch.float32).view(-1)
    else:
        tensor = torch.tensor(list(policy), dtype=torch.float32).view(-1)

    if tensor.numel() == target_cells:
        return tensor

    expected_cells = board_size * board_size
    if tensor.numel() != expected_cells:
        msg = (
            f"Expected policy of length {expected_cells} for board_size={board_size} "
            f"or {target_cells} for padded format, got {tensor.numel()}"
        )
        raise ValueError(msg)

    padded = torch.zeros(target_cells, dtype=torch.float32)
    for idx, value in enumerate(tensor.tolist()):
        row, col = divmod(idx, board_size)
        padded[row * board_max + col] = value
    return padded
