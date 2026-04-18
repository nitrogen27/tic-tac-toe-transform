"""D4 symmetry augmentation for 16x16 board planes and policy vectors."""

from __future__ import annotations

from typing import Sequence

import torch

from trainer_lab.specs import PADDED_BOARD_SIZE, PADDED_POLICY_SIZE


# ---------------------------------------------------------------------------
# Low-level transforms on [C, 16, 16] planes and [256] policy vectors
# ---------------------------------------------------------------------------

def _identity(planes: torch.Tensor) -> torch.Tensor:
    return planes


def _rot90(planes: torch.Tensor) -> torch.Tensor:
    return torch.rot90(planes, k=1, dims=(1, 2))


def _rot180(planes: torch.Tensor) -> torch.Tensor:
    return torch.rot90(planes, k=2, dims=(1, 2))


def _rot270(planes: torch.Tensor) -> torch.Tensor:
    return torch.rot90(planes, k=3, dims=(1, 2))


def _mirror_v(planes: torch.Tensor) -> torch.Tensor:
    """Flip vertically (top <-> bottom)."""
    return planes.flip(1)


def _mirror_h(planes: torch.Tensor) -> torch.Tensor:
    """Flip horizontally (left <-> right)."""
    return planes.flip(2)


def _diag_main(planes: torch.Tensor) -> torch.Tensor:
    """Transpose across the main diagonal."""
    return planes.transpose(1, 2)


def _diag_anti(planes: torch.Tensor) -> torch.Tensor:
    """Transpose across the anti-diagonal (= rot90 + flip vertical)."""
    return torch.rot90(planes, k=1, dims=(1, 2)).flip(1)


_TRANSFORMS = [
    _identity,
    _rot90,
    _rot180,
    _rot270,
    _mirror_v,
    _mirror_h,
    _diag_main,
    _diag_anti,
]


def _transform_policy(policy: torch.Tensor, transform_fn) -> torch.Tensor:
    """Apply the same spatial transform to a flat [256] policy vector.

    The policy is reshaped to [1, 16, 16], transformed, then flattened back.
    """
    grid = policy.view(1, PADDED_BOARD_SIZE, PADDED_BOARD_SIZE)
    grid = transform_fn(grid)
    return grid.reshape(PADDED_POLICY_SIZE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def augment_sample(
    planes: torch.Tensor,
    policy: torch.Tensor,
    value: float | torch.Tensor,
    board_size: int = 16,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return all 8 D4 symmetry variants of a single sample.

    Parameters
    ----------
    planes : Tensor [6, 16, 16]
    policy : Tensor [256]   (flat probability / target vector)
    value  : float or scalar Tensor
    board_size : int
        Actual board size.  For boards smaller than 16, rotations are applied
        only to the top-left board_size×board_size subgrid so that policy mass
        never leaks into the padded region.

    Returns
    -------
    list of (planes, policy, value) tuples — length 8.
    """
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=torch.float32)

    results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for fn in _TRANSFORMS:
        if board_size >= 16:
            aug_planes = fn(planes.clone())
            aug_policy = _transform_policy(policy.clone(), fn)
        else:
            bs = board_size
            aug_planes = torch.zeros_like(planes)
            aug_planes[:, :bs, :bs] = fn(planes[:, :bs, :bs].clone())
            pol_grid = policy.view(1, PADDED_BOARD_SIZE, PADDED_BOARD_SIZE)
            aug_pol_grid = torch.zeros_like(pol_grid)
            aug_pol_grid[:, :bs, :bs] = fn(pol_grid[:, :bs, :bs].clone())
            aug_policy = aug_pol_grid.reshape(PADDED_POLICY_SIZE)
        results.append((aug_planes, aug_policy, value.clone()))
    return results
