"""D4 symmetry augmentation for 16x16 board planes and policy vectors."""

from __future__ import annotations

from typing import Sequence

import torch


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
    grid = policy.view(1, 16, 16)
    grid = transform_fn(grid)
    return grid.reshape(256)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def augment_sample(
    planes: torch.Tensor,
    policy: torch.Tensor,
    value: float | torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return all 8 D4 symmetry variants of a single sample.

    Parameters
    ----------
    planes : Tensor [6, 16, 16]
    policy : Tensor [256]   (flat probability / target vector)
    value  : float or scalar Tensor

    Returns
    -------
    list of (planes, policy, value) tuples — length 8.
    """
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=torch.float32)

    results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for fn in _TRANSFORMS:
        aug_planes = fn(planes.clone())
        aug_policy = _transform_policy(policy.clone(), fn)
        results.append((aug_planes, aug_policy, value.clone()))
    return results
