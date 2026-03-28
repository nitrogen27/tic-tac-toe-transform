"""Learning-rate scheduler: cosine annealing with linear warmup."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Create a LambdaLR that does linear warmup then cosine decay.

    Parameters
    ----------
    optimizer    : the optimizer to schedule
    warmup_steps : steps of linear warmup from 0 to base LR
    total_steps  : total training steps (warmup + cosine)
    min_lr_ratio : minimum LR as a fraction of base LR (default 0)
    """

    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(warmup_steps, 1)
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)
