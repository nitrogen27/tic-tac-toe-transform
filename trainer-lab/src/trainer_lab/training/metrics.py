"""Training metrics: policy accuracy and value MAE."""

from __future__ import annotations

import torch


def policy_accuracy(
    policy_logits: torch.Tensor,
    policy_target: torch.Tensor,
) -> float:
    """Top-1 accuracy: fraction of samples where argmax(logits) == argmax(target).

    Parameters
    ----------
    policy_logits : [B, 256]
    policy_target : [B, 256]
    """
    pred = policy_logits.argmax(dim=1)
    label = policy_target.argmax(dim=1)
    return (pred == label).float().mean().item()


def value_mae(
    value_pred: torch.Tensor,
    value_target: torch.Tensor,
) -> float:
    """Mean absolute error for value predictions.

    Parameters
    ----------
    value_pred   : [B, 1] or [B]
    value_target : [B, 1] or [B]
    """
    return (value_pred.view(-1) - value_target.view(-1)).abs().mean().item()
