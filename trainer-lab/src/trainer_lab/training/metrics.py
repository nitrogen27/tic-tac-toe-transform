"""Training metrics: policy accuracy, value MAE, and strength diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def policy_accuracy(
    policy_logits: torch.Tensor,
    policy_target: torch.Tensor,
    legal_mask: torch.Tensor | None = None,
) -> float:
    """Top-1 accuracy: fraction of samples where argmax(logits) == argmax(target).

    Also referred to as policyTop1Acc.  This measures agreement with the
    training target, **not** playing strength.

    Parameters
    ----------
    policy_logits : [B, 256]
    policy_target : [B, 256]
    """
    if legal_mask is not None:
        policy_logits = policy_logits + (1.0 - legal_mask) * (-1e8)
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


# ---------------------------------------------------------------------------
# Extended metrics for training quality diagnostics
# ---------------------------------------------------------------------------


def teacher_mass_on_pred(
    policy_logits: torch.Tensor,
    policy_target: torch.Tensor,
    legal_mask: torch.Tensor | None = None,
) -> float:
    """How much probability mass the teacher assigns to the model's top choice.

    High value = teacher agrees with model's best move (even if argmax differs).
    Low value  = model picks moves the teacher considers bad.
    """
    if legal_mask is not None:
        policy_logits = policy_logits + (1.0 - legal_mask) * (-1e8)
    pred = policy_logits.argmax(dim=1)  # [B]
    mass = policy_target.gather(1, pred.unsqueeze(1)).squeeze(1)  # [B]
    return mass.mean().item()


def value_sign_agreement(
    value_pred: torch.Tensor,
    value_target: torch.Tensor,
) -> float:
    """Fraction of positions where sign(pred) == sign(target).

    Zeros in target are always counted as agreement.
    """
    pred_sign = value_pred.view(-1).sign()
    target_sign = value_target.view(-1).sign()
    agree = ((pred_sign == target_sign) | (target_sign == 0)).float()
    return agree.mean().item()


def policy_entropy(
    policy_logits: torch.Tensor,
    legal_mask: torch.Tensor | None = None,
) -> float:
    """Mean entropy of the policy distribution (nats).

    High = uncertain / exploratory.  Low = confident / peaked.
    """
    if legal_mask is not None:
        policy_logits = policy_logits + (1.0 - legal_mask) * (-1e8)
    probs = F.softmax(policy_logits, dim=1)
    log_probs = F.log_softmax(policy_logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1)
    return entropy.mean().item()


def policy_kl_divergence(
    policy_logits: torch.Tensor,
    policy_target: torch.Tensor,
    legal_mask: torch.Tensor | None = None,
) -> float:
    """KL(target || model) — how much information the model loses vs teacher.

    Lower = model distribution is closer to teacher.
    More informative than top-1 accuracy for soft targets.
    """
    if legal_mask is not None:
        policy_logits = policy_logits + (1.0 - legal_mask) * (-1e8)
    log_probs = F.log_softmax(policy_logits, dim=1)
    target_log = (policy_target + 1e-8).log()
    kl = (policy_target * (target_log - log_probs)).sum(dim=1)
    return kl.mean().item()
