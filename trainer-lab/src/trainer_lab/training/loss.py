"""Combined policy + value loss for the dual-head network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GomokuLoss(nn.Module):
    """Combined loss: masked cross-entropy for policy + MSE for value.

    Parameters
    ----------
    weight_value : float
        Relative weight of the value loss term.  Total loss is::

            L = policy_loss + weight_value * value_loss
    """

    def __init__(self, weight_value: float = 0.5) -> None:
        super().__init__()
        self.weight_value = weight_value

    def forward(
        self,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        policy_target: torch.Tensor,
        value_target: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss components.

        Parameters
        ----------
        policy_logits : [B, 256]
        value_pred    : [B, 1]
        policy_target : [B, 256]  — probability distribution
        value_target  : [B, 1]    — scalar in [-1, 1]
        legal_mask    : [B, 256]  — optional; 1 = legal, 0 = illegal

        Returns
        -------
        total_loss, policy_loss, value_loss
        """
        # Mask illegal moves with large negative logits
        if legal_mask is not None:
            policy_logits = policy_logits + (1.0 - legal_mask) * (-1e8)

        # Log-softmax + target distribution -> cross-entropy
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -torch.sum(policy_target * log_probs, dim=1).mean()

        # Value MSE
        value_loss = F.mse_loss(value_pred.view(-1), value_target.view(-1))

        total_loss = policy_loss + self.weight_value * value_loss
        return total_loss, policy_loss, value_loss
