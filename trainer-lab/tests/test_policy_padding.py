from __future__ import annotations

import torch

from trainer_lab.data.policy import pad_policy_target
from trainer_lab.training.metrics import policy_accuracy


def test_pad_policy_target_expands_board_to_16x16() -> None:
    policy = [0.0] * 49
    policy[8] = 1.0

    padded = pad_policy_target(policy, board_size=7, board_max=16)

    assert padded.shape == (256,)
    assert padded.sum().item() == 1.0
    assert padded[1 * 16 + 1].item() == 1.0


def test_policy_accuracy_respects_legal_mask() -> None:
    logits = torch.tensor([[10.0, 1.0, 2.0]])
    target = torch.tensor([[0.0, 0.0, 1.0]])
    legal_mask = torch.tensor([[0.0, 1.0, 1.0]])

    acc = policy_accuracy(logits, target, legal_mask=legal_mask)

    assert acc == 1.0
