"""PolicyValueResNet — dual-head residual network for Gomoku."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer_lab.models.blocks import ResBlock


class PolicyValueResNet(nn.Module):
    """ResNet with separate policy and value heads.

    Input shape : ``[B, 6, 16, 16]``
    Policy output: ``[B, 256]``  (logits over 16*16 = 256 moves)
    Value output : ``[B, 1]``   (tanh, range [-1, 1])
    """

    def __init__(
        self,
        in_channels: int = 6,
        res_filters: int = 128,
        res_blocks: int = 8,
        policy_filters: int = 2,
        value_fc: int = 128,
        board_max: int = 16,
    ) -> None:
        super().__init__()
        self.board_max = board_max
        board_cells = board_max * board_max  # 256

        # --- shared tower ---------------------------------------------------
        self.input_conv = nn.Conv2d(in_channels, res_filters, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(res_filters)
        self.tower = nn.Sequential(*[ResBlock(res_filters) for _ in range(res_blocks)])

        # --- policy head -----------------------------------------------------
        self.policy_conv = nn.Conv2d(res_filters, policy_filters, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_filters)
        self.policy_fc = nn.Linear(policy_filters * board_cells, board_cells)

        # --- value head ------------------------------------------------------
        self.value_conv = nn.Conv2d(res_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_cells, value_fc)
        self.value_fc2 = nn.Linear(value_fc, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # shared tower
        s = F.relu(self.input_bn(self.input_conv(x)))
        s = self.tower(s)

        # policy head
        p = F.relu(self.policy_bn(self.policy_conv(s)))
        p = p.reshape(p.size(0), -1)
        p = self.policy_fc(p)  # [B, 256] raw logits

        # value head
        v = F.relu(self.value_bn(self.value_conv(s)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # [B, 1]

        return p, v
