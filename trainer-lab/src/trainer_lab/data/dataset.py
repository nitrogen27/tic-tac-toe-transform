"""PyTorch Dataset that loads positions from JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from trainer_lab.data.encoder import board_to_tensor
from trainer_lab.data.policy import pad_policy_target
from trainer_lab.data.augmentation import augment_sample


class PositionDataset(Dataset):
    """Load training positions from one or more JSON files.

    Each JSON file should contain a list of position dicts with keys:

    * ``board_size``, ``board``, ``current_player``, ``last_move``
    * ``policy`` — list of 256 floats (flat target distribution)
    * ``value``  — float in [-1, 1]

    When *augment=True* every position is expanded to its 8 D4 symmetries,
    increasing the effective dataset size by 8x.
    """

    def __init__(
        self,
        data_dir: str | Path,
        augment: bool = False,
        glob_pattern: str = "*.json",
    ) -> None:
        super().__init__()
        self.augment = augment
        self.samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        data_path = Path(data_dir)
        if not data_path.exists():
            return  # allow empty dataset for testing

        for fp in sorted(data_path.glob(glob_pattern)):
            with open(fp, "r", encoding="utf-8") as f:
                records = json.load(f)
            for rec in records:
                planes = board_to_tensor(rec)
                policy = pad_policy_target(rec["policy"], rec["board_size"])
                value = torch.tensor(rec["value"], dtype=torch.float32)

                if self.augment:
                    self.samples.extend(augment_sample(planes, policy, value))
                else:
                    self.samples.append((planes, policy, value))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[idx]
