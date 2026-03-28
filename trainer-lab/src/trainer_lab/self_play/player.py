"""Self-play player that uses the neural network + MCTS to generate games."""

from __future__ import annotations

from typing import Optional

import torch

from trainer_lab.config import SelfPlayConfig
from trainer_lab.models.resnet import PolicyValueResNet


class SelfPlayPlayer:
    """Wrapper around a PolicyValueResNet for self-play game generation.

    This is a Phase-6 stub. The full implementation will integrate MCTS
    with the neural network to produce training positions.
    """

    def __init__(
        self,
        model: PolicyValueResNet,
        config: SelfPlayConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config or SelfPlayConfig()
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def play_game(self, board_size: int = 15) -> list[dict]:
        """Play a single self-play game and return a list of position records.

        Each record is a dict suitable for the training pipeline:
        ``{board_size, board, current_player, last_move, policy, value}``.

        Returns
        -------
        list[dict] — empty in this stub; will be populated in Phase 6.
        """
        # TODO: implement MCTS-guided self-play
        return []

    def generate_games(self, num_games: int | None = None) -> list[dict]:
        """Generate multiple self-play games and collect all position records.

        Parameters
        ----------
        num_games : override ``self.config.games`` if provided.
        """
        n = num_games if num_games is not None else self.config.games
        all_positions: list[dict] = []
        for _ in range(n):
            all_positions.extend(self.play_game())
        return all_positions
