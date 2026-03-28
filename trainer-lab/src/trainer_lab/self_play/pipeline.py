"""End-to-end self-play training pipeline."""

from __future__ import annotations

import logging
from typing import Optional

import torch

from trainer_lab.config import ModelConfig, TrainConfig, SelfPlayConfig
from trainer_lab.models.resnet import PolicyValueResNet
from trainer_lab.self_play.player import SelfPlayPlayer
from trainer_lab.self_play.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class SelfPlayPipeline:
    """Orchestrates the generate-train loop.

    1. Use the current model to play self-play games.
    2. Collect positions into the replay buffer.
    3. Train the model on sampled positions.
    4. Repeat for *generations* iterations.

    Phase-6 stub — the skeleton is in place; MCTS integration is deferred.
    """

    def __init__(
        self,
        model_cfg: ModelConfig | None = None,
        train_cfg: TrainConfig | None = None,
        selfplay_cfg: SelfPlayConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model_cfg = model_cfg or ModelConfig()
        self.train_cfg = train_cfg or TrainConfig()
        self.selfplay_cfg = selfplay_cfg or SelfPlayConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = PolicyValueResNet(
            in_channels=self.model_cfg.in_channels,
            res_filters=self.model_cfg.res_filters,
            res_blocks=self.model_cfg.res_blocks,
            policy_filters=self.model_cfg.policy_filters,
            value_fc=self.model_cfg.value_fc,
            board_max=self.model_cfg.board_max,
        ).to(self.device)

        self.player = SelfPlayPlayer(self.model, self.selfplay_cfg, self.device)
        self.buffer = ReplayBuffer(self.selfplay_cfg)

    def run(self, generations: int = 10) -> None:
        """Execute the self-play loop for *generations* iterations."""
        for gen in range(1, generations + 1):
            logger.info("Generation %d: generating self-play games...", gen)
            positions = self.player.generate_games()
            self.buffer.add_many(positions)
            logger.info(
                "Generation %d: buffer size = %d (new positions = %d)",
                gen, len(self.buffer), len(positions),
            )

            # TODO: train on sampled buffer positions
            logger.info("Generation %d: training step (stub)", gen)
