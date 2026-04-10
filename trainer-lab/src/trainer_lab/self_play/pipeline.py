"""End-to-end self-play training pipeline: generate → train → evaluate → export."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from trainer_lab.config import ModelConfig, TrainConfig, SelfPlayConfig
from trainer_lab.data.encoder import board_to_tensor
from trainer_lab.data.policy import pad_policy_target
from trainer_lab.evaluation.eval_script import evaluate_vs_random
from trainer_lab.export.onnx_export import export_to_onnx
from trainer_lab.models.resnet import PolicyValueResNet
from trainer_lab.self_play.player import SelfPlayPlayer
from trainer_lab.self_play.mixed_replay import MixedReplay
from trainer_lab.training.loss import GomokuLoss
from trainer_lab.training.trainer import train_epoch

logger = logging.getLogger(__name__)


class SelfPlayPipeline:
    """Orchestrates the generate → train → evaluate → export loop.

    1. Use the current model to play self-play games via MCTS.
    2. Collect positions into the replay buffer.
    3. Train the model on sampled positions for N epochs.
    4. Evaluate the model against a random opponent every E generations.
    5. Export to ONNX every E generations.
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
        self.buffer = MixedReplay(total_capacity=self.selfplay_cfg.replay_buffer_max)

    def run(self, generations: int = 10, eval_every: int = 5) -> None:
        """Execute the self-play loop for *generations* iterations."""
        checkpoint_dir = Path(self.train_cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        buffer_path = checkpoint_dir / "replay_buffer.json"

        # Try restoring buffer from previous run
        self.buffer.load(buffer_path)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_cfg.lr)
        criterion = GomokuLoss(weight_value=self.train_cfg.weight_value)
        use_amp = self.train_cfg.mixed_precision and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        for gen in range(1, generations + 1):
            # 1. Generate self-play games
            logger.info("Gen %d: generating %d games...", gen, self.selfplay_cfg.games)
            self.model.eval()
            positions = self.player.generate_games()
            self.buffer.add_many("self_play", positions)
            logger.info("Gen %d: buffer=%d, new=%d", gen, len(self.buffer), len(positions))

            # 2. Train on sampled positions
            if len(self.buffer) >= self.train_cfg.batch_size:
                self.model.train()
                sampled = self.buffer.sample(
                    self.train_cfg.batch_size * 4,
                    source_weights={"self_play": 1.0},
                )
                loader = self._make_loader(sampled)
                for epoch in range(1, self.train_cfg.epochs + 1):
                    metrics = train_epoch(
                        self.model, loader, criterion, optimizer, scaler,
                        self.device, use_amp=use_amp,
                    )
                    if epoch % 5 == 0 or epoch == self.train_cfg.epochs:
                        logger.info(
                            "Gen %d epoch %d/%d: loss=%.4f policy_acc=%.3f value_mae=%.3f",
                            gen, epoch, self.train_cfg.epochs,
                            metrics["loss"], metrics.get("policy_acc", 0),
                            metrics.get("value_mae", 0),
                        )

            # 3. Evaluate every N generations
            if gen % eval_every == 0:
                self.model.eval()
                results = evaluate_vs_random(self.model, num_games=20, device=self.device)
                logger.info("Gen %d eval: wins=%.0f%% draws=%.0f%% losses=%.0f%%",
                            gen, results["wins"] * 100, results["draws"] * 100, results["losses"] * 100)

            # 4. Export ONNX every N generations
            if gen % eval_every == 0:
                onnx_path = checkpoint_dir / f"model_gen_{gen:03d}.onnx"
                self.model.eval()
                export_to_onnx(self.model, str(onnx_path))
                logger.info("Gen %d: exported %s", gen, onnx_path)

            # 5. Save buffer
            self.buffer.save(buffer_path)

    def _make_loader(self, positions: list[dict]) -> DataLoader:
        """Convert position dicts to a DataLoader for training."""
        planes_list, policy_list, value_list = [], [], []
        for p in positions:
            tensor = board_to_tensor(p)
            planes_list.append(tensor)
            policy_list.append(pad_policy_target(p["policy"], p["board_size"], self.model_cfg.board_max))
            value_list.append(torch.tensor([p["value"]], dtype=torch.float32))

        dataset = TensorDataset(
            torch.stack(planes_list),
            torch.stack(policy_list),
            torch.stack(value_list),
        )
        return DataLoader(dataset, batch_size=self.train_cfg.batch_size, shuffle=True)
