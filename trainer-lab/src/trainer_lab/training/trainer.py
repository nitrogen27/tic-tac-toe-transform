"""End-to-end training driver with mixed-precision, checkpointing, and logging."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from trainer_lab.config import ModelConfig, TrainConfig
from trainer_lab.models.resnet import PolicyValueResNet
from trainer_lab.training.loss import GomokuLoss
from trainer_lab.training.scheduler import cosine_warmup_scheduler
from trainer_lab.training.metrics import policy_accuracy, value_mae

logger = logging.getLogger(__name__)


def _to_device(
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return tuple(t.to(device, non_blocking=True) for t in batch)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: GomokuLoss,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    use_amp: bool = True,
) -> dict[str, float]:
    """Run one training epoch. Returns dict of averaged metrics."""
    model.train()
    total_loss = 0.0
    total_ploss = 0.0
    total_vloss = 0.0
    total_pacc = 0.0
    total_vmae = 0.0
    n_batches = 0

    for planes, policy_target, value_target in tqdm(loader, desc="train", leave=False):
        planes, policy_target, value_target = _to_device(
            (planes, policy_target, value_target), device
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            policy_logits, value_pred = model(planes)
            loss, ploss, vloss = criterion(
                policy_logits, value_pred, policy_target, value_target
            )

        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_ploss += ploss.item()
        total_vloss += vloss.item()
        total_pacc += policy_accuracy(policy_logits.detach(), policy_target)
        total_vmae += value_mae(value_pred.detach(), value_target)
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "policy_loss": total_ploss / n,
        "value_loss": total_vloss / n,
        "policy_acc": total_pacc / n,
        "value_mae": total_vmae / n,
    }


def train(
    train_loader: DataLoader,
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
    device: torch.device | None = None,
) -> PolicyValueResNet:
    """Full training loop with checkpointing, LR scheduling, and TensorBoard.

    Returns the trained model.
    """
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = PolicyValueResNet(
        in_channels=model_cfg.in_channels,
        res_filters=model_cfg.res_filters,
        res_blocks=model_cfg.res_blocks,
        policy_filters=model_cfg.policy_filters,
        value_fc=model_cfg.value_fc,
        board_max=model_cfg.board_max,
    ).to(device)

    criterion = GomokuLoss(weight_value=train_cfg.weight_value)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    total_steps = train_cfg.epochs * len(train_loader)
    warmup_steps = min(total_steps // 10, 500)
    scheduler = cosine_warmup_scheduler(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda") if train_cfg.mixed_precision else None

    # Checkpoints & logging
    ckpt_dir = Path(train_cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(ckpt_dir / "tb_logs"))

    logger.info("Starting training: %d epochs, device=%s", train_cfg.epochs, device)

    for epoch in range(1, train_cfg.epochs + 1):
        metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            use_amp=train_cfg.mixed_precision,
        )
        scheduler.step()

        # Log
        for k, v in metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        logger.info(
            "Epoch %02d | loss=%.4f  p_loss=%.4f  v_loss=%.4f  p_acc=%.3f  v_mae=%.4f",
            epoch,
            metrics["loss"],
            metrics["policy_loss"],
            metrics["value_loss"],
            metrics["policy_acc"],
            metrics["value_mae"],
        )

        # Checkpoint every 5 epochs and at the end
        if epoch % 5 == 0 or epoch == train_cfg.epochs:
            path = ckpt_dir / f"model_epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), path)
            logger.info("Saved checkpoint: %s", path)

    writer.close()
    return model
