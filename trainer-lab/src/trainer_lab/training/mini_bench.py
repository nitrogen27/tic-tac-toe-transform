"""Fast synthetic benchmark for policy-value training throughput."""

from __future__ import annotations

import random
import time
from typing import Any

import torch

from trainer_lab.config import ModelConfig
from trainer_lab.models.resnet import PolicyValueResNet
from trainer_lab.training.loss import GomokuLoss


def _prepare_runtime(device: torch.device) -> dict[str, bool]:
    flags = {
        "mixed_precision": False,
        "tf32": False,
        "channels_last": False,
        "torch_compile": False,
        "compile_mode": None,
    }
    if device.type != "cuda":
        return flags

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    flags.update({
        "mixed_precision": True,
        "tf32": True,
        "channels_last": True,
    })
    return flags


def _maybe_compile_model(model: PolicyValueResNet, device: torch.device, runtime: dict[str, bool]) -> PolicyValueResNet:
    if device.type != "cuda" or not hasattr(torch, "compile"):
        return model
    compile_mode = "reduce-overhead"
    try:
        compiled = torch.compile(model, mode=compile_mode, fullgraph=False)
    except Exception:
        return model
    runtime["torch_compile"] = True
    runtime["compile_mode"] = compile_mode
    return compiled


def _make_synthetic_batch(
    batch_size: int,
    board_size: int = 15,
    board_max: int = 16,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = device or torch.device("cpu")
    planes = torch.zeros(batch_size, 6, board_max, board_max, dtype=torch.float32, device=device)
    policy = torch.zeros(batch_size, board_max * board_max, dtype=torch.float32, device=device)
    value = torch.empty(batch_size, 1, dtype=torch.float32, device=device).uniform_(-1.0, 1.0)

    for sample in range(batch_size):
        occupied: set[int] = set()
        stones = random.randint(board_size // 2, board_size * 2)
        last_flat = None
        current_player_is_one = random.choice([True, False])

        for stone_idx in range(stones):
            while True:
                row = random.randrange(board_size)
                col = random.randrange(board_size)
                flat = row * board_size + col
                if flat not in occupied:
                    occupied.add(flat)
                    break

            plane = 0 if (stone_idx % 2 == 0) == current_player_is_one else 1
            planes[sample, plane, row, col] = 1.0
            last_flat = (row, col)

        planes[sample, 5, :board_size, :board_size] = 1.0
        planes[sample, 2, :board_size, :board_size] = 1.0
        planes[sample, 2] -= (planes[sample, 0] + planes[sample, 1]).clamp(max=1.0)
        if current_player_is_one:
            planes[sample, 4, :board_size, :board_size] = 1.0
        if last_flat is not None:
            planes[sample, 3, last_flat[0], last_flat[1]] = 1.0

        legal_indices = torch.nonzero(planes[sample, 2].reshape(-1) > 0, as_tuple=False).view(-1)
        chosen = legal_indices[random.randrange(len(legal_indices))]
        policy[sample, chosen] = 1.0

    legal_mask = planes[:, 2].reshape(batch_size, -1)
    return planes, policy, value, legal_mask


def run_mini_benchmark(
    *,
    steps: int = 6,
    warmup_steps: int = 2,
    batch_size: int = 64,
    board_size: int = 15,
    device: str | torch.device | None = None,
    model_cfg: ModelConfig | None = None,
) -> dict[str, Any]:
    """Run a short synthetic training benchmark and return structured metrics."""
    model_cfg = model_cfg or ModelConfig()
    resolved_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    runtime = _prepare_runtime(resolved_device)

    model = PolicyValueResNet(
        in_channels=model_cfg.in_channels,
        res_filters=model_cfg.res_filters,
        res_blocks=model_cfg.res_blocks,
        policy_filters=model_cfg.policy_filters,
        value_fc=model_cfg.value_fc,
        board_max=model_cfg.board_max,
    )
    if resolved_device.type == "cuda":
        model = model.to(device=resolved_device, memory_format=torch.channels_last)
        torch.cuda.reset_peak_memory_stats(resolved_device)
    else:
        model = model.to(resolved_device)
    model = _maybe_compile_model(model, resolved_device, runtime)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = GomokuLoss(weight_value=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime["mixed_precision"] and resolved_device.type == "cuda")

    step_times_ms: list[float] = []
    final_loss = 0.0
    total_samples = 0

    for step in range(steps + warmup_steps):
        planes, policy_target, value_target, legal_mask = _make_synthetic_batch(
            batch_size=batch_size,
            board_size=board_size,
            board_max=model_cfg.board_max,
            device=resolved_device,
        )
        if resolved_device.type == "cuda":
            planes = planes.contiguous(memory_format=torch.channels_last)
            torch.cuda.synchronize()

        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=runtime["mixed_precision"] and resolved_device.type == "cuda"):
            policy_logits, value_pred = model(planes)
            loss, _, _ = criterion(
                policy_logits,
                value_pred,
                policy_target,
                value_target,
                legal_mask=legal_mask,
            )

        if runtime["mixed_precision"] and resolved_device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if resolved_device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if step >= warmup_steps:
            step_times_ms.append(elapsed_ms)
            total_samples += batch_size
            final_loss = loss.item()

    avg_step_ms = sum(step_times_ms) / max(len(step_times_ms), 1)
    samples_per_sec = total_samples / max(sum(step_times_ms) / 1000.0, 1e-6)
    max_memory_mb = 0.0
    if resolved_device.type == "cuda":
        max_memory_mb = torch.cuda.max_memory_allocated(resolved_device) / 1024 / 1024

    return {
        "device": resolved_device.type,
        "steps": steps,
        "warmupSteps": warmup_steps,
        "batchSize": batch_size,
        "boardSize": board_size,
        "avgStepMs": round(avg_step_ms, 2),
        "peakStepMs": round(max(step_times_ms), 2) if step_times_ms else 0.0,
        "minStepMs": round(min(step_times_ms), 2) if step_times_ms else 0.0,
        "samplesPerSec": round(samples_per_sec, 2),
        "finalLoss": round(final_loss, 6),
        "mixedPrecision": runtime["mixed_precision"],
        "tf32": runtime["tf32"],
        "channelsLast": runtime["channels_last"],
        "torchCompile": runtime["torch_compile"],
        "compileMode": runtime["compile_mode"],
        "maxMemoryMB": round(max_memory_mb, 2),
        "modelParams": sum(parameter.numel() for parameter in model.parameters()),
    }
