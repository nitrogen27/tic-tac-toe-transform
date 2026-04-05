"""Server-only training diagnostics and mini-tests."""

from __future__ import annotations

import time
from typing import Any

import torch

from gomoku_api.ws.model_profiles import variant_model_hparams
from gomoku_api.ws.model_registry import ModelRegistry
from gomoku_api.ws.train_service_ws import (
    _build_frozen_benchmark_suites,
    _compute_target_sanity_metrics,
    _evaluate_supervised_dataset,
    _generate_tactical_curriculum_positions,
    _load_offline_dataset_positions,
    _prepare_cuda_runtime,
    _resolve_variant_spec,
    _run_engine_exam,
    _run_training_steps,
    _split_holdout_positions,
)


async def _noop_callback(_event: dict[str, Any]) -> None:
    return None


async def run_training_diagnostics(
    variant: str,
    *,
    dataset_limit: int = 256,
    holdout_ratio: float = 0.20,
    tiny_steps: int = 32,
    batch_size: int = 128,
    model_profile: str = "auto",
    include_quick_probe: bool = True,
) -> dict[str, Any]:
    from trainer_lab.config import ModelConfig
    from trainer_lab.models.resnet import PolicyValueResNet
    from gomoku_api.ws.engine_evaluator import EngineEvaluator

    board_size, win_len = _resolve_variant_spec(variant)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime_flags = _prepare_cuda_runtime(device)
    cfg = ModelConfig()
    manifest = ModelRegistry(variant).read_manifest()
    resolved_profile, (res_filters, res_blocks, value_fc) = variant_model_hparams(
        variant,
        board_size,
        cfg,
        model_profile=model_profile,
        manifest=manifest,
    )

    offline_positions, offline_path, offline_type = _load_offline_dataset_positions(
        variant,
        max_positions=max(dataset_limit, 64),
    )
    dataset_source = offline_type or "tactical_fallback"
    required_positions = max(int(dataset_limit / max(1.0 - holdout_ratio, 0.1)), 64)
    if len(offline_positions) < required_positions:
        fallback_positions = await _generate_tactical_curriculum_positions(
            required_positions - len(offline_positions),
            board_size,
            win_len,
            _noop_callback,
        )
        offline_positions = list(offline_positions) + fallback_positions
        dataset_source = "mixed_fallback" if offline_type else "tactical_fallback"
    if not offline_positions:
        offline_positions = await _generate_tactical_curriculum_positions(
            required_positions,
            board_size,
            win_len,
            _noop_callback,
        )
        dataset_source = "tactical_fallback"

    train_positions, holdout_positions = _split_holdout_positions(
        offline_positions,
        holdout_ratio=holdout_ratio,
        max_holdout=max(int(dataset_limit * holdout_ratio), 16),
    )
    train_positions = train_positions[:dataset_limit]
    holdout_positions = holdout_positions[: max(int(dataset_limit * holdout_ratio), 16)]
    if len(train_positions) < 16:
        raise ValueError(f"Diagnostics dataset too small for {variant}: {len(train_positions)} positions")

    model = PolicyValueResNet(
        in_channels=cfg.in_channels,
        res_filters=res_filters,
        res_blocks=res_blocks,
        policy_filters=cfg.policy_filters,
        value_fc=value_fc,
        board_max=cfg.board_max,
    )
    if device.type == "cuda":
        model = model.to(device=device, memory_format=torch.channels_last)
    else:
        model = model.to(device)

    before_train = _evaluate_supervised_dataset(model, train_positions, device)
    before_holdout = _evaluate_supervised_dataset(model, holdout_positions, device) if holdout_positions else {}

    metrics_history: list[dict[str, Any]] = []
    started_at = time.monotonic()
    await _run_training_steps(
        model,
        train_positions,
        tiny_steps,
        min(batch_size, max(len(train_positions), 16)),
        device,
        runtime_flags,
        _noop_callback,
        metrics_history,
        phase="diagnostics",
        stage="tiny_overfit",
        overall_started_at=started_at,
        overall_percent_base=0,
        overall_percent_range=100,
        augment=True,
        augment_mode="random" if board_size <= 5 else "full",
        time_budget=0.0,
        variant=variant,
    )

    after_train = _evaluate_supervised_dataset(model, train_positions, device)
    after_holdout = _evaluate_supervised_dataset(model, holdout_positions, device) if holdout_positions else {}
    target_sanity = _compute_target_sanity_metrics(train_positions + holdout_positions)

    report: dict[str, Any] = {
        "variant": variant,
        "boardSize": board_size,
        "winLength": win_len,
        "device": device.type,
        "modelProfile": resolved_profile,
        "modelParams": sum(p.numel() for p in model.parameters()),
        "datasetSource": dataset_source,
        "datasetPath": str(offline_path) if offline_path is not None else None,
        "datasetSize": len(train_positions) + len(holdout_positions),
        "trainSize": len(train_positions),
        "holdoutSize": len(holdout_positions),
        "targetSanity": target_sanity,
        "beforeTrain": before_train,
        "beforeHoldout": before_holdout,
        "afterTrain": after_train,
        "afterHoldout": after_holdout,
        "trainHoldoutGap": round(
            max(0.0, after_train.get("policyTop1Acc", 0.0) - after_holdout.get("policyTop1Acc", 0.0)),
            2,
        ) if after_holdout else None,
        "tinyOverfitPassed": after_train.get("policyTop1Acc", 0.0) >= 90.0,
        "steps": tiny_steps,
        "batchSize": min(batch_size, max(len(train_positions), 16)),
        "elapsed": round(time.monotonic() - started_at, 2),
    }

    engine_eval = None
    try:
        engine_eval = EngineEvaluator()
        await engine_eval.start()
        if engine_eval.alive:
            frozen_suites = await _build_frozen_benchmark_suites(
                variant,
                board_size,
                win_len,
                engine_eval,
            )
            report["frozenBench"] = {
                name: _evaluate_supervised_dataset(model, positions, device)
                for name, positions in frozen_suites.items()
                if positions
            }
            if include_quick_probe:
                probe_result, _, probe_summary = await _run_engine_exam(
                    model,
                    board_size,
                    win_len,
                    device,
                    _noop_callback,
                    engine_eval,
                    variant=variant,
                    cycle=0,
                    total_cycles=0,
                    num_pairs=2,
                    phase="diagnostics_exam",
                    stage="quick_probe",
                    collect_failures=False,
                )
                report["quickProbe"] = {
                    "winrate": round(probe_result.winrate_a, 4),
                    **probe_summary,
                }
    finally:
        if engine_eval is not None:
            await engine_eval.stop()

    return report
