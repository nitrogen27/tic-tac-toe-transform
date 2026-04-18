"""Training service for WebSocket streaming — uses PyTorch via trainer-lab."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import torch
from torch.utils.data import DataLoader, TensorDataset
from trainer_lab.specs import VariantSpec, resolve_variant_spec as _shared_resolve_variant_spec

from gomoku_api.ws.gpu_info import get_gpu_info
from gomoku_api.ws.model_profiles import (
    current_model_profile_from_manifest,
    variant_model_hparams as _shared_variant_model_hparams,
)

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).resolve().parents[5] / "saved"
TRAIN_CALLBACK = Callable[[dict[str, Any]], Awaitable[None]]
_GPU_POLL_INTERVAL_S = 1.5
_PROGRESS_EMIT_INTERVAL_S = 0.75


def _ensure_saved_dir(variant: str) -> Path:
    d = SAVED_DIR / f"{variant}_resnet"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_variant_spec(variant: str) -> tuple[int, int]:
    spec = _shared_resolve_variant_spec(variant)
    return spec.board_size, spec.win_length


def _variant_metric_metadata(variant_spec: VariantSpec) -> dict[str, Any]:
    return {
        "variantSpec": variant_spec.to_metadata(),
        "curriculumStage": variant_spec.curriculum_stage,
    }


def _count_model_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _clone_model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _restore_model_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state_dict)


def _checkpoint_selection_score(summary: dict[str, Any], cycle: int) -> tuple[float, float, float, float, float, float, float, float, float, float, int]:
    """Rank candidate checkpoints by actual game strength.

    Priority:
    1. challenger-vs-champion winrate when available,
    2. self-play challenger-vs-previous-checkpoint winrate,
    3. balanced strength across both sides,
    4. strength as P2 specifically,
    5. challenger-vs-champion decisive wins,
    6. previous-checkpoint decisive wins,
    7. algorithm/engine decisive wins,
    8. total winrate,
    9. lower draw rate and lower pure-gap,
    10. later cycle as a tie-break.
    """
    champion_wr = float(summary.get("winrateVsChampion", -1.0) or 0.0) if summary.get("winrateVsChampion") is not None else -1.0
    champion_decisive = float(summary.get("decisiveWinRateVsChampion", 0.0) or 0.0)
    previous_wr = float(summary.get("winrateVsPreviousCheckpoint", -1.0) or 0.0) if summary.get("winrateVsPreviousCheckpoint") is not None else -1.0
    previous_decisive = float(summary.get("decisiveWinRateVsPreviousCheckpoint", 0.0) or 0.0)
    decisive = float(summary.get("decisiveWinRate", 0.0) or 0.0)
    winrate = float(summary.get("winrate", 0.0) or 0.0)
    winrate_as_p1_raw = summary.get("winrateAsP1")
    winrate_as_p2_raw = summary.get("winrateAsP2")
    winrate_as_p1 = winrate if winrate_as_p1_raw is None else float(winrate_as_p1_raw)
    winrate_as_p2 = winrate if winrate_as_p2_raw is None else float(winrate_as_p2_raw)
    balanced_side = min(winrate_as_p1, winrate_as_p2)
    pure_gap_rate = float(summary.get("pureGapRate", 1.0) or 0.0)
    pure_gap_rate_as_p1 = float(summary.get("pureGapRateAsP1", pure_gap_rate) or 0.0)
    pure_gap_rate_as_p2 = float(summary.get("pureGapRateAsP2", pure_gap_rate) or 0.0)
    pure_alignment = 1.0 - pure_gap_rate
    balanced_pure_alignment = min(1.0 - pure_gap_rate_as_p1, 1.0 - pure_gap_rate_as_p2)
    draw_rate = float(summary.get("drawRate", 0.0) or 0.0)
    return (
        champion_wr,
        previous_wr,
        balanced_side,
        winrate_as_p2,
        champion_decisive,
        previous_decisive,
        decisive,
        winrate,
        balanced_pure_alignment,
        pure_alignment - draw_rate,
        int(cycle),
    )


def _selfplay_previous_checkpoint_accepted(summary: dict[str, Any] | None) -> bool:
    """Accept a self-play iteration only if it is non-regressive vs previous."""
    if not summary:
        return False
    previous_wr_raw = summary.get("winrateVsPreviousCheckpoint")
    if previous_wr_raw is None:
        return False
    previous_wr = float(previous_wr_raw or 0.0)
    return previous_wr >= 0.50


def _selfplay_eval_num_pairs(
    iteration: int,
    total_iterations: int,
    *,
    purpose: str,
) -> int:
    """Keep self-play eval responsive during training and stricter near the end."""
    total_iterations = max(int(total_iterations or 1), 1)
    iteration = max(1, min(int(iteration or 1), total_iterations))
    progress = iteration / total_iterations

    if purpose == "engine":
        if progress >= 1.0:
            return 6
        if progress >= 0.75:
            return 5
        return 4

    if purpose == "challenger":
        if progress >= 1.0:
            return 3
        return 2

    if progress >= 1.0:
        return 4
    if progress >= 0.75:
        return 3
    return 2


def _latest_train_done_payload(variant: str) -> dict[str, Any] | None:
    log_dir = SAVED_DIR / "training_logs" / variant
    if not log_dir.exists():
        return None
    logs = sorted(log_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in logs:
        try:
            for raw in reversed(path.read_text(encoding="utf-8").splitlines()):
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                if obj.get("event") == "train.done":
                    return obj.get("payload") or {}
        except Exception:
            continue
    return None


def _candidate_summary_from_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metrics:
        return None
    if metrics.get("selectedCheckpointWinrate") is not None:
        return {
            "winrate": metrics.get("selectedCheckpointWinrate"),
            "decisiveWinRate": metrics.get("selectedCheckpointDecisiveWinRate", metrics.get("confirmDecisiveWinRate")),
            "drawRate": metrics.get("selectedCheckpointDrawRate", metrics.get("confirmDrawRate")),
            "winrateAsP1": metrics.get("selectedCheckpointWinrateAsP1"),
            "winrateAsP2": metrics.get("selectedCheckpointWinrateAsP2"),
            "balancedSideWinrate": metrics.get("selectedCheckpointBalancedSideWinrate"),
            "pureGapRate": metrics.get("selectedCheckpointPureGapRate", metrics.get("confirmPureGapRate")),
            "pureGapRateAsP1": metrics.get("selectedCheckpointPureGapRateAsP1"),
            "pureGapRateAsP2": metrics.get("selectedCheckpointPureGapRateAsP2"),
            "cycle": metrics.get("selectedCheckpointCycle", 0),
        }
    if metrics.get("winrateVsAlgorithm") is not None:
        return {
            "winrate": metrics.get("winrateVsAlgorithm"),
            "decisiveWinRate": metrics.get("confirmDecisiveWinRate", metrics.get("decisiveWinRate")),
            "drawRate": metrics.get("confirmDrawRate", metrics.get("drawRate")),
            "winrateAsP1": metrics.get("confirmWinrateAsP1", metrics.get("winrateAsP1")),
            "winrateAsP2": metrics.get("confirmWinrateAsP2", metrics.get("winrateAsP2")),
            "balancedSideWinrate": metrics.get("confirmBalancedSideWinrate", metrics.get("balancedSideWinrate")),
            "pureGapRate": metrics.get("confirmPureGapRate", metrics.get("pureGapRate")),
            "pureGapRateAsP1": metrics.get("confirmPureGapRateAsP1", metrics.get("pureGapRateAsP1")),
            "pureGapRateAsP2": metrics.get("confirmPureGapRateAsP2", metrics.get("pureGapRateAsP2")),
            "cycle": metrics.get("selectedCheckpointCycle", 0),
        }
    return None


def _read_active_candidate_summary(registry: Any, variant: str) -> dict[str, Any] | None:
    meta = registry.read_candidate_meta()
    summary = _candidate_summary_from_metrics(meta)
    if summary is not None:
        return summary
    if registry.has_active_candidate() and not registry.has_champion():
        return _candidate_summary_from_metrics(_latest_train_done_payload(variant))
    return None


def _populate_selected_checkpoint_payload(
    payload: dict[str, Any],
    selected_checkpoint_summary: dict[str, Any] | None,
) -> None:
    if not selected_checkpoint_summary:
        return
    payload["selectedCheckpointWinrate"] = round(float(selected_checkpoint_summary.get("winrate", 0.0) or 0.0), 4)
    payload["selectedCheckpointDecisiveWinRate"] = round(float(selected_checkpoint_summary.get("decisiveWinRate", 0.0) or 0.0), 4)
    payload["selectedCheckpointDrawRate"] = round(float(selected_checkpoint_summary.get("drawRate", 0.0) or 0.0), 4)
    payload["selectedCheckpointWinrateAsP1"] = round(float(selected_checkpoint_summary.get("winrateAsP1", 0.0) or 0.0), 4)
    payload["selectedCheckpointWinrateAsP2"] = round(float(selected_checkpoint_summary.get("winrateAsP2", 0.0) or 0.0), 4)
    payload["selectedCheckpointBalancedSideWinrate"] = round(float(selected_checkpoint_summary.get("balancedSideWinrate", 0.0) or 0.0), 4)
    if selected_checkpoint_summary.get("winrateVsChampion") is not None:
        payload["selectedCheckpointWinrateVsChampion"] = round(float(selected_checkpoint_summary.get("winrateVsChampion", 0.0) or 0.0), 4)
        payload["selectedCheckpointDecisiveWinRateVsChampion"] = round(float(selected_checkpoint_summary.get("decisiveWinRateVsChampion", 0.0) or 0.0), 4)
    if selected_checkpoint_summary.get("winrateVsPreviousCheckpoint") is not None:
        payload["selectedCheckpointWinrateVsPreviousCheckpoint"] = round(float(selected_checkpoint_summary.get("winrateVsPreviousCheckpoint", 0.0) or 0.0), 4)
        payload["selectedCheckpointDecisiveWinRateVsPreviousCheckpoint"] = round(float(selected_checkpoint_summary.get("decisiveWinRateVsPreviousCheckpoint", 0.0) or 0.0), 4)


def _populate_exam_summary_payload(
    payload: dict[str, Any],
    summary: dict[str, Any] | None,
    *,
    prefix: str = "",
) -> None:
    if not summary:
        return
    count_like_keys = {
        "pureGapCount",
        "pureTacticalGapCount",
        "pureConversionGapCount",
        "pureMissedWinCount",
        "pureMissedBlockCount",
        "pureMissedWinAsP1",
        "pureMissedWinAsP2",
        "pureMissedBlockAsP1",
        "pureMissedBlockAsP2",
    }
    for key in (
        "winrateAsP1",
        "winrateAsP2",
        "balancedSideWinrate",
        "tacticalOverrideRate",
        "valueGuidedRate",
        "modelPolicyRate",
        "avgUnsafeMovesFiltered",
        "pureGapRate",
        "pureGapRateAsP1",
        "pureGapRateAsP2",
        "pureAlignmentRate",
        "pureGapCount",
        "pureTacticalGapCount",
        "pureConversionGapCount",
        "pureMissedWinCount",
        "pureMissedBlockCount",
        "pureMissedWinAsP1",
        "pureMissedWinAsP2",
        "pureMissedBlockAsP1",
        "pureMissedBlockAsP2",
    ):
        if summary.get(key) is None:
            continue
        value = summary.get(key, 0.0)
        target_key = key if not prefix else f"{prefix}{key[0].upper()}{key[1:]}"
        payload[target_key] = int(value) if key in count_like_keys else round(float(value or 0.0), 4)
    if summary.get("decisionReasonCounts") is not None:
        target_key = "decisionReasonCounts" if not prefix else f"{prefix}DecisionReasonCounts"
        payload[target_key] = dict(summary.get("decisionReasonCounts") or {})


def _populate_validation_payload(
    payload: dict[str, Any],
    validation_history: list[dict[str, Any]] | None,
) -> None:
    if not validation_history:
        return
    payload["validationHistory"] = validation_history
    latest_validation = validation_history[-1]
    for key in (
        "holdoutPolicyAcc",
        "holdoutTeacherMass",
        "holdoutPolicyKL",
        "holdoutValueMAE",
        "holdoutValueSignAgreement",
        "holdoutLegalTargetRate",
        "holdoutDuplicateMergeRate",
        "holdoutPolicyMassMeanAbsError",
        "holdoutNonFiniteTargetRate",
        "frozenBlockAcc",
        "frozenWinAcc",
        "frozenExactAcc",
        "frozenMidAcc",
        "frozenLateAcc",
        "pureFrozenWinRecall",
        "pureFrozenBlockRecall",
        "hybridFrozenWinRecall",
        "hybridFrozenBlockRecall",
        "pureExactTrapRecall",
        "hybridExactTrapRecall",
        "pureWorstTrapFamilyRecall",
        "hybridWorstTrapFamilyRecall",
        "pureP1TrapRecall",
        "pureP2TrapRecall",
        "hybridP1TrapRecall",
        "hybridP2TrapRecall",
        "exactPackSize",
        "exactPackFamilyCount",
    ):
        if latest_validation.get(key) is not None:
            payload[key] = latest_validation[key]
    for key in (
        "pureWorstTrapFamily",
        "hybridWorstTrapFamily",
        "pureExactFamilyRecall",
        "hybridExactFamilyRecall",
    ):
        if latest_validation.get(key) is not None:
            payload[key] = latest_validation[key]


def _build_deferred_train_done_payload(
    *,
    registry: Any,
    variant_spec: VariantSpec,
    variant: str,
    epochs: int,
    started_at: float,
    metrics_history: list[dict[str, Any]],
    device: torch.device,
    model_profile: str,
    positions_count: int,
    failure_bank_size: int,
    cycle_count: int,
    selected_checkpoint_cycle: int | None,
    selected_checkpoint_summary: dict[str, Any] | None,
    winrate_history: list[dict[str, Any]],
    validation_history: list[dict[str, Any]],
    teacher_backend_requested: str,
    teacher_backend_resolved: str,
    confirm_backend_requested: str,
    confirm_backend_resolved: str,
) -> dict[str, Any]:
    manifest = registry.read_manifest()
    payload: dict[str, Any] = {
        "variant": variant,
        "curriculumStage": variant_spec.curriculum_stage,
        "variantSpec": variant_spec.to_metadata(),
        "epochs": epochs,
        "elapsed": round(time.monotonic() - started_at, 1),
        "metricsHistory": metrics_history,
        "device": device.type,
        "modelProfile": model_profile,
        "positions": positions_count,
        "failureBankSize": failure_bank_size,
        "championGeneration": manifest.get("current_champion_generation"),
        "promoted": False,
        "promotionPending": True,
        "evaluationQueued": True,
        "cycles": cycle_count,
        "selectedCheckpointCycle": selected_checkpoint_cycle,
        "teacherBackendRequested": teacher_backend_requested,
        "teacherBackendResolved": teacher_backend_resolved,
        "confirmBackendRequested": confirm_backend_requested,
        "confirmBackendResolved": confirm_backend_resolved,
        "message": "Основное обучение завершено; confirm и promotion продолжаются в фоне.",
        **registry.serving_summary(expected_spec=variant_spec),
    }
    _populate_selected_checkpoint_payload(payload, selected_checkpoint_summary)
    if selected_checkpoint_summary:
        payload["winrateVsAlgorithm"] = round(float(selected_checkpoint_summary.get("winrate", 0.0) or 0.0), 4)
        payload["decisiveWinRate"] = round(float(selected_checkpoint_summary.get("decisiveWinRate", 0.0) or 0.0), 4)
        payload["drawRate"] = round(float(selected_checkpoint_summary.get("drawRate", 0.0) or 0.0), 4)
    if winrate_history:
        payload["winrateHistory"] = winrate_history
    _populate_exam_summary_payload(payload, selected_checkpoint_summary)
    _populate_validation_payload(payload, validation_history)
    return payload


def _build_background_progress_payload(
    payload: dict[str, Any],
    *,
    phase_label: str,
    base_percent: int,
    span_percent: int,
    message_prefix: str,
) -> dict[str, Any]:
    current = payload.get("game") or payload.get("step") or payload.get("iteration") or payload.get("cycle") or 0
    total = payload.get("totalGames") or payload.get("totalSteps") or payload.get("totalIterations") or payload.get("totalCycles") or 1
    frac = 0.0
    try:
        frac = min(max(float(current) / max(float(total), 1.0), 0.0), 1.0)
    except Exception:
        frac = 0.0
    percent = int(round(base_percent + span_percent * frac))
    return {
        "epoch": 1,
        "epochs": 1,
        "epochPercent": percent,
        "phase": payload.get("phase"),
        "stage": payload.get("stage"),
        "variant": payload.get("variant"),
        "game": payload.get("game"),
        "totalGames": payload.get("totalGames"),
        "step": payload.get("step"),
        "totalSteps": payload.get("totalSteps"),
        "loss": payload.get("loss"),
        "acc": payload.get("acc") or payload.get("accuracy"),
        "batchProgress": payload.get("batchProgress"),
        "currentBatch": payload.get("currentBatch"),
        "batchesPerEpoch": payload.get("batchesPerEpoch"),
        "message": payload.get("message") or f"{message_prefix}: {phase_label}",
        "winrateVsAlgorithm": payload.get("winrate"),
        "winrateVsChampion": payload.get("winrateVsChampion"),
        "winrateVsPreviousCheckpoint": payload.get("winrateVsPreviousCheckpoint"),
        "balancedSideWinrate": payload.get("balancedSideWinrate"),
        "acceptedVsPreviousCheckpoint": payload.get("acceptedVsPreviousCheckpoint"),
    }


async def run_deferred_evaluator_tail(
    context: dict[str, Any],
    callback: TRAIN_CALLBACK,
) -> dict[str, Any]:
    from trainer_lab.config import ModelConfig
    from trainer_lab.models.resnet import PolicyValueResNet
    from gomoku_api.ws.model_registry import ModelRegistry
    from gomoku_api.ws.oracle_backends import create_oracle_evaluator
    from gomoku_api.ws.promotion import evaluate_promotion
    from gomoku_api.ws.predict_service import clear_cached_model

    variant = str(context["variant"])
    variant_spec = _shared_resolve_variant_spec(variant)
    variant_metric_metadata = _variant_metric_metadata(variant_spec)
    board_size = int(context["boardSize"])
    win_len = int(context["winLen"])
    cycle_count = int(context["cycleCount"])
    epochs = int(context["epochs"])
    started_at = float(context["startedAt"])
    model_profile = str(context["modelProfile"])
    res_filters = int(context["resFilters"])
    res_blocks = int(context["resBlocks"])
    value_fc = int(context["valueFc"])
    selected_checkpoint_cycle = context.get("selectedCheckpointCycle")
    selected_checkpoint_summary = context.get("selectedCheckpointSummary")
    metrics_history = list(context.get("metricsHistory") or [])
    validation_history = list(context.get("validationHistory") or [])
    winrate_history = list(context.get("winrateHistory") or [])
    teacher_backend_requested = str(context.get("teacherBackendRequested", "auto"))
    teacher_backend_resolved = str(context.get("teacherBackendResolved", "builtin"))
    confirm_backend_requested = str(context.get("confirmBackendRequested", "auto"))
    confirm_backend_resolved = str(context.get("confirmBackendResolved", "builtin"))
    positions_count = int(context.get("positionsCount", 0))
    failure_bank_size = int(context.get("failureBankSize", 0))

    registry = ModelRegistry(variant)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig()
    model = PolicyValueResNet(
        in_channels=cfg.in_channels,
        res_filters=res_filters,
        res_blocks=res_blocks,
        policy_filters=cfg.policy_filters,
        value_fc=value_fc,
        board_max=cfg.board_max,
    )
    load_path = registry.working_candidate_path if registry.working_candidate_path.exists() else registry.candidate_path
    if not load_path.exists():
        raise FileNotFoundError(f"No deferred evaluator checkpoint found for {variant}: {load_path}")
    registry.load_into(model, load_path, device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    async def evaluator_cb(event: dict[str, Any]) -> None:
        if event.get("type") != "train.progress":
            await callback(event)
            return
        payload = event.get("payload") or {}
        phase = str(payload.get("phase") or "")
        if phase == "arena":
            progress = _build_background_progress_payload(
                payload,
                phase_label="challenger vs previous checkpoint",
                base_percent=10,
                span_percent=20,
                message_prefix="Фоновая арена",
            )
        elif phase == "confirm_exam":
            progress = _build_background_progress_payload(
                payload,
                phase_label="confirm exam",
                base_percent=35,
                span_percent=45,
                message_prefix="Фоновый confirm",
            )
        elif phase == "promotion":
            progress = _build_background_progress_payload(
                payload,
                phase_label="promotion",
                base_percent=82,
                span_percent=18,
                message_prefix="Фоновый promotion",
            )
        else:
            progress = _build_background_progress_payload(
                payload,
                phase_label=phase or "evaluation",
                base_percent=0,
                span_percent=100,
                message_prefix="Фоновая оценка",
            )
        await callback({"type": "background_train.progress", "payload": progress})

    await callback({
        "type": "background_train.started",
        "payload": {
            "variant": variant,
            "curriculumStage": variant_spec.curriculum_stage,
            "variantSpec": variant_spec.to_metadata(),
            "epoch": 1,
            "epochs": 1,
            "epochPercent": 0,
            "message": "Основное обучение завершено. Запущены confirm и promotion в фоне.",
        },
    })

    engine_eval = None
    confirm_eval = None
    engine_available = False
    confirm_available = False
    quick_result = None
    strong_result = None
    confirm_result = None
    confirm_summary: dict[str, Any] | None = None
    promoted_this_run = False
    try:
        engine_eval, teacher_backend_resolved = create_oracle_evaluator(
            board_size,
            win_len,
            backend=teacher_backend_requested,
            role="teacher",
        )
        await engine_eval.start()
        engine_available = bool(engine_eval.alive)
    except Exception as exc:
        logger.warning("Deferred teacher oracle unavailable for %s: %s", variant, exc)
        if engine_eval is not None:
            try:
                await engine_eval.stop()
            except Exception:
                pass
        engine_eval = None
        engine_available = False

    try:
        if confirm_backend_requested == teacher_backend_requested and engine_eval is not None:
            confirm_eval = engine_eval
            confirm_backend_resolved = teacher_backend_resolved
            confirm_available = engine_available
        else:
            confirm_eval, confirm_backend_resolved = create_oracle_evaluator(
                board_size,
                win_len,
                backend=confirm_backend_requested,
                role="confirm",
            )
            await confirm_eval.start()
            confirm_available = bool(confirm_eval.alive)
    except Exception as exc:
        logger.warning("Deferred confirm oracle unavailable for %s: %s", variant, exc)
        if confirm_eval is not None and confirm_eval is not engine_eval:
            try:
                await confirm_eval.stop()
            except Exception:
                pass
        confirm_eval = engine_eval
        confirm_available = engine_available
        confirm_backend_resolved = teacher_backend_resolved

    if registry.has_champion():
        quick_result = await _run_challenger_vs_champion(
            model,
            registry,
            board_size=board_size,
            win_len=win_len,
            device=device,
            callback=evaluator_cb,
            variant=variant,
            num_pairs=4,
            phase="arena",
            stage="deferred_challenger",
        )

    if confirm_available and confirm_eval is not None:
        confirm_result, _, confirm_summary = await _run_engine_exam(
            model,
            board_size,
            win_len,
            device,
            evaluator_cb,
            confirm_eval,
            variant=variant,
            cycle=cycle_count + 1,
            total_cycles=cycle_count,
            num_pairs=10,
            previous_result=selected_checkpoint_summary,
            phase="confirm_exam",
            stage="engine_eval",
            collect_failures=False,
        )
        strong_result = confirm_result

    if board_size <= 9:
        block_acc = _compute_tactical_accuracy(model, board_size, win_len, device, n_samples=200, motif_filter="block")
        win_acc = _compute_tactical_accuracy(model, board_size, win_len, device, n_samples=200, motif_filter="win")
    else:
        block_acc = 90.0
        win_acc = 90.0

    manifest_before = registry.read_manifest()
    prev_algo_winrate = None
    if manifest_before.get("history"):
        prev_algo_winrate = manifest_before["history"][-1].get("winrateVsAlgorithm")

    decision = evaluate_promotion(
        quick_result,
        strong_result,
        block_accuracy=block_acc,
        win_accuracy=win_acc,
        balanced_side_winrate=confirm_summary.get("balancedSideWinrate") if confirm_summary else None,
        winrate_as_p2=confirm_summary.get("winrateAsP2") if confirm_summary else None,
        prev_algo_winrate=prev_algo_winrate,
        require_champion_match=registry.has_champion(),
    )
    await evaluator_cb({
        "type": "train.progress",
            "payload": {
                "phase": "promotion",
                "stage": "decision",
                "variant": variant,
                "curriculumStage": variant_spec.curriculum_stage,
                "variantSpec": variant_spec.to_metadata(),
                "promotionDecision": decision.promoted,
                "message": "Фоновый promotion завершает проверку кандидата.",
                **decision.to_dict(),
        },
    })

    active_candidate_summary = _read_active_candidate_summary(registry, variant)
    if decision.promoted:
        registry.promote_candidate(
            generation=cycle_count,
            metrics={
                **variant_metric_metadata,
                **decision.to_dict(),
                "modelProfile": model_profile,
                "resFilters": res_filters,
                "resBlocks": res_blocks,
                "valueFc": value_fc,
            },
            reason=decision.reason,
            source_path=registry.working_candidate_path if registry.working_candidate_path.exists() else None,
        )
        promoted_this_run = True
        await callback({
            "type": "model.promoted",
            "payload": {
                "variant": variant,
                "curriculumStage": variant_spec.curriculum_stage,
                "variantSpec": variant_spec.to_metadata(),
                "generation": cycle_count,
                "promotionDecision": True,
                **decision.to_dict(),
            },
        })
    else:
        selected_summary_for_commit = selected_checkpoint_summary or confirm_summary or {}
        should_commit_candidate = False
        if not registry.has_champion():
            if not registry.has_active_candidate():
                should_commit_candidate = True
            elif selected_summary_for_commit:
                selected_score = _checkpoint_selection_score(
                    selected_summary_for_commit,
                    int((selected_checkpoint_cycle or cycle_count) or cycle_count),
                )
                current_score = (
                    _checkpoint_selection_score(
                        active_candidate_summary,
                        int(active_candidate_summary.get("cycle", 0) or 0),
                    )
                    if active_candidate_summary
                    else None
                )
                should_commit_candidate = current_score is None or selected_score > current_score
        if should_commit_candidate and registry.working_candidate_path.exists():
            registry.commit_working_candidate(
                generation=cycle_count,
                metrics={
                    **variant_metric_metadata,
                    "phase": "checkpoint_selection",
                    "selectedCheckpointCycle": selected_checkpoint_cycle,
                    "selectedCheckpointWinrate": selected_summary_for_commit.get("winrate"),
                    "selectedCheckpointDecisiveWinRate": selected_summary_for_commit.get("decisiveWinRate"),
                    "selectedCheckpointDrawRate": selected_summary_for_commit.get("drawRate"),
                    "selectedCheckpointWinrateAsP1": selected_summary_for_commit.get("winrateAsP1"),
                    "selectedCheckpointWinrateAsP2": selected_summary_for_commit.get("winrateAsP2"),
                    "selectedCheckpointBalancedSideWinrate": selected_summary_for_commit.get("balancedSideWinrate"),
                    "selectedCheckpointPureGapRate": selected_summary_for_commit.get("pureGapRate"),
                    "selectedCheckpointPureGapRateAsP1": selected_summary_for_commit.get("pureGapRateAsP1"),
                    "selectedCheckpointPureGapRateAsP2": selected_summary_for_commit.get("pureGapRateAsP2"),
                    "selectedCheckpointWinrateVsChampion": selected_summary_for_commit.get("winrateVsChampion"),
                    "selectedCheckpointDecisiveWinRateVsChampion": selected_summary_for_commit.get("decisiveWinRateVsChampion"),
                    "selectedCheckpointWinrateVsPreviousCheckpoint": selected_summary_for_commit.get("winrateVsPreviousCheckpoint"),
                    "selectedCheckpointDecisiveWinRateVsPreviousCheckpoint": selected_summary_for_commit.get("decisiveWinRateVsPreviousCheckpoint"),
                    "winrateVsAlgorithm": decision.winrate_vs_algorithm,
                    "confirmDecisiveWinRate": confirm_summary.get("decisiveWinRate") if confirm_summary else None,
                    "confirmDrawRate": confirm_summary.get("drawRate") if confirm_summary else None,
                    "confirmWinrateAsP1": confirm_summary.get("winrateAsP1") if confirm_summary else None,
                    "confirmWinrateAsP2": confirm_summary.get("winrateAsP2") if confirm_summary else None,
                    "confirmBalancedSideWinrate": confirm_summary.get("balancedSideWinrate") if confirm_summary else None,
                    "confirmPureGapRate": confirm_summary.get("pureGapRate") if confirm_summary else None,
                },
            )
        await callback({
            "type": "promotion.rejected",
            "payload": {
                "variant": variant,
                "curriculumStage": variant_spec.curriculum_stage,
                "variantSpec": variant_spec.to_metadata(),
                "generation": cycle_count,
                "promotionDecision": False,
                **decision.to_dict(),
            },
        })

    if promoted_this_run:
        clear_cached_model(variant)

    manifest = registry.read_manifest()
    done_payload: dict[str, Any] = {
        "variant": variant,
        "curriculumStage": variant_spec.curriculum_stage,
        "variantSpec": variant_spec.to_metadata(),
        "epochs": epochs,
        "elapsed": round(time.monotonic() - started_at, 1),
        "metricsHistory": metrics_history,
        "device": device.type,
        "modelProfile": model_profile,
        "positions": positions_count,
        "failureBankSize": failure_bank_size,
        "championGeneration": manifest.get("current_champion_generation"),
        "promoted": promoted_this_run,
        "promotionDecision": promoted_this_run,
        "promotionPending": False,
        "evaluationQueued": False,
        "cycles": cycle_count,
        "selectedCheckpointCycle": selected_checkpoint_cycle,
        "teacherBackendRequested": teacher_backend_requested,
        "teacherBackendResolved": teacher_backend_resolved,
        "confirmBackendRequested": confirm_backend_requested,
        "confirmBackendResolved": confirm_backend_resolved,
        "message": "Фоновая оценка завершена.",
        **registry.serving_summary(expected_spec=variant_spec),
    }
    _populate_selected_checkpoint_payload(done_payload, selected_checkpoint_summary)
    if quick_result:
        done_payload["winrateVsChampion"] = round(quick_result.winrate_a, 4)
        done_payload["decisiveWinRateVsChampion"] = round(quick_result.decisive_winrate_a, 4)
    if strong_result:
        done_payload["winrateVsAlgorithm"] = round(strong_result.winrate_a, 4)
        done_payload["decisiveWinRate"] = round(strong_result.decisive_winrate_a, 4)
        done_payload["drawRate"] = round(strong_result.draw_rate, 4)
    final_exam_summary = confirm_summary or selected_checkpoint_summary
    _populate_exam_summary_payload(done_payload, final_exam_summary)
    if winrate_history:
        done_payload["winrateHistory"] = winrate_history
    if confirm_result is not None:
        done_payload["confirmWins"] = confirm_result.wins_a
        done_payload["confirmLosses"] = confirm_result.wins_b
        done_payload["confirmDraws"] = confirm_result.draws
        done_payload["confirmDecisiveWinRate"] = round(confirm_result.decisive_winrate_a, 4)
        done_payload["confirmDrawRate"] = round(confirm_result.draw_rate, 4)
    _populate_exam_summary_payload(done_payload, confirm_summary, prefix="confirm")
    _populate_validation_payload(done_payload, validation_history)
    await callback({"type": "background_train.done", "payload": done_payload})
    return done_payload
    


def _variant_model_hparams(
    board_size: int,
    cfg: Any,
    *,
    variant: str = "",
    model_profile: str | None = None,
    manifest: dict[str, Any] | None = None,
    variant_spec: VariantSpec | None = None,
) -> tuple[str, tuple[int, int, int]]:
    return _shared_variant_model_hparams(
        variant or f"gomoku{board_size}",
        board_size,
        cfg,
        model_profile=model_profile,
        manifest=manifest,
        spec=variant_spec,
    )


def _prepare_cuda_runtime(device: torch.device) -> dict[str, bool]:
    enabled = {
        "mixedPrecision": False,
        "tf32": False,
        "channelsLast": False,
        "torchCompile": False,
        "compileMode": None,
    }
    if device.type != "cuda":
        return enabled

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    enabled.update({
        "mixedPrecision": True,
        "tf32": True,
        "channelsLast": True,
    })
    return enabled


def _maybe_compile_model(model: torch.nn.Module, device: torch.device, runtime_flags: dict[str, Any]) -> torch.nn.Module:
    """Best-effort torch.compile() wrapper for CUDA training workloads."""
    if device.type != "cuda" or not hasattr(torch, "compile"):
        return model

    # Triton is required for torch.compile on CUDA; skip if unavailable (Windows)
    try:
        import triton  # noqa: F401
    except ImportError:
        logger.info("Triton not installed — skipping torch.compile (eager mode)")
        return model

    compile_mode = "reduce-overhead"
    try:
        compiled = torch.compile(model, mode=compile_mode, fullgraph=False)
    except Exception as exc:
        logger.warning("torch.compile unavailable for current run, falling back to eager mode: %s", exc)
        return model

    runtime_flags["torchCompile"] = True
    runtime_flags["compileMode"] = compile_mode
    return compiled


def _load_champion_model_for_arena(
    registry: Any,
    board_size: int,
    *,
    variant: str,
    device: torch.device,
) -> Any | None:
    if not registry.has_champion():
        return None

    from trainer_lab.config import ModelConfig as _MC
    from trainer_lab.models.resnet import PolicyValueResNet as _PVR

    cfg = _MC()
    champion_manifest = registry.read_manifest()
    _, (res_filters, res_blocks, value_fc) = _variant_model_hparams(
        board_size,
        cfg,
        variant=variant,
        manifest=champion_manifest,
    )
    champion_model = _PVR(
        in_channels=cfg.in_channels,
        res_filters=res_filters,
        res_blocks=res_blocks,
        policy_filters=cfg.policy_filters,
        value_fc=value_fc,
        board_max=cfg.board_max,
    )
    champion_model = registry.load_champion_into(champion_model, device)
    if champion_model is not None and device.type == "cuda":
        champion_model = champion_model.to(memory_format=torch.channels_last)
    return champion_model


async def _run_challenger_vs_champion(
    model: Any,
    registry: Any,
    *,
    board_size: int,
    win_len: int,
    device: torch.device,
    callback: TRAIN_CALLBACK,
    variant: str,
    num_pairs: int,
    phase: str,
    stage: str,
) -> Any | None:
    from gomoku_api.ws.arena_eval import arena_match

    champion_model = _load_champion_model_for_arena(
        registry,
        board_size,
        variant=variant,
        device=device,
    )
    if champion_model is None:
        return None
    try:
        return await arena_match(
            model,
            champion_model,
            board_size,
            win_len,
            num_pairs=num_pairs,
            device=device,
            callback=callback,
            variant=variant,
            phase=phase,
            stage=stage,
        )
    finally:
        del champion_model


async def _run_model_vs_model_arena(
    model_a: Any,
    model_b: Any,
    *,
    board_size: int,
    win_len: int,
    device: torch.device,
    callback: TRAIN_CALLBACK,
    variant: str,
    num_pairs: int,
    phase: str,
    stage: str,
) -> Any:
    from gomoku_api.ws.arena_eval import arena_match

    return await arena_match(
        model_a,
        model_b,
        board_size,
        win_len,
        num_pairs=num_pairs,
        device=device,
        callback=callback,
        variant=variant,
        phase=phase,
        stage=stage,
    )


def _should_emit_progress(now: float, last_emit_at: float, *, force: bool = False) -> bool:
    return force or (now - last_emit_at) >= _PROGRESS_EMIT_INTERVAL_S


def _maybe_refresh_gpu_info(now: float, last_gpu_probe: float, live_gpu: dict[str, Any]) -> tuple[dict[str, Any], float]:
    if now - last_gpu_probe >= _GPU_POLL_INTERVAL_S:
        return get_gpu_info(), now
    return live_gpu, last_gpu_probe


def _flat_to_board2d(board: list[int], board_size: int) -> list[list[int]]:
    return [
        [board[row * board_size + col] for col in range(board_size)]
        for row in range(board_size)
    ]


def _board2d_to_flat(board: list[list[int]]) -> list[int]:
    return [cell for row in board for cell in row]


def _policy_cell_index(flat_index: int, board_size: int) -> int:
    row, col = divmod(flat_index, board_size)
    return row * 16 + col


def _one_hot_policy(move: int, board_size: int) -> list[float]:
    policy = [0.0] * 256
    if move >= 0:
        policy[_policy_cell_index(move, board_size)] = 1.0
    return policy


def _soft_policy_from_engine_hints(
    best_move: int,
    board: list[int],
    board_size: int,
    hints: list[dict[str, Any]] | None,
) -> list[float]:
    """Build a soft target distribution from engine hints / top moves."""
    padded = [0.0] * 256
    legal = [idx for idx, cell in enumerate(board) if cell == 0]
    if not legal:
        return padded

    scored: list[tuple[int, float]] = []
    for hint in hints or []:
        try:
            move = int(hint.get("move", -1))
            score = float(hint.get("score", 0.0))
        except Exception:
            continue
        if move < 0 or move >= len(board) or board[move] != 0:
            continue
        scored.append((move, score))

    if not scored:
        return _one_hot_policy(best_move, board_size)

    max_score = max(score for _, score in scored)
    weights: list[tuple[int, float]] = []
    total = 0.0
    for move, score in scored:
        weight = math.exp((score - max_score) / 2.5)
        weights.append((move, weight))
        total += weight

    if total <= 0:
        return _one_hot_policy(best_move, board_size)

    for move, weight in weights:
        padded[_policy_cell_index(move, board_size)] = weight / total

    if best_move >= 0 and board[best_move] == 0 and padded[_policy_cell_index(best_move, board_size)] == 0.0:
        padded[_policy_cell_index(best_move, board_size)] = 1.0
        return _normalize_policy_vector(padded)

    return _normalize_policy_vector(padded)


def _find_immediate_move(board: list[int], board_size: int, win_len: int, player: int) -> int | None:
    for move, cell in enumerate(board):
        if cell != 0:
            continue
        board[move] = player
        winner = _nxn_winner(board, board_size, win_len, move)
        board[move] = 0
        if winner == player:
            return move
    return None


def _extract_telemetry(gpu_info: dict[str, Any]) -> dict[str, Any]:
    telemetry = gpu_info.get("telemetry") or {}
    vram = gpu_info.get("vram") or {}
    return {
        "gpuUtilization": telemetry.get("utilizationGpu"),
        "gpuMemoryUtilization": telemetry.get("utilizationMemory"),
        "gpuPowerW": telemetry.get("powerDrawW"),
        "gpuPowerLimitW": telemetry.get("powerLimitW"),
        "gpuTemperatureC": telemetry.get("temperatureC"),
        "gpuClockSmMHz": telemetry.get("clockSmMHz"),
        "gpuClockMemMHz": telemetry.get("clockMemMHz"),
        "gpuMemoryUsedMB": telemetry.get("memoryUsedMB", vram.get("usedMB")),
        "gpuMemoryTotalMB": telemetry.get("memoryTotalMB", vram.get("totalMB")),
        "gpuAllocatedMB": vram.get("allocatedMB"),
        "gpuReservedMB": vram.get("reservedMB"),
        "gpuTelemetryTimestamp": telemetry.get("timestamp"),
    }


def _compute_tactical_accuracy(
    model: Any, board_size: int, win_len: int, device: Any, n_samples: int = 200,
    motif_filter: str | None = None,
) -> float:
    """Evaluate model on tactical positions. Returns accuracy 0-100.

    Parameters
    ----------
    motif_filter : "win", "block", or None (both mixed).
    """
    from trainer_lab.data.encoder import board_to_tensor
    import torch.nn.functional as F

    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            board = [0] * (board_size * board_size)
            motif = motif_filter or random.choice(("win", "block"))
            current = random.choice((1, 2))
            line_player = current if motif == "win" else (2 if current == 1 else 1)
            dr, dc = random.choice(directions)

            valid_starts = [
                (r, c) for r in range(board_size) for c in range(board_size)
                if 0 <= r + dr * (win_len - 1) < board_size and 0 <= c + dc * (win_len - 1) < board_size
            ]
            if not valid_starts:
                continue
            sr, sc = random.choice(valid_starts)
            gap = random.randrange(win_len)
            target_move = (sr + dr * gap) * board_size + (sc + dc * gap)

            for step in range(win_len):
                r, c = sr + dr * step, sc + dc * step
                flat = r * board_size + c
                if flat != target_move:
                    board[flat] = line_player

            pos = {
                "board_size": board_size,
                "board": [[board[r * board_size + c] for c in range(board_size)] for r in range(board_size)],
                "current_player": current,
                "last_move": None,
            }
            planes = board_to_tensor(pos).unsqueeze(0).to(device)
            if device.type == "cuda":
                planes = planes.contiguous(memory_format=torch.channels_last)
            logits, _ = model(planes)
            legal_mask = planes[:, 2].reshape(1, -1)
            masked = logits + (1.0 - legal_mask) * (-1e8)
            pred = masked.argmax(dim=1).item()

            target_r, target_c = divmod(target_move, board_size)
            target_idx = target_r * 16 + target_c
            if pred == target_idx:
                correct += 1
            total += 1

    model.train()
    return round(100.0 * correct / max(total, 1), 1)


async def _emit_dataset_progress(
    callback: TRAIN_CALLBACK,
    *,
    generated: int,
    total: int,
    stage: str,
    message: str,
    start_time: float,
    games: int = 0,
    unit: str = "positions",
) -> None:
    elapsed = max(time.monotonic() - start_time, 0.001)
    percent = round(min(generated / max(total, 1), 1.0) * 100, 1)
    await callback({
        "type": "dataset.progress",
        "payload": {
            "generated": generated,
            "total": total,
            "games": games,
            "percent": percent,
            "rate": round(generated / elapsed, 2),
            "elapsed": round(elapsed, 1),
            "stage": stage,
            "message": message,
            "unit": unit,
            "workers": 1,
        },
    })


def _position_last_move_index(position: dict[str, Any]) -> int:
    last_move = position.get("last_move")
    if not last_move:
        return -1
    row, col = last_move
    return row * int(position["board_size"]) + col


def _position_fingerprint(position: dict[str, Any]) -> tuple[Any, ...]:
    board_2d = position.get("board") or []
    return (
        int(position.get("board_size", 0)),
        tuple(_board2d_to_flat(board_2d)),
        int(position.get("current_player", 0)),
        tuple(position.get("last_move") or (-1, -1)),
    )


def _transform_square_tensor(grid: torch.Tensor, transform_idx: int) -> torch.Tensor:
    if transform_idx == 0:
        return grid
    if transform_idx == 1:
        return torch.rot90(grid, 1, dims=(0, 1))
    if transform_idx == 2:
        return torch.rot90(grid, 2, dims=(0, 1))
    if transform_idx == 3:
        return torch.rot90(grid, 3, dims=(0, 1))
    if transform_idx == 4:
        return grid.flip(0)
    if transform_idx == 5:
        return grid.flip(1)
    if transform_idx == 6:
        return grid.transpose(0, 1)
    return torch.rot90(grid, 1, dims=(0, 1)).flip(0)


def _transform_last_move(
    last_move: list[int] | tuple[int, int] | None,
    board_size: int,
    transform_idx: int,
) -> list[int] | None:
    if not last_move:
        return None
    row, col = int(last_move[0]), int(last_move[1])
    marker = torch.zeros((board_size, board_size), dtype=torch.int8)
    marker[row, col] = 1
    transformed = _transform_square_tensor(marker, transform_idx).reshape(-1)
    flat_idx = int(transformed.argmax().item())
    return [flat_idx // board_size, flat_idx % board_size]


def _transform_policy_vector(policy: list[float], board_size: int, transform_idx: int) -> list[float]:
    grid = torch.tensor(policy, dtype=torch.float32).reshape(16, 16)
    if board_size >= 16:
        transformed = _transform_square_tensor(grid, transform_idx)
        return transformed.reshape(256).tolist()

    transformed = torch.zeros_like(grid)
    transformed[:board_size, :board_size] = _transform_square_tensor(
        grid[:board_size, :board_size],
        transform_idx,
    )
    return transformed.reshape(256).tolist()


def _normalize_policy_vector(policy: list[float]) -> list[float]:
    total = sum(policy)
    if total <= 1e-8:
        return policy
    return [float(v / total) for v in policy]


def _classify_engine_phase(board_size: int, occupied_cells: int) -> str:
    total_cells = board_size * board_size
    if board_size <= 5:
        if occupied_cells <= 3:
            return "opening"
        if occupied_cells <= 10:
            return "early_mid"
        if occupied_cells <= 16:
            return "mid"
        return "late"
    if occupied_cells <= max(3, total_cells // 8):
        return "opening"
    if occupied_cells <= max(10, total_cells // 3):
        return "mid"
    return "late"


def _resolve_engine_sampling_bounds(
    board_size: int,
    total_cells: int,
    *,
    phase_focus: str | None = None,
    rng_value: float | None = None,
) -> tuple[int, int]:
    focus = (phase_focus or "").strip().lower()
    if board_size <= 5:
        if focus in {"opening", "early"}:
            return 0, min(6, total_cells - 1)
        if focus in {"mid", "midgame"}:
            return min(9, total_cells - 1), min(16, total_cells - 1)
        if focus in {"late", "endgame"}:
            return min(17, total_cells - 1), total_cells - 1
    else:
        if focus in {"opening", "early"}:
            return 0, min(max(4, total_cells // 8), total_cells - 1)
        if focus in {"mid", "midgame"}:
            return min(max(5, total_cells // 5), total_cells - 1), min(max(18, (total_cells * 2) // 3), total_cells - 1)
        if focus in {"late", "endgame"}:
            return min(max(19, (total_cells * 2) // 3), total_cells - 1), total_cells - 1

    r = 0.5 if rng_value is None else rng_value
    if board_size <= 5:
        # Default mixed mode still prefers tactical positions, but leaves
        # enough midgame coverage for conversion quality to improve.
        if r < 0.05:
            return 0, min(3, total_cells - 1)
        if r < 0.20:
            return min(4, total_cells - 1), min(10, total_cells - 1)
        if r < 0.55:
            return min(11, total_cells - 1), min(16, total_cells - 1)
        return min(17, total_cells - 1), total_cells - 1

    if r < 0.20:
        return 0, min(3, total_cells - 1)
    if r < 0.60:
        return min(4, total_cells - 1), min(10, total_cells - 1)
    if r < 0.90:
        return min(11, total_cells - 1), min(18, total_cells - 1)
    return min(19, total_cells - 1), total_cells - 1


def _canonicalize_position(position: dict[str, Any]) -> dict[str, Any]:
    if "_canonicalFingerprint" in position:
        return position

    board_size = int(position.get("board_size", 0))
    board_tensor = torch.tensor(position.get("board") or [], dtype=torch.int16)
    if board_tensor.numel() == 0:
        canonical = dict(position)
        canonical["_canonicalFingerprint"] = _position_fingerprint(position)
        return canonical

    best_key: tuple[Any, ...] | None = None
    best_variant: dict[str, Any] | None = None
    policy = position.get("policy")
    current_player = int(position.get("current_player", 0))

    for transform_idx in range(8):
        board_variant = _transform_square_tensor(board_tensor, transform_idx)
        last_move_variant = _transform_last_move(position.get("last_move"), board_size, transform_idx)
        key = (
            board_size,
            tuple(int(v) for v in board_variant.reshape(-1).tolist()),
            current_player,
            tuple(last_move_variant or (-1, -1)),
        )
        if best_key is not None and key >= best_key:
            continue

        candidate = dict(position)
        candidate["board"] = board_variant.tolist()
        candidate["last_move"] = last_move_variant
        if policy is not None:
            candidate["policy"] = _transform_policy_vector(policy, board_size, transform_idx)
        candidate["_canonicalFingerprint"] = key
        best_key = key
        best_variant = candidate

    assert best_variant is not None
    return best_variant


def _position_bank_importance(position: dict[str, Any]) -> float:
    source = position.get("source", "")
    motif = position.get("motif", "")
    sample_weight = float(position.get("sampleWeight", 1.0))
    merge_count = float(position.get("mergeCount", 1))
    importance = sample_weight + min(math.log2(max(merge_count, 1.0) + 1.0) * 0.15, 0.75)
    if source == "failure":
        importance += 0.35
    elif source == "failure_conversion":
        importance += 0.55
    elif source == "failure_pure_gap":
        importance += 0.45
    elif source == "user_mistake":
        importance += 0.45
    elif source == "user_conversion":
        importance += 0.40
    elif source == "user_game":
        importance += 0.15
    elif source == "engine_conversion":
        importance += 0.35
    elif source == "engine_side_conversion":
        importance += 0.45
    elif source == "engine_side_focus":
        importance += 0.28
    elif source == "engine":
        importance += 0.20
    if motif in {"pure_missed_win", "pure_missed_block"}:
        importance += 0.45
    elif str(motif).startswith("exact_") or motif == "exact_trap":
        importance += 0.30
    elif motif in {"block", "win"}:
        importance += 0.20
    elif motif == "conversion" or bool(position.get("conversionTarget")):
        importance += 0.35
    if int(position.get("playerFocus", 0) or 0) in (1, 2):
        importance += 0.12
    return importance


def _merge_position_records(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    existing = _canonicalize_position(existing)
    incoming = _canonicalize_position(incoming)

    existing_count = float(existing.get("mergeCount", 1))
    incoming_count = float(incoming.get("mergeCount", 1))
    total_count = max(existing_count + incoming_count, 1.0)

    prefer_incoming = incoming.get("source") == "engine" and existing.get("source") != "engine"
    merged = dict(incoming if prefer_incoming else existing)

    existing_policy = existing.get("policy")
    incoming_policy = incoming.get("policy")
    if isinstance(existing_policy, list) and isinstance(incoming_policy, list):
        merged_policy = [
            ((float(a) * existing_count) + (float(b) * incoming_count)) / total_count
            for a, b in zip(existing_policy, incoming_policy)
        ]
        merged["policy"] = _normalize_policy_vector(merged_policy)

    existing_value = float(existing.get("value", 0.0))
    incoming_value = float(incoming.get("value", 0.0))
    merged["value"] = ((existing_value * existing_count) + (incoming_value * incoming_count)) / total_count
    merged["mergeCount"] = int(total_count)
    baseline_weight = min(1.0 + math.log2(total_count + 1.0) * 0.35, 3.0)
    merged["sampleWeight"] = round(max(
        baseline_weight,
        float(existing.get("sampleWeight", 1.0)),
        float(incoming.get("sampleWeight", 1.0)),
    ), 4)
    merged["_canonicalFingerprint"] = existing.get("_canonicalFingerprint") or incoming.get("_canonicalFingerprint")
    merged["_order"] = max(int(existing.get("_order", 0)), int(incoming.get("_order", 0)))
    if existing.get("source") == "engine" or incoming.get("source") == "engine":
        merged["source"] = "engine"
    elif existing.get("source") == "failure_conversion" or incoming.get("source") == "failure_conversion":
        merged["source"] = "failure_conversion"
    elif existing.get("source") == "failure_pure_gap" or incoming.get("source") == "failure_pure_gap":
        merged["source"] = "failure_pure_gap"
    motif_priority = {
        "pure_missed_win": 4,
        "pure_missed_block": 3,
        "conversion": 2,
        "pure_gap": 1,
    }
    existing_motif = str(existing.get("motif", "") or "")
    incoming_motif = str(incoming.get("motif", "") or "")
    merged_motif = max(
        (existing_motif, incoming_motif),
        key=lambda motif: motif_priority.get(motif, 0),
    )
    if motif_priority.get(merged_motif, 0) > 0:
        merged["motif"] = merged_motif
    if bool(existing.get("conversionTarget")) or bool(incoming.get("conversionTarget")):
        merged["conversionTarget"] = True
    merged["playerFocus"] = int(incoming.get("playerFocus", existing.get("playerFocus", 0)) or 0)
    return merged


def _build_pure_gap_relabel_candidate(
    state: dict[str, Any],
    hybrid_decision: dict[str, Any],
    pure_decision: dict[str, Any],
) -> dict[str, Any] | None:
    hybrid_move = int(hybrid_decision.get("move", -1) or -1)
    pure_move = int(pure_decision.get("move", -1) or -1)
    if hybrid_move < 0 or pure_move < 0 or hybrid_move == pure_move:
        return None

    hybrid_reason = str(hybrid_decision.get("tacticalReason", "") or "")
    hybrid_pressure = int(hybrid_decision.get("winningPressure", 0) or 0)
    hybrid_forcing = int(hybrid_decision.get("forcingThreatsAfterMove", 0) or 0)
    hybrid_search_score = float(hybrid_decision.get("searchScore", 0.0) or 0.0)
    hybrid_unsafe_filtered = int(hybrid_decision.get("unsafeMovesFiltered", 0) or 0)
    pure_reason = str(pure_decision.get("tacticalReason", "") or "")
    pure_pressure = int(pure_decision.get("winningPressure", 0) or 0)

    hybrid_is_conversion = (
        hybrid_reason in {"press_winning_advantage", "create_forcing_threat", "search_exact_win"}
        or hybrid_pressure > pure_pressure
        or hybrid_forcing >= 2
        or hybrid_search_score > 0.0
    )
    hybrid_is_tactical = hybrid_reason in {
        "immediate_win",
        "block_immediate",
        "reject_unsafe_model_move",
        "search_exact_hold",
        "search_exact_draw",
        "search_exact_survival",
        "least_bad_move",
    } or hybrid_unsafe_filtered > 0
    pure_is_quiet = pure_reason in {"model_policy", "policy_value"} and pure_pressure <= hybrid_pressure
    pure_gap = hybrid_move != pure_move and (pure_is_quiet or pure_reason != hybrid_reason)
    if not pure_gap or not (hybrid_is_conversion or hybrid_is_tactical):
        return None

    current_player = int(state.get("current_player", 0) or 0)
    motif = "conversion"
    source = "failure_conversion"
    conversion_target = True
    if hybrid_reason == "immediate_win":
        motif = "pure_missed_win"
        source = "failure_conversion"
        conversion_target = True
        sample_weight = 2.35
    elif hybrid_reason == "block_immediate":
        motif = "pure_missed_block"
        source = "failure_pure_gap"
        conversion_target = False
        sample_weight = 2.10
    elif hybrid_is_conversion:
        source = "failure_conversion"
        motif = "conversion"
        conversion_target = True
        sample_weight = 1.85
    else:
        source = "failure_pure_gap"
        motif = "pure_gap"
        conversion_target = False
        sample_weight = 1.70
    enriched = dict(state)
    enriched["source"] = source
    enriched["motif"] = motif
    enriched["sampleWeight"] = sample_weight
    enriched["playerFocus"] = current_player
    enriched["conversionTarget"] = conversion_target
    enriched["hybridMove"] = hybrid_move
    enriched["pureMove"] = pure_move
    enriched["hybridReason"] = hybrid_reason
    enriched["pureReason"] = pure_reason
    enriched["pureMissedWinInOne"] = motif == "pure_missed_win"
    enriched["pureMissedBlockInOne"] = motif == "pure_missed_block"
    return enriched


def _sanitize_bank_positions(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for pos in positions:
        cleaned = {k: v for k, v in pos.items() if not k.startswith("_")}
        sanitized.append(cleaned)
    return sanitized


def _merge_position_bank(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    max_size: int,
) -> list[dict[str, Any]]:
    merged_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    order = 0

    def ingest(raw_position: dict[str, Any]) -> None:
        nonlocal order
        canonical = _canonicalize_position(raw_position)
        order += 1
        canonical["_order"] = order
        key = canonical["_canonicalFingerprint"]
        if key in merged_by_key:
            merged_by_key[key] = _merge_position_records(merged_by_key[key], canonical)
            merged_by_key[key]["_order"] = order
        else:
            canonical.setdefault("mergeCount", 1)
            canonical.setdefault("sampleWeight", 1.0)
            merged_by_key[key] = canonical

    for position in existing:
        ingest(position)
    for position in incoming:
        ingest(position)

    merged = list(merged_by_key.values())
    if len(merged) > max_size:
        merged.sort(key=lambda pos: (_position_bank_importance(pos), int(pos.get("_order", 0))))
        merged = merged[-max_size:]
    merged.sort(key=lambda pos: int(pos.get("_order", 0)))
    return _sanitize_bank_positions(merged)


def _split_holdout_positions(
    positions: list[dict[str, Any]],
    *,
    holdout_ratio: float = 0.10,
    max_holdout: int = 64,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(positions) < 8 or holdout_ratio <= 0:
        return list(positions), []

    holdout_count = min(max(int(round(len(positions) * holdout_ratio)), 1), max_holdout, len(positions) // 4)
    holdout_indices = set(random.sample(range(len(positions)), holdout_count))
    train_split: list[dict[str, Any]] = []
    holdout_split: list[dict[str, Any]] = []
    for idx, position in enumerate(positions):
        if idx in holdout_indices:
            held = dict(position)
            held["source"] = held.get("source", "holdout")
            holdout_split.append(held)
        else:
            train_split.append(position)
    return train_split, holdout_split


def _load_offline_dataset_positions(
    variant: str,
    *,
    max_positions: int | None = None,
    preferred_backend: str | None = None,
) -> tuple[list[dict[str, Any]], Path | None, str | None]:
    datasets_dir = SAVED_DIR / "datasets"
    dataset_paths = {
        "rapfi": datasets_dir / f"{variant}_rapfi.json",
        "engine": datasets_dir / f"{variant}_engine.json",
        "minimax": datasets_dir / f"{variant}_minimax.json",
    }
    backend_aliases = {
        "builtin": "engine",
        "engine": "engine",
        "cpp": "engine",
        "rapfi": "rapfi",
        "minimax": "minimax",
    }
    preferred_type = backend_aliases.get(str(preferred_backend or "").strip().lower())
    ordered_types: list[str] = []
    if preferred_type is not None:
        ordered_types.append(preferred_type)
    for dataset_type in ("rapfi", "engine", "minimax"):
        if dataset_type not in ordered_types:
            ordered_types.append(dataset_type)
    candidates = [(dataset_type, dataset_paths[dataset_type]) for dataset_type in ordered_types]
    for dataset_type, path in candidates:
        if not path.exists():
            continue
        try:
            positions = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load offline dataset %s: %s", path, exc)
            continue
        if max_positions is not None and max_positions > 0:
            positions = positions[:max_positions]
        return positions, path, dataset_type
    return [], None, None


def _compute_target_sanity_metrics(positions: list[dict[str, Any]]) -> dict[str, float]:
    from trainer_lab.data.encoder import board_to_tensor

    if not positions:
        return {}

    policy_sum_error = 0.0
    legal_target_hits = 0
    non_finite_targets = 0
    unique_fingerprints: set[tuple[Any, ...]] = set()

    for position in positions:
        policy = position.get("policy") or []
        if not policy:
            non_finite_targets += 1
            continue

        policy_sum_error += abs(float(sum(policy)) - 1.0)
        if any(not math.isfinite(float(value)) for value in policy):
            non_finite_targets += 1
            continue

        tensor = board_to_tensor(position)
        legal_mask = tensor[2].reshape(-1)
        target_idx = max(range(len(policy)), key=policy.__getitem__)
        if target_idx < legal_mask.numel() and float(legal_mask[target_idx].item()) > 0.5:
            legal_target_hits += 1

        canonical = _canonicalize_position(position)
        fingerprint = canonical.get("_canonicalFingerprint")
        if isinstance(fingerprint, tuple):
            unique_fingerprints.add(fingerprint)

    total = max(len(positions), 1)
    unique = len(unique_fingerprints) or total
    return {
        "policyMassMeanAbsError": round(policy_sum_error / total, 6),
        "legalTargetRate": round((legal_target_hits / total) * 100.0, 2),
        "nonFiniteTargetRate": round((non_finite_targets / total) * 100.0, 2),
        "uniqueCanonicalPositions": float(unique),
        "duplicateMergeRate": round(max(0.0, 1.0 - (unique / total)), 4),
    }


def _sample_positions(positions: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not positions:
        return []
    if len(positions) <= count:
        return list(positions)
    weights = torch.tensor([
        max(float(position.get("sampleWeight", 1.0)), 0.05) * (1.0 + 0.30 * ((idx + 1) / max(len(positions), 1)))
        for idx, position in enumerate(positions)
    ], dtype=torch.float32)
    picks = torch.multinomial(weights, num_samples=count, replacement=False).tolist()
    return [positions[idx] for idx in picks]


def _position_focus_side(position: dict[str, Any]) -> int:
    focus = int(position.get("playerFocus", 0) or 0)
    if focus in (1, 2):
        return focus
    current = int(position.get("current_player", 0) or 0)
    if current in (1, 2):
        return current
    return 0


def _sample_positions_balanced(
    positions: list[dict[str, Any]],
    count: int,
    *,
    focus_player: int | None = None,
    focus_ratio: float = 0.0,
) -> list[dict[str, Any]]:
    if count <= 0 or not positions:
        return []
    if focus_player not in (1, 2) or focus_ratio <= 0.0:
        return _sample_positions(positions, count)

    clamped_ratio = max(0.0, min(float(focus_ratio), 1.0))
    focus_positions = [pos for pos in positions if _position_focus_side(pos) == int(focus_player)]
    if not focus_positions:
        return _sample_positions(positions, count)

    focus_quota = min(len(focus_positions), max(int(round(count * clamped_ratio)), 0))
    focused = _sample_positions(focus_positions, focus_quota) if focus_quota > 0 else []
    if len(focused) >= count:
        return focused[:count]

    focused_ids = {id(pos) for pos in focused}
    remaining_positions = [pos for pos in positions if id(pos) not in focused_ids]
    remaining_needed = count - len(focused)
    remainder = _sample_positions(remaining_positions, remaining_needed) if remaining_positions else []
    combined = focused + remainder
    if len(combined) < count:
        combined.extend(_sample_positions(positions, min(count - len(combined), len(positions))))
    return combined[:count]


def _is_conversion_training_position(position: dict[str, Any]) -> bool:
    source = str(position.get("source", "") or "")
    motif = str(position.get("motif", "") or "")
    return (
        bool(position.get("conversionTarget"))
        or source in {"failure_conversion", "engine_conversion", "engine_side_conversion", "user_conversion"}
        or motif in {"conversion", "pure_missed_win"}
    )


def _sample_focus_conversion_positions(
    positions: list[dict[str, Any]],
    count: int,
    *,
    focus_player: int | None = None,
) -> list[dict[str, Any]]:
    if count <= 0 or not positions or focus_player not in (1, 2):
        return []
    candidates = [
        pos
        for pos in positions
        if _position_focus_side(pos) == int(focus_player) and _is_conversion_training_position(pos)
    ]
    if not candidates:
        return []
    return _sample_positions(candidates, count)


def _sample_counter_conversion_positions(
    positions: list[dict[str, Any]],
    count: int,
    *,
    focus_player: int | None = None,
) -> list[dict[str, Any]]:
    if focus_player not in (1, 2):
        return []
    counter_player = 1 if int(focus_player) == 2 else 2
    return _sample_focus_conversion_positions(positions, count, focus_player=counter_player)


def _merge_failure_bank(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    max_size: int,
) -> list[dict[str, Any]]:
    tagged_existing = [{**pos, "source": pos.get("source", "failure")} for pos in existing]
    tagged_incoming = [{**pos, "source": pos.get("source", "failure")} for pos in incoming]
    return _merge_position_bank(tagged_existing, tagged_incoming, max_size=max_size)


def _build_repair_pool(
    failure_bank: list[dict[str, Any]],
    anchor_positions: list[dict[str, Any]],
    tactical_positions: list[dict[str, Any]],
    user_corpus_positions: list[dict[str, Any]] | None = None,
    *,
    data_count: int,
    focus_player: int | None = None,
    focus_ratio: float = 0.0,
    focus_conversion_ratio: float = 0.0,
    counter_conversion_ratio: float = 0.0,
) -> list[dict[str, Any]]:
    """Prioritize recent mistakes while keeping anchor/tactical stability."""
    if data_count <= 0:
        return []

    user_corpus_positions = list(user_corpus_positions or [])
    pool: list[dict[str, Any]] = []
    if focus_player in (1, 2) and focus_conversion_ratio > 0.0:
        conversion_quota = max(int(round(data_count * max(0.0, min(float(focus_conversion_ratio), 1.0)))), 0)
        conversion_sources = failure_bank + anchor_positions + user_corpus_positions
        pool.extend(
            _sample_focus_conversion_positions(
                conversion_sources,
                conversion_quota,
                focus_player=focus_player,
            )
        )
    if focus_player in (1, 2) and counter_conversion_ratio > 0.0:
        counter_quota = max(int(round(data_count * max(0.0, min(float(counter_conversion_ratio), 1.0)))), 0)
        conversion_sources = failure_bank + anchor_positions + user_corpus_positions
        pool.extend(
            _sample_counter_conversion_positions(
                conversion_sources,
                counter_quota,
                focus_player=focus_player,
            )
        )

    remaining_count = max(data_count - len(pool), 0)
    if user_corpus_positions:
        fail_quota = min(len(failure_bank), max(int(remaining_count * 0.55), 0))
        anchor_quota = min(len(anchor_positions), max(int(remaining_count * 0.20), 0))
        tactical_quota = min(len(tactical_positions), max(int(remaining_count * 0.10), 0))
        user_quota = min(len(user_corpus_positions), max(remaining_count - fail_quota - anchor_quota - tactical_quota, 0))
    else:
        fail_quota = min(len(failure_bank), max(int(remaining_count * 0.65), 0))
        anchor_quota = min(len(anchor_positions), max(int(remaining_count * 0.25), 0))
        tactical_quota = min(len(tactical_positions), max(remaining_count - fail_quota - anchor_quota, 0))
        user_quota = 0

    if fail_quota > 0:
        pool.extend(_sample_positions_balanced(failure_bank, fail_quota, focus_player=focus_player, focus_ratio=focus_ratio))
    if anchor_quota > 0:
        pool.extend(_sample_positions_balanced(anchor_positions, anchor_quota, focus_player=focus_player, focus_ratio=focus_ratio))
    if tactical_quota > 0:
        pool.extend(_sample_positions_balanced(tactical_positions, tactical_quota, focus_player=focus_player, focus_ratio=focus_ratio))
    if user_quota > 0:
        pool.extend(_sample_positions_balanced(user_corpus_positions, user_quota, focus_player=focus_player, focus_ratio=focus_ratio))

    if len(pool) < data_count:
        leftovers = failure_bank + anchor_positions + tactical_positions + user_corpus_positions
        needed = min(data_count - len(pool), len(leftovers))
        if needed > 0:
            pool.extend(_sample_positions_balanced(leftovers, needed, focus_player=focus_player, focus_ratio=focus_ratio))

    random.shuffle(pool)
    return pool[:data_count]


def _build_turbo_pool(
    anchor_positions: list[dict[str, Any]],
    tactical_positions: list[dict[str, Any]],
    failure_bank: list[dict[str, Any]],
    user_corpus_positions: list[dict[str, Any]] | None = None,
    *,
    data_count: int,
    tactical_ratio: float = 0.20,
    focus_player: int | None = None,
    focus_ratio: float = 0.0,
    focus_conversion_ratio: float = 0.0,
    counter_conversion_ratio: float = 0.0,
) -> list[dict[str, Any]]:
    """Anchor-heavy pool for high-throughput short training bursts."""
    if data_count <= 0:
        return []

    user_corpus_positions = list(user_corpus_positions or [])
    pool: list[dict[str, Any]] = []
    if focus_player in (1, 2) and focus_conversion_ratio > 0.0:
        conversion_quota = max(int(round(data_count * max(0.0, min(float(focus_conversion_ratio), 1.0)))), 0)
        conversion_sources = anchor_positions + failure_bank + user_corpus_positions
        pool.extend(
            _sample_focus_conversion_positions(
                conversion_sources,
                conversion_quota,
                focus_player=focus_player,
            )
        )
    if focus_player in (1, 2) and counter_conversion_ratio > 0.0:
        counter_quota = max(int(round(data_count * max(0.0, min(float(counter_conversion_ratio), 1.0)))), 0)
        conversion_sources = anchor_positions + failure_bank + user_corpus_positions
        pool.extend(
            _sample_counter_conversion_positions(
                conversion_sources,
                counter_quota,
                focus_player=focus_player,
            )
        )

    remaining_count = max(data_count - len(pool), 0)
    user_ratio = 0.10 if user_corpus_positions else 0.0
    failure_ratio = 0.10 if user_corpus_positions else 0.20
    anchor_ratio = max(0.10, 1.0 - tactical_ratio - failure_ratio - user_ratio)
    anchor_quota = min(len(anchor_positions), max(int(remaining_count * anchor_ratio), 0))
    tactical_quota = min(len(tactical_positions), max(int(remaining_count * tactical_ratio), 0))
    failure_quota = min(len(failure_bank), max(int(remaining_count * failure_ratio), 0))
    user_quota = min(len(user_corpus_positions), max(remaining_count - anchor_quota - tactical_quota - failure_quota, 0))

    if anchor_quota > 0:
        pool.extend(_sample_positions_balanced(anchor_positions, anchor_quota, focus_player=focus_player, focus_ratio=focus_ratio))
    if tactical_quota > 0:
        pool.extend(_sample_positions_balanced(tactical_positions, tactical_quota, focus_player=focus_player, focus_ratio=focus_ratio))
    if failure_quota > 0:
        pool.extend(_sample_positions_balanced(failure_bank, failure_quota, focus_player=focus_player, focus_ratio=focus_ratio))
    if user_quota > 0:
        pool.extend(_sample_positions_balanced(user_corpus_positions, user_quota, focus_player=focus_player, focus_ratio=focus_ratio))

    if len(pool) < data_count:
        leftovers = anchor_positions + tactical_positions + failure_bank + user_corpus_positions
        needed = min(data_count - len(pool), len(leftovers))
        if needed > 0:
            pool.extend(_sample_positions_balanced(leftovers, needed, focus_player=focus_player, focus_ratio=focus_ratio))

    random.shuffle(pool)
    return pool[:data_count]


def _selfplay_replay_path(variant: str) -> Path:
    return _ensure_saved_dir(variant) / "self_play_mixed_replay.json"


def _selfplay_mixed_source_weights(iteration: int, total_iterations: int) -> dict[str, float]:
    """Blend replay sources for the AlphaZero-style self-play phase.

    We keep a very small stabilizing teacher/failure/user contribution, but
    once self-play has warmed up the learner should train predominantly on
    visit-count targets coming from self-play itself.
    """
    total = max(int(total_iterations), 1)
    progress = max(0.0, min((float(iteration) - 1.0) / max(total - 1, 1), 1.0))

    # AlphaZero-like shape: self-play dominates, while anchors/tactical/failure
    # act only as a light stabilizer to avoid forgetting hard traps too early.
    self_play_weight = 0.58 + 0.22 * progress
    anchor_weight = 0.18 - 0.10 * progress
    tactical_weight = 0.12 - 0.05 * progress
    failure_weight = 0.08 - 0.03 * progress
    user_weight = 1.0 - (self_play_weight + anchor_weight + tactical_weight + failure_weight)

    # Keep every auxiliary source alive, but intentionally small.
    anchor_weight = max(anchor_weight, 0.06)
    tactical_weight = max(tactical_weight, 0.05)
    failure_weight = max(failure_weight, 0.03)
    user_weight = max(user_weight, 0.04)

    # Renormalize after floors to keep exact sum 1.0.
    total_weight = self_play_weight + anchor_weight + tactical_weight + failure_weight + user_weight
    self_play_weight /= total_weight
    anchor_weight /= total_weight
    tactical_weight /= total_weight
    failure_weight /= total_weight
    user_weight /= total_weight
    return {
        "anchor": round(anchor_weight, 4),
        "tactical": round(tactical_weight, 4),
        "failure": round(failure_weight, 4),
        "user": round(user_weight, 4),
        "self_play": round(self_play_weight, 4),
    }


def _choose_rapid_cycle_strategy(
    validation_payload: dict[str, Any],
    *,
    corrected_rate: float,
    failure_bank_size: int,
    engine_per_cycle: int,
    exam_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    strategy = {
        "tacticalRatio": 0.50,
        "tacticalFocus": None,
        "failureSlice": 256,
        "engineFocus": None,
        "engineCount": engine_per_cycle,
        "engineCurrentPlayerFocus": None,
        "playerFocusRatio": 0.0,
        "focusConversionRatio": 0.0,
        "counterConversionRatio": 0.0,
        "conversionFocus": False,
        "weakestFrozenSuite": None,
        "midLateGap": None,
    }

    block_bench = validation_payload.get("frozenBlockAcc")
    win_bench = validation_payload.get("frozenWinAcc")
    mid_bench = validation_payload.get("frozenMidAcc")
    late_bench = validation_payload.get("frozenLateAcc")
    pure_block_recall = validation_payload.get("pureFrozenBlockRecall")
    pure_win_recall = validation_payload.get("pureFrozenWinRecall")
    pure_exact_recall = validation_payload.get("pureExactTrapRecall")
    pure_p2_trap_recall = validation_payload.get("pureP2TrapRecall")
    pure_worst_trap_family_recall = validation_payload.get("pureWorstTrapFamilyRecall")
    holdout_delta = float(validation_payload.get("holdoutDeltaAcc", 0.0) or 0.0)

    suites = [
        ("block", block_bench),
        ("win", win_bench),
        ("mid", mid_bench),
        ("late", late_bench),
    ]
    available_suites = [(name, float(value)) for name, value in suites if value is not None]
    if available_suites:
        weakest_name, _ = min(available_suites, key=lambda item: item[1])
        strategy["weakestFrozenSuite"] = weakest_name

    if mid_bench is not None and late_bench is not None:
        strategy["midLateGap"] = round(float(late_bench) - float(mid_bench), 2)

    if block_bench is not None and win_bench is not None:
        if float(block_bench) + 4.0 < float(win_bench):
            strategy["tacticalFocus"] = "block"
            strategy["tacticalRatio"] = 0.60
        elif float(win_bench) + 6.0 < float(block_bench):
            strategy["tacticalFocus"] = "win"
            strategy["tacticalRatio"] = 0.55

    if pure_block_recall is not None and pure_win_recall is not None:
        if float(pure_block_recall) + 6.0 < float(pure_win_recall):
            strategy["tacticalFocus"] = "block"
            strategy["tacticalRatio"] = max(float(strategy["tacticalRatio"]), 0.65)
            strategy["failureSlice"] = max(int(strategy["failureSlice"]), 320)
        elif float(pure_win_recall) + 6.0 < float(pure_block_recall):
            strategy["tacticalFocus"] = "win"
            strategy["tacticalRatio"] = max(float(strategy["tacticalRatio"]), 0.65)
            strategy["failureSlice"] = max(int(strategy["failureSlice"]), 320)
            strategy["conversionFocus"] = True
        if float(pure_win_recall) < 20.0:
            strategy["tacticalFocus"] = "win"
            strategy["tacticalRatio"] = max(float(strategy["tacticalRatio"]), 0.75)
            strategy["failureSlice"] = max(int(strategy["failureSlice"]), 448)
            strategy["conversionFocus"] = True
        if float(pure_block_recall) < 20.0:
            strategy["tacticalFocus"] = "block"
            strategy["tacticalRatio"] = max(float(strategy["tacticalRatio"]), 0.75)
            strategy["failureSlice"] = max(int(strategy["failureSlice"]), 448)
    if pure_exact_recall is not None:
        exact_recall = float(pure_exact_recall)
        if exact_recall < 85.0:
            strategy["tacticalRatio"] = max(float(strategy["tacticalRatio"]), 0.65)
            strategy["failureSlice"] = max(int(strategy["failureSlice"]), 384)
        if exact_recall < 70.0:
            strategy["conversionFocus"] = True
            if strategy["engineFocus"] is None:
                strategy["engineFocus"] = "mid"
    if pure_worst_trap_family_recall is not None:
        worst_family_recall = float(pure_worst_trap_family_recall)
        if worst_family_recall < 80.0:
            strategy["tacticalRatio"] = max(float(strategy["tacticalRatio"]), 0.68)
            strategy["failureSlice"] = max(int(strategy["failureSlice"]), 416)
        if worst_family_recall < 65.0:
            strategy["conversionFocus"] = True
            if strategy["engineFocus"] is None:
                strategy["engineFocus"] = "mid"
    if pure_p2_trap_recall is not None and float(pure_p2_trap_recall) < 85.0:
        strategy["engineCurrentPlayerFocus"] = 2
        strategy["playerFocusRatio"] = max(float(strategy["playerFocusRatio"]), 0.45)
        strategy["failureSlice"] = max(int(strategy["failureSlice"]), 384)
        strategy["tacticalRatio"] = max(float(strategy["tacticalRatio"]), 0.65)

    # Midgame conversion is the current bottleneck once tactical suites are strong.
    if mid_bench is not None and (late_bench is None or float(mid_bench) + 8.0 < float(late_bench)):
        strategy["engineFocus"] = "mid"
        strategy["engineCount"] = min(
            128,
            max(engine_per_cycle + 24, int(round(engine_per_cycle * 1.75))),
        )
        strategy["tacticalRatio"] = min(float(strategy["tacticalRatio"]), 0.35)
    elif late_bench is not None and float(late_bench) < 60.0:
        strategy["engineFocus"] = "late"
        strategy["engineCount"] = min(
            112,
            max(engine_per_cycle + 16, int(round(engine_per_cycle * 1.5))),
        )
        strategy["tacticalRatio"] = min(float(strategy["tacticalRatio"]), 0.40)

    if holdout_delta < -1.0:
        strategy["tacticalRatio"] = max(0.35, float(strategy["tacticalRatio"]) - 0.05)
        strategy["engineCount"] = min(128, int(strategy["engineCount"]) + 16)

    if exam_summary:
        winrate_as_p1 = float(exam_summary.get("winrateAsP1", 0.0) or 0.0)
        winrate_as_p2 = float(exam_summary.get("winrateAsP2", 0.0) or 0.0)
        decisive_winrate = float(exam_summary.get("decisiveWinRate", 0.0) or 0.0)
        draw_rate = float(exam_summary.get("drawRate", 0.0) or 0.0)
        pure_gap_rate = float(exam_summary.get("pureGapRate", 0.0) or 0.0)
        pure_gap_rate_as_p1 = float(exam_summary.get("pureGapRateAsP1", 0.0) or 0.0)
        pure_gap_rate_as_p2 = float(exam_summary.get("pureGapRateAsP2", 0.0) or 0.0)
        conversion_failures_as_p1 = int(exam_summary.get("conversionFailuresAsP1", 0) or 0)
        conversion_failures_as_p2 = int(exam_summary.get("conversionFailuresAsP2", 0) or 0)
        pure_missed_win_count = int(exam_summary.get("pureMissedWinCount", 0) or 0)
        pure_missed_block_count = int(exam_summary.get("pureMissedBlockCount", 0) or 0)
        balanced_side = float(exam_summary.get("balancedSideWinrate", min(winrate_as_p1, winrate_as_p2)) or 0.0)

        if abs(winrate_as_p1 - winrate_as_p2) >= 0.10:
            strategy["engineCurrentPlayerFocus"] = 1 if winrate_as_p1 < winrate_as_p2 else 2
            side_gap = abs(winrate_as_p1 - winrate_as_p2)
            if side_gap >= 0.50:
                strategy["playerFocusRatio"] = max(float(strategy["playerFocusRatio"]), 0.60)
                strategy["focusConversionRatio"] = max(float(strategy["focusConversionRatio"]), 0.18)
            elif side_gap >= 0.25:
                strategy["playerFocusRatio"] = max(float(strategy["playerFocusRatio"]), 0.45)
                strategy["focusConversionRatio"] = max(float(strategy["focusConversionRatio"]), 0.12)
            else:
                strategy["playerFocusRatio"] = max(float(strategy["playerFocusRatio"]), 0.30)
            strategy["engineCount"] = min(144, max(int(strategy["engineCount"]), engine_per_cycle + 12))

        if decisive_winrate < 0.20 or draw_rate > 0.40:
            strategy["conversionFocus"] = True
            strategy["engineCount"] = min(160, max(int(strategy["engineCount"]), engine_per_cycle + 24))
            strategy["tacticalRatio"] = min(float(strategy["tacticalRatio"]), 0.32)
            if strategy["engineFocus"] is None:
                strategy["engineFocus"] = "mid"
            if strategy["engineCurrentPlayerFocus"] in (1, 2):
                strategy["focusConversionRatio"] = max(float(strategy["focusConversionRatio"]), 0.18)
        if balanced_side < 0.25:
            strategy["conversionFocus"] = True
            strategy["failureSlice"] = max(int(strategy["failureSlice"]), 384)
            if strategy["engineFocus"] is None:
                strategy["engineFocus"] = "mid"
            strategy["engineCount"] = min(160, max(int(strategy["engineCount"]), engine_per_cycle + 24))
            if strategy["engineCurrentPlayerFocus"] in (1, 2):
                strategy["focusConversionRatio"] = max(float(strategy["focusConversionRatio"]), 0.24)
        if winrate_as_p2 <= 0.05:
            strategy["engineCurrentPlayerFocus"] = 2
            strategy["conversionFocus"] = True
            strategy["failureSlice"] = max(int(strategy["failureSlice"]), 448)
            strategy["playerFocusRatio"] = max(float(strategy["playerFocusRatio"]), 0.70)
            strategy["focusConversionRatio"] = max(float(strategy["focusConversionRatio"]), 0.32)
            strategy["counterConversionRatio"] = max(float(strategy["counterConversionRatio"]), 0.10)
            strategy["engineCount"] = min(160, max(int(strategy["engineCount"]), engine_per_cycle + 32))
            strategy["tacticalRatio"] = max(float(strategy["tacticalRatio"]), 0.42)
            if strategy["engineFocus"] is None:
                strategy["engineFocus"] = "mid"
        elif winrate_as_p1 <= 0.05:
            strategy["engineCurrentPlayerFocus"] = 1
            strategy["conversionFocus"] = True
            strategy["failureSlice"] = max(int(strategy["failureSlice"]), 448)
            strategy["playerFocusRatio"] = max(float(strategy["playerFocusRatio"]), 0.70)
            strategy["focusConversionRatio"] = max(float(strategy["focusConversionRatio"]), 0.32)
            strategy["counterConversionRatio"] = max(float(strategy["counterConversionRatio"]), 0.10)
            strategy["engineCount"] = min(160, max(int(strategy["engineCount"]), engine_per_cycle + 32))
            strategy["tacticalRatio"] = max(float(strategy["tacticalRatio"]), 0.42)
            if strategy["engineFocus"] is None:
                strategy["engineFocus"] = "mid"

        if pure_gap_rate > 0.25:
            strategy["failureSlice"] = 384
            strategy["engineCount"] = min(160, max(int(strategy["engineCount"]), engine_per_cycle + 16))
        if pure_gap_rate_as_p1 > pure_gap_rate_as_p2 + 0.05:
            strategy["engineCurrentPlayerFocus"] = 1
            strategy["conversionFocus"] = True
            strategy["playerFocusRatio"] = max(float(strategy["playerFocusRatio"]), 0.45)
            strategy["focusConversionRatio"] = max(float(strategy["focusConversionRatio"]), 0.18)
        elif pure_gap_rate_as_p2 > pure_gap_rate_as_p1 + 0.05:
            strategy["engineCurrentPlayerFocus"] = 2
            strategy["conversionFocus"] = True
            strategy["playerFocusRatio"] = max(float(strategy["playerFocusRatio"]), 0.45)
            strategy["focusConversionRatio"] = max(float(strategy["focusConversionRatio"]), 0.18)
        if pure_missed_win_count >= 6 or pure_missed_block_count >= 6:
            strategy["tacticalRatio"] = max(float(strategy["tacticalRatio"]), 0.60)
            if pure_missed_win_count >= pure_missed_block_count:
                strategy["tacticalFocus"] = "win"
            else:
                strategy["tacticalFocus"] = "block"
            strategy["failureSlice"] = 384
        if pure_missed_win_count >= 8:
            strategy["conversionFocus"] = True
            if strategy["engineFocus"] is None:
                strategy["engineFocus"] = "mid"
            if strategy["engineCurrentPlayerFocus"] in (1, 2):
                strategy["playerFocusRatio"] = max(float(strategy["playerFocusRatio"]), 0.50)
                strategy["focusConversionRatio"] = max(float(strategy["focusConversionRatio"]), 0.22)
        if abs(conversion_failures_as_p1 - conversion_failures_as_p2) >= 4:
            weaker_side = 1 if conversion_failures_as_p1 > conversion_failures_as_p2 else 2
            strategy["engineCurrentPlayerFocus"] = weaker_side
            strategy["conversionFocus"] = True
            strategy["playerFocusRatio"] = max(float(strategy["playerFocusRatio"]), 0.55)
            strategy["focusConversionRatio"] = max(float(strategy["focusConversionRatio"]), 0.30)
            strategy["failureSlice"] = max(int(strategy["failureSlice"]), 384)
            if strategy["engineFocus"] is None:
                strategy["engineFocus"] = "mid"

        if strategy["engineCurrentPlayerFocus"] in (1, 2) and float(strategy["focusConversionRatio"]) > 0.0:
            strategy["counterConversionRatio"] = max(
                float(strategy["counterConversionRatio"]),
                min(float(strategy["focusConversionRatio"]) * 0.45, 0.12),
            )

    if corrected_rate < 0.20 and failure_bank_size > 8:
        strategy["failureSlice"] = 384
    elif corrected_rate > 0.45:
        strategy["failureSlice"] = 192

    return strategy


_BATCH_D4_TRANSFORMS = [
    lambda p: p,                                          # identity
    lambda p: torch.rot90(p, 1, [2, 3]),                  # rot90
    lambda p: torch.rot90(p, 2, [2, 3]),                  # rot180
    lambda p: torch.rot90(p, 3, [2, 3]),                  # rot270
    lambda p: p.flip(2),                                  # mirror_v
    lambda p: p.flip(3),                                  # mirror_h
    lambda p: p.transpose(2, 3),                          # diag_main
    lambda p: torch.rot90(p, 1, [2, 3]).flip(2),          # diag_anti
]


def _apply_random_d4_batch(
    planes: torch.Tensor,
    policy: torch.Tensor,
    board_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ONE random D4 transform to entire batch (planes + policy synced).

    For boards smaller than 16×16, operates only on the board_size×board_size
    subgrid to avoid rotating content outside the legal area.
    """
    fn = random.choice(_BATCH_D4_TRANSFORMS)
    if board_size >= 16:
        planes = fn(planes).contiguous()
        pol_grid = fn(policy.reshape(-1, 1, 16, 16)).contiguous()
        return planes, pol_grid.reshape(-1, 256)

    # Extract subgrid, transform, place back
    bs = board_size
    sub_planes = planes[:, :, :bs, :bs].contiguous()
    sub_planes = fn(sub_planes).contiguous()
    new_planes = torch.zeros_like(planes)
    new_planes[:, :, :bs, :bs] = sub_planes

    pol_grid = policy.reshape(-1, 1, 16, 16)
    sub_pol = pol_grid[:, :, :bs, :bs].contiguous()
    sub_pol = fn(sub_pol).contiguous()
    new_pol = torch.zeros_like(pol_grid)
    new_pol[:, :, :bs, :bs] = sub_pol

    return new_planes.contiguous(), new_pol.reshape(-1, 256)


def _materialize_training_tensors(
    positions: list[dict[str, Any]],
    *,
    augment: bool = False,
    augment_mode: str = "full",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    from trainer_lab.data.augmentation import augment_sample
    from trainer_lab.data.encoder import board_to_tensor

    if not positions:
        empty_planes = torch.empty((0, 6, 16, 16), dtype=torch.float32)
        empty_policy = torch.empty((0, 256), dtype=torch.float32)
        empty_value = torch.empty((0, 1), dtype=torch.float32)
        return empty_planes, empty_policy, empty_value, 0

    planes_list: list[torch.Tensor] = []
    policy_list: list[torch.Tensor] = []
    value_list: list[torch.Tensor] = []

    do_full_augment = augment and augment_mode == "full"

    for pos in positions:
        planes = board_to_tensor(pos)
        policy = torch.tensor(pos["policy"], dtype=torch.float32)
        value_scalar = torch.tensor(float(pos["value"]), dtype=torch.float32)

        if do_full_augment:
            _bs = pos.get("board_size", 16)
            for aug_planes, aug_policy, aug_value in augment_sample(planes, policy, value_scalar, board_size=_bs):
                planes_list.append(aug_planes)
                policy_list.append(aug_policy)
                value_list.append(aug_value.view(1))
        else:
            planes_list.append(planes)
            policy_list.append(policy)
            value_list.append(value_scalar.view(1))

    return (
        torch.stack(planes_list),
        torch.stack(policy_list),
        torch.stack(value_list),
        len(planes_list),
    )


def _score_policy_matches(
    model: Any,
    positions: list[dict[str, Any]],
    device: Any,
    *,
    batch_size: int = 256,
) -> list[bool]:
    from trainer_lab.data.encoder import board_to_tensor

    if not positions:
        return []

    encoded = torch.stack([board_to_tensor(p) for p in positions])
    targets = torch.tensor([
        max(range(len(p["policy"])), key=p["policy"].__getitem__)
        for p in positions
    ], dtype=torch.long)

    results: list[bool] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, encoded.size(0), batch_size):
            planes = encoded[start : start + batch_size].to(device, non_blocking=device.type == "cuda")
            if device.type == "cuda":
                planes = planes.contiguous(memory_format=torch.channels_last)
            logits, _ = model(planes)
            legal_mask = planes[:, 2].reshape(planes.size(0), -1)
            masked = logits + (1.0 - legal_mask) * (-1e8)
            preds = masked.argmax(dim=1).cpu()
            target_slice = targets[start : start + batch_size]
            results.extend((preds == target_slice).tolist())
    model.train()
    return results


def _evaluate_supervised_dataset(
    model: Any,
    positions: list[dict[str, Any]],
    device: Any,
    *,
    batch_size: int = 256,
) -> dict[str, float]:
    from trainer_lab.training.metrics import (
        policy_accuracy,
        policy_kl_divergence,
        teacher_mass_on_pred,
        value_mae,
        value_sign_agreement,
    )

    if not positions:
        return {}

    target_sanity = _compute_target_sanity_metrics(positions)
    planes, policy, value, _ = _materialize_training_tensors(positions, augment=False)
    if device.type == "cuda":
        planes = planes.pin_memory()
        policy = policy.pin_memory()
        value = value.pin_memory()

    model.eval()
    total = 0
    acc = tmass = pkl = vmae = vsign = 0.0
    with torch.inference_mode():
        for start in range(0, planes.size(0), batch_size):
            batch_planes = planes[start : start + batch_size].to(device, non_blocking=device.type == "cuda")
            batch_policy = policy[start : start + batch_size].to(device, non_blocking=device.type == "cuda")
            batch_value = value[start : start + batch_size].to(device, non_blocking=device.type == "cuda")
            if device.type == "cuda":
                batch_planes = batch_planes.contiguous(memory_format=torch.channels_last)
            logits, vpred = model(batch_planes)
            legal_mask = batch_planes[:, 2].reshape(batch_planes.size(0), -1)
            bs = batch_planes.size(0)
            acc += policy_accuracy(logits, batch_policy, legal_mask=legal_mask) * bs
            tmass += teacher_mass_on_pred(logits, batch_policy, legal_mask=legal_mask) * bs
            pkl += policy_kl_divergence(logits, batch_policy, legal_mask=legal_mask) * bs
            vmae += value_mae(vpred, batch_value) * bs
            vsign += value_sign_agreement(vpred, batch_value) * bs
            total += bs
    model.train()

    return {
        "policyTop1Acc": round((acc / max(total, 1)) * 100.0, 2),
        "teacherMassOnPred": round(tmass / max(total, 1), 4),
        "policyKL": round(pkl / max(total, 1), 6),
        "valueMAE": round(vmae / max(total, 1), 6),
        "valueSignAgreement": round((vsign / max(total, 1)) * 100.0, 2),
        **target_sanity,
    }


def _sample_engine_position(
    board_size: int,
    win_len: int,
    *,
    rng: random.Random | None = None,
    phase_focus: str | None = None,
) -> tuple[list[int], int, int] | None:
    rng = rng or random
    total_cells = board_size * board_size
    low, high = _resolve_engine_sampling_bounds(
        board_size,
        total_cells,
        phase_focus=phase_focus,
        rng_value=rng.random(),
    )

    if high < low:
        low = 0
    plies = rng.randint(low, high) if high > 0 else 0
    board = [0] * total_cells
    current = 1
    last_move = -1

    for _ in range(plies):
        legal = [idx for idx, cell in enumerate(board) if cell == 0]
        if not legal:
            return None
        move = rng.choice(legal)
        board[move] = current
        last_move = move
        if _nxn_winner(board, board_size, win_len, move) != 0:
            return None
        current = 2 if current == 1 else 1

    if all(cell != 0 for cell in board):
        return None
    return board, current, last_move


async def _generate_engine_labeled_positions(
    count: int,
    board_size: int,
    win_len: int,
    callback: TRAIN_CALLBACK,
    engine_eval: Any,
    *,
    variant: str = "",
    rng: random.Random | None = None,
    source: str = "engine",
    phase_focus: str | None = None,
    current_player_focus: int | None = None,
    min_value: float | None = None,
    boost_weight: float = 1.0,
) -> list[dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    started_at = time.monotonic()
    last_reported = 0
    attempts = 0
    max_attempts = max(count * 25, 200)
    rng = rng or random

    while len(positions) < count and attempts < max_attempts:
        attempts += 1
        sampled = _sample_engine_position(board_size, win_len, rng=rng, phase_focus=phase_focus)
        if sampled is None:
            continue

        board, current, last_move = sampled
        if current_player_focus in (1, 2) and current != int(current_player_focus):
            continue
        phase_bucket = _classify_engine_phase(board_size, sum(1 for cell in board if cell != 0))
        analysis = await engine_eval.analyze_position(board, current, board_size, win_len)
        move = int(analysis.get("bestMove", -1))
        value = max(-1.0, min(1.0, float(analysis.get("value", 0.0))))
        if min_value is not None and value < float(min_value):
            continue
        if move < 0 or move >= len(board) or board[move] != 0:
            continue
        hints = await engine_eval.suggest_moves(board, current, board_size, win_len, top_n=5)

        positions.append({
            "board_size": board_size,
            "board": _flat_to_board2d(board, board_size),
            "current_player": current,
            "last_move": list(divmod(last_move, board_size)) if last_move >= 0 else None,
            "policy": _soft_policy_from_engine_hints(move, board, board_size, hints),
            "value": value,
            "source": source,
            "phaseBucket": phase_bucket,
            "sampleWeight": max(float(boost_weight), 0.1),
            "playerFocus": int(current_player_focus) if current_player_focus in (1, 2) else None,
            "conversionTarget": bool(min_value is not None and min_value > 0.0),
        })

        if len(positions) - last_reported >= 64 or len(positions) == count:
            last_reported = len(positions)
            await _emit_dataset_progress(
                callback,
                generated=len(positions),
                total=count,
                stage="engine_teacher",
                message=f"Generated {len(positions)}/{count} engine-labeled positions",
                start_time=started_at,
            )
            await asyncio.sleep(0)

    return positions


async def _relabel_positions_with_engine(
    positions: list[dict[str, Any]],
    board_size: int,
    win_len: int,
    engine_eval: Any,
    callback: TRAIN_CALLBACK | None = None,
) -> list[dict[str, Any]]:
    relabeled: list[dict[str, Any]] = []
    started_at = time.monotonic()
    last_reported = 0

    for idx, pos in enumerate(positions, 1):
        board = _board2d_to_flat(pos["board"])
        current = int(pos["current_player"])
        analysis = await engine_eval.analyze_position(board, current, board_size, win_len)
        move = int(analysis.get("bestMove", -1))
        value = max(-1.0, min(1.0, float(analysis.get("value", 0.0))))
        if move < 0 or move >= len(board) or board[move] != 0:
            continue
        hints = await engine_eval.suggest_moves(board, current, board_size, win_len, top_n=5)

        relabeled.append({
            "board_size": board_size,
            "board": pos["board"],
            "current_player": current,
            "last_move": pos.get("last_move"),
            "policy": _soft_policy_from_engine_hints(move, board, board_size, hints),
            "value": value,
            "source": pos.get("source", "failure"),
            "motif": pos.get("motif"),
            "sampleWeight": float(pos.get("sampleWeight", 1.0)),
            "playerFocus": int(pos.get("playerFocus", 0) or 0),
            "conversionTarget": bool(pos.get("conversionTarget", False)),
        })

        if callback is not None and (idx - last_reported >= 24 or idx == len(positions)):
            last_reported = idx
            elapsed = max(time.monotonic() - started_at, 0.01)
            await callback({
                "type": "train.progress",
                "payload": {
                    "phase": "repair",
                    "stage": "relabel",
                    "generated": idx,
                    "total": len(positions),
                    "positions": len(relabeled),
                    "percent": round((idx / max(len(positions), 1)) * 100.0, 1),
                    "elapsed": round(elapsed, 1),
                    "speed": round(idx / elapsed, 2),
                    "speedUnit": "pos/s",
                },
            })
            await asyncio.sleep(0)

    return relabeled


async def _run_engine_exam(
    model: Any,
    board_size: int,
    win_len: int,
    device: Any,
    callback: TRAIN_CALLBACK,
    engine_eval: Any,
    *,
    variant: str,
    cycle: int,
    total_cycles: int,
    num_pairs: int,
    previous_result: dict[str, Any] | None = None,
    max_failure_positions: int = 96,
    max_failure_turns_per_game: int = 4,
    phase: str = "exam",
    stage: str = "engine_eval",
    collect_failures: bool = True,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    from gomoku_api.ws.arena_eval import ArenaResult, _model_greedy_decision
    from gomoku_api.ws.predict_service import _loaded_model_decision

    wins = losses = draws = 0
    total_games = max(num_pairs * 2, 1)
    raw_failures: list[dict[str, Any]] = []
    raw_conversion_failures: list[dict[str, Any]] = []
    started_at = time.monotonic()
    side_stats: dict[int, dict[str, int]] = {
        1: {"wins": 0, "losses": 0, "draws": 0, "total": 0},
        2: {"wins": 0, "losses": 0, "draws": 0, "total": 0},
    }
    candidate_move_count = 0
    tactical_override_count = 0
    value_guided_count = 0
    model_policy_count = 0
    unsafe_moves_filtered_total = 0
    decision_reason_counts: dict[str, int] = {}
    conversion_failure_count = 0
    conversion_failure_side_counts: dict[int, int] = {1: 0, 2: 0}
    pure_gap_count = 0
    pure_gap_side_counts: dict[int, int] = {1: 0, 2: 0}
    candidate_move_side_counts: dict[int, int] = {1: 0, 2: 0}
    pure_tactical_gap_count = 0
    pure_conversion_gap_count = 0
    pure_missed_win_count = 0
    pure_missed_block_count = 0
    pure_missed_win_side_counts: dict[int, int] = {1: 0, 2: 0}
    pure_missed_block_side_counts: dict[int, int] = {1: 0, 2: 0}

    def _side_rate(side: int, numerator: str, *, draw_weight: float = 0.0) -> float:
        bucket = side_stats[side]
        total = max(bucket["total"], 1)
        return (bucket[numerator] + draw_weight * bucket["draws"]) / total

    def _pure_gap_rate(side: int | None = None) -> float:
        if side is None:
            return pure_gap_count / max(candidate_move_count, 1)
        side_total = max(candidate_move_side_counts.get(side, 0), 1)
        return pure_gap_side_counts.get(side, 0) / side_total

    for pair in range(num_pairs):
        for candidate_side in (1, 2):
            board = [0] * (board_size * board_size)
            current = 1
            last_move = -1
            winner = 0
            forfeit_by = 0
            candidate_states: list[dict[str, Any]] = []

            for _ in range(board_size * board_size):
                legal = [i for i, c in enumerate(board) if c == 0]
                if not legal:
                    break

                if current == candidate_side:
                    candidate_state = {
                        "board_size": board_size,
                        "board": _flat_to_board2d(board, board_size),
                        "current_player": current,
                        "last_move": list(divmod(last_move, board_size)) if last_move >= 0 else None,
                    }
                    candidate_states.append(candidate_state)
                    decision = _model_greedy_decision(board, board_size, win_len, current, model, device)
                    move = int(decision.get("move", -1))
                    pure_decision = _loaded_model_decision(
                        board,
                        current,
                        board_size,
                        win_len,
                        model,
                        decision_mode="pure",
                    )
                    pure_gap_candidate = _build_pure_gap_relabel_candidate(candidate_state, decision, pure_decision)
                    if pure_gap_candidate is not None:
                        pure_gap_count += 1
                        pure_gap_side_counts[current] = pure_gap_side_counts.get(current, 0) + 1
                        if str(pure_gap_candidate.get("source")) == "failure_conversion":
                            raw_conversion_failures.append(pure_gap_candidate)
                            pure_conversion_gap_count += 1
                            conversion_failure_count += 1
                            conversion_failure_side_counts[current] = conversion_failure_side_counts.get(current, 0) + 1
                        else:
                            raw_failures.append(pure_gap_candidate)
                            pure_tactical_gap_count += 1
                        motif = str(pure_gap_candidate.get("motif", "") or "")
                        if motif == "pure_missed_win":
                            pure_missed_win_count += 1
                            pure_missed_win_side_counts[current] = pure_missed_win_side_counts.get(current, 0) + 1
                        elif motif == "pure_missed_block":
                            pure_missed_block_count += 1
                            pure_missed_block_side_counts[current] = pure_missed_block_side_counts.get(current, 0) + 1
                    candidate_move_count += 1
                    candidate_move_side_counts[current] = candidate_move_side_counts.get(current, 0) + 1
                    if bool(decision.get("tacticalOverride")):
                        tactical_override_count += 1
                    if bool(decision.get("valueGuided")):
                        value_guided_count += 1
                    if str(decision.get("tacticalReason", "")) == "model_policy":
                        model_policy_count += 1
                    unsafe_moves_filtered_total += int(decision.get("unsafeMovesFiltered", 0) or 0)
                    reason = str(decision.get("tacticalReason", "unknown") or "unknown")
                    decision_reason_counts[reason] = decision_reason_counts.get(reason, 0) + 1
                else:
                    move = await engine_eval.best_move(board, current, board_size, win_len)

                if move < 0 or move >= len(board) or board[move] != 0:
                    forfeit_by = current
                    break

                board[move] = current
                last_move = move
                winner = _nxn_winner(board, board_size, win_len, move)
                if winner != 0:
                    break
                current = 2 if current == 1 else 1

            candidate_lost = False
            side_stats[candidate_side]["total"] += 1
            if forfeit_by != 0:
                if forfeit_by == candidate_side:
                    losses += 1
                    side_stats[candidate_side]["losses"] += 1
                    candidate_lost = True
                else:
                    wins += 1
                    side_stats[candidate_side]["wins"] += 1
            elif winner == candidate_side:
                wins += 1
                side_stats[candidate_side]["wins"] += 1
            elif winner != 0:
                losses += 1
                side_stats[candidate_side]["losses"] += 1
                candidate_lost = True
            else:
                draws += 1
                side_stats[candidate_side]["draws"] += 1

            if candidate_lost:
                raw_failures.extend(candidate_states[-max_failure_turns_per_game:])

        games_done = (pair + 1) * 2
        winrate = (wins + 0.5 * draws) / max(games_done, 1)
        decisive_winrate = wins / max(games_done, 1)
        draw_rate = draws / max(games_done, 1)
        winrate_as_p1 = _side_rate(1, "wins", draw_weight=0.5)
        winrate_as_p2 = _side_rate(2, "wins", draw_weight=0.5)
        balanced_side_winrate = min(winrate_as_p1, winrate_as_p2)
        tactical_override_rate = tactical_override_count / max(candidate_move_count, 1)
        value_guided_rate = value_guided_count / max(candidate_move_count, 1)
        model_policy_rate = model_policy_count / max(candidate_move_count, 1)
        avg_unsafe_filtered = unsafe_moves_filtered_total / max(candidate_move_count, 1)
        prev_wr = previous_result.get("winrate") if previous_result else None
        delta_wr = None if prev_wr is None else round(winrate - prev_wr, 4)
        trend = "flat"
        if delta_wr is not None:
            if delta_wr > 0.02:
                trend = "improving"
            elif delta_wr < -0.02:
                trend = "regressing"

        elapsed = max(time.monotonic() - started_at, 0.01)
        await callback({
            "type": "train.progress",
            "payload": {
                "phase": phase,
                "stage": stage,
                "variant": variant,
                "cycle": cycle,
                "totalCycles": total_cycles,
                "iteration": cycle,
                "totalIterations": total_cycles,
                "game": games_done,
                "totalGames": total_games,
                "arenaWins": wins,
                "arenaLosses": losses,
                "arenaDraws": draws,
                "winrateVsAlgorithm": round(winrate, 4),
                "decisiveWinRate": round(decisive_winrate, 4),
                "drawRate": round(draw_rate, 4),
                "winrateAsP1": round(winrate_as_p1, 4),
                "winrateAsP2": round(winrate_as_p2, 4),
                "balancedSideWinrate": round(balanced_side_winrate, 4),
                "tacticalOverrideCount": tactical_override_count,
                "candidateMoveCount": candidate_move_count,
                "tacticalOverrideRate": round(tactical_override_rate, 4),
                "valueGuidedRate": round(value_guided_rate, 4),
                "modelPolicyRate": round(model_policy_rate, 4),
                "avgUnsafeMovesFiltered": round(avg_unsafe_filtered, 4),
                "deltaWinrate": delta_wr,
                "progressTrend": trend,
                "elapsed": round(elapsed, 1),
                "speed": round(games_done / elapsed, 2),
                "speedUnit": "g/s",
                "positions": len(raw_failures) + len(raw_conversion_failures),
                "conversionFailures": conversion_failure_count,
                "conversionFailuresAsP1": conversion_failure_side_counts[1],
                "conversionFailuresAsP2": conversion_failure_side_counts[2],
                "pureGapCount": pure_gap_count,
                "pureGapRate": round(_pure_gap_rate(), 4),
                "pureGapRateAsP1": round(_pure_gap_rate(1), 4),
                "pureGapRateAsP2": round(_pure_gap_rate(2), 4),
                "pureAlignmentRate": round(1.0 - _pure_gap_rate(), 4),
                "pureTacticalGapCount": pure_tactical_gap_count,
                "pureConversionGapCount": pure_conversion_gap_count,
                "pureMissedWinCount": pure_missed_win_count,
                "pureMissedBlockCount": pure_missed_block_count,
                "pureMissedWinAsP1": pure_missed_win_side_counts[1],
                "pureMissedWinAsP2": pure_missed_win_side_counts[2],
                "pureMissedBlockAsP1": pure_missed_block_side_counts[1],
                "pureMissedBlockAsP2": pure_missed_block_side_counts[2],
            },
        })
        await asyncio.sleep(0)

    if collect_failures:
        unique_raw = _merge_failure_bank([], raw_failures + raw_conversion_failures, max_size=max_failure_positions)
        relabeled = await _relabel_positions_with_engine(
            unique_raw,
            board_size,
            win_len,
            engine_eval,
            callback,
        )
    else:
        relabeled = []
    result = ArenaResult(wins_a=wins, wins_b=losses, draws=draws, total=total_games)
    summary = {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winrate": result.winrate_a,
        "decisiveWinRate": result.decisive_winrate_a,
        "drawRate": result.draw_rate,
        "winrateAsP1": round(_side_rate(1, "wins", draw_weight=0.5), 4),
        "winrateAsP2": round(_side_rate(2, "wins", draw_weight=0.5), 4),
        "decisiveWinRateAsP1": round(_side_rate(1, "wins"), 4),
        "decisiveWinRateAsP2": round(_side_rate(2, "wins"), 4),
        "drawRateAsP1": round(_side_rate(1, "draws"), 4),
        "drawRateAsP2": round(_side_rate(2, "draws"), 4),
        "balancedSideWinrate": round(min(_side_rate(1, "wins", draw_weight=0.5), _side_rate(2, "wins", draw_weight=0.5)), 4),
        "candidateMoveCount": candidate_move_count,
        "tacticalOverrideCount": tactical_override_count,
        "tacticalOverrideRate": round(tactical_override_count / max(candidate_move_count, 1), 4),
        "valueGuidedRate": round(value_guided_count / max(candidate_move_count, 1), 4),
        "modelPolicyRate": round(model_policy_count / max(candidate_move_count, 1), 4),
        "avgUnsafeMovesFiltered": round(unsafe_moves_filtered_total / max(candidate_move_count, 1), 4),
        "decisionReasonCounts": dict(sorted(decision_reason_counts.items())),
        "conversionFailures": conversion_failure_count,
        "conversionFailuresAsP1": conversion_failure_side_counts[1],
        "conversionFailuresAsP2": conversion_failure_side_counts[2],
        "pureGapCount": pure_gap_count,
        "pureGapRate": round(_pure_gap_rate(), 4),
        "pureGapRateAsP1": round(_pure_gap_rate(1), 4),
        "pureGapRateAsP2": round(_pure_gap_rate(2), 4),
        "pureAlignmentRate": round(1.0 - _pure_gap_rate(), 4),
        "pureTacticalGapCount": pure_tactical_gap_count,
        "pureConversionGapCount": pure_conversion_gap_count,
        "pureMissedWinCount": pure_missed_win_count,
        "pureMissedBlockCount": pure_missed_block_count,
        "pureMissedWinAsP1": pure_missed_win_side_counts[1],
        "pureMissedWinAsP2": pure_missed_win_side_counts[2],
        "pureMissedBlockAsP1": pure_missed_block_side_counts[1],
        "pureMissedBlockAsP2": pure_missed_block_side_counts[2],
        "conversionRelabels": sum(1 for pos in relabeled if pos.get("motif") == "conversion"),
        "newFailures": len(relabeled),
    }
    return result, relabeled, summary


# ---------------------------------------------------------------------------
# Data generation for TTT3 (3x3, win_length=3)
# ---------------------------------------------------------------------------


def _ttt3_winner(board: list[int]) -> int:
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    ]
    for a, b, c in lines:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def _minimax_value(board: list[int], current: int) -> float:
    winner = _ttt3_winner(board)
    if winner == current:
        return 1.0
    if winner == -current:
        return -1.0
    empty = [i for i in range(9) if board[i] == 0]
    if not empty:
        return 0.0

    best = -2.0
    for move in empty:
        board[move] = current
        value = -_minimax_value(board, -current)
        board[move] = 0
        best = max(best, value)
    return best


def _minimax_policy(board: list[int], current: int) -> list[float]:
    empty = [i for i in range(9) if board[i] == 0]
    if not empty:
        return [0.0] * 9

    scores: dict[int, float] = {}
    for move in empty:
        board[move] = current
        scores[move] = -_minimax_value(board, -current)
        board[move] = 0

    best_value = max(scores.values())
    best_moves = [move for move, value in scores.items() if value == best_value]
    policy = [0.0] * 9
    for move in best_moves:
        policy[move] = 1.0 / len(best_moves)
    return policy


async def _generate_ttt3_positions(count: int, callback: TRAIN_CALLBACK) -> list[dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    start_time = time.monotonic()
    last_reported = 0

    while len(positions) < count:
        board = [0] * 9
        current = 1
        moves_played = random.randint(0, 6)
        available = list(range(9))
        random.shuffle(available)

        for idx in range(min(moves_played, len(available))):
            board[available[idx]] = current
            current = -current

        if _ttt3_winner(board) != 0:
            continue
        if all(cell != 0 for cell in board):
            continue

        policy = _minimax_policy(list(board), current)
        value = _minimax_value(list(board), current)

        board_2d = []
        for row in range(3):
            encoded_row = []
            for col in range(3):
                cell = board[row * 3 + col]
                if cell == 1:
                    encoded_row.append(1)
                elif cell == -1:
                    encoded_row.append(2)
                else:
                    encoded_row.append(0)
            board_2d.append(encoded_row)

        policy_256 = [0.0] * 256
        for idx, prob in enumerate(policy):
            row, col = divmod(idx, 3)
            policy_256[row * 16 + col] = prob

        positions.append({
            "board_size": 3,
            "board": board_2d,
            "current_player": 1 if current == 1 else 2,
            "last_move": None,
            "policy": policy_256,
            "value": value,
        })

        if len(positions) - last_reported >= 128 or len(positions) == count:
            last_reported = len(positions)
            await _emit_dataset_progress(
                callback,
                generated=len(positions),
                total=count,
                stage="generating",
                message=f"Generated {len(positions)} tactical positions",
                start_time=start_time,
            )
            await asyncio.sleep(0)

    return positions


# ---------------------------------------------------------------------------
# Alpha-beta minimax for small NxN boards (5x5 with depth limit)
# ---------------------------------------------------------------------------


def _nxn_evaluate_heuristic(board: list[int], n: int, win_len: int, player: int) -> float:
    """Quick heuristic evaluation for NxN board from player's perspective."""
    score = 0.0
    opponent = 3 - player
    for r in range(n):
        for c in range(n):
            if board[r * n + c] != 0:
                continue
            # Count threats around empty cell
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                my_count = opp_count = 0
                for s in range(1, win_len):
                    nr, nc = r + dr * s, c + dc * s
                    if 0 <= nr < n and 0 <= nc < n:
                        v = board[nr * n + nc]
                        if v == player:
                            my_count += 1
                        elif v == opponent:
                            opp_count += 1
                        else:
                            break
                    else:
                        break
                for s in range(1, win_len):
                    nr, nc = r - dr * s, c - dc * s
                    if 0 <= nr < n and 0 <= nc < n:
                        v = board[nr * n + nc]
                        if v == player:
                            my_count += 1
                        elif v == opponent:
                            opp_count += 1
                        else:
                            break
                    else:
                        break
                if my_count >= win_len - 1 and opp_count == 0:
                    score += 10.0
                elif my_count >= win_len - 2 and opp_count == 0:
                    score += 1.0
                if opp_count >= win_len - 1 and my_count == 0:
                    score -= 8.0
                elif opp_count >= win_len - 2 and my_count == 0:
                    score -= 0.8
    return score


def _nxn_minimax(
    board: list[int], n: int, win_len: int, current: int,
    depth: int, alpha: float, beta: float, last_move: int,
) -> float:
    """Alpha-beta minimax for NxN boards with depth limit."""
    if last_move >= 0:
        w = _nxn_winner(board, n, win_len, last_move)
        if w != 0:
            return 100.0 if w == current else -100.0

    if depth <= 0:
        return _nxn_evaluate_heuristic(board, n, win_len, current)

    empty = [i for i in range(n * n) if board[i] == 0]
    if not empty:
        return 0.0

    # Order moves: center first, then near existing stones
    center = n // 2
    def move_priority(m: int) -> float:
        r, c = divmod(m, n)
        dist = abs(r - center) + abs(c - center)
        # Bonus for adjacent to existing stones
        adj = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and board[nr * n + nc] != 0:
                    adj += 1
        return -adj * 10 + dist
    empty.sort(key=move_priority)

    best = -200.0
    opp = 3 - current
    for move in empty:
        board[move] = current
        val = -_nxn_minimax(board, n, win_len, opp, depth - 1, -beta, -alpha, move)
        board[move] = 0
        if val > best:
            best = val
        alpha = max(alpha, val)
        if alpha >= beta:
            break
    return best


def _nxn_minimax_policy(board: list[int], n: int, win_len: int, current: int, depth: int = 6) -> tuple[list[float], float]:
    """Get minimax-evaluated policy for NxN board. Returns (policy_N*N, value)."""
    empty = [i for i in range(n * n) if board[i] == 0]
    if not empty:
        return [0.0] * (n * n), 0.0

    scores: dict[int, float] = {}
    opp = 3 - current
    for move in empty:
        board[move] = current
        w = _nxn_winner(board, n, win_len, move)
        if w != 0:
            scores[move] = 100.0
            board[move] = 0
            continue
        val = -_nxn_minimax(board, n, win_len, opp, depth - 1, -200.0, 200.0, move)
        scores[move] = val
        board[move] = 0

    best_val = max(scores.values())
    # Softmax-like distribution: boost good moves
    import math
    temperature = 1.0
    exp_scores = {}
    for m, s in scores.items():
        exp_scores[m] = math.exp(min((s - best_val) / max(temperature, 0.1), 20))
    total = sum(exp_scores.values())

    policy = [0.0] * (n * n)
    for m, e in exp_scores.items():
        policy[m] = e / total

    value = max(-1.0, min(1.0, best_val / 100.0))
    return policy, value


# ---------------------------------------------------------------------------
# Generic NxN data generation (with minimax for small boards)
# ---------------------------------------------------------------------------


def _nxn_winner(board: list[int], n: int, win_len: int, last_move: int) -> int:
    if last_move < 0:
        return 0

    player = board[last_move]
    if player == 0:
        return 0

    row, col = divmod(last_move, n)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for step in range(1, win_len):
            nr, nc = row + dr * step, col + dc * step
            if 0 <= nr < n and 0 <= nc < n and board[nr * n + nc] == player:
                count += 1
            else:
                break
        for step in range(1, win_len):
            nr, nc = row - dr * step, col - dc * step
            if 0 <= nr < n and 0 <= nc < n and board[nr * n + nc] == player:
                count += 1
            else:
                break
        if count >= win_len:
            return player
    return 0


async def _generate_nxn_positions(
    count: int,
    board_size: int,
    win_len: int,
    callback: TRAIN_CALLBACK,
) -> list[dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    games_played = 0
    start_time = time.monotonic()
    last_reported = 0
    while len(positions) < count:
        board = [0] * (board_size * board_size)
        current_player = 1
        last_move_flat = -1
        history: list[dict[str, Any]] = []
        winner = 0

        for _move in range(board_size * board_size):
            empty = [idx for idx, cell in enumerate(board) if cell == 0]
            if not empty:
                break

            # Uniform policy over legal moves (minimax moved to offline gen)
            policy = [0.0] * 256
            probability = 1.0 / len(empty)
            for idx in empty:
                r, c = divmod(idx, board_size)
                policy[r * 16 + c] = probability
            pos_value = 0.0

            board_2d = [
                [board[row * board_size + col] for col in range(board_size)]
                for row in range(board_size)
            ]
            history.append({
                "board_size": board_size,
                "board": board_2d,
                "current_player": current_player,
                "last_move": list(divmod(last_move_flat, board_size)) if last_move_flat >= 0 else None,
                "policy": policy,
                "value": pos_value,
            })

            # Choose move: random (minimax moved to offline gen)
            move = random.choice(empty)
            board[move] = current_player
            last_move_flat = move

            winner = _nxn_winner(board, board_size, win_len, move)
            if winner != 0:
                break
            current_player = 2 if current_player == 1 else 1

        for pos in history:
            if winner == 0:
                pos["value"] = 0.0
            elif pos["current_player"] == winner:
                pos["value"] = 1.0
            else:
                pos["value"] = -1.0

        positions.extend(history)
        games_played += 1

        if len(positions) - last_reported >= max(board_size * 2, 128) or len(positions) >= count:
            last_reported = len(positions)
            await _emit_dataset_progress(
                callback,
                generated=min(len(positions), count),
                total=count,
                stage="generating",
                message=f"Generated {min(len(positions), count)} positions from {games_played} games",
                start_time=start_time,
                games=games_played,
            )
            await asyncio.sleep(0)

    return positions[:count]


async def _build_positions(
    variant: str,
    data_count: int,
    callback: TRAIN_CALLBACK,
) -> tuple[list[dict[str, Any]], int, int]:
    board_size, win_len = _resolve_variant_spec(variant)
    if variant == "ttt3":
        positions = await _generate_ttt3_positions(data_count, callback)
    else:
        positions = await _generate_nxn_positions(data_count, board_size, win_len, callback)
    return positions, board_size, win_len


# ---------------------------------------------------------------------------
# Tactical curriculum for large boards
# ---------------------------------------------------------------------------


async def _generate_tactical_curriculum_positions(
    count: int,
    board_size: int,
    win_len: int,
    callback: TRAIN_CALLBACK,
    *,
    motif_filter: str | None = None,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Generate immediate-win / immediate-block positions for strong tactical supervision."""
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    positions: list[dict[str, Any]] = []
    started_at = time.monotonic()
    last_reported = 0
    rng = rng or random

    while len(positions) < count:
        board = [0] * (board_size * board_size)
        motif = motif_filter or rng.choice(("win", "block"))
        current_player = rng.choice((1, 2))
        line_player = current_player if motif == "win" else (2 if current_player == 1 else 1)
        target_player = current_player
        direction = rng.choice(directions)
        dr, dc = direction

        # Pick a start cell where a full win_len segment fits.
        valid_starts: list[tuple[int, int]] = []
        for row in range(board_size):
            for col in range(board_size):
                end_row = row + dr * (win_len - 1)
                end_col = col + dc * (win_len - 1)
                if 0 <= end_row < board_size and 0 <= end_col < board_size:
                    valid_starts.append((row, col))
        if not valid_starts:
            break

        start_row, start_col = rng.choice(valid_starts)
        gap_index = rng.randrange(win_len)
        target_move = (start_row + dr * gap_index) * board_size + (start_col + dc * gap_index)

        occupied: set[int] = {target_move}
        for step in range(win_len):
            row = start_row + dr * step
            col = start_col + dc * step
            flat = row * board_size + col
            if flat == target_move:
                continue
            board[flat] = line_player
            occupied.add(flat)

        # Add a few random context stones far from the tactical segment.
        extra_stones = rng.randint(0, max(2, board_size // 3))
        for _ in range(extra_stones):
            placed = False
            for _attempt in range(20):
                move = rng.randrange(board_size * board_size)
                if move in occupied or board[move] != 0:
                    continue
                if abs((move // board_size) - (target_move // board_size)) <= 1 and abs((move % board_size) - (target_move % board_size)) <= 1:
                    continue
                board[move] = rng.choice((1, 2))
                occupied.add(move)
                placed = True
                break
            if not placed:
                break

        # Reject accidental terminal boards or broken motifs.
        last_stone = next((idx for idx, cell in enumerate(board) if cell == line_player), -1)
        if last_stone >= 0 and _nxn_winner(board, board_size, win_len, last_stone) != 0:
            continue

        policy = _one_hot_policy(target_move, board_size)
        value = 1.0 if motif == "win" else 0.35
        positions.append({
            "board_size": board_size,
            "board": _flat_to_board2d(board, board_size),
            "current_player": target_player,
            "last_move": None,
            "policy": policy,
            "value": value,
            "motif": motif,
            "source": "tactical",
            "sampleWeight": 1.0,
        })

        if len(positions) - last_reported >= 128 or len(positions) == count:
            last_reported = len(positions)
            await _emit_dataset_progress(
                callback,
                generated=len(positions),
                total=count,
                stage="generating",
                message=f"Generated {len(positions)} tactical {board_size}x{board_size} positions",
                start_time=started_at,
            )
            await asyncio.sleep(0)

    return positions


def _build_exact_ttt5_validation_pack() -> list[dict[str, Any]]:
    """Small deterministic solved pack for user-found 5x5 traps."""

    def _board(stones: dict[int, int]) -> list[int]:
        board = [0] * 25
        for move, player in stones.items():
            board[move] = player
        return board

    def _position(
        stones: dict[int, int],
        current_player: int,
        target_move: int,
        *,
        family: str,
        motif: str,
        value: float,
        conversion_target: bool,
        sample_weight: float,
    ) -> dict[str, Any]:
        board = _board(stones)
        return {
            "board_size": 5,
            "board": _flat_to_board2d(board, 5),
            "current_player": current_player,
            "last_move": None,
            "policy": _one_hot_policy(target_move, 5),
            "value": value,
            "motif": motif,
            "source": "exact",
            "sampleWeight": sample_weight,
            "playerFocus": current_player,
            "conversionTarget": conversion_target,
            "exactFamily": family,
        }

    return [
        _position({4: 1, 9: 1, 14: 1, 1: 2, 7: 2, 16: 2}, 2, 19, family="right_edge_vertical_block", motif="exact_block_edge_vertical", value=0.35, conversion_target=False, sample_weight=2.15),
        _position({4: 1, 9: 1, 14: 1, 6: 2, 12: 2, 15: 2}, 2, 19, family="right_edge_vertical_block", motif="exact_block_edge_vertical_alt", value=0.35, conversion_target=False, sample_weight=2.2),
        _position({2: 1, 3: 1, 4: 1, 11: 2, 12: 2, 13: 2}, 2, 1, family="top_edge_horizontal_block", motif="exact_block_top_edge", value=0.35, conversion_target=False, sample_weight=2.2),
        _position({1: 1, 2: 1, 3: 1, 7: 2, 12: 2, 17: 2}, 2, 4, family="top_edge_horizontal_block", motif="exact_block_top_edge_open_right", value=0.35, conversion_target=False, sample_weight=2.2),
        _position({3: 1, 7: 1, 11: 1, 1: 2, 12: 2, 22: 2}, 2, 15, family="edge_diagonal_block", motif="exact_block_edge_diagonal", value=0.35, conversion_target=False, sample_weight=2.15),
        _position({0: 1, 6: 1, 12: 1, 4: 2, 8: 2, 17: 2}, 2, 18, family="main_diagonal_block", motif="exact_block_main_diagonal", value=0.35, conversion_target=False, sample_weight=2.15),
        _position({20: 2, 21: 2, 22: 2, 1: 1, 7: 1, 14: 1}, 1, 23, family="bottom_edge_horizontal_block", motif="exact_block_bottom_edge", value=0.35, conversion_target=False, sample_weight=2.1),
        _position({20: 2, 21: 2, 22: 2, 4: 1, 8: 1, 13: 1}, 1, 23, family="bottom_edge_horizontal_block", motif="exact_block_bottom_edge_alt", value=0.35, conversion_target=False, sample_weight=2.1),
        _position({0: 2, 5: 2, 10: 2, 12: 1, 18: 1}, 1, 15, family="left_edge_vertical_block", motif="exact_block_left_edge", value=0.35, conversion_target=False, sample_weight=2.1),
        _position({0: 2, 5: 2, 10: 2, 6: 1, 12: 1, 24: 1}, 1, 15, family="left_edge_vertical_block", motif="exact_block_left_edge_alt", value=0.35, conversion_target=False, sample_weight=2.15),
        _position({6: 2, 12: 2, 18: 2, 3: 1, 9: 1}, 1, 24, family="long_diagonal_block", motif="exact_block_long_diagonal", value=0.35, conversion_target=False, sample_weight=2.15),
        _position({2: 2, 7: 2, 12: 2, 4: 1, 9: 1}, 2, 17, family="vertical_win", motif="exact_win_vertical", value=1.0, conversion_target=True, sample_weight=2.3),
        _position({15: 1, 16: 1, 17: 1, 2: 2, 7: 2, 12: 2}, 1, 18, family="horizontal_win", motif="exact_win_horizontal", value=1.0, conversion_target=True, sample_weight=2.3),
        _position({1: 1, 7: 1, 13: 1, 5: 2, 10: 2, 22: 2}, 1, 19, family="diagonal_win", motif="exact_win_diagonal", value=1.0, conversion_target=True, sample_weight=2.3),
        _position({4: 2, 8: 2, 12: 2, 0: 1, 6: 1}, 2, 16, family="edge_diagonal_win", motif="exact_win_edge_diagonal", value=1.0, conversion_target=True, sample_weight=2.3),
    ]


async def _build_frozen_benchmark_suites(
    variant: str,
    board_size: int,
    win_len: int,
    engine_eval: Any | None,
) -> dict[str, list[dict[str, Any]]]:
    async def _silent_callback(_event: dict[str, Any]) -> None:
        return None

    suites: dict[str, list[dict[str, Any]]] = {
        "block": await _generate_tactical_curriculum_positions(
            48,
            board_size,
            win_len,
            _silent_callback,
            motif_filter="block",
            rng=random.Random(f"{variant}:block"),
        ),
        "win": await _generate_tactical_curriculum_positions(
            48,
            board_size,
            win_len,
            _silent_callback,
            motif_filter="win",
            rng=random.Random(f"{variant}:win"),
        ),
    }
    if variant == "ttt5" and board_size == 5 and win_len == 4:
        suites["exact"] = _build_exact_ttt5_validation_pack()

    if engine_eval is not None:
        suites["mid"] = await _generate_engine_labeled_positions(
            48 if board_size <= 5 else 24,
            board_size,
            win_len,
            _silent_callback,
            engine_eval,
            variant=variant,
            rng=random.Random(f"{variant}:mid"),
            source="benchmark",
            phase_focus="mid",
        )
        suites["late"] = await _generate_engine_labeled_positions(
            48 if board_size <= 5 else 24,
            board_size,
            win_len,
            _silent_callback,
            engine_eval,
            variant=variant,
            rng=random.Random(f"{variant}:late"),
            source="benchmark",
            phase_focus="late",
        )
    return suites


def _position_flat_board(position: dict[str, Any]) -> list[int]:
    board_2d = position.get("board") or []
    if not board_2d:
        return []
    return [int(cell) for row in board_2d for cell in row]


def _position_policy_target_move(position: dict[str, Any]) -> int:
    board_size = int(position.get("board_size", 0) or 0)
    policy = position.get("policy") or []
    board = _position_flat_board(position)
    if board_size <= 0 or not policy or not board:
        return -1

    legal = [idx for idx, cell in enumerate(board) if cell == 0]
    if not legal:
        return -1
    return max(
        legal,
        key=lambda move: float(policy[_policy_cell_index(move, board_size)]),
    )


def _build_decision_suite_failure(
    position: dict[str, Any],
    *,
    suite_name: str,
    decision_mode: str,
    chosen_move: int,
    target_move: int,
) -> dict[str, Any]:
    failure = {
        key: value
        for key, value in position.items()
    }
    current_player = int(position.get("current_player", 1) or 1)
    failure["source"] = "failure_pure_gap" if decision_mode == "pure" else "benchmark_miss"
    if suite_name == "win":
        failure["motif"] = "pure_missed_win" if decision_mode == "pure" else "win"
        failure["conversionTarget"] = True
        failure["sampleWeight"] = max(float(failure.get("sampleWeight", 1.0) or 1.0), 2.35 if current_player == 1 else 2.05)
        if decision_mode == "pure":
            failure["pureMissedWinInOne"] = True
    elif suite_name == "block":
        failure["motif"] = "pure_missed_block" if decision_mode == "pure" else "block"
        failure["conversionTarget"] = False
        failure["sampleWeight"] = max(float(failure.get("sampleWeight", 1.0) or 1.0), 2.10 if current_player == 1 else 1.85)
        if decision_mode == "pure":
            failure["pureMissedBlockInOne"] = True
    elif suite_name == "exact":
        failure["motif"] = str(position.get("motif", "") or "exact_trap")
        failure["conversionTarget"] = bool(position.get("conversionTarget", False))
        failure["sampleWeight"] = max(float(failure.get("sampleWeight", 1.0) or 1.0), 2.25 if failure["conversionTarget"] else 2.0)
        failure["exactFamily"] = str(position.get("exactFamily", "") or position.get("motif", "") or "exact_trap")
        if decision_mode == "pure":
            failure["pureExactTrapMiss"] = True
    failure["playerFocus"] = current_player
    failure["decisionMode"] = decision_mode
    failure["targetMove"] = target_move
    failure["chosenMove"] = chosen_move
    return failure


def _evaluate_decision_suite(
    model: Any,
    positions: list[dict[str, Any]],
    board_size: int,
    win_len: int,
    *,
    decision_mode: str = "pure",
    suite_name: str = "suite",
    collect_failures: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from gomoku_api.ws.predict_service import _loaded_model_decision

    total = 0
    correct = 0
    failures: list[dict[str, Any]] = []
    family_total: dict[str, int] = {}
    family_correct: dict[str, int] = {}
    player_total: dict[int, int] = {}
    player_correct: dict[int, int] = {}

    for position in positions:
        target_move = _position_policy_target_move(position)
        if target_move < 0:
            continue
        flat_board = _position_flat_board(position)
        if not flat_board:
            continue
        current_player = int(position.get("current_player", 1) or 1)
        decision = _loaded_model_decision(
            flat_board,
            current_player,
            board_size,
            win_len,
            model,
            decision_mode=decision_mode,
        )
        chosen_move = int(decision.get("move", -1))
        family = str(position.get("exactFamily", "") or position.get("motif", "") or suite_name)
        total += 1
        family_total[family] = int(family_total.get(family, 0)) + 1
        player_total[current_player] = int(player_total.get(current_player, 0)) + 1
        if chosen_move == target_move:
            correct += 1
            family_correct[family] = int(family_correct.get(family, 0)) + 1
            player_correct[current_player] = int(player_correct.get(current_player, 0)) + 1
            continue
        if collect_failures:
            failures.append(
                _build_decision_suite_failure(
                    position,
                    suite_name=suite_name,
                    decision_mode=decision_mode,
                    chosen_move=chosen_move,
                    target_move=target_move,
                )
            )

    accuracy = (correct / total) if total else 0.0
    family_recall = {
        family: round(float(family_correct.get(family, 0)) / float(count), 4)
        for family, count in family_total.items()
        if count > 0
    }
    worst_family_name = min(family_recall, key=family_recall.get) if family_recall else None
    p1_total = int(player_total.get(1, 0))
    p2_total = int(player_total.get(2, 0))
    p1_recall = (int(player_correct.get(1, 0)) / p1_total) if p1_total else 0.0
    p2_recall = (int(player_correct.get(2, 0)) / p2_total) if p2_total else 0.0
    return {
        "correct": correct,
        "total": total,
        "accuracy": round(accuracy, 4),
        "familyRecall": family_recall,
        "familyCount": len(family_recall),
        "worstFamilyRecall": round(float(family_recall.get(worst_family_name, 0.0)), 4) if worst_family_name else 0.0,
        "worstFamilyName": worst_family_name,
        "p1Recall": round(p1_recall, 4),
        "p2Recall": round(p2_recall, 4),
    }, failures


async def _run_validation_snapshot(
    model: Any,
    holdout_positions: list[dict[str, Any]],
    frozen_suites: dict[str, list[dict[str, Any]]],
    device: Any,
    callback: TRAIN_CALLBACK,
    *,
    variant: str,
    cycle: int,
    total_cycles: int,
    previous_holdout: dict[str, float] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "phase": "holdout",
        "stage": "validation",
        "variant": variant,
        "cycle": cycle,
        "totalCycles": total_cycles,
        "iteration": cycle,
        "totalIterations": total_cycles,
    }

    holdout_metrics = _evaluate_supervised_dataset(model, holdout_positions, device) if holdout_positions else {}
    if holdout_metrics:
        payload.update({
            "holdoutPolicyAcc": holdout_metrics["policyTop1Acc"],
            "holdoutTeacherMass": holdout_metrics["teacherMassOnPred"],
            "holdoutPolicyKL": holdout_metrics["policyKL"],
            "holdoutValueMAE": holdout_metrics["valueMAE"],
            "holdoutValueSignAgreement": holdout_metrics["valueSignAgreement"],
            "holdoutLegalTargetRate": holdout_metrics.get("legalTargetRate"),
            "holdoutDuplicateMergeRate": holdout_metrics.get("duplicateMergeRate"),
            "holdoutPolicyMassMeanAbsError": holdout_metrics.get("policyMassMeanAbsError"),
            "holdoutNonFiniteTargetRate": holdout_metrics.get("nonFiniteTargetRate"),
            "holdoutPositions": len(holdout_positions),
        })
        prev_acc = previous_holdout.get("holdoutPolicyAcc") if previous_holdout else None
        if prev_acc is not None:
            payload["holdoutDeltaAcc"] = round(holdout_metrics["policyTop1Acc"] - prev_acc, 2)

    benchmark_results: dict[str, float] = {}
    for name, suite_positions in frozen_suites.items():
        if not suite_positions:
            continue
        suite_metrics = _evaluate_supervised_dataset(model, suite_positions, device)
        benchmark_results[name] = suite_metrics.get("policyTop1Acc", 0.0)

    if benchmark_results:
        payload["frozenBlockAcc"] = round(benchmark_results.get("block", 0.0), 2)
        payload["frozenWinAcc"] = round(benchmark_results.get("win", 0.0), 2)
        if "exact" in benchmark_results:
            payload["frozenExactAcc"] = round(benchmark_results["exact"], 2)
        if "mid" in benchmark_results:
            payload["frozenMidAcc"] = round(benchmark_results["mid"], 2)
        if "late" in benchmark_results:
            payload["frozenLateAcc"] = round(benchmark_results["late"], 2)

    win_suite = frozen_suites.get("win") or []
    block_suite = frozen_suites.get("block") or []
    exact_suite = frozen_suites.get("exact") or []
    if win_suite:
        pure_win_metrics, _ = _evaluate_decision_suite(
            model,
            win_suite,
            int(win_suite[0].get("board_size", 0) or 0),
            4 if variant == "ttt5" else (3 if variant == "ttt3" else 5),
            decision_mode="pure",
            suite_name="win",
        )
        hybrid_win_metrics, _ = _evaluate_decision_suite(
            model,
            win_suite,
            int(win_suite[0].get("board_size", 0) or 0),
            4 if variant == "ttt5" else (3 if variant == "ttt3" else 5),
            decision_mode="hybrid",
            suite_name="win",
        )
        payload["pureFrozenWinRecall"] = round(pure_win_metrics["accuracy"] * 100.0, 2)
        payload["hybridFrozenWinRecall"] = round(hybrid_win_metrics["accuracy"] * 100.0, 2)
    if block_suite:
        pure_block_metrics, _ = _evaluate_decision_suite(
            model,
            block_suite,
            int(block_suite[0].get("board_size", 0) or 0),
            4 if variant == "ttt5" else (3 if variant == "ttt3" else 5),
            decision_mode="pure",
            suite_name="block",
        )
        hybrid_block_metrics, _ = _evaluate_decision_suite(
            model,
            block_suite,
            int(block_suite[0].get("board_size", 0) or 0),
            4 if variant == "ttt5" else (3 if variant == "ttt3" else 5),
            decision_mode="hybrid",
            suite_name="block",
        )
        payload["pureFrozenBlockRecall"] = round(pure_block_metrics["accuracy"] * 100.0, 2)
        payload["hybridFrozenBlockRecall"] = round(hybrid_block_metrics["accuracy"] * 100.0, 2)
    if exact_suite:
        pure_exact_metrics, _ = _evaluate_decision_suite(
            model,
            exact_suite,
            int(exact_suite[0].get("board_size", 0) or 0),
            4 if variant == "ttt5" else (3 if variant == "ttt3" else 5),
            decision_mode="pure",
            suite_name="exact",
        )
        hybrid_exact_metrics, _ = _evaluate_decision_suite(
            model,
            exact_suite,
            int(exact_suite[0].get("board_size", 0) or 0),
            4 if variant == "ttt5" else (3 if variant == "ttt3" else 5),
            decision_mode="hybrid",
            suite_name="exact",
        )
        payload["exactPackSize"] = len(exact_suite)
        payload["exactPackFamilyCount"] = int(pure_exact_metrics.get("familyCount", 0) or 0)
        payload["pureExactTrapRecall"] = round(pure_exact_metrics["accuracy"] * 100.0, 2)
        payload["hybridExactTrapRecall"] = round(hybrid_exact_metrics["accuracy"] * 100.0, 2)
        payload["pureWorstTrapFamilyRecall"] = round(float(pure_exact_metrics.get("worstFamilyRecall", 0.0) or 0.0) * 100.0, 2)
        payload["hybridWorstTrapFamilyRecall"] = round(float(hybrid_exact_metrics.get("worstFamilyRecall", 0.0) or 0.0) * 100.0, 2)
        payload["pureWorstTrapFamily"] = pure_exact_metrics.get("worstFamilyName")
        payload["hybridWorstTrapFamily"] = hybrid_exact_metrics.get("worstFamilyName")
        payload["pureP1TrapRecall"] = round(float(pure_exact_metrics.get("p1Recall", 0.0) or 0.0) * 100.0, 2)
        payload["pureP2TrapRecall"] = round(float(pure_exact_metrics.get("p2Recall", 0.0) or 0.0) * 100.0, 2)
        payload["hybridP1TrapRecall"] = round(float(hybrid_exact_metrics.get("p1Recall", 0.0) or 0.0) * 100.0, 2)
        payload["hybridP2TrapRecall"] = round(float(hybrid_exact_metrics.get("p2Recall", 0.0) or 0.0) * 100.0, 2)
        payload["pureExactFamilyRecall"] = {
            family: round(float(recall) * 100.0, 2)
            for family, recall in dict(pure_exact_metrics.get("familyRecall") or {}).items()
        }
        payload["hybridExactFamilyRecall"] = {
            family: round(float(recall) * 100.0, 2)
            for family, recall in dict(hybrid_exact_metrics.get("familyRecall") or {}).items()
        }

    await callback({"type": "train.progress", "payload": payload})
    return payload


# ---------------------------------------------------------------------------
# Self-play game generation (bootstrap + self-play)
# ---------------------------------------------------------------------------


async def _batched_model_forward(
    game_states: list[dict[str, Any]],
    board_size: int,
    model: Any,
    device: Any,
) -> tuple[list[list[float]], list[float]]:
    """Run one batched policy/value inference for the active game states."""
    from trainer_lab.data.encoder import board_to_tensor
    import torch.nn.functional as F

    if not game_states:
        return [], []

    pos_dicts = []
    for state in game_states:
        pos_dicts.append({
            "board_size": board_size,
            "board": _flat_to_board2d(state["board"], board_size),
            "current_player": state["current"],
            "last_move": list(divmod(state["last_move"], board_size)) if state["last_move"] >= 0 else None,
        })

    planes = torch.stack([board_to_tensor(pos) for pos in pos_dicts])
    planes = planes.to(device, non_blocking=device.type == "cuda")
    if device.type == "cuda":
        planes = planes.contiguous(memory_format=torch.channels_last)

    model.eval()
    with torch.inference_mode():
        logits, values = model(planes)

    logits_cpu = logits.detach().cpu()
    values_cpu = values.detach().cpu().view(-1).tolist()
    policies: list[list[float]] = []

    for idx, state in enumerate(game_states):
        legal = [move for move, cell in enumerate(state["board"]) if cell == 0]
        policy = [0.0] * 256
        if legal:
            masked = torch.full_like(logits_cpu[idx], float("-inf"))
            for move in legal:
                masked[_policy_cell_index(move, board_size)] = logits_cpu[idx, _policy_cell_index(move, board_size)]
            probs = F.softmax(masked, dim=0)
            for move in legal:
                policy[_policy_cell_index(move, board_size)] = probs[_policy_cell_index(move, board_size)].item()
        policies.append(policy)
    return policies, values_cpu


def _select_selfplay_move(
    board: list[int],
    board_size: int,
    win_len: int,
    current: int,
    policy256: list[float],
    move_count: int,
    *,
    temperature_moves: int = 10,
    dirichlet_weight: float = 0.15,
) -> int:
    legal = [move for move, cell in enumerate(board) if cell == 0]
    if not legal:
        return -1

    winning_move = _find_immediate_move(board, board_size, win_len, current)
    if winning_move is not None:
        return winning_move

    blocking_move = _find_immediate_move(board, board_size, win_len, 2 if current == 1 else 1)
    if blocking_move is not None:
        return blocking_move

    weights = [max(policy256[_policy_cell_index(move, board_size)], 0.0) for move in legal]
    total = sum(weights)
    if total <= 0:
        return random.choice(legal)

    if move_count < temperature_moves:
        noisy = list(weights)
        if len(noisy) > 1 and dirichlet_weight > 0:
            noise = torch.distributions.dirichlet.Dirichlet(
                torch.full((len(noisy),), 0.3, dtype=torch.float32)
            ).sample().tolist()
            noisy = [
                (1.0 - dirichlet_weight) * w + dirichlet_weight * n
                for w, n in zip(noisy, noise)
            ]
        return random.choices(legal, weights=noisy, k=1)[0]

    return legal[max(range(len(legal)), key=lambda idx: weights[idx])]


def _build_train_pool(
    latest_positions: list[dict[str, Any]],
    replay_positions: list[dict[str, Any]],
    *,
    data_count: int,
    seed_positions: list[dict[str, Any]] | None = None,
    minimax_positions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Mix fresh self-play, replay, tactical seed, and minimax positions.

    Hardened quotas:
      20% tactical seed (min)
      20% minimax seed (min)
      60% recent self-play (remaining)
    Replay fills gaps if any bucket is too small.
    """
    if data_count <= 0:
        return []

    seed_positions = seed_positions or []
    minimax_positions = minimax_positions or []

    # Minimum quotas: 20% tactical, 20% minimax, rest = latest + replay
    tactical_quota = min(len(seed_positions), max(data_count * 20 // 100, 0))
    minimax_quota = min(len(minimax_positions), max(data_count * 20 // 100, 0))
    latest_budget = data_count - tactical_quota - minimax_quota
    latest_quota = min(len(latest_positions), latest_budget)
    remaining = max(latest_budget - latest_quota, 0)
    replay_quota = min(len(replay_positions), remaining)

    pool: list[dict[str, Any]] = []
    if latest_quota > 0 and latest_positions:
        pool.extend(random.sample(latest_positions, latest_quota))
    if replay_quota > 0 and replay_positions:
        pool.extend(random.sample(replay_positions, replay_quota))
    if tactical_quota > 0 and seed_positions:
        pool.extend(random.sample(seed_positions, tactical_quota))
    if minimax_quota > 0 and minimax_positions:
        pool.extend(random.sample(minimax_positions, minimax_quota))

    # Backfill if any bucket was too small.
    if len(pool) < data_count:
        leftovers = latest_positions + replay_positions + seed_positions + minimax_positions
        needed = min(data_count - len(pool), len(leftovers))
        if needed > 0:
            pool.extend(random.sample(leftovers, needed))

    return pool[:data_count]


async def _play_selfplay_games_batched(
    num_games: int,
    board_size: int,
    win_len: int,
    model: Any,
    device: Any,
    callback: TRAIN_CALLBACK,
    phase: str,
    *,
    teacher_mode: str = "policy",
    completed_phases: list[str] | None = None,
    iteration: int = 0,
    total_iterations: int = 0,
    overall_percent_base: float = 0.0,
    overall_percent_range: float = 10.0,
    runtime_flags: dict[str, bool] | None = None,
    **extra_fields: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Generate self-play games using batched model inference.

    This is much more GPU-friendly than per-move single-state inference and is
    the default path for bootstrap and large-board self-play.
    """
    positions: list[dict[str, Any]] = []
    stats = {"wins": 0, "losses": 0, "draws": 0}
    started_at = time.monotonic()
    stage = "self_play_game"
    last_emit_at = 0.0
    last_gpu_probe = 0.0
    live_gpu = get_gpu_info()

    games: list[dict[str, Any]] = []
    for _ in range(num_games):
        games.append({
            "board": [0] * (board_size * board_size),
            "current": 1,
            "last_move": -1,
            "history": [],
            "finished": False,
        })

    finished_games = 0
    while finished_games < num_games:
        active = [game for game in games if not game["finished"]]
        if not active:
            break

        policies256, _values = await _batched_model_forward(active, board_size, model, device)

        for game, model_policy in zip(active, policies256):
            board = game["board"]
            current = game["current"]
            last_move = game["last_move"]
            move_count = sum(1 for cell in board if cell != 0)

            # Always use model policy as target (no online minimax)
            target_policy = list(model_policy)

            winning_move = _find_immediate_move(board, board_size, win_len, current)
            blocking_move = None if winning_move is not None else _find_immediate_move(
                board, board_size, win_len, 2 if current == 1 else 1
            )
            if winning_move is not None:
                target_policy = _one_hot_policy(winning_move, board_size)
            elif blocking_move is not None:
                target_policy = _one_hot_policy(blocking_move, board_size)

            game["history"].append({
                "board_size": board_size,
                "board": _flat_to_board2d(board, board_size),
                "current_player": current,
                "last_move": list(divmod(last_move, board_size)) if last_move >= 0 else None,
                "policy": target_policy,
                "value": 0.0,
            })

            move = _select_selfplay_move(
                board,
                board_size,
                win_len,
                current,
                model_policy,
                move_count,
            )
            if move < 0:
                game["finished"] = True
                finished_games += 1
                stats["draws"] += 1
                continue

            board[move] = current
            game["last_move"] = move
            winner = _nxn_winner(board, board_size, win_len, move)
            if winner != 0 or all(cell != 0 for cell in board):
                result = winner
                for pos in game["history"]:
                    if result == 0:
                        pos["value"] = 0.0
                    elif pos["current_player"] == result:
                        pos["value"] = 1.0
                    else:
                        pos["value"] = -1.0
                positions.extend(game["history"])
                game["finished"] = True
                finished_games += 1
                if result == 1:
                    stats["wins"] += 1
                elif result == 2:
                    stats["losses"] += 1
                else:
                    stats["draws"] += 1
            else:
                game["current"] = 2 if current == 1 else 1

        now = time.monotonic()
        force_emit = finished_games >= num_games
        if _should_emit_progress(now, last_emit_at, force=force_emit):
            elapsed = max(now - started_at, 0.01)
            speed = finished_games / elapsed
            pct = overall_percent_base + overall_percent_range * (finished_games / max(num_games, 1)) * 0.6
            live_gpu, last_gpu_probe = _maybe_refresh_gpu_info(now, last_gpu_probe, live_gpu)
            telemetry = _extract_telemetry(live_gpu)
            rf = runtime_flags or {}
            await callback({
                "type": "train.progress",
                "payload": {
                    "phase": phase,
                    "stage": stage,
                    "variant": extra_fields.get("variant", ""),
                    "completedPhases": completed_phases or [],
                    "game": finished_games,
                    "totalGames": num_games,
                    "iteration": iteration,
                    "totalIterations": total_iterations,
                    "selfPlayStats": stats,
                    "positions": len(positions),
                    "epoch": 0,
                    "totalEpochs": 0,
                    "percent": round(pct, 1),
                    "elapsed": round(elapsed, 1),
                    "eta": round((elapsed / max(finished_games, 1)) * max(num_games - finished_games, 0), 1),
                    "speed": round(speed, 2),
                    "speedUnit": "g/s",
                    "teacherMode": teacher_mode,
                    "mixedPrecision": rf.get("mixedPrecision", False),
                    "tf32": rf.get("tf32", False),
                    "torchCompile": rf.get("torchCompile", False),
                    "compileMode": rf.get("compileMode"),
                    "gpu": live_gpu,
                    **telemetry,
                    **{k: v for k, v in extra_fields.items() if k not in ("variant",)},
                },
            })
            last_emit_at = now
            await asyncio.sleep(0)

    return positions, stats


async def _play_selfplay_games(
    num_games: int,
    board_size: int,
    win_len: int,
    model: Any,
    device: Any,
    callback: TRAIN_CALLBACK,
    phase: str,
    use_mcts: bool = False,
    mcts_simulations: int = 16,
    completed_phases: list[str] | None = None,
    iteration: int = 0,
    total_iterations: int = 0,
    overall_percent_base: float = 0.0,
    overall_percent_range: float = 10.0,
    runtime_flags: dict[str, bool] | None = None,
    **extra_fields: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Play self-play games and return (positions, selfPlayStats)."""
    import torch.nn.functional as F
    from trainer_lab.data.encoder import board_to_tensor

    positions: list[dict[str, Any]] = []
    stats = {"wins": 0, "losses": 0, "draws": 0}
    started_at = time.monotonic()
    stage = "mcts_game" if use_mcts else "generating"
    last_emit_at = 0.0
    last_gpu_probe = 0.0
    live_gpu = get_gpu_info()

    for g in range(num_games):
        board = [0] * (board_size * board_size)
        current = 1
        last_move = -1
        history: list[dict[str, Any]] = []
        winner = 0

        for _ in range(board_size * board_size):
            empty = [i for i, c in enumerate(board) if c == 0]
            if not empty:
                break

            # Choose move
            mcts_policy = None  # Reset each turn
            if use_mcts and model is not None:
                from trainer_lab.self_play.player import GameState, mcts_search
                gs = GameState(board_size)
                gs.board = [[board[r * board_size + c] for c in range(board_size)] for r in range(board_size)]
                gs.current_player = 1 if current == 1 else 2
                gs.move_count = sum(1 for x in board if x != 0)
                if last_move >= 0:
                    gs.last_move = divmod(last_move, board_size)
                model.eval()
                # Run MCTS in thread pool to avoid blocking event loop
                mcts_policy, _ = await asyncio.to_thread(
                    mcts_search, gs, model, device, num_simulations=mcts_simulations
                )
                # Sample with temperature
                total_p = sum(mcts_policy)
                if total_p > 0:
                    move = random.choices(range(len(mcts_policy)), weights=mcts_policy, k=1)[0]
                else:
                    move = random.choice(empty)
            elif model is not None:
                # Model inference without MCTS
                board_2d = [[0] * board_size for _ in range(board_size)]
                for idx in range(board_size * board_size):
                    r, c = divmod(idx, board_size)
                    v = board[idx]
                    if v == current:
                        board_2d[r][c] = 1
                    elif v != 0:
                        board_2d[r][c] = 2
                pos_dict = {"board_size": board_size, "board": board_2d, "current_player": 1, "last_move": list(divmod(last_move, board_size)) if last_move >= 0 else None}
                tensor = board_to_tensor(pos_dict).unsqueeze(0).to(device)
                model.eval()
                with torch.no_grad():
                    logits, _ = model(tensor)
                logits = logits.squeeze(0).cpu()
                mask = torch.full_like(logits, float("-inf"))
                for idx in empty:
                    r, c = divmod(idx, board_size)
                    mask[r * 16 + c] = 0.0
                probs = F.softmax(logits + mask, dim=0)
                move = max(empty, key=lambda i: probs[divmod(i, board_size)[0] * 16 + divmod(i, board_size)[1]].item())
            else:
                move = random.choice(empty)

            # Build policy target
            if use_mcts and mcts_policy is not None:
                # MCTS: use visit count distribution
                policy = [0.0] * 256
                for idx in range(board_size * board_size):
                    r, c = divmod(idx, board_size)
                    policy[r * 16 + c] = mcts_policy[idx] if idx < len(mcts_policy) else 0.0
            else:
                # No MCTS: uniform over legal moves (fallback)
                policy = [0.0] * 256
                for idx in empty:
                    r, c = divmod(idx, board_size)
                    policy[r * 16 + c] = 1.0 / len(empty)

            # Record position
            board_2d = [[0] * board_size for _ in range(board_size)]
            for idx in range(board_size * board_size):
                r, c = divmod(idx, board_size)
                board_2d[r][c] = board[idx]
            history.append({
                "board_size": board_size, "board": board_2d,
                "current_player": current, "last_move": list(divmod(last_move, board_size)) if last_move >= 0 else None,
                "policy": policy, "value": 0.0,
            })

            board[move] = current
            last_move = move
            winner = _nxn_winner(board, board_size, win_len, move)
            if winner != 0:
                break
            current = 2 if current == 1 else 1

        # Fill outcome values
        for pos in history:
            if winner == 0:
                pos["value"] = 0.0
            elif pos["current_player"] == winner:
                pos["value"] = 1.0
            else:
                pos["value"] = -1.0
        positions.extend(history)

        if winner == 1:
            stats["wins"] += 1
        elif winner == 2:
            stats["losses"] += 1
        else:
            stats["draws"] += 1

        # Emit progress every 2 games (or every game for slow MCTS)
        emit_interval = 1 if use_mcts else 2
        now = time.monotonic()
        if ((g + 1) % emit_interval == 0 and _should_emit_progress(now, last_emit_at)) or g + 1 == num_games:
            elapsed = now - started_at
            game_pct = (g + 1) / num_games
            speed = (g + 1) / max(elapsed, 0.01)
            pct = overall_percent_base + overall_percent_range * game_pct * 0.6

            live_gpu, last_gpu_probe = _maybe_refresh_gpu_info(now, last_gpu_probe, live_gpu)
            telemetry = _extract_telemetry(live_gpu)
            rf = runtime_flags or {}
            await callback({
                "type": "train.progress",
                "payload": {
                    "phase": phase, "stage": stage, "variant": extra_fields.get("variant", ""),
                    "completedPhases": completed_phases or [],
                    "game": g + 1, "totalGames": num_games,
                    "iteration": iteration, "totalIterations": total_iterations,
                    "selfPlayStats": stats,
                    "positions": len(positions),
                    "epoch": 0, "totalEpochs": 0,
                    "percent": round(pct, 1),
                    "elapsed": round(elapsed, 1),
                    "eta": round((elapsed / max(g + 1, 1)) * max(num_games - g - 1, 0), 1),
                    "speed": round(speed, 2), "speedUnit": "g/s",
                    "mixedPrecision": rf.get("mixedPrecision", False),
                    "tf32": rf.get("tf32", False),
                    "torchCompile": rf.get("torchCompile", False),
                    "compileMode": rf.get("compileMode"),
                    "gpu": live_gpu, **telemetry,
                    **{k: v for k, v in extra_fields.items() if k not in ("variant",)},
                },
            })
            last_emit_at = now
            await asyncio.sleep(0)

    return positions, stats


# ---------------------------------------------------------------------------
# Training epoch loop (extracted for reuse across phases)
# ---------------------------------------------------------------------------


async def _run_training_epochs(
    model: Any,
    positions: list[dict[str, Any]],
    num_epochs: int,
    batch_size: int,
    device: Any,
    runtime_flags: dict[str, bool],
    callback: TRAIN_CALLBACK,
    metrics_history: list[dict[str, Any]],
    phase: str,
    stage: str = "training",
    completed_phases: list[str] | None = None,
    iteration: int = 0,
    total_iterations: int = 0,
    selfplay_stats: dict[str, int] | None = None,
    overall_started_at: float = 0.0,
    overall_percent_base: float = 0.0,
    overall_percent_range: float = 10.0,
    augment: bool = False,
    **extra_fields: Any,
) -> None:
    """Run training epochs on positions, streaming progress to callback."""
    from trainer_lab.training.loss import GomokuLoss
    from trainer_lab.training.metrics import (
        policy_accuracy, value_mae,
        teacher_mass_on_pred, value_sign_agreement, policy_entropy, policy_kl_divergence,
    )

    if not positions:
        return

    all_planes, all_policy, all_value, effective_positions = _materialize_training_tensors(
        positions,
        augment=augment,
    )
    dataset = TensorDataset(all_planes, all_policy, all_value)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                        pin_memory=device.type == "cuda")

    criterion = GomokuLoss(weight_value=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    use_amp = runtime_flags["mixedPrecision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()

    total_batches = len(loader)
    total_steps = max(total_batches * num_epochs, 1)
    model_params = _count_model_parameters(model)
    samples_seen = 0
    last_gpu_probe = 0.0
    last_emit_at = 0.0
    live_gpu = get_gpu_info()
    phase_start = time.monotonic()

    for epoch in range(1, num_epochs + 1):
        ep_loss = ep_ploss = ep_vloss = ep_acc = ep_mae = 0.0
        ep_tmop = ep_vsa = ep_entropy = ep_kl = 0.0
        ep_samples = 0

        for bi, (planes, pol_t, val_t) in enumerate(loader, 1):
            step_t = time.perf_counter()
            planes = planes.to(device, non_blocking=True)
            pol_t = pol_t.to(device, non_blocking=True)
            val_t = val_t.to(device, non_blocking=True)
            if device.type == "cuda":
                planes = planes.contiguous(memory_format=torch.channels_last)

            legal_mask = planes[:, 2].reshape(planes.size(0), -1)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, vpred = model(planes)
                loss, pl, vl = criterion(logits, vpred, pol_t, val_t, legal_mask=legal_mask)

            if use_amp:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()

            bt = max(time.perf_counter() - step_t, 1e-6)

            bs = planes.size(0)
            det_logits = logits.detach()
            det_vpred = vpred.detach()
            acc = policy_accuracy(det_logits, pol_t, legal_mask=legal_mask)
            mae = value_mae(det_vpred, val_t)
            tmop = teacher_mass_on_pred(det_logits, pol_t, legal_mask=legal_mask)
            vsa = value_sign_agreement(det_vpred, val_t)
            p_entropy = policy_entropy(det_logits, legal_mask=legal_mask)
            p_kl = policy_kl_divergence(det_logits, pol_t, legal_mask=legal_mask)

            ep_loss += loss.item() * bs; ep_ploss += pl.item() * bs; ep_vloss += vl.item() * bs
            ep_acc += acc * bs; ep_mae += mae * bs; ep_samples += bs; samples_seen += bs
            ep_tmop += tmop * bs; ep_vsa += vsa * bs
            ep_entropy += p_entropy * bs; ep_kl += p_kl * bs

            a_loss = ep_loss / max(ep_samples, 1)
            a_acc = (ep_acc / max(ep_samples, 1)) * 100.0
            a_mae = ep_mae / max(ep_samples, 1)
            a_tmop = ep_tmop / max(ep_samples, 1)
            a_vsa = (ep_vsa / max(ep_samples, 1)) * 100.0
            a_entropy = ep_entropy / max(ep_samples, 1)
            a_kl = ep_kl / max(ep_samples, 1)

            done_steps = (epoch - 1) * total_batches + bi
            elapsed = max(time.monotonic() - overall_started_at, 0.01)
            phase_elapsed = max(time.monotonic() - phase_start, 0.01)
            sps = samples_seen / phase_elapsed
            pct = overall_percent_base + overall_percent_range * (0.6 + 0.4 * done_steps / total_steps)

            now = time.monotonic()
            force_emit = bi == total_batches
            if _should_emit_progress(now, last_emit_at, force=force_emit):
                live_gpu, last_gpu_probe = _maybe_refresh_gpu_info(now, last_gpu_probe, live_gpu)
                telemetry = _extract_telemetry(live_gpu)

                cur_hist = metrics_history + [{
                    "epoch": epoch,
                    "loss": round(a_loss, 6),
                    "acc": round(a_acc, 2),
                    "mae": round(a_mae, 6),
                }]
                await callback({
                    "type": "train.progress",
                    "payload": {
                        "phase": phase, "stage": stage, "variant": extra_fields.get("variant", ""),
                        "completedPhases": completed_phases or [],
                        "iteration": iteration, "totalIterations": total_iterations,
                        "selfPlayStats": selfplay_stats or {},
                        "epoch": epoch, "totalEpochs": num_epochs, "epochs": num_epochs,
                        "batch": bi, "totalBatches": total_batches, "batchesPerEpoch": total_batches,
                        "loss": round(a_loss, 6), "policyLoss": round(ep_ploss / max(ep_samples, 1), 6),
                        "valueLoss": round(ep_vloss / max(ep_samples, 1), 6),
                        "accuracy": round(a_acc, 2), "acc": round(a_acc, 2),
                        "policyTop1Acc": round(a_acc, 2),
                        "teacherMassOnPred": round(a_tmop, 4),
                        "valueSignAgreement": round(a_vsa, 2),
                        "policyEntropy": round(a_entropy, 4),
                        "policyKL": round(a_kl, 6),
                        "mae": round(a_mae, 6),
                        "percent": round(pct, 1), "epochPercent": round(bi / max(total_batches, 1) * 100, 1),
                        "elapsed": round(elapsed, 1),
                        "eta": round((phase_elapsed / done_steps) * max(total_steps - done_steps, 0), 1),
                        "speed": round(sps, 2), "speedUnit": "samples/s",
                        "samplesPerSec": round(sps, 2), "batchesPerSec": round(done_steps / phase_elapsed, 3),
                        "batchTimeMs": round(bt * 1000, 2),
                        "positions": len(positions), "effectivePositions": effective_positions,
                        "batchSize": batch_size, "learningRate": 2e-3,
                        "modelParams": model_params,
                        "device": device.type, "deviceName": live_gpu.get("name", str(device)),
                        "mixedPrecision": runtime_flags["mixedPrecision"],
                        "tf32": runtime_flags["tf32"], "channelsLast": runtime_flags["channelsLast"],
                        "torchCompile": runtime_flags.get("torchCompile", False),
                        "compileMode": runtime_flags.get("compileMode"),
                        "metricsHistory": cur_hist, "gpu": live_gpu, **telemetry, **extra_fields,
                    },
                })
                last_emit_at = now
                await asyncio.sleep(0)

        metrics_history.append({
            "epoch": len(metrics_history) + 1,
            "loss": round(ep_loss / max(ep_samples, 1), 6),
            "acc": round((ep_acc / max(ep_samples, 1)) * 100.0, 2),
            "mae": round(ep_mae / max(ep_samples, 1), 6),
        })


async def _run_training_steps(
    model: Any,
    positions: list[dict[str, Any]],
    max_steps: int,
    batch_size: int,
    device: Any,
    runtime_flags: dict[str, bool],
    callback: TRAIN_CALLBACK,
    metrics_history: list[dict[str, Any]],
    phase: str,
    stage: str = "training",
    completed_phases: list[str] | None = None,
    iteration: int = 0,
    total_iterations: int = 0,
    selfplay_stats: dict[str, int] | None = None,
    overall_started_at: float = 0.0,
    overall_percent_base: float = 0.0,
    overall_percent_range: float = 10.0,
    augment: bool = False,
    augment_mode: str = "full",
    time_budget: float = 0.0,
    optimizer_type: str = "adam",
    learning_rate: float = 2e-3,
    **extra_fields: Any,
) -> None:
    """Step-based training with manual batching — keeps GPU busy longer."""
    from trainer_lab.training.loss import GomokuLoss
    from trainer_lab.training.metrics import (
        policy_accuracy, value_mae,
        teacher_mass_on_pred, value_sign_agreement, policy_entropy, policy_kl_divergence,
    )

    if not positions or max_steps <= 0:
        return

    # Pre-stack tensors on CPU with pin_memory
    all_planes, all_policy, all_value, effective_positions = _materialize_training_tensors(
        positions,
        augment=augment,
        augment_mode=augment_mode,
    )
    if device.type == "cuda":
        all_planes = all_planes.pin_memory()
        all_policy = all_policy.pin_memory()
        all_value = all_value.pin_memory()

    n = all_planes.size(0)
    use_random_augment = augment and augment_mode == "random"

    criterion = GomokuLoss(weight_value=0.5)
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    use_amp = runtime_flags["mixedPrecision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()

    model_params = _count_model_parameters(model)
    samples_seen = 0
    last_gpu_probe = 0.0
    last_emit_at = 0.0
    live_gpu = get_gpu_info()
    phase_start = time.monotonic()
    step = 0
    epoch = 0
    cum_loss = cum_acc = cum_mae = 0.0
    cum_tmop = cum_vsa = cum_entropy = cum_kl = 0.0
    cum_samples = 0

    # EMA early stop state
    ema_acc = 0.0
    prev_ema = 0.0
    plateau_count = 0
    ema_alpha = 0.3
    min_steps_for_stop = max(5, max_steps // 5)

    while step < max_steps:
        epoch += 1
        perm = torch.randperm(n)
        for bi_start in range(0, n, batch_size):
            if step >= max_steps:
                break
            idx = perm[bi_start : bi_start + batch_size]
            planes = all_planes[idx].to(device, non_blocking=True)
            pol_t = all_policy[idx].to(device, non_blocking=True)
            val_t = all_value[idx].to(device, non_blocking=True)
            if use_random_augment:
                _aug_board_size = extra_fields.get("boardSize", 16)
                planes, pol_t = _apply_random_d4_batch(planes, pol_t, board_size=_aug_board_size)
            if device.type == "cuda":
                planes = planes.contiguous(memory_format=torch.channels_last)

            legal_mask = planes[:, 2].reshape(planes.size(0), -1)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, vpred = model(planes)
                loss, pl, vl = criterion(logits, vpred, pol_t, val_t, legal_mask=legal_mask)

            if use_amp:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()

            step += 1
            bs = planes.size(0)
            det_logits = logits.detach()
            det_vpred = vpred.detach()
            acc = policy_accuracy(det_logits, pol_t, legal_mask=legal_mask)
            mae = value_mae(det_vpred, val_t)
            tmop = teacher_mass_on_pred(det_logits, pol_t, legal_mask=legal_mask)
            vsa = value_sign_agreement(det_vpred, val_t)
            p_entropy = policy_entropy(det_logits, legal_mask=legal_mask)
            p_kl = policy_kl_divergence(det_logits, pol_t, legal_mask=legal_mask)

            cum_loss += loss.item() * bs; cum_acc += acc * bs; cum_mae += mae * bs
            cum_tmop += tmop * bs; cum_vsa += vsa * bs
            cum_entropy += p_entropy * bs; cum_kl += p_kl * bs
            cum_samples += bs; samples_seen += bs

            a_loss = cum_loss / max(cum_samples, 1)
            a_acc = (cum_acc / max(cum_samples, 1)) * 100.0
            a_mae = cum_mae / max(cum_samples, 1)
            a_tmop = cum_tmop / max(cum_samples, 1)
            a_vsa = (cum_vsa / max(cum_samples, 1)) * 100.0
            a_entropy = cum_entropy / max(cum_samples, 1)
            a_kl = cum_kl / max(cum_samples, 1)

            # EMA early stop: time budget or plateau
            ema_acc = ema_alpha * acc + (1 - ema_alpha) * ema_acc
            if abs(ema_acc - prev_ema) < 0.005:
                plateau_count += 1
            else:
                plateau_count = 0
            prev_ema = ema_acc
            phase_elapsed_check = time.monotonic() - phase_start
            if time_budget > 0 and step >= min_steps_for_stop and phase_elapsed_check >= time_budget:
                step = max_steps  # exit
                break
            if plateau_count >= 3 and step >= min_steps_for_stop * 2 and ema_acc >= 0.50:
                step = max_steps  # exit
                break

            now = time.monotonic()
            if _should_emit_progress(now, last_emit_at, force=(step >= max_steps)):
                live_gpu, last_gpu_probe = _maybe_refresh_gpu_info(now, last_gpu_probe, live_gpu)
                telemetry = _extract_telemetry(live_gpu)
                phase_elapsed = max(now - phase_start, 0.01)
                sps = samples_seen / phase_elapsed
                pct = overall_percent_base + overall_percent_range * (0.6 + 0.4 * step / max_steps)
                elapsed = max(now - overall_started_at, 0.01)

                cur_hist = metrics_history + [{"epoch": epoch, "loss": round(a_loss, 6), "acc": round(a_acc, 2), "mae": round(a_mae, 6)}]
                await callback({
                    "type": "train.progress",
                    "payload": {
                        "phase": phase, "stage": stage, "variant": extra_fields.get("variant", ""),
                        "completedPhases": completed_phases or [],
                        "iteration": iteration, "totalIterations": total_iterations,
                        "selfPlayStats": selfplay_stats or {},
                        "step": step, "totalSteps": max_steps,
                        "epoch": epoch, "totalEpochs": max(1, (max_steps * batch_size) // max(n, 1)),
                        "loss": round(a_loss, 6), "accuracy": round(a_acc, 2), "acc": round(a_acc, 2),
                        "policyTop1Acc": round(a_acc, 2),
                        "teacherMassOnPred": round(a_tmop, 4),
                        "valueSignAgreement": round(a_vsa, 2),
                        "policyEntropy": round(a_entropy, 4),
                        "policyKL": round(a_kl, 6),
                        "mae": round(a_mae, 6),
                        "percent": round(pct, 1),
                        "elapsed": round(elapsed, 1),
                        "eta": round((phase_elapsed / step) * max(max_steps - step, 0), 1),
                        "speed": round(sps, 2), "speedUnit": "samples/s",
                        "samplesPerSec": round(sps, 2),
                        "positions": len(positions),
                        "effectivePositions": effective_positions,
                        "batchSize": batch_size, "learningRate": 2e-3,
                        "modelParams": model_params,
                        "device": device.type, "deviceName": live_gpu.get("name", str(device)),
                        "mixedPrecision": runtime_flags["mixedPrecision"],
                        "tf32": runtime_flags["tf32"], "channelsLast": runtime_flags["channelsLast"],
                        "metricsHistory": cur_hist, "gpu": live_gpu, **telemetry,
                    },
                })
                last_emit_at = now
                await asyncio.sleep(0)

        # End-of-epoch metrics
        if cum_samples > 0:
            metrics_history.append({
                "epoch": len(metrics_history) + 1,
                "loss": round(cum_loss / cum_samples, 6),
                "acc": round((cum_acc / cum_samples) * 100.0, 2),
                "mae": round(cum_mae / cum_samples, 6),
            })
            cum_loss = cum_acc = cum_mae = 0.0
            cum_tmop = cum_vsa = cum_entropy = cum_kl = 0.0
            cum_samples = 0


# ---------------------------------------------------------------------------
# Curriculum training: Tactical → Supervised → Bootstrap → Self-Play
# ---------------------------------------------------------------------------


async def train_variant(
    variant: str,
    callback: TRAIN_CALLBACK,
    epochs: int = 8,
    batch_size: int = 256,
    data_count: int = 4000,
    **kwargs: Any,
) -> None:
    """Train with multi-phase curriculum: Tactical → Bootstrap → MCTS iterations."""
    from trainer_lab.config import ModelConfig
    from trainer_lab.models.resnet import PolicyValueResNet
    from gomoku_api.ws.model_registry import ModelRegistry

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime_flags = _prepare_cuda_runtime(device)
    cfg = ModelConfig()

    variant_spec = _shared_resolve_variant_spec(variant)
    board_size, win_len = variant_spec.board_size, variant_spec.win_length
    rapid_mode = board_size <= 5

    registry = ModelRegistry(variant)
    manifest = registry.read_manifest()
    serving_summary = registry.serving_summary(expected_spec=variant_spec)
    model_dir = _ensure_saved_dir(variant)
    requested_model_profile = str(kwargs.get("modelProfile", "auto"))
    current_manifest_profile = current_model_profile_from_manifest(manifest)

    # Determine if this is a fresh model or a resume.
    # Try candidate → champion → legacy, falling through on load failure
    # so a corrupt candidate doesn't nuke the verified champion.
    resume_candidates = [p for p in (registry.candidate_path, registry.champion_path, registry.legacy_path) if p.exists()]
    is_new_model = len(resume_candidates) == 0

    model_profile, (res_filters, res_blocks, value_fc) = _variant_model_hparams(
        board_size,
        cfg,
        variant=variant,
        model_profile=requested_model_profile,
        manifest=manifest,
        variant_spec=variant_spec,
    )
    if requested_model_profile.strip().lower() not in {"", "auto"} and current_manifest_profile and current_manifest_profile != model_profile:
        logger.info(
            "Model profile changed for %s (%s -> %s); starting from fresh weights",
            variant,
            current_manifest_profile,
            model_profile,
        )
        resume_candidates = []
        is_new_model = True

    model = PolicyValueResNet(
        in_channels=cfg.in_channels, res_filters=res_filters,
        res_blocks=res_blocks, policy_filters=cfg.policy_filters,
        value_fc=value_fc, board_max=cfg.board_max,
    )
    loaded = False
    for resume_path in resume_candidates:
        try:
            model.load_state_dict(torch.load(resume_path, map_location="cpu", weights_only=True))
            logger.info("Resumed model from %s", resume_path)
            loaded = True
            break
        except Exception as exc:
            logger.warning("Checkpoint %s failed to load, trying next: %s", resume_path, exc)
    if not loaded and not is_new_model:
        # All existing checkpoints are corrupt — treat as fresh
        is_new_model = True
        logger.warning("All checkpoints corrupt for %s, starting fresh", variant)
    if device.type == "cuda":
        model = model.to(device=device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    model = _maybe_compile_model(model, device, runtime_flags)

    bootstrap_games = int(kwargs.get("bootstrapGames", 64))
    cycle_count = max(1, int(kwargs.get("mctsIterations", 3)))
    exam_games = max(8, int(kwargs.get("mctsGamesPerIter", 32)))
    teacher_backend_requested = str(kwargs.get("teacherBackend", "default"))
    confirm_backend_requested = str(kwargs.get("confirmBackend", "default"))
    prefer_offline_dataset = bool(kwargs.get("preferOfflineDataset", True))
    offline_dataset_limit = max(0, int(kwargs.get("offlineDatasetLimit", 0)))
    foundation_dataset_count = max(0, int(kwargs.get("foundationDatasetCount", 0)))
    if board_size <= 5:
        exam_pairs = max(3, min(8, exam_games // 4))
    else:
        exam_pairs = max(2, exam_games // 2)
    if board_size <= 5:
        teacher_seed_count = max(128, min(300, bootstrap_games * 3))
    else:
        teacher_seed_count = max(256, min(data_count, bootstrap_games * 12))
    if board_size <= 5:
        turbo_steps = 50  # base, overridden per-cycle by adaptive budget
        repair_steps = 25
        cycle_count = max(5, min(12, cycle_count * 3))
        rapid_batch_size = min(batch_size, 256)
        engine_per_cycle = 50
        tactical_per_cycle = 200
        anchor_bank_max = 800
        exam_every = max(2, cycle_count // 4)
        exam_threshold_acc = float(kwargs.get("examThresholdAcc", 30.0))
    else:
        turbo_steps = max(48, min(320, epochs * 8))
        repair_steps = max(32, min(192, max(epochs * 4, turbo_steps // 2)))
        exam_every = 1  # large boards: exam every cycle
        exam_threshold_acc = float(kwargs.get("examThresholdAcc", 0.0))
    if variant_spec.is_curriculum and board_size <= 9:
        bootstrap_games = int(kwargs.get("bootstrapGames", min(bootstrap_games, 48)))
        cycle_count = max(3, min(8, cycle_count))
        exam_games = max(8, min(exam_games, 24))
        exam_pairs = max(2, min(6, exam_games // 4))
        teacher_seed_count = max(192, min(data_count, max(teacher_seed_count, bootstrap_games * 4)))
        turbo_steps = max(40, min(160, epochs * 6))
        repair_steps = max(24, min(96, max(epochs * 3, turbo_steps // 2)))
        exam_every = 1
        exam_threshold_acc = float(kwargs.get("examThresholdAcc", 0.0))
    repair_pool_size = max(512, min(data_count, teacher_seed_count))
    failure_bank_max = max(256, min(data_count * 2, 2048))

    completed_phases: list[str] = []
    metrics_history: list[dict[str, Any]] = []
    winrate_history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    tactical_positions: list[dict[str, Any]] = []
    anchor_positions: list[dict[str, Any]] = []
    holdout_bank: list[dict[str, Any]] = []
    failure_bank: list[dict[str, Any]] = []
    frozen_suites: dict[str, list[dict[str, Any]]] = {}
    started_at = time.monotonic()
    common = {
        "variant": variant,
        "boardSize": board_size,
        "winLength": win_len,
        "curriculumStage": variant_spec.curriculum_stage,
        "variantSpec": variant_spec.to_metadata(),
    }
    variant_metric_metadata = _variant_metric_metadata(variant_spec)
    use_user_corpus = bool(kwargs.get("useCorpus", False)) and variant == "ttt5"
    user_corpus_mode = str(kwargs.get("corpusMode", "quick_repair")).strip().lower()
    user_corpus = None
    user_corpus_stats: dict[str, Any] = {}
    user_corpus_budget = 0
    user_corpus_positions: list[dict[str, Any]] = []
    if use_user_corpus:
        from gomoku_api.ws.user_game_corpus import UserGameCorpus

        user_corpus = UserGameCorpus(variant)
        user_corpus.load()
        user_corpus_stats = user_corpus.stats()
        if user_corpus_mode in {"consolidate", "consolidation"}:
            user_corpus_budget = min(max(data_count // 2, 192), 768)
        else:
            user_corpus_budget = min(max(data_count // 4, 96), 384)
        if user_corpus_budget > 0:
            user_corpus_positions = user_corpus.get_pool_for_builder(user_corpus_budget, user_corpus_mode)

    engine_eval = None
    engine_available = False
    engine_backend_resolved = "builtin"
    confirm_eval = None
    confirm_available = False
    confirm_backend_resolved = "builtin"
    try:
        from gomoku_api.ws.oracle_backends import create_oracle_evaluator

        engine_eval, engine_backend_resolved = create_oracle_evaluator(
            board_size,
            win_len,
            backend=teacher_backend_requested,
            role="teacher",
        )
        await engine_eval.start()
        engine_available = bool(engine_eval.alive)
    except Exception as exc:
        logger.warning("Teacher oracle unavailable for %s: %s", variant, exc)
        if engine_eval is not None:
            try:
                await engine_eval.stop()
            except Exception:
                pass
        engine_eval = None

    try:
        if confirm_backend_requested == teacher_backend_requested and engine_eval is not None:
            confirm_eval = engine_eval
            confirm_backend_resolved = engine_backend_resolved
            confirm_available = engine_available
        else:
            from gomoku_api.ws.oracle_backends import create_oracle_evaluator

            confirm_eval, confirm_backend_resolved = create_oracle_evaluator(
                board_size,
                win_len,
                backend=confirm_backend_requested,
                role="confirm",
            )
            await confirm_eval.start()
            confirm_available = bool(confirm_eval.alive)
    except Exception as exc:
        logger.warning("Confirm oracle unavailable for %s: %s", variant, exc)
        if confirm_eval is not None and confirm_eval is not engine_eval:
            try:
                await confirm_eval.stop()
            except Exception:
                pass
        confirm_eval = engine_eval
        confirm_available = engine_available
        confirm_backend_resolved = engine_backend_resolved

    live_gpu = get_gpu_info()
    await callback({
        "type": "train.start",
        "payload": {
            "variant": variant,
            "epochs": epochs,
            "batchSize": batch_size,
            "boardSize": board_size,
            "winLength": win_len,
            "curriculumStage": variant_spec.curriculum_stage,
            "variantSpec": variant_spec.to_metadata(),
            "device": device.type,
            "deviceName": live_gpu.get("name", str(device)),
            "mixedPrecision": runtime_flags["mixedPrecision"],
            "tf32": runtime_flags["tf32"],
            "channelsLast": runtime_flags["channelsLast"],
            "learningRate": 2e-3,
            "modelParams": _count_model_parameters(model),
            "totalIterations": cycle_count,
            "totalCycles": cycle_count,
            "trainingMode": "size_curriculum" if variant_spec.is_curriculum else "cyclic_engine_exam",
            "modelProfile": model_profile,
            "teacherBackend": teacher_backend_requested,
            "teacherBackendResolved": engine_backend_resolved,
            "confirmBackend": confirm_backend_requested,
            "confirmBackendResolved": confirm_backend_resolved,
            "userCorpusEnabled": use_user_corpus,
            "userCorpusMode": user_corpus_mode if use_user_corpus else None,
            "userCorpusStats": user_corpus_stats if use_user_corpus else None,
            **serving_summary,
            "gpu": live_gpu,
        },
    })

    offline_target = offline_dataset_limit
    if prefer_offline_dataset and board_size <= 5:
        offline_target = max(
            offline_target,
            foundation_dataset_count,
            teacher_seed_count * 6,
            4096,
        )
    offline_positions: list[dict[str, Any]] = []
    offline_dataset_path: Path | None = None
    offline_dataset_type: str | None = None
    if prefer_offline_dataset:
        offline_positions, offline_dataset_path, offline_dataset_type = _load_offline_dataset_positions(
            variant,
            max_positions=offline_target if offline_target > 0 else None,
            preferred_backend=engine_backend_resolved,
        )
        if offline_positions:
            logger.info(
                "Using offline %s dataset for %s: %d positions from %s",
                offline_dataset_type,
                variant,
                len(offline_positions),
                offline_dataset_path,
            )

    if variant == "ttt3":
        tactical_positions, _, _ = await _build_positions(variant, min(teacher_seed_count // 2, 512), callback)
    elif board_size <= 5:
        tactical_positions = await _generate_tactical_curriculum_positions(
            max(600, min(1500, teacher_seed_count * 5 if offline_positions else teacher_seed_count * 3)),
            board_size,
            win_len,
            callback,
        )
    else:
        tactical_positions = await _generate_tactical_curriculum_positions(
            min(max(teacher_seed_count // 3, 128), 512),
            board_size,
            win_len,
            callback,
        )

    if offline_positions:
        anchor_positions = list(offline_positions)
    elif engine_available and engine_eval is not None:
        anchor_positions = await _generate_engine_labeled_positions(
            teacher_seed_count,
            board_size,
            win_len,
            callback,
            engine_eval,
            variant=variant,
        )
    elif variant in {"ttt3", "ttt5"}:
        _, dataset_path, dataset_type = _load_offline_dataset_positions(
            variant,
            max_positions=teacher_seed_count,
            preferred_backend=engine_backend_resolved,
        )
        if dataset_path is None:
            dataset_path = SAVED_DIR / "datasets" / f"{variant}_minimax.json"
        if not dataset_path.exists():
            logger.info("No offline dataset at %s — auto-generating fallback dataset", dataset_path)
            from gomoku_api.ws.offline_gen import generate_minimax_dataset

            await generate_minimax_dataset(variant, min(max(teacher_seed_count, 1000), 5000), callback)
        try:
            anchor_positions = json.loads(dataset_path.read_text())[:teacher_seed_count]
            offline_dataset_path = dataset_path
            offline_dataset_type = dataset_type or "minimax"
        except Exception as exc:
            logger.warning("Failed to load fallback dataset %s: %s", dataset_path, exc)

    if not anchor_positions:
        anchor_positions = list(tactical_positions)

    if offline_positions and rapid_mode:
        repair_pool_size = max(repair_pool_size, min(len(anchor_positions), 1536))

    initial_anchor_cap = len(anchor_positions) if (offline_positions and rapid_mode) else max(anchor_bank_max if rapid_mode else len(anchor_positions), teacher_seed_count)
    anchor_positions = _merge_position_bank([], anchor_positions, max_size=initial_anchor_cap)
    initial_holdout_cap = 48 if rapid_mode else 96
    if offline_positions and rapid_mode:
        initial_holdout_cap = min(max(int(len(anchor_positions) * 0.12), 96), 384)
    anchor_positions, initial_holdout = _split_holdout_positions(
        anchor_positions,
        holdout_ratio=0.12 if rapid_mode else 0.08,
        max_holdout=initial_holdout_cap,
    )
    holdout_bank = _merge_position_bank(
        holdout_bank,
        initial_holdout,
        max_size=512 if (rapid_mode and offline_positions) else (192 if rapid_mode else 512),
    )
    frozen_suites = await _build_frozen_benchmark_suites(
        variant,
        board_size,
        win_len,
        engine_eval if engine_available else None,
    )

    _cycle_batch_size = rapid_batch_size if rapid_mode else batch_size
    _cycle_augment_mode = "random" if rapid_mode else "full"

    if is_new_model:
        foundation_pool = list(anchor_positions) + list(tactical_positions)
        if user_corpus is not None and user_corpus_budget > 0:
            foundation_user_count = min(
                max(user_corpus_budget * (2 if user_corpus_mode in {"consolidate", "consolidation"} else 1), 64),
                768,
            )
            foundation_pool.extend(user_corpus.get_pool_for_builder(foundation_user_count, user_corpus_mode))
        random.shuffle(foundation_pool)
        if rapid_mode and foundation_pool:
            foundation_batch_size = _cycle_batch_size if rapid_mode else batch_size
            foundation_batches = max(1, len(foundation_pool) // max(foundation_batch_size, 1))
            default_foundation_epochs = 10 if (offline_positions and board_size <= 5) else (6 if offline_positions else 3)
            foundation_epochs = max(2, min(12, int(kwargs.get("foundationEpochs", default_foundation_epochs))))
            min_foundation_steps = 80 if (offline_positions and board_size <= 5) else 60
            max_foundation_steps = 360 if (offline_positions and board_size <= 5) else 240
            foundation_steps = max(min_foundation_steps, min(max_foundation_steps, foundation_batches * foundation_epochs))
            if foundation_dataset_count > 0:
                dataset_step_cap = 512 if board_size <= 5 else 320
                foundation_steps = max(
                    foundation_steps,
                    min(dataset_step_cap, foundation_dataset_count // max(foundation_batch_size, 1)),
                )
        else:
            foundation_steps = max(60, turbo_steps * 2) if rapid_mode else turbo_steps
        await _run_training_steps(
            model,
            foundation_pool,
            foundation_steps,
            _cycle_batch_size if rapid_mode else batch_size,
            device,
            runtime_flags,
            callback,
            metrics_history,
            phase="foundation",
            stage="turbo_train",
            completed_phases=completed_phases,
            overall_started_at=started_at,
            overall_percent_base=0,
            overall_percent_range=18,
            augment=True,
            augment_mode=_cycle_augment_mode,
            time_budget=90.0 if rapid_mode else 0.0,
            **common,
        )
        completed_phases.append("foundation")
        registry.save_working_candidate(
            model,
            generation=0,
            metrics={**variant_metric_metadata, "phase": "foundation"},
        )
        foundation_validation = await _run_validation_snapshot(
            model,
            holdout_bank,
            frozen_suites,
            device,
            callback,
            variant=variant,
            cycle=0,
            total_cycles=cycle_count,
        )
        validation_history.append(dict(foundation_validation))

    previous_exam: dict[str, Any] | None = None
    last_validation: dict[str, Any] | None = validation_history[-1] if validation_history else None
    strong_result = None
    cycle_percent = 62.0 / max(cycle_count, 1)
    base_percent = 18.0 if is_new_model else 5.0

    _cycle_time_turbo = 45.0 if rapid_mode else 0.0
    _cycle_time_repair = 30.0 if rapid_mode else 0.0
    current_tactical_ratio = 0.50 if rapid_mode else 0.20
    current_tactical_focus: str | None = None
    current_engine_focus: str | None = None
    current_engine_count = engine_per_cycle
    current_engine_current_player_focus: int | None = 2 if (rapid_mode and board_size <= 5) else None
    current_player_focus_ratio = 0.35 if current_engine_current_player_focus in (1, 2) else 0.0
    current_focus_conversion_ratio = 0.0
    current_counter_conversion_ratio = 0.0
    current_conversion_focus = False
    current_failure_slice = 256
    repair_effectiveness_history: list[float] = []
    best_quick_checkpoint_state: dict[str, torch.Tensor] | None = None
    best_quick_checkpoint_summary: dict[str, Any] | None = None
    best_quick_checkpoint_cycle: int | None = None
    best_quick_checkpoint_score: tuple[float, float, float, float, int] | None = None

    for cycle in range(1, cycle_count + 1):
        cycle_base = base_percent + (cycle - 1) * cycle_percent

        # Incremental data generation per cycle (rapid mode)
        if rapid_mode and engine_available and engine_eval is not None:
            new_engine = await _generate_engine_labeled_positions(
                current_engine_count,
                board_size,
                win_len,
                callback,
                engine_eval,
                phase_focus=current_engine_focus,
            )
            if current_engine_current_player_focus in (1, 2):
                side_focus_count = max(8, min(32, current_engine_count // 4))
                side_focus_positions = await _generate_engine_labeled_positions(
                    side_focus_count,
                    board_size,
                    win_len,
                    callback,
                    engine_eval,
                    source="engine_side_focus",
                    phase_focus=current_engine_focus,
                    current_player_focus=current_engine_current_player_focus,
                    boost_weight=1.15,
                )
                new_engine.extend(side_focus_positions)
            if current_conversion_focus:
                conversion_count = max(12, min(40, current_engine_count // 3))
                conversion_positions = await _generate_engine_labeled_positions(
                    conversion_count,
                    board_size,
                    win_len,
                    callback,
                    engine_eval,
                    source="engine_conversion",
                    phase_focus=current_engine_focus or "mid",
                    current_player_focus=current_engine_current_player_focus,
                    min_value=0.45,
                    boost_weight=1.35,
                )
                new_engine.extend(conversion_positions)
                if current_engine_current_player_focus in (1, 2):
                    side_conversion_count = max(10, min(48, current_engine_count // 2))
                    side_conversion_positions = await _generate_engine_labeled_positions(
                        side_conversion_count,
                        board_size,
                        win_len,
                        callback,
                        engine_eval,
                        source="engine_side_conversion",
                        phase_focus=current_engine_focus or "mid",
                        current_player_focus=current_engine_current_player_focus,
                        min_value=0.55,
                        boost_weight=1.55,
                    )
                    new_engine.extend(side_conversion_positions)
            train_engine, holdout_engine = _split_holdout_positions(
                new_engine,
                holdout_ratio=0.10,
                max_holdout=8,
            )
            if offline_positions:
                current_anchor_cap = min(max(anchor_bank_max, len(anchor_positions)), 4096)
            else:
                current_anchor_cap = min(anchor_bank_max, 200 + cycle * 75)
            anchor_positions = _merge_position_bank(anchor_positions, train_engine, max_size=current_anchor_cap)
            holdout_bank = _merge_position_bank(holdout_bank, holdout_engine, max_size=192)
            if current_tactical_focus is None:
                tactical_positions = await _generate_tactical_curriculum_positions(
                    tactical_per_cycle, board_size, win_len, callback,
                )
            else:
                focused_count = max(int(tactical_per_cycle * 0.60), 1)
                mixed_count = max(tactical_per_cycle - focused_count, 0)
                focused_positions = await _generate_tactical_curriculum_positions(
                    focused_count,
                    board_size,
                    win_len,
                    callback,
                    motif_filter=current_tactical_focus,
                )
                mixed_positions = await _generate_tactical_curriculum_positions(
                    mixed_count,
                    board_size,
                    win_len,
                    callback,
                ) if mixed_count > 0 else []
                tactical_positions = focused_positions + mixed_positions

        turbo_pool = _build_turbo_pool(
            anchor_positions,
            tactical_positions,
            failure_bank[-min(len(failure_bank), current_failure_slice):],
            user_corpus_positions if user_corpus_positions else None,
            data_count=repair_pool_size,
            tactical_ratio=current_tactical_ratio if rapid_mode else 0.20,
            focus_player=current_engine_current_player_focus,
            focus_ratio=current_player_focus_ratio,
            focus_conversion_ratio=current_focus_conversion_ratio,
            counter_conversion_ratio=current_counter_conversion_ratio,
        )

        # Adaptive training budget: steps scale with pool size
        if rapid_mode:
            _pool_size = len(turbo_pool)
            _batches_per_pass = max(1, _pool_size // _cycle_batch_size)
            effective_turbo = max(30, min(100, 8 * _batches_per_pass))
            effective_repair = max(15, effective_turbo // 2)
        else:
            effective_turbo = turbo_steps
            effective_repair = repair_steps

        await _run_training_steps(
            model,
            turbo_pool,
            effective_turbo,
            _cycle_batch_size,
            device,
            runtime_flags,
            callback,
            metrics_history,
            phase="turbo_train",
            stage="training",
            completed_phases=completed_phases,
            iteration=cycle,
            total_iterations=cycle_count,
            overall_started_at=started_at,
            overall_percent_base=cycle_base,
            overall_percent_range=cycle_percent * 0.42,
            augment=True,
            augment_mode=_cycle_augment_mode,
            time_budget=_cycle_time_turbo,
            failureBankSize=len(failure_bank),
            **common,
        )
        registry.save_working_candidate(
            model,
            generation=cycle,
            metrics={**variant_metric_metadata, "phase": "turbo_train", "cycle": cycle},
        )
        pre_repair_state = _clone_model_state_dict(model)

        cycle_failures: list[dict[str, Any]] = []
        exam_result = None
        exam_summary_current: dict[str, Any] | None = None
        quick_exam_scheduled = not rapid_mode or (cycle % exam_every == 1 or cycle == cycle_count)
        latest_holdout_acc = float(last_validation.get("holdoutPolicyAcc", 0.0) or 0.0) if last_validation else 0.0
        holdout_ready = (not rapid_mode) or latest_holdout_acc >= exam_threshold_acc or cycle == cycle_count
        do_exam = quick_exam_scheduled and holdout_ready
        if do_exam:
            if engine_available and engine_eval is not None:
                exam_result, cycle_failures, exam_summary_current = await _run_engine_exam(
                    model,
                    board_size,
                    win_len,
                    device,
                    callback,
                    engine_eval,
                    variant=variant,
                    cycle=cycle,
                    total_cycles=cycle_count,
                    num_pairs=exam_pairs,
                    previous_result=previous_exam,
                    phase="exam",
                    stage="engine_eval",
                    collect_failures=True,
                )
                strong_result = exam_result
                previous_exam = exam_summary_current
                if exam_summary_current:
                    challenger_result = await _run_challenger_vs_champion(
                        model,
                        registry,
                        board_size=board_size,
                        win_len=win_len,
                        device=device,
                        callback=callback,
                        variant=variant,
                        num_pairs=min(exam_pairs, 6),
                        phase="arena",
                        stage="challenger_eval",
                    )
                    if challenger_result is not None:
                        exam_summary_current["winrateVsChampion"] = round(challenger_result.winrate_a, 4)
                        exam_summary_current["decisiveWinRateVsChampion"] = round(challenger_result.decisive_winrate_a, 4)
                        exam_summary_current["drawRateVsChampion"] = round(challenger_result.draw_rate, 4)
                        exam_summary_current["winsVsChampion"] = int(challenger_result.wins_a)
                        exam_summary_current["lossesVsChampion"] = int(challenger_result.wins_b)
                        exam_summary_current["drawsVsChampion"] = int(challenger_result.draws)
                    winrate_history.append({
                        "cycle": cycle,
                        "winrate": round(exam_summary_current.get("winrate", 0), 4),
                        "decisiveWinRate": round(exam_summary_current.get("decisiveWinRate", 0), 4),
                        "drawRate": round(exam_summary_current.get("drawRate", 0), 4),
                        "winrateAsP1": round(exam_summary_current.get("winrateAsP1", 0), 4),
                        "winrateAsP2": round(exam_summary_current.get("winrateAsP2", 0), 4),
                        "balancedSideWinrate": round(exam_summary_current.get("balancedSideWinrate", 0), 4),
                        "tacticalOverrideRate": round(exam_summary_current.get("tacticalOverrideRate", 0), 4),
                        "valueGuidedRate": round(exam_summary_current.get("valueGuidedRate", 0), 4),
                        "modelPolicyRate": round(exam_summary_current.get("modelPolicyRate", 0), 4),
                        "pureGapRate": round(exam_summary_current.get("pureGapRate", 0), 4),
                        "pureGapRateAsP1": round(exam_summary_current.get("pureGapRateAsP1", 0), 4),
                        "pureGapRateAsP2": round(exam_summary_current.get("pureGapRateAsP2", 0), 4),
                        "pureAlignmentRate": round(exam_summary_current.get("pureAlignmentRate", 0), 4),
                        "pureMissedWinCount": int(exam_summary_current.get("pureMissedWinCount", 0) or 0),
                        "pureMissedBlockCount": int(exam_summary_current.get("pureMissedBlockCount", 0) or 0),
                        "winrateVsChampion": round(exam_summary_current.get("winrateVsChampion", 0), 4),
                        "decisiveWinRateVsChampion": round(exam_summary_current.get("decisiveWinRateVsChampion", 0), 4),
                        "wins": exam_summary_current.get("wins", 0),
                        "losses": exam_summary_current.get("losses", 0),
                        "draws": exam_summary_current.get("draws", 0),
                    })
                    current_score = _checkpoint_selection_score(exam_summary_current, cycle)
                    if best_quick_checkpoint_score is None or current_score > best_quick_checkpoint_score:
                        best_quick_checkpoint_score = current_score
                        best_quick_checkpoint_state = pre_repair_state
                        best_quick_checkpoint_summary = dict(exam_summary_current)
                        best_quick_checkpoint_cycle = cycle
            else:
                await callback({
                    "type": "train.progress",
                    "payload": {
                        "phase": "exam",
                        "stage": "unavailable",
                        "variant": variant,
                        "cycle": cycle,
                        "totalCycles": cycle_count,
                        "iteration": cycle,
                        "totalIterations": cycle_count,
                        "message": "Engine exam unavailable: persistent engine not running",
                    },
                })
        elif quick_exam_scheduled and not holdout_ready:
            await callback({
                "type": "train.progress",
                "payload": {
                    "phase": "exam",
                    "stage": "holdout_gate",
                    "variant": variant,
                    "cycle": cycle,
                    "totalCycles": cycle_count,
                    "iteration": cycle,
                    "totalIterations": cycle_count,
                    "holdoutPolicyAcc": round(latest_holdout_acc, 2),
                    "examThresholdAcc": round(exam_threshold_acc, 2),
                    "message": "Quick probe deferred until holdout accuracy gate is met",
                },
            })
        else:
            await callback({
                "type": "train.progress",
                "payload": {
                    "phase": "exam",
                    "stage": "deferred",
                    "variant": variant,
                    "cycle": cycle,
                    "totalCycles": cycle_count,
                    "iteration": cycle,
                    "totalIterations": cycle_count,
                    "message": "Quick probe deferred this cycle",
                },
            })

        before_matches = _score_policy_matches(model, cycle_failures, device) if cycle_failures else []
        failure_bank = _merge_failure_bank(failure_bank, cycle_failures, max_size=failure_bank_max)

        if failure_bank:
            repair_pool = _build_repair_pool(
                failure_bank,
                anchor_positions,
                tactical_positions,
                user_corpus_positions if user_corpus_positions else None,
                data_count=max(batch_size, repair_pool_size),
                focus_player=current_engine_current_player_focus,
                focus_ratio=max(current_player_focus_ratio, 0.40 if current_engine_current_player_focus in (1, 2) else 0.0),
                focus_conversion_ratio=max(current_focus_conversion_ratio, 0.18 if current_engine_current_player_focus in (1, 2) and current_conversion_focus else 0.0),
                counter_conversion_ratio=current_counter_conversion_ratio,
            )
            await _run_training_steps(
                model,
                repair_pool,
                effective_repair,
                _cycle_batch_size,
                device,
                runtime_flags,
                callback,
                metrics_history,
                phase="repair",
                stage="training",
                completed_phases=completed_phases,
                iteration=cycle,
                total_iterations=cycle_count,
                overall_started_at=started_at,
                overall_percent_base=cycle_base + cycle_percent * 0.42,
                overall_percent_range=cycle_percent * 0.42,
                augment=True,
                augment_mode=_cycle_augment_mode,
                time_budget=_cycle_time_repair,
                failureBankSize=len(failure_bank),
                newFailures=len(cycle_failures),
                **common,
            )
            registry.save_working_candidate(
                model,
                generation=cycle,
                metrics={**variant_metric_metadata, "phase": "repair", "cycle": cycle},
            )

        after_matches = _score_policy_matches(model, cycle_failures, device) if cycle_failures else []
        fixed_errors = sum((not before) and after for before, after in zip(before_matches, after_matches))
        regressed_errors = sum(before and (not after) for before, after in zip(before_matches, after_matches))
        corrected_errors = sum(after_matches)
        corrected_rate = corrected_errors / max(len(after_matches), 1) if after_matches else 0.0
        await callback({
            "type": "train.progress",
            "payload": {
                "phase": "repair_eval",
                "stage": "assessment",
                "variant": variant,
                "cycle": cycle,
                "totalCycles": cycle_count,
                "iteration": cycle,
                "totalIterations": cycle_count,
                "fixedErrors": fixed_errors,
                "regressedErrors": regressed_errors,
                "correctedErrors": corrected_errors,
                "correctedRate": round(corrected_rate, 4),
                "failureBankSize": len(failure_bank),
                "newFailures": len(cycle_failures),
                "winrateVsAlgorithm": round(exam_result.winrate_a, 4) if exam_result is not None else None,
                "decisiveWinRate": round(exam_result.decisive_winrate_a, 4) if exam_result is not None else None,
                "drawRate": round(exam_result.draw_rate, 4) if exam_result is not None else None,
                "percent": round(cycle_base + cycle_percent * 0.90, 1),
                "elapsed": round(time.monotonic() - started_at, 1),
            },
        })
        repair_effectiveness_history.append(corrected_rate)

        validation_payload = await _run_validation_snapshot(
            model,
            holdout_bank,
            frozen_suites,
            device,
            callback,
            variant=variant,
            cycle=cycle,
            total_cycles=cycle_count,
            previous_holdout=last_validation,
        )
        validation_history.append(dict(validation_payload))
        last_validation = validation_payload

        pure_tactical_failures: list[dict[str, Any]] = []
        win_suite = frozen_suites.get("win") or []
        block_suite = frozen_suites.get("block") or []
        exact_suite = frozen_suites.get("exact") or []
        if win_suite:
            _win_bench, pure_win_failures = _evaluate_decision_suite(
                model,
                win_suite,
                board_size,
                win_len,
                decision_mode="pure",
                suite_name="win",
                collect_failures=True,
            )
            pure_tactical_failures.extend(pure_win_failures[:16])
        if block_suite:
            _block_bench, pure_block_failures = _evaluate_decision_suite(
                model,
                block_suite,
                board_size,
                win_len,
                decision_mode="pure",
                suite_name="block",
                collect_failures=True,
            )
            pure_tactical_failures.extend(pure_block_failures[:16])
        if exact_suite:
            _exact_bench, pure_exact_failures = _evaluate_decision_suite(
                model,
                exact_suite,
                board_size,
                win_len,
                decision_mode="pure",
                suite_name="exact",
                collect_failures=True,
            )
            pure_tactical_failures.extend(pure_exact_failures[:24])
        if pure_tactical_failures:
            failure_bank = _merge_failure_bank(
                failure_bank,
                pure_tactical_failures,
                max_size=512 if rapid_mode else 768,
            )

        if rapid_mode:
            strategy = _choose_rapid_cycle_strategy(
                validation_payload,
                corrected_rate=corrected_rate,
                failure_bank_size=len(failure_bank),
                engine_per_cycle=engine_per_cycle,
                exam_summary=exam_summary_current or previous_exam,
            )
            current_tactical_ratio = float(strategy["tacticalRatio"])
            current_tactical_focus = strategy["tacticalFocus"]
            current_engine_focus = strategy["engineFocus"]
            current_engine_count = int(strategy["engineCount"])
            current_engine_current_player_focus = strategy.get("engineCurrentPlayerFocus")
            current_player_focus_ratio = float(strategy.get("playerFocusRatio", 0.0) or 0.0)
            current_focus_conversion_ratio = float(strategy.get("focusConversionRatio", 0.0) or 0.0)
            current_counter_conversion_ratio = float(strategy.get("counterConversionRatio", 0.0) or 0.0)
            current_conversion_focus = bool(strategy.get("conversionFocus", False))
            current_failure_slice = int(strategy["failureSlice"])

        # Winrate-driven scheduler: early exit if stable or stagnated
        if rapid_mode and len(winrate_history) >= 3:
            last3 = [h["winrate"] for h in winrate_history[-3:]]
            recent_holdout = [
                float(item["holdoutPolicyAcc"])
                for item in validation_history[-3:]
                if item.get("holdoutPolicyAcc") is not None
            ]
            recent_corrected = repair_effectiveness_history[-3:]
            avg_corrected = sum(recent_corrected) / max(len(recent_corrected), 1)
            holdout_regressing = len(recent_holdout) >= 2 and recent_holdout[-1] + 1.0 < recent_holdout[0]
            holdout_flat = len(recent_holdout) >= 2 and abs(recent_holdout[-1] - recent_holdout[0]) < 1.0

            if min(last3) >= 0.60 and not holdout_regressing and avg_corrected >= 0.25:
                logger.info("Winrate stable above 60%% after cycle %d, exiting early", cycle)
                break
            if max(last3) - min(last3) < 0.03 and cycle >= 5 and holdout_flat and avg_corrected < 0.15:
                logger.info("Winrate stagnated after cycle %d, exiting", cycle)
                break

    quick_result = None
    selected_checkpoint_cycle = None
    selected_checkpoint_summary = None
    if best_quick_checkpoint_state is not None:
        _restore_model_state_dict(model, best_quick_checkpoint_state)
        selected_checkpoint_cycle = best_quick_checkpoint_cycle
        selected_checkpoint_summary = dict(best_quick_checkpoint_summary or {})
        registry.save_working_candidate(
            model,
            generation=cycle_count,
            metrics={
                **variant_metric_metadata,
                "phase": "checkpoint_selection",
                "cycle": selected_checkpoint_cycle,
                "winrate": selected_checkpoint_summary.get("winrate"),
                "decisiveWinRate": selected_checkpoint_summary.get("decisiveWinRate"),
                "drawRate": selected_checkpoint_summary.get("drawRate"),
                "winrateAsP1": selected_checkpoint_summary.get("winrateAsP1"),
                "winrateAsP2": selected_checkpoint_summary.get("winrateAsP2"),
                "balancedSideWinrate": selected_checkpoint_summary.get("balancedSideWinrate"),
                "pureGapRate": selected_checkpoint_summary.get("pureGapRate"),
                "pureGapRateAsP1": selected_checkpoint_summary.get("pureGapRateAsP1"),
                "pureGapRateAsP2": selected_checkpoint_summary.get("pureGapRateAsP2"),
                "selectedCheckpointCycle": selected_checkpoint_cycle,
                "selectedCheckpointWinrate": selected_checkpoint_summary.get("winrate"),
                "selectedCheckpointDecisiveWinRate": selected_checkpoint_summary.get("decisiveWinRate"),
                "selectedCheckpointDrawRate": selected_checkpoint_summary.get("drawRate"),
                "selectedCheckpointWinrateAsP1": selected_checkpoint_summary.get("winrateAsP1"),
                "selectedCheckpointWinrateAsP2": selected_checkpoint_summary.get("winrateAsP2"),
                "selectedCheckpointBalancedSideWinrate": selected_checkpoint_summary.get("balancedSideWinrate"),
                "selectedCheckpointPureGapRate": selected_checkpoint_summary.get("pureGapRate"),
                "selectedCheckpointPureGapRateAsP1": selected_checkpoint_summary.get("pureGapRateAsP1"),
                "selectedCheckpointPureGapRateAsP2": selected_checkpoint_summary.get("pureGapRateAsP2"),
                "selectedCheckpointWinrateVsChampion": selected_checkpoint_summary.get("winrateVsChampion"),
                "selectedCheckpointDecisiveWinRateVsChampion": selected_checkpoint_summary.get("decisiveWinRateVsChampion"),
                "selectedCheckpointWinrateVsPreviousCheckpoint": selected_checkpoint_summary.get("winrateVsPreviousCheckpoint"),
                "selectedCheckpointDecisiveWinRateVsPreviousCheckpoint": selected_checkpoint_summary.get("decisiveWinRateVsPreviousCheckpoint"),
            },
        )
        await callback({
            "type": "train.progress",
            "payload": {
                "phase": "checkpoint_selection",
                "stage": "best_quick_probe",
                "variant": variant,
                "cycle": selected_checkpoint_cycle,
                "selectedCheckpointCycle": selected_checkpoint_cycle,
                "selectedCheckpointWinrate": selected_checkpoint_summary.get("winrate"),
                "selectedCheckpointDecisiveWinRate": selected_checkpoint_summary.get("decisiveWinRate"),
                "selectedCheckpointDrawRate": selected_checkpoint_summary.get("drawRate"),
                "message": "Restored best pre-repair checkpoint before final confirm exam",
                "selectedCheckpointWinrateVsChampion": selected_checkpoint_summary.get("winrateVsChampion"),
                "selectedCheckpointDecisiveWinRateVsChampion": selected_checkpoint_summary.get("decisiveWinRateVsChampion"),
                "selectedCheckpointWinrateVsPreviousCheckpoint": selected_checkpoint_summary.get("winrateVsPreviousCheckpoint"),
                "selectedCheckpointDecisiveWinRateVsPreviousCheckpoint": selected_checkpoint_summary.get("decisiveWinRateVsPreviousCheckpoint"),
            },
        })
        selected_validation = await _run_validation_snapshot(
            model,
            holdout_bank,
            frozen_suites,
            device,
            callback,
            variant=variant,
            cycle=cycle_count,
            total_cycles=cycle_count,
            previous_holdout=last_validation,
        )
        validation_history.append(dict(selected_validation))
        last_validation = selected_validation

    if registry.has_champion():
        quick_result = await _run_challenger_vs_champion(
            model,
            registry,
            board_size=board_size,
            win_len=win_len,
            device=device,
            callback=callback,
            variant=variant,
            num_pairs=min(exam_pairs, 8),
            phase="arena",
            stage="quick_eval",
        )

    # ── Self-Play MCTS Loop (optional, for strong models) ────────────
    auto_selfplay = variant == "ttt5" and board_size <= 5 and kwargs.get("selfPlay") is None
    do_selfplay = bool(kwargs.get("selfPlay", auto_selfplay))
    # Alpha-zero inspired defaults: more games, more sims, more training
    sp_iterations_default = 6 if auto_selfplay else 15
    sp_games_default = 60 if auto_selfplay else 200
    sp_sims_default = 100 if auto_selfplay else 200
    sp_train_steps_default = 80 if auto_selfplay else 200
    sp_min_replay_default = 512 if auto_selfplay else 2000
    sp_iterations = max(1, min(30, int(kwargs.get("selfPlayIterations", sp_iterations_default))))
    sp_games = max(20, min(500, int(kwargs.get("selfPlayGames", sp_games_default))))
    sp_sims = max(25, min(400, int(kwargs.get("selfPlaySims", sp_sims_default))))
    sp_train_steps = max(20, min(500, int(kwargs.get("selfPlayTrainSteps", sp_train_steps_default))))
    sp_min_replay_samples = max(64, min(50_000, int(kwargs.get("selfPlayMinReplaySamples", sp_min_replay_default))))

    if do_selfplay and board_size <= 9:
        from trainer_lab.self_play.mixed_replay import MixedReplay
        from trainer_lab.self_play.player import SelfPlayPlayer
        from trainer_lab.config import SelfPlayConfig

        logger.info("Self-play phase: %d iterations × %d games × %d sims", sp_iterations, sp_games, sp_sims)

        replay_path = _selfplay_replay_path(variant)
        sp_replay = MixedReplay(
            total_capacity=50_000,
            source_limits={
                "anchor": 12_000,
                "tactical": 8_000,
                "failure": 10_000,
                "user": 8_000,
                "self_play": 20_000,
            },
        )
        sp_replay.load(replay_path)
        sp_replay.replace("anchor", anchor_positions[:2500])
        sp_replay.replace("tactical", tactical_positions[:1500])
        sp_replay.replace("failure", failure_bank[:1500])
        sp_replay.replace("user", list(user_corpus_positions[:1500]) if user_corpus_positions else [])

        sp_config = SelfPlayConfig(
            games=1,
            simulations=sp_sims,
            min_replay_samples=sp_min_replay_samples,
        )
        best_sp_winrate = -1.0
        best_sp_score: tuple[float, float, float, float, float, float, float, float, float, float, int] | None = None
        best_sp_state = None
        previous_selfplay_model = copy.deepcopy(model).to(device)
        previous_selfplay_model.eval()

        for sp_iter in range(1, sp_iterations + 1):
            # --- Generate self-play games (parallel) ---
            model.eval()
            from trainer_lab.self_play.player import generate_games_parallel

            sp_generation_started_at = time.monotonic()
            sp_generation_progress = {"completed": 0, "positions": 0}
            loop = asyncio.get_running_loop()
            sp_generation_done = asyncio.Event()

            def _on_selfplay_generation_progress(completed: int, total_games: int, positions_count: int) -> None:
                def _store_progress() -> None:
                    sp_generation_progress["completed"] = completed
                    sp_generation_progress["positions"] = positions_count

                loop.call_soon_threadsafe(_store_progress)

            async def _emit_selfplay_generation_progress() -> None:
                while not sp_generation_done.is_set():
                    completed_games = int(sp_generation_progress.get("completed", 0) or 0)
                    positions_count = int(sp_generation_progress.get("positions", 0) or 0)
                    fraction = min(max(completed_games / max(sp_games, 1), 0.0), 1.0)
                    elapsed = time.monotonic() - sp_generation_started_at
                    await callback({
                        "type": "train.progress",
                        "payload": {
                            "phase": "self_play_gen",
                            "stage": "generating",
                            "variant": variant,
                            "iteration": sp_iter,
                            "totalIterations": sp_iterations,
                            "game": completed_games,
                            "totalGames": sp_games,
                            "positionsCollected": positions_count,
                            "elapsed": round(elapsed, 1),
                            "percent": round(90 + 8 * ((sp_iter - 1 + fraction) / sp_iterations), 1),
                            "message": f"Self-play generation: {completed_games}/{sp_games} games",
                        },
                    })
                    await asyncio.sleep(1.0)

            generation_progress_task = asyncio.create_task(_emit_selfplay_generation_progress())
            try:
                sp_positions = await asyncio.to_thread(
                    generate_games_parallel,
                    model,
                    sp_games,
                    board_size=board_size,
                    win_length=win_len,
                    num_simulations=sp_sims,
                    num_workers=min(4, sp_games),
                    device=device,
                    warm_up_steps=sp_config.warm_up_steps,
                    c_puct=sp_config.c_puct,
                    dirichlet_alpha=sp_config.dirichlet_alpha,
                    dirichlet_weight=sp_config.dirichlet_weight,
                    progress_callback=_on_selfplay_generation_progress,
                )
            finally:
                sp_generation_done.set()
                generation_progress_task.cancel()
                try:
                    await generation_progress_task
                except asyncio.CancelledError:
                    pass

            sp_stats = {"wins_p1": 0, "wins_p2": 0, "draws": 0}
            # Count game outcomes from positions (group by game via value sign changes)
            game_values = []
            for pos in sp_positions:
                v = pos.get("value", 0)
                if v > 0:
                    game_values.append(1)
                elif v < 0:
                    game_values.append(-1)
                else:
                    game_values.append(0)
            # Approximate: count unique game results from last positions
            sp_stats["wins_p1"] = sum(1 for v in game_values if v > 0) // max(1, len(game_values) // sp_games)
            sp_stats["wins_p2"] = sum(1 for v in game_values if v < 0) // max(1, len(game_values) // sp_games)
            sp_stats["draws"] = sp_games - sp_stats["wins_p1"] - sp_stats["wins_p2"]

            # Emit generation progress
            elapsed = time.monotonic() - started_at
            await callback({
                "type": "train.progress",
                "payload": {
                    "phase": "self_play_gen",
                    "stage": "generating",
                    "variant": variant,
                    "iteration": sp_iter,
                    "totalIterations": sp_iterations,
                    "game": sp_games,
                    "totalGames": sp_games,
                    "positionsCollected": len(sp_positions),
                    "selfPlayStats": sp_stats,
                    "elapsed": round(elapsed, 1),
                    "percent": round(90 + 8 * ((sp_iter - 0.5) / sp_iterations), 1),
                },
            })
            await asyncio.sleep(0)

            # Add to replay buffer
            sp_replay.add_many("self_play", sp_positions)
            replay_weights = _selfplay_mixed_source_weights(sp_iter, sp_iterations)
            replay_summary = sp_replay.summary()
            self_play_replay_count = int(replay_summary["sources"].get("self_play", 0) or 0)
            logger.info(
                "Self-play iter %d: %d new positions, mixed replay=%d (%s)",
                sp_iter,
                len(sp_positions),
                len(sp_replay),
                replay_summary["sources"],
            )

            if self_play_replay_count < sp_config.min_replay_samples:
                logger.info(
                    "Self-play iter %d: replay warm-up %d/%d self-play samples",
                    sp_iter,
                    self_play_replay_count,
                    sp_config.min_replay_samples,
                )
                await callback({
                    "type": "train.progress",
                    "payload": {
                        "phase": "self_play_warmup",
                        "stage": "replay_warmup",
                        "variant": variant,
                        "iteration": sp_iter,
                        "totalIterations": sp_iterations,
                        "selfPlayReplaySamples": self_play_replay_count,
                        "selfPlayMinReplaySamples": sp_config.min_replay_samples,
                        "mixedReplaySources": replay_summary["sources"],
                        "mixedReplayWeights": replay_weights,
                    },
                })
                sp_replay.save(replay_path)
                continue

            # --- Train on replay buffer ---
            train_sample = sp_replay.sample(
                min(len(sp_replay), sp_train_steps * _cycle_batch_size),
                source_weights=replay_weights,
            )
            # SGD + decaying LR for self-play (alpha_zero-inspired)
            sp_lr = 0.01 * max(0.1, 1.0 - sp_iter / (sp_iterations + 1))
            model.train()
            await _run_training_steps(
                model, train_sample, sp_train_steps, _cycle_batch_size,
                device, runtime_flags, callback, metrics_history,
                phase="self_play_train", stage="training",
                iteration=sp_iter, total_iterations=sp_iterations,
                overall_started_at=started_at,
                overall_percent_base=90 + 8 * (sp_iter - 0.5) / sp_iterations,
                overall_percent_range=4 / sp_iterations,
                optimizer_type="sgd", learning_rate=sp_lr,
                variant=variant, boardSize=board_size, winLength=win_len,
                mixedReplaySources=replay_summary["sources"],
                mixedReplayWeights=replay_weights,
            )

            # --- Arena eval vs engine ---
            if engine_available and engine_eval is not None:
                engine_eval_pairs = _selfplay_eval_num_pairs(
                    sp_iter,
                    sp_iterations,
                    purpose="engine",
                )
                sp_exam_result, sp_failures, sp_exam_summary = await _run_engine_exam(
                    model, board_size, win_len, device, callback, engine_eval,
                    variant=variant, cycle=cycle_count + sp_iter, total_cycles=cycle_count + sp_iterations,
                    num_pairs=engine_eval_pairs, previous_result=previous_exam,
                    phase="self_play_exam", stage="engine_eval",
                    collect_failures=True,
                )
                sp_wr = sp_exam_summary.get("winrate", 0.0) if sp_exam_summary else 0.0
                logger.info("Self-play iter %d: winrate=%.2f", sp_iter, sp_wr)

                challenger_pairs = _selfplay_eval_num_pairs(
                    sp_iter,
                    sp_iterations,
                    purpose="challenger",
                )
                challenger_result = await _run_challenger_vs_champion(
                    model,
                    registry,
                    board_size=board_size,
                    win_len=win_len,
                    device=device,
                    callback=callback,
                    variant=variant,
                    num_pairs=challenger_pairs,
                    phase="arena",
                    stage="self_play_challenger",
                )
                if challenger_result is not None and sp_exam_summary is not None:
                    sp_exam_summary["winrateVsChampion"] = round(challenger_result.winrate_a, 4)
                    sp_exam_summary["decisiveWinRateVsChampion"] = round(challenger_result.decisive_winrate_a, 4)
                    sp_exam_summary["drawRateVsChampion"] = round(challenger_result.draw_rate, 4)

                previous_pairs = _selfplay_eval_num_pairs(
                    sp_iter,
                    sp_iterations,
                    purpose="previous",
                )
                previous_result = await _run_model_vs_model_arena(
                    model,
                    previous_selfplay_model,
                    board_size=board_size,
                    win_len=win_len,
                    device=device,
                    callback=callback,
                    variant=variant,
                    num_pairs=previous_pairs,
                    phase="arena",
                    stage="self_play_previous_checkpoint",
                )
                if sp_exam_summary is not None:
                    sp_exam_summary["winrateVsPreviousCheckpoint"] = round(previous_result.winrate_a, 4)
                    sp_exam_summary["decisiveWinRateVsPreviousCheckpoint"] = round(previous_result.decisive_winrate_a, 4)
                    sp_exam_summary["drawRateVsPreviousCheckpoint"] = round(previous_result.draw_rate, 4)

                accepted_vs_previous = _selfplay_previous_checkpoint_accepted(sp_exam_summary)
                current_sp_score = _checkpoint_selection_score(
                    sp_exam_summary or {"winrate": sp_wr, "decisiveWinRate": 0.0, "drawRate": 0.0},
                    cycle_count + sp_iter,
                )
                if accepted_vs_previous and (best_sp_score is None or current_sp_score > best_sp_score):
                    best_sp_score = current_sp_score
                    best_sp_winrate = sp_wr
                    best_sp_state = {k: v.clone() for k, v in model.state_dict().items()}

                if sp_failures:
                    sp_replay.add_many("failure", sp_failures)

                previous_exam = sp_exam_summary
                strong_result = sp_exam_result

                winrate_history.append({
                    "cycle": cycle_count + sp_iter,
                    "winrate": round(sp_wr, 4),
                    "winrateVsChampion": round((sp_exam_summary or {}).get("winrateVsChampion", 0), 4),
                    "decisiveWinRateVsChampion": round((sp_exam_summary or {}).get("decisiveWinRateVsChampion", 0), 4),
                    "winrateVsPreviousCheckpoint": round((sp_exam_summary or {}).get("winrateVsPreviousCheckpoint", 0), 4),
                    "decisiveWinRateVsPreviousCheckpoint": round((sp_exam_summary or {}).get("decisiveWinRateVsPreviousCheckpoint", 0), 4),
                    "wins": sp_exam_summary.get("wins", 0) if sp_exam_summary else 0,
                    "losses": sp_exam_summary.get("losses", 0) if sp_exam_summary else 0,
                    "draws": sp_exam_summary.get("draws", 0) if sp_exam_summary else 0,
                })
                await callback({
                    "type": "train.progress",
                    "payload": {
                        "phase": "self_play_acceptance",
                        "stage": "accepted_previous_checkpoint" if accepted_vs_previous else "rejected_previous_checkpoint",
                        "variant": variant,
                        "iteration": sp_iter,
                        "totalIterations": sp_iterations,
                        "winrateVsPreviousCheckpoint": round((sp_exam_summary or {}).get("winrateVsPreviousCheckpoint", 0.0), 4),
                        "decisiveWinRateVsPreviousCheckpoint": round((sp_exam_summary or {}).get("decisiveWinRateVsPreviousCheckpoint", 0.0), 4),
                        "acceptedVsPreviousCheckpoint": accepted_vs_previous,
                    },
                })
                if accepted_vs_previous:
                    previous_selfplay_model.load_state_dict(model.state_dict())
                    previous_selfplay_model.eval()
                    registry.save_working_candidate(
                        model,
                        generation=cycle_count + sp_iter,
                        metrics={
                            **variant_metric_metadata,
                            "phase": "self_play",
                            "iteration": sp_iter,
                            "acceptedVsPreviousCheckpoint": True,
                            "winrateVsPreviousCheckpoint": (sp_exam_summary or {}).get("winrateVsPreviousCheckpoint"),
                            "decisiveWinRateVsPreviousCheckpoint": (sp_exam_summary or {}).get("decisiveWinRateVsPreviousCheckpoint"),
                            "winrateVsChampion": (sp_exam_summary or {}).get("winrateVsChampion"),
                            "decisiveWinRateVsChampion": (sp_exam_summary or {}).get("decisiveWinRateVsChampion"),
                        },
                    )

            sp_replay.save(replay_path)

            # Convergence check
            sp_winrates = [h["winrate"] for h in winrate_history[-3:]]
            if len(sp_winrates) >= 3 and min(sp_winrates) >= 0.70:
                logger.info("Self-play converged at iter %d (winrate %.2f)", sp_iter, min(sp_winrates))
                break

        # Restore best self-play checkpoint
        if best_sp_state is not None and best_sp_winrate > 0:
            model.load_state_dict(best_sp_state)
            logger.info("Restored best self-play checkpoint (winrate=%.2f)", best_sp_winrate)
            registry.save_working_candidate(
                model,
                generation=cycle_count,
                metrics={
                    **variant_metric_metadata,
                    "phase": "self_play_best",
                    "winrateVsAlgorithm": best_sp_winrate,
                    "acceptedVsPreviousCheckpoint": True,
                },
            )

    done_payload = _build_deferred_train_done_payload(
        registry=registry,
        variant_spec=variant_spec,
        variant=variant,
        epochs=epochs,
        started_at=started_at,
        metrics_history=metrics_history,
        device=device,
        model_profile=model_profile,
        positions_count=len(anchor_positions) + len(tactical_positions) + len(failure_bank),
        failure_bank_size=len(failure_bank),
        cycle_count=cycle_count,
        selected_checkpoint_cycle=selected_checkpoint_cycle,
        selected_checkpoint_summary=selected_checkpoint_summary or previous_exam,
        winrate_history=winrate_history,
        validation_history=validation_history,
        teacher_backend_requested=teacher_backend_requested,
        teacher_backend_resolved=engine_backend_resolved,
        confirm_backend_requested=confirm_backend_requested,
        confirm_backend_resolved=confirm_backend_resolved,
    )
    await callback({"type": "train.done", "payload": done_payload})

    if confirm_eval is not None and confirm_eval is not engine_eval:
        try:
            await confirm_eval.stop()
        except Exception:
            logger.debug("Failed to stop confirm oracle cleanly", exc_info=True)

    if engine_eval is not None:
        try:
            await engine_eval.stop()
        except Exception:
            logger.debug("Failed to stop persistent engine cleanly", exc_info=True)

    return {
        "deferredEvaluator": {
            "variant": variant,
            "variantSpec": variant_spec.to_metadata(),
            "curriculumStage": variant_spec.curriculum_stage,
            "boardSize": board_size,
            "winLen": win_len,
            "cycleCount": cycle_count,
            "epochs": epochs,
            "startedAt": started_at,
            "modelProfile": model_profile,
            "resFilters": res_filters,
            "resBlocks": res_blocks,
            "valueFc": value_fc,
            "selectedCheckpointCycle": selected_checkpoint_cycle,
            "selectedCheckpointSummary": selected_checkpoint_summary or previous_exam,
            "metricsHistory": metrics_history,
            "validationHistory": validation_history,
            "winrateHistory": winrate_history,
            "teacherBackendRequested": teacher_backend_requested,
            "teacherBackendResolved": engine_backend_resolved,
            "confirmBackendRequested": confirm_backend_requested,
            "confirmBackendResolved": confirm_backend_resolved,
            "positionsCount": len(anchor_positions) + len(tactical_positions) + len(failure_bank),
            "failureBankSize": len(failure_bank),
        }
    }


def clear_model(variant: str = "all") -> dict[str, Any]:
    """Delete saved model files."""
    from gomoku_api.ws.predict_service import clear_cached_model
    from gomoku_api.ws.model_registry import ModelRegistry

    variants = ["ttt3", "ttt5"] if variant == "all" else [variant]
    for current_variant in variants:
        registry = ModelRegistry(current_variant)
        registry.clear_checkpoints(preserve_history=True)
        logger.info("Cleared model checkpoints for %s", current_variant)
        clear_cached_model(current_variant)
    return {"cleared": True, "variant": variant}
