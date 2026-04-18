"""Shared model-profile helpers for training and prediction.

Keeps training and inference on the same variant/profile-specific
architecture without hard-coding the shape logic in multiple files.
"""

from __future__ import annotations

from typing import Any

from trainer_lab.specs import VariantSpec, resolve_variant_spec


def current_model_profile_from_manifest(manifest: dict[str, Any] | None) -> str | None:
    if not manifest:
        return None
    current = manifest.get("current_model_profile")
    if isinstance(current, str) and current:
        return current
    history = manifest.get("history") or []
    if history:
        last = history[-1] or {}
        profile = last.get("modelProfile") or last.get("model_profile")
        if isinstance(profile, str) and profile:
            return profile
    return None


def resolve_model_profile(
    variant: str,
    board_size: int,
    *,
    requested: str | None = None,
    manifest: dict[str, Any] | None = None,
    spec: VariantSpec | None = None,
) -> str:
    variant_spec = spec or resolve_variant_spec(variant)
    profile = (requested or "").strip().lower()
    if profile and profile != "auto":
        if board_size <= 3:
            return "tiny"
        if board_size <= 5 and profile in {"small", "standard"}:
            return profile
        if variant_spec.curriculum_stage == "curriculum" and profile in {"curriculum", "standard"}:
            return "curriculum"
        return "standard"

    manifest_profile = current_model_profile_from_manifest(manifest)
    if manifest_profile:
        return manifest_profile
    if board_size <= 3:
        return "tiny"
    if variant_spec.curriculum_stage == "curriculum":
        return "curriculum"
    return "standard"


def variant_model_hparams(
    variant: str,
    board_size: int,
    cfg: Any,
    *,
    model_profile: str | None = None,
    manifest: dict[str, Any] | None = None,
    spec: VariantSpec | None = None,
) -> tuple[str, tuple[int, int, int]]:
    variant_spec = spec or resolve_variant_spec(variant)
    profile = resolve_model_profile(
        variant,
        board_size,
        requested=model_profile,
        manifest=manifest,
        spec=variant_spec,
    )

    if board_size <= 3:
        return "tiny", (32, 3, 64)

    if board_size <= 5:
        if profile == "small":
            return "small", (64, 6, 128)
        return "standard", (96, 8, 160)

    if variant_spec.curriculum_stage == "curriculum":
        return "curriculum", (96, 6, 160)

    if board_size <= 9:
        return "standard", (96, 6, 160)

    return "standard", (cfg.res_filters, cfg.res_blocks, max(cfg.value_fc, 192))
