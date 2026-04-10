"""Champion/Candidate model lifecycle management.

Separates the model being trained (candidate) from the model used for
prediction/play (champion).  A candidate only becomes champion after
passing the promotion gate (arena evaluation).

File layout per variant (e.g. ``saved/ttt5_resnet/``):

    candidate.pt   – latest training checkpoint
    champion.pt    – best verified model (used by predict_service)
    model.pt       – legacy alias, kept in sync with champion.pt
    manifest.json  – promotion history log
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).resolve().parents[5] / "saved"


@dataclass
class ModelRegistry:
    """Manages candidate/champion model files for a variant."""

    variant: str
    base_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.base_dir = SAVED_DIR / f"{self.variant}_resnet"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def candidate_path(self) -> Path:
        return self.base_dir / "candidate.pt"

    @property
    def working_candidate_path(self) -> Path:
        return self.base_dir / "candidate_working.pt"

    @property
    def champion_path(self) -> Path:
        return self.base_dir / "champion.pt"

    @property
    def legacy_path(self) -> Path:
        return self.base_dir / "model.pt"

    @property
    def manifest_path(self) -> Path:
        return self.base_dir / "manifest.json"

    @property
    def candidate_meta_path(self) -> Path:
        return self.base_dir / "candidate_meta.json"

    @property
    def working_candidate_meta_path(self) -> Path:
        return self.base_dir / "candidate_working_meta.json"

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def _write_meta(self, path: Path, *, generation: int = 0, metrics: dict[str, Any] | None = None) -> None:
        payload = {
            "generation": generation,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **(metrics or {}),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _read_meta(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def save_candidate(self, model: nn.Module, *, generation: int = 0, metrics: dict[str, Any] | None = None) -> None:
        """Save model state_dict as active candidate checkpoint."""
        torch.save(model.state_dict(), self.candidate_path)
        self._write_meta(self.candidate_meta_path, generation=generation, metrics=metrics)
        logger.info("Saved active candidate checkpoint: %s (gen %d)", self.candidate_path, generation)

    def save_working_candidate(self, model: nn.Module, *, generation: int = 0, metrics: dict[str, Any] | None = None) -> None:
        """Save model state_dict as working checkpoint for the current run only."""
        torch.save(model.state_dict(), self.working_candidate_path)
        self._write_meta(self.working_candidate_meta_path, generation=generation, metrics=metrics)
        logger.info("Saved working candidate checkpoint: %s (gen %d)", self.working_candidate_path, generation)

    def read_candidate_meta(self) -> dict[str, Any] | None:
        return self._read_meta(self.candidate_meta_path)

    def read_working_candidate_meta(self) -> dict[str, Any] | None:
        return self._read_meta(self.working_candidate_meta_path)

    def commit_working_candidate(self, *, generation: int = 0, metrics: dict[str, Any] | None = None) -> None:
        """Commit working checkpoint to active candidate without touching champion."""
        if not self.working_candidate_path.exists():
            raise FileNotFoundError(f"No working candidate to commit: {self.working_candidate_path}")
        tmp_path = self.base_dir / "candidate.pt.tmp"
        try:
            shutil.copy2(self.working_candidate_path, tmp_path)
            tmp_path.replace(self.candidate_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        self._write_meta(self.candidate_meta_path, generation=generation, metrics=metrics)
        logger.info("Committed working candidate → active candidate (gen %d)", generation)

    def load_into(self, model: nn.Module, path: Path, device: torch.device) -> nn.Module:
        """Load state_dict from *path* into *model* and move to *device*."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model

    def load_champion_into(self, model: nn.Module, device: torch.device) -> nn.Module | None:
        """Load champion.pt into *model*.  Returns None if no champion exists."""
        path = self.champion_path
        if not path.exists():
            path = self.legacy_path
        if not path.exists():
            return None
        return self.load_into(model, path, device)

    def has_champion(self) -> bool:
        return self.champion_path.exists() or self.legacy_path.exists()

    def resolve_serving_checkpoint(self) -> tuple[Path | None, str]:
        """Return the verified checkpoint that runtime is allowed to serve.

        Serving must stay on the promoted champion. The legacy ``model.pt`` is
        only kept as a compatibility alias for the same promoted weights.
        """
        if self.champion_path.exists():
            return self.champion_path, "champion"
        if self.legacy_path.exists():
            return self.legacy_path, "legacy"
        return None, "none"

    def serving_summary(self) -> dict[str, Any]:
        """Return lightweight metadata describing the active serving source."""
        path, source = self.resolve_serving_checkpoint()
        manifest = self.read_manifest()
        return {
            "variant": self.variant,
            "servingReady": path is not None,
            "servingSource": source,
            "servingCheckpointPath": str(path) if path is not None else None,
            "servingGeneration": manifest.get("current_champion_generation"),
        }

    def has_active_candidate(self) -> bool:
        return self.candidate_path.exists()

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def promote_candidate(
        self,
        *,
        generation: int = 0,
        metrics: dict[str, Any] | None = None,
        reason: str = "passed_promotion_gate",
        source_path: Path | None = None,
    ) -> None:
        """Atomic promotion: candidate → champion (+ legacy model.pt sync).

        Uses temp file + os.replace for crash safety: if the process dies
        mid-copy, the previous champion remains intact.

        Also appends an entry to manifest.json.
        """
        promote_source = source_path or self.candidate_path
        if not promote_source.exists():
            raise FileNotFoundError(f"No candidate to promote: {promote_source}")

        # 0. Sync active candidate to the promoted source first.
        tmp_candidate = self.base_dir / "candidate.pt.tmp"
        try:
            shutil.copy2(promote_source, tmp_candidate)
            tmp_candidate.replace(self.candidate_path)
        except BaseException:
            tmp_candidate.unlink(missing_ok=True)
            raise
        self._write_meta(self.candidate_meta_path, generation=generation, metrics=metrics)

        # 1. Copy candidate → champion (crash-safe: copy to tmp, then rename)
        tmp_path = self.base_dir / "champion.pt.tmp"
        try:
            shutil.copy2(promote_source, tmp_path)
            tmp_path.replace(self.champion_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

        # 2. Keep legacy model.pt in sync
        tmp_path2 = self.base_dir / "model.pt.tmp"
        try:
            shutil.copy2(self.champion_path, tmp_path2)
            tmp_path2.replace(self.legacy_path)
        except BaseException:
            tmp_path2.unlink(missing_ok=True)
            raise

        logger.info("Promoted candidate → champion (gen %d, reason=%s)", generation, reason)

        # 3. Record in manifest
        manifest = self._read_manifest()
        parent_gen = manifest.get("current_champion_generation")
        entry = {
            "generation": generation,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "promoted_from": parent_gen,
            "reason": reason,
            **(metrics or {}),
        }
        manifest["current_champion_generation"] = generation
        if metrics and metrics.get("modelProfile"):
            manifest["current_model_profile"] = metrics["modelProfile"]
        manifest.setdefault("history", []).append(entry)
        self._write_manifest(manifest)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _read_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"variant": self.variant, "current_champion_generation": None, "history": []}

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def read_manifest(self) -> dict[str, Any]:
        """Public access to manifest data."""
        return self._read_manifest()

    # ------------------------------------------------------------------
    # Legacy migration
    # ------------------------------------------------------------------

    def _migrate_legacy(self) -> None:
        """If model.pt exists but champion.pt doesn't, bootstrap champion from it."""
        if self.legacy_path.exists() and not self.champion_path.exists():
            shutil.copy2(self.legacy_path, self.champion_path)
            logger.info("Migrated legacy model.pt → champion.pt for %s", self.variant)
