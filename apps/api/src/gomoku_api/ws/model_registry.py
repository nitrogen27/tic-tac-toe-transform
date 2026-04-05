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
    def champion_path(self) -> Path:
        return self.base_dir / "champion.pt"

    @property
    def legacy_path(self) -> Path:
        return self.base_dir / "model.pt"

    @property
    def manifest_path(self) -> Path:
        return self.base_dir / "manifest.json"

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_candidate(self, model: nn.Module, *, generation: int = 0, metrics: dict[str, Any] | None = None) -> None:
        """Save model state_dict as candidate checkpoint."""
        torch.save(model.state_dict(), self.candidate_path)
        logger.info("Saved candidate checkpoint: %s (gen %d)", self.candidate_path, generation)

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

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def promote_candidate(self, *, generation: int = 0, metrics: dict[str, Any] | None = None, reason: str = "passed_promotion_gate") -> None:
        """Atomic promotion: candidate → champion (+ legacy model.pt sync).

        Uses temp file + os.replace for crash safety: if the process dies
        mid-copy, the previous champion remains intact.

        Also appends an entry to manifest.json.
        """
        if not self.candidate_path.exists():
            raise FileNotFoundError(f"No candidate to promote: {self.candidate_path}")

        # 1. Copy candidate → champion (crash-safe: copy to tmp, then rename)
        tmp_path = self.base_dir / "champion.pt.tmp"
        try:
            shutil.copy2(self.candidate_path, tmp_path)
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
