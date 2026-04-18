from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from gomoku_api.ws import model_registry
from gomoku_api.ws.model_registry import ModelRegistry
from trainer_lab.specs import resolve_variant_spec


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))


def test_working_candidate_does_not_overwrite_active_candidate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(model_registry, "SAVED_DIR", tmp_path)
    registry = ModelRegistry("ttt5")
    model = _tiny_model()

    registry.save_working_candidate(model, generation=1, metrics={"phase": "foundation"})

    assert registry.working_candidate_path.exists()
    assert registry.read_working_candidate_meta() is not None
    assert not registry.candidate_path.exists()
    assert registry.read_candidate_meta() is None


def test_commit_working_candidate_creates_active_candidate_with_meta(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(model_registry, "SAVED_DIR", tmp_path)
    registry = ModelRegistry("ttt5")
    model = _tiny_model()

    registry.save_working_candidate(model, generation=3, metrics={"selectedCheckpointWinrate": 0.75})
    registry.commit_working_candidate(generation=3, metrics={"selectedCheckpointWinrate": 0.75})

    assert registry.candidate_path.exists()
    meta = registry.read_candidate_meta()
    assert meta is not None
    assert meta["selectedCheckpointWinrate"] == 0.75


def test_promote_candidate_can_use_working_checkpoint_source(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(model_registry, "SAVED_DIR", tmp_path)
    registry = ModelRegistry("ttt5")
    model = _tiny_model()

    registry.save_working_candidate(model, generation=9, metrics={"winrateVsAlgorithm": 0.9})
    registry.promote_candidate(
        generation=9,
        metrics={"winrateVsAlgorithm": 0.9},
        source_path=registry.working_candidate_path,
    )

    assert registry.candidate_path.exists()
    assert registry.champion_path.exists()
    assert registry.legacy_path.exists()
    manifest = registry.read_manifest()
    assert manifest["history"][-1]["winrateVsAlgorithm"] == 0.9


def test_serving_summary_prefers_champion(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(model_registry, "SAVED_DIR", tmp_path)
    registry = ModelRegistry("ttt5")
    model = _tiny_model()

    registry.save_working_candidate(model, generation=9, metrics={"winrateVsAlgorithm": 0.9})
    registry.promote_candidate(
        generation=9,
        metrics={"winrateVsAlgorithm": 0.9},
        source_path=registry.working_candidate_path,
    )

    summary = registry.serving_summary()

    assert summary["servingReady"] is True
    assert summary["servingSource"] == "champion"
    assert summary["servingGeneration"] == 9


def test_registry_heals_missing_serving_files_from_working_candidate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(model_registry, "SAVED_DIR", tmp_path)
    first = ModelRegistry("ttt5")
    model = _tiny_model()

    first.save_working_candidate(model, generation=4, metrics={"phase": "foundation"})

    healed = ModelRegistry("ttt5")

    assert healed.working_candidate_path.exists()
    assert healed.candidate_path.exists()
    assert healed.champion_path.exists()
    assert healed.legacy_path.exists()


def test_clear_checkpoints_removes_serving_and_working_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(model_registry, "SAVED_DIR", tmp_path)
    registry = ModelRegistry("ttt5")
    model = _tiny_model()

    registry.save_working_candidate(model, generation=4, metrics={"phase": "foundation"})
    registry.promote_candidate(
        generation=4,
        metrics={"winrateVsAlgorithm": 0.8},
        source_path=registry.working_candidate_path,
    )

    registry.clear_checkpoints(preserve_history=True)

    assert not registry.working_candidate_path.exists()
    assert not registry.candidate_path.exists()
    assert not registry.champion_path.exists()
    assert not registry.legacy_path.exists()
    manifest = registry.read_manifest()
    assert manifest["current_champion_generation"] is None
    assert len(manifest["history"]) == 1


def test_resolve_serving_checkpoint_rejects_incompatible_structured_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(model_registry, "SAVED_DIR", tmp_path)
    registry = ModelRegistry("gomoku15")
    model = _tiny_model()
    incompatible_spec = resolve_variant_spec("gomoku9_curriculum")

    registry.save_working_candidate(
        model,
        generation=7,
        metrics={"variantSpec": incompatible_spec.to_metadata()},
    )
    registry.promote_candidate(
        generation=7,
        metrics={"variantSpec": incompatible_spec.to_metadata()},
        source_path=registry.working_candidate_path,
    )

    path, source = registry.resolve_serving_checkpoint(expected_spec=resolve_variant_spec("gomoku15"))

    assert path is None
    assert source == "incompatible"
