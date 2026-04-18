from __future__ import annotations

import pytest

from trainer_lab.specs import (
    NEURAL_CONTRACT_ID,
    PADDED_BOARD_SIZE,
    PADDED_POLICY_SIZE,
    resolve_variant_spec,
    variant_metadata_matches,
)


def test_gomoku9_curriculum_spec_uses_unified_padded_contract() -> None:
    spec = resolve_variant_spec("gomoku9_curriculum")

    assert spec.board_size == 9
    assert spec.win_length == 5
    assert spec.curriculum_stage == "curriculum"
    assert spec.padded_size == PADDED_BOARD_SIZE
    assert spec.policy_size == PADDED_POLICY_SIZE
    assert spec.compatibility_key == NEURAL_CONTRACT_ID
    assert spec.production_variant_id == "gomoku9"
    assert spec.promotion_scope == "curriculum_only"


def test_production_gomoku15_spec_is_separate_from_curriculum() -> None:
    spec = resolve_variant_spec("gomoku15")

    assert spec.board_size == 15
    assert spec.curriculum_stage == "production"
    assert spec.production_variant_id == "gomoku15"
    assert spec.promotion_scope == "variant_exact"


def test_variant_metadata_matches_requires_exact_structured_spec() -> None:
    curriculum = resolve_variant_spec("gomoku9_curriculum")
    production = resolve_variant_spec("gomoku9")

    assert variant_metadata_matches(curriculum, {"variantSpec": curriculum.to_metadata()}) is True
    assert variant_metadata_matches(production, {"variantSpec": curriculum.to_metadata()}) is False


def test_unsupported_gomoku_size_raises() -> None:
    with pytest.raises(ValueError):
        resolve_variant_spec("gomoku17")
