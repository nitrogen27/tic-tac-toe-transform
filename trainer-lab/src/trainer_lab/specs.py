"""Unified board and variant specifications for padded-board training.

This module is the single source of truth for board-aware training/runtime
contracts across curriculum and production lanes.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

PADDED_BOARD_SIZE = 16
PADDED_POLICY_SIZE = PADDED_BOARD_SIZE * PADDED_BOARD_SIZE
NEURAL_CONTRACT_ID = f"padded{PADDED_BOARD_SIZE}-policy{PADDED_POLICY_SIZE}-top_left-v1"


@dataclass(frozen=True)
class VariantSpec:
    """Canonical description of a playable/training variant."""

    variant_id: str
    board_size: int
    win_length: int
    padded_size: int = PADDED_BOARD_SIZE
    policy_size: int = PADDED_POLICY_SIZE
    rule_constraints: tuple[str, ...] = ()
    curriculum_stage: str = "production"
    promotion_scope: str = "variant_exact"
    compatibility_key: str = NEURAL_CONTRACT_ID

    @property
    def serving_variant_id(self) -> str:
        return self.variant_id

    @property
    def production_variant_id(self) -> str:
        if self.variant_id.startswith("gomoku") and self.curriculum_stage == "curriculum":
            return f"gomoku{self.board_size}"
        return self.variant_id

    @property
    def is_curriculum(self) -> bool:
        return self.curriculum_stage == "curriculum"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "variantId": self.variant_id,
            "boardSize": self.board_size,
            "winLength": self.win_length,
            "paddedSize": self.padded_size,
            "policySize": self.policy_size,
            "ruleConstraints": list(self.rule_constraints),
            "curriculumStage": self.curriculum_stage,
            "promotionScope": self.promotion_scope,
            "compatibilityKey": self.compatibility_key,
            "servingVariantId": self.serving_variant_id,
            "productionVariantId": self.production_variant_id,
        }


def _gomoku_rule_constraints() -> tuple[str, ...]:
    return ("freestyle", "five_in_row", "no_forbidden_moves")


def resolve_variant_spec(variant: str) -> VariantSpec:
    normalized = str(variant or "").strip().lower()
    if normalized == "ttt3":
        return VariantSpec(
            variant_id="ttt3",
            board_size=3,
            win_length=3,
            rule_constraints=("tic_tac_toe", "three_in_row"),
        )
    if normalized == "ttt5":
        return VariantSpec(
            variant_id="ttt5",
            board_size=5,
            win_length=4,
            rule_constraints=("tic_tac_toe", "four_in_row"),
        )

    match = re.fullmatch(r"gomoku(?P<size>\d+)(?:[_-](?P<stage>curriculum))?", normalized)
    if match:
        board_size = int(match.group("size"))
        if not 7 <= board_size <= PADDED_BOARD_SIZE:
            raise ValueError(f"Unsupported Gomoku board size: {board_size}")
        stage = "curriculum" if match.group("stage") else "production"
        return VariantSpec(
            variant_id=normalized,
            board_size=board_size,
            win_length=5,
            rule_constraints=_gomoku_rule_constraints(),
            curriculum_stage=stage,
            promotion_scope="curriculum_only" if stage == "curriculum" else "variant_exact",
        )

    raise ValueError(f"Unsupported variant: {variant}")


def variant_metadata_matches(expected: VariantSpec, metadata: dict[str, Any] | None) -> bool:
    """Return True when serialized metadata is compatible with *expected*.

    Legacy checkpoints may not carry structured spec metadata yet; callers can
    treat missing metadata as a soft-compatibility case outside this helper.
    """

    if not metadata:
        return False
    candidate = metadata.get("variantSpec")
    if not isinstance(candidate, dict):
        candidate = metadata
    return (
        str(candidate.get("variantId") or "") == expected.variant_id
        and int(candidate.get("boardSize") or 0) == expected.board_size
        and int(candidate.get("winLength") or 0) == expected.win_length
        and int(candidate.get("paddedSize") or 0) == expected.padded_size
        and int(candidate.get("policySize") or 0) == expected.policy_size
        and str(candidate.get("curriculumStage") or "") == expected.curriculum_stage
        and str(candidate.get("compatibilityKey") or "") == expected.compatibility_key
        and str(candidate.get("servingVariantId") or "") == expected.serving_variant_id
    )
