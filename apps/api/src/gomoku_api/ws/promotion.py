"""Promotion gate: decides whether a candidate model should replace the champion.

A candidate must pass multiple criteria to be promoted:
1. winrate vs champion >= threshold (default 55%)
2. block accuracy >= threshold (default 88%)
3. win accuracy >= threshold (default 70%)
4. winrate vs algorithm should not degrade significantly
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from gomoku_api.ws.arena_eval import ArenaResult

logger = logging.getLogger(__name__)


@dataclass
class PromotionDecision:
    """Result of the promotion gate evaluation."""
    promoted: bool
    winrate_vs_champion: float | None
    winrate_vs_algorithm: float | None
    block_accuracy: float
    win_accuracy: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "promoted": self.promoted,
            "winrateVsChampion": self.winrate_vs_champion,
            "winrateVsAlgorithm": self.winrate_vs_algorithm,
            "blockAccuracy": round(self.block_accuracy, 1),
            "winAccuracy": round(self.win_accuracy, 1),
            "reason": self.reason,
        }


def evaluate_promotion(
    quick_arena: ArenaResult | None,
    strong_arena: ArenaResult | None,
    block_accuracy: float,
    win_accuracy: float,
    *,
    champion_threshold: float = 0.55,
    block_threshold: float = 88.0,
    win_threshold: float = 70.0,
    prev_algo_winrate: float | None = None,
    algo_regression_margin: float = 0.10,
    require_champion_match: bool = False,
) -> PromotionDecision:
    """Evaluate whether candidate should be promoted to champion.

    Parameters
    ----------
    quick_arena : Arena result vs champion (can be None if no champion exists)
    strong_arena : Arena result vs algorithm/engine (can be None)
    block_accuracy : Model accuracy on block-only tactical positions (0-100)
    win_accuracy : Model accuracy on win-only tactical positions (0-100)
    champion_threshold : Minimum winrate vs champion to promote
    block_threshold : Minimum block accuracy to promote
    win_threshold : Minimum win accuracy to promote
    prev_algo_winrate : Previous champion's winrate vs algorithm (for regression check)
    algo_regression_margin : Max allowed drop in algo winrate
    """
    wr_champion = quick_arena.winrate_a if quick_arena and quick_arena.total > 0 else None
    wr_algo = strong_arena.winrate_a if strong_arena and strong_arena.total > 0 else None

    reasons: list[str] = []

    # Check block accuracy
    if block_accuracy < block_threshold:
        reasons.append(f"blockAcc {block_accuracy:.1f}% < {block_threshold:.0f}%")

    # Check win accuracy
    if win_accuracy < win_threshold:
        reasons.append(f"winAcc {win_accuracy:.1f}% < {win_threshold:.0f}%")

    if require_champion_match and quick_arena is None:
        reasons.append("wrVsChampion unavailable")

    # Check winrate vs champion
    if wr_champion is not None and wr_champion < champion_threshold:
        reasons.append(f"wrVsChampion {wr_champion:.2f} < {champion_threshold:.2f}")

    # Check algorithm regression
    if wr_algo is not None and prev_algo_winrate is not None:
        if wr_algo < prev_algo_winrate - algo_regression_margin:
            reasons.append(f"wrVsAlgo regressed {wr_algo:.2f} < {prev_algo_winrate:.2f} - {algo_regression_margin}")

    if reasons:
        reason_str = "; ".join(reasons)
        logger.info("Promotion REJECTED: %s", reason_str)
        return PromotionDecision(
            promoted=False,
            winrate_vs_champion=wr_champion,
            winrate_vs_algorithm=wr_algo,
            block_accuracy=block_accuracy,
            win_accuracy=win_accuracy,
            reason=reason_str,
        )

    reason_str = "passed_all_gates"
    logger.info("Promotion APPROVED: wrChampion=%s wrAlgo=%s blockAcc=%.1f winAcc=%.1f",
                wr_champion, wr_algo, block_accuracy, win_accuracy)
    return PromotionDecision(
        promoted=True,
        winrate_vs_champion=wr_champion,
        winrate_vs_algorithm=wr_algo,
        block_accuracy=block_accuracy,
        win_accuracy=win_accuracy,
        reason=reason_str,
    )
