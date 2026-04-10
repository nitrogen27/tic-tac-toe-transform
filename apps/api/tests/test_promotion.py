from gomoku_api.ws.arena_eval import ArenaResult
from gomoku_api.ws.promotion import evaluate_promotion


def test_evaluate_promotion_rejects_when_champion_match_required_but_missing() -> None:
    decision = evaluate_promotion(
        quick_arena=None,
        strong_arena=ArenaResult(wins_a=8, wins_b=2, draws=0, total=10),
        block_accuracy=95.0,
        win_accuracy=95.0,
        require_champion_match=True,
    )

    assert decision.promoted is False
    assert "wrVsChampion unavailable" in decision.reason


def test_evaluate_promotion_accepts_when_champion_match_is_present_and_good() -> None:
    decision = evaluate_promotion(
        quick_arena=ArenaResult(wins_a=6, wins_b=2, draws=2, total=10),
        strong_arena=ArenaResult(wins_a=8, wins_b=2, draws=0, total=10),
        block_accuracy=95.0,
        win_accuracy=95.0,
        require_champion_match=True,
    )

    assert decision.promoted is True
    assert decision.winrate_vs_champion == 0.7
