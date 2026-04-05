from __future__ import annotations

from gomoku_api.ws import commentary_service


def test_analyze_move_commentary_detects_found_win() -> None:
    board = [
        1, 1, 1, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    ]

    result = commentary_service.analyze_move_commentary(
        board,
        move=3,
        current=1,
        variant="ttt5",
        style="coach",
        actor="player",
    )

    assert result["category"] == "found_win"
    assert result["mood"] == "positive"
    assert result["moveLabel"] == "D1"


def test_analyze_move_commentary_detects_missed_block() -> None:
    board = [
        2, 2, 2, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    ]

    result = commentary_service.analyze_move_commentary(
        board,
        move=6,
        current=1,
        variant="ttt5",
        style="hint",
        actor="player",
    )

    assert result["category"] == "missed_block"
    assert result["mood"] == "danger"
    assert result["suggestedMove"] == 3
    assert result["bestMoveLabel"] == "D1"


def test_analyze_move_commentary_uses_model_scores_for_lost_advantage(monkeypatch) -> None:
    board = [0] * 25

    monkeypatch.setattr(commentary_service, "_get_model", lambda _variant: object())
    monkeypatch.setattr(
        commentary_service,
        "_evaluate_afterstate_values",
        lambda _model, _board, _current, _board_size, legal: {move: (0.8 if move == 12 else 0.05) for move in legal},
    )

    result = commentary_service.analyze_move_commentary(
        board,
        move=0,
        current=1,
        variant="ttt5",
        style="emotional",
        actor="player",
    )

    assert result["category"] == "lost_advantage"
    assert result["bestMove"] == 12
    assert result["scoreGap"] > 0.35
