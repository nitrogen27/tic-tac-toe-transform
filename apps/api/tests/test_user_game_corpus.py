from __future__ import annotations

from pathlib import Path

import pytest

from gomoku_api.ws.user_game_corpus import (
    CONVERSION_TYPES,
    UserGameCorpus,
    _classify_move_quality,
    analyze_game,
)


class FakeEngine:
    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def analyze_position(self, board, current, board_size, win_len):
        if current == 1 and board[:4] == [1, 1, 1, 0]:
            return {"bestMove": 3, "value": 1.0}
        if current == 2 and board[10] == 1:
            return {"bestMove": 3, "value": 0.0}
        if current == 2 and board[3] == 1:
            return {"bestMove": 10, "value": -1.0}
        return {"bestMove": next((idx for idx, cell in enumerate(board) if cell == 0), -1), "value": 0.0}

    async def suggest_moves(self, board, current, board_size, win_len, *, top_n=5):
        if current == 1 and board[:4] == [1, 1, 1, 0]:
            return [
                {"move": 3, "score": 10.0},
                {"move": 10, "score": 1.0},
            ]
        return [{"move": next((idx for idx, cell in enumerate(board) if cell == 0), -1), "score": 1.0}]


class MissingBestMoveEngine(FakeEngine):
    async def analyze_position(self, board, current, board_size, win_len):
        return {"bestMove": -1, "value": 0.4}

    async def suggest_moves(self, board, current, board_size, win_len, *, top_n=5):
        legal = next((idx for idx, cell in enumerate(board) if cell == 0), -1)
        return [{"move": legal, "score": 1.0}] if legal >= 0 else []


def _make_position(*, source: str, mistake_type: str = "positional", quality: str = "good", player: int = 1, conversion: bool = False, losing: bool = False, marker: int = 0) -> dict:
    policy = [0.0] * 256
    policy[marker] = 1.0
    board = [[0] * 5 for _ in range(5)]
    r, c = divmod(marker % 25, 5)
    board[r][c] = player
    return {
        "board_size": 5,
        "board": board,
        "current_player": player,
        "last_move": [r, c],
        "policy": policy,
        "value": 0.25,
        "source": source,
        "motif": "conversion" if conversion else "mid",
        "sampleWeight": 1.5,
        "playerFocus": player,
        "conversionTarget": conversion,
        "teacher_best_move": 0,
        "teacher_value": 0.5,
        "user_move": 1,
        "user_move_rank": 5,
        "move_quality": quality,
        "mistake_type": mistake_type,
        "value_loss": 0.4,
        "game_id": "g1",
        "game_result": 2,
        "move_number": 8,
        "game_phase": "mid",
        "is_losing_side": losing,
        "had_immediate_win": False,
        "had_immediate_block": False,
        "actor": "model",
    }


def test_classify_move_quality_flags_missed_win() -> None:
    quality, mistake = _classify_move_quality(
        0.95,
        user_move=10,
        teacher_best=3,
        rank=None,
        had_win=True,
        had_block=False,
        teacher_value=1.0,
        actual_value=0.0,
    )

    assert quality == "blunder"
    assert mistake == "missed_win"


def test_classify_move_quality_detects_conversion_miss() -> None:
    quality, mistake = _classify_move_quality(
        0.35,
        user_move=10,
        teacher_best=3,
        rank=5,
        had_win=False,
        had_block=False,
        teacher_value=0.65,
        actual_value=0.0,
    )

    assert quality == "mistake"
    assert mistake in CONVERSION_TYPES


def test_classify_move_quality_does_not_overcall_conversion_for_top_rank_move() -> None:
    quality, mistake = _classify_move_quality(
        0.25,
        user_move=10,
        teacher_best=3,
        rank=2,
        had_win=False,
        had_block=False,
        teacher_value=0.70,
        actual_value=0.10,
    )

    assert quality == "inaccuracy"
    assert mistake == "positional"


@pytest.mark.asyncio
async def test_analyze_game_builds_teacher_backed_position() -> None:
    board = [0] * 25
    board[0] = 1
    board[1] = 1
    board[2] = 1
    game = {
        "gameId": "g1",
        "playerRole": 1,
        "winner": 1,
        "moves": [
            {"board": board, "move": 10, "current": 1, "variant": "ttt5"},
        ],
    }

    positions = await analyze_game(game, 5, 4, FakeEngine())

    assert len(positions) == 1
    pos = positions[0]
    assert pos["teacher_best_move"] == 3
    assert pos["mistake_type"] == "missed_win"
    assert pos["source"] == "user_conversion"
    assert pos["conversionTarget"] is True
    assert sum(pos["policy"]) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.asyncio
async def test_analyze_game_recovers_teacher_move_from_hints_when_best_move_missing() -> None:
    board = [0] * 25
    board[0] = 1
    game = {
        "gameId": "g2",
        "playerRole": 1,
        "winner": 0,
        "moves": [
            {"board": board, "move": 6, "current": 1, "variant": "ttt5"},
        ],
    }

    positions = await analyze_game(game, 5, 4, MissingBestMoveEngine())

    assert len(positions) == 1
    pos = positions[0]
    assert pos["teacher_best_move"] == 1
    assert sum(pos["policy"]) == pytest.approx(1.0, abs=1e-6)


def test_user_game_corpus_ingest_populates_buckets(tmp_path: Path) -> None:
    corpus = UserGameCorpus("ttt5")
    corpus.path = tmp_path / "user_corpus.json"

    positions = [
        _make_position(source="user_game", quality="good", marker=0),
        _make_position(source="user_mistake", quality="mistake", mistake_type="positional", marker=1),
        _make_position(source="user_conversion", quality="blunder", mistake_type="missed_win", conversion=True, marker=2),
        _make_position(source="user_game", quality="good", player=2, losing=True, marker=3),
    ]

    stats = corpus.ingest_analyzed_game(positions)

    assert stats["recentCount"] == 4
    assert stats["hardMistakeCount"] >= 2
    assert stats["conversionCount"] >= 1
    assert stats["weakSideCount"] >= 1
    assert stats["p1FocusCount"] >= 1


def test_user_game_corpus_separates_weak_side_from_p1_focus(tmp_path: Path) -> None:
    corpus = UserGameCorpus("ttt5")
    corpus.path = tmp_path / "user_corpus.json"

    positions = [
        _make_position(source="user_game", quality="good", player=1, losing=False, marker=8),
        _make_position(source="user_game", quality="good", player=2, losing=True, marker=9),
    ]

    stats = corpus.ingest_analyzed_game(positions)

    assert stats["weakSideCount"] == 1
    assert stats["p1FocusCount"] == 1


def test_user_game_corpus_save_and_load_roundtrip(tmp_path: Path) -> None:
    corpus = UserGameCorpus("ttt5")
    corpus.path = tmp_path / "user_corpus.json"
    corpus.ingest_analyzed_game([
        _make_position(source="user_mistake", quality="mistake", mistake_type="positional", marker=4),
        _make_position(source="user_conversion", quality="blunder", mistake_type="missed_win", conversion=True, marker=5),
    ])
    saved = corpus.save()

    loaded = UserGameCorpus("ttt5")
    loaded.path = saved
    assert loaded.load() is True
    stats = loaded.stats()
    assert stats["recentCount"] >= 2
    assert stats["conversionCount"] >= 1
    assert "p1FocusCount" in stats


def test_pool_for_builder_strips_analysis_fields(tmp_path: Path) -> None:
    corpus = UserGameCorpus("ttt5")
    corpus.path = tmp_path / "user_corpus.json"
    corpus.ingest_analyzed_game([
        _make_position(source="user_mistake", quality="mistake", mistake_type="positional", marker=6),
        _make_position(source="user_conversion", quality="blunder", mistake_type="missed_win", conversion=True, marker=7),
    ])

    pool = corpus.get_pool_for_builder(4, mode="consolidate")

    assert pool
    assert "teacher_best_move" not in pool[0]
    assert "move_quality" not in pool[0]
    assert {"board", "policy", "value", "source", "sampleWeight"} <= set(pool[0].keys())
