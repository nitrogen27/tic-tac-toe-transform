"""Teacher-backed user game corpus for TTT5 post-hoc analysis.

This module turns finished product games into structured training positions.
It is intentionally focused on the small-board TTT5 workflow where user games
should become a useful supervised corpus instead of raw move logs.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

from gomoku_api.ws.offline_gen import _soft_policy_from_engine_hints
from gomoku_api.ws.oracle_backends import create_oracle_evaluator
from gomoku_api.ws.predict_service import _find_immediate_move

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).resolve().parents[5] / "saved"

RECENT_MAX = 2048
MISTAKE_MAX = 1024
CONVERSION_MAX = 512
WEAK_SIDE_MAX = 512
P1_FOCUS_MAX = 512

TRAINING_FIELDS = {
    "board",
    "board_size",
    "current_player",
    "last_move",
    "policy",
    "value",
    "source",
    "motif",
    "sampleWeight",
    "playerFocus",
    "conversionTarget",
}

CONVERSION_TYPES = {
    "missed_win",
    "missed_block",
    "missed_fork",
    "conversion_miss",
    "draw_instead_of_win",
}


def resolve_variant_spec(variant: str) -> tuple[int, int]:
    if variant == "ttt3":
        return 3, 3
    if variant == "ttt5":
        return 5, 4
    if variant.startswith("gomoku"):
        digits = "".join(ch for ch in variant if ch.isdigit())
        board_size = int(digits) if digits else 15
        return board_size, 5
    return 15, 5


def _flat_to_board2d(board: list[int], board_size: int) -> list[list[int]]:
    return [list(board[r * board_size:(r + 1) * board_size]) for r in range(board_size)]


def _nxn_winner(board: list[int], board_size: int, win_len: int, last_move: int) -> int:
    if last_move < 0 or last_move >= len(board):
        return 0
    player = board[last_move]
    if player == 0:
        return 0
    r, c = divmod(last_move, board_size)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for step in range(1, win_len):
            nr, nc = r + dr * step, c + dc * step
            if 0 <= nr < board_size and 0 <= nc < board_size and board[nr * board_size + nc] == player:
                count += 1
            else:
                break
        for step in range(1, win_len):
            nr, nc = r - dr * step, c - dc * step
            if 0 <= nr < board_size and 0 <= nc < board_size and board[nr * board_size + nc] == player:
                count += 1
            else:
                break
        if count >= win_len:
            return player
    return 0


def _classify_phase(board_size: int, move_number: int) -> str:
    total_cells = board_size * board_size
    if board_size <= 5:
        if move_number <= 3:
            return "opening"
        if move_number <= 10:
            return "early_mid"
        if move_number <= 16:
            return "mid"
        return "late"
    if move_number <= max(3, total_cells // 8):
        return "opening"
    if move_number <= max(10, total_cells // 3):
        return "mid"
    return "late"


def _normalize_engine_value(value: Any) -> float:
    try:
        return max(-1.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _rank_move_from_hints(move: int, hints: list[dict[str, Any]] | None) -> int | None:
    if move < 0:
        return None
    rank = 1
    for hint in hints or []:
        try:
            hint_move = int(hint.get("move", -1))
        except Exception:
            continue
        if hint_move == move:
            return rank
        rank += 1
    return None


def _classify_move_quality(
    value_loss: float,
    user_move: int,
    teacher_best: int,
    rank: int | None,
    had_win: bool,
    had_block: bool,
    *,
    teacher_value: float = 0.0,
    actual_value: float = 0.0,
) -> tuple[str, str]:
    if had_win and user_move != teacher_best:
        return "blunder", "missed_win"
    if had_block and user_move != teacher_best:
        return "blunder", "missed_block"
    if user_move == teacher_best or value_loss < 0.05:
        return "best", "best"
    if rank is not None and rank <= 3 and value_loss < 0.15:
        return "good", "good"
    if teacher_value >= 0.45 and actual_value <= 0.20 and value_loss >= 0.20:
        if rank is not None and rank <= 2 and value_loss < 0.35:
            return "inaccuracy", "positional"
        return ("blunder", "draw_instead_of_win") if value_loss >= 0.65 else ("mistake", "conversion_miss")
    if teacher_value >= 0.35 and actual_value <= 0.10 and value_loss >= 0.25:
        return ("mistake", "missed_fork") if rank is None or rank > 1 else ("inaccuracy", "conversion_miss")
    if value_loss < 0.35:
        return "inaccuracy", "positional"
    if value_loss < 0.65:
        return "mistake", "positional"
    return "blunder", "positional"


def _sample_weight_for_position(
    *,
    quality: str,
    mistake_type: str,
    actor: str,
    current_player: int,
    teacher_value: float,
) -> float:
    weight = 1.0
    if quality == "good":
        weight += 0.05
    elif quality == "mistake":
        weight += 0.35
    elif quality == "blunder":
        weight += 0.65

    if mistake_type in CONVERSION_TYPES:
        weight += 0.35
    if teacher_value >= 0.45:
        weight += 0.20
    if actor == "model":
        weight += 0.10
    if current_player == 1:
        weight += 0.12
    return round(weight, 4)


def _position_source(quality: str, mistake_type: str) -> str:
    if mistake_type in CONVERSION_TYPES:
        return "user_conversion"
    if quality in {"mistake", "blunder"}:
        return "user_mistake"
    return "user_game"


def _strip_training_position(position: dict[str, Any]) -> dict[str, Any]:
    cleaned = {key: value for key, value in position.items() if key in TRAINING_FIELDS}
    cleaned.setdefault("source", "user_game")
    cleaned.setdefault("motif", "user")
    cleaned.setdefault("sampleWeight", 1.0)
    cleaned.setdefault("playerFocus", int(position.get("current_player", 0) or 0))
    cleaned.setdefault("conversionTarget", bool(position.get("conversionTarget", False)))
    return cleaned


def _resolve_teacher_move(
    teacher_best: int,
    legal: list[int],
    hints: list[dict[str, Any]] | None,
) -> int:
    if teacher_best in legal:
        return teacher_best
    for hint in hints or []:
        try:
            move = int(hint.get("move", -1))
        except Exception:
            continue
        if move in legal:
            return move
    return -1


async def analyze_game(
    game: dict[str, Any],
    board_size: int,
    win_len: int,
    engine_eval: Any,
) -> list[dict[str, Any]]:
    """Analyze a finished game and relabel positions with teacher targets."""
    positions: list[dict[str, Any]] = []
    moves = list(game.get("moves", []))
    if not moves:
        return positions

    winner = int(game.get("winner", 0) or 0)
    model_player = int(game.get("playerRole", 2) or 2)
    previous_move = -1
    game_id = str(game.get("gameId", ""))

    for move_index, item in enumerate(moves, start=1):
        board = [int(v) for v in item.get("board", [])]
        current = int(item.get("current", 0) or 0)
        user_move = int(item.get("move", -1))
        if len(board) != board_size * board_size or current not in (1, 2):
            previous_move = user_move if user_move >= 0 else previous_move
            continue
        legal = [idx for idx, cell in enumerate(board) if cell == 0]
        if user_move not in legal:
            previous_move = user_move if user_move >= 0 else previous_move
            continue

        teacher = await engine_eval.analyze_position(board, current, board_size, win_len)
        teacher_best = int(teacher.get("bestMove", -1))
        teacher_value = _normalize_engine_value(teacher.get("value", 0.0))
        hints = await engine_eval.suggest_moves(board, current, board_size, win_len, top_n=5)
        teacher_best = _resolve_teacher_move(teacher_best, legal, hints)
        if teacher_best < 0:
            logger.debug("Skipping user corpus position without valid teacher move [game=%s move=%s]", game_id, move_index)
            previous_move = user_move
            continue
        policy = _soft_policy_from_engine_hints(teacher_best, board, board_size, hints)
        if sum(float(v) for v in policy) <= 0.0:
            logger.debug("Skipping user corpus position with empty policy target [game=%s move=%s]", game_id, move_index)
            previous_move = user_move
            continue

        after_board = list(board)
        after_board[user_move] = current
        next_player = 2 if current == 1 else 1
        after_value = -_normalize_engine_value(
            (await engine_eval.analyze_position(after_board, next_player, board_size, win_len)).get("value", 0.0)
        )
        value_loss = max(0.0, teacher_value - after_value)

        immediate_win = _find_immediate_move(board, board_size, win_len, current)
        immediate_block = _find_immediate_move(board, board_size, win_len, next_player)
        had_immediate_win = immediate_win is not None
        had_immediate_block = immediate_block is not None
        rank = _rank_move_from_hints(user_move, hints)
        quality, mistake_type = _classify_move_quality(
            value_loss,
            user_move,
            teacher_best,
            rank,
            had_immediate_win,
            had_immediate_block,
            teacher_value=teacher_value,
            actual_value=after_value,
        )
        actor = "model" if current == model_player else "user"
        source = _position_source(quality, mistake_type)
        phase = _classify_phase(board_size, move_index)
        losing_side = winner in (1, 2) and winner != current
        conversion_target = mistake_type in CONVERSION_TYPES or teacher_value >= 0.45
        motif = mistake_type if mistake_type not in {"best", "good"} else phase

        positions.append({
            "board_size": board_size,
            "board": _flat_to_board2d(board, board_size),
            "current_player": current,
            "last_move": list(divmod(previous_move, board_size)) if previous_move >= 0 else None,
            "policy": policy,
            "value": teacher_value,
            "source": source,
            "motif": motif,
            "sampleWeight": _sample_weight_for_position(
                quality=quality,
                mistake_type=mistake_type,
                actor=actor,
                current_player=current,
                teacher_value=teacher_value,
            ),
            "playerFocus": current,
            "conversionTarget": conversion_target,
            "teacher_best_move": teacher_best,
            "teacher_value": teacher_value,
            "user_move": user_move,
            "user_move_rank": rank,
            "move_quality": quality,
            "mistake_type": mistake_type,
            "value_loss": round(value_loss, 4),
            "game_id": game_id,
            "game_result": winner,
            "move_number": move_index,
            "game_phase": phase,
            "is_losing_side": bool(losing_side),
            "had_immediate_win": had_immediate_win,
            "had_immediate_block": had_immediate_block,
            "actor": actor,
        })
        previous_move = user_move

    return positions


class UserGameCorpus:
    """Persistent multi-bucket corpus built from analyzed user games."""

    def __init__(self, variant: str) -> None:
        self.variant = variant
        self.path = SAVED_DIR / f"{variant}_resnet" / "user_corpus.json"
        self._recent_positions: deque[dict[str, Any]] = deque(maxlen=RECENT_MAX)
        self._hard_mistakes: list[dict[str, Any]] = []
        self._conversion_failures: list[dict[str, Any]] = []
        self._weak_side: list[dict[str, Any]] = []
        self._p1_focus: list[dict[str, Any]] = []

    def _merge_positions(self, existing: list[dict[str, Any]], incoming: list[dict[str, Any]], *, max_size: int) -> list[dict[str, Any]]:
        from gomoku_api.ws.train_service_ws import _merge_position_bank

        return _merge_position_bank(existing, incoming, max_size=max_size)

    def ingest_analyzed_game(self, positions: list[dict[str, Any]]) -> dict[str, Any]:
        if not positions:
            return self.stats()

        trainable = [_strip_training_position(pos) | {k: v for k, v in pos.items() if k not in TRAINING_FIELDS} for pos in positions]
        recent_before = len(self._recent_positions)
        recent_merged = self._merge_positions(list(self._recent_positions), trainable, max_size=RECENT_MAX)
        self._recent_positions = deque(recent_merged, maxlen=RECENT_MAX)

        hard = [
            pos for pos in trainable
            if str(pos.get("move_quality")) in {"mistake", "blunder"} or str(pos.get("source")) == "user_mistake"
        ]
        if hard:
            self._hard_mistakes = self._merge_positions(self._hard_mistakes, hard, max_size=MISTAKE_MAX)

        conversions = [
            pos for pos in trainable
            if str(pos.get("mistake_type")) in CONVERSION_TYPES or str(pos.get("source")) == "user_conversion"
        ]
        if conversions:
            self._conversion_failures = self._merge_positions(self._conversion_failures, conversions, max_size=CONVERSION_MAX)

        weak_side = [
            pos for pos in trainable
            if bool(pos.get("is_losing_side"))
        ]
        if weak_side:
            self._weak_side = self._merge_positions(self._weak_side, weak_side, max_size=WEAK_SIDE_MAX)

        p1_focus = [
            pos for pos in trainable
            if int(pos.get("playerFocus", 0) or 0) == 1
        ]
        if p1_focus:
            self._p1_focus = self._merge_positions(self._p1_focus, p1_focus, max_size=P1_FOCUS_MAX)

        stats = self.stats()
        stats["ingestedPositions"] = len(positions)
        stats["recentAdded"] = max(0, stats["recentCount"] - recent_before)
        return stats

    def _sample_bucket(self, bucket: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        if count <= 0 or not bucket:
            return []
        from gomoku_api.ws.train_service_ws import _sample_positions

        return _sample_positions(bucket, min(count, len(bucket)))

    def get_quick_repair_pool(self, count: int) -> list[dict[str, Any]]:
        if count <= 0:
            return []
        mistake_count = max(1, int(count * 0.50))
        conversion_count = max(1, int(count * 0.30))
        recent_count = max(0, count - mistake_count - conversion_count)

        pool: list[dict[str, Any]] = []
        pool.extend(self._sample_bucket(self._hard_mistakes, mistake_count))
        pool.extend(self._sample_bucket(self._conversion_failures, conversion_count))
        pool.extend(self._sample_bucket(list(self._recent_positions), recent_count))
        return [_strip_training_position(pos) for pos in pool[:count]]

    def get_consolidation_pool(self, count: int) -> list[dict[str, Any]]:
        if count <= 0:
            return []
        mistake_count = max(1, int(count * 0.35))
        conversion_count = max(1, int(count * 0.25))
        weak_count = max(1, int(count * 0.15))
        p1_count = max(1, int(count * 0.10))
        recent_count = max(0, count - mistake_count - conversion_count - weak_count - p1_count)

        pool: list[dict[str, Any]] = []
        pool.extend(self._sample_bucket(self._hard_mistakes, mistake_count))
        pool.extend(self._sample_bucket(self._conversion_failures, conversion_count))
        pool.extend(self._sample_bucket(self._weak_side, weak_count))
        pool.extend(self._sample_bucket(self._p1_focus, p1_count))
        pool.extend(self._sample_bucket(list(self._recent_positions), recent_count))
        return [_strip_training_position(pos) for pos in pool[:count]]

    def get_pool_for_builder(self, count: int, mode: str = "quick_repair") -> list[dict[str, Any]]:
        normalized = str(mode or "quick_repair").strip().lower()
        if normalized in {"consolidate", "consolidation"}:
            return self.get_consolidation_pool(count)
        return self.get_quick_repair_pool(count)

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "variant": self.variant,
            "updatedAt": int(time.time()),
            "recent": list(self._recent_positions),
            "hardMistakes": self._hard_mistakes,
            "conversionFailures": self._conversion_failures,
            "weakSide": self._weak_side,
            "p1Focus": self._p1_focus,
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return self.path

    def load(self) -> bool:
        if not self.path.exists():
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load user corpus %s: %s", self.path, exc)
            return False
        self._recent_positions = deque(payload.get("recent", []), maxlen=RECENT_MAX)
        self._hard_mistakes = list(payload.get("hardMistakes", []))[:MISTAKE_MAX]
        self._conversion_failures = list(payload.get("conversionFailures", []))[:CONVERSION_MAX]
        self._weak_side = list(payload.get("weakSide", []))[:WEAK_SIDE_MAX]
        self._p1_focus = list(payload.get("p1Focus", []))[:P1_FOCUS_MAX]
        return True

    def stats(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "recentCount": len(self._recent_positions),
            "hardMistakeCount": len(self._hard_mistakes),
            "conversionCount": len(self._conversion_failures),
            "weakSideCount": len(self._weak_side),
            "p1FocusCount": len(self._p1_focus),
            "path": str(self.path),
        }


async def analyze_finished_games_to_corpus(
    variant: str,
    games: list[dict[str, Any]],
    *,
    backend: str | None = None,
) -> dict[str, Any]:
    board_size, win_len = resolve_variant_spec(variant)
    engine_eval, _resolved = create_oracle_evaluator(board_size, win_len, backend=backend, role="teacher")
    await engine_eval.start()
    corpus = UserGameCorpus(variant)
    corpus.load()
    analyzed_games = 0
    analyzed_positions = 0
    try:
        for game in games:
            positions = await analyze_game(game, board_size, win_len, engine_eval)
            if not positions:
                continue
            corpus.ingest_analyzed_game(positions)
            analyzed_games += 1
            analyzed_positions += len(positions)
        corpus.save()
    finally:
        await engine_eval.stop()
    stats = corpus.stats()
    stats["analyzedGames"] = analyzed_games
    stats["analyzedPositions"] = analyzed_positions
    return stats
