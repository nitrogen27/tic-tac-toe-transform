"""Arena evaluation: model vs model and model vs algorithm.

Quick Arena (model vs model): 8-16 paired games, fast.
Strong Arena (model vs engine): 10-20 paired games, uses persistent C++ engine.

Paired matches ensure fairness: each pair = 2 games with swapped sides.
All games use the same policy + value + tactical decision path as the real bot,
so confirm/quick evaluation measures the deployed behavior instead of a stale
greedy-policy approximation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import torch
from gomoku_api.ws.predict_service import _loaded_model_decision

logger = logging.getLogger(__name__)

TRAIN_CALLBACK = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class ArenaResult:
    """Result of an arena evaluation."""
    wins_a: int
    wins_b: int
    draws: int
    total: int

    @property
    def winrate_a(self) -> float:
        """Winrate of player A: (wins + 0.5*draws) / total."""
        if self.total == 0:
            return 0.0
        return (self.wins_a + 0.5 * self.draws) / self.total

    @property
    def winrate_b(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.wins_b + 0.5 * self.draws) / self.total

    @property
    def decisive_winrate_a(self) -> float:
        if self.total == 0:
            return 0.0
        return self.wins_a / self.total

    @property
    def decisive_winrate_b(self) -> float:
        if self.total == 0:
            return 0.0
        return self.wins_b / self.total

    @property
    def draw_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.draws / self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "winsA": self.wins_a,
            "winsB": self.wins_b,
            "draws": self.draws,
            "total": self.total,
            "winrateA": round(self.winrate_a, 4),
            "winrateB": round(self.winrate_b, 4),
            "decisiveWinrateA": round(self.decisive_winrate_a, 4),
            "decisiveWinrateB": round(self.decisive_winrate_b, 4),
            "drawRate": round(self.draw_rate, 4),
        }


# ---------------------------------------------------------------------------
# Board helpers (reuse patterns from train_service_ws)
# ---------------------------------------------------------------------------


def _nxn_winner(board: list[int], n: int, win_len: int, last_move: int) -> int:
    if last_move < 0:
        return 0
    player = board[last_move]
    if player == 0:
        return 0
    row, col = divmod(last_move, n)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for step in range(1, win_len):
            nr, nc = row + dr * step, col + dc * step
            if 0 <= nr < n and 0 <= nc < n and board[nr * n + nc] == player:
                count += 1
            else:
                break
        for step in range(1, win_len):
            nr, nc = row - dr * step, col - dc * step
            if 0 <= nr < n and 0 <= nc < n and board[nr * n + nc] == player:
                count += 1
            else:
                break
        if count >= win_len:
            return player
    return 0


def _find_immediate_move(board: list[int], board_size: int, win_len: int, player: int) -> int | None:
    """Check if *player* can win in one move. Returns flat index or None."""
    for move, cell in enumerate(board):
        if cell != 0:
            continue
        board[move] = player
        winner = _nxn_winner(board, board_size, win_len, move)
        board[move] = 0
        if winner == player:
            return move
    return None


# ---------------------------------------------------------------------------
# Model inference for arena (greedy, no noise)
# ---------------------------------------------------------------------------


def _model_greedy_move(
    board: list[int],
    board_size: int,
    win_len: int,
    current: int,
    model: torch.nn.Module,
    device: torch.device,
) -> int:
    """Select a move using the same decision path as predict_service."""
    selection = _model_greedy_decision(
        board,
        board_size,
        win_len,
        current,
        model,
        device,
    )
    return int(selection.get("move", -1))


def _model_greedy_decision(
    board: list[int],
    board_size: int,
    win_len: int,
    current: int,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Return the full shared decision payload used by the real bot."""
    return _loaded_model_decision(
        board,
        current,
        board_size,
        win_len,
        model,
        decision_mode="hybrid",
    )


# ---------------------------------------------------------------------------
# Play a single game between two move-selection functions
# ---------------------------------------------------------------------------


def _play_arena_game(
    move_fn_p1: Callable[[list[int], int], int],
    move_fn_p2: Callable[[list[int], int], int],
    board_size: int,
    win_len: int,
) -> int:
    """Play one game. Returns winner (1 or 2) or 0 for draw."""
    board = [0] * (board_size * board_size)
    current = 1
    last_move = -1

    for _ in range(board_size * board_size):
        legal = [i for i, c in enumerate(board) if c == 0]
        if not legal:
            return 0

        fn = move_fn_p1 if current == 1 else move_fn_p2
        move = fn(board, current)
        if move < 0 or board[move] != 0:
            return 0  # invalid move = draw

        board[move] = current
        last_move = move
        winner = _nxn_winner(board, board_size, win_len, move)
        if winner != 0:
            return winner
        current = 2 if current == 1 else 1

    return 0


# ---------------------------------------------------------------------------
# Quick Arena: candidate vs champion
# ---------------------------------------------------------------------------


async def arena_match(
    candidate: torch.nn.Module,
    champion: torch.nn.Module | None,
    board_size: int,
    win_len: int,
    num_pairs: int = 8,
    device: torch.device = torch.device("cpu"),
    callback: TRAIN_CALLBACK | None = None,
    variant: str = "",
) -> ArenaResult:
    """Paired matches: candidate vs champion.

    Each pair = 2 games (candidate as P1 + candidate as P2).
    Returns ArenaResult where A=candidate, B=champion.
    If champion is None, returns empty result.
    """
    if champion is None:
        return ArenaResult(wins_a=0, wins_b=0, draws=0, total=0)

    wins_candidate = 0
    wins_champion = 0
    draws = 0
    total = num_pairs * 2
    started_at = time.monotonic()

    def make_move_fn(model: torch.nn.Module):
        def fn(board: list[int], current: int) -> int:
            return _model_greedy_move(board, board_size, win_len, current, model, device)
        return fn

    fn_candidate = make_move_fn(candidate)
    fn_champion = make_move_fn(champion)

    for pair in range(num_pairs):
        # Game 1: candidate = P1, champion = P2
        winner = _play_arena_game(fn_candidate, fn_champion, board_size, win_len)
        if winner == 1:
            wins_candidate += 1
        elif winner == 2:
            wins_champion += 1
        else:
            draws += 1

        # Game 2: champion = P1, candidate = P2
        winner = _play_arena_game(fn_champion, fn_candidate, board_size, win_len)
        if winner == 1:
            wins_champion += 1
        elif winner == 2:
            wins_candidate += 1
        else:
            draws += 1

        if callback is not None:
            games_done = (pair + 1) * 2
            elapsed = time.monotonic() - started_at
            await callback({
                "type": "train.progress",
                "payload": {
                    "phase": "arena",
                    "stage": "quick_eval",
                    "variant": variant,
                    "game": games_done,
                    "totalGames": total,
                    "arenaWins": wins_candidate,
                    "arenaLosses": wins_champion,
                    "arenaDraws": draws,
                    "winrateVsChampion": round((wins_candidate + 0.5 * draws) / max(games_done, 1), 4),
                    "decisiveWinRate": round(wins_candidate / max(games_done, 1), 4),
                    "drawRate": round(draws / max(games_done, 1), 4),
                    "elapsed": round(elapsed, 1),
                },
            })
            await asyncio.sleep(0)

    return ArenaResult(
        wins_a=wins_candidate,
        wins_b=wins_champion,
        draws=draws,
        total=total,
    )


# ---------------------------------------------------------------------------
# Strong Arena: candidate vs C++ engine
# ---------------------------------------------------------------------------


async def arena_vs_algorithm(
    candidate: torch.nn.Module,
    board_size: int,
    win_len: int,
    num_pairs: int = 10,
    device: torch.device = torch.device("cpu"),
    engine_move_fn: Callable[[list[int], int, int, int], Awaitable[int]] | None = None,
    callback: TRAIN_CALLBACK | None = None,
    variant: str = "",
) -> ArenaResult:
    """Paired matches: candidate vs algorithm/engine.

    *engine_move_fn(board, current, board_size, win_len) -> int* is an async
    callable that returns the engine's best move.  If None, returns empty result.
    """
    if engine_move_fn is None:
        return ArenaResult(wins_a=0, wins_b=0, draws=0, total=0)

    wins_candidate = 0
    wins_engine = 0
    draws = 0
    total = num_pairs * 2
    started_at = time.monotonic()

    def fn_candidate(board: list[int], current: int) -> int:
        return _model_greedy_move(board, board_size, win_len, current, candidate, device)

    for pair in range(num_pairs):
        for candidate_side in (1, 2):
            board = [0] * (board_size * board_size)
            current = 1
            winner = 0
            forfeit_by = 0  # track which side made an invalid/failed move

            for _ in range(board_size * board_size):
                legal = [i for i, c in enumerate(board) if c == 0]
                if not legal:
                    break

                if current == candidate_side:
                    move = fn_candidate(board, current)
                else:
                    move = await engine_move_fn(board, current, board_size, win_len)

                if move < 0 or board[move] != 0:
                    # Invalid move = forfeit by the side that produced it
                    forfeit_by = current
                    break

                board[move] = current
                winner = _nxn_winner(board, board_size, win_len, move)
                if winner != 0:
                    break
                current = 2 if current == 1 else 1

            if forfeit_by != 0:
                # The forfeiting side loses; the other side wins
                if forfeit_by == candidate_side:
                    wins_engine += 1
                else:
                    wins_candidate += 1
            elif winner == candidate_side:
                wins_candidate += 1
            elif winner != 0:
                wins_engine += 1
            else:
                draws += 1

        if callback is not None:
            games_done = (pair + 1) * 2
            elapsed = time.monotonic() - started_at
            await callback({
                "type": "train.progress",
                "payload": {
                    "phase": "arena",
                    "stage": "strong_eval",
                    "variant": variant,
                    "game": games_done,
                    "totalGames": total,
                    "arenaWins": wins_candidate,
                    "arenaLosses": wins_engine,
                    "arenaDraws": draws,
                    "winrateVsAlgorithm": round((wins_candidate + 0.5 * draws) / max(games_done, 1), 4),
                    "decisiveWinRate": round(wins_candidate / max(games_done, 1), 4),
                    "drawRate": round(draws / max(games_done, 1), 4),
                    "elapsed": round(elapsed, 1),
                },
            })
            await asyncio.sleep(0)

    return ArenaResult(
        wins_a=wins_candidate,
        wins_b=wins_engine,
        draws=draws,
        total=total,
    )
