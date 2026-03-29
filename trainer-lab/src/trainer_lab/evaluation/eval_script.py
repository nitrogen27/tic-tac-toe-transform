"""Evaluate a trained model against random or engine-based opponents."""

from __future__ import annotations

import logging
import random
from typing import Optional

import torch

from trainer_lab.config import ModelConfig
from trainer_lab.data.encoder import board_to_tensor
from trainer_lab.models.resnet import PolicyValueResNet

logger = logging.getLogger(__name__)


def _make_empty_position(board_size: int = 15) -> dict:
    """Create an empty board position dictionary."""
    return {
        "board_size": board_size,
        "board": [[0] * board_size for _ in range(board_size)],
        "current_player": 1,
        "last_move": None,
    }


def _random_move(position: dict) -> tuple[int, int] | None:
    """Pick a random legal move."""
    bs = position["board_size"]
    legal = [
        (r, c)
        for r in range(bs)
        for c in range(bs)
        if position["board"][r][c] == 0
    ]
    return random.choice(legal) if legal else None


def _model_move(
    model: PolicyValueResNet,
    position: dict,
    device: torch.device,
) -> tuple[int, int] | None:
    """Select the move with highest policy logit among legal moves."""
    bs = position["board_size"]
    planes = board_to_tensor(position).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        policy_logits, value = model(planes)

    logits = policy_logits.squeeze(0).cpu()

    best_score = float("-inf")
    best_move = None
    for r in range(bs):
        for c in range(bs):
            if position["board"][r][c] == 0:
                idx = r * 16 + c
                if logits[idx].item() > best_score:
                    best_score = logits[idx].item()
                    best_move = (r, c)
    return best_move


def _apply_move(position: dict, move: tuple[int, int]) -> dict:
    """Return a new position with *move* applied (in-place mutation of board copy)."""
    import copy

    pos = copy.deepcopy(position)
    r, c = move
    pos["board"][r][c] = pos["current_player"]
    pos["current_player"] = 3 - pos["current_player"]
    pos["last_move"] = list(move)
    return pos


def _check_winner(position: dict, win_length: int = 5) -> int:
    """Check if the last move resulted in a win. Returns winner (1 or 2) or 0."""
    last = position.get("last_move")
    if last is None:
        return 0
    r, c = last[0], last[1]
    bs = position["board_size"]
    board = position["board"]
    player = board[r][c]
    if player == 0:
        return 0

    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for s in range(1, win_length):
            nr, nc = r + dr * s, c + dc * s
            if 0 <= nr < bs and 0 <= nc < bs and board[nr][nc] == player:
                count += 1
            else:
                break
        for s in range(1, win_length):
            nr, nc = r - dr * s, c - dc * s
            if 0 <= nr < bs and 0 <= nc < bs and board[nr][nc] == player:
                count += 1
            else:
                break
        if count >= win_length:
            return player
    return 0


def evaluate_vs_random(
    model: PolicyValueResNet,
    num_games: int = 20,
    board_size: int = 15,
    max_moves: int = 225,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Play *num_games* against a random opponent and report win/draw/loss rates.

    The model always plays as player 1 (first to move).

    Returns
    -------
    dict with keys ``wins``, ``losses``, ``draws`` (fractions).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wins = draws = losses = 0

    for g in range(num_games):
        pos = _make_empty_position(board_size)
        game_result = 0  # 0=draw, 1=player1 wins, 2=player2 wins

        for _ in range(max_moves):
            if pos["current_player"] == 1:
                move = _model_move(model, pos, device)
            else:
                move = _random_move(pos)

            if move is None:
                break

            pos = _apply_move(pos, move)
            winner = _check_winner(pos)
            if winner > 0:
                game_result = winner
                break

        if game_result == 1:
            wins += 1
        elif game_result == 2:
            losses += 1
        else:
            draws += 1

    total = max(num_games, 1)
    return {
        "wins": wins / total,
        "losses": losses / total,
        "draws": draws / total,
    }
