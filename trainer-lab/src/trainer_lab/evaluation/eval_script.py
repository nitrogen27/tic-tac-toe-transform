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
        for _ in range(max_moves):
            if pos["current_player"] == 1:
                move = _model_move(model, pos, device)
            else:
                move = _random_move(pos)

            if move is None:
                draws += 1
                break

            pos = _apply_move(pos, move)

            # (Simplified: no win detection — full implementation deferred)
        else:
            draws += 1

    total = max(num_games, 1)
    return {
        "wins": wins / total,
        "losses": losses / total,
        "draws": draws / total,
    }
