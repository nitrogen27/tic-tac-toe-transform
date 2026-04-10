"""Evaluate a trained model against random or previous-checkpoint opponents."""

from __future__ import annotations

import logging
import random
from typing import Optional

import torch

from trainer_lab.config import ModelConfig
from trainer_lab.data.encoder import board_to_tensor
from trainer_lab.models.resnet import PolicyValueResNet
from trainer_lab.self_play.player import GameState, mcts_search

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


def _mcts_model_move(
    model: PolicyValueResNet,
    state: GameState,
    device: torch.device,
    *,
    simulations: int,
    deterministic: bool,
) -> int | None:
    """Choose a move from MCTS visit counts."""
    policy_flat, _ = mcts_search(
        state,
        model,
        device,
        num_simulations=simulations,
        root_noise=False,
    )
    legal = state.legal_moves()
    if not legal:
        return None
    if deterministic:
        return max(legal, key=lambda idx: policy_flat[idx])
    total = sum(policy_flat[idx] for idx in legal)
    if total <= 0:
        return random.choice(legal)
    weights = [policy_flat[idx] for idx in legal]
    return random.choices(legal, weights=weights, k=1)[0]


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


def evaluate_vs_previous_checkpoint(
    current_model: PolicyValueResNet,
    previous_model: PolicyValueResNet,
    *,
    num_games: int = 20,
    board_size: int = 15,
    win_length: int | None = None,
    max_moves: int | None = None,
    simulations: int = 400,
    deterministic: bool = True,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Head-to-head evaluation of current model against previous checkpoint.

    Colors alternate by game so the result is not biased by first-move advantage.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wl = win_length if win_length is not None else (4 if board_size <= 5 else 5)
    limit = max_moves if max_moves is not None else board_size * board_size

    current_model.eval()
    previous_model.eval()
    current_model.to(device)
    previous_model.to(device)

    wins = draws = losses = 0
    wins_as_first = wins_as_second = 0

    for game_idx in range(num_games):
        state = GameState(board_size, wl)
        current_is_first = (game_idx % 2 == 0)
        game_result = 0

        for _ in range(limit):
            if (state.current_player == 1 and current_is_first) or (state.current_player == 2 and not current_is_first):
                move = _mcts_model_move(
                    current_model,
                    state,
                    device,
                    simulations=simulations,
                    deterministic=deterministic,
                )
            else:
                move = _mcts_model_move(
                    previous_model,
                    state,
                    device,
                    simulations=simulations,
                    deterministic=deterministic,
                )
            if move is None:
                break
            state = state.apply_move(move)
            terminal, winner = state.is_terminal()
            if terminal:
                game_result = winner
                break

        if game_result == 0:
            draws += 1
            continue

        current_won = (game_result == 1 and current_is_first) or (game_result == 2 and not current_is_first)
        if current_won:
            wins += 1
            if current_is_first:
                wins_as_first += 1
            else:
                wins_as_second += 1
        else:
            losses += 1

    total = max(num_games, 1)
    return {
        "wins": wins / total,
        "losses": losses / total,
        "draws": draws / total,
        "wins_as_first": wins_as_first / max(1, (num_games + 1) // 2),
        "wins_as_second": wins_as_second / max(1, num_games // 2),
    }
