"""Move prediction: algorithm (minimax/C++ engine) or model (PyTorch)."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from gomoku_api.config import settings
from gomoku_api.ws.model_profiles import variant_model_hparams
from gomoku_api.ws.model_registry import ModelRegistry
from gomoku_api.ws.subprocess_utils import windows_hidden_subprocess_kwargs
from trainer_lab.data.encoder import board_to_tensor
from trainer_lab.specs import PADDED_BOARD_SIZE, resolve_variant_spec

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).resolve().parents[5] / "saved"

# ---------------------------------------------------------------------------
# Minimax for 3x3 (perfect play, instant)
# ---------------------------------------------------------------------------

def _ttt3_winner(board: list[int]) -> int:
    """Check 3x3 winner. Returns 1, -1, or 0."""
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    ]
    for a, b, c in lines:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def _minimax(board: list[int], current: int, alpha: float = -2, beta: float = 2) -> tuple[float, int]:
    """Minimax with alpha-beta for 3x3. Returns (score, best_move)."""
    w = _ttt3_winner(board)
    if w != 0:
        return (1.0 if w == current else -1.0), -1
    empty = [i for i in range(9) if board[i] == 0]
    if not empty:
        return 0.0, -1

    best_score = -2.0
    best_move = empty[0]
    for m in empty:
        board[m] = current
        opp = -current
        score, _ = _minimax(board, opp, -beta, -alpha)
        score = -score
        board[m] = 0
        if score > best_score:
            best_score = score
            best_move = m
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best_score, best_move


# ---------------------------------------------------------------------------
# NxN winner check (for 5x5 with win_length=4)
# ---------------------------------------------------------------------------

def _nxn_winner(board: list[int], n: int, win_len: int, last_move: int) -> int:
    """Check NxN winner from last_move. Returns player or 0."""
    if last_move < 0 or last_move >= len(board):
        return 0
    player = board[last_move]
    if player == 0:
        return 0
    r, c = divmod(last_move, n)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for s in range(1, win_len):
            nr, nc = r + dr * s, c + dc * s
            if 0 <= nr < n and 0 <= nc < n and board[nr * n + nc] == player:
                count += 1
            else:
                break
        for s in range(1, win_len):
            nr, nc = r - dr * s, c - dc * s
            if 0 <= nr < n and 0 <= nc < n and board[nr * n + nc] == player:
                count += 1
            else:
                break
        if count >= win_len:
            return player
    return 0


def _policy_cell_index(flat_index: int, board_size: int) -> int:
    row, col = divmod(flat_index, board_size)
    return row * PADDED_BOARD_SIZE + col


def _flat_to_board2d(board: list[int], board_size: int, current: int) -> list[list[int]]:
    board_2d: list[list[int]] = []
    for r in range(board_size):
        row: list[int] = []
        for c in range(board_size):
            v = board[r * board_size + c]
            if v == current:
                row.append(1)
            elif v == -current or (v != 0 and v != current):
                row.append(2)
            else:
                row.append(0)
        board_2d.append(row)
    return board_2d


def _find_immediate_move(board: list[int], board_size: int, win_len: int, player: int) -> int | None:
    for move, cell in enumerate(board):
        if cell != 0:
            continue
        board[move] = player
        winner = _nxn_winner(board, board_size, win_len, move)
        board[move] = 0
        if winner == player:
            return move
    return None


def _list_immediate_wins(board: list[int], board_size: int, win_len: int, player: int) -> list[int]:
    wins: list[int] = []
    for move, cell in enumerate(board):
        if cell != 0:
            continue
        board[move] = player
        winner = _nxn_winner(board, board_size, win_len, move)
        board[move] = 0
        if winner == player:
            wins.append(move)
    return wins


def _count_double_threat_responses(board: list[int], board_size: int, win_len: int, player: int) -> int:
    """Count replies that create at least two distinct immediate wins for *player*."""
    count = 0
    for move, cell in enumerate(board):
        if cell != 0:
            continue
        board[move] = player
        if _nxn_winner(board, board_size, win_len, move) != player:
            immediate_wins = _list_immediate_wins(board, board_size, win_len, player)
            if len(immediate_wins) >= 2:
                count += 1
        board[move] = 0
    return count


def _count_winning_pressure_moves(board: list[int], board_size: int, win_len: int, player: int) -> int:
    """Count follow-up moves that either win immediately or create a forcing fork."""
    count = 0
    for move, cell in enumerate(board):
        if cell != 0:
            continue
        board[move] = player
        if _nxn_winner(board, board_size, win_len, move) == player:
            count += 1
        else:
            immediate_wins = _list_immediate_wins(board, board_size, win_len, player)
            if len(immediate_wins) >= 2:
                count += 1
        board[move] = 0
    return count


def _board_winner(board: list[int], board_size: int, win_len: int) -> int:
    """Check whether either side has already won in the current position."""
    for idx, cell in enumerate(board):
        if cell != 0 and _nxn_winner(board, board_size, win_len, idx) == cell:
            return cell
    return 0


def _uniform_legal_probs(board: list[int]) -> list[float]:
    legal = [idx for idx, cell in enumerate(board) if cell == 0]
    if not legal:
        return [0.0] * len(board)
    p = 1.0 / len(legal)
    return [p if cell == 0 else 0.0 for cell in board]


def _pure_fallback_move(board: list[int]) -> tuple[int, list[float]]:
    probs_raw = _uniform_legal_probs(board)
    legal = [idx for idx, cell in enumerate(board) if cell == 0]
    if not legal:
        return -1, probs_raw
    # Honest pure fallback: no tactical help if there is no checkpoint.
    return legal[0], probs_raw


def _evaluate_afterstate_values(
    model: Any,
    board: list[int],
    current: int,
    board_size: int,
    candidate_moves: list[int],
) -> dict[int, float]:
    if not candidate_moves:
        return {}

    device = next(model.parameters()).device
    opponent = 2 if current == 1 else 1
    tensors: list[torch.Tensor] = []
    for move in candidate_moves:
        after = list(board)
        after[move] = current
        pos_dict = {
            "board_size": board_size,
            "board": _flat_to_board2d(after, board_size, opponent),
            "current_player": 1,
            "last_move": list(divmod(move, board_size)),
        }
        tensors.append(board_to_tensor(pos_dict))

    batch = torch.stack(tensors, dim=0).to(device)
    if device.type == "cuda":
        batch = batch.contiguous(memory_format=torch.channels_last)

    model.eval()
    with torch.inference_mode():
        _, values = model(batch)

    values_cpu = values.view(-1).detach().cpu().tolist()
    # Values are from the next side-to-move perspective (opponent), so negate.
    return {
        move: float(max(-1.0, min(1.0, -value)))
        for move, value in zip(candidate_moves, values_cpu)
    }


def _select_policy_value_move(
    board: list[int],
    probs_raw: list[float],
    value_scores: dict[int, float] | None = None,
) -> tuple[int, dict[str, Any]]:
    legal = [idx for idx, cell in enumerate(board) if cell == 0]
    if not legal:
        return -1, {
            "tacticalReason": "no_legal_moves",
            "tacticalOverride": False,
            "valueGuided": False,
            "unsafeMovesFiltered": 0,
            "searchBacked": False,
            "searchMode": "none",
            "searchScore": 0.0,
            "searchDepth": 0,
        }

    value_scores = value_scores or {}
    model_best = max(legal, key=lambda move: probs_raw[move])
    best_prob = float(probs_raw[model_best])
    sorted_by_policy = sorted(
        legal,
        key=lambda move: (
            probs_raw[move],
            float(value_scores.get(move, 0.0)),
            -move,
        ),
        reverse=True,
    )

    if best_prob > 0.0:
        abs_margin = 0.03 if len(legal) <= 25 else 0.02
        rel_floor = best_prob * 0.85
        min_prob = max(best_prob - abs_margin, rel_floor)
        candidate_pool = [move for move in legal if probs_raw[move] >= min_prob]
    else:
        candidate_pool = []
    if not candidate_pool:
        candidate_pool = sorted_by_policy[: min(3, len(sorted_by_policy))]

    best_move = max(
        candidate_pool,
        key=lambda move: (
            float(value_scores.get(move, 0.0)),
            probs_raw[move],
            -move,
        ),
    )
    used_value = best_move != model_best and len(candidate_pool) > 1
    reason = "policy_value" if used_value and best_move != model_best else "model_policy"
    return best_move, {
        "tacticalReason": reason,
        "tacticalOverride": False,
        "valueGuided": used_value,
        "afterstateValue": round(float(value_scores.get(best_move, 0.0)), 4),
        "unsafeMovesFiltered": 0,
        "searchBacked": False,
        "searchMode": "none",
        "searchScore": 0.0,
        "searchDepth": 0,
    }


def _evaluate_threat_move(
    board: list[int],
    current: int,
    board_size: int,
    win_len: int,
    move: int,
    *,
    opponent_threats_before: list[int],
    probs_raw: list[float] | None = None,
    value_scores: dict[int, float] | None = None,
) -> dict[str, Any]:
    opponent = 2 if current == 1 else 1
    probs_raw = probs_raw or [0.0] * len(board)
    value_scores = value_scores or {}

    board[move] = current
    winner = _nxn_winner(board, board_size, win_len, move)
    opp_immediate_wins = _list_immediate_wins(board, board_size, win_len, opponent)
    self_immediate_next = _list_immediate_wins(board, board_size, win_len, current)
    opponent_fork_responses = 0
    if not opp_immediate_wins:
        opponent_fork_responses = _count_double_threat_responses(board, board_size, win_len, opponent)
    self_winning_pressure = _count_winning_pressure_moves(board, board_size, win_len, current)
    board[move] = 0

    safe_now = len(opp_immediate_wins) == 0
    safe_vs_fork = opponent_fork_responses == 0
    creates_fork = len(self_immediate_next) >= 2
    blocks_immediate = bool(opponent_threats_before) and safe_now

    return {
        "move": move,
        "isImmediateWin": winner == current,
        "safeNow": safe_now,
        "safeVsFork": safe_vs_fork,
        "blocksImmediate": blocks_immediate,
        "createsFork": creates_fork,
        "opponentImmediateWins": len(opp_immediate_wins),
        "opponentForkResponses": opponent_fork_responses,
        "selfImmediateWinsNext": len(self_immediate_next),
        "selfWinningPressure": self_winning_pressure,
        "modelProb": probs_raw[move],
        "afterstateValue": float(value_scores.get(move, 0.0)),
    }


def _threat_candidate_sort_key(evaluation: dict[str, Any]) -> tuple[Any, ...]:
    winning_pressure = int(evaluation.get("selfWinningPressure", 0))
    return (
        1 if evaluation["isImmediateWin"] else 0,
        1 if evaluation["blocksImmediate"] else 0,
        1 if evaluation["safeNow"] else 0,
        1 if evaluation["safeVsFork"] else 0,
        winning_pressure,
        1 if evaluation["createsFork"] else 0,
        evaluation["selfImmediateWinsNext"],
        -evaluation["opponentImmediateWins"],
        -evaluation["opponentForkResponses"],
        evaluation["afterstateValue"],
        evaluation["modelProb"],
        -evaluation["move"],
    )


def _candidate_move_evaluations(
    board: list[int],
    current: int,
    board_size: int,
    win_len: int,
    *,
    probs_raw: list[float] | None = None,
    value_scores: dict[int, float] | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    legal = [idx for idx, cell in enumerate(board) if cell == 0]
    opponent = 2 if current == 1 else 1
    opponent_threats_before = _list_immediate_wins(board, board_size, win_len, opponent)
    evaluations = [
        _evaluate_threat_move(
            board,
            current,
            board_size,
            win_len,
            move,
            opponent_threats_before=opponent_threats_before,
            probs_raw=probs_raw,
            value_scores=value_scores,
        )
        for move in legal
    ]
    evaluations.sort(key=_threat_candidate_sort_key, reverse=True)
    return evaluations, opponent_threats_before


def _select_exact_endgame_move(
    board: list[int],
    current: int,
    board_size: int,
    win_len: int,
    probs_raw: list[float],
    value_scores: dict[int, float] | None = None,
    *,
    max_empties: int = 10,
) -> tuple[int, dict[str, Any]] | None:
    legal = [idx for idx, cell in enumerate(board) if cell == 0]
    if not legal or len(legal) > max_empties:
        return None

    root_evaluations, opponent_threats_before = _candidate_move_evaluations(
        board,
        current,
        board_size,
        win_len,
        probs_raw=probs_raw,
        value_scores=value_scores,
    )
    if not root_evaluations:
        return None

    @lru_cache(maxsize=20000)
    def solve(state: tuple[int, ...], side_to_move: int) -> int:
        local_board = list(state)
        winner = _board_winner(local_board, board_size, win_len)
        if winner != 0:
            return 1 if winner == current else -1

        local_legal = [idx for idx, cell in enumerate(local_board) if cell == 0]
        if not local_legal:
            return 0

        evaluations, _ = _candidate_move_evaluations(local_board, side_to_move, board_size, win_len)
        if side_to_move == current:
            best = -2
            for evaluation in evaluations:
                move = int(evaluation["move"])
                local_board[move] = side_to_move
                score = solve(tuple(local_board), 2 if side_to_move == 1 else 1)
                local_board[move] = 0
                if score > best:
                    best = score
                if best == 1:
                    break
            return best

        best = 2
        for evaluation in evaluations:
            move = int(evaluation["move"])
            local_board[move] = side_to_move
            score = solve(tuple(local_board), 2 if side_to_move == 1 else 1)
            local_board[move] = 0
            if score < best:
                best = score
            if best == -1:
                break
        return best

    best_move = -1
    best_eval: dict[str, Any] | None = None
    best_score = -2
    best_key: tuple[Any, ...] | None = None
    model_best = max(legal, key=lambda idx: probs_raw[idx])

    for evaluation in root_evaluations:
        move = int(evaluation["move"])
        board[move] = current
        score = solve(tuple(board), 2 if current == 1 else 1)
        board[move] = 0
        key = (score, *_threat_candidate_sort_key(evaluation))
        if best_key is None or key > best_key:
            best_key = key
            best_score = score
            best_move = move
            best_eval = evaluation

    if best_eval is None or best_move < 0:
        return None

    if best_score > 0:
        reason = "search_exact_win"
    elif best_score == 0 and best_eval["blocksImmediate"]:
        reason = "search_exact_hold"
    elif best_score == 0:
        reason = "search_exact_draw"
    else:
        reason = "search_exact_survival"

    unsafe_filtered = sum(
        1 for evaluation in root_evaluations if not evaluation["safeNow"] or not evaluation["safeVsFork"]
    )

    return best_move, {
        "tacticalReason": reason,
        "tacticalOverride": best_move != model_best or reason != "model_policy",
        "unsafeMovesFiltered": unsafe_filtered,
        "opponentThreatsBefore": len(opponent_threats_before),
        "forcingThreatsAfterMove": best_eval["selfImmediateWinsNext"],
        "opponentThreatsAfterMove": best_eval["opponentImmediateWins"],
        "opponentForkResponses": best_eval["opponentForkResponses"],
        "valueGuided": False,
        "afterstateValue": round(best_eval["afterstateValue"], 4),
        "searchBacked": True,
        "searchMode": "exact_endgame",
        "searchScore": float(best_score),
        "searchDepth": len(legal),
    }


def _select_threat_aware_move(
    board: list[int],
    current: int,
    board_size: int,
    win_len: int,
    probs_raw: list[float],
    value_scores: dict[int, float] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Choose a move using model probabilities plus hard tactical safety rules.

    The priority order is:
    1. Immediate win now.
    2. Forced immediate block against opponent's win.
    3. Reject moves that allow opponent an immediate win.
    4. Prefer safe moves that preserve the strongest winning pressure.
    5. Prefer moves that create multiple immediate wins next turn (forcing fork).
    6. Reject moves that allow opponent a forcing fork on the next reply.
    7. Use model policy/value only as a final tie-break among tactically similar moves.
    """
    legal = [idx for idx, cell in enumerate(board) if cell == 0]
    if not legal:
        return -1, {
            "tacticalReason": "no_legal_moves",
            "tacticalOverride": False,
            "unsafeMovesFiltered": 0,
            "opponentThreatsBefore": 0,
            "forcingThreatsAfterMove": 0,
            "searchBacked": False,
            "searchMode": "none",
            "searchScore": 0.0,
            "searchDepth": 0,
        }

    opponent = 2 if current == 1 else 1
    model_best = max(legal, key=lambda idx: probs_raw[idx]) if legal else -1

    immediate_win = _find_immediate_move(board, board_size, win_len, current)
    if immediate_win is not None:
        return immediate_win, {
            "tacticalReason": "immediate_win",
            "tacticalOverride": immediate_win != model_best,
            "unsafeMovesFiltered": 0,
            "opponentThreatsBefore": 0,
            "forcingThreatsAfterMove": 0,
            "searchBacked": False,
            "searchMode": "none",
            "searchScore": 0.0,
            "searchDepth": 0,
        }

    if board_size <= 5:
        exact_endgame = _select_exact_endgame_move(
            board,
            current,
            board_size,
            win_len,
            probs_raw,
            value_scores=value_scores,
        )
        if exact_endgame is not None:
            return exact_endgame

    evaluations, opponent_threats_before = _candidate_move_evaluations(
        board,
        current,
        board_size,
        win_len,
        probs_raw=probs_raw,
        value_scores=value_scores,
    )
    best_move = -1
    best_key: tuple[Any, ...] | None = None
    best_eval: dict[str, Any] | None = None
    unsafe_filtered = 0

    for evaluation in evaluations:
        move = int(evaluation["move"])
        if not evaluation["safeNow"] or not evaluation["safeVsFork"]:
            unsafe_filtered += 1
        key = _threat_candidate_sort_key(evaluation)
        if best_key is None or key > best_key:
            best_key = key
            best_move = move
            best_eval = evaluation

    if best_eval is None:
        return model_best, {
            "tacticalReason": "model_policy",
            "tacticalOverride": False,
            "unsafeMovesFiltered": 0,
            "opponentThreatsBefore": len(opponent_threats_before),
            "forcingThreatsAfterMove": 0,
            "searchBacked": False,
            "searchMode": "none",
            "searchScore": 0.0,
            "searchDepth": 0,
        }

    if best_eval["blocksImmediate"]:
        reason = "block_immediate"
    elif best_eval["createsFork"]:
        reason = "create_forcing_threat"
    elif (
        best_eval["safeNow"]
        and best_eval["safeVsFork"]
        and int(best_eval.get("selfWinningPressure", 0)) > 0
        and best_move != model_best
    ):
        reason = "press_winning_advantage"
    elif best_eval["safeNow"] and best_eval["safeVsFork"] and abs(best_eval["afterstateValue"]) > 1e-6 and best_move != model_best:
        reason = "policy_value"
    elif best_eval["safeNow"] and best_eval["safeVsFork"] and best_move != model_best:
        reason = "reject_unsafe_model_move"
    elif not best_eval["safeNow"] or not best_eval["safeVsFork"]:
        reason = "least_bad_move"
    else:
        reason = "model_policy"

    return best_move, {
        "tacticalReason": reason,
        "tacticalOverride": best_move != model_best or reason != "model_policy",
        "unsafeMovesFiltered": unsafe_filtered,
        "opponentThreatsBefore": len(opponent_threats_before),
        "forcingThreatsAfterMove": best_eval["selfImmediateWinsNext"],
        "opponentThreatsAfterMove": best_eval["opponentImmediateWins"],
        "opponentForkResponses": best_eval["opponentForkResponses"],
        "winningPressure": int(best_eval.get("selfWinningPressure", 0)),
        "valueGuided": reason == "policy_value",
        "afterstateValue": round(best_eval["afterstateValue"], 4),
        "searchBacked": False,
        "searchMode": "none",
        "searchScore": 0.0,
        "searchDepth": 0,
    }


def _loaded_model_decision(
    board: list[int],
    current: int,
    board_size: int,
    win_length: int,
    model: Any,
    *,
    decision_mode: str = "hybrid",
) -> dict[str, Any]:
    """Run the shared loaded-model decision path used by predict and arena.

    This keeps arena/confirm evaluation aligned with the real bot behavior:
    the same board encoding, policy softmax, afterstate value scoring, and
    hybrid/pure move selection logic are used everywhere.
    """
    from trainer_lab.data.encoder import board_to_tensor

    resolved_mode = "pure" if str(decision_mode).strip().lower() == "pure" else "hybrid"
    pos_dict = {
        "board_size": board_size,
        "board": _flat_to_board2d(board, board_size, current),
        "current_player": 1,
        "last_move": None,
    }
    tensor = board_to_tensor(pos_dict).unsqueeze(0)
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    if device.type == "cuda":
        tensor = tensor.contiguous(memory_format=torch.channels_last)

    model.eval()
    with torch.inference_mode():
        policy_logits, value = model(tensor)

    logits = policy_logits.squeeze(0).cpu()
    legal_flat = [i for i, v in enumerate(board) if v == 0]
    probs_raw = [0.0] * len(board)

    mask = torch.full_like(logits, float("-inf"))
    for idx in legal_flat:
        mask[_policy_cell_index(idx, board_size)] = 0.0
    probs_tensor = F.softmax(logits + mask, dim=0)

    for idx in legal_flat:
        probs_raw[idx] = probs_tensor[_policy_cell_index(idx, board_size)].item()

    value_scores = _evaluate_afterstate_values(model, board, current, board_size, legal_flat)
    if resolved_mode == "pure":
        best_move, tactical_meta = _select_policy_value_move(board, probs_raw, value_scores=value_scores)
    else:
        best_move, tactical_meta = _select_threat_aware_move(
            board,
            current,
            board_size,
            win_length,
            probs_raw,
            value_scores=value_scores,
        )
    confidence = probs_raw[best_move] if best_move >= 0 else 0.0

    return {
        "move": best_move,
        "confidence": round(confidence, 4),
        "probs": [round(p, 6) for p in probs_raw],
        "decisionMode": resolved_mode,
        "value": round(value.item(), 4),
        "afterstateValue": round(value_scores.get(best_move, 0.0), 4) if best_move >= 0 else 0.0,
        **tactical_meta,
    }


# ---------------------------------------------------------------------------
# C++ Engine subprocess (for algorithm mode on 5x5+)
# ---------------------------------------------------------------------------

async def _engine_predict(board: list[int], current: int, board_size: int, win_length: int) -> dict:
    """Call C++ engine for algorithm-mode prediction."""
    binary = settings.engine_binary
    cells = []
    for v in board:
        if v == 1:
            cells.append(1)
        elif v == -1 or v == 2:
            cells.append(-1)
        else:
            cells.append(0)

    payload = {
        "command": "best-move",
        "position": {
            "boardSize": board_size,
            "winLength": win_length,
            "cells": cells,
            "sideToMove": 1 if current == 1 else -1,
            "moveCount": sum(1 for c in cells if c != 0),
            "lastMove": -1,
            "moveHistory": [],
        },
    }
    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **windows_hidden_subprocess_kwargs(),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(json.dumps(payload).encode()), timeout=10)
        result = json.loads(stdout.decode())
        return {
            "move": result.get("bestMove", -1),
            "confidence": 0.0,
            "probs": [],
            "mode": "algorithm",
            "isRandom": False,
            "fallback": False,
        }
    except Exception as exc:
        logger.error("Engine predict failed: %s", exc)
        legal = [i for i, v in enumerate(board) if v == 0]
        return {
            "move": random.choice(legal) if legal else -1,
            "confidence": 0.0,
            "probs": [],
            "mode": "algorithm",
            "isRandom": True,
            "fallback": True,
        }


# ---------------------------------------------------------------------------
# PyTorch model inference
# ---------------------------------------------------------------------------

_loaded_models: dict[str, Any] = {}
_loaded_model_sources: dict[str, str] = {}


def _current_model_source(variant: str) -> str:
    return _loaded_model_sources.get(variant, "none")


def _maybe_compile_model(model: Any) -> Any:
    if not torch.cuda.is_available() or not hasattr(torch, "compile"):
        return model
    try:
        import triton  # noqa: F401
    except ImportError:
        logger.info("Triton not installed — predict model stays in eager mode")
        return model
    try:
        return torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception as exc:
        logger.warning("Predict model stays in eager mode because torch.compile failed: %s", exc)
        return model


def _get_model(variant: str):
    """Load or return cached PyTorch model."""
    if variant in _loaded_models:
        return _loaded_models[variant]

    try:
        variant_spec = resolve_variant_spec(variant)
    except ValueError:
        _loaded_model_sources[variant] = "none"
        return None

    registry = ModelRegistry(variant)
    # Serve only the promoted checkpoint. ``model.pt`` is just a legacy alias
    # for the same promoted weights and remains safe to use as compatibility
    # fallback, but active candidates must not leak into runtime serving.
    model_path, model_source = registry.resolve_serving_checkpoint(expected_spec=variant_spec)
    if model_path is None:
        _loaded_model_sources[variant] = "none"
        return None

    try:
        from trainer_lab.config import ModelConfig
        from trainer_lab.models.resnet import PolicyValueResNet

        cfg = ModelConfig()
        manifest = ModelRegistry(variant).read_manifest()
        model_profile, (res_filters, res_blocks, value_fc) = variant_model_hparams(
            variant,
            variant_spec.board_size,
            cfg,
            manifest=manifest,
            spec=variant_spec,
        )
        model = PolicyValueResNet(
            in_channels=cfg.in_channels,
            res_filters=res_filters,
            res_blocks=res_blocks,
            policy_filters=cfg.policy_filters,
            value_fc=value_fc,
            board_max=cfg.board_max,
        )
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        except Exception as exc:
            logger.warning("Model checkpoint %s is incompatible with current architecture: %s", model_path, exc)
            _loaded_model_sources[variant] = "none"
            return None
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            model = _maybe_compile_model(model)
        _loaded_models[variant] = model
        _loaded_model_sources[variant] = model_source
        logger.info("Loaded serving model for %s from %s (source=%s, profile=%s)", variant, model_path, model_source, model_profile)
        return model
    except Exception as exc:
        logger.error("Failed to load model %s: %s", variant, exc)
        _loaded_model_sources[variant] = "none"
        return None


def _model_predict(
    board: list[int],
    current: int,
    variant: str,
    board_size: int,
    *,
    decision_mode: str = "hybrid",
) -> dict:
    """Use PyTorch model for prediction."""
    win_length = 4 if board_size == 5 else 5
    if variant == "ttt3":
        win_length = 3
    resolved_mode = "pure" if str(decision_mode).strip().lower() == "pure" else "hybrid"
    model = _get_model(variant)
    if model is None:
        if resolved_mode == "pure":
            best_move, probs_raw = _pure_fallback_move(board)
            tactical_meta = {
                "tacticalReason": "no_model_checkpoint",
                "tacticalOverride": False,
                "valueGuided": False,
                "unsafeMovesFiltered": 0,
                "afterstateValue": 0.0,
                "searchBacked": False,
                "searchMode": "none",
                "searchScore": 0.0,
                "searchDepth": 0,
            }
        else:
            probs_raw = _uniform_legal_probs(board)
            best_move, tactical_meta = _select_threat_aware_move(board, current, board_size, win_length, probs_raw)
        return {
            "move": best_move,
            "confidence": round(probs_raw[best_move], 4) if best_move >= 0 else 0.0,
            "probs": [round(p, 6) for p in probs_raw],
            "mode": "model",
            "isRandom": resolved_mode == "pure",
            "fallback": True,
            "decisionMode": resolved_mode,
            "modelSource": "none",
            **tactical_meta,
        }

    try:
        selection = _loaded_model_decision(
            board,
            current,
            board_size,
            win_length,
            model,
            decision_mode=resolved_mode,
        )

        return {
            "mode": "model",
            "isRandom": False,
            "fallback": False,
            "modelSource": _current_model_source(variant),
            **selection,
        }
    except Exception as exc:
        logger.error("Model predict error: %s", exc)
        probs_raw = _uniform_legal_probs(board)
        best_move, tactical_meta = _select_threat_aware_move(board, current, board_size, win_length, probs_raw)
        return {
            "move": best_move,
            "confidence": round(probs_raw[best_move], 4) if best_move >= 0 else 0.0,
            "probs": [round(p, 6) for p in probs_raw],
            "mode": "model",
            "isRandom": False,
            "fallback": True,
            "decisionMode": resolved_mode,
            "modelSource": _current_model_source(variant),
            **tactical_meta,
        }


def clear_cached_model(variant: str) -> None:
    """Remove model from cache so it's reloaded next time."""
    _loaded_models.pop(variant, None)
    _loaded_model_sources.pop(variant, None)


# ---------------------------------------------------------------------------
# MCTS-enhanced prediction
# ---------------------------------------------------------------------------


def _mcts_predict(
    board: list[int],
    current: int,
    variant: str,
    board_size: int,
    *,
    num_simulations: int = 50,
) -> dict:
    """Use MCTS + neural network for stronger move selection."""
    import asyncio
    from trainer_lab.self_play.player import GameState, mcts_search

    win_length = 4 if board_size == 5 else (3 if board_size == 3 else 5)
    model = _get_model(variant)
    if model is None:
        # Fallback to regular predict if no model
        return _model_predict(board, current, variant, board_size)

    device = next(model.parameters()).device

    # Build GameState from flat board
    state = GameState(board_size, win_length)
    for r in range(board_size):
        for c in range(board_size):
            v = board[r * board_size + c]
            if v == current:
                state.board[r][c] = 1
            elif v != 0:
                state.board[r][c] = 2
    state.current_player = 1  # encoder expects current=1
    state.move_count = sum(1 for c in board if c != 0)

    # Run MCTS
    model.eval()
    policy_flat, root_value = mcts_search(
        state, model, device,
        num_simulations=num_simulations,
        c_puct=1.5,
        dirichlet_alpha=0.03,
        dirichlet_weight=0.0,  # no noise at inference
    )

    # Convert policy to probs_raw format
    probs_raw = [0.0] * (board_size * board_size)
    for idx, prob in enumerate(policy_flat):
        probs_raw[idx] = prob

    legal = [i for i, v in enumerate(board) if v == 0]
    value_scores = _evaluate_afterstate_values(model, board, current, board_size, legal)
    best_move, tactical_meta = _select_threat_aware_move(
        board,
        current,
        board_size,
        win_length,
        probs_raw,
        value_scores=value_scores,
    )
    confidence = probs_raw[best_move] if best_move >= 0 else 0.0
    search_mode = tactical_meta.get("searchMode") or "none"
    if search_mode == "none":
        search_mode = "mcts"

    return {
        "move": best_move,
        "confidence": round(confidence, 4),
        "probs": [round(p, 6) for p in probs_raw],
        "mode": "model",
        "isRandom": False,
        "fallback": False,
        "modelSource": _current_model_source(variant),
        "value": round(root_value, 4),
        "searchBacked": True,
        "searchMode": search_mode,
        "searchDepth": num_simulations,
        "mctsConfidence": round(confidence, 4),
        **tactical_meta,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def predict(
    board: list[int],
    current: int,
    mode: str = "model",
    variant: str | None = None,
    model_decision_mode: str = "hybrid",
) -> dict:
    """Predict next move. Dispatches to minimax, C++ engine, or PyTorch model."""
    n = len(board)

    # Auto-detect variant
    if variant is None or variant == "":
        if n == 9:
            variant = "ttt3"
        elif n == 25:
            variant = "ttt5"
        else:
            board_size = int(math.sqrt(n))
            variant = f"gomoku{board_size}"

    variant_spec = resolve_variant_spec(variant)
    board_size, win_length = variant_spec.board_size, variant_spec.win_length

    if mode == "algorithm":
        if variant == "ttt3":
            _, best = _minimax(list(board), current)
            return {
                "move": best,
                "confidence": 1.0,
                "probs": [],
                "mode": "algorithm",
                "isRandom": False,
                "fallback": False,
            }
        else:
            return await _engine_predict(board, current, board_size, win_length)
    elif mode == "mcts" or model_decision_mode == "mcts":
        return _mcts_predict(board, current, variant, board_size, num_simulations=50)
    else:
        return _model_predict(board, current, variant, board_size, decision_mode=model_decision_mode)
