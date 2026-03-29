"""Move prediction: algorithm (minimax/C++ engine) or model (PyTorch)."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from gomoku_api.config import settings

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


def _variant_model_hparams(variant: str, board_size: int, cfg: Any) -> tuple[int, int, int]:
    if board_size <= 3:
        return 32, 3, 64
    if board_size <= 5:
        return 96, 8, 160
    if board_size <= 9:
        return 96, 6, 160
    return cfg.res_filters, cfg.res_blocks, max(cfg.value_fc, 192)


def _get_model(variant: str):
    """Load or return cached PyTorch model."""
    if variant in _loaded_models:
        return _loaded_models[variant]

    model_path = SAVED_DIR / f"{variant}_resnet" / "model.pt"
    if not model_path.exists():
        return None

    try:
        from trainer_lab.config import ModelConfig
        from trainer_lab.models.resnet import PolicyValueResNet

        cfg = ModelConfig()
        if variant == "ttt3":
            board_size = 3
        elif variant == "ttt5":
            board_size = 5
        elif variant.startswith("gomoku"):
            board_size = int(variant.replace("gomoku", ""))
        else:
            board_size = 15
        res_filters, res_blocks, value_fc = _variant_model_hparams(variant, board_size, cfg)
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
            return None
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            model = _maybe_compile_model(model)
        _loaded_models[variant] = model
        logger.info("Loaded model for %s from %s", variant, model_path)
        return model
    except Exception as exc:
        logger.error("Failed to load model %s: %s", variant, exc)
        return None


def _model_predict(board: list[int], current: int, variant: str, board_size: int) -> dict:
    """Use PyTorch model for prediction."""
    model = _get_model(variant)
    if model is None:
        legal = [i for i, v in enumerate(board) if v == 0]
        return {
            "move": random.choice(legal) if legal else -1,
            "confidence": 0.0,
            "probs": [0.0] * len(board),
            "mode": "model",
            "isRandom": True,
            "fallback": True,
        }

    try:
        from trainer_lab.data.encoder import board_to_tensor

        # Build position dict for encoder
        board_2d = []
        for r in range(board_size):
            row = []
            for c in range(board_size):
                v = board[r * board_size + c]
                if v == current:
                    row.append(1)
                elif v == -current or (v != 0 and v != current):
                    row.append(2)
                else:
                    row.append(0)
            board_2d.append(row)

        pos_dict = {
            "board_size": board_size,
            "board": board_2d,
            "current_player": 1,
            "last_move": None,
        }
        tensor = board_to_tensor(pos_dict).unsqueeze(0)
        device = next(model.parameters()).device
        tensor = tensor.to(device)

        with torch.inference_mode():
            policy_logits, value = model(tensor)

        logits = policy_logits.squeeze(0).cpu()
        # Mask to valid cells only (board uses 16×16 padded grid)
        legal_flat = [i for i, v in enumerate(board) if v == 0]
        probs_raw = [0.0] * len(board)

        mask = torch.full_like(logits, float("-inf"))
        for idx in legal_flat:
            r, c = divmod(idx, board_size)
            mask[r * 16 + c] = 0.0
        probs_tensor = F.softmax(logits + mask, dim=0)

        for idx in legal_flat:
            r, c = divmod(idx, board_size)
            probs_raw[idx] = probs_tensor[r * 16 + c].item()

        best_move = max(legal_flat, key=lambda i: probs_raw[i]) if legal_flat else -1
        confidence = probs_raw[best_move] if best_move >= 0 else 0.0

        return {
            "move": best_move,
            "confidence": round(confidence, 4),
            "probs": [round(p, 6) for p in probs_raw],
            "mode": "model",
            "isRandom": False,
            "fallback": False,
            "value": round(value.item(), 4),
        }
    except Exception as exc:
        logger.error("Model predict error: %s", exc)
        legal = [i for i, v in enumerate(board) if v == 0]
        return {
            "move": random.choice(legal) if legal else -1,
            "confidence": 0.0,
            "probs": [0.0] * len(board),
            "mode": "model",
            "isRandom": True,
            "fallback": True,
        }


def clear_cached_model(variant: str) -> None:
    """Remove model from cache so it's reloaded next time."""
    _loaded_models.pop(variant, None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def predict(board: list[int], current: int, mode: str = "model", variant: str | None = None) -> dict:
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

    if variant == "ttt3":
        board_size, win_length = 3, 3
    elif variant == "ttt5":
        board_size, win_length = 5, 4
    else:
        board_size = int(variant.replace("gomoku", ""))
        win_length = 5

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
    else:
        return _model_predict(board, current, variant, board_size)
