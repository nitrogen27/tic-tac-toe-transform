"""Training service for WebSocket streaming — uses PyTorch via trainer-lab."""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).resolve().parents[5] / "saved"


def _ensure_saved_dir(variant: str) -> Path:
    d = SAVED_DIR / f"{variant}_resnet"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Data generation for TTT3 (3×3, win_length=3)
# ---------------------------------------------------------------------------

def _ttt3_winner(board: list[int]) -> int:
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    ]
    for a, b, c in lines:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def _minimax_value(board: list[int], current: int) -> float:
    w = _ttt3_winner(board)
    if w == current:
        return 1.0
    if w == -current:
        return -1.0
    empty = [i for i in range(9) if board[i] == 0]
    if not empty:
        return 0.0
    best = -2.0
    for m in empty:
        board[m] = current
        val = -_minimax_value(board, -current)
        board[m] = 0
        best = max(best, val)
    return best


def _minimax_policy(board: list[int], current: int) -> list[float]:
    """Get minimax-optimal policy for 3×3. Returns 9 probabilities."""
    empty = [i for i in range(9) if board[i] == 0]
    if not empty:
        return [0.0] * 9
    scores = {}
    for m in empty:
        board[m] = current
        scores[m] = -_minimax_value(board, -current)
        board[m] = 0
    best_val = max(scores.values())
    best_moves = [m for m, v in scores.items() if v == best_val]
    policy = [0.0] * 9
    for m in best_moves:
        policy[m] = 1.0 / len(best_moves)
    return policy


def _generate_ttt3_positions(count: int = 2000) -> list[dict]:
    """Generate training positions for 3×3 via random play + minimax labels."""
    positions = []
    for _ in range(count):
        board = [0] * 9
        current = 1
        moves_played = random.randint(0, 6)
        available = list(range(9))
        random.shuffle(available)
        for i in range(min(moves_played, len(available))):
            board[available[i]] = current
            current = -current
        if _ttt3_winner(board) != 0:
            continue
        empty = [i for i in range(9) if board[i] == 0]
        if not empty:
            continue

        policy = _minimax_policy(list(board), current)
        value = _minimax_value(list(board), current)

        # Build 2D board for encoder (3×3)
        board_2d = []
        for r in range(3):
            row = []
            for c in range(3):
                v = board[r * 3 + c]
                if v == current:
                    row.append(1)
                elif v == -current:
                    row.append(2)
                else:
                    row.append(0)
            board_2d.append(row)

        # Policy: 9 cells → 256 padded (16×16 grid)
        policy_256 = [0.0] * 256
        for idx in range(9):
            r, c = divmod(idx, 3)
            policy_256[r * 16 + c] = policy[idx]

        positions.append({
            "board_size": 3,
            "board": board_2d,
            "current_player": 1,
            "last_move": None,
            "policy": policy_256,
            "value": value,
        })
    return positions


# ---------------------------------------------------------------------------
# Data generation for TTT5 (5×5, win_length=4) — random self-play
# ---------------------------------------------------------------------------

def _nxn_winner(board: list[int], n: int, win_len: int, last_move: int) -> int:
    if last_move < 0:
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


def _generate_ttt5_positions(count: int = 2000) -> list[dict]:
    """Generate training positions for 5×5 via random self-play with outcome labels."""
    positions = []
    games_played = 0
    N, WIN_LEN = 5, 4

    while len(positions) < count:
        board = [0] * 25
        current = 1
        history = []
        winner = 0

        for _ in range(25):
            empty = [i for i in range(25) if board[i] == 0]
            if not empty:
                break
            move = random.choice(empty)

            # Uniform policy for random play
            policy = [0.0] * 25
            for e in empty:
                policy[e] = 1.0 / len(empty)

            # Store position
            board_2d = []
            for r in range(N):
                row = []
                for c in range(N):
                    v = board[r * N + c]
                    if v == current:
                        row.append(1)
                    elif v == -current:
                        row.append(2)
                    else:
                        row.append(0)
                board_2d.append(row)

            policy_256 = [0.0] * 256
            for idx in range(25):
                r, c = divmod(idx, N)
                policy_256[r * 16 + c] = policy[idx]

            history.append({
                "board_size": N,
                "board": board_2d,
                "current_player": 1,
                "last_move": None,
                "policy": policy_256,
                "value": 0.0,
                "side": current,
            })

            board[move] = current
            winner = _nxn_winner(board, N, WIN_LEN, move)
            if winner != 0:
                break
            current = -current

        # Fill values from game outcome
        for pos in history:
            if winner == 0:
                pos["value"] = 0.0
            elif pos["side"] == winner:
                pos["value"] = 1.0
            else:
                pos["value"] = -1.0
            del pos["side"]

        positions.extend(history)
        games_played += 1

    return positions[:count]


# ---------------------------------------------------------------------------
# Training loop with progress callback
# ---------------------------------------------------------------------------

async def train_variant(
    variant: str,
    callback: Callable,
    epochs: int = 30,
    batch_size: int = 256,
    data_count: int = 4000,
    **kwargs,
) -> None:
    """Train a ResNet model and stream progress via callback."""
    from trainer_lab.config import ModelConfig
    from trainer_lab.data.encoder import board_to_tensor
    from trainer_lab.models.resnet import PolicyValueResNet
    from trainer_lab.training.loss import GomokuLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig()

    # Load or create model
    model_dir = _ensure_saved_dir(variant)
    model_path = model_dir / "model.pt"
    model = PolicyValueResNet(
        in_channels=cfg.in_channels,
        res_filters=cfg.res_filters,
        res_blocks=cfg.res_blocks,
        policy_filters=cfg.policy_filters,
        value_fc=cfg.value_fc,
        board_max=cfg.board_max,
    )
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        logger.info("Resumed model from %s", model_path)
    model = model.to(device)

    # Send start event
    await callback({"type": "train.start", "payload": {"epochs": epochs, "variant": variant}})

    # Generate data
    await callback({"type": "train.status", "payload": {"message": f"Generating {data_count} positions..."}})
    await asyncio.sleep(0)

    if variant == "ttt3":
        positions = _generate_ttt3_positions(data_count)
    elif variant == "ttt5":
        positions = _generate_ttt5_positions(data_count)
    else:
        positions = _generate_ttt5_positions(data_count)

    await callback({"type": "dataset.progress", "payload": {
        "generated": len(positions), "total": data_count, "percent": 100,
        "stage": "complete", "message": f"{len(positions)} positions generated",
    }})

    # Build DataLoader
    planes_list, policy_list, value_list = [], [], []
    for p in positions:
        planes_list.append(board_to_tensor(p))
        policy_list.append(torch.tensor(p["policy"], dtype=torch.float32))
        value_list.append(torch.tensor([p["value"]], dtype=torch.float32))

    dataset = TensorDataset(torch.stack(planes_list), torch.stack(policy_list), torch.stack(value_list))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Training loop
    criterion = GomokuLoss(weight_value=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()

    start_time = time.monotonic()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_ploss = 0.0
        total_vloss = 0.0
        correct = 0
        total = 0
        batches = 0

        for planes, policy_target, value_target in loader:
            planes = planes.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)

            optimizer.zero_grad()
            policy_logits, value_pred = model(planes)
            loss, ploss, vloss = criterion(policy_logits, value_pred, policy_target, value_target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ploss += ploss.item()
            total_vloss += vloss.item()
            batches += 1

            pred = policy_logits.argmax(dim=1)
            target = policy_target.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += planes.size(0)

        avg_loss = total_loss / max(batches, 1)
        avg_ploss = total_ploss / max(batches, 1)
        avg_vloss = total_vloss / max(batches, 1)
        accuracy = correct / max(total, 1)
        elapsed = time.monotonic() - start_time

        await callback({
            "type": "train.progress",
            "payload": {
                "epoch": epoch,
                "totalEpochs": epochs,
                "loss": round(avg_loss, 6),
                "policyLoss": round(avg_ploss, 6),
                "valueLoss": round(avg_vloss, 6),
                "accuracy": round(accuracy, 4),
                "epochPercent": round(epoch / epochs * 100, 1),
                "elapsed": round(elapsed, 1),
                "phase": "training",
                "variant": variant,
            },
        })
        # Yield to event loop so WS can send
        await asyncio.sleep(0)

    # Save model
    torch.save(model.state_dict(), model_path)
    logger.info("Saved model to %s", model_path)

    # Invalidate predict cache
    from gomoku_api.ws.predict_service import clear_cached_model
    clear_cached_model(variant)

    await callback({"type": "train.done", "payload": {"variant": variant, "epochs": epochs}})


def clear_model(variant: str = "all") -> dict:
    """Delete saved model files."""
    from gomoku_api.ws.predict_service import clear_cached_model

    variants = ["ttt3", "ttt5"] if variant == "all" else [variant]
    for v in variants:
        model_path = SAVED_DIR / f"{v}_resnet" / "model.pt"
        if model_path.exists():
            model_path.unlink()
            logger.info("Cleared model: %s", model_path)
        clear_cached_model(v)
    return {"cleared": True, "variant": variant}
