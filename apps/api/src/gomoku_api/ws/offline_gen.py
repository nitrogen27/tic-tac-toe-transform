"""Offline minimax dataset generator for supervised warm start.

Generates minimax-labeled positions and saves to JSON. Slow but runs once.
Usage:
    python -m gomoku_api.ws.offline_gen ttt5 --count 5000
    python -m gomoku_api.ws.offline_gen ttt3 --count 2000
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).resolve().parents[5] / "saved"

TRAIN_CALLBACK = Callable[[dict[str, Any]], Awaitable[None]]


def _resolve_variant(variant: str) -> tuple[int, int]:
    if variant == "ttt3":
        return 3, 3
    if variant == "ttt5":
        return 5, 4
    raise ValueError(f"Unknown variant: {variant}")


def _nxn_winner(board: list[int], n: int, win_len: int, last_move: int) -> int:
    """Check if last_move created a win."""
    if last_move < 0:
        return 0
    r, c = divmod(last_move, n)
    player = board[last_move]
    if player == 0:
        return 0
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for dr, dc in directions:
        count = 1
        for sign in (1, -1):
            for step in range(1, win_len):
                nr, nc = r + dr * step * sign, c + dc * step * sign
                if 0 <= nr < n and 0 <= nc < n and board[nr * n + nc] == player:
                    count += 1
                else:
                    break
        if count >= win_len:
            return player
    return 0


def _nxn_minimax(board: list[int], n: int, win_len: int, current: int,
                 depth: int, alpha: float, beta: float, last_move: int) -> float:
    """Minimax with alpha-beta pruning."""
    if last_move >= 0:
        w = _nxn_winner(board, n, win_len, last_move)
        if w != 0:
            return 1.0 if w == current else -1.0
    if depth == 0 or all(c != 0 for c in board):
        return 0.0

    opponent = 2 if current == 1 else 1
    best = -2.0
    for move in range(n * n):
        if board[move] != 0:
            continue
        board[move] = current
        val = -_nxn_minimax(board, n, win_len, opponent, depth - 1, -beta, -alpha, move)
        board[move] = 0
        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def _nxn_minimax_policy(board: list[int], n: int, win_len: int,
                        current: int, depth: int = 6) -> tuple[list[float], float]:
    """Minimax-evaluated policy for NxN board. Returns (policy_N*N, value)."""
    empty = [i for i in range(n * n) if board[i] == 0]
    if not empty:
        return [0.0] * (n * n), 0.0

    opponent = 2 if current == 1 else 1
    move_values: list[tuple[int, float]] = []
    for move in empty:
        board[move] = current
        val = -_nxn_minimax(board, n, win_len, opponent, depth - 1, -2.0, 2.0, move)
        board[move] = 0
        move_values.append((move, val))

    best_val = max(v for _, v in move_values)
    # Softmax-ish: best moves get most weight
    policy = [0.0] * (n * n)
    top_moves = [m for m, v in move_values if v >= best_val - 0.01]
    weight = 1.0 / len(top_moves)
    for m in top_moves:
        policy[m] = weight
    return policy, best_val


def _flat_to_board2d(board: list[int], board_size: int) -> list[list[int]]:
    return [
        [board[row * board_size + col] for col in range(board_size)]
        for row in range(board_size)
    ]


def _policy_to_256(policy: list[float], board_size: int) -> list[float]:
    """Convert board_size² policy to padded 256 (16×16) format."""
    padded = [0.0] * 256
    for idx, prob in enumerate(policy):
        if prob > 0:
            r, c = divmod(idx, board_size)
            padded[r * 16 + c] = prob
    return padded


def _one_hot_policy(move: int, board_size: int) -> list[float]:
    padded = [0.0] * 256
    if move >= 0:
        row, col = divmod(move, board_size)
        padded[row * 16 + col] = 1.0
    return padded


def _classify_engine_phase(board_size: int, occupied_cells: int) -> str:
    total_cells = board_size * board_size
    if board_size <= 5:
        if occupied_cells <= 3:
            return "opening"
        if occupied_cells <= 10:
            return "early_mid"
        if occupied_cells <= 16:
            return "mid"
        return "late"
    if occupied_cells <= max(3, total_cells // 8):
        return "opening"
    if occupied_cells <= max(10, total_cells // 3):
        return "mid"
    return "late"


def _resolve_engine_sampling_bounds(
    board_size: int,
    total_cells: int,
    *,
    phase_focus: str | None = None,
    rng_value: float | None = None,
) -> tuple[int, int]:
    focus = (phase_focus or "").strip().lower()
    if board_size <= 5:
        if focus in {"opening", "early"}:
            return 0, min(6, total_cells - 1)
        if focus in {"mid", "midgame"}:
            return min(9, total_cells - 1), min(16, total_cells - 1)
        if focus in {"late", "endgame"}:
            return min(17, total_cells - 1), total_cells - 1
    else:
        if focus in {"opening", "early"}:
            return 0, min(max(4, total_cells // 8), total_cells - 1)
        if focus in {"mid", "midgame"}:
            return min(max(5, total_cells // 5), total_cells - 1), min(max(18, (total_cells * 2) // 3), total_cells - 1)
        if focus in {"late", "endgame"}:
            return min(max(19, (total_cells * 2) // 3), total_cells - 1), total_cells - 1

    r = 0.5 if rng_value is None else rng_value
    if board_size <= 5:
        if r < 0.05:
            return 0, min(3, total_cells - 1)
        if r < 0.20:
            return min(4, total_cells - 1), min(10, total_cells - 1)
        if r < 0.55:
            return min(11, total_cells - 1), min(16, total_cells - 1)
        return min(17, total_cells - 1), total_cells - 1

    if r < 0.20:
        return 0, min(3, total_cells - 1)
    if r < 0.60:
        return min(4, total_cells - 1), min(10, total_cells - 1)
    if r < 0.90:
        return min(11, total_cells - 1), min(18, total_cells - 1)
    return min(19, total_cells - 1), total_cells - 1


def _sample_engine_position(
    board_size: int,
    win_len: int,
    rng: random.Random,
    *,
    phase_focus: str | None = None,
) -> tuple[list[int], int, int] | None:
    total_cells = board_size * board_size
    low, high = _resolve_engine_sampling_bounds(
        board_size,
        total_cells,
        phase_focus=phase_focus,
        rng_value=rng.random(),
    )

    if high < low:
        low = 0
    plies = rng.randint(low, high) if high > 0 else 0
    board = [0] * total_cells
    current = 1
    last_move = -1

    for _ in range(plies):
        legal = [idx for idx, cell in enumerate(board) if cell == 0]
        if not legal:
            return None
        move = rng.choice(legal)
        board[move] = current
        last_move = move
        if _nxn_winner(board, board_size, win_len, move) != 0:
            return None
        current = 2 if current == 1 else 1

    if all(cell != 0 for cell in board):
        return None
    return board, current, last_move


async def generate_minimax_dataset(
    variant: str,
    count: int = 5000,
    callback: TRAIN_CALLBACK | None = None,
) -> Path:
    """Generate minimax-labeled positions and save to JSON file."""
    board_size, win_len = _resolve_variant(variant)
    minimax_depth = 4 if board_size <= 3 else 3

    output_dir = SAVED_DIR / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{variant}_minimax.json"

    positions: list[dict[str, Any]] = []
    started_at = time.monotonic()
    last_reported = 0

    while len(positions) < count:
        board = [0] * (board_size * board_size)
        current = 1
        last_move = -1
        history: list[dict[str, Any]] = []
        winner = 0

        for _step in range(board_size * board_size):
            empty = [i for i, c in enumerate(board) if c == 0]
            if not empty:
                break

            # Minimax policy — the expensive part, run in thread
            mm_policy, mm_value = await asyncio.to_thread(
                _nxn_minimax_policy, list(board), board_size, win_len, current, minimax_depth
            )
            policy_256 = _policy_to_256(mm_policy, board_size)

            history.append({
                "board_size": board_size,
                "board": _flat_to_board2d(board, board_size),
                "current_player": current,
                "last_move": list(divmod(last_move, board_size)) if last_move >= 0 else None,
                "policy": policy_256,
                "value": mm_value,
            })

            # Choose move: weighted by policy with some exploration
            weights = [mm_policy[m] for m in empty]
            total = sum(weights)
            if total > 0:
                move = random.choices(empty, weights=weights, k=1)[0]
            else:
                move = random.choice(empty)

            board[move] = current
            winner = _nxn_winner(board, board_size, win_len, move)
            last_move = move
            if winner != 0:
                break
            current = 2 if current == 1 else 1

        # Backfill values based on game outcome
        for pos in history:
            if winner == 0:
                pos["value"] = 0.0
            elif pos["current_player"] == winner:
                pos["value"] = 1.0
            else:
                pos["value"] = -1.0

        positions.extend(history)

        # Progress reporting
        if callback and (len(positions) - last_reported >= 100 or len(positions) >= count):
            last_reported = len(positions)
            elapsed = time.monotonic() - started_at
            speed = len(positions) / max(elapsed, 0.01)
            eta = (count - len(positions)) / max(speed, 0.01)
            await callback({
                "type": "dataset.progress",
                "payload": {
                    "generated": min(len(positions), count),
                    "total": count,
                    "percent": round(100.0 * min(len(positions), count) / count, 1),
                    "elapsed": round(elapsed, 1),
                    "eta": round(eta, 1),
                    "speed": round(speed, 1),
                    "message": f"Generated {min(len(positions), count)}/{count} minimax positions",
                },
            })
            await asyncio.sleep(0)

    # Trim to exact count and save
    positions = positions[:count]
    output_path.write_text(json.dumps(positions, separators=(",", ":")))
    logger.info("Saved %d minimax positions to %s", len(positions), output_path)
    return output_path


async def generate_engine_dataset(
    variant: str,
    count: int = 10_000,
    callback: TRAIN_CALLBACK | None = None,
    phase_focus: str | None = None,
) -> Path:
    """Generate engine-labeled offline dataset via persistent engine worker."""
    from gomoku_api.ws.engine_evaluator import EngineEvaluator

    board_size, win_len = _resolve_variant(variant)

    output_dir = SAVED_DIR / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{variant}_engine.json"

    positions: list[dict[str, Any]] = []
    started_at = time.monotonic()
    last_reported = 0
    attempts = 0
    rng = random.Random(f"{variant}:engine:{count}")
    max_attempts = max(count * 25, 500)

    async with EngineEvaluator() as engine_eval:
        while len(positions) < count and attempts < max_attempts:
            attempts += 1
            sampled = _sample_engine_position(board_size, win_len, rng, phase_focus=phase_focus)
            if sampled is None:
                continue

            board, current, last_move = sampled
            phase_bucket = _classify_engine_phase(board_size, sum(1 for cell in board if cell != 0))
            move, value = await engine_eval.best_move_with_value(board, current, board_size, win_len)
            if move < 0 or move >= len(board) or board[move] != 0:
                continue

            positions.append({
                "board_size": board_size,
                "board": _flat_to_board2d(board, board_size),
                "current_player": current,
                "last_move": list(divmod(last_move, board_size)) if last_move >= 0 else None,
                "policy": _one_hot_policy(move, board_size),
                "value": max(-1.0, min(1.0, float(value))),
                "source": "engine",
                "phaseBucket": phase_bucket,
                "sampleWeight": 1.0,
            })

            if callback and (len(positions) - last_reported >= 100 or len(positions) >= count):
                last_reported = len(positions)
                elapsed = time.monotonic() - started_at
                speed = len(positions) / max(elapsed, 0.01)
                eta = (count - len(positions)) / max(speed, 0.01)
                await callback({
                    "type": "dataset.progress",
                    "payload": {
                        "generated": min(len(positions), count),
                        "total": count,
                        "percent": round(100.0 * min(len(positions), count) / count, 1),
                        "elapsed": round(elapsed, 1),
                        "eta": round(eta, 1),
                        "speed": round(speed, 1),
                        "message": f"Generated {min(len(positions), count)}/{count} engine positions",
                        "stage": "engine_dataset",
                    },
                })
                await asyncio.sleep(0)

    positions = positions[:count]
    output_path.write_text(json.dumps(positions, separators=(",", ":")), encoding="utf-8")
    logger.info("Saved %d engine positions to %s", len(positions), output_path)
    return output_path


# ── CLI entry point ───────────────────────────────────────────────────

async def _cli_callback(event: dict[str, Any]) -> None:
    payload = event.get("payload", {})
    msg = payload.get("message", "")
    pct = payload.get("percent", "")
    if msg:
        print(f"  [{pct}%] {msg}")


async def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate offline dataset")
    parser.add_argument("variant", choices=["ttt3", "ttt5"], help="Game variant")
    parser.add_argument("--count", type=int, default=5000, help="Number of positions")
    parser.add_argument("--mode", choices=["minimax", "engine"], default="minimax", help="Dataset generator")
    parser.add_argument("--phase-focus", choices=["opening", "mid", "late"], default=None, help="Optional engine dataset phase bias")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    print(f"Generating {args.count} {args.mode} positions for {args.variant}...")
    started = time.monotonic()
    if args.mode == "engine":
        path = await generate_engine_dataset(args.variant, args.count, _cli_callback, phase_focus=args.phase_focus)
    else:
        path = await generate_minimax_dataset(args.variant, args.count, _cli_callback)
    elapsed = time.monotonic() - started
    print(f"Done: {path} ({elapsed:.1f}s)")


if __name__ == "__main__":
    asyncio.run(_main())
