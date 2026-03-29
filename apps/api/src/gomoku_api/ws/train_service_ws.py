"""Training service for WebSocket streaming — uses PyTorch via trainer-lab."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import torch
from torch.utils.data import DataLoader, TensorDataset

from gomoku_api.ws.gpu_info import get_gpu_info

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).resolve().parents[5] / "saved"
TRAIN_CALLBACK = Callable[[dict[str, Any]], Awaitable[None]]
_GPU_POLL_INTERVAL_S = 1.5
_PROGRESS_EMIT_INTERVAL_S = 0.75


def _ensure_saved_dir(variant: str) -> Path:
    d = SAVED_DIR / f"{variant}_resnet"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_variant_spec(variant: str) -> tuple[int, int]:
    if variant == "ttt3":
        return 3, 3
    if variant == "ttt5":
        return 5, 4

    match = re.fullmatch(r"gomoku(?P<size>\d+)", variant)
    if match:
        board_size = int(match.group("size"))
        if 7 <= board_size <= 16:
            return board_size, 5

    raise ValueError(f"Unsupported training variant: {variant}")


def _count_model_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _variant_model_hparams(board_size: int, cfg: Any) -> tuple[int, int, int]:
    """Return variant-specific network size.

    Small boards still use compact models, but 5x5 and larger boards get a
    meaningfully wider/deeper network so the GPU has enough work and the model
    has more capacity for tactics.
    """
    if board_size <= 3:
        return 32, 3, 64
    if board_size <= 5:
        return 96, 8, 160
    if board_size <= 9:
        return 96, 6, 160
    return cfg.res_filters, cfg.res_blocks, max(cfg.value_fc, 192)


def _prepare_cuda_runtime(device: torch.device) -> dict[str, bool]:
    enabled = {
        "mixedPrecision": False,
        "tf32": False,
        "channelsLast": False,
        "torchCompile": False,
        "compileMode": None,
    }
    if device.type != "cuda":
        return enabled

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    enabled.update({
        "mixedPrecision": True,
        "tf32": True,
        "channelsLast": True,
    })
    return enabled


def _maybe_compile_model(model: torch.nn.Module, device: torch.device, runtime_flags: dict[str, Any]) -> torch.nn.Module:
    """Best-effort torch.compile() wrapper for CUDA training workloads."""
    if device.type != "cuda" or not hasattr(torch, "compile"):
        return model

    # Triton is required for torch.compile on CUDA; skip if unavailable (Windows)
    try:
        import triton  # noqa: F401
    except ImportError:
        logger.info("Triton not installed — skipping torch.compile (eager mode)")
        return model

    compile_mode = "reduce-overhead"
    try:
        compiled = torch.compile(model, mode=compile_mode, fullgraph=False)
    except Exception as exc:
        logger.warning("torch.compile unavailable for current run, falling back to eager mode: %s", exc)
        return model

    runtime_flags["torchCompile"] = True
    runtime_flags["compileMode"] = compile_mode
    return compiled


def _should_emit_progress(now: float, last_emit_at: float, *, force: bool = False) -> bool:
    return force or (now - last_emit_at) >= _PROGRESS_EMIT_INTERVAL_S


def _maybe_refresh_gpu_info(now: float, last_gpu_probe: float, live_gpu: dict[str, Any]) -> tuple[dict[str, Any], float]:
    if now - last_gpu_probe >= _GPU_POLL_INTERVAL_S:
        return get_gpu_info(), now
    return live_gpu, last_gpu_probe


def _flat_to_board2d(board: list[int], board_size: int) -> list[list[int]]:
    return [
        [board[row * board_size + col] for col in range(board_size)]
        for row in range(board_size)
    ]


def _policy_cell_index(flat_index: int, board_size: int) -> int:
    row, col = divmod(flat_index, board_size)
    return row * 16 + col


def _one_hot_policy(move: int, board_size: int) -> list[float]:
    policy = [0.0] * 256
    if move >= 0:
        policy[_policy_cell_index(move, board_size)] = 1.0
    return policy


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


def _extract_telemetry(gpu_info: dict[str, Any]) -> dict[str, Any]:
    telemetry = gpu_info.get("telemetry") or {}
    vram = gpu_info.get("vram") or {}
    return {
        "gpuUtilization": telemetry.get("utilizationGpu"),
        "gpuMemoryUtilization": telemetry.get("utilizationMemory"),
        "gpuPowerW": telemetry.get("powerDrawW"),
        "gpuPowerLimitW": telemetry.get("powerLimitW"),
        "gpuTemperatureC": telemetry.get("temperatureC"),
        "gpuClockSmMHz": telemetry.get("clockSmMHz"),
        "gpuClockMemMHz": telemetry.get("clockMemMHz"),
        "gpuMemoryUsedMB": telemetry.get("memoryUsedMB", vram.get("usedMB")),
        "gpuMemoryTotalMB": telemetry.get("memoryTotalMB", vram.get("totalMB")),
        "gpuAllocatedMB": vram.get("allocatedMB"),
        "gpuReservedMB": vram.get("reservedMB"),
        "gpuTelemetryTimestamp": telemetry.get("timestamp"),
    }


def _compute_tactical_accuracy(
    model: Any, board_size: int, win_len: int, device: Any, n_samples: int = 200
) -> float:
    """Evaluate model on tactical positions (win/block). Returns accuracy 0-100."""
    from trainer_lab.data.encoder import board_to_tensor
    import torch.nn.functional as F

    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            board = [0] * (board_size * board_size)
            motif = random.choice(("win", "block"))
            current = random.choice((1, 2))
            line_player = current if motif == "win" else (2 if current == 1 else 1)
            dr, dc = random.choice(directions)

            valid_starts = [
                (r, c) for r in range(board_size) for c in range(board_size)
                if 0 <= r + dr * (win_len - 1) < board_size and 0 <= c + dc * (win_len - 1) < board_size
            ]
            if not valid_starts:
                continue
            sr, sc = random.choice(valid_starts)
            gap = random.randrange(win_len)
            target_move = (sr + dr * gap) * board_size + (sc + dc * gap)

            for step in range(win_len):
                r, c = sr + dr * step, sc + dc * step
                flat = r * board_size + c
                if flat != target_move:
                    board[flat] = line_player

            pos = {
                "board_size": board_size,
                "board": [[board[r * board_size + c] for c in range(board_size)] for r in range(board_size)],
                "current_player": current,
                "last_move": None,
            }
            planes = board_to_tensor(pos).unsqueeze(0).to(device)
            if device.type == "cuda":
                planes = planes.contiguous(memory_format=torch.channels_last)
            logits, _ = model(planes)
            legal_mask = planes[:, 2].reshape(1, -1)
            masked = logits + (1.0 - legal_mask) * (-1e8)
            pred = masked.argmax(dim=1).item()

            target_r, target_c = divmod(target_move, board_size)
            target_idx = target_r * 16 + target_c
            if pred == target_idx:
                correct += 1
            total += 1

    model.train()
    return round(100.0 * correct / max(total, 1), 1)


async def _emit_dataset_progress(
    callback: TRAIN_CALLBACK,
    *,
    generated: int,
    total: int,
    stage: str,
    message: str,
    start_time: float,
    games: int = 0,
    unit: str = "positions",
) -> None:
    elapsed = max(time.monotonic() - start_time, 0.001)
    percent = round(min(generated / max(total, 1), 1.0) * 100, 1)
    await callback({
        "type": "dataset.progress",
        "payload": {
            "generated": generated,
            "total": total,
            "games": games,
            "percent": percent,
            "rate": round(generated / elapsed, 2),
            "elapsed": round(elapsed, 1),
            "stage": stage,
            "message": message,
            "unit": unit,
            "workers": 1,
        },
    })


# ---------------------------------------------------------------------------
# Data generation for TTT3 (3x3, win_length=3)
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
    winner = _ttt3_winner(board)
    if winner == current:
        return 1.0
    if winner == -current:
        return -1.0
    empty = [i for i in range(9) if board[i] == 0]
    if not empty:
        return 0.0

    best = -2.0
    for move in empty:
        board[move] = current
        value = -_minimax_value(board, -current)
        board[move] = 0
        best = max(best, value)
    return best


def _minimax_policy(board: list[int], current: int) -> list[float]:
    empty = [i for i in range(9) if board[i] == 0]
    if not empty:
        return [0.0] * 9

    scores: dict[int, float] = {}
    for move in empty:
        board[move] = current
        scores[move] = -_minimax_value(board, -current)
        board[move] = 0

    best_value = max(scores.values())
    best_moves = [move for move, value in scores.items() if value == best_value]
    policy = [0.0] * 9
    for move in best_moves:
        policy[move] = 1.0 / len(best_moves)
    return policy


async def _generate_ttt3_positions(count: int, callback: TRAIN_CALLBACK) -> list[dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    start_time = time.monotonic()
    last_reported = 0

    while len(positions) < count:
        board = [0] * 9
        current = 1
        moves_played = random.randint(0, 6)
        available = list(range(9))
        random.shuffle(available)

        for idx in range(min(moves_played, len(available))):
            board[available[idx]] = current
            current = -current

        if _ttt3_winner(board) != 0:
            continue
        if all(cell != 0 for cell in board):
            continue

        policy = _minimax_policy(list(board), current)
        value = _minimax_value(list(board), current)

        board_2d = []
        for row in range(3):
            encoded_row = []
            for col in range(3):
                cell = board[row * 3 + col]
                if cell == 1:
                    encoded_row.append(1)
                elif cell == -1:
                    encoded_row.append(2)
                else:
                    encoded_row.append(0)
            board_2d.append(encoded_row)

        policy_256 = [0.0] * 256
        for idx, prob in enumerate(policy):
            row, col = divmod(idx, 3)
            policy_256[row * 16 + col] = prob

        positions.append({
            "board_size": 3,
            "board": board_2d,
            "current_player": 1 if current == 1 else 2,
            "last_move": None,
            "policy": policy_256,
            "value": value,
        })

        if len(positions) - last_reported >= 128 or len(positions) == count:
            last_reported = len(positions)
            await _emit_dataset_progress(
                callback,
                generated=len(positions),
                total=count,
                stage="generating",
                message=f"Generated {len(positions)} tactical positions",
                start_time=start_time,
            )
            await asyncio.sleep(0)

    return positions


# ---------------------------------------------------------------------------
# Alpha-beta minimax for small NxN boards (5x5 with depth limit)
# ---------------------------------------------------------------------------


def _nxn_evaluate_heuristic(board: list[int], n: int, win_len: int, player: int) -> float:
    """Quick heuristic evaluation for NxN board from player's perspective."""
    score = 0.0
    opponent = 3 - player
    for r in range(n):
        for c in range(n):
            if board[r * n + c] != 0:
                continue
            # Count threats around empty cell
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                my_count = opp_count = 0
                for s in range(1, win_len):
                    nr, nc = r + dr * s, c + dc * s
                    if 0 <= nr < n and 0 <= nc < n:
                        v = board[nr * n + nc]
                        if v == player:
                            my_count += 1
                        elif v == opponent:
                            opp_count += 1
                        else:
                            break
                    else:
                        break
                for s in range(1, win_len):
                    nr, nc = r - dr * s, c - dc * s
                    if 0 <= nr < n and 0 <= nc < n:
                        v = board[nr * n + nc]
                        if v == player:
                            my_count += 1
                        elif v == opponent:
                            opp_count += 1
                        else:
                            break
                    else:
                        break
                if my_count >= win_len - 1 and opp_count == 0:
                    score += 10.0
                elif my_count >= win_len - 2 and opp_count == 0:
                    score += 1.0
                if opp_count >= win_len - 1 and my_count == 0:
                    score -= 8.0
                elif opp_count >= win_len - 2 and my_count == 0:
                    score -= 0.8
    return score


def _nxn_minimax(
    board: list[int], n: int, win_len: int, current: int,
    depth: int, alpha: float, beta: float, last_move: int,
) -> float:
    """Alpha-beta minimax for NxN boards with depth limit."""
    if last_move >= 0:
        w = _nxn_winner(board, n, win_len, last_move)
        if w != 0:
            return 100.0 if w == current else -100.0

    if depth <= 0:
        return _nxn_evaluate_heuristic(board, n, win_len, current)

    empty = [i for i in range(n * n) if board[i] == 0]
    if not empty:
        return 0.0

    # Order moves: center first, then near existing stones
    center = n // 2
    def move_priority(m: int) -> float:
        r, c = divmod(m, n)
        dist = abs(r - center) + abs(c - center)
        # Bonus for adjacent to existing stones
        adj = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and board[nr * n + nc] != 0:
                    adj += 1
        return -adj * 10 + dist
    empty.sort(key=move_priority)

    best = -200.0
    opp = 3 - current
    for move in empty:
        board[move] = current
        val = -_nxn_minimax(board, n, win_len, opp, depth - 1, -beta, -alpha, move)
        board[move] = 0
        if val > best:
            best = val
        alpha = max(alpha, val)
        if alpha >= beta:
            break
    return best


def _nxn_minimax_policy(board: list[int], n: int, win_len: int, current: int, depth: int = 6) -> tuple[list[float], float]:
    """Get minimax-evaluated policy for NxN board. Returns (policy_N*N, value)."""
    empty = [i for i in range(n * n) if board[i] == 0]
    if not empty:
        return [0.0] * (n * n), 0.0

    scores: dict[int, float] = {}
    opp = 3 - current
    for move in empty:
        board[move] = current
        w = _nxn_winner(board, n, win_len, move)
        if w != 0:
            scores[move] = 100.0
            board[move] = 0
            continue
        val = -_nxn_minimax(board, n, win_len, opp, depth - 1, -200.0, 200.0, move)
        scores[move] = val
        board[move] = 0

    best_val = max(scores.values())
    # Softmax-like distribution: boost good moves
    import math
    temperature = 1.0
    exp_scores = {}
    for m, s in scores.items():
        exp_scores[m] = math.exp(min((s - best_val) / max(temperature, 0.1), 20))
    total = sum(exp_scores.values())

    policy = [0.0] * (n * n)
    for m, e in exp_scores.items():
        policy[m] = e / total

    value = max(-1.0, min(1.0, best_val / 100.0))
    return policy, value


# ---------------------------------------------------------------------------
# Generic NxN data generation (with minimax for small boards)
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


async def _generate_nxn_positions(
    count: int,
    board_size: int,
    win_len: int,
    callback: TRAIN_CALLBACK,
) -> list[dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    games_played = 0
    start_time = time.monotonic()
    last_reported = 0
    while len(positions) < count:
        board = [0] * (board_size * board_size)
        current_player = 1
        last_move_flat = -1
        history: list[dict[str, Any]] = []
        winner = 0

        for _move in range(board_size * board_size):
            empty = [idx for idx, cell in enumerate(board) if cell == 0]
            if not empty:
                break

            # Uniform policy over legal moves (minimax moved to offline gen)
            policy = [0.0] * 256
            probability = 1.0 / len(empty)
            for idx in empty:
                r, c = divmod(idx, board_size)
                policy[r * 16 + c] = probability
            pos_value = 0.0

            board_2d = [
                [board[row * board_size + col] for col in range(board_size)]
                for row in range(board_size)
            ]
            history.append({
                "board_size": board_size,
                "board": board_2d,
                "current_player": current_player,
                "last_move": list(divmod(last_move_flat, board_size)) if last_move_flat >= 0 else None,
                "policy": policy,
                "value": pos_value,
            })

            # Choose move: random (minimax moved to offline gen)
            move = random.choice(empty)
            board[move] = current_player
            last_move_flat = move

            winner = _nxn_winner(board, board_size, win_len, move)
            if winner != 0:
                break
            current_player = 2 if current_player == 1 else 1

        for pos in history:
            if winner == 0:
                pos["value"] = 0.0
            elif pos["current_player"] == winner:
                pos["value"] = 1.0
            else:
                pos["value"] = -1.0

        positions.extend(history)
        games_played += 1

        if len(positions) - last_reported >= max(board_size * 2, 128) or len(positions) >= count:
            last_reported = len(positions)
            await _emit_dataset_progress(
                callback,
                generated=min(len(positions), count),
                total=count,
                stage="generating",
                message=f"Generated {min(len(positions), count)} positions from {games_played} games",
                start_time=start_time,
                games=games_played,
            )
            await asyncio.sleep(0)

    return positions[:count]


async def _build_positions(
    variant: str,
    data_count: int,
    callback: TRAIN_CALLBACK,
) -> tuple[list[dict[str, Any]], int, int]:
    board_size, win_len = _resolve_variant_spec(variant)
    if variant == "ttt3":
        positions = await _generate_ttt3_positions(data_count, callback)
    else:
        positions = await _generate_nxn_positions(data_count, board_size, win_len, callback)
    return positions, board_size, win_len


# ---------------------------------------------------------------------------
# Tactical curriculum for large boards
# ---------------------------------------------------------------------------


async def _generate_tactical_curriculum_positions(
    count: int,
    board_size: int,
    win_len: int,
    callback: TRAIN_CALLBACK,
) -> list[dict[str, Any]]:
    """Generate immediate-win / immediate-block positions for strong tactical supervision."""
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    positions: list[dict[str, Any]] = []
    started_at = time.monotonic()
    last_reported = 0

    while len(positions) < count:
        board = [0] * (board_size * board_size)
        motif = random.choice(("win", "block"))
        current_player = random.choice((1, 2))
        line_player = current_player if motif == "win" else (2 if current_player == 1 else 1)
        target_player = current_player
        direction = random.choice(directions)
        dr, dc = direction

        # Pick a start cell where a full win_len segment fits.
        valid_starts: list[tuple[int, int]] = []
        for row in range(board_size):
            for col in range(board_size):
                end_row = row + dr * (win_len - 1)
                end_col = col + dc * (win_len - 1)
                if 0 <= end_row < board_size and 0 <= end_col < board_size:
                    valid_starts.append((row, col))
        if not valid_starts:
            break

        start_row, start_col = random.choice(valid_starts)
        gap_index = random.randrange(win_len)
        target_move = (start_row + dr * gap_index) * board_size + (start_col + dc * gap_index)

        occupied: set[int] = {target_move}
        for step in range(win_len):
            row = start_row + dr * step
            col = start_col + dc * step
            flat = row * board_size + col
            if flat == target_move:
                continue
            board[flat] = line_player
            occupied.add(flat)

        # Add a few random context stones far from the tactical segment.
        extra_stones = random.randint(0, max(2, board_size // 3))
        for _ in range(extra_stones):
            placed = False
            for _attempt in range(20):
                move = random.randrange(board_size * board_size)
                if move in occupied or board[move] != 0:
                    continue
                if abs((move // board_size) - (target_move // board_size)) <= 1 and abs((move % board_size) - (target_move % board_size)) <= 1:
                    continue
                board[move] = random.choice((1, 2))
                occupied.add(move)
                placed = True
                break
            if not placed:
                break

        # Reject accidental terminal boards or broken motifs.
        last_stone = next((idx for idx, cell in enumerate(board) if cell == line_player), -1)
        if last_stone >= 0 and _nxn_winner(board, board_size, win_len, last_stone) != 0:
            continue

        policy = _one_hot_policy(target_move, board_size)
        value = 1.0 if motif == "win" else 0.35
        positions.append({
            "board_size": board_size,
            "board": _flat_to_board2d(board, board_size),
            "current_player": target_player,
            "last_move": None,
            "policy": policy,
            "value": value,
            "motif": motif,
        })

        if len(positions) - last_reported >= 128 or len(positions) == count:
            last_reported = len(positions)
            await _emit_dataset_progress(
                callback,
                generated=len(positions),
                total=count,
                stage="generating",
                message=f"Generated {len(positions)} tactical {board_size}x{board_size} positions",
                start_time=started_at,
            )
            await asyncio.sleep(0)

    return positions


# ---------------------------------------------------------------------------
# Self-play game generation (bootstrap + self-play)
# ---------------------------------------------------------------------------


async def _batched_model_forward(
    game_states: list[dict[str, Any]],
    board_size: int,
    model: Any,
    device: Any,
) -> tuple[list[list[float]], list[float]]:
    """Run one batched policy/value inference for the active game states."""
    from trainer_lab.data.encoder import board_to_tensor
    import torch.nn.functional as F

    if not game_states:
        return [], []

    pos_dicts = []
    for state in game_states:
        pos_dicts.append({
            "board_size": board_size,
            "board": _flat_to_board2d(state["board"], board_size),
            "current_player": state["current"],
            "last_move": list(divmod(state["last_move"], board_size)) if state["last_move"] >= 0 else None,
        })

    planes = torch.stack([board_to_tensor(pos) for pos in pos_dicts])
    planes = planes.to(device, non_blocking=device.type == "cuda")
    if device.type == "cuda":
        planes = planes.contiguous(memory_format=torch.channels_last)

    model.eval()
    with torch.inference_mode():
        logits, values = model(planes)

    logits_cpu = logits.detach().cpu()
    values_cpu = values.detach().cpu().view(-1).tolist()
    policies: list[list[float]] = []

    for idx, state in enumerate(game_states):
        legal = [move for move, cell in enumerate(state["board"]) if cell == 0]
        policy = [0.0] * 256
        if legal:
            masked = torch.full_like(logits_cpu[idx], float("-inf"))
            for move in legal:
                masked[_policy_cell_index(move, board_size)] = logits_cpu[idx, _policy_cell_index(move, board_size)]
            probs = F.softmax(masked, dim=0)
            for move in legal:
                policy[_policy_cell_index(move, board_size)] = probs[_policy_cell_index(move, board_size)].item()
        policies.append(policy)
    return policies, values_cpu


def _select_selfplay_move(
    board: list[int],
    board_size: int,
    win_len: int,
    current: int,
    policy256: list[float],
    move_count: int,
    *,
    temperature_moves: int = 10,
    dirichlet_weight: float = 0.15,
) -> int:
    legal = [move for move, cell in enumerate(board) if cell == 0]
    if not legal:
        return -1

    winning_move = _find_immediate_move(board, board_size, win_len, current)
    if winning_move is not None:
        return winning_move

    blocking_move = _find_immediate_move(board, board_size, win_len, 2 if current == 1 else 1)
    if blocking_move is not None:
        return blocking_move

    weights = [max(policy256[_policy_cell_index(move, board_size)], 0.0) for move in legal]
    total = sum(weights)
    if total <= 0:
        return random.choice(legal)

    if move_count < temperature_moves:
        noisy = list(weights)
        if len(noisy) > 1 and dirichlet_weight > 0:
            noise = torch.distributions.dirichlet.Dirichlet(
                torch.full((len(noisy),), 0.3, dtype=torch.float32)
            ).sample().tolist()
            noisy = [
                (1.0 - dirichlet_weight) * w + dirichlet_weight * n
                for w, n in zip(noisy, noise)
            ]
        return random.choices(legal, weights=noisy, k=1)[0]

    return legal[max(range(len(legal)), key=lambda idx: weights[idx])]


def _build_train_pool(
    latest_positions: list[dict[str, Any]],
    replay_positions: list[dict[str, Any]],
    *,
    data_count: int,
    seed_positions: list[dict[str, Any]] | None = None,
    minimax_positions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Mix fresh self-play, replay, tactical seed, and minimax positions.

    Hardened quotas:
      20% tactical seed (min)
      20% minimax seed (min)
      60% recent self-play (remaining)
    Replay fills gaps if any bucket is too small.
    """
    if data_count <= 0:
        return []

    seed_positions = seed_positions or []
    minimax_positions = minimax_positions or []

    # Minimum quotas: 20% tactical, 20% minimax, rest = latest + replay
    tactical_quota = min(len(seed_positions), max(data_count * 20 // 100, 0))
    minimax_quota = min(len(minimax_positions), max(data_count * 20 // 100, 0))
    latest_budget = data_count - tactical_quota - minimax_quota
    latest_quota = min(len(latest_positions), latest_budget)
    remaining = max(latest_budget - latest_quota, 0)
    replay_quota = min(len(replay_positions), remaining)

    pool: list[dict[str, Any]] = []
    if latest_quota > 0 and latest_positions:
        pool.extend(random.sample(latest_positions, latest_quota))
    if replay_quota > 0 and replay_positions:
        pool.extend(random.sample(replay_positions, replay_quota))
    if tactical_quota > 0 and seed_positions:
        pool.extend(random.sample(seed_positions, tactical_quota))
    if minimax_quota > 0 and minimax_positions:
        pool.extend(random.sample(minimax_positions, minimax_quota))

    # Backfill if any bucket was too small.
    if len(pool) < data_count:
        leftovers = latest_positions + replay_positions + seed_positions + minimax_positions
        needed = min(data_count - len(pool), len(leftovers))
        if needed > 0:
            pool.extend(random.sample(leftovers, needed))

    return pool[:data_count]


async def _play_selfplay_games_batched(
    num_games: int,
    board_size: int,
    win_len: int,
    model: Any,
    device: Any,
    callback: TRAIN_CALLBACK,
    phase: str,
    *,
    teacher_mode: str = "policy",
    completed_phases: list[str] | None = None,
    iteration: int = 0,
    total_iterations: int = 0,
    overall_percent_base: float = 0.0,
    overall_percent_range: float = 10.0,
    runtime_flags: dict[str, bool] | None = None,
    **extra_fields: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Generate self-play games using batched model inference.

    This is much more GPU-friendly than per-move single-state inference and is
    the default path for bootstrap and large-board self-play.
    """
    positions: list[dict[str, Any]] = []
    stats = {"wins": 0, "losses": 0, "draws": 0}
    started_at = time.monotonic()
    stage = "self_play_game"
    last_emit_at = 0.0
    last_gpu_probe = 0.0
    live_gpu = get_gpu_info()

    games: list[dict[str, Any]] = []
    for _ in range(num_games):
        games.append({
            "board": [0] * (board_size * board_size),
            "current": 1,
            "last_move": -1,
            "history": [],
            "finished": False,
        })

    finished_games = 0
    while finished_games < num_games:
        active = [game for game in games if not game["finished"]]
        if not active:
            break

        policies256, _values = await _batched_model_forward(active, board_size, model, device)

        for game, model_policy in zip(active, policies256):
            board = game["board"]
            current = game["current"]
            last_move = game["last_move"]
            move_count = sum(1 for cell in board if cell != 0)

            # Always use model policy as target (no online minimax)
            target_policy = list(model_policy)

            winning_move = _find_immediate_move(board, board_size, win_len, current)
            blocking_move = None if winning_move is not None else _find_immediate_move(
                board, board_size, win_len, 2 if current == 1 else 1
            )
            if winning_move is not None:
                target_policy = _one_hot_policy(winning_move, board_size)
            elif blocking_move is not None:
                target_policy = _one_hot_policy(blocking_move, board_size)

            game["history"].append({
                "board_size": board_size,
                "board": _flat_to_board2d(board, board_size),
                "current_player": current,
                "last_move": list(divmod(last_move, board_size)) if last_move >= 0 else None,
                "policy": target_policy,
                "value": 0.0,
            })

            move = _select_selfplay_move(
                board,
                board_size,
                win_len,
                current,
                model_policy,
                move_count,
            )
            if move < 0:
                game["finished"] = True
                finished_games += 1
                stats["draws"] += 1
                continue

            board[move] = current
            game["last_move"] = move
            winner = _nxn_winner(board, board_size, win_len, move)
            if winner != 0 or all(cell != 0 for cell in board):
                result = winner
                for pos in game["history"]:
                    if result == 0:
                        pos["value"] = 0.0
                    elif pos["current_player"] == result:
                        pos["value"] = 1.0
                    else:
                        pos["value"] = -1.0
                positions.extend(game["history"])
                game["finished"] = True
                finished_games += 1
                if result == 1:
                    stats["wins"] += 1
                elif result == 2:
                    stats["losses"] += 1
                else:
                    stats["draws"] += 1
            else:
                game["current"] = 2 if current == 1 else 1

        now = time.monotonic()
        force_emit = finished_games >= num_games
        if _should_emit_progress(now, last_emit_at, force=force_emit):
            elapsed = max(now - started_at, 0.01)
            speed = finished_games / elapsed
            pct = overall_percent_base + overall_percent_range * (finished_games / max(num_games, 1)) * 0.6
            live_gpu, last_gpu_probe = _maybe_refresh_gpu_info(now, last_gpu_probe, live_gpu)
            telemetry = _extract_telemetry(live_gpu)
            rf = runtime_flags or {}
            await callback({
                "type": "train.progress",
                "payload": {
                    "phase": phase,
                    "stage": stage,
                    "variant": extra_fields.get("variant", ""),
                    "completedPhases": completed_phases or [],
                    "game": finished_games,
                    "totalGames": num_games,
                    "iteration": iteration,
                    "totalIterations": total_iterations,
                    "selfPlayStats": stats,
                    "positions": len(positions),
                    "epoch": 0,
                    "totalEpochs": 0,
                    "percent": round(pct, 1),
                    "elapsed": round(elapsed, 1),
                    "eta": round((elapsed / max(finished_games, 1)) * max(num_games - finished_games, 0), 1),
                    "speed": round(speed, 2),
                    "speedUnit": "g/s",
                    "teacherMode": teacher_mode,
                    "mixedPrecision": rf.get("mixedPrecision", False),
                    "tf32": rf.get("tf32", False),
                    "torchCompile": rf.get("torchCompile", False),
                    "compileMode": rf.get("compileMode"),
                    "gpu": live_gpu,
                    **telemetry,
                    **{k: v for k, v in extra_fields.items() if k not in ("variant",)},
                },
            })
            last_emit_at = now
            await asyncio.sleep(0)

    return positions, stats


async def _play_selfplay_games(
    num_games: int,
    board_size: int,
    win_len: int,
    model: Any,
    device: Any,
    callback: TRAIN_CALLBACK,
    phase: str,
    use_mcts: bool = False,
    mcts_simulations: int = 16,
    completed_phases: list[str] | None = None,
    iteration: int = 0,
    total_iterations: int = 0,
    overall_percent_base: float = 0.0,
    overall_percent_range: float = 10.0,
    runtime_flags: dict[str, bool] | None = None,
    **extra_fields: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Play self-play games and return (positions, selfPlayStats)."""
    import torch.nn.functional as F
    from trainer_lab.data.encoder import board_to_tensor

    positions: list[dict[str, Any]] = []
    stats = {"wins": 0, "losses": 0, "draws": 0}
    started_at = time.monotonic()
    stage = "mcts_game" if use_mcts else "generating"
    last_emit_at = 0.0
    last_gpu_probe = 0.0
    live_gpu = get_gpu_info()

    for g in range(num_games):
        board = [0] * (board_size * board_size)
        current = 1
        last_move = -1
        history: list[dict[str, Any]] = []
        winner = 0

        for _ in range(board_size * board_size):
            empty = [i for i, c in enumerate(board) if c == 0]
            if not empty:
                break

            # Choose move
            mcts_policy = None  # Reset each turn
            if use_mcts and model is not None:
                from trainer_lab.self_play.player import GameState, mcts_search
                gs = GameState(board_size)
                gs.board = [[board[r * board_size + c] for c in range(board_size)] for r in range(board_size)]
                gs.current_player = 1 if current == 1 else 2
                gs.move_count = sum(1 for x in board if x != 0)
                if last_move >= 0:
                    gs.last_move = divmod(last_move, board_size)
                model.eval()
                # Run MCTS in thread pool to avoid blocking event loop
                mcts_policy, _ = await asyncio.to_thread(
                    mcts_search, gs, model, device, num_simulations=mcts_simulations
                )
                # Sample with temperature
                total_p = sum(mcts_policy)
                if total_p > 0:
                    move = random.choices(range(len(mcts_policy)), weights=mcts_policy, k=1)[0]
                else:
                    move = random.choice(empty)
            elif model is not None:
                # Model inference without MCTS
                board_2d = [[0] * board_size for _ in range(board_size)]
                for idx in range(board_size * board_size):
                    r, c = divmod(idx, board_size)
                    v = board[idx]
                    if v == current:
                        board_2d[r][c] = 1
                    elif v != 0:
                        board_2d[r][c] = 2
                pos_dict = {"board_size": board_size, "board": board_2d, "current_player": 1, "last_move": list(divmod(last_move, board_size)) if last_move >= 0 else None}
                tensor = board_to_tensor(pos_dict).unsqueeze(0).to(device)
                model.eval()
                with torch.no_grad():
                    logits, _ = model(tensor)
                logits = logits.squeeze(0).cpu()
                mask = torch.full_like(logits, float("-inf"))
                for idx in empty:
                    r, c = divmod(idx, board_size)
                    mask[r * 16 + c] = 0.0
                probs = F.softmax(logits + mask, dim=0)
                move = max(empty, key=lambda i: probs[divmod(i, board_size)[0] * 16 + divmod(i, board_size)[1]].item())
            else:
                move = random.choice(empty)

            # Build policy target
            if use_mcts and mcts_policy is not None:
                # MCTS: use visit count distribution
                policy = [0.0] * 256
                for idx in range(board_size * board_size):
                    r, c = divmod(idx, board_size)
                    policy[r * 16 + c] = mcts_policy[idx] if idx < len(mcts_policy) else 0.0
            else:
                # No MCTS: uniform over legal moves (fallback)
                policy = [0.0] * 256
                for idx in empty:
                    r, c = divmod(idx, board_size)
                    policy[r * 16 + c] = 1.0 / len(empty)

            # Record position
            board_2d = [[0] * board_size for _ in range(board_size)]
            for idx in range(board_size * board_size):
                r, c = divmod(idx, board_size)
                board_2d[r][c] = board[idx]
            history.append({
                "board_size": board_size, "board": board_2d,
                "current_player": current, "last_move": list(divmod(last_move, board_size)) if last_move >= 0 else None,
                "policy": policy, "value": 0.0,
            })

            board[move] = current
            last_move = move
            winner = _nxn_winner(board, board_size, win_len, move)
            if winner != 0:
                break
            current = 2 if current == 1 else 1

        # Fill outcome values
        for pos in history:
            if winner == 0:
                pos["value"] = 0.0
            elif pos["current_player"] == winner:
                pos["value"] = 1.0
            else:
                pos["value"] = -1.0
        positions.extend(history)

        if winner == 1:
            stats["wins"] += 1
        elif winner == 2:
            stats["losses"] += 1
        else:
            stats["draws"] += 1

        # Emit progress every 2 games (or every game for slow MCTS)
        emit_interval = 1 if use_mcts else 2
        now = time.monotonic()
        if ((g + 1) % emit_interval == 0 and _should_emit_progress(now, last_emit_at)) or g + 1 == num_games:
            elapsed = now - started_at
            game_pct = (g + 1) / num_games
            speed = (g + 1) / max(elapsed, 0.01)
            pct = overall_percent_base + overall_percent_range * game_pct * 0.6

            live_gpu, last_gpu_probe = _maybe_refresh_gpu_info(now, last_gpu_probe, live_gpu)
            telemetry = _extract_telemetry(live_gpu)
            rf = runtime_flags or {}
            await callback({
                "type": "train.progress",
                "payload": {
                    "phase": phase, "stage": stage, "variant": extra_fields.get("variant", ""),
                    "completedPhases": completed_phases or [],
                    "game": g + 1, "totalGames": num_games,
                    "iteration": iteration, "totalIterations": total_iterations,
                    "selfPlayStats": stats,
                    "positions": len(positions),
                    "epoch": 0, "totalEpochs": 0,
                    "percent": round(pct, 1),
                    "elapsed": round(elapsed, 1),
                    "eta": round((elapsed / max(g + 1, 1)) * max(num_games - g - 1, 0), 1),
                    "speed": round(speed, 2), "speedUnit": "g/s",
                    "mixedPrecision": rf.get("mixedPrecision", False),
                    "tf32": rf.get("tf32", False),
                    "torchCompile": rf.get("torchCompile", False),
                    "compileMode": rf.get("compileMode"),
                    "gpu": live_gpu, **telemetry,
                    **{k: v for k, v in extra_fields.items() if k not in ("variant",)},
                },
            })
            last_emit_at = now
            await asyncio.sleep(0)

    return positions, stats


# ---------------------------------------------------------------------------
# Training epoch loop (extracted for reuse across phases)
# ---------------------------------------------------------------------------


async def _run_training_epochs(
    model: Any,
    positions: list[dict[str, Any]],
    num_epochs: int,
    batch_size: int,
    device: Any,
    runtime_flags: dict[str, bool],
    callback: TRAIN_CALLBACK,
    metrics_history: list[dict[str, Any]],
    phase: str,
    stage: str = "training",
    completed_phases: list[str] | None = None,
    iteration: int = 0,
    total_iterations: int = 0,
    selfplay_stats: dict[str, int] | None = None,
    overall_started_at: float = 0.0,
    overall_percent_base: float = 0.0,
    overall_percent_range: float = 10.0,
    **extra_fields: Any,
) -> None:
    """Run training epochs on positions, streaming progress to callback."""
    from trainer_lab.data.encoder import board_to_tensor
    from trainer_lab.training.loss import GomokuLoss
    from trainer_lab.training.metrics import policy_accuracy, value_mae

    if not positions:
        return

    planes_list = [board_to_tensor(p) for p in positions]
    policy_list = [torch.tensor(p["policy"], dtype=torch.float32) for p in positions]
    value_list = [torch.tensor([p["value"]], dtype=torch.float32) for p in positions]

    dataset = TensorDataset(torch.stack(planes_list), torch.stack(policy_list), torch.stack(value_list))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                        pin_memory=device.type == "cuda")

    criterion = GomokuLoss(weight_value=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    use_amp = runtime_flags["mixedPrecision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()

    total_batches = len(loader)
    total_steps = max(total_batches * num_epochs, 1)
    model_params = _count_model_parameters(model)
    samples_seen = 0
    last_gpu_probe = 0.0
    last_emit_at = 0.0
    live_gpu = get_gpu_info()
    phase_start = time.monotonic()

    for epoch in range(1, num_epochs + 1):
        ep_loss = ep_ploss = ep_vloss = ep_acc = ep_mae = 0.0
        ep_samples = 0

        for bi, (planes, pol_t, val_t) in enumerate(loader, 1):
            step_t = time.perf_counter()
            planes = planes.to(device, non_blocking=True)
            pol_t = pol_t.to(device, non_blocking=True)
            val_t = val_t.to(device, non_blocking=True)
            if device.type == "cuda":
                planes = planes.contiguous(memory_format=torch.channels_last)

            legal_mask = planes[:, 2].reshape(planes.size(0), -1)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, vpred = model(planes)
                loss, pl, vl = criterion(logits, vpred, pol_t, val_t, legal_mask=legal_mask)

            if use_amp:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()

            bt = max(time.perf_counter() - step_t, 1e-6)

            bs = planes.size(0)
            acc = policy_accuracy(logits.detach(), pol_t, legal_mask=legal_mask)
            mae = value_mae(vpred.detach(), val_t)

            ep_loss += loss.item() * bs; ep_ploss += pl.item() * bs; ep_vloss += vl.item() * bs
            ep_acc += acc * bs; ep_mae += mae * bs; ep_samples += bs; samples_seen += bs

            a_loss = ep_loss / max(ep_samples, 1)
            a_acc = (ep_acc / max(ep_samples, 1)) * 100.0
            a_mae = ep_mae / max(ep_samples, 1)

            done_steps = (epoch - 1) * total_batches + bi
            elapsed = max(time.monotonic() - overall_started_at, 0.01)
            phase_elapsed = max(time.monotonic() - phase_start, 0.01)
            sps = samples_seen / phase_elapsed
            pct = overall_percent_base + overall_percent_range * (0.6 + 0.4 * done_steps / total_steps)

            now = time.monotonic()
            force_emit = bi == total_batches
            if _should_emit_progress(now, last_emit_at, force=force_emit):
                live_gpu, last_gpu_probe = _maybe_refresh_gpu_info(now, last_gpu_probe, live_gpu)
                telemetry = _extract_telemetry(live_gpu)

                cur_hist = metrics_history + [{
                    "epoch": epoch,
                    "loss": round(a_loss, 6),
                    "acc": round(a_acc, 2),
                    "mae": round(a_mae, 6),
                }]
                await callback({
                    "type": "train.progress",
                    "payload": {
                        "phase": phase, "stage": stage, "variant": extra_fields.get("variant", ""),
                        "completedPhases": completed_phases or [],
                        "iteration": iteration, "totalIterations": total_iterations,
                        "selfPlayStats": selfplay_stats or {},
                        "epoch": epoch, "totalEpochs": num_epochs, "epochs": num_epochs,
                        "batch": bi, "totalBatches": total_batches, "batchesPerEpoch": total_batches,
                        "loss": round(a_loss, 6), "policyLoss": round(ep_ploss / max(ep_samples, 1), 6),
                        "valueLoss": round(ep_vloss / max(ep_samples, 1), 6),
                        "accuracy": round(a_acc, 2), "acc": round(a_acc, 2), "mae": round(a_mae, 6),
                        "percent": round(pct, 1), "epochPercent": round(bi / max(total_batches, 1) * 100, 1),
                        "elapsed": round(elapsed, 1),
                        "eta": round((phase_elapsed / done_steps) * max(total_steps - done_steps, 0), 1),
                        "speed": round(sps, 2), "speedUnit": "samples/s",
                        "samplesPerSec": round(sps, 2), "batchesPerSec": round(done_steps / phase_elapsed, 3),
                        "batchTimeMs": round(bt * 1000, 2),
                        "positions": len(positions), "effectivePositions": len(positions),
                        "batchSize": batch_size, "learningRate": 2e-3,
                        "modelParams": model_params,
                        "device": device.type, "deviceName": live_gpu.get("name", str(device)),
                        "mixedPrecision": runtime_flags["mixedPrecision"],
                        "tf32": runtime_flags["tf32"], "channelsLast": runtime_flags["channelsLast"],
                        "torchCompile": runtime_flags.get("torchCompile", False),
                        "compileMode": runtime_flags.get("compileMode"),
                        "metricsHistory": cur_hist, "gpu": live_gpu, **telemetry,
                    },
                })
                last_emit_at = now
                await asyncio.sleep(0)

        metrics_history.append({
            "epoch": len(metrics_history) + 1,
            "loss": round(ep_loss / max(ep_samples, 1), 6),
            "acc": round((ep_acc / max(ep_samples, 1)) * 100.0, 2),
            "mae": round(ep_mae / max(ep_samples, 1), 6),
        })


async def _run_training_steps(
    model: Any,
    positions: list[dict[str, Any]],
    max_steps: int,
    batch_size: int,
    device: Any,
    runtime_flags: dict[str, bool],
    callback: TRAIN_CALLBACK,
    metrics_history: list[dict[str, Any]],
    phase: str,
    stage: str = "training",
    completed_phases: list[str] | None = None,
    iteration: int = 0,
    total_iterations: int = 0,
    selfplay_stats: dict[str, int] | None = None,
    overall_started_at: float = 0.0,
    overall_percent_base: float = 0.0,
    overall_percent_range: float = 10.0,
    **extra_fields: Any,
) -> None:
    """Step-based training with manual batching — keeps GPU busy longer."""
    from trainer_lab.data.encoder import board_to_tensor
    from trainer_lab.training.loss import GomokuLoss
    from trainer_lab.training.metrics import policy_accuracy, value_mae

    if not positions or max_steps <= 0:
        return

    # Pre-stack tensors on CPU with pin_memory
    all_planes = torch.stack([board_to_tensor(p) for p in positions])
    all_policy = torch.stack([torch.tensor(p["policy"], dtype=torch.float32) for p in positions])
    all_value = torch.stack([torch.tensor([p["value"]], dtype=torch.float32) for p in positions])
    if device.type == "cuda":
        all_planes = all_planes.pin_memory()
        all_policy = all_policy.pin_memory()
        all_value = all_value.pin_memory()

    n = all_planes.size(0)
    criterion = GomokuLoss(weight_value=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    use_amp = runtime_flags["mixedPrecision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()

    model_params = _count_model_parameters(model)
    samples_seen = 0
    last_gpu_probe = 0.0
    last_emit_at = 0.0
    live_gpu = get_gpu_info()
    phase_start = time.monotonic()
    step = 0
    epoch = 0
    cum_loss = cum_acc = cum_mae = 0.0
    cum_samples = 0

    while step < max_steps:
        epoch += 1
        perm = torch.randperm(n)
        for bi_start in range(0, n, batch_size):
            if step >= max_steps:
                break
            idx = perm[bi_start : bi_start + batch_size]
            planes = all_planes[idx].to(device, non_blocking=True)
            pol_t = all_policy[idx].to(device, non_blocking=True)
            val_t = all_value[idx].to(device, non_blocking=True)
            if device.type == "cuda":
                planes = planes.contiguous(memory_format=torch.channels_last)

            legal_mask = planes[:, 2].reshape(planes.size(0), -1)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, vpred = model(planes)
                loss, pl, vl = criterion(logits, vpred, pol_t, val_t, legal_mask=legal_mask)

            if use_amp:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()

            step += 1
            bs = planes.size(0)
            acc = policy_accuracy(logits.detach(), pol_t, legal_mask=legal_mask)
            mae = value_mae(vpred.detach(), val_t)
            cum_loss += loss.item() * bs; cum_acc += acc * bs; cum_mae += mae * bs
            cum_samples += bs; samples_seen += bs

            a_loss = cum_loss / max(cum_samples, 1)
            a_acc = (cum_acc / max(cum_samples, 1)) * 100.0
            a_mae = cum_mae / max(cum_samples, 1)

            now = time.monotonic()
            if _should_emit_progress(now, last_emit_at, force=(step >= max_steps)):
                live_gpu, last_gpu_probe = _maybe_refresh_gpu_info(now, last_gpu_probe, live_gpu)
                telemetry = _extract_telemetry(live_gpu)
                phase_elapsed = max(now - phase_start, 0.01)
                sps = samples_seen / phase_elapsed
                pct = overall_percent_base + overall_percent_range * (0.6 + 0.4 * step / max_steps)
                elapsed = max(now - overall_started_at, 0.01)

                cur_hist = metrics_history + [{"epoch": epoch, "loss": round(a_loss, 6), "acc": round(a_acc, 2), "mae": round(a_mae, 6)}]
                await callback({
                    "type": "train.progress",
                    "payload": {
                        "phase": phase, "stage": stage, "variant": extra_fields.get("variant", ""),
                        "completedPhases": completed_phases or [],
                        "iteration": iteration, "totalIterations": total_iterations,
                        "selfPlayStats": selfplay_stats or {},
                        "step": step, "totalSteps": max_steps,
                        "epoch": epoch, "totalEpochs": max(1, (max_steps * batch_size) // max(n, 1)),
                        "loss": round(a_loss, 6), "accuracy": round(a_acc, 2), "acc": round(a_acc, 2),
                        "mae": round(a_mae, 6),
                        "percent": round(pct, 1),
                        "elapsed": round(elapsed, 1),
                        "eta": round((phase_elapsed / step) * max(max_steps - step, 0), 1),
                        "speed": round(sps, 2), "speedUnit": "samples/s",
                        "samplesPerSec": round(sps, 2),
                        "positions": len(positions),
                        "batchSize": batch_size, "learningRate": 2e-3,
                        "modelParams": model_params,
                        "device": device.type, "deviceName": live_gpu.get("name", str(device)),
                        "mixedPrecision": runtime_flags["mixedPrecision"],
                        "tf32": runtime_flags["tf32"], "channelsLast": runtime_flags["channelsLast"],
                        "metricsHistory": cur_hist, "gpu": live_gpu, **telemetry,
                    },
                })
                last_emit_at = now
                await asyncio.sleep(0)

        # End-of-epoch metrics
        if cum_samples > 0:
            metrics_history.append({
                "epoch": len(metrics_history) + 1,
                "loss": round(cum_loss / cum_samples, 6),
                "acc": round((cum_acc / cum_samples) * 100.0, 2),
                "mae": round(cum_mae / cum_samples, 6),
            })
            cum_loss = cum_acc = cum_mae = 0.0
            cum_samples = 0


# ---------------------------------------------------------------------------
# Curriculum training: Tactical → Supervised → Bootstrap → Self-Play
# ---------------------------------------------------------------------------


async def train_variant(
    variant: str,
    callback: TRAIN_CALLBACK,
    epochs: int = 8,
    batch_size: int = 256,
    data_count: int = 4000,
    **kwargs: Any,
) -> None:
    """Train with multi-phase curriculum: Tactical → Bootstrap → MCTS iterations."""
    from trainer_lab.config import ModelConfig
    from trainer_lab.models.resnet import PolicyValueResNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime_flags = _prepare_cuda_runtime(device)
    cfg = ModelConfig()

    board_size, win_len = _resolve_variant_spec(variant)

    model_dir = _ensure_saved_dir(variant)
    model_path = model_dir / "model.pt"
    is_new_model = not model_path.exists()

    res_filters, res_blocks, value_fc = _variant_model_hparams(board_size, cfg)

    model = PolicyValueResNet(
        in_channels=cfg.in_channels, res_filters=res_filters,
        res_blocks=res_blocks, policy_filters=cfg.policy_filters,
        value_fc=value_fc, board_max=cfg.board_max,
    )
    if model_path.exists():
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            logger.info("Resumed model from %s", model_path)
        except Exception as exc:
            is_new_model = True
            logger.warning("Checkpoint %s is incompatible with current architecture, starting fresh: %s", model_path, exc)
    if device.type == "cuda":
        model = model.to(device=device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    model = _maybe_compile_model(model, device, runtime_flags)

    # Curriculum parameters
    bootstrap_games = int(kwargs.get("bootstrapGames", 64))
    selfplay_iterations = int(kwargs.get("mctsIterations", 3))
    selfplay_games = int(kwargs.get("mctsGamesPerIter", 32))
    tactical_epochs = min(2, epochs)
    supervised_epochs = min(12, epochs) if board_size <= 5 else min(3, epochs)
    bootstrap_steps = 500 if board_size <= 5 else 300
    selfplay_steps_per_iter = 500 if board_size <= 5 else 300

    live_gpu = get_gpu_info()
    await callback({
        "type": "train.start",
        "payload": {
            "variant": variant, "epochs": epochs, "batchSize": batch_size,
            "boardSize": board_size, "winLength": win_len,
            "device": device.type, "deviceName": live_gpu.get("name", str(device)),
            "mixedPrecision": runtime_flags["mixedPrecision"],
            "tf32": runtime_flags["tf32"], "channelsLast": runtime_flags["channelsLast"],
            "learningRate": 2e-3, "modelParams": _count_model_parameters(model),
            "totalIterations": selfplay_iterations, "gpu": live_gpu,
        },
    })

    completed_phases: list[str] = []
    metrics_history: list[dict[str, Any]] = []
    all_positions: list[dict[str, Any]] = []
    tactical_positions: list[dict[str, Any]] = []
    bootstrap_positions: list[dict[str, Any]] = []
    minimax_positions: list[dict[str, Any]] = []
    started_at = time.monotonic()
    common = {"variant": variant, "boardSize": board_size, "winLength": win_len}

    # ── Phase 1: Tactical ──────────────────────────────────────────────
    if is_new_model:
        logger.info("Phase 1: Tactical — generating positions")
        tactical_count = min(data_count // 2, 1000)
        if board_size > 5:
            tactical_positions = await _generate_tactical_curriculum_positions(
                tactical_count, board_size, win_len, callback
            )
        else:
            tactical_positions, _, _ = await _build_positions(variant, tactical_count, callback)
        all_positions.extend(tactical_positions)

        await _run_training_epochs(
            model, tactical_positions, tactical_epochs, batch_size, device, runtime_flags,
            callback, metrics_history,
            phase="tactical", stage="training", completed_phases=completed_phases,
            overall_started_at=started_at, overall_percent_base=0, overall_percent_range=10,
            **common,
        )
        completed_phases.append("tactical")
        torch.save(model.state_dict(), model_path)

    # ── Phase 2: Supervised warm start (offline minimax seed) ─────────
    if is_new_model:
        dataset_path = SAVED_DIR / "datasets" / f"{variant}_minimax.json"
        if not dataset_path.exists():
            logger.info("No offline dataset at %s — auto-generating...", dataset_path)
            from gomoku_api.ws.offline_gen import generate_minimax_dataset
            await generate_minimax_dataset(variant, 5000, callback)
        logger.info("Phase 2: Supervised — loading %s", dataset_path)
        minimax_positions = json.loads(dataset_path.read_text())
        # Shard: 40% tactical + 60% minimax for balanced supervised pool
        tactical_shard_count = min(len(minimax_positions) * 2 // 3, 2000)
        tactical_shard = await _generate_tactical_curriculum_positions(
            tactical_shard_count, board_size, win_len, callback
        )
        supervised_positions = tactical_shard + minimax_positions
        random.shuffle(supervised_positions)
        all_positions.extend(supervised_positions)
        await _run_training_epochs(
            model, all_positions, supervised_epochs, batch_size, device, runtime_flags,
            callback, metrics_history,
            phase="supervised", stage="training", completed_phases=completed_phases,
            overall_started_at=started_at, overall_percent_base=10, overall_percent_range=10,
            **common,
        )
        completed_phases.append("supervised")
        torch.save(model.state_dict(), model_path)
        logger.info("Supervised warm start: %d positions loaded", len(supervised_positions))

    # ── Phase 3: Bootstrap (GPU policy self-play) ─────────────────────
    if is_new_model:
        logger.info("Phase 3: Bootstrap — %d games (policy teacher)", bootstrap_games)
        bootstrap_positions, bootstrap_stats = await _play_selfplay_games_batched(
            bootstrap_games, board_size, win_len, model, device, callback,
            phase="bootstrap", teacher_mode="policy",
            completed_phases=completed_phases,
            overall_percent_base=20, overall_percent_range=10,
            runtime_flags=runtime_flags, **common,
        )
        all_positions.extend(bootstrap_positions)

        await _run_training_steps(
            model, bootstrap_positions, bootstrap_steps, batch_size, device, runtime_flags,
            callback, metrics_history,
            phase="bootstrap", stage="training", completed_phases=completed_phases,
            selfplay_stats=bootstrap_stats,
            overall_started_at=started_at, overall_percent_base=25, overall_percent_range=5,
            **common,
        )
        completed_phases.append("bootstrap")
        torch.save(model.state_dict(), model_path)

    # ── Phase 4: Self-play iterations (GPU policy only) ───────────────
    sp_pct_per_iter = 60.0 / max(selfplay_iterations, 1)
    sp_base_pct = 30.0 if is_new_model else 0.0

    for it in range(1, selfplay_iterations + 1):
        logger.info("Phase 4: Self-play iteration %d/%d — %d games", it, selfplay_iterations, selfplay_games)
        iter_base = sp_base_pct + (it - 1) * sp_pct_per_iter

        sp_positions, sp_stats = await _play_selfplay_games_batched(
            selfplay_games, board_size, win_len, model, device, callback,
            phase="self_play", teacher_mode="policy",
            completed_phases=completed_phases,
            iteration=it, total_iterations=selfplay_iterations,
            overall_percent_base=iter_base, overall_percent_range=sp_pct_per_iter,
            runtime_flags=runtime_flags, **common,
        )
        all_positions.extend(sp_positions)

        # Train on recent + replay
        replay_tail = all_positions[:-len(sp_positions)] if sp_positions else list(all_positions)
        seed_replay = tactical_positions[: min(len(tactical_positions), max(data_count // 4, 256))]
        if bootstrap_positions:
            seed_replay = seed_replay + bootstrap_positions[: min(len(bootstrap_positions), max(data_count // 6, 128))]
        train_pool = _build_train_pool(
            sp_positions,
            replay_tail[-max(data_count, len(replay_tail) // 2):],
            data_count=data_count,
            seed_positions=seed_replay,
            minimax_positions=minimax_positions,
        )
        await _run_training_steps(
            model, train_pool, selfplay_steps_per_iter, batch_size, device, runtime_flags,
            callback, metrics_history,
            phase="self_play", stage="training", completed_phases=completed_phases,
            selfplay_stats=sp_stats,
            iteration=it, total_iterations=selfplay_iterations,
            overall_started_at=started_at,
            overall_percent_base=iter_base + sp_pct_per_iter * 0.5,
            overall_percent_range=sp_pct_per_iter * 0.5,
            **common,
        )
        torch.save(model.state_dict(), model_path)

        # Tactical accuracy eval
        if board_size <= 9:
            tac_acc = _compute_tactical_accuracy(model, board_size, win_len, device)
            logger.info("Self-play %d/%d tactical accuracy: %.1f%%", it, selfplay_iterations, tac_acc)
            await callback({
                "type": "train.progress",
                "payload": {
                    "phase": "self_play", "stage": "eval", "variant": variant,
                    "completedPhases": completed_phases or [],
                    "iteration": it, "totalIterations": selfplay_iterations,
                    "tacticalAccuracy": tac_acc,
                    "message": f"Tactical accuracy: {tac_acc}%",
                },
            })

    # ── Done ───────────────────────────────────────────────────────────
    from gomoku_api.ws.predict_service import clear_cached_model
    clear_cached_model(variant)

    final_elapsed = round(time.monotonic() - started_at, 1)
    await callback({
        "type": "train.done",
        "payload": {
            "variant": variant, "epochs": epochs, "elapsed": final_elapsed,
            "metricsHistory": metrics_history, "device": device.type,
            "positions": len(all_positions),
        },
    })


def clear_model(variant: str = "all") -> dict[str, Any]:
    """Delete saved model files."""
    from gomoku_api.ws.predict_service import clear_cached_model

    variants = ["ttt3", "ttt5"] if variant == "all" else [variant]
    for current_variant in variants:
        model_path = SAVED_DIR / f"{current_variant}_resnet" / "model.pt"
        if model_path.exists():
            model_path.unlink()
            logger.info("Cleared model: %s", model_path)
        clear_cached_model(current_variant)
    return {"cleared": True, "variant": variant}
