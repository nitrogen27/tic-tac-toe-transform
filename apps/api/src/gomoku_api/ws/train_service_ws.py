"""Training service for WebSocket streaming — uses PyTorch via trainer-lab."""

from __future__ import annotations

import asyncio
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


def _prepare_cuda_runtime(device: torch.device) -> dict[str, bool]:
    enabled = {"mixedPrecision": False, "tf32": False, "channelsLast": False}
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
    use_minimax = board_size <= 5  # Minimax for small boards only
    minimax_depth = 4 if board_size <= 3 else 3  # depth 3 for 5x5 (fast enough)

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

            # Get policy: minimax for small boards, random for large
            if use_minimax:
                mm_policy, mm_value = _nxn_minimax_policy(
                    list(board), board_size, win_len, current_player, minimax_depth
                )
                policy = [0.0] * 256
                for idx in range(board_size * board_size):
                    r, c = divmod(idx, board_size)
                    policy[r * 16 + c] = mm_policy[idx]
                pos_value = mm_value
            else:
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

            # Choose move based on policy (weighted sample for diversity)
            if use_minimax:
                weights = [mm_policy[i] for i in empty]
                total_w = sum(weights)
                if total_w > 0:
                    move = random.choices(empty, weights=weights, k=1)[0]
                else:
                    move = random.choice(empty)
            else:
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
# Self-play game generation (bootstrap + MCTS)
# ---------------------------------------------------------------------------


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
            elif board_size <= 5:
                # Small boards: use minimax policy for quality targets
                mm_pol, _ = _nxn_minimax_policy(list(board), board_size, win_len, current, 3)
                policy = [0.0] * 256
                for idx in range(board_size * board_size):
                    r, c = divmod(idx, board_size)
                    policy[r * 16 + c] = mm_pol[idx]
            else:
                # Large boards: uniform (fallback)
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
        if (g + 1) % emit_interval == 0 or g + 1 == num_games:
            elapsed = time.monotonic() - started_at
            game_pct = (g + 1) / num_games
            speed = (g + 1) / max(elapsed, 0.01)
            pct = overall_percent_base + overall_percent_range * game_pct * 0.6

            live_gpu = get_gpu_info()
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
                    "gpu": live_gpu, **telemetry,
                    **{k: v for k, v in extra_fields.items() if k not in ("variant",)},
                },
            })
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
    live_gpu = get_gpu_info()
    phase_start = time.monotonic()

    for epoch in range(1, num_epochs + 1):
        ep_loss = ep_ploss = ep_vloss = ep_acc = ep_mae = 0.0
        ep_samples = 0

        for bi, (planes, pol_t, val_t) in enumerate(loader, 1):
            step_t = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.synchronize()

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

            if device.type == "cuda":
                torch.cuda.synchronize()
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
            if now - last_gpu_probe >= 1.0:
                live_gpu = get_gpu_info(); last_gpu_probe = now
            telemetry = _extract_telemetry(live_gpu)

            cur_hist = metrics_history + [{"epoch": epoch, "loss": round(a_loss, 6),
                                           "acc": round(a_acc, 2), "mae": round(a_mae, 6)}]
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
                    "metricsHistory": cur_hist, "gpu": live_gpu, **telemetry,
                },
            })
            await asyncio.sleep(0)

        metrics_history.append({
            "epoch": len(metrics_history) + 1,
            "loss": round(ep_loss / max(ep_samples, 1), 6),
            "acc": round((ep_acc / max(ep_samples, 1)) * 100.0, 2),
            "mae": round(ep_mae / max(ep_samples, 1), 6),
        })


# ---------------------------------------------------------------------------
# Curriculum training: Tactical → Bootstrap → MCTS
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

    # Use smaller model for small boards (3x3, 5x5) — large ResNet overfits
    if board_size <= 5:
        res_filters, res_blocks, value_fc = 32, 3, 64
    elif board_size <= 9:
        res_filters, res_blocks, value_fc = 64, 4, 128
    else:
        res_filters, res_blocks, value_fc = cfg.res_filters, cfg.res_blocks, cfg.value_fc

    model = PolicyValueResNet(
        in_channels=cfg.in_channels, res_filters=res_filters,
        res_blocks=res_blocks, policy_filters=cfg.policy_filters,
        value_fc=value_fc, board_max=cfg.board_max,
    )
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        logger.info("Resumed model from %s", model_path)
    if device.type == "cuda":
        model = model.to(device=device, memory_format=torch.channels_last)
    else:
        model = model.to(device)

    # Curriculum parameters
    bootstrap_games = int(kwargs.get("bootstrapGames", 40))
    mcts_iterations = int(kwargs.get("mctsIterations", 3))
    mcts_games = int(kwargs.get("mctsGamesPerIter", 20))
    mcts_sims = int(kwargs.get("mctsTrainingSims", 16))
    tactical_epochs = min(2, epochs)
    bootstrap_epochs = min(3, epochs)
    mcts_epochs_per_iter = min(3, epochs)

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
            "totalIterations": mcts_iterations, "gpu": live_gpu,
        },
    })

    completed_phases: list[str] = []
    metrics_history: list[dict[str, Any]] = []
    all_positions: list[dict[str, Any]] = []
    started_at = time.monotonic()
    common = {"variant": variant, "boardSize": board_size, "winLength": win_len}

    # ── Phase 1: Tactical ──────────────────────────────────────────────
    if is_new_model:
        logger.info("Phase 1: Tactical — generating positions")
        tactical_count = min(data_count // 2, 1000)
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

    # ── Phase 2: Bootstrap ─────────────────────────────────────────────
    if is_new_model:
        logger.info("Phase 2: Bootstrap — %d games", bootstrap_games)
        bootstrap_positions, bootstrap_stats = await _play_selfplay_games(
            bootstrap_games, board_size, win_len, model, device, callback,
            phase="bootstrap", use_mcts=False,
            completed_phases=completed_phases,
            overall_percent_base=10, overall_percent_range=20,
            runtime_flags=runtime_flags, **common,
        )
        all_positions.extend(bootstrap_positions)

        await _run_training_epochs(
            model, bootstrap_positions, bootstrap_epochs, batch_size, device, runtime_flags,
            callback, metrics_history,
            phase="bootstrap", stage="training", completed_phases=completed_phases,
            selfplay_stats=bootstrap_stats,
            overall_started_at=started_at, overall_percent_base=20, overall_percent_range=10,
            **common,
        )
        completed_phases.append("bootstrap")
        torch.save(model.state_dict(), model_path)

    # ── Phase 3: MCTS iterations ───────────────────────────────────────
    mcts_pct_per_iter = 60.0 / max(mcts_iterations, 1)
    mcts_base_pct = 30.0 if is_new_model else 0.0

    for it in range(1, mcts_iterations + 1):
        logger.info("Phase 3: MCTS iteration %d/%d — %d games", it, mcts_iterations, mcts_games)
        iter_base = mcts_base_pct + (it - 1) * mcts_pct_per_iter

        # Self-play with MCTS
        mcts_positions, mcts_stats = await _play_selfplay_games(
            mcts_games, board_size, win_len, model, device, callback,
            phase="mcts", use_mcts=True, mcts_simulations=mcts_sims,
            completed_phases=completed_phases,
            iteration=it, total_iterations=mcts_iterations,
            overall_percent_base=iter_base, overall_percent_range=mcts_pct_per_iter,
            runtime_flags=runtime_flags, **common,
        )
        all_positions.extend(mcts_positions)

        # Train on recent + replay
        train_pool = (mcts_positions + all_positions[-len(all_positions)//2:])[:data_count]
        await _run_training_epochs(
            model, train_pool, mcts_epochs_per_iter, batch_size, device, runtime_flags,
            callback, metrics_history,
            phase="mcts", stage="mcts_train", completed_phases=completed_phases,
            selfplay_stats=mcts_stats,
            iteration=it, total_iterations=mcts_iterations,
            overall_started_at=started_at,
            overall_percent_base=iter_base + mcts_pct_per_iter * 0.5,
            overall_percent_range=mcts_pct_per_iter * 0.5,
            **common,
        )
        torch.save(model.state_dict(), model_path)

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
