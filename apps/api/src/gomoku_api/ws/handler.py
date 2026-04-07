"""WebSocket message router — legacy Vue.js protocol compatibility layer."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from gomoku_api.ws.game_service import GameService
from gomoku_api.ws.gpu_info import get_gpu_info
from gomoku_api.ws.commentary_service import analyze_move_commentary
from gomoku_api.ws.predict_service import predict
from gomoku_api.ws.offline_gen import generate_engine_dataset, generate_minimax_dataset
from gomoku_api.ws.training_diagnostics import run_training_diagnostics
from gomoku_api.ws.train_service_ws import clear_model, train_variant
from gomoku_api.ws.user_game_corpus import UserGameCorpus, analyze_game, resolve_variant_spec
from gomoku_api.ws.oracle_backends import create_oracle_evaluator

logger = logging.getLogger(__name__)

# Shared game service (one per process)
_game_service = GameService()

# Active training tasks — non-blocking dispatch
_active_train_tasks: dict[str, asyncio.Task] = {}
_corpus_analysis_lock = asyncio.Lock()
_corpus_analysis_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=32)
_corpus_analysis_worker: asyncio.Task | None = None
_queued_analysis_game_ids: set[str] = set()


async def _run_training(variant: str, cb, **kwargs):
    """Wrapper that runs train_variant as a background task."""
    from gomoku_api.ws.training_run_logger import TrainingRunLogger

    run_logger = TrainingRunLogger(variant)

    async def logged_cb(event: dict[str, Any]) -> None:
        run_logger.log(event)
        await cb(event)

    try:
        await train_variant(variant, logged_cb, **kwargs)
    except asyncio.CancelledError:
        logger.info("Training cancelled [%s]", variant)
        event = {"type": "train.cancelled", "payload": {"variant": variant}}
        run_logger.log(event)
        await cb(event)
    except Exception as exc:
        logger.error("Training failed [%s]: %s", variant, exc, exc_info=True)
        event = {"type": "train.error", "payload": {"error": str(exc), "variant": variant}}
        run_logger.log(event)
        await cb(event)
    finally:
        _active_train_tasks.pop(variant, None)


async def _start_training_task(
    ws: WebSocket,
    variant: str,
    *,
    epochs: int,
    batch_size: int,
    data_count: int,
    extra: dict[str, Any] | None = None,
) -> bool:
    if _is_training_active(variant):
        await _send(ws, {"type": "train.rejected", "payload": {"variant": variant, "reason": "already running"}})
        return False
    cb = await _ws_callback(ws)
    task = asyncio.create_task(
        _run_training(
            variant,
            cb,
            epochs=epochs,
            batch_size=batch_size,
            data_count=data_count,
            **(extra or {}),
        )
    )
    _active_train_tasks[variant] = task
    await _send(ws, {"type": "train.accepted", "payload": {"variant": variant}})
    return True


async def _analyze_games_into_corpus(
    variant: str,
    games: list[dict[str, Any]],
    *,
    backend: str | None = None,
) -> dict[str, Any]:
    board_size, win_len = resolve_variant_spec(variant)
    engine_eval, resolved_backend = create_oracle_evaluator(board_size, win_len, backend=backend, role="teacher")
    corpus = UserGameCorpus(variant)
    corpus.load()
    analyzed_positions = 0
    analyzed_games = 0
    per_game: list[tuple[str, int]] = []
    await engine_eval.start()
    try:
        for game in games:
            positions = await analyze_game(game, board_size, win_len, engine_eval)
            if not positions:
                continue
            corpus.ingest_analyzed_game(positions)
            analyzed_games += 1
            analyzed_positions += len(positions)
            per_game.append((str(game.get("gameId", "")), len(positions)))
        if analyzed_games > 0:
            corpus.save()
    finally:
        await engine_eval.stop()

    stats = corpus.stats()
    stats.update({
        "backendResolved": resolved_backend,
        "analyzedGames": analyzed_games,
        "analyzedPositions": analyzed_positions,
        "perGame": per_game,
    })
    return stats


async def _analyze_finished_games(
    ws: WebSocket,
    *,
    variant: str,
    game_id: str | None = None,
    unanalyzed_only: bool = True,
    backend: str | None = None,
) -> dict[str, Any]:
    async with _corpus_analysis_lock:
        if game_id:
            game = _game_service.get_finished_game(game_id)
            if game is None:
                return {"variant": variant, "analyzedGames": 0, "analyzedPositions": 0, "reason": "game_not_found"}
            if unanalyzed_only and bool(game.get("analyzed")):
                stats = UserGameCorpus(variant)
                stats.load()
                payload = stats.stats()
                payload.update({"variant": variant, "analyzedGames": 0, "analyzedPositions": 0, "reason": "already_analyzed"})
                return payload
            games = [game]
        else:
            games = _game_service.get_finished_games(variant=variant, unanalyzed_only=unanalyzed_only)
        if not games:
            stats = UserGameCorpus(variant)
            stats.load()
            payload = stats.stats()
            payload.update({"variant": variant, "analyzedGames": 0, "analyzedPositions": 0})
            return payload

        result = await _analyze_games_into_corpus(variant, games, backend=backend)
        for analyzed_game_id, count in result.get("perGame", []):
            _game_service.mark_game_analyzed(analyzed_game_id, positions=count)
        await _send(ws, {
            "type": "corpus.updated",
            "payload": {
                **{k: v for k, v in result.items() if k != "perGame"},
                "variant": variant,
                "gameId": game_id,
            },
        })
        return result


async def _run_training_after_corpus_analysis(
    ws: WebSocket,
    variant: str,
    *,
    epochs: int,
    batch_size: int,
    data_count: int,
    corpus_mode: str,
    only_unanalyzed: bool = True,
) -> None:
    cb = await _ws_callback(ws)
    try:
        if variant == "ttt5":
            await _analyze_finished_games(ws, variant=variant, unanalyzed_only=only_unanalyzed)
        await _run_training(
            variant,
            cb,
            epochs=epochs,
            batch_size=batch_size,
            data_count=data_count,
            useCorpus=True,
            corpusMode=corpus_mode,
        )
    except asyncio.CancelledError:
        event = {"type": "train.cancelled", "payload": {"variant": variant}}
        await cb(event)
        raise
    except Exception as exc:
        logger.error("Training-with-corpus failed [%s]: %s", variant, exc, exc_info=True)
        await cb({"type": "train.error", "payload": {"error": str(exc), "variant": variant}})
    finally:
        _active_train_tasks.pop(variant, None)


async def _ensure_corpus_analysis_worker() -> None:
    global _corpus_analysis_worker
    if _corpus_analysis_worker is None or _corpus_analysis_worker.done():
        _corpus_analysis_worker = asyncio.create_task(_corpus_analysis_worker_loop())


async def _corpus_analysis_worker_loop() -> None:
    while True:
        job = await _corpus_analysis_queue.get()
        try:
            variant = str(job.get("variant", "ttt5"))
            ws = job["ws"]
            result = await _analyze_finished_games(
                ws,
                variant=variant,
                game_id=job.get("game_id"),
                unanalyzed_only=bool(job.get("unanalyzed_only", True)),
            )
            if bool(job.get("trigger_training")):
                epochs = int(job.get("epochs", 3))
                batch_size = int(job.get("batch_size", 256))
                data_count = int(job.get("data_count", 1000))
                await _start_training_task(
                    ws,
                    variant,
                    epochs=epochs,
                    batch_size=batch_size,
                    data_count=data_count,
                    extra={"useCorpus": True, "corpusMode": "quick_repair"},
                )
            await _send(ws, {
                "type": "corpus.analysis.done",
                "payload": {
                    "variant": variant,
                    "gameId": job.get("game_id"),
                    "analyzedGames": result.get("analyzedGames", 0),
                    "analyzedPositions": result.get("analyzedPositions", 0),
                },
            })
        except Exception as exc:
            logger.error("Corpus analysis failed: %s", exc, exc_info=True)
            ws = job.get("ws")
            if ws is not None:
                await _send(ws, {"type": "error", "error": f"corpus_analysis_failed: {exc}"})
        finally:
            game_id = job.get("game_id")
            if isinstance(game_id, str) and game_id:
                _queued_analysis_game_ids.discard(game_id)
            _corpus_analysis_queue.task_done()


async def _queue_finished_game_analysis(
    ws: WebSocket,
    *,
    game_id: str,
    variant: str,
    trigger_training: bool = False,
    epochs: int = 3,
    batch_size: int = 256,
    data_count: int = 1000,
) -> bool:
    if not game_id or variant != "ttt5":
        return False
    if game_id in _queued_analysis_game_ids:
        return True
    await _ensure_corpus_analysis_worker()
    try:
        _corpus_analysis_queue.put_nowait({
            "ws": ws,
            "game_id": game_id,
            "variant": variant,
            "unanalyzed_only": True,
            "trigger_training": trigger_training,
            "epochs": epochs,
            "batch_size": batch_size,
            "data_count": data_count,
        })
        _queued_analysis_game_ids.add(game_id)
        await _send(ws, {
            "type": "corpus.analysis.queued",
            "payload": {"variant": variant, "gameId": game_id, "triggerTraining": trigger_training},
        })
        return True
    except asyncio.QueueFull:
        await _send(ws, {"type": "error", "error": "corpus_analysis_queue_full"})
        return False


def _is_training_active(variant: str) -> bool:
    task = _active_train_tasks.get(variant)
    return task is not None and not task.done()


async def _send(ws: WebSocket, msg: dict) -> None:
    """Send JSON to websocket, ignoring errors if disconnected."""
    try:
        await ws.send_json(msg)
    except Exception:
        pass


async def _ws_callback(ws: WebSocket):
    """Return a callback that sends events to this websocket."""
    async def cb(event: dict):
        await _send(ws, event)
    return cb


async def ws_handler(websocket: WebSocket) -> None:
    """Handle a single WebSocket client connection (legacy Vue protocol)."""
    await websocket.accept()

    # Send GPU info on connect (matches legacy server behavior)
    gpu = get_gpu_info()
    await _send(websocket, {"type": "gpu.info", "payload": gpu})

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send(websocket, {"type": "error", "error": "invalid_json"})
                continue

            msg_type = msg.get("type", "")
            payload = msg.get("payload", msg)

            try:
                await _dispatch(websocket, msg_type, payload)
            except Exception as exc:
                logger.error("WS handler error [%s]: %s", msg_type, exc, exc_info=True)
                await _send(websocket, {"type": "error", "error": str(exc)})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)


async def _dispatch(ws: WebSocket, msg_type: str, payload: dict) -> None:
    """Route a message to the appropriate handler."""

    if msg_type == "ping":
        await _send(ws, {"type": "pong"})

    elif msg_type == "get_gpu_info":
        await _send(ws, {"type": "gpu.info", "payload": get_gpu_info()})

    elif msg_type == "health":
        gpu = get_gpu_info()
        await _send(ws, {
            "type": "health.result",
            "payload": {"status": "ok", "ws_port": 8080, "gpu": gpu},
        })

    elif msg_type in ("predict",):
        board = payload.get("board", [])
        current = payload.get("current", 1)
        mode = payload.get("mode", "model")
        variant = payload.get("variant", None)
        decision_mode = payload.get("modelDecisionMode", payload.get("decisionMode", "hybrid"))
        result = await predict(board, current, mode, variant, model_decision_mode=decision_mode)
        await _send(ws, {"type": "predict.result", "payload": result})

    elif msg_type == "comment_move":
        result = analyze_move_commentary(
            payload.get("boardBefore", []),
            int(payload.get("move", -1)),
            int(payload.get("current", 1)),
            variant=payload.get("variant"),
            style=str(payload.get("style", "coach")),
            actor=str(payload.get("actor", "player")),
        )
        await _send(ws, {"type": "commentary.result", "payload": result})

    elif msg_type in ("train_ttt3", "train"):
        if _is_training_active("ttt3"):
            await _send(ws, {"type": "train.rejected", "payload": {"variant": "ttt3", "reason": "already running"}})
            return
        epochs = min(max(int(payload.get("epochs", 30)), 1), 50)
        batch_size = min(max(int(payload.get("batchSize", 256)), 32), 4096)
        cb = await _ws_callback(ws)
        task = asyncio.create_task(_run_training("ttt3", cb, epochs=epochs, batch_size=batch_size, data_count=3000))
        _active_train_tasks["ttt3"] = task
        await _send(ws, {"type": "train.accepted", "payload": {"variant": "ttt3"}})

    elif msg_type == "train_ttt5":
        epochs = min(max(int(payload.get("epochs", 30)), 1), 60)
        batch_size = min(max(int(payload.get("batchSize", 256)), 32), 4096)
        extra = {}
        for key in ("bootstrapGames", "mctsIterations", "mctsGamesPerIter", "foundationDatasetCount", "offlineDatasetLimit", "examThresholdAcc", "modelProfile", "teacherBackend", "confirmBackend"):
            if key in payload:
                extra[key] = payload[key]
        if "preferOfflineDataset" in payload:
            extra["preferOfflineDataset"] = payload["preferOfflineDataset"]
        await _start_training_task(ws, "ttt5", epochs=epochs, batch_size=batch_size, data_count=5000, extra=extra)

    elif msg_type == "train_gomoku":
        variant_g = payload.get("variant", "gomoku15")
        if _is_training_active(variant_g):
            await _send(ws, {"type": "train.rejected", "payload": {"variant": variant_g, "reason": "already running"}})
            return
        epochs = min(max(int(payload.get("epochs", 30)), 1), 60)
        batch_size = min(max(int(payload.get("batchSize", 256)), 32), 4096)
        cb = await _ws_callback(ws)
        extra = {}
        for key in ("teacherBackend", "confirmBackend", "modelProfile"):
            if key in payload:
                extra[key] = payload[key]
        task = asyncio.create_task(_run_training(variant_g, cb, epochs=epochs, batch_size=batch_size, data_count=5000, **extra))
        _active_train_tasks[variant_g] = task
        await _send(ws, {"type": "train.accepted", "payload": {"variant": variant_g}})

    elif msg_type == "generate_dataset":
        variant = payload.get("variant", "ttt5")
        count = min(max(int(payload.get("count", 5000)), 100), 50000)
        cb = await _ws_callback(ws)
        path = await generate_minimax_dataset(variant, count, cb)
        await _send(ws, {"type": "dataset.done", "payload": {"path": str(path), "count": count, "variant": variant}})

    elif msg_type == "generate_engine_dataset":
        variant = payload.get("variant", "ttt5")
        count = min(max(int(payload.get("count", 10000)), 100), 100000)
        backend = str(payload.get("backend", "auto"))
        phase_focus = payload.get("phaseFocus", payload.get("phase_focus"))
        cb = await _ws_callback(ws)
        path = await generate_engine_dataset(variant, count, cb, phase_focus=phase_focus, backend=backend)
        await _send(ws, {"type": "dataset.done", "payload": {"path": str(path), "count": count, "variant": variant, "mode": "engine", "backend": backend}})

    elif msg_type == "run_training_diagnostics":
        variant = payload.get("variant", "ttt5")
        report = await run_training_diagnostics(
            variant,
            dataset_limit=min(max(int(payload.get("datasetLimit", 256)), 64), 4096),
            holdout_ratio=min(max(float(payload.get("holdoutRatio", 0.2)), 0.05), 0.4),
            tiny_steps=min(max(int(payload.get("tinySteps", 32)), 8), 256),
            batch_size=min(max(int(payload.get("batchSize", 128)), 16), 1024),
            model_profile=str(payload.get("modelProfile", "auto")),
            include_quick_probe=bool(payload.get("includeQuickProbe", True)),
        )
        await _send(ws, {"type": "training.diagnostics", "payload": report})

    elif msg_type == "clear_model":
        variant = payload.get("variant", "all")
        result = clear_model(variant)
        await _send(ws, {"type": "clear_model.success", "payload": result})

    elif msg_type == "start_game":
        player_role = payload.get("playerRole", 1)
        variant = payload.get("variant", "ttt3")
        game_id = _game_service.start_game(player_role, variant)
        await _send(ws, {"type": "game.started", "payload": {"gameId": game_id}})

    elif msg_type == "save_move":
        stats = _game_service.save_move(
            board=payload.get("board", []),
            move=payload.get("move", -1),
            current=payload.get("current", 1),
            game_id=payload.get("gameId"),
            variant=payload.get("variant", "ttt3"),
        )
        await _send(ws, {"type": "move.saved", "payload": stats})

    elif msg_type == "finish_game":
        stats = _game_service.finish_game(
            game_id=payload.get("gameId", ""),
            winner=payload.get("winner", 0),
        )
        await _send(ws, {"type": "game.finished", "payload": stats})

        variant = payload.get("variant", "ttt3")
        if variant == "ttt5":
            await _queue_finished_game_analysis(
                ws,
                game_id=str(payload.get("gameId", "")),
                variant=variant,
                trigger_training=bool(payload.get("autoTrain")),
                epochs=min(max(int(payload.get("epochs", 3)), 1), 10),
                batch_size=min(max(int(payload.get("incrementalBatchSize", 256)), 32), 1024),
                data_count=1000,
            )
        elif payload.get("autoTrain"):
            await _start_training_task(
                ws,
                variant,
                epochs=min(max(int(payload.get("epochs", 3)), 1), 10),
                batch_size=min(max(int(payload.get("incrementalBatchSize", 256)), 32), 1024),
                data_count=1000,
            )

    elif msg_type == "train_on_games":
        variant = payload.get("variant", "ttt3")
        if _is_training_active(variant):
            await _send(ws, {"type": "train.rejected", "payload": {"variant": variant, "reason": "already running"}})
            return
        epochs = min(max(int(payload.get("epochs", 3)), 1), 10)
        batch_size = min(max(int(payload.get("batchSize", 256)), 32), 1024)
        use_corpus = bool(payload.get("useCorpus", variant == "ttt5"))
        corpus_mode = str(payload.get("corpusMode", "consolidate"))
        if variant == "ttt5" and use_corpus:
            task = asyncio.create_task(
                _run_training_after_corpus_analysis(
                    ws,
                    variant,
                    epochs=epochs,
                    batch_size=batch_size,
                    data_count=2000,
                    corpus_mode=corpus_mode,
                    only_unanalyzed=True,
                )
            )
        else:
            cb = await _ws_callback(ws)
            task = asyncio.create_task(_run_training(variant, cb, epochs=epochs, batch_size=batch_size, data_count=2000))
        _active_train_tasks[variant] = task
        await _send(ws, {"type": "train.accepted", "payload": {"variant": variant}})

    elif msg_type == "analyze_game_corpus":
        variant = payload.get("variant", "ttt5")
        task = asyncio.create_task(
            _analyze_finished_games(
                ws,
                variant=variant,
                unanalyzed_only=bool(payload.get("onlyUnanalyzed", True)),
            )
        )
        await _send(ws, {"type": "corpus.analysis.accepted", "payload": {"variant": variant}})

    elif msg_type == "cancel_training":
        variant = payload.get("variant", "")
        task = _active_train_tasks.get(variant)
        if task and not task.done():
            task.cancel()
            await _send(ws, {"type": "train.cancelled", "payload": {"variant": variant}})
        else:
            await _send(ws, {"type": "error", "error": f"no active training for {variant}"})

    elif msg_type == "clear_history":
        result = _game_service.clear_history()
        await _send(ws, {"type": "history.cleared", "payload": result})

    elif msg_type == "get_history_stats":
        stats = _game_service.get_stats()
        await _send(ws, {"type": "history.stats", "payload": stats})
        await _send(ws, {"type": "gpu.info", "payload": get_gpu_info()})

    else:
        await _send(ws, {"type": "error", "error": f"unknown_type: {msg_type}"})
