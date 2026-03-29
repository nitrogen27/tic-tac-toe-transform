"""WebSocket message router — legacy Vue.js protocol compatibility layer."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from gomoku_api.ws.game_service import GameService
from gomoku_api.ws.gpu_info import get_gpu_info
from gomoku_api.ws.predict_service import predict
from gomoku_api.ws.offline_gen import generate_minimax_dataset
from gomoku_api.ws.train_service_ws import clear_model, train_variant

logger = logging.getLogger(__name__)

# Shared game service (one per process)
_game_service = GameService()


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
        result = await predict(board, current, mode, variant)
        await _send(ws, {"type": "predict.result", "payload": result})

    elif msg_type in ("train_ttt3", "train"):
        epochs = min(max(int(payload.get("epochs", 30)), 1), 50)
        batch_size = min(max(int(payload.get("batchSize", 256)), 32), 4096)
        cb = await _ws_callback(ws)
        await train_variant("ttt3", cb, epochs=epochs, batch_size=batch_size, data_count=3000)

    elif msg_type == "train_ttt5":
        epochs = min(max(int(payload.get("epochs", 30)), 1), 60)
        batch_size = min(max(int(payload.get("batchSize", 256)), 32), 4096)
        cb = await _ws_callback(ws)
        # Forward preset curriculum params from Vue client
        extra = {}
        for key in ("bootstrapGames", "mctsIterations", "mctsGamesPerIter"):
            if key in payload:
                extra[key] = payload[key]
        await train_variant("ttt5", cb, epochs=epochs, batch_size=batch_size, data_count=5000, **extra)

    elif msg_type == "train_gomoku":
        epochs = min(max(int(payload.get("epochs", 30)), 1), 60)
        batch_size = min(max(int(payload.get("batchSize", 256)), 32), 4096)
        variant = payload.get("variant", "gomoku15")
        cb = await _ws_callback(ws)
        await train_variant(variant, cb, epochs=epochs, batch_size=batch_size, data_count=5000)

    elif msg_type == "generate_dataset":
        variant = payload.get("variant", "ttt5")
        count = min(max(int(payload.get("count", 5000)), 100), 50000)
        cb = await _ws_callback(ws)
        path = await generate_minimax_dataset(variant, count, cb)
        await _send(ws, {"type": "dataset.done", "payload": {"path": str(path), "count": count, "variant": variant}})

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

        # Optional auto-train
        if payload.get("autoTrain"):
            variant = payload.get("variant", "ttt3")
            cb = await _ws_callback(ws)
            epochs = min(max(int(payload.get("epochs", 3)), 1), 10)
            batch_size = min(max(int(payload.get("incrementalBatchSize", 256)), 32), 1024)
            await train_variant(variant, cb, epochs=epochs, batch_size=batch_size, data_count=1000)

    elif msg_type == "train_on_games":
        variant = payload.get("variant", "ttt3")
        epochs = min(max(int(payload.get("epochs", 3)), 1), 10)
        batch_size = min(max(int(payload.get("batchSize", 256)), 32), 1024)
        cb = await _ws_callback(ws)
        await train_variant(variant, cb, epochs=epochs, batch_size=batch_size, data_count=2000)

    elif msg_type == "clear_history":
        result = _game_service.clear_history()
        await _send(ws, {"type": "history.cleared", "payload": result})

    elif msg_type == "get_history_stats":
        stats = _game_service.get_stats()
        await _send(ws, {"type": "history.stats", "payload": stats})
        await _send(ws, {"type": "gpu.info", "payload": get_gpu_info()})

    else:
        await _send(ws, {"type": "error", "error": f"unknown_type: {msg_type}"})
