"""Persistent C++ engine subprocess for arena evaluation.

Unlike the per-request pattern in engine_adapter.py / predict_service.py,
this keeps a single long-running engine process and sends/receives JSON
over stdin/stdout.  Eliminates subprocess spawn overhead during arena
(300+ moves per arena session).

The C++ engine already supports multi-request mode: it reads JSON lines
from stdin and writes JSON lines to stdout in a loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from gomoku_api.config import settings
from gomoku_api.ws.subprocess_utils import windows_hidden_subprocess_kwargs

logger = logging.getLogger(__name__)


class EngineEvaluator:
    """Persistent engine subprocess for batched arena evaluation."""

    def __init__(self, binary_path: str | None = None) -> None:
        self._binary = binary_path or settings.engine_binary
        self._process: asyncio.subprocess.Process | None = None
        self._request_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the engine subprocess."""
        if self.alive:
            return
        self._process = await asyncio.create_subprocess_exec(
            self._binary,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **windows_hidden_subprocess_kwargs(),
        )
        logger.info("Started persistent engine evaluator: pid=%s", self._process.pid)

    async def stop(self) -> None:
        """Terminate the engine subprocess."""
        if self._process is not None:
            try:
                self._process.stdin.close()  # type: ignore[union-attr]
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                self._process.kill()
            logger.info("Stopped persistent engine evaluator")
            self._process = None

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def _restart(self) -> None:
        logger.warning("Restarting persistent engine evaluator")
        await self.stop()
        await self.start()

    async def _send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send JSON request and read JSON response (single line)."""
        async with self._request_lock:
            attempts = 0
            while attempts < 2:
                attempts += 1
                if not self.alive:
                    await self.start()

                proc = self._process
                assert proc is not None
                assert proc.stdin is not None
                assert proc.stdout is not None

                try:
                    data = json.dumps(payload).encode() + b"\n"
                    proc.stdin.write(data)
                    await proc.stdin.drain()

                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=30)
                    if not line:
                        raise RuntimeError("Engine process returned empty response")

                    return json.loads(line.decode())
                except Exception:
                    if attempts >= 2:
                        raise
                    await self._restart()

            raise RuntimeError("Engine request retry budget exhausted")

    def _build_position_payload(
        self,
        board: list[int],
        current: int,
        board_size: int,
        win_len: int,
    ) -> dict[str, Any]:
        cells = []
        for v in board:
            if v == 1:
                cells.append(1)
            elif v == 2:
                cells.append(-1)
            else:
                cells.append(0)

        return {
            "boardSize": board_size,
            "winLength": win_len,
            "cells": cells,
            "sideToMove": 1 if current == 1 else -1,
            "moveCount": sum(1 for c in cells if c != 0),
            "lastMove": -1,
            "moveHistory": [],
        }

    async def analyze_position(
        self,
        board: list[int],
        current: int,
        board_size: int,
        win_len: int,
    ) -> dict[str, Any]:
        payload = {
            "command": "best-move",
            "position": self._build_position_payload(board, current, board_size, win_len),
        }
        return await self._send_request(payload)

    async def suggest_moves(
        self,
        board: list[int],
        current: int,
        board_size: int,
        win_len: int,
        *,
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        payload = {
            "command": "suggest",
            "position": self._build_position_payload(board, current, board_size, win_len),
            "topN": int(max(1, top_n)),
        }
        try:
            result = await self._send_request(payload)
            hints = result.get("hints", [])
            return [hint for hint in hints if isinstance(hint, dict)]
        except Exception as exc:
            logger.error("Engine suggest_moves failed: %s", exc)
            return []

    async def best_move(
        self,
        board: list[int],
        current: int,
        board_size: int,
        win_len: int,
    ) -> int:
        """Get engine's best move for the given position."""
        try:
            result = await self.analyze_position(board, current, board_size, win_len)
            return result.get("bestMove", -1)
        except Exception as exc:
            logger.error("Engine best_move failed: %s", exc)
            return -1

    async def best_move_with_value(
        self,
        board: list[int],
        current: int,
        board_size: int,
        win_len: int,
    ) -> tuple[int, float]:
        """Get engine's best move plus value for the given position."""
        try:
            result = await self.analyze_position(board, current, board_size, win_len)
            move = int(result.get("bestMove", -1))
            value = float(result.get("value", 0.0))
            value = max(-1.0, min(1.0, value))
            return move, value
        except Exception as exc:
            logger.error("Engine best_move_with_value failed: %s", exc)
            return -1, 0.0

    async def __aenter__(self) -> EngineEvaluator:
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()
