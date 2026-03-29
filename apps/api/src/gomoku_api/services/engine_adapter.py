"""Adapter that shells out to the C++ Gomoku engine CLI binary."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import shutil
from dataclasses import dataclass, field
from typing import Any

from gomoku_api.config import settings
from gomoku_api.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BestMoveRequest,
    BestMoveResponse,
    EngineInfo,
    EngineSource,
    MoveCandidate,
    Position,
    SuggestRequest,
    SuggestResponse,
)

logger = logging.getLogger(__name__)


@dataclass
class EngineAdapter:
    """Communicates with the native gomoku engine via subprocess JSON protocol."""

    binary_path: str = field(default_factory=lambda: settings.engine_binary)
    _available: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        binary = Path(self.binary_path)
        self._available = binary.is_file() or shutil.which(self.binary_path) is not None

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Low-level subprocess call
    # ------------------------------------------------------------------

    async def execute(
        self,
        command: str,
        position: Position,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a JSON command to the engine binary via stdin and read JSON from stdout."""
        # Build position payload matching the C++ CLI protocol
        pos_data = position.model_dump(by_alias=True)
        cli_position = {
            "boardSize": pos_data["boardSize"],
            "winLength": pos_data["winLength"],
            "cells": pos_data["cells"],
            "sideToMove": pos_data["currentPlayer"],
            "moveCount": sum(1 for c in pos_data["cells"] if c != 0),
            "lastMove": pos_data.get("lastMove", -1),
            "moveHistory": [],
        }
        payload = {
            "command": command,
            "position": cli_position,
        }
        if options:
            payload["options"] = options
            payload.update(options)
        stdin_bytes = json.dumps(payload).encode()

        if not self._available:
            logger.warning("Engine binary not found at %s — returning fallback.", self.binary_path)
            return self._fallback(command, position)

        try:
            proc = await asyncio.create_subprocess_exec(
                self.binary_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(stdin_bytes), timeout=30)
            if proc.returncode != 0:
                logger.error("Engine exited %d: %s", proc.returncode, stderr.decode(errors="replace"))
                return self._fallback(command, position)
            return json.loads(stdout.decode())
        except (asyncio.TimeoutError, FileNotFoundError, json.JSONDecodeError) as exc:
            logger.error("Engine call failed: %s", exc)
            return self._fallback(command, position)

    # ------------------------------------------------------------------
    # Fallback: centre move
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback(command: str, position: Position) -> dict[str, Any]:
        centre = (position.board_size * position.board_size) // 2
        row = centre // position.board_size
        col = centre % position.board_size
        return {
            "bestMove": centre,
            "value": 0.0,
            "confidence": 0.0,
            "source": "alpha_beta",
            "depth": 0,
            "nodesSearched": 0,
            "timeMs": 0,
            "topMoves": [{"move": centre, "score": 0.0, "row": row, "col": col}],
            "pvLine": [centre],
        }

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_source(source: str) -> str:
        """Map engine source strings (e.g. 'alpha-beta') to enum values ('alpha_beta')."""
        return source.replace("-", "_")

    async def analyze(self, req: AnalyzeRequest) -> AnalyzeResponse:
        raw = await self.execute(
            "analyze",
            req.position,
            {"topK": req.top_k, "timeLimitMs": req.time_limit_ms, "includePv": req.include_pv},
        )
        raw["source"] = self._normalize_source(raw.get("source", "alpha_beta"))
        raw.setdefault("confidence", 0.0)
        raw["timeMs"] = int(raw.get("timeMs", 0))
        raw.setdefault("topMoves", raw.get("hints", []))
        return AnalyzeResponse(**raw)

    async def best_move(self, req: BestMoveRequest) -> BestMoveResponse:
        raw = await self.execute(
            "best-move",
            req.position,
            {"timeLimitMs": req.time_limit_ms},
        )
        move = raw["bestMove"]
        bs = req.position.board_size
        return BestMoveResponse(
            move=move,
            row=move // bs,
            col=move % bs,
            value=raw.get("value", 0.0),
            source=EngineSource(self._normalize_source(raw.get("source", "alpha_beta"))),
        )

    async def suggest(self, req: SuggestRequest) -> SuggestResponse:
        raw = await self.execute(
            "suggest",
            req.position,
            {"topN": req.top_k},
        )
        candidates = [
            MoveCandidate(**m)
            for m in raw.get("hints", raw.get("topMoves", raw.get("suggestions", [])))
        ]
        return SuggestResponse(suggestions=candidates)

    def info(self) -> EngineInfo:
        return EngineInfo()
