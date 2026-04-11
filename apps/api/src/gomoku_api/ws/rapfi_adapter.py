"""Optional Rapfi oracle adapter.

This adapter integrates Rapfi as a stronger external oracle for standard
Gomoku-style boards (win_len=5). It currently supports a conservative
Piskvork-compatible move-query flow and gracefully degrades advanced calls
like suggest/value to best-move-only responses.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
from pathlib import Path
from typing import Any

from gomoku_api.config import settings
from gomoku_api.ws.subprocess_utils import windows_hidden_subprocess_kwargs

logger = logging.getLogger(__name__)

_MOVE_RE = re.compile(r"^\s*(-?\d+)\s*[, ]\s*(-?\d+)\s*$")


def rapfi_supports_variant(board_size: int, win_len: int) -> bool:
    """Rapfi is intended for standard Gomoku-like rules with five-in-a-row."""
    return board_size >= 5 and win_len == 5


class RapfiAdapter:
    """Best-move oracle wrapper for a Rapfi subprocess.

    The baseline integration uses the widely supported Piskvork text protocol.
    Advanced analysis fields are currently approximated from the best move so
    that the rest of the training/eval pipeline can already consume Rapfi as an
    optional backend without breaking the existing JSON-based interface.
    """

    def __init__(
        self,
        binary_path: str | None = None,
        *,
        extra_args: str | None = None,
        workdir: str | None = None,
    ) -> None:
        self._binary = binary_path or settings.rapfi_binary
        self._extra_args = shlex.split(extra_args if extra_args is not None else settings.rapfi_args)
        default_workdir = str(Path(self._binary).resolve().parent) if self._binary else None
        self._workdir = workdir or settings.rapfi_workdir or default_workdir
        self._process: asyncio.subprocess.Process | None = None
        self._request_lock = asyncio.Lock()
        self._board_size: int | None = None

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def start(self) -> None:
        if self.alive:
            return
        if not self._binary:
            raise RuntimeError("Rapfi binary is not configured")
        self._process = await asyncio.create_subprocess_exec(
            self._binary,
            *self._extra_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._workdir or None,
            **windows_hidden_subprocess_kwargs(),
        )
        self._board_size = None
        logger.info("Started Rapfi adapter: pid=%s binary=%s", self._process.pid, self._binary)

    async def stop(self) -> None:
        if self._process is None:
            return
        proc = self._process
        try:
            if proc.stdin is not None:
                try:
                    proc.stdin.write(b"END\n")
                    await proc.stdin.drain()
                except Exception:
                    pass
                proc.stdin.close()
            await asyncio.wait_for(proc.wait(), timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        finally:
            self._process = None
            self._board_size = None

    async def _restart(self) -> None:
        await self.stop()
        await self.start()

    async def _write_line(self, line: str) -> None:
        proc = self._process
        assert proc is not None and proc.stdin is not None
        proc.stdin.write((line.rstrip() + "\n").encode("utf-8"))
        await proc.stdin.drain()

    async def _read_until(self, *, expect_ok: bool = False, expect_move: bool = False) -> tuple[int, int] | None:
        proc = self._process
        assert proc is not None and proc.stdout is not None
        deadline = asyncio.get_running_loop().time() + 15.0
        while asyncio.get_running_loop().time() < deadline:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=15)
            if not line:
                raise RuntimeError("Rapfi returned empty response")
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            if expect_ok and text.upper().startswith("OK"):
                return None
            if expect_move:
                match = _MOVE_RE.match(text)
                if match:
                    return int(match.group(1)), int(match.group(2))
            # Ignore info / banner lines.
            if text.upper().startswith("MESSAGE") or text.upper().startswith("INFO") or text.upper().startswith("DEBUG"):
                continue
        raise RuntimeError("Timed out waiting for Rapfi response")

    async def _ensure_session(self, board_size: int) -> None:
        if not self.alive:
            await self.start()
        try:
            if self._board_size == board_size:
                await self._write_line("RESTART")
                await self._read_until(expect_ok=True)
                return
        except Exception:
            logger.debug("Rapfi RESTART failed, recreating process", exc_info=True)
        await self._restart()
        await self._write_line(f"START {board_size}")
        await self._read_until(expect_ok=True)
        self._board_size = board_size

    async def _query_best_move(
        self,
        board: list[int],
        current: int,
        board_size: int,
        win_len: int,
    ) -> int:
        if not rapfi_supports_variant(board_size, win_len):
            raise RuntimeError(f"Rapfi does not support board_size={board_size}, win_len={win_len}")

        async with self._request_lock:
            await self._ensure_session(board_size)
            await self._write_line("BOARD")
            for idx, cell in enumerate(board):
                if cell == 0:
                    continue
                row, col = divmod(idx, board_size)
                player = 1 if cell == 1 else 2
                # Piskvork uses x,y where x=column, y=row.
                await self._write_line(f"{col},{row},{player}")
            await self._write_line("DONE")
            coords = await self._read_until(expect_move=True)
            if coords is None:
                return -1
            col, row = coords
            if not (0 <= row < board_size and 0 <= col < board_size):
                return -1
            move = row * board_size + col
            if move < 0 or move >= len(board) or board[move] != 0:
                return -1
            return move

    async def analyze_position(
        self,
        board: list[int],
        current: int,
        board_size: int,
        win_len: int,
    ) -> dict[str, Any]:
        move = await self._query_best_move(board, current, board_size, win_len)
        return {
            "bestMove": move,
            "value": 0.0,
            "backend": "rapfi",
        }

    async def suggest_moves(
        self,
        board: list[int],
        current: int,
        board_size: int,
        win_len: int,
        *,
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        move = await self._query_best_move(board, current, board_size, win_len)
        if move < 0:
            return []
        return [{"move": move, "score": 0.0, "rank": 1, "backend": "rapfi"}]

    async def best_move(
        self,
        board: list[int],
        current: int,
        board_size: int,
        win_len: int,
    ) -> int:
        try:
            return await self._query_best_move(board, current, board_size, win_len)
        except Exception as exc:
            logger.error("Rapfi best_move failed: %s", exc)
            return -1

    async def best_move_with_value(
        self,
        board: list[int],
        current: int,
        board_size: int,
        win_len: int,
    ) -> tuple[int, float]:
        move = await self.best_move(board, current, board_size, win_len)
        return move, 0.0

    async def __aenter__(self) -> RapfiAdapter:
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()
