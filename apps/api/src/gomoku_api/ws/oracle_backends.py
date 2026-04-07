"""Factory helpers for selecting oracle backends.

Backends:
- builtin: existing JSON-line engine evaluator
- rapfi: optional Rapfi subprocess adapter for standard Gomoku rules
"""

from __future__ import annotations

import logging
from typing import Any

from gomoku_api.config import settings
from gomoku_api.ws.engine_evaluator import EngineEvaluator
from gomoku_api.ws.rapfi_adapter import RapfiAdapter, rapfi_supports_variant

logger = logging.getLogger(__name__)


def _auto_oracle_backend(board_size: int, win_len: int) -> str:
    if settings.rapfi_enabled and settings.rapfi_binary and rapfi_supports_variant(board_size, win_len):
        return "rapfi"
    return "builtin"


def normalize_oracle_backend(name: str | None, *, role: str = "teacher", board_size: int | None = None, win_len: int | None = None) -> str:
    raw = (name or "").strip().lower()
    if raw in {"", "default"}:
        if role == "confirm":
            raw = str(settings.default_confirm_backend or "auto").strip().lower()
        else:
            raw = str(settings.default_teacher_backend or "auto").strip().lower()
    if raw == "auto":
        if board_size is None or win_len is None:
            return "builtin"
        return _auto_oracle_backend(board_size, win_len)
    if raw in {"engine", "builtin", "cpp"}:
        return "builtin"
    if raw in {"rapfi"}:
        return "rapfi"
    return "builtin"


def create_oracle_evaluator(
    board_size: int,
    win_len: int,
    *,
    backend: str | None = None,
    role: str = "teacher",
) -> tuple[Any, str]:
    """Build the requested oracle evaluator with safe fallback."""
    requested = normalize_oracle_backend(backend, role=role, board_size=board_size, win_len=win_len)
    if requested == "rapfi":
        if not settings.rapfi_enabled:
            logger.warning("Rapfi requested for %s but GOMOKU_RAPFI_ENABLED is false; falling back to builtin", role)
            return EngineEvaluator(), "builtin"
        if not settings.rapfi_binary:
            logger.warning("Rapfi requested for %s but no GOMOKU_RAPFI_BINARY configured; falling back to builtin", role)
            return EngineEvaluator(), "builtin"
        if not rapfi_supports_variant(board_size, win_len):
            logger.warning(
                "Rapfi requested for %s but unsupported rules board_size=%s win_len=%s; falling back to builtin",
                role,
                board_size,
                win_len,
            )
            return EngineEvaluator(), "builtin"
        return RapfiAdapter(), "rapfi"
    return EngineEvaluator(), "builtin"
