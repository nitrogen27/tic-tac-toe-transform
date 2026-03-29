"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings


def _default_engine_binary() -> str:
    root = Path(__file__).resolve().parents[4] / "engine-core"
    candidates = [
        root / "build" / "Release" / "gomoku-engine.exe",
        root / "build" / "gomoku-engine.exe",
        root / "build" / "Release" / "gomoku-engine",
        root / "build" / "gomoku-engine",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0] if os.name == "nt" else candidates[2])


class Settings(BaseSettings):
    """Runtime configuration for the Gomoku API."""

    engine_binary: str = _default_engine_binary()
    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: list[str] = ["*"]
    log_level: str = "info"

    model_config = {"env_prefix": "GOMOKU_"}


settings = Settings()
