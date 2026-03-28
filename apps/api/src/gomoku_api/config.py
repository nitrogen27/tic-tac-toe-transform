"""Application settings loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration for the Gomoku API."""

    engine_binary: str = str(
        Path(__file__).resolve().parents[4] / "engine-core" / "build" / "gomoku_engine"
    )
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]
    log_level: str = "info"

    model_config = {"env_prefix": "GOMOKU_"}


settings = Settings()
