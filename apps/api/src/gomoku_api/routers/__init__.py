"""API routers."""

from gomoku_api.routers.engine import router as engine_router
from gomoku_api.routers.health import router as health_router
from gomoku_api.routers.training import router as training_router

__all__ = ["engine_router", "health_router", "training_router"]
