"""FastAPI application factory with WebSocket support for legacy Vue client."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from gomoku_api.config import settings
from gomoku_api.routers import engine_router, health_router, training_router
from gomoku_api.services.engine_adapter import EngineAdapter
from gomoku_api.services.train_service import TrainService
from gomoku_api.ws.handler import ws_handler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared services on startup."""
    app.state.engine = EngineAdapter()
    app.state.train_service = TrainService()
    yield


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    app = FastAPI(
        title="Gomoku Platform API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(engine_router)
    app.include_router(training_router)

    @app.websocket("/")
    async def websocket_root(websocket: WebSocket):
        """Legacy Vue.js client WebSocket endpoint (ws://host:port/)."""
        await ws_handler(websocket)

    return app


app = create_app()
