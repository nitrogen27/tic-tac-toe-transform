"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict:
    """Liveness probe."""
    engine = request.app.state.engine
    return {"status": "ok", "engineAvailable": engine.available}
