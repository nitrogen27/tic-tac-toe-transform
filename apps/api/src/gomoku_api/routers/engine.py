"""Engine analysis endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from gomoku_api.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BestMoveRequest,
    BestMoveResponse,
    EngineInfo,
    SuggestRequest,
    SuggestResponse,
)

router = APIRouter(tags=["engine"])


def _engine(request: Request):
    return request.app.state.engine


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest, request: Request) -> AnalyzeResponse:
    """Run full analysis on a board position."""
    return await _engine(request).analyze(body)


@router.post("/best-move", response_model=BestMoveResponse)
async def best_move(body: BestMoveRequest, request: Request) -> BestMoveResponse:
    """Return the single best move quickly."""
    return await _engine(request).best_move(body)


@router.post("/suggest", response_model=SuggestResponse)
async def suggest(body: SuggestRequest, request: Request) -> SuggestResponse:
    """Return top-K move suggestions for UI hints."""
    return await _engine(request).suggest(body)


@router.get("/engine/info", response_model=EngineInfo)
async def engine_info(request: Request) -> EngineInfo:
    """Return engine version and capabilities."""
    return _engine(request).info()
