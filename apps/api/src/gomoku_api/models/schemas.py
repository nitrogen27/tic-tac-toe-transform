"""Pydantic v2 models matching packages/shared/schemas/*.schema.json."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EngineSource(str, Enum):
    safety_win = "safety_win"
    safety_block = "safety_block"
    safety_multi_block = "safety_multi_block"
    vcf_win = "vcf_win"
    vcf_defense = "vcf_defense"
    fork = "fork"
    alpha_beta = "alpha_beta"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class TrainPhase(str, Enum):
    tactical = "tactical"
    bootstrap = "bootstrap"
    self_play = "self_play"
    training = "training"
    evaluating = "evaluating"


class ModelFormat(str, Enum):
    onnx = "onnx"
    pytorch = "pytorch"
    checkpoint = "checkpoint"


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

class Position(BaseModel):
    board_size: int = Field(..., ge=7, le=16, alias="boardSize")
    win_length: int = Field(5, ge=5, le=5, alias="winLength")
    current_player: int = Field(..., alias="currentPlayer")
    cells: list[int]
    last_move: int = Field(-1, ge=-1, le=255, alias="lastMove")
    variant: Optional[str] = None

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class MoveCandidate(BaseModel):
    move: int
    score: float
    confidence: Optional[float] = None
    row: Optional[int] = None
    col: Optional[int] = None


class EngineMeta(BaseModel):
    tt_hit_rate: Optional[float] = Field(None, alias="ttHitRate")
    tt_size: Optional[int] = Field(None, alias="ttSize")
    raw_score: Optional[float] = Field(None, alias="rawScore")

    model_config = {"populate_by_name": True}


class AnalyzeRequest(BaseModel):
    position: Position
    top_k: int = Field(5, ge=1, le=25, alias="topK")
    time_limit_ms: int = Field(1000, ge=1, le=60000, alias="timeLimitMs")
    include_pv: bool = Field(True, alias="includePv")

    model_config = {"populate_by_name": True}


class AnalyzeResponse(BaseModel):
    best_move: int = Field(..., alias="bestMove")
    value: float = Field(..., ge=-1, le=1)
    confidence: float = Field(0.0, ge=0, le=1)
    source: EngineSource
    depth: int = Field(0, ge=0)
    nodes_searched: int = Field(0, ge=0, alias="nodesSearched")
    time_ms: float = Field(0, ge=0, alias="timeMs")
    top_moves: list[MoveCandidate] = Field(default_factory=list, alias="topMoves")
    pv_line: list[int] = Field(default_factory=list, alias="pvLine")
    policy: Optional[list[float]] = None
    engine_meta: Optional[EngineMeta] = Field(None, alias="engineMeta")

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Best move (simplified)
# ---------------------------------------------------------------------------

class BestMoveRequest(BaseModel):
    position: Position
    time_limit_ms: int = Field(500, ge=1, le=60000, alias="timeLimitMs")

    model_config = {"populate_by_name": True}


class BestMoveResponse(BaseModel):
    move: int
    row: int
    col: int
    value: float
    source: EngineSource


# ---------------------------------------------------------------------------
# Suggest (top-K hints for UI)
# ---------------------------------------------------------------------------

class SuggestRequest(BaseModel):
    position: Position
    top_k: int = Field(5, ge=1, le=10, alias="topK")

    model_config = {"populate_by_name": True}


class SuggestResponse(BaseModel):
    suggestions: list[MoveCandidate]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainJobConfig(BaseModel):
    variant: str = "gomoku15"
    batch_size: int = Field(256, alias="batchSize")
    epochs: int = 30
    lr: float = 0.001
    self_play_games: int = Field(200, alias="selfPlayGames")
    self_play_simulations: int = Field(400, alias="selfPlaySimulations")
    resume_from_checkpoint: Optional[str] = Field(None, alias="resumeFromCheckpoint")

    model_config = {"populate_by_name": True}


class TrainJobProgress(BaseModel):
    phase: TrainPhase = TrainPhase.tactical
    epoch: int = 0
    total_epochs: int = Field(0, alias="totalEpochs")
    loss: float = 0.0
    policy_accuracy: float = Field(0.0, alias="policyAccuracy")
    value_mae: float = Field(0.0, alias="valueMae")
    games_generated: int = Field(0, alias="gamesGenerated")
    positions_collected: int = Field(0, alias="positionsCollected")
    elapsed_sec: float = Field(0.0, alias="elapsedSec")

    model_config = {"populate_by_name": True}


class TrainJob(BaseModel):
    job_id: str = Field(..., alias="jobId")
    variant: str = "gomoku15"
    status: JobStatus = JobStatus.queued
    config: Optional[TrainJobConfig] = None
    progress: Optional[TrainJobProgress] = None
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: Optional[datetime] = Field(None, alias="updatedAt")
    completed_at: Optional[datetime] = Field(None, alias="completedAt")
    artifact_id: Optional[str] = Field(None, alias="artifactId")
    error: Optional[str] = None

    model_config = {"populate_by_name": True}


class ModelArtifact(BaseModel):
    artifact_id: str = Field(..., alias="artifactId")
    name: str
    version: str
    format: ModelFormat
    board_sizes: list[int] = Field(default_factory=list, alias="boardSizes")
    input_shape: Optional[list[int]] = Field(None, alias="inputShape")
    created_at: datetime = Field(..., alias="createdAt")
    metrics: Optional[dict] = None

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Engine info
# ---------------------------------------------------------------------------

class EngineInfo(BaseModel):
    version: str = "0.1.0"
    supported_board_sizes: list[int] = Field(
        default_factory=lambda: [7, 9, 11, 13, 15],
        alias="supportedBoardSizes",
    )
    capabilities: list[str] = Field(
        default_factory=lambda: ["analyze", "best_move", "suggest"],
    )

    model_config = {"populate_by_name": True}
