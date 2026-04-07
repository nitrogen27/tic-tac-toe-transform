"""Training job endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import Response

from gomoku_api.models.schemas import TrainJob, TrainJobConfig
from gomoku_api.ws.offline_gen import generate_engine_dataset
from gomoku_api.ws.training_diagnostics import run_training_diagnostics

router = APIRouter(prefix="/train", tags=["training"])


class EngineDatasetRequest(BaseModel):
    variant: str = "ttt5"
    count: int = Field(default=10_000, ge=100, le=100_000)
    backend: str = Field(default="auto")
    phase_focus: str | None = Field(default=None, alias="phaseFocus")

    model_config = {"populate_by_name": True}


class DiagnosticsRequest(BaseModel):
    variant: str = "ttt5"
    dataset_limit: int = Field(default=256, ge=64, le=4096, alias="datasetLimit")
    holdout_ratio: float = Field(default=0.20, ge=0.05, le=0.4, alias="holdoutRatio")
    tiny_steps: int = Field(default=32, ge=8, le=256, alias="tinySteps")
    batch_size: int = Field(default=128, ge=16, le=1024, alias="batchSize")
    model_profile: str = Field(default="auto", alias="modelProfile")
    include_quick_probe: bool = Field(default=True, alias="includeQuickProbe")

    model_config = {"populate_by_name": True}


def _train_service(request: Request):
    return request.app.state.train_service


@router.post("/jobs", response_model=TrainJob, status_code=201)
async def create_job(config: TrainJobConfig, request: Request) -> TrainJob:
    """Create and start a new training job."""
    svc = _train_service(request)
    job = svc.create_job(config)
    await svc.start_job(job.job_id)
    return svc.get_job(job.job_id)


@router.get("/jobs", response_model=list[TrainJob])
async def list_jobs(request: Request) -> list[TrainJob]:
    """List all training jobs."""
    return _train_service(request).list_jobs()


@router.get("/jobs/{job_id}", response_model=TrainJob)
async def get_job(job_id: str, request: Request) -> TrainJob:
    """Get training job status."""
    job = _train_service(request).get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@router.delete("/jobs/{job_id}", status_code=204)
async def cancel_job(job_id: str, request: Request) -> Response:
    """Cancel a running training job."""
    svc = _train_service(request)
    job = svc.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if not svc.cancel_job(job_id):
        raise HTTPException(status_code=409, detail="Job is not running")
    return Response(status_code=204)


@router.post("/datasets/engine")
async def create_engine_dataset(config: EngineDatasetRequest) -> dict:
    path = await generate_engine_dataset(config.variant, config.count, phase_focus=config.phase_focus, backend=config.backend)
    return {
        "variant": config.variant,
        "count": config.count,
        "mode": "engine",
        "backend": config.backend,
        "path": str(path),
    }


@router.post("/diagnostics")
async def run_diagnostics(config: DiagnosticsRequest) -> dict:
    return await run_training_diagnostics(
        config.variant,
        dataset_limit=config.dataset_limit,
        holdout_ratio=config.holdout_ratio,
        tiny_steps=config.tiny_steps,
        batch_size=config.batch_size,
        model_profile=config.model_profile,
        include_quick_probe=config.include_quick_probe,
    )
