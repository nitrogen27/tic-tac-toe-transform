"""Training job endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from gomoku_api.models.schemas import TrainJob, TrainJobConfig

router = APIRouter(prefix="/train", tags=["training"])


def _train_service(request: Request):
    return request.app.state.train_service


@router.post("/jobs", response_model=TrainJob, status_code=201)
async def create_job(config: TrainJobConfig, request: Request) -> TrainJob:
    """Create a new training job."""
    return _train_service(request).create_job(config)


@router.get("/jobs/{job_id}", response_model=TrainJob)
async def get_job(job_id: str, request: Request) -> TrainJob:
    """Get training job status."""
    job = _train_service(request).get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job
