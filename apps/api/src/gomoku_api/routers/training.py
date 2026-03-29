"""Training job endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import Response

from gomoku_api.models.schemas import TrainJob, TrainJobConfig

router = APIRouter(prefix="/train", tags=["training"])


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
