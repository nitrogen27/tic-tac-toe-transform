"""In-memory training job management (stub for Phase 1)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from gomoku_api.models.schemas import (
    JobStatus,
    TrainJob,
    TrainJobConfig,
    TrainJobProgress,
    TrainPhase,
)


class TrainService:
    """Manages training jobs in an in-memory store."""

    def __init__(self) -> None:
        self._jobs: dict[str, TrainJob] = {}

    def create_job(self, config: TrainJobConfig) -> TrainJob:
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        job = TrainJob(
            jobId=job_id,
            variant=config.variant,
            status=JobStatus.queued,
            config=config,
            progress=TrainJobProgress(phase=TrainPhase.tactical),
            createdAt=now,
            updatedAt=now,
        )
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[TrainJob]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[TrainJob]:
        return list(self._jobs.values())
