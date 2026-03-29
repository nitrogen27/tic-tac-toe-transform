"""In-memory training job management with async background execution."""

from __future__ import annotations

import asyncio
import logging
import time
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

logger = logging.getLogger(__name__)

_PHASE_SEQUENCE = [
    TrainPhase.tactical,
    TrainPhase.bootstrap,
    TrainPhase.self_play,
    TrainPhase.training,
    TrainPhase.evaluating,
]


class TrainService:
    """Manages training jobs in an in-memory store."""

    def __init__(self) -> None:
        self._jobs: dict[str, TrainJob] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def create_job(self, config: TrainJobConfig) -> TrainJob:
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        job = TrainJob(
            jobId=job_id,
            variant=config.variant,
            status=JobStatus.queued,
            config=config,
            progress=TrainJobProgress(
                phase=TrainPhase.tactical,
                totalEpochs=config.epochs,
            ),
            createdAt=now,
            updatedAt=now,
        )
        self._jobs[job_id] = job
        return job

    async def start_job(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.running
        job.updated_at = datetime.now(timezone.utc)
        task = asyncio.create_task(self._run_job(job_id))
        self._tasks[job_id] = task

    async def _run_job(self, job_id: str) -> None:
        """Background worker — iterates through training phases.

        This is a stub that simulates progress. Real training integration
        will be added in Phase 6 via trainer-lab.
        """
        job = self._jobs[job_id]
        assert job.progress is not None
        assert job.config is not None
        start = time.monotonic()

        try:
            for phase in _PHASE_SEQUENCE:
                job.progress.phase = phase
                job.updated_at = datetime.now(timezone.utc)

                if phase == TrainPhase.training:
                    for epoch in range(1, job.config.epochs + 1):
                        await asyncio.sleep(0.5)
                        job.progress.epoch = epoch
                        job.progress.elapsed_sec = time.monotonic() - start
                        job.updated_at = datetime.now(timezone.utc)
                else:
                    await asyncio.sleep(0.5)

                job.progress.elapsed_sec = time.monotonic() - start

            job.status = JobStatus.completed
            job.completed_at = datetime.now(timezone.utc)
            job.updated_at = job.completed_at
            logger.info("Job %s completed in %.1fs", job_id, job.progress.elapsed_sec)

        except asyncio.CancelledError:
            job.status = JobStatus.cancelled
            job.updated_at = datetime.now(timezone.utc)
            logger.info("Job %s cancelled", job_id)

        except Exception as exc:
            job.status = JobStatus.failed
            job.error = str(exc)
            job.updated_at = datetime.now(timezone.utc)
            logger.error("Job %s failed: %s", job_id, exc)

        finally:
            self._tasks.pop(job_id, None)

    def cancel_job(self, job_id: str) -> bool:
        task = self._tasks.get(job_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    def get_job(self, job_id: str) -> Optional[TrainJob]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[TrainJob]:
        return list(self._jobs.values())
