"""
Pipeline runner.

Executes the six reconstruction stages sequentially in a background thread.
Each stage updates its own status in the job metadata file so callers can
poll /jobs/{id} for real-time progress.

Stage execution order:
  validate → features → matching → sfm → dense → export

Any unhandled exception inside a stage marks that stage and the whole job as
FAILED and records the error message. Subsequent stages are skipped.
"""

import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..models.job import Job, JobStatus, StageStatus
from ..pipeline.stages import dense, export, features, matching, sfm, validate
from ..storage.local import storage


def _run_stage(
    job: Job,
    stage_name: str,
    fn: Callable[..., Dict[str, Any]],
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    stage = job.stages[stage_name]
    stage.status = StageStatus.RUNNING
    stage.started_at = datetime.utcnow()
    storage.save_job(job)

    try:
        result = fn(*args, **kwargs)
        stage.status = StageStatus.COMPLETED
        stage.message = str(result)
        stage.completed_at = datetime.utcnow()
        storage.save_job(job)
        return result
    except Exception as exc:
        stage.status = StageStatus.FAILED
        stage.message = str(exc)
        stage.completed_at = datetime.utcnow()
        job.status = JobStatus.FAILED
        job.error = f"Stage '{stage_name}' failed: {exc}"
        storage.save_job(job)
        raise


def _skip_remaining(job: Job, from_stage: str) -> None:
    stages = list(job.stages.keys())
    start = stages.index(from_stage) if from_stage in stages else len(stages)
    for name in stages[start:]:
        if job.stages[name].status == StageStatus.PENDING:
            job.stages[name].status = StageStatus.SKIPPED
    storage.save_job(job)


def run_pipeline(job_id: str) -> None:
    job = storage.load_job(job_id)
    if job is None:
        return

    job.status = JobStatus.RUNNING
    storage.save_job(job)

    images_dir: Path = storage.images_dir(job_id)
    artifacts_dir: Path = storage.artifacts_dir(job_id)

    try:
        _run_stage(job, "validate", validate.run, images_dir, job.images)
        _run_stage(job, "features", features.run, images_dir, artifacts_dir, job.images)
        _run_stage(job, "matching", matching.run, images_dir, artifacts_dir, job.images)
        _run_stage(job, "sfm", sfm.run, images_dir, artifacts_dir, job.images)
        _run_stage(job, "dense", dense.run, artifacts_dir, job.images)
        _run_stage(job, "export", export.run, artifacts_dir)

        job.status = JobStatus.COMPLETED
        storage.save_job(job)

    except Exception:
        # Find the first failed stage and skip everything after it
        failed = next(
            (name for name, s in job.stages.items() if s.status == StageStatus.FAILED),
            None,
        )
        if failed:
            next_stages = list(job.stages.keys())
            idx = next_stages.index(failed) + 1
            _skip_remaining(job, next_stages[idx] if idx < len(next_stages) else "")


def start_pipeline_async(job_id: str) -> None:
    thread = threading.Thread(target=run_pipeline, args=(job_id,), daemon=True)
    thread.start()
