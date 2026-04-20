"""
Background pipeline runner (used by the FastAPI server).

Executes reconstruction stages sequentially in a daemon thread so the
API can return the job immediately and the caller polls for progress.

For the CLI, stages are driven directly in helium/cli/main.py so that
output can be printed to the terminal as each stage completes.

Stage order:  validate → features → matching → sfm → dense → export

Stages that return {"status": "placeholder"} are marked SKIPPED rather
than COMPLETED so the job is never falsely reported as a real
reconstruction.  job.real_reconstruction is only True when sfm produces
sparse.ply + cameras.json.
"""

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict

from ..models.job import Job, JobStatus, StageStatus
from ..storage.local import storage
from .stages import dense, export, features, matching, sfm, validate

_PLACEHOLDER = "placeholder"


def _run_stage(
    job: Job,
    stage_name: str,
    fn: Callable[..., Dict[str, Any]],
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    stage = job.stages[stage_name]
    stage.status = StageStatus.RUNNING
    stage.started_at = datetime.now(timezone.utc)
    storage.save_job(job)

    try:
        result = fn(*args, **kwargs)

        if isinstance(result, dict) and result.get("status") == _PLACEHOLDER:
            stage.status = StageStatus.SKIPPED
            stage.message = result.get("note", "Not yet implemented.")
        else:
            stage.status = StageStatus.COMPLETED
            stage.message = str(result)
            if isinstance(result, dict) and "artifacts" in result:
                stage.artifacts = list(result["artifacts"])

        stage.completed_at = datetime.now(timezone.utc)
        storage.save_job(job)
        return result
    except Exception as exc:
        stage.status = StageStatus.FAILED
        stage.message = str(exc)
        stage.completed_at = datetime.now(timezone.utc)
        job.status = JobStatus.FAILED
        job.error = f"Stage '{stage_name}' failed: {exc}"
        storage.save_job(job)
        raise


def _skip_remaining(job: Job, from_index: int) -> None:
    stages = list(job.stages.keys())
    for name in stages[from_index:]:
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

        sfm_stage = job.stages["sfm"]
        if sfm_stage.status == StageStatus.COMPLETED and sfm_stage.artifacts:
            job.real_reconstruction = True
            for rel in sfm_stage.artifacts:
                name = Path(rel).name
                if name == "sparse.ply":
                    job.artifacts.sparse_ply = rel
                elif name == "cameras.json":
                    job.artifacts.cameras_json = rel

        job.status = JobStatus.COMPLETED
        storage.save_job(job)

    except Exception:
        failed_index = next(
            (i for i, (_, s) in enumerate(job.stages.items()) if s.status == StageStatus.FAILED),
            None,
        )
        if failed_index is not None:
            _skip_remaining(job, failed_index + 1)


def start_pipeline_async(job_id: str) -> None:
    thread = threading.Thread(target=run_pipeline, args=(job_id,), daemon=True)
    thread.start()
