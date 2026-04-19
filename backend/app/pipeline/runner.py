"""
Pipeline runner.

Executes reconstruction stages sequentially in a background thread.
Each stage updates its own status in metadata so callers can poll /jobs/{id}.

Stage execution order:
  validate → features → matching → sfm → dense → export

Stages that return {"status": "placeholder"} are marked SKIPPED rather than
COMPLETED — the job is never falsely reported as a real reconstruction when
sfm hasn't run. The job is only marked real_reconstruction=True when sfm
produces expected artifacts (sparse.ply + cameras.json).
"""

import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..config import settings
from ..db.repository import JobRepository
from ..models.job import Job, JobStatus, StageStatus
from ..pipeline.stages import dense, export, features, matching, sfm, validate
from ..storage.local import storage

_PLACEHOLDER_STATUS = "placeholder"


def _make_engine():
    return create_async_engine(
        settings.database_url, connect_args={"ssl": True}, pool_size=1, max_overflow=0
    )


async def _db_save(job: Job) -> None:
    engine = _make_engine()
    try:
        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with session_factory() as session:
            await JobRepository(session).save(job)
            await session.commit()
    finally:
        await engine.dispose()


async def _db_load(job_id: str) -> Optional[Job]:
    engine = _make_engine()
    try:
        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with session_factory() as session:
            return await JobRepository(session).get(job_id)
    finally:
        await engine.dispose()


def _save_job(job: Job) -> None:
    asyncio.run(_db_save(job))


def _load_job(job_id: str) -> Optional[Job]:
    return asyncio.run(_db_load(job_id))


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
    _save_job(job)

    try:
        result = fn(*args, **kwargs)

        if isinstance(result, dict) and result.get("status") == _PLACEHOLDER_STATUS:
            stage.status = StageStatus.SKIPPED
            stage.message = result.get("note", "Not yet implemented.")
        else:
            stage.status = StageStatus.COMPLETED
            stage.message = str(result)
            if isinstance(result, dict) and "artifacts" in result:
                stage.artifacts = list(result["artifacts"])

        stage.completed_at = datetime.utcnow()
        _save_job(job)
        return result
    except Exception as exc:
        stage.status = StageStatus.FAILED
        stage.message = str(exc)
        stage.completed_at = datetime.utcnow()
        job.status = JobStatus.FAILED
        job.error = f"Stage '{stage_name}' failed: {exc}"
        _save_job(job)
        raise


def _skip_remaining(job: Job, from_index: int) -> None:
    stages = list(job.stages.keys())
    for name in stages[from_index:]:
        if job.stages[name].status == StageStatus.PENDING:
            job.stages[name].status = StageStatus.SKIPPED
    _save_job(job)


def run_pipeline(job_id: str) -> None:
    job = _load_job(job_id)
    if job is None:
        return

    job.status = JobStatus.RUNNING
    _save_job(job)

    images_dir: Path = storage.images_dir(job_id)
    artifacts_dir: Path = storage.artifacts_dir(job_id)

    try:
        _run_stage(job, "validate", validate.run, images_dir, job.images)
        _run_stage(job, "features", features.run, images_dir, artifacts_dir, job.images)
        _run_stage(job, "matching", matching.run, images_dir, artifacts_dir, job.images)
        _run_stage(job, "sfm", sfm.run, images_dir, artifacts_dir, job.images)
        _run_stage(job, "dense", dense.run, artifacts_dir, job.images)
        _run_stage(job, "export", export.run, artifacts_dir)

        # Only flag real_reconstruction when sfm produced actual artifact files
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
        _save_job(job)

    except Exception:
        failed_index = next(
            (
                i
                for i, (name, s) in enumerate(job.stages.items())
                if s.status == StageStatus.FAILED
            ),
            None,
        )
        if failed_index is not None:
            _skip_remaining(job, failed_index + 1)


def start_pipeline_async(job_id: str) -> None:
    thread = threading.Thread(target=run_pipeline, args=(job_id,), daemon=True)
    thread.start()
