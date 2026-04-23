"""
Background pipeline runner (used by the FastAPI server).

Executes pipeline stages sequentially in a daemon thread so the
API can return the job immediately and the caller polls for progress.

For the CLI, stages are driven directly in helium/cli/main.py so that
output can be printed to the terminal as each stage completes.

Stage order:  validate → diarize → separate → convert → evaluate → export

Stages that return {"status": "placeholder"} are marked SKIPPED rather
than COMPLETED so the job is never falsely reported as running a real
model-backed pipeline. Placeholder stages may still emit planning
artifacts so the research workflow stays inspectable.
"""

import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict

from ..models.job import Job, JobStatus, StageStatus
from ..storage.local import storage
from .stages import convert, diarize, evaluate, export, separate, validate

_PLACEHOLDER = "placeholder"
_ARTIFACT_FIELD_BY_STAGE = {
    "validate": "validation_report",
    "diarize": "diarization_output",
    "separate": "separation_output",
    "convert": "conversion_output",
    "evaluate": "evaluation_report",
    "export": "export_manifest",
}


def _sync_job_artifacts(job: Job) -> None:
    for stage_name, field_name in _ARTIFACT_FIELD_BY_STAGE.items():
        stage = job.stages.get(stage_name)
        setattr(job.artifacts, field_name, stage.artifacts[0] if stage and stage.artifacts else None)

    job.model_outputs_ready = any(
        job.stages[name].status == StageStatus.COMPLETED and bool(job.stages[name].artifacts)
        for name in ("diarize", "separate", "convert")
    )


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
            if "artifacts" in result:
                stage.artifacts = list(result["artifacts"])
        else:
            stage.status = StageStatus.COMPLETED
            stage.message = str(result)
            if isinstance(result, dict) and "artifacts" in result:
                stage.artifacts = list(result["artifacts"])

        stage.completed_at = datetime.now(timezone.utc)
        _sync_job_artifacts(job)
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

    audio_dir = storage.audio_dir(job_id)
    artifacts_dir = storage.artifacts_dir(job_id)

    try:
        _run_stage(job, "validate", validate.run, audio_dir, artifacts_dir, job.audio_files)
        _run_stage(job, "diarize", diarize.run, audio_dir, artifacts_dir, job.audio_files, job.target_speakers)
        _run_stage(job, "separate", separate.run, audio_dir, artifacts_dir, job.audio_files, job.target_speakers)
        _run_stage(job, "convert", convert.run, audio_dir, artifacts_dir, job.audio_files, job.target_speakers)
        _run_stage(job, "evaluate", evaluate.run, artifacts_dir, job.audio_files, job.target_speakers)
        _run_stage(job, "export", export.run, artifacts_dir, job.id)

        job.status = JobStatus.COMPLETED
        _sync_job_artifacts(job)
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
