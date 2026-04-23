from typing import List

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from ...models.job import Job
from ...pipeline.runner import start_pipeline_async
from ...services import job_service, upload_service
from ...storage.local import storage

router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=Job, status_code=201)
async def upload_audio(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="1–6 WAV files for a speech research job"),
) -> Job:
    """
    Upload WAV files and kick off a local speech research pipeline.

    - Accepts 1–6 WAV files.
    - Returns the created job immediately; analysis runs in the background.
    - Poll GET /jobs/{id} for progress.
    """
    job = job_service.create_job()
    job.target_speakers = 2

    try:
        saved = upload_service.save_audio(job, files)
    except HTTPException:
        job_service.delete_job(job.id)
        raise
    except Exception as exc:
        job_service.delete_job(job.id)
        raise HTTPException(status_code=500, detail=f"Failed to save audio: {exc}") from exc

    job.audio_count = len(saved)
    job.audio_files = saved
    storage.save_job(job)

    background_tasks.add_task(start_pipeline_async, job.id)

    return job
