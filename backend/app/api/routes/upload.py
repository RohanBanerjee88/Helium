from typing import List

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from ...models.job import Job
from ...pipeline.runner import start_pipeline_async
from ...services import job_service, upload_service
from ...storage.local import storage

router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=Job, status_code=201)
async def upload_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="8–20 photos of the object"),
) -> Job:
    """
    Upload photos and kick off a reconstruction job.

    - Accepts 8–20 images (JPEG, PNG, BMP, TIFF).
    - Returns the created job immediately; reconstruction runs in the background.
    - Poll GET /jobs/{id} for progress.
    """
    job = job_service.create_job()

    try:
        saved = upload_service.save_images(job, files)
    except HTTPException:
        job_service.delete_job(job.id)
        raise
    except Exception as exc:
        job_service.delete_job(job.id)
        raise HTTPException(status_code=500, detail=f"Failed to save images: {exc}") from exc

    job.image_count = len(saved)
    job.images = saved
    storage.save_job(job)

    background_tasks.add_task(start_pipeline_async, job.id)

    return job
