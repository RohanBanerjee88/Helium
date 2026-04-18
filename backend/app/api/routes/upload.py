from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from ...db import JobRepository, get_db
from ...models.job import Job
from ...pipeline.runner import start_pipeline_async
from ...services import job_service, upload_service

router = APIRouter(tags=["upload"])


def get_repo(db: AsyncSession = Depends(get_db)) -> JobRepository:
    return JobRepository(db)


@router.post("/upload", response_model=Job, status_code=201)
async def upload_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="8–20 photos of the object"),
    repo: JobRepository = Depends(get_repo),
) -> Job:
    """
    Upload photos and kick off a reconstruction job.

    - Accepts 8–20 images (JPEG, PNG, BMP, TIFF).
    - Returns the created job immediately; reconstruction runs in the background.
    - Poll GET /jobs/{id} for progress.
    """
    job = await job_service.create_job(repo)

    try:
        saved = upload_service.save_images(job, files)
    except HTTPException:
        await job_service.delete_job(job.id, repo)
        raise
    except Exception as exc:
        await job_service.delete_job(job.id, repo)
        raise HTTPException(status_code=500, detail=f"Failed to save images: {exc}") from exc

    job.image_count = len(saved)
    job.images = saved
    await job_service.save_job(job, repo)

    background_tasks.add_task(start_pipeline_async, job.id)

    return job
