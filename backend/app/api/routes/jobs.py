from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ...db import JobRepository, get_db
from ...models.job import Job
from ...services import job_service

router = APIRouter(prefix="/jobs", tags=["jobs"])


def get_repo(db: AsyncSession = Depends(get_db)) -> JobRepository:
    return JobRepository(db)


@router.get("/", response_model=List[Job])
async def list_jobs(repo: JobRepository = Depends(get_repo)) -> List[Job]:
    """Return all jobs, newest first."""
    return await job_service.list_jobs(repo)


@router.get("/{job_id}", response_model=Job)
async def get_job(job_id: str, repo: JobRepository = Depends(get_repo)) -> Job:
    """Return a single job's status and per-stage detail."""
    job = await job_service.get_job(job_id, repo)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job
