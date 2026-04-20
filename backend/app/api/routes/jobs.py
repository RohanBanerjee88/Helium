from typing import List

from fastapi import APIRouter, HTTPException

from ...models.job import Job
from ...services import job_service

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/", response_model=List[Job])
def list_jobs() -> List[Job]:
    """Return all jobs, newest first."""
    return job_service.list_jobs()


@router.get("/{job_id}", response_model=Job)
def get_job(job_id: str) -> Job:
    """Return a single job's status and per-stage detail."""
    job = job_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job
