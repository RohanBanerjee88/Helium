from typing import List, Optional

from ..db.repository import JobRepository
from ..models.job import Job
from ..storage.local import storage


async def create_job(repo: JobRepository) -> Job:
    job = Job()
    storage.create_job_dirs(job.id)
    return await repo.create(job)


async def get_job(job_id: str, repo: JobRepository) -> Optional[Job]:
    return await repo.get(job_id)


async def list_jobs(repo: JobRepository) -> List[Job]:
    return await repo.list_all()


async def save_job(job: Job, repo: JobRepository) -> Job:
    return await repo.save(job)


async def delete_job(job_id: str, repo: JobRepository) -> None:
    storage.delete_job(job_id)
    await repo.delete(job_id)
