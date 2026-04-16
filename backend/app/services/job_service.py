from typing import List, Optional

from ..models.job import Job, JobStatus
from ..storage.local import storage


def create_job() -> Job:
    job = Job()
    storage.create_job_dirs(job.id)
    storage.save_job(job)
    return job


def get_job(job_id: str) -> Optional[Job]:
    return storage.load_job(job_id)


def list_jobs() -> List[Job]:
    jobs = []
    for job_id in storage.list_job_ids():
        job = storage.load_job(job_id)
        if job is not None:
            jobs.append(job)
    return sorted(jobs, key=lambda j: j.created_at, reverse=True)
