from ..models.job import Job, JobArtifacts, PIPELINE_STAGES, StageResult
from .models import JobRow


def job_row_to_pydantic(row: JobRow) -> Job:
    stages_raw = row.stages or {}
    stages = {
        name: StageResult.model_validate(stages_raw.get(name, {}))
        for name in PIPELINE_STAGES
    }
    return Job(
        id=row.id,
        status=row.status,
        created_at=row.created_at,
        updated_at=row.updated_at,
        image_count=row.image_count,
        images=row.images or [],
        stages=stages,
        error=row.error,
        artifacts=JobArtifacts.model_validate(row.artifacts or {}),
        real_reconstruction=row.real_reconstruction,
    )


def pydantic_to_dict(job: Job) -> dict:
    return {
        "id": job.id,
        "status": job.status.value,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "image_count": job.image_count,
        "images": job.images,
        "stages": {
            name: result.model_dump(mode="json")
            for name, result in job.stages.items()
        },
        "error": job.error,
        "artifacts": job.artifacts.model_dump(mode="json"),
        "real_reconstruction": job.real_reconstruction,
    }
