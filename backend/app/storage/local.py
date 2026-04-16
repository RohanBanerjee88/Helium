import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..config import settings
from ..models.job import Job


class LocalStorage:
    """
    All job data lives under:
      {data_dir}/jobs/{job_id}/
        images/           raw uploaded images
        artifacts/
          features/       keypoints + descriptors per image
          matches/        pairwise match files
          point_cloud/    sparse + dense PLY files (future)
          mesh/           OBJ / STL exports (future)
        metadata.json     job state
    """

    def __init__(self) -> None:
        self._jobs_root = Path(settings.data_dir) / "jobs"
        self._jobs_root.mkdir(parents=True, exist_ok=True)

    # --- Path helpers ---

    def job_dir(self, job_id: str) -> Path:
        return self._jobs_root / job_id

    def images_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "images"

    def artifacts_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "artifacts"

    def metadata_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "metadata.json"

    # --- Lifecycle ---

    def create_job_dirs(self, job_id: str) -> None:
        artifacts = self.artifacts_dir(job_id)
        for subdir in ("features", "matches", "point_cloud", "mesh"):
            (artifacts / subdir).mkdir(parents=True, exist_ok=True)
        self.images_dir(job_id).mkdir(parents=True, exist_ok=True)

    # --- Persistence ---

    def save_job(self, job: Job) -> None:
        job.updated_at = datetime.utcnow()
        with open(self.metadata_path(job.id), "w") as fh:
            fh.write(job.model_dump_json(indent=2))

    def load_job(self, job_id: str) -> Optional[Job]:
        path = self.metadata_path(job_id)
        if not path.exists():
            return None
        with open(path) as fh:
            return Job.model_validate_json(fh.read())

    def list_job_ids(self) -> List[str]:
        return [d.name for d in self._jobs_root.iterdir() if d.is_dir()]


storage = LocalStorage()
