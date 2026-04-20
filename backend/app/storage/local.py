import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
          point_cloud/    sparse + dense PLY files
          mesh/           OBJ / STL exports
        metadata.json     job state (atomic write on every status change)
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

    def delete_job(self, job_id: str) -> None:
        job_dir = self.job_dir(job_id)
        if job_dir.exists():
            shutil.rmtree(job_dir)

    # --- Persistence ---

    def save_job(self, job: Job) -> None:
        job.updated_at = datetime.now(timezone.utc)
        path = self.metadata_path(job.id)
        # Atomic write: write to sibling temp file, then rename
        with tempfile.NamedTemporaryFile(
            mode="w", dir=str(path.parent), delete=False, suffix=".tmp"
        ) as tmp:
            tmp.write(job.model_dump_json(indent=2))
            tmp_path = tmp.name
        os.replace(tmp_path, path)

    def load_job(self, job_id: str) -> Optional[Job]:
        path = self.metadata_path(job_id)
        if not path.exists():
            return None
        with open(path) as fh:
            return Job.model_validate_json(fh.read())

    def list_job_ids(self) -> List[str]:
        return [d.name for d in self._jobs_root.iterdir() if d.is_dir()]

    def list_artifacts(self, job_id: str) -> List[Dict[str, Any]]:
        artifacts_dir = self.artifacts_dir(job_id)
        if not artifacts_dir.exists():
            return []
        result = []
        for path in sorted(artifacts_dir.rglob("*")):
            if path.is_file():
                rel = path.relative_to(artifacts_dir)
                result.append(
                    {
                        "name": path.name,
                        "path": str(rel),
                        "size_bytes": path.stat().st_size,
                    }
                )
        return result


storage = LocalStorage()
