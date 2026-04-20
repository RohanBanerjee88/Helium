from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ...services import job_service
from ...storage.local import storage

router = APIRouter(prefix="/jobs", tags=["artifacts"])


@router.get("/{job_id}/artifacts", response_model=List[Dict[str, Any]])
def list_artifacts(job_id: str) -> List[Dict[str, Any]]:
    """List all artifact files produced for a job."""
    if job_service.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return storage.list_artifacts(job_id)


@router.get("/{job_id}/artifacts/{file_path:path}")
def download_artifact(job_id: str, file_path: str) -> FileResponse:
    """Download a specific artifact (e.g. point_cloud/sparse.ply)."""
    if job_service.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    artifacts_dir = storage.artifacts_dir(job_id)
    target = (artifacts_dir / file_path).resolve()

    # Reject path traversal attempts
    if not str(target).startswith(str(artifacts_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid artifact path.")

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact '{file_path}' not found.")

    return FileResponse(path=str(target), filename=target.name)


@router.get("/{job_id}/summary")
def job_summary(job_id: str) -> Dict[str, Any]:
    """Compact reconstruction summary: status, real_reconstruction flag, per-stage artifacts."""
    job = job_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    sfm_stage = job.stages.get("sfm")
    sfm_info: Dict[str, Any] = {
        "status": sfm_stage.status if sfm_stage else "unknown",
        "artifacts": sfm_stage.artifacts if sfm_stage else [],
        "message": sfm_stage.message if sfm_stage else "",
    }

    return {
        "job_id": job.id,
        "status": job.status,
        "real_reconstruction": job.real_reconstruction,
        "image_count": job.image_count,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "stages": {
            name: {"status": s.status, "artifacts": s.artifacts}
            for name, s in job.stages.items()
        },
        "sfm": sfm_info,
        "artifacts": {
            "sparse_ply": job.artifacts.sparse_ply,
            "cameras_json": job.artifacts.cameras_json,
            "dense_ply": job.artifacts.dense_ply,
            "mesh_obj": job.artifacts.mesh_obj,
            "mesh_stl": job.artifacts.mesh_stl,
        },
    }
