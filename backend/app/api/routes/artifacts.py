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
    """Download a specific artifact (for example manifests/validation_report.json)."""
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
    """Compact research summary: job status, model readiness, and per-stage artifacts."""
    job = job_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    diarize_stage = job.stages.get("diarize")
    diarize_info: Dict[str, Any] = {
        "status": diarize_stage.status if diarize_stage else "unknown",
        "artifacts": diarize_stage.artifacts if diarize_stage else [],
        "message": diarize_stage.message if diarize_stage else "",
    }

    return {
        "job_id": job.id,
        "status": job.status,
        "model_outputs_ready": job.model_outputs_ready,
        "audio_count": job.audio_count,
        "target_speakers": job.target_speakers,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "stages": {
            name: {"status": s.status, "artifacts": s.artifacts}
            for name, s in job.stages.items()
        },
        "diarize": diarize_info,
        "artifacts": {
            "validation_report": job.artifacts.validation_report,
            "diarization_output": job.artifacts.diarization_output,
            "separation_output": job.artifacts.separation_output,
            "conversion_output": job.artifacts.conversion_output,
            "evaluation_report": job.artifacts.evaluation_report,
            "export_manifest": job.artifacts.export_manifest,
        },
    }
