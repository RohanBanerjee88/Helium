"""
Pipeline stage unit tests, job lifecycle tests, and artifact endpoint tests.
"""

import io
import math
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from helium.models.job import Job, StageStatus
from helium.pipeline.runner import _run_stage
from helium.pipeline.stages import diarize as diarize_stage
from helium.pipeline.stages import validate as validate_stage
from helium.storage.local import storage

client = TestClient(app)


def _make_wav_bytes(duration_seconds: float = 0.25, sample_rate: int = 16000) -> bytes:
    frames = int(duration_seconds * sample_rate)
    payload = io.BytesIO()
    with wave.open(payload, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        samples = bytearray()
        for idx in range(frames):
            value = int(16000 * math.sin(2 * math.pi * 220 * (idx / sample_rate)))
            samples.extend(value.to_bytes(2, byteorder="little", signed=True))
        wav_file.writeframes(bytes(samples))
    return payload.getvalue()


def _upload_files(n: int) -> list:
    return [
        ("files", (f"clip_{i:02d}.wav", io.BytesIO(_make_wav_bytes()), "audio/wav"))
        for i in range(n)
    ]


def _write_wavs(audio_dir: Path, n: int = 3) -> list:
    filenames = []
    for idx in range(n):
        name = f"clip_{idx:03d}.wav"
        (audio_dir / name).write_bytes(_make_wav_bytes())
        filenames.append(name)
    return filenames


def test_validate_accepts_valid_audio(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    files = _write_wavs(tmp_path, n=3)
    result = validate_stage.run(tmp_path, artifacts_dir, files)
    assert result["validated"] == 3
    assert len(result["clips"]) == 3
    assert (artifacts_dir / "manifests" / "validation_report.json").exists()


def test_validate_rejects_missing_file(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    with pytest.raises(validate_stage.ValidationError, match="missing"):
        validate_stage.run(tmp_path, artifacts_dir, ["ghost.wav"])


def test_validate_rejects_corrupt_wav(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (tmp_path / "broken.wav").write_bytes(b"nope")
    with pytest.raises(validate_stage.ValidationError, match="readable WAV"):
        validate_stage.run(tmp_path, artifacts_dir, ["broken.wav"])


def test_diarize_returns_placeholder_with_artifact(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    result = diarize_stage.run(tmp_path, artifacts_dir, ["clip_000.wav"], 2)
    assert result["status"] == "placeholder"
    assert result["artifacts"] == ["diarization/diarization_plan.json"]
    assert (artifacts_dir / "diarization" / "diarization_plan.json").exists()


def test_runner_marks_placeholder_stage_skipped_and_keeps_artifacts():
    job = Job()
    storage.create_job_dirs(job.id)
    storage.save_job(job)

    placeholder_fn = lambda: {
        "status": "placeholder",
        "note": "not implemented",
        "artifacts": ["diarization/diarization_plan.json"],
    }
    _run_stage(job, "diarize", placeholder_fn)

    assert job.stages["diarize"].status == StageStatus.SKIPPED
    assert job.stages["diarize"].artifacts == ["diarization/diarization_plan.json"]
    assert job.model_outputs_ready is False

    storage.delete_job(job.id)


def test_runner_marks_real_stage_completed_with_artifacts():
    job = Job()
    storage.create_job_dirs(job.id)
    storage.save_job(job)

    real_fn = lambda: {
        "metrics_planned": ["DER", "SI-SDR"],
        "artifacts": ["evaluation/evaluation_report.json"],
    }
    _run_stage(job, "evaluate", real_fn)

    assert job.stages["evaluate"].status == StageStatus.COMPLETED
    assert job.artifacts.evaluation_report == "evaluation/evaluation_report.json"

    storage.delete_job(job.id)


def test_upload_job_starts_as_pending_with_no_model_outputs():
    resp = client.post("/upload", files=_upload_files(2))
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "pending"
    assert body["model_outputs_ready"] is False
    assert body["artifacts"]["validation_report"] is None
    assert body["artifacts"]["diarization_output"] is None


def test_list_artifacts_returns_list_for_existing_job():
    resp = client.post("/upload", files=_upload_files(2))
    job_id = resp.json()["id"]

    resp2 = client.get(f"/jobs/{job_id}/artifacts")
    assert resp2.status_code == 200
    artifacts = resp2.json()
    assert isinstance(artifacts, list)
    assert any(entry["path"] == "manifests/validation_report.json" for entry in artifacts)


def test_download_artifact_404_for_nonexistent_file():
    resp = client.post("/upload", files=_upload_files(2))
    job_id = resp.json()["id"]

    resp2 = client.get(f"/jobs/{job_id}/artifacts/separation/speaker_0.wav")
    assert resp2.status_code == 404


def test_download_artifact_rejects_path_traversal():
    resp = client.post("/upload", files=_upload_files(2))
    job_id = resp.json()["id"]

    resp2 = client.get(f"/jobs/{job_id}/artifacts/../../metadata.json")
    assert resp2.status_code in (400, 404)


def test_summary_endpoint_returns_expected_shape():
    resp = client.post("/upload", files=_upload_files(2))
    job_id = resp.json()["id"]

    resp2 = client.get(f"/jobs/{job_id}/summary")
    assert resp2.status_code == 200
    body = resp2.json()

    assert body["job_id"] == job_id
    assert "status" in body
    assert "model_outputs_ready" in body
    assert "diarize" in body
    for key in (
        "validation_report",
        "diarization_output",
        "separation_output",
        "conversion_output",
        "evaluation_report",
        "export_manifest",
    ):
        assert key in body["artifacts"]
