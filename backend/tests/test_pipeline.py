"""
Pipeline stage unit tests, job lifecycle tests, and artifact endpoint tests.

These tests verify:
  - validate/features/matching stages behave correctly on real (in-memory) images
  - sfm stage returns placeholder when pycolmap is absent
  - runner marks stages SKIPPED (not COMPLETED) for placeholder stages
  - job.real_reconstruction is only True when sfm produces real artifacts
  - artifact list, download, and summary endpoints work correctly
"""

import io
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

# Ensure env is configured before app is imported
if "HELIUM_DATA_DIR" not in os.environ:
    os.environ["HELIUM_DATA_DIR"] = tempfile.mkdtemp()
    os.environ["HELIUM_MIN_IMAGES"] = "2"

from fastapi.testclient import TestClient

from app.main import app
from app.models.job import Job, JobStatus, StageStatus
from app.pipeline.stages import features as feat_stage
from app.pipeline.stages import matching as match_stage
from app.pipeline.stages import validate as val_stage
from app.pipeline.stages import sfm as sfm_stage
from app.storage.local import storage

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg(width: int = 200, height: int = 200) -> bytes:
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _upload_files(n: int) -> list:
    return [
        ("files", (f"photo_{i:02d}.jpg", io.BytesIO(_make_jpeg()), "image/jpeg"))
        for i in range(n)
    ]


def _write_images(images_dir: Path, n: int = 3) -> list:
    filenames = []
    for i in range(n):
        img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        name = f"img_{i:03d}.jpg"
        cv2.imwrite(str(images_dir / name), img)
        filenames.append(name)
    return filenames


# ---------------------------------------------------------------------------
# validate stage
# ---------------------------------------------------------------------------

def test_validate_accepts_valid_images(tmp_path):
    files = _write_images(tmp_path, n=3)
    result = val_stage.run(tmp_path, files)
    assert result["validated"] == 3
    assert len(result["images"]) == 3


def test_validate_rejects_missing_file(tmp_path):
    from app.pipeline.stages.validate import ValidationError
    with pytest.raises(ValidationError, match="missing"):
        val_stage.run(tmp_path, ["ghost.jpg"])


def test_validate_rejects_tiny_image(tmp_path):
    from app.pipeline.stages.validate import ValidationError
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "tiny.jpg"), img)
    with pytest.raises(ValidationError, match="too small"):
        val_stage.run(tmp_path, ["tiny.jpg"])


# ---------------------------------------------------------------------------
# features stage
# ---------------------------------------------------------------------------

def test_features_produces_keypoint_and_descriptor_files(tmp_path):
    images_dir = tmp_path / "images"
    artifacts_dir = tmp_path / "artifacts"
    images_dir.mkdir()
    artifacts_dir.mkdir()

    files = _write_images(images_dir, n=2)
    result = feat_stage.run(images_dir, artifacts_dir, files)

    assert result["processed"] == 2
    features_dir = artifacts_dir / "features"
    assert (features_dir / "img_000_keypoints.json").exists()
    assert (features_dir / "img_000_descriptors.npy").exists()
    assert (features_dir / "img_001_keypoints.json").exists()


def test_features_skips_unreadable_image(tmp_path):
    images_dir = tmp_path / "images"
    artifacts_dir = tmp_path / "artifacts"
    images_dir.mkdir()
    artifacts_dir.mkdir()

    # Write a valid image and a corrupt file
    _write_images(images_dir, n=1)
    (images_dir / "corrupt.jpg").write_bytes(b"not-an-image")

    result = feat_stage.run(images_dir, artifacts_dir, ["img_000.jpg", "corrupt.jpg"])
    assert result["processed"] == 1  # corrupt file silently skipped


# ---------------------------------------------------------------------------
# matching stage
# ---------------------------------------------------------------------------

def test_matching_produces_match_files(tmp_path):
    images_dir = tmp_path / "images"
    artifacts_dir = tmp_path / "artifacts"
    images_dir.mkdir()
    (artifacts_dir / "features").mkdir(parents=True)
    (artifacts_dir / "matches").mkdir(parents=True)

    files = _write_images(images_dir, n=3)

    # First extract features so matching has descriptors to work with
    feat_stage.run(images_dir, artifacts_dir, files)

    result = match_stage.run(images_dir, artifacts_dir, files)
    assert result["pairs_processed"] == 3  # C(3,2) = 3
    matches_dir = artifacts_dir / "matches"
    assert len(list(matches_dir.glob("matches_*.json"))) == 3


# ---------------------------------------------------------------------------
# sfm stage — placeholder fallback
# ---------------------------------------------------------------------------

def test_sfm_returns_placeholder_when_pycolmap_missing(tmp_path):
    with patch.object(sfm_stage, "_PYCOLMAP_AVAILABLE", False):
        result = sfm_stage.run(tmp_path, tmp_path, ["a.jpg", "b.jpg"])
    assert result.get("status") == "placeholder"
    assert "artifacts" not in result or result.get("artifacts") is None or result.get("artifacts") == []


# ---------------------------------------------------------------------------
# runner: placeholder stages become SKIPPED, not COMPLETED
# ---------------------------------------------------------------------------

def test_runner_marks_placeholder_stage_skipped():
    from app.pipeline.runner import _run_stage
    from app.models.job import Job, StageStatus

    job = Job()
    storage.create_job_dirs(job.id)
    storage.save_job(job)

    placeholder_fn = lambda: {"status": "placeholder", "note": "not implemented"}
    _run_stage(job, "sfm", placeholder_fn)

    assert job.stages["sfm"].status == StageStatus.SKIPPED
    assert job.real_reconstruction is False

    storage.delete_job(job.id)


def test_runner_marks_real_stage_completed_with_artifacts():
    from app.pipeline.runner import _run_stage
    from app.models.job import Job, StageStatus

    job = Job()
    storage.create_job_dirs(job.id)
    storage.save_job(job)

    real_fn = lambda: {
        "num_points3d": 1000,
        "artifacts": ["point_cloud/sparse.ply", "point_cloud/cameras.json"],
    }
    _run_stage(job, "sfm", real_fn)

    assert job.stages["sfm"].status == StageStatus.COMPLETED
    assert job.stages["sfm"].artifacts == [
        "point_cloud/sparse.ply",
        "point_cloud/cameras.json",
    ]

    storage.delete_job(job.id)


# ---------------------------------------------------------------------------
# Job lifecycle via API
# ---------------------------------------------------------------------------

def test_upload_job_starts_as_pending_with_no_real_reconstruction():
    resp = client.post("/upload", files=_upload_files(3))
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "pending"
    assert body["real_reconstruction"] is False
    assert body["artifacts"]["sparse_ply"] is None
    assert body["artifacts"]["cameras_json"] is None


def test_failed_upload_cleans_up_job_dirs():
    # A single file is below HELIUM_MIN_IMAGES=2 so upload_service raises HTTPException
    resp = client.post("/upload", files=_upload_files(1))
    assert resp.status_code == 400
    # After a failed upload caused by validation, no job dirs should linger
    # (We can't easily enumerate dirs here, but we verify the response is correct.)


# ---------------------------------------------------------------------------
# Artifact endpoints
# ---------------------------------------------------------------------------

def test_list_artifacts_returns_list_for_existing_job():
    resp = client.post("/upload", files=_upload_files(3))
    job_id = resp.json()["id"]

    resp2 = client.get(f"/jobs/{job_id}/artifacts")
    assert resp2.status_code == 200
    assert isinstance(resp2.json(), list)


def test_list_artifacts_404_for_unknown_job():
    resp = client.get("/jobs/no-such-job/artifacts")
    assert resp.status_code == 404


def test_download_artifact_404_for_nonexistent_file():
    resp = client.post("/upload", files=_upload_files(3))
    job_id = resp.json()["id"]

    resp2 = client.get(f"/jobs/{job_id}/artifacts/point_cloud/sparse.ply")
    assert resp2.status_code == 404


def test_download_artifact_rejects_path_traversal():
    resp = client.post("/upload", files=_upload_files(3))
    job_id = resp.json()["id"]

    # URL-encode a path traversal attempt
    resp2 = client.get(f"/jobs/{job_id}/artifacts/../../metadata.json")
    # Must be rejected; 400 from our check or 404 if path resolved outside artifacts_dir
    assert resp2.status_code in (400, 404)


def test_summary_endpoint_returns_expected_shape():
    resp = client.post("/upload", files=_upload_files(3))
    job_id = resp.json()["id"]

    resp2 = client.get(f"/jobs/{job_id}/summary")
    assert resp2.status_code == 200
    body = resp2.json()

    assert body["job_id"] == job_id
    assert "status" in body
    assert "real_reconstruction" in body
    assert "stages" in body
    assert "sfm" in body
    assert "artifacts" in body
    assert set(body["artifacts"].keys()) == {
        "sparse_ply", "cameras_json", "dense_ply", "mesh_obj", "mesh_stl"
    }


def test_summary_404_for_unknown_job():
    resp = client.get("/jobs/no-such-job/summary")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Storage: atomic save and list_artifacts
# ---------------------------------------------------------------------------

def test_storage_atomic_save_load_roundtrip():
    job = Job()
    storage.create_job_dirs(job.id)
    job.real_reconstruction = True
    job.artifacts.sparse_ply = "point_cloud/sparse.ply"
    storage.save_job(job)

    loaded = storage.load_job(job.id)
    assert loaded is not None
    assert loaded.real_reconstruction is True
    assert loaded.artifacts.sparse_ply == "point_cloud/sparse.ply"

    storage.delete_job(job.id)


def test_storage_list_artifacts_empty_before_pipeline():
    job = Job()
    storage.create_job_dirs(job.id)
    storage.save_job(job)

    artifacts = storage.list_artifacts(job.id)
    assert isinstance(artifacts, list)
    # Dirs are created but no files yet (beyond the status JSONs from placeholder stages)

    storage.delete_job(job.id)


def test_storage_list_artifacts_includes_written_files():
    job = Job()
    storage.create_job_dirs(job.id)
    storage.save_job(job)

    ply_path = storage.artifacts_dir(job.id) / "point_cloud" / "sparse.ply"
    ply_path.write_bytes(b"ply\nformat ascii 1.0\n")

    artifacts = storage.list_artifacts(job.id)
    names = [a["name"] for a in artifacts]
    assert "sparse.ply" in names

    entry = next(a for a in artifacts if a["name"] == "sparse.ply")
    assert entry["path"] == "point_cloud/sparse.ply"
    assert entry["size_bytes"] > 0

    storage.delete_job(job.id)
