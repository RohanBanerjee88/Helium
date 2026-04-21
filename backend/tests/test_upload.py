"""
Integration tests for the upload and job status endpoints.

These tests use an in-process TestClient and a temp directory for storage
so they run without Docker and leave no residue.
"""

import io
import os
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from helium.storage.local import storage

client = TestClient(app)


def _make_jpeg(width: int = 200, height: int = 200) -> bytes:
    """Create a tiny valid JPEG in memory."""
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _image_files(n: int) -> list:
    return [
        ("files", (f"photo_{i:02d}.jpg", io.BytesIO(_make_jpeg()), "image/jpeg"))
        for i in range(n)
    ]


# --- health ---

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# --- upload ---

def test_upload_too_few_images():
    resp = client.post("/upload", files=_image_files(1))
    assert resp.status_code == 400
    assert "minimum" in resp.json()["detail"].lower() or "least" in resp.json()["detail"].lower()


def test_upload_too_many_images():
    with patch("app.config.settings") as mock_settings:
        mock_settings.min_images = 2
        mock_settings.max_images = 3
        mock_settings.max_image_size_mb = 20
    resp = client.post("/upload", files=_image_files(25))
    assert resp.status_code == 400


def test_upload_success():
    resp = client.post("/upload", files=_image_files(3))
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert "id" in body
    assert body["status"] == "pending"
    assert body["image_count"] == 3
    assert len(body["images"]) == 3


def test_upload_bad_content_type():
    files = [("files", ("doc.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf"))]
    resp = client.post("/upload", files=files)
    assert resp.status_code == 400


def test_failed_upload_cleans_up_partial_job_state():
    before = set(storage.list_job_ids())
    oversized = b"x" * (2 * 1024 * 1024)

    with patch("app.services.upload_service.settings.max_image_size_mb", 1):
        resp = client.post(
            "/upload",
            files=[
                ("files", ("photo_ok.jpg", io.BytesIO(_make_jpeg()), "image/jpeg")),
                ("files", ("photo_big.jpg", io.BytesIO(oversized), "image/jpeg")),
            ],
        )

    assert resp.status_code == 400
    after = set(storage.list_job_ids())
    assert after == before


# --- jobs ---

def test_get_job_not_found():
    resp = client.get("/jobs/nonexistent-id")
    assert resp.status_code == 404


def test_list_jobs_empty():
    # Uses a fresh tmp dir per test run; may not be empty across tests but should not crash
    resp = client.get("/jobs/")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_full_flow():
    # Upload
    resp = client.post("/upload", files=_image_files(4))
    assert resp.status_code == 201
    job_id = resp.json()["id"]

    # Immediately readable
    resp2 = client.get(f"/jobs/{job_id}")
    assert resp2.status_code == 200
    body = resp2.json()
    assert body["id"] == job_id
    assert body["image_count"] == 4

    # Appears in list
    resp3 = client.get("/jobs/")
    ids = [j["id"] for j in resp3.json()]
    assert job_id in ids
