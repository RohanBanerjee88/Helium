"""
Integration tests for the upload and job status endpoints.
"""

import io
import math
import wave

from fastapi.testclient import TestClient

from app.main import app
from app.services import upload_service

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
            value = int(16000 * math.sin(2 * math.pi * 440 * (idx / sample_rate)))
            samples.extend(value.to_bytes(2, byteorder="little", signed=True))
        wav_file.writeframes(bytes(samples))
    return payload.getvalue()


def _audio_files(n: int) -> list:
    return [
        ("files", (f"clip_{i:02d}.wav", io.BytesIO(_make_wav_bytes()), "audio/wav"))
        for i in range(n)
    ]


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_upload_too_few_audio_files():
    resp = client.post("/upload", files=_audio_files(1))
    assert resp.status_code == 400
    assert "at least" in resp.json()["detail"].lower()


def test_upload_too_many_audio_files():
    original_min = upload_service.settings.min_audio_files
    original_max = upload_service.settings.max_audio_files
    original_size = upload_service.settings.max_audio_size_mb
    upload_service.settings.min_audio_files = 1
    upload_service.settings.max_audio_files = 3
    upload_service.settings.max_audio_size_mb = 100
    resp = client.post("/upload", files=_audio_files(5))
    upload_service.settings.min_audio_files = original_min
    upload_service.settings.max_audio_files = original_max
    upload_service.settings.max_audio_size_mb = original_size
    assert resp.status_code == 400


def test_upload_success():
    resp = client.post("/upload", files=_audio_files(3))
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert "id" in body
    assert body["status"] == "pending"
    assert body["audio_count"] == 3
    assert len(body["audio_files"]) == 3


def test_upload_bad_content_type():
    files = [("files", ("clip.mp3", io.BytesIO(b"not-mp3"), "audio/mpeg"))]
    resp = client.post("/upload", files=files)
    assert resp.status_code == 400


def test_get_job_not_found():
    resp = client.get("/jobs/nonexistent-id")
    assert resp.status_code == 404


def test_list_jobs_returns_array():
    resp = client.get("/jobs/")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_full_flow():
    resp = client.post("/upload", files=_audio_files(4))
    assert resp.status_code == 201
    job_id = resp.json()["id"]

    resp2 = client.get(f"/jobs/{job_id}")
    assert resp2.status_code == 200
    body = resp2.json()
    assert body["id"] == job_id
    assert body["audio_count"] == 4

    resp3 = client.get("/jobs/")
    ids = [job["id"] for job in resp3.json()]
    assert job_id in ids
