# Helium

**Local-first photo-to-3D reconstruction backend.**

Upload 8–20 photos of a real-world object and Helium runs a photogrammetry-style pipeline to produce a 3D model suitable for future 3D printing export.

## What it does (V1)

- Accepts 8–20 images via REST API
- Validates images (format, size, decodability)
- Extracts SIFT keypoints and descriptors (OpenCV)
- Matches features between all image pairs (Lowe ratio test)
- Tracks per-job, per-stage status in real time
- Placeholder hooks for SfM (COLMAP), dense reconstruction, and mesh export

## What it does NOT do yet

- Full camera pose estimation (COLMAP integration is a TODO stub)
- Dense multi-view stereo reconstruction
- Mesh repair or guaranteed print-ready output
- Handle reflective, transparent, or dark objects
- Auth, cloud deployment, or any kind of paid infrastructure

## Object requirements

- One object at a time
- Matte, non-transparent, non-reflective surface
- Decent, even lighting
- 8–20 images from varied angles with good overlap (~60% between adjacent shots)

## Tech stack

- Python 3.11 + FastAPI + uvicorn
- OpenCV (SIFT feature extraction and matching)
- Open3D (future: point cloud processing)
- trimesh (future: mesh export)
- Docker + docker-compose

---

## Quick start

```bash
cp .env.example .env
docker-compose up --build
```

API available at `http://localhost:8000`  
Docs at `http://localhost:8000/docs`

### Upload images

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@photo1.jpg" \
  -F "files=@photo2.jpg" \
  ... \
  -F "files=@photo12.jpg"
```

Response includes a `job_id`.

### Check job status

```bash
curl http://localhost:8000/jobs/{job_id}
```

### List all jobs

```bash
curl http://localhost:8000/jobs/
```

---

## Local development (without Docker)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

---

## Project structure

```
helium/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + middleware
│   │   ├── config.py            # Settings (env vars via pydantic-settings)
│   │   ├── api/routes/
│   │   │   ├── upload.py        # POST /upload
│   │   │   └── jobs.py          # GET /jobs, GET /jobs/{id}
│   │   ├── models/
│   │   │   └── job.py           # Job, StageResult, enums
│   │   ├── services/
│   │   │   ├── job_service.py
│   │   │   └── upload_service.py
│   │   ├── pipeline/
│   │   │   ├── runner.py        # Stage orchestrator (background thread)
│   │   │   └── stages/
│   │   │       ├── validate.py  # Image validation (OpenCV)
│   │   │       ├── features.py  # SIFT feature extraction
│   │   │       ├── matching.py  # BFMatcher + Lowe ratio
│   │   │       ├── sfm.py       # TODO: COLMAP / SfM integration
│   │   │       ├── dense.py     # TODO: dense reconstruction
│   │   │       └── export.py    # TODO: mesh export (trimesh / Open3D)
│   │   └── storage/
│   │       └── local.py         # Filesystem layout + JSON metadata
│   ├── tests/
│   │   └── test_upload.py
│   ├── Dockerfile
│   └── requirements.txt
├── data/                        # Runtime data (gitignored)
├── docker-compose.yml
├── .env.example
├── Makefile
└── README.md
```

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 0 — Bootstrap | ✅ Done | Scaffold, Docker, config, health check |
| 1 — Upload + Jobs | ✅ Done | POST /upload, job creation, file storage |
| 2 — Pipeline Skeleton | ✅ Done | Background runner, 6 stages wired |
| 3 — Real CV | ✅ Done | SIFT features, BFMatcher, Lowe ratio |
| 4 — SfM Integration | 🔲 TODO | COLMAP subprocess or pycolmap |
| 5 — Dense + Export | 🔲 TODO | Open3D MVS, Poisson mesh, trimesh STL/OBJ |

## Contributing

PRs welcome. Keep changes modular and scoped. See `CONTRIBUTING.md` (coming soon).

## License

Apache 2.0
