import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import artifacts, jobs, upload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(
    title="Helium",
    description=(
        "Local-first photo-to-3D reconstruction backend. "
        "Upload 8–20 photos of an object and get a 3D model."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(jobs.router)
app.include_router(artifacts.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}
