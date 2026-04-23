from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import artifacts, jobs, upload

app = FastAPI(
    title="Helium",
    description=(
        "Local-first speech research backend. "
        "Upload WAV files and orchestrate diarization, separation, voice conversion, and evaluation."
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
