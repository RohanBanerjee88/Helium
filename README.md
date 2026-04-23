# Helium

**Local-first speech research backend for speaker diarization, source separation, and controllable voice conversion.**

Helium is now aimed at a research workflow, not cloud hosting. The project is designed to orchestrate local audio jobs, keep every intermediate artifact on disk, and make it easy to compare open-source backends for:

- speaker diarization in crowded or noisy scenes
- two-speaker speech separation
- style-preserving voice conversion
- evaluation and experiment tracking for publishable work

## Current Scope

Helium currently provides:

- a FastAPI backend for uploading 1-6 WAV files
- a local CLI for running the same pipeline on folders of audio
- a job model with stage-by-stage tracking
- audio validation with metadata export
- research scaffold stages for diarization, separation, conversion, evaluation, and export
- placeholder artifacts that document how to plug in local Hugging Face-compatible backends

Helium does not yet bundle pretrained diarization, separation, or voice conversion checkpoints. That is deliberate: the goal is to keep the repo open-source, local-first, and research-friendly instead of tightly coupling it to one gated stack.

## Research Direction

The current target problem is:

`noisy two-speaker speech disentanglement + style-preserving voice conversion`

Example use case:

- two people speaking in a cafe
- detect two distinct speakers
- isolate each speaker
- convert one speaker to a different target timbre while preserving words, rhythm, and turn-taking

## Suggested Baselines

- Diarization: `pyannote/speaker-diarization`
- Separation: `speechbrain/sepformer-whamr`
- Voice conversion: `RedRepter/seed-vc-api`

These are not hardcoded into the repo yet, but the scaffold writes per-stage planning artifacts that point to them.

## Quick Start

```bash
cp .env.example .env
docker-compose up --build
```

API available at `http://localhost:8000`  
Docs at `http://localhost:8000/docs`

### Upload audio

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@mixture.wav" \
  -F "files=@target_reference.wav"
```

Response includes a `job_id`.

### Check job status

```bash
curl http://localhost:8000/jobs/{job_id}
```

### Get a compact summary

```bash
curl http://localhost:8000/jobs/{job_id}/summary
```

## Local Development

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## CLI

Run the local pipeline on a directory of WAV files:

```bash
helium run ./sample_audio --target-speakers 2
```

Inspect jobs:

```bash
helium jobs list
helium jobs show <job-id>
```

## Pipeline

The current stage order is:

1. `validate`
2. `diarize`
3. `separate`
4. `convert`
5. `evaluate`
6. `export`

Only `validate`, `evaluate`, and `export` are implemented as real scaffold stages today. The model-backed stages currently emit planning artifacts and are marked as skipped until you wire in local backends.

## Project Structure

```text
helium/
├── backend/
│   ├── app/
│   │   ├── api/routes/
│   │   │   ├── upload.py
│   │   │   ├── jobs.py
│   │   │   └── artifacts.py
│   │   ├── services/
│   │   │   ├── job_service.py
│   │   │   └── upload_service.py
│   │   └── main.py
│   └── tests/
├── helium/
│   ├── cli/
│   ├── models/job.py
│   ├── pipeline/
│   │   ├── runner.py
│   │   └── stages/
│   │       ├── validate.py
│   │       ├── diarize.py
│   │       ├── separate.py
│   │       ├── convert.py
│   │       ├── evaluate.py
│   │       └── export.py
│   └── storage/local.py
└── README.md
```

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 0 - Pivot | ✅ Done | Repo moved from 3D reconstruction to speech research scaffold |
| 1 - Audio Jobs | ✅ Done | WAV uploads, local storage, job model, CLI |
| 2 - Research Scaffolding | ✅ Done | Diarization, separation, conversion, evaluation, export stages |
| 3 - Local Baselines | 🔲 TODO | Plug in pyannote, SepFormer, Seed-VC locally |
| 4 - Benchmarking | 🔲 TODO | Add Libri2Mix/WHAMR/VoxCeleb evaluation harness |
| 5 - Research Contribution | 🔲 TODO | Build and evaluate a diarization-guided conversion improvement |

## License

Apache 2.0
