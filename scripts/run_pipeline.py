"""
Run the Helium pipeline on a directory of WAV files.

Usage:
    python scripts/run_pipeline.py --audio-dir ./sample_audio
    python scripts/run_pipeline.py --audio-dir ./sample_audio --target-speakers 2 --output ./results

Pipeline stages: validate → diarize → separate → convert → evaluate → export

Scaffold stages (diarize, separate, convert) emit planning artifacts and are
marked SKIPPED until you wire in local model backends. Edit the corresponding
stage files under helium/pipeline/stages/ to plug in your model.
"""

import argparse
import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helium.models.job import Job, JobStatus, StageStatus
from helium.pipeline.stages import convert, diarize, evaluate, export, separate, validate
from helium.storage.local import storage

_PLACEHOLDER = "placeholder"

_STAGE_FNS = {
    "validate": validate.run,
    "diarize": diarize.run,
    "separate": separate.run,
    "convert": convert.run,
    "evaluate": evaluate.run,
    "export": export.run,
}

_STAGE_ARGS = {
    # (audio_dir, artifacts_dir, audio_files, [target_speakers])
    "validate": lambda audio_dir, artifacts_dir, audio_files, n: (audio_dir, artifacts_dir, audio_files),
    "diarize":  lambda audio_dir, artifacts_dir, audio_files, n: (audio_dir, artifacts_dir, audio_files, n),
    "separate": lambda audio_dir, artifacts_dir, audio_files, n: (audio_dir, artifacts_dir, audio_files, n),
    "convert":  lambda audio_dir, artifacts_dir, audio_files, n: (audio_dir, artifacts_dir, audio_files, n),
    "evaluate": lambda audio_dir, artifacts_dir, audio_files, n: (artifacts_dir, audio_files, n),
    "export":   lambda audio_dir, artifacts_dir, audio_files, n: (artifacts_dir, None),  # job.id patched below
}


def _run_stage(job, stage_name, fn, *args):
    stage = job.stages[stage_name]
    stage.status = StageStatus.RUNNING
    stage.started_at = datetime.now(timezone.utc)
    storage.save_job(job)
    print(f"  [{stage_name}] running...", flush=True)

    try:
        result = fn(*args)
        if isinstance(result, dict) and result.get("status") == _PLACEHOLDER:
            stage.status = StageStatus.SKIPPED
            stage.message = result.get("note", "Not yet implemented.")
            if "artifacts" in result:
                stage.artifacts = list(result["artifacts"])
            print(f"  [{stage_name}] skipped (scaffold placeholder)")
        else:
            stage.status = StageStatus.COMPLETED
            stage.message = str(result)
            if isinstance(result, dict) and "artifacts" in result:
                stage.artifacts = list(result["artifacts"])
            print(f"  [{stage_name}] completed")
        stage.completed_at = datetime.now(timezone.utc)
        storage.save_job(job)
    except Exception as exc:
        stage.status = StageStatus.FAILED
        stage.message = str(exc)
        stage.completed_at = datetime.now(timezone.utc)
        job.status = JobStatus.FAILED
        job.error = f"Stage '{stage_name}' failed: {exc}"
        storage.save_job(job)
        raise


def run(audio_dir: Path, target_speakers: int = 2, output_dir: Path = None) -> Job:
    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {audio_dir}", file=sys.stderr)
        sys.exit(1)

    job = Job(
        audio_count=len(wav_files),
        audio_files=[f.name for f in wav_files],
        target_speakers=target_speakers,
    )
    storage.create_job_dirs(job.id)

    # Copy WAV files into job audio dir
    job_audio_dir = storage.audio_dir(job.id)
    for wav in wav_files:
        shutil.copy2(wav, job_audio_dir / wav.name)

    artifacts_dir = storage.artifacts_dir(job.id)
    job.status = JobStatus.RUNNING
    storage.save_job(job)

    print(f"\nJob {job.id}")
    print(f"Audio files: {[f.name for f in wav_files]}")
    print(f"Target speakers: {target_speakers}\n")

    try:
        _run_stage(job, "validate", validate.run, job_audio_dir, artifacts_dir, job.audio_files)
        _run_stage(job, "diarize",  diarize.run,  job_audio_dir, artifacts_dir, job.audio_files, target_speakers)
        _run_stage(job, "separate", separate.run, job_audio_dir, artifacts_dir, job.audio_files, target_speakers)
        _run_stage(job, "convert",  convert.run,  job_audio_dir, artifacts_dir, job.audio_files, target_speakers)
        _run_stage(job, "evaluate", evaluate.run, artifacts_dir, job.audio_files, target_speakers)
        _run_stage(job, "export",   export.run,   artifacts_dir, job.id)

        job.status = JobStatus.COMPLETED
        storage.save_job(job)
        print(f"\nPipeline complete. Artifacts at: {artifacts_dir}")
    except Exception as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(artifacts_dir, output_dir / job.id, dirs_exist_ok=True)
        print(f"Artifacts copied to: {output_dir / job.id}")

    return job


def main():
    parser = argparse.ArgumentParser(description="Run the Helium speech research pipeline")
    parser.add_argument("--audio-dir", required=True, type=Path, help="Directory of WAV files to process")
    parser.add_argument("--target-speakers", type=int, default=2, help="Expected number of speakers (default: 2)")
    parser.add_argument("--output", type=Path, default=None, help="Copy artifacts here after completion")
    args = parser.parse_args()

    if not args.audio_dir.is_dir():
        print(f"Error: {args.audio_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    job = run(args.audio_dir, args.target_speakers, args.output)
    sys.exit(0 if job.status == JobStatus.COMPLETED else 1)


if __name__ == "__main__":
    main()
