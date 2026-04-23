"""
Stage: validate

Validates uploaded WAV files and records lightweight metadata before any
model-backed processing begins.
"""

import json
import wave
from pathlib import Path
from typing import Any, Dict, List


class ValidationError(RuntimeError):
    pass


def _inspect_wav(path: Path) -> Dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
    except (wave.Error, EOFError) as exc:
        raise ValidationError(f"'{path.name}' is not a readable WAV file.") from exc

    duration_seconds = frames / float(sample_rate) if sample_rate else 0.0
    if duration_seconds <= 0:
        raise ValidationError(f"'{path.name}' is empty.")

    return {
        "filename": path.name,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frames": frames,
        "duration_seconds": round(duration_seconds, 3),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
    }


def run(audio_dir: Path, artifacts_dir: Path, audio_files: List[str]) -> Dict[str, Any]:
    if not audio_files:
        raise ValidationError("No audio files were provided.")

    manifests_dir = artifacts_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    clips = []
    for name in audio_files:
        path = audio_dir / name
        if not path.exists():
            raise ValidationError(f"Audio file '{name}' is missing.")
        clips.append(_inspect_wav(path))

    total_duration = round(sum(item["duration_seconds"] for item in clips), 3)
    report = {
        "validated": len(clips),
        "total_duration_seconds": total_duration,
        "clips": clips,
        "recommended_backends": {
            "diarization": "pyannote/speaker-diarization",
            "separation": "speechbrain/sepformer-whamr",
            "conversion": "RedRepter/seed-vc-api",
        },
    }

    report_path = manifests_dir / "validation_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    return {
        "validated": len(clips),
        "total_duration_seconds": total_duration,
        "clips": clips,
        "artifacts": [str(report_path.relative_to(artifacts_dir))],
    }
