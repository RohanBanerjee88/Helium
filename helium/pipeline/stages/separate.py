import json
from pathlib import Path
from typing import Any, Dict, List

from ...config import settings


def run(
    audio_dir: Path,
    artifacts_dir: Path,
    audio_files: List[str],
    target_speakers: int,
) -> Dict[str, Any]:
    separation_dir = artifacts_dir / "separation"
    separation_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "stage": "separate",
        "status": "placeholder",
        "backend": settings.separation_backend,
        "target_speakers": target_speakers,
        "input_files": audio_files,
        "expected_outputs": [
            "speaker_0.wav",
            "speaker_1.wav",
            "separation_manifest.json",
        ],
        "research_hypothesis": (
            "Compare diarization-guided separation against plain mixture separation "
            "for noisy two-speaker conversations."
        ),
        "notes": [
            "Start with SepFormer on WHAMR-style mixtures.",
            "Then add overlap-aware conditioning from diarization segments.",
        ],
        "primary_input_dir": str(audio_dir),
    }

    plan_path = separation_dir / "separation_plan.json"
    with open(plan_path, "w") as fh:
        json.dump(plan, fh, indent=2)

    return {
        "status": "placeholder",
        "note": (
            "Speech separation backend not wired yet. "
            f"Plug in '{settings.separation_backend}' and write isolated stems here."
        ),
        "artifacts": [str(plan_path.relative_to(artifacts_dir))],
    }
