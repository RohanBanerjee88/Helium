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
    diarization_dir = artifacts_dir / "diarization"
    diarization_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "stage": "diarize",
        "status": "placeholder",
        "backend": settings.diarization_backend,
        "target_speakers": target_speakers,
        "input_files": audio_files,
        "expected_outputs": ["speaker_turns.rttm", "speaker_turns.json"],
        "notes": [
            "This scaffold is ready for pyannote integration but does not bundle gated checkpoints.",
            "Recommended next step: load pyannote locally, pass num_speakers=2, and store RTTM plus JSON turns here.",
        ],
        "primary_input_dir": str(audio_dir),
    }

    plan_path = diarization_dir / "diarization_plan.json"
    with open(plan_path, "w") as fh:
        json.dump(plan, fh, indent=2)

    return {
        "status": "placeholder",
        "note": (
            "Speaker diarization backend not wired yet. "
            f"Plug in '{settings.diarization_backend}' and save RTTM or JSON turns."
        ),
        "artifacts": [str(plan_path.relative_to(artifacts_dir))],
    }
