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
    conversion_dir = artifacts_dir / "conversion"
    conversion_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "stage": "convert",
        "status": "placeholder",
        "backend": settings.conversion_backend,
        "target_speakers": target_speakers,
        "input_files": audio_files,
        "expected_outputs": [
            "speaker_0_converted.wav",
            "speaker_1_converted.wav",
            "conversion_manifest.json",
        ],
        "controls": {
            "target_voice_gender_swap": "example_only",
            "preserve_content": True,
            "preserve_rhythm": True,
            "preserve_turn_taking": True,
        },
        "notes": [
            "Recommended baseline: Seed-VC for style-preserving voice conversion.",
            "Recommended ablation: compare against OpenVoice V2 or GenVC.",
        ],
        "primary_input_dir": str(audio_dir),
    }

    plan_path = conversion_dir / "conversion_plan.json"
    with open(plan_path, "w") as fh:
        json.dump(plan, fh, indent=2)

    return {
        "status": "placeholder",
        "note": (
            "Voice conversion backend not wired yet. "
            f"Plug in '{settings.conversion_backend}' and save converted speech here."
        ),
        "artifacts": [str(plan_path.relative_to(artifacts_dir))],
    }
