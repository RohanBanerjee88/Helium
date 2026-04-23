import json
from pathlib import Path
from typing import Any, Dict, List

from ...config import settings


def run(artifacts_dir: Path, audio_files: List[str], target_speakers: int) -> Dict[str, Any]:
    evaluation_dir = artifacts_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "stage": "evaluate",
        "target_speakers": target_speakers,
        "input_files": audio_files,
        "metrics": [
            "DER",
            "SI-SDR",
            "SDRi",
            "WER",
            "speaker_similarity",
            "MOS",
        ],
        "baselines": {
            "diarization": settings.diarization_backend,
            "separation": settings.separation_backend,
            "conversion": settings.conversion_backend,
        },
        "research_question": (
            "Can diarization-guided source separation plus style-preserving voice conversion "
            "outperform plain separation-plus-conversion baselines in noisy two-speaker mixtures?"
        ),
        "recommended_datasets": [
            "WHAMR! for noisy reverberant separation",
            "Libri2Mix for controllable two-speaker mixtures",
            "VoxCeleb for speaker identity and similarity checks",
            "A small in-house cafe benchmark for real-world robustness",
        ],
    }

    report_path = evaluation_dir / "evaluation_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    return {
        "metrics_planned": report["metrics"],
        "dataset_count": len(report["recommended_datasets"]),
        "artifacts": [str(report_path.relative_to(artifacts_dir))],
    }
