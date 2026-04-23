import json
from pathlib import Path
from typing import Any, Dict


def run(artifacts_dir: Path, job_id: str) -> Dict[str, Any]:
    export_dir = artifacts_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    artifact_files = [
        str(path.relative_to(artifacts_dir))
        for path in sorted(artifacts_dir.rglob("*"))
        if path.is_file() and path.parent != export_dir
    ]

    manifest = {
        "job_id": job_id,
        "artifact_count": len(artifact_files),
        "artifacts": artifact_files,
        "next_steps": [
            "Integrate pyannote diarization outputs.",
            "Integrate local speech separation to emit isolated stems.",
            "Integrate style-preserving voice conversion and compare against baselines.",
        ],
    }

    manifest_path = export_dir / "run_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    return {
        "artifact_count": len(artifact_files),
        "artifacts": [str(manifest_path.relative_to(artifacts_dir))],
    }
