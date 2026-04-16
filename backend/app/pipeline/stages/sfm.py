"""
Stage: sfm (Structure from Motion)

TODO: This is a placeholder. Implement one of the following:

Option A — COLMAP CLI (recommended for production quality):
    import subprocess
    subprocess.run([
        "colmap", "automatic_reconstructor",
        "--workspace_path", str(workspace),
        "--image_path", str(images_dir),
    ], check=True)

Option B — pycolmap Python bindings:
    import pycolmap
    pycolmap.extract_features(database_path, images_dir)
    pycolmap.match_exhaustive(database_path)
    maps = pycolmap.incremental_mapping(database_path, images_dir, workspace)

Option C — OpenCV essential matrix (educational, lower quality):
    Use matched keypoint coordinates to estimate E via cv2.findEssentialMat,
    recover R and t via cv2.recoverPose, and triangulate with
    cv2.triangulatePoints.

Expected outputs:
    artifacts/point_cloud/sparse.ply      — sparse 3D point cloud
    artifacts/point_cloud/cameras.json    — estimated camera poses per image
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def run(images_dir: Path, artifacts_dir: Path, image_files: List[str]) -> Dict[str, Any]:
    point_cloud_dir = artifacts_dir / "point_cloud"
    point_cloud_dir.mkdir(parents=True, exist_ok=True)

    status = {
        "status": "placeholder",
        "note": (
            "SfM not yet implemented. "
            "Connect COLMAP or pycolmap here to produce sparse.ply and cameras.json."
        ),
        "image_count": len(image_files),
        "expected_outputs": ["sparse.ply", "cameras.json"],
    }

    with open(point_cloud_dir / "sfm_status.json", "w") as fh:
        json.dump(status, fh, indent=2)

    return status
