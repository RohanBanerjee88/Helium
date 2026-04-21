"""
Stage: dense (Dense Reconstruction)

TODO: This is a placeholder. Implement one of the following:

Option A — COLMAP dense pipeline (MVS):
    subprocess.run(["colmap", "patch_match_stereo", ...], check=True)
    subprocess.run(["colmap", "stereo_fusion", ...], check=True)

Option B — Open3D TSDF fusion:
    Requires per-image depth maps (from depth camera or estimated depth).
    volume = o3d.pipelines.integration.ScalableTSDFVolume(...)
    for color, depth in image_pairs:
        volume.integrate(rgbd, intrinsic, extrinsic)
    mesh = volume.extract_triangle_mesh()

Option C — OpenMVS:
    External tool; call via subprocess similar to COLMAP.

Expected inputs:
    artifacts/point_cloud/sparse.ply      — from sfm stage
    artifacts/point_cloud/cameras.json    — camera poses from sfm stage
    artifacts/colmap_workspace/           — persisted COLMAP workspace for dense prep

Expected outputs:
    artifacts/point_cloud/dense.ply       — dense point cloud
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def run(artifacts_dir: Path, image_files: List[str]) -> Dict[str, Any]:
    point_cloud_dir = artifacts_dir / "point_cloud"
    point_cloud_dir.mkdir(parents=True, exist_ok=True)

    status = {
        "status": "placeholder",
        "note": (
            "Dense reconstruction not yet implemented. "
            "Connect COLMAP MVS or Open3D TSDF fusion here to produce dense.ply."
        ),
        "expected_inputs": ["sparse.ply", "cameras.json"],
        "expected_outputs": ["dense.ply"],
    }

    with open(point_cloud_dir / "dense_status.json", "w") as fh:
        json.dump(status, fh, indent=2)

    return status
