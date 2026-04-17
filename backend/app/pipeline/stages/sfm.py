"""
Stage: sfm (Structure from Motion)

Uses pycolmap to run incremental SfM and produce:
  artifacts/point_cloud/sparse.ply      — sparse 3D point cloud
  artifacts/point_cloud/cameras.json    — estimated camera poses per image

Falls back to a placeholder when pycolmap is not installed.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_PYCOLMAP_AVAILABLE = False
try:
    import pycolmap  # type: ignore[import]
    _PYCOLMAP_AVAILABLE = True
except ImportError:
    pass


def _export_cameras(reconstruction: Any, out_path: Path) -> None:
    cameras: Dict[str, Any] = {}
    for _image_id, image in reconstruction.images.items():
        cameras[image.name] = {
            "camera_id": image.camera_id,
            "rotation_matrix": image.rotmat().tolist(),
            "translation": image.tvec.tolist(),
        }
    with open(out_path, "w") as fh:
        json.dump(cameras, fh, indent=2)


def run(images_dir: Path, artifacts_dir: Path, image_files: List[str]) -> Dict[str, Any]:
    if not _PYCOLMAP_AVAILABLE:
        return {
            "status": "placeholder",
            "note": (
                "pycolmap not installed. "
                "Add 'pycolmap>=0.6.0' to requirements.txt and rebuild the container."
            ),
            "image_count": len(image_files),
            "expected_outputs": ["sparse.ply", "cameras.json"],
        }

    point_cloud_dir = artifacts_dir / "point_cloud"
    point_cloud_dir.mkdir(parents=True, exist_ok=True)

    workspace = artifacts_dir / "colmap_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    database_path = workspace / "database.db"
    sparse_path = workspace / "sparse"
    sparse_path.mkdir(parents=True, exist_ok=True)

    # Remove stale database so re-runs start clean
    if database_path.exists():
        database_path.unlink()

    logger.info("SfM: extracting features for %d images in %s", len(image_files), images_dir)
    pycolmap.extract_features(database_path, images_dir)

    logger.info("SfM: exhaustive feature matching")
    pycolmap.match_exhaustive(database_path)

    logger.info("SfM: incremental mapping")
    maps = pycolmap.incremental_mapping(database_path, images_dir, sparse_path)

    if not maps:
        raise RuntimeError(
            "SfM produced no reconstruction. "
            "Insufficient feature matches — ensure photos overlap by at least 60 % "
            "and that the subject has visible texture."
        )

    # Use the reconstruction with the most registered images
    reconstruction = max(maps.values(), key=lambda r: len(r.images))

    num_images = len(reconstruction.images)
    num_points = len(reconstruction.points3D)

    if num_images < 3:
        raise RuntimeError(
            f"SfM registered only {num_images} image(s); at least 3 are needed. "
            "Try photos with more overlap and avoid featureless surfaces."
        )

    ply_path = point_cloud_dir / "sparse.ply"
    reconstruction.export_PLY(str(ply_path))

    cameras_path = point_cloud_dir / "cameras.json"
    _export_cameras(reconstruction, cameras_path)

    logger.info(
        "SfM complete: %d/%d images registered, %d 3D points",
        num_images,
        len(image_files),
        num_points,
    )

    return {
        "num_images_registered": num_images,
        "num_images_total": len(image_files),
        "num_points3d": num_points,
        "artifacts": [
            str(ply_path.relative_to(artifacts_dir)),
            str(cameras_path.relative_to(artifacts_dir)),
        ],
    }
