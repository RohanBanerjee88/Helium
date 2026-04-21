"""
Stage: sfm (Structure from Motion)

Runs the COLMAP CLI sparse reconstruction pipeline and produces:
  artifacts/point_cloud/sparse.ply      — sparse 3D point cloud
  artifacts/point_cloud/cameras.json    — estimated camera poses per image

The full COLMAP workspace is kept under artifacts/colmap_workspace/ so the
dense stage can consume it later.
"""

import json
import logging
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from helium.config import settings

logger = logging.getLogger(__name__)


def _resolve_colmap_binary() -> Optional[str]:
    configured = settings.colmap_bin.strip()
    if not configured:
        return None

    if os.sep in configured:
        candidate = Path(configured)
        return str(candidate) if candidate.exists() else None

    return shutil.which(configured)


def _clean_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _run_colmap(
    colmap_bin: str,
    command: str,
    args: Sequence[str],
    step_name: str,
) -> None:
    cmd = [colmap_bin, command, *args]
    logger.info("SfM: running COLMAP %s", command)
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"COLMAP binary '{colmap_bin}' was not found while running {step_name}."
        ) from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        if detail:
            detail = detail.splitlines()[-1]
            raise RuntimeError(f"{step_name} failed: {detail}")
        raise RuntimeError(f"{step_name} failed with exit code {completed.returncode}.")


def _parse_images_txt(images_txt_path: Path) -> Dict[str, Any]:
    cameras: Dict[str, Any] = {}
    with open(images_txt_path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if len(parts) < 10:
                continue

            try:
                image_id = int(parts[0])
                qvec = [float(value) for value in parts[1:5]]
                tvec = [float(value) for value in parts[5:8]]
                camera_id = int(parts[8])
            except ValueError:
                continue

            name = " ".join(parts[9:])
            cameras[name] = {
                "image_id": image_id,
                "camera_id": camera_id,
                "rotation_matrix": _qvec_to_rotmat(qvec),
                "translation": tvec,
            }

    return cameras


def _parse_points3d_txt(points3d_txt_path: Path) -> int:
    count = 0
    with open(points3d_txt_path) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
    return count


def _qvec_to_rotmat(qvec: Sequence[float]) -> List[List[float]]:
    w, x, y, z = qvec
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        raise RuntimeError("COLMAP returned an invalid zero-length quaternion.")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return [
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ]


def _find_sparse_models(sparse_dir: Path) -> List[Path]:
    return sorted(path for path in sparse_dir.iterdir() if path.is_dir())


def _export_text_model(colmap_bin: str, model_dir: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_colmap(
        colmap_bin,
        "model_converter",
        [
            "--input_path",
            str(model_dir),
            "--output_path",
            str(output_dir),
            "--output_type",
            "TXT",
        ],
        step_name=f"text export for model '{model_dir.name}'",
    )
    cameras = _parse_images_txt(output_dir / "images.txt")
    num_points = _parse_points3d_txt(output_dir / "points3D.txt")
    return {"cameras": cameras, "num_points3d": num_points}


def _select_best_model(
    colmap_bin: str,
    sparse_dir: Path,
    text_models_dir: Path,
) -> Tuple[Path, Path, Dict[str, Any]]:
    models = _find_sparse_models(sparse_dir)
    if not models:
        raise RuntimeError(
            "SfM produced no reconstruction. "
            "Insufficient feature matches — ensure photos overlap by at least 60 % "
            "and that the subject has visible texture."
        )

    best: Optional[Tuple[Path, Path, Dict[str, Any]]] = None
    for model_dir in models:
        exported_dir = text_models_dir / model_dir.name
        model_info = _export_text_model(colmap_bin, model_dir, exported_dir)
        if best is None or len(model_info["cameras"]) > len(best[2]["cameras"]):
            best = (model_dir, exported_dir, model_info)

    if best is None:
        raise RuntimeError("SfM produced sparse models, but none could be analyzed.")
    return best


def run(images_dir: Path, artifacts_dir: Path, image_files: List[str]) -> Dict[str, Any]:
    colmap_bin = _resolve_colmap_binary()
    if not colmap_bin:
        return {
            "status": "placeholder",
            "note": (
                "COLMAP CLI is not installed or not on PATH. "
                "Install COLMAP locally or in Docker, or set HELIUM_COLMAP_BIN to the binary path."
            ),
            "image_count": len(image_files),
            "expected_outputs": ["sparse.ply", "cameras.json"],
        }

    point_cloud_dir = artifacts_dir / "point_cloud"
    point_cloud_dir.mkdir(parents=True, exist_ok=True)

    workspace = artifacts_dir / "colmap_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    database_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    text_models_dir = workspace / "text_models"

    for stale_path in (database_path, sparse_dir, text_models_dir):
        _clean_path(stale_path)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    text_models_dir.mkdir(parents=True, exist_ok=True)

    _run_colmap(
        colmap_bin,
        "feature_extractor",
        [
            "--database_path",
            str(database_path),
            "--image_path",
            str(images_dir),
            "--ImageReader.camera_model",
            settings.colmap_camera_model,
            # TODO: Re-enable COLMAP GPU SIFT in production environments that
            # provide a stable CUDA/OpenGL-capable runtime. We force CPU here
            # because the current Docker target is headless and non-CUDA.
            "--SiftExtraction.use_gpu",
            "0",
            "--SiftExtraction.num_threads",
            "1",
            "--SiftExtraction.max_num_features",
            str(settings.colmap_sift_max_num_features),
        ],
        step_name="feature extraction",
    )
    _run_colmap(
        colmap_bin,
        "exhaustive_matcher",
        [
            "--database_path",
            str(database_path),
            # TODO: Re-enable GPU matching alongside GPU SIFT once the
            # production container/runtime supports it reliably.
            "--SiftMatching.use_gpu",
            "0",
        ],
        step_name="feature matching",
    )
    _run_colmap(
        colmap_bin,
        "mapper",
        [
            "--database_path",
            str(database_path),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(sparse_dir),
        ],
        step_name="incremental mapping",
    )

    selected_model_dir, selected_text_dir, model_info = _select_best_model(
        colmap_bin,
        sparse_dir,
        text_models_dir,
    )

    num_images = len(model_info["cameras"])
    num_points = model_info["num_points3d"]
    if num_images < settings.colmap_min_registered_images:
        raise RuntimeError(
            f"SfM registered only {num_images} image(s); at least "
            f"{settings.colmap_min_registered_images} are needed. "
            "Try photos with more overlap and avoid featureless surfaces."
        )

    ply_path = point_cloud_dir / "sparse.ply"
    _clean_path(ply_path)
    _run_colmap(
        colmap_bin,
        "model_converter",
        [
            "--input_path",
            str(selected_model_dir),
            "--output_path",
            str(ply_path),
            "--output_type",
            "PLY",
        ],
        step_name=f"PLY export for model '{selected_model_dir.name}'",
    )

    cameras_path = point_cloud_dir / "cameras.json"
    with open(cameras_path, "w") as fh:
        json.dump(
            {
                name: {
                    "camera_id": camera["camera_id"],
                    "rotation_matrix": camera["rotation_matrix"],
                    "translation": camera["translation"],
                }
                for name, camera in model_info["cameras"].items()
            },
            fh,
            indent=2,
        )

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
        "selected_model": str(selected_model_dir.relative_to(artifacts_dir)),
        "selected_text_model": str(selected_text_dir.relative_to(artifacts_dir)),
        "artifacts": [
            str(ply_path.relative_to(artifacts_dir)),
            str(cameras_path.relative_to(artifacts_dir)),
        ],
    }
