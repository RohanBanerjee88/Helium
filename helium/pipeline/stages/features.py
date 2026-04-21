"""
Stage: features

Extracts SIFT keypoints and descriptors from every image.

Outputs per image (written to artifacts/features/):
  {stem}_keypoints.json   — list of {x, y, size, angle, response, octave}
  {stem}_descriptors.npy  — float32 array of shape (N, 128)

SIFT is patent-free as of 2020 and gives good results for textured,
matte objects. For low-texture objects, consider ORB or SuperPoint.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

_SIFT_MAX_FEATURES = 2000
_MIN_USABLE_IMAGES = 2
logger = logging.getLogger(__name__)


def run(
    images_dir: Path,
    artifacts_dir: Path,
    image_files: List[str],
    progress_callback=None,
) -> Dict[str, Any]:
    features_dir = artifacts_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    cv2.setNumThreads(1)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)

    detector = cv2.SIFT_create(nfeatures=_SIFT_MAX_FEATURES)
    results = []
    usable_images = 0

    for filename in image_files:
        logger.info("Features: processing %s", filename)
        img = cv2.imread(str(images_dir / filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning("Features: skipping unreadable image %s", filename)
            continue

        keypoints, descriptors = detector.detectAndCompute(img, None)

        stem = Path(filename).stem

        kp_data = [
            {
                "x": float(kp.pt[0]),
                "y": float(kp.pt[1]),
                "size": float(kp.size),
                "angle": float(kp.angle),
                "response": float(kp.response),
                "octave": int(kp.octave),
            }
            for kp in keypoints
        ]
        with open(features_dir / f"{stem}_keypoints.json", "w") as fh:
            json.dump(kp_data, fh)

        if descriptors is not None:
            np.save(str(features_dir / f"{stem}_descriptors.npy"), descriptors)
            if descriptors.shape[0] >= 2:
                usable_images += 1

        results.append({"filename": filename, "keypoints": len(keypoints)})
        logger.info("Features: %s -> %d keypoints", filename, len(keypoints))
        if progress_callback is not None:
            progress_callback(filename, len(keypoints))

    if usable_images < _MIN_USABLE_IMAGES:
        raise RuntimeError(
            "Feature extraction produced usable descriptors for only "
            f"{usable_images} image(s); at least {_MIN_USABLE_IMAGES} are required. "
            "Use sharper, more textured photos with better overlap."
        )

    return {"processed": len(results), "images": results}
