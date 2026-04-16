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
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

_SIFT_MAX_FEATURES = 2000


def run(images_dir: Path, artifacts_dir: Path, image_files: List[str]) -> Dict[str, Any]:
    features_dir = artifacts_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    detector = cv2.SIFT_create(nfeatures=_SIFT_MAX_FEATURES)
    results = []

    for filename in image_files:
        img = cv2.imread(str(images_dir / filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
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

        results.append({"filename": filename, "keypoints": len(keypoints)})

    return {"processed": len(results), "images": results}
