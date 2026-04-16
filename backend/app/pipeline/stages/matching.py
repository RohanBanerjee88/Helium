"""
Stage: matching

Matches SIFT descriptors between all pairs of images using:
  - BFMatcher (L2 norm)
  - Lowe ratio test (0.75 threshold) to reject ambiguous matches

Outputs per pair (written to artifacts/matches/):
  matches_{i:03d}_{j:03d}.json — {image_i, image_j, matches: [{queryIdx, trainIdx, distance}]}

Only adjacent and cross-image pairs are attempted. For large image sets,
consider a vocabulary-tree-based retrieval to limit pair count.
"""

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

_LOWE_RATIO = 0.75
_MIN_DESCRIPTORS = 2


def run(images_dir: Path, artifacts_dir: Path, image_files: List[str]) -> Dict[str, Any]:
    features_dir = artifacts_dir / "features"
    matches_dir = artifacts_dir / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    stems = [Path(f).stem for f in image_files]

    pairs_processed = 0
    total_good = 0

    for i, j in combinations(range(len(stems)), 2):
        desc_i_path = features_dir / f"{stems[i]}_descriptors.npy"
        desc_j_path = features_dir / f"{stems[j]}_descriptors.npy"

        if not desc_i_path.exists() or not desc_j_path.exists():
            continue

        desc_i = np.load(str(desc_i_path))
        desc_j = np.load(str(desc_j_path))

        if desc_i.shape[0] < _MIN_DESCRIPTORS or desc_j.shape[0] < _MIN_DESCRIPTORS:
            continue

        raw = matcher.knnMatch(desc_i, desc_j, k=2)

        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < _LOWE_RATIO * n.distance:
                    good.append({
                        "queryIdx": int(m.queryIdx),
                        "trainIdx": int(m.trainIdx),
                        "distance": float(m.distance),
                    })

        match_data = {
            "image_i": image_files[i],
            "image_j": image_files[j],
            "total_candidates": len(raw),
            "good_matches": len(good),
            "matches": good,
        }

        out_path = matches_dir / f"matches_{i:03d}_{j:03d}.json"
        with open(out_path, "w") as fh:
            json.dump(match_data, fh)

        pairs_processed += 1
        total_good += len(good)

    return {
        "pairs_processed": pairs_processed,
        "total_good_matches": total_good,
        "avg_good_per_pair": round(total_good / pairs_processed, 1) if pairs_processed else 0,
    }
