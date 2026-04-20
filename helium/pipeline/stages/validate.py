"""
Stage: validate

Checks every uploaded image is readable and within acceptable bounds.
Runs before any compute-heavy steps so we fail fast on bad inputs.
"""

from pathlib import Path
from typing import Any, Dict, List

import cv2

from ...config import settings


class ValidationError(Exception):
    pass


def run(images_dir: Path, image_files: List[str]) -> Dict[str, Any]:
    results = []

    for filename in image_files:
        path = images_dir / filename

        if not path.exists():
            raise ValidationError(f"Image file missing on disk: {filename}")

        img = cv2.imread(str(path))
        if img is None:
            raise ValidationError(
                f"OpenCV could not decode '{filename}'. "
                "File may be corrupt or an unsupported format."
            )

        h, w = img.shape[:2]
        if h < 100 or w < 100:
            raise ValidationError(
                f"Image '{filename}' is too small ({w}x{h}). Minimum is 100x100 px."
            )

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > settings.max_image_size_mb:
            raise ValidationError(
                f"Image '{filename}' is {size_mb:.1f} MB, "
                f"exceeding the {settings.max_image_size_mb} MB limit."
            )

        results.append({"filename": filename, "width": w, "height": h, "size_mb": round(size_mb, 2)})

    return {"validated": len(results), "images": results}
