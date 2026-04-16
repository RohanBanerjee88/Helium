import shutil
from pathlib import Path
from typing import List

from fastapi import HTTPException, UploadFile

from ..config import settings
from ..models.job import Job
from ..storage.local import storage

_ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/tif",
}

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _validate_file_meta(file: UploadFile) -> None:
    if file.content_type and file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}' for '{file.filename}'. "
                   f"Accepted: JPEG, PNG, BMP, TIFF.",
        )
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in _ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported extension '{ext}' for '{file.filename}'.",
            )


def save_images(job: Job, files: List[UploadFile]) -> List[str]:
    if len(files) < settings.min_images:
        raise HTTPException(
            status_code=400,
            detail=f"At least {settings.min_images} images are required; got {len(files)}.",
        )
    if len(files) > settings.max_images:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.max_images} images allowed; got {len(files)}.",
        )

    for file in files:
        _validate_file_meta(file)

    images_dir = storage.images_dir(job.id)
    saved: List[str] = []

    for idx, file in enumerate(files):
        ext = Path(file.filename).suffix.lower() if file.filename else ".jpg"
        filename = f"img_{idx:03d}{ext}"
        dest = images_dir / filename

        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)

        file_size_mb = dest.stat().st_size / (1024 * 1024)
        if file_size_mb > settings.max_image_size_mb:
            dest.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail=f"Image '{file.filename}' exceeds {settings.max_image_size_mb} MB limit "
                       f"({file_size_mb:.1f} MB).",
            )

        saved.append(filename)

    return saved
