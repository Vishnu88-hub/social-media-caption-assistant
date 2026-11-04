"""Safe in-memory image validation and preprocessing."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_TYPES = {"JPEG", "PNG", "WEBP"}


class ImageValidationError(ValueError):
    """Raised when an uploaded image is invalid or unsafe to process."""


def process_image(data: bytes, *, max_bytes: int = 10 * 1024 * 1024, max_dimension: int = 1600) -> bytes:
    if not data:
        raise ImageValidationError("The uploaded image is empty.")
    if len(data) > max_bytes:
        raise ImageValidationError(f"Image exceeds the {max_bytes // (1024 * 1024)} MB limit.")
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in SUPPORTED_TYPES:
                raise ImageValidationError("Use a JPG, PNG, or WEBP image.")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            output = BytesIO()
            image.save(output, format="JPEG", quality=88, optimize=True)
            return output.getvalue()
    except UnidentifiedImageError as exc:
        raise ImageValidationError("The uploaded file is not a valid image.") from exc


def image_dimensions(data: bytes) -> tuple[int, int]:
    try:
        with Image.open(BytesIO(data)) as image:
            return image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError("The uploaded file is not a valid image.") from exc
