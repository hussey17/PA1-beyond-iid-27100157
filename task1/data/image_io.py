"""Reliable image-file validation and atomic PNG writes.

Colab runtimes can be interrupted while an image is being written. A partial
file still satisfies ``Path.exists()``, so a naive cache check can preserve a
corrupt PNG and fail much later inside a DataLoader worker. These helpers make
generated-image caches self-healing: reuse only a decodable image of the
expected size, and publish new files only after a complete temporary write.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def is_valid_image_file(
    path: str | Path, expected_size: tuple[int, int] | None = (224, 224)
) -> bool:
    """Return whether ``path`` is a complete, decodable image of the right size."""

    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        # verify() checks file structure without decoding pixel data.
        with Image.open(path) as image:
            image.verify()
        # Reopen because verify() invalidates the first Image object; load()
        # forces full decoding so truncated pixel data is also detected.
        with Image.open(path) as image:
            image.load()
            if expected_size is not None and image.size != expected_size:
                return False
        return True
    except (OSError, ValueError, UnidentifiedImageError):
        return False


def atomic_save_pil(image: Image.Image, destination: str | Path) -> None:
    """Write a PNG completely, then atomically replace the destination path."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}_",
        suffix=".tmp.png",
        dir=destination.parent,
        delete=False,
    )
    temporary_path = Path(handle.name)
    handle.close()
    try:
        image.save(temporary_path, format="PNG")
        if not is_valid_image_file(temporary_path, expected_size=image.size):
            raise OSError(f"Temporary PNG failed validation: {temporary_path}")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_save_tensor(image_tensor, destination: str | Path) -> None:
    """Atomically save a CHW image tensor through torchvision's PNG writer."""

    from torchvision.utils import save_image

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}_",
        suffix=".tmp.png",
        dir=destination.parent,
        delete=False,
    )
    temporary_path = Path(handle.name)
    handle.close()
    try:
        save_image(image_tensor, temporary_path, format="png")
        if not is_valid_image_file(temporary_path):
            raise OSError(f"Temporary tensor PNG failed validation: {temporary_path}")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def invalid_manifest_paths(frame, path_column: str = "path") -> list[str]:
    """List corrupt/missing generated files referenced by a manifest."""

    return [
        str(path)
        for path in frame[path_column].astype(str).unique()
        if not is_valid_image_file(path)
    ]
