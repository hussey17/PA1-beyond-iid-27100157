"""Controlled Task 1 image interventions.

All functions operate on the common 224x224 RGB image and return a new PIL
image.  The generated files are cached before model evaluation so every model
receives byte-identical transformed inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import functional as TF


def grayscale_rgb(image: Image.Image) -> Image.Image:
    """Remove chromatic information while retaining three input channels."""

    return TF.rgb_to_grayscale(image.convert("RGB"), num_output_channels=3)


def rotate_hue(image: Image.Image, degrees: float = 90.0) -> Image.Image:
    """Rotate hue by a fixed angle while preserving HSV saturation and value."""

    if not -180.0 <= degrees <= 180.0:
        raise ValueError("torchvision hue rotation must lie in [-180, 180] degrees")
    return TF.adjust_hue(image.convert("RGB"), hue_factor=degrees / 360.0)


def translate_reflect(
    image: Image.Image, displacement: int, direction: str
) -> Image.Image:
    """Translate with reflection padding followed by a shifted 224x224 crop."""

    if displacement == 0:
        return image.copy()
    direction_to_offset = {
        "up": (0, -displacement),
        "down": (0, displacement),
        "left": (-displacement, 0),
        "right": (displacement, 0),
    }
    if direction not in direction_to_offset:
        raise ValueError(f"Unknown direction: {direction}")

    array = np.asarray(image.convert("RGB"))
    height, width = array.shape[:2]
    padded = np.pad(
        array,
        ((displacement, displacement), (displacement, displacement), (0, 0)),
        mode="reflect",
    )
    dx, dy = direction_to_offset[direction]
    y0 = displacement - dy
    x0 = displacement - dx
    shifted = padded[y0 : y0 + height, x0 : x0 + width]
    return Image.fromarray(shifted.astype(np.uint8), mode="RGB")


def shuffle_patch_grid(
    image: Image.Image, image_seed: int, grid_size: int = 4
) -> tuple[Image.Image, list[int]]:
    """Apply one deterministic non-identity permutation of a square patch grid."""

    array = np.asarray(image.convert("RGB"))
    height, width = array.shape[:2]
    if height % grid_size or width % grid_size:
        raise ValueError("Image dimensions must be divisible by grid_size")
    patch_h, patch_w = height // grid_size, width // grid_size
    patches = [
        array[
            row * patch_h : (row + 1) * patch_h,
            col * patch_w : (col + 1) * patch_w,
        ].copy()
        for row in range(grid_size)
        for col in range(grid_size)
    ]
    rng = np.random.default_rng(image_seed)
    identity = np.arange(grid_size * grid_size)
    permutation = rng.permutation(identity)
    while np.array_equal(permutation, identity):
        permutation = rng.permutation(identity)

    output = np.empty_like(array)
    for destination, source in enumerate(permutation):
        row, col = divmod(destination, grid_size)
        output[
            row * patch_h : (row + 1) * patch_h,
            col * patch_w : (col + 1) * patch_w,
        ] = patches[int(source)]
    return Image.fromarray(output, mode="RGB"), permutation.tolist()


def generate_simple_intervention(
    clean_manifest: pd.DataFrame,
    output_dir: Path,
    transform_name: str,
    transform: Callable[[Image.Image], Image.Image],
) -> pd.DataFrame:
    """Generate one cached transformed image per clean example."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in clean_manifest.itertuples(index=False):
        output_path = output_dir / f"{row.image_id}.png"
        if not output_path.exists():
            image = Image.open(row.path).convert("RGB")
            transform(image).save(output_path)
        rows.append(
            {
                "image_id": row.image_id,
                "original_index": row.original_index,
                "label": row.label,
                "class_name": row.class_name,
                "condition": transform_name,
                "path": str(output_path),
            }
        )
    return pd.DataFrame(rows)


def generate_patch_shuffle_set(
    clean_manifest: pd.DataFrame, output_dir: Path, seed: int, grid_size: int = 4
) -> pd.DataFrame:
    """Generate and record a distinct deterministic permutation for each image."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in clean_manifest.itertuples(index=False):
        output_path = output_dir / f"{row.image_id}.png"
        image = Image.open(row.path).convert("RGB")
        shuffled, permutation = shuffle_patch_grid(
            image, image_seed=seed + int(row.original_index), grid_size=grid_size
        )
        if not output_path.exists():
            shuffled.save(output_path)
        rows.append(
            {
                "image_id": row.image_id,
                "original_index": row.original_index,
                "label": row.label,
                "class_name": row.class_name,
                "condition": "patch_shuffle",
                "permutation": ",".join(map(str, permutation)),
                "path": str(output_path),
            }
        )
    return pd.DataFrame(rows)


def generate_translation_sets(
    clean_manifest: pd.DataFrame,
    output_root: Path,
    displacements: list[int],
    directions: list[str],
) -> pd.DataFrame:
    """Generate every required displacement/direction combination once."""

    rows = []
    for displacement in displacements:
        active_directions = ["none"] if displacement == 0 else directions
        for direction in active_directions:
            condition_dir = output_root / f"delta_{displacement}" / direction
            condition_dir.mkdir(parents=True, exist_ok=True)
            for row in clean_manifest.itertuples(index=False):
                output_path = condition_dir / f"{row.image_id}.png"
                if not output_path.exists():
                    image = Image.open(row.path).convert("RGB")
                    transformed = (
                        image
                        if displacement == 0
                        else translate_reflect(image, displacement, direction)
                    )
                    transformed.save(output_path)
                rows.append(
                    {
                        "image_id": row.image_id,
                        "original_index": row.original_index,
                        "label": row.label,
                        "class_name": row.class_name,
                        "condition": "translation",
                        "displacement": displacement,
                        "direction": direction,
                        "path": str(output_path),
                    }
                )
    return pd.DataFrame(rows)
