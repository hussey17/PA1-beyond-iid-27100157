"""Deterministic STL-10 splits and image export helpers.

The assignment uses the official STL-10 training partition for an 80/20
stratified head-training split and a class-balanced 500-image subset of the
official test partition.  Saving the selected original indices is essential:
every backbone and intervention must evaluate the same underlying examples.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


STL10_CLASSES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic CUDA behavior."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_train_val_indices(
    targets: Sequence[int], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Return deterministic stratified indices into the official train split."""

    all_indices = np.arange(len(targets))
    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=np.asarray(targets),
    )
    return sorted(train_indices.tolist()), sorted(validation_indices.tolist())


def balanced_test_indices(
    targets: Sequence[int], examples_per_class: int, seed: int
) -> list[int]:
    """Select the same number of official test examples from every class."""

    targets_array = np.asarray(targets)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in sorted(np.unique(targets_array)):
        candidates = np.flatnonzero(targets_array == class_id)
        if len(candidates) < examples_per_class:
            raise ValueError(
                f"Class {class_id} has {len(candidates)} examples; "
                f"cannot select {examples_per_class}."
            )
        chosen = rng.choice(candidates, size=examples_per_class, replace=False)
        selected.extend(sorted(chosen.tolist()))
    return selected


def save_split_manifest(
    output_path: Path,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    test_indices: Sequence[int],
    seed: int,
) -> None:
    """Persist original dataset indices so the exact protocol is recoverable."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "official_train_indices": list(train_indices),
        "official_validation_indices": list(validation_indices),
        "official_test_subset_indices": list(test_indices),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def common_geometry(resize_shorter_edge: int = 256, crop_size: int = 224):
    """Create the shared deterministic geometry applied before normalization."""

    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize(resize_shorter_edge, antialias=True),
            transforms.CenterCrop(crop_size),
        ]
    )


class IndexedImageDataset(Dataset):
    """Apply common geometry and model normalization to selected base images."""

    def __init__(
        self,
        base_dataset: Dataset,
        indices: Sequence[int],
        tensor_transform: Callable[[Image.Image], torch.Tensor],
        geometry: Callable[[Image.Image], Image.Image],
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.tensor_transform = tensor_transform
        self.geometry = geometry

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        original_index = self.indices[position]
        image, label = self.base_dataset[original_index]
        image = self.geometry(image)
        return self.tensor_transform(image), int(label), int(original_index)


class FileImageDataset(Dataset):
    """Load an intervention manifest while preserving identifiers and labels."""

    def __init__(self, frame, tensor_transform: Callable[[Image.Image], torch.Tensor]):
        self.frame = frame.reset_index(drop=True).copy()
        self.tensor_transform = tensor_transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, position: int):
        row = self.frame.iloc[position]
        image = Image.open(row["path"]).convert("RGB")
        return (
            self.tensor_transform(image),
            int(row["label"]),
            str(row["image_id"]),
        )


def export_clean_subset(
    dataset: Dataset,
    indices: Sequence[int],
    output_dir: Path,
    geometry: Callable[[Image.Image], Image.Image],
):
    """Export the fixed 224x224 clean images and return their manifest."""

    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for subset_position, original_index in enumerate(indices):
        image, label = dataset[original_index]
        image = geometry(image)
        image_id = f"test_{original_index:05d}"
        path = output_dir / f"{image_id}.png"
        if not path.exists():
            image.save(path)
        rows.append(
            {
                "image_id": image_id,
                "subset_position": subset_position,
                "original_index": int(original_index),
                "label": int(label),
                "class_name": STL10_CLASSES[int(label)],
                "path": str(path),
            }
        )
    return pd.DataFrame(rows)
