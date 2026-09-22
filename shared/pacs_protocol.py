"""Deterministic PACS source splits and shared image transformations."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.models import ResNet18_Weights

from shared.pacs import PACS_CLASSES, PACSSample

SOURCE_DOMAINS = ("photo", "art_painting", "cartoon")
TARGET_DOMAIN = "sketch"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_source_splits(
    samples_by_domain: Mapping[str, Sequence[PACSSample]],
    validation_fraction: float,
    seed: int,
) -> dict[str, dict[str, list[str]]]:
    """Create the manual's independent stratified 80/20 source splits."""
    manifest: dict[str, dict[str, list[str]]] = {}
    for domain in SOURCE_DOMAINS:
        samples = list(samples_by_domain[domain])
        indices = np.arange(len(samples))
        labels = np.asarray([sample.label for sample in samples])
        train_indices, validation_indices = train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )
        manifest[domain] = {
            "train": [
                samples[index].sample_id for index in sorted(train_indices.tolist())
            ],
            "validation": [
                samples[index].sample_id
                for index in sorted(validation_indices.tolist())
            ],
        }
    return manifest


def save_split_manifest(
    path: str | Path,
    splits: Mapping[str, Mapping[str, Sequence[str]]],
    seed: int,
    validation_fraction: float,
) -> None:
    payload = {
        "dataset": "PACS",
        "source_domains": list(SOURCE_DOMAINS),
        "target_domain": TARGET_DOMAIN,
        "classes": list(PACS_CLASSES),
        "seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "splits": splits,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def load_split_manifest(
    path: str | Path,
    samples_by_domain: Mapping[str, Sequence[PACSSample]],
    expected_seed: int,
) -> dict[str, dict[str, list[PACSSample]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload["seed"] != expected_seed:
        raise ValueError("PACS split manifest seed does not match the experiment seed")
    result: dict[str, dict[str, list[PACSSample]]] = {}
    for domain in SOURCE_DOMAINS:
        lookup = {sample.sample_id: sample for sample in samples_by_domain[domain]}
        train_ids = payload["splits"][domain]["train"]
        validation_ids = payload["splits"][domain]["validation"]
        missing = (set(train_ids) | set(validation_ids)) - set(lookup)
        if missing:
            raise ValueError(
                f"Split manifest references missing {domain} samples: {sorted(missing)[:3]}"
            )
        if set(train_ids) & set(validation_ids):
            raise ValueError(f"Train/validation overlap detected for {domain}")
        if set(train_ids) | set(validation_ids) != set(lookup):
            raise ValueError(f"Split manifest does not exactly cover {domain}")
        result[domain] = {
            "train": [lookup[sample_id] for sample_id in train_ids],
            "validation": [lookup[sample_id] for sample_id in validation_ids],
        }
    return result


def build_transforms(resize_size: int = 256, crop_size: int = 224):
    weights = ResNet18_Weights.IMAGENET1K_V1
    preset = weights.transforms()
    normalization = transforms.Normalize(mean=preset.mean, std=preset.std)
    train_transform = transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, evaluation_transform
