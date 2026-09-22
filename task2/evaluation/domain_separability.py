"""Held-out source-versus-target domain-separability probe."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@torch.no_grad()
def extract_features(
    backbone, loader, device: torch.device
) -> tuple[np.ndarray, list[str]]:
    backbone.eval()
    batches: list[np.ndarray] = []
    identifiers: list[str] = []
    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        sample_ids = batch[-1]
        batches.append(backbone(images).cpu().numpy())
        identifiers.extend(str(sample_id) for sample_id in sample_ids)
    if not batches:
        raise ValueError("Cannot extract features from an empty loader")
    return np.concatenate(batches, axis=0), identifiers


def balanced_domain_separability(
    source_features: np.ndarray,
    target_features: np.ndarray,
    seed: int,
    test_fraction: float = 0.30,
    c: float = 1.0,
) -> dict[str, float | int]:
    """Fit the manual's balanced C=1 logistic source/target probe."""
    rng = np.random.default_rng(seed)
    count = min(len(source_features), len(target_features))
    if count < 4:
        raise ValueError("At least four examples per domain are required for the probe")
    source_indices = rng.choice(len(source_features), size=count, replace=False)
    target_indices = rng.choice(len(target_features), size=count, replace=False)
    features = np.concatenate(
        [source_features[source_indices], target_features[target_indices]], axis=0
    )
    domains = np.concatenate(
        [np.zeros(count, dtype=np.int64), np.ones(count, dtype=np.int64)]
    )
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        domains,
        test_size=test_fraction,
        random_state=seed,
        stratify=domains,
    )
    probe = LogisticRegression(C=c, max_iter=2000, random_state=seed)
    probe.fit(train_x, train_y)
    prediction = probe.predict(test_x)
    return {
        "domain_separability": float(accuracy_score(test_y, prediction)),
        "balanced_examples_per_domain": int(count),
        "probe_train_examples": len(train_y),
        "probe_test_examples": len(test_y),
    }
