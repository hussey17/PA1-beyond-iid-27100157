"""Paired cosine representation-stability calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rowwise_cosine(clean_features: np.ndarray, transformed_features: np.ndarray):
    """Return one cosine similarity for each paired clean/transformed example."""

    if clean_features.shape != transformed_features.shape:
        raise ValueError(
            f"Paired feature shapes differ: {clean_features.shape} vs "
            f"{transformed_features.shape}"
        )
    numerator = np.sum(clean_features * transformed_features, axis=1)
    denominator = np.linalg.norm(clean_features, axis=1) * np.linalg.norm(
        transformed_features, axis=1
    )
    return numerator / np.clip(denominator, 1e-12, None)


def stability_frame(
    image_ids,
    labels,
    similarities,
    backbone: str,
    intervention: str,
    **metadata,
) -> pd.DataFrame:
    """Create traceable per-example stability records."""

    frame = pd.DataFrame(
        {
            "image_id": list(image_ids),
            "label": labels,
            "cosine_similarity": similarities,
            "backbone": backbone,
            "intervention": intervention,
        }
    )
    for key, value in metadata.items():
        frame[key] = value
    return frame


def summarize_stability(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean cosine stability without discarding per-example records."""

    grouping = [
        column
        for column in ["backbone", "intervention", "displacement", "direction"]
        if column in frame.columns
    ]
    return (
        frame.groupby(grouping, dropna=False).cosine_similarity
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_cosine_stability"})
    )
