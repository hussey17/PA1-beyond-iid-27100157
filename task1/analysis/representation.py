"""UMAP projections fitted jointly to clean and transformed features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def joint_umap_projection(
    clean_features: np.ndarray,
    transformed_features: np.ndarray,
    labels,
    image_ids,
    backbone: str,
    intervention: str,
    seed: int,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
) -> pd.DataFrame:
    """Fit one projection to both conditions so their locations are comparable."""

    import umap

    combined = np.concatenate([clean_features, transformed_features], axis=0)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        transform_seed=seed,
    )
    coordinates = reducer.fit_transform(combined)
    count = len(clean_features)
    return pd.DataFrame(
        {
            "umap_1": coordinates[:, 0],
            "umap_2": coordinates[:, 1],
            "label": np.concatenate([labels, labels]),
            "image_id": list(image_ids) + list(image_ids),
            "condition": ["clean"] * count + ["transformed"] * count,
            "backbone": backbone,
            "intervention": intervention,
        }
    )
