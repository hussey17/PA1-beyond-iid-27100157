"""Classification evaluation with traceable per-example predictions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score


@torch.no_grad()
def predict_labeled(backbone, classifier, loader, device: torch.device) -> pd.DataFrame:
    backbone.eval()
    classifier.eval()
    rows = []
    for images, labels, sample_ids in loader:
        images = images.to(device, non_blocking=True)
        logits = classifier(backbone(images))
        probabilities = logits.softmax(dim=1)
        confidence, predictions = probabilities.max(dim=1)
        for sample_id, label, prediction, score in zip(
            sample_ids,
            labels.numpy(),
            predictions.cpu().numpy(),
            confidence.cpu().numpy(),
        ):
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "label": int(label),
                    "prediction": int(prediction),
                    "confidence": float(score),
                }
            )
    return pd.DataFrame(rows)


def classification_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    if predictions.empty:
        raise ValueError("Cannot score an empty prediction table")
    return {
        "n": len(predictions),
        "accuracy": float(
            accuracy_score(predictions["label"], predictions["prediction"])
        ),
        "macro_f1": float(
            f1_score(
                predictions["label"],
                predictions["prediction"],
                average="macro",
                zero_division=0,
            )
        ),
    }


def mean_source_metric(
    metrics_by_domain: Mapping[str, Mapping[str, float]], metric: str
) -> float:
    return float(
        np.mean(
            [domain_metrics[metric] for domain_metrics in metrics_by_domain.values()]
        )
    )
