"""Target per-class accuracy changes and confusion summaries."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import confusion_matrix


def per_class_accuracy(
    predictions: pd.DataFrame, class_names: list[str]
) -> pd.DataFrame:
    rows = []
    for label, class_name in enumerate(class_names):
        subset = predictions[predictions["label"] == label]
        rows.append(
            {
                "label": label,
                "class_name": class_name,
                "n": len(subset),
                "accuracy": float((subset["label"] == subset["prediction"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def class_accuracy_changes(
    baseline: pd.DataFrame,
    adapted: pd.DataFrame,
    class_names: list[str],
) -> pd.DataFrame:
    baseline_scores = per_class_accuracy(baseline, class_names).rename(
        columns={"accuracy": "source_only_accuracy"}
    )
    adapted_scores = per_class_accuracy(adapted, class_names).rename(
        columns={"accuracy": "adapted_accuracy"}
    )
    merged = baseline_scores[["label", "class_name", "source_only_accuracy"]].merge(
        adapted_scores[["label", "adapted_accuracy"]], on="label", validate="one_to_one"
    )
    merged["accuracy_change"] = (
        merged["adapted_accuracy"] - merged["source_only_accuracy"]
    )
    return merged


def confusion_table(predictions: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    matrix = confusion_matrix(
        predictions["label"],
        predictions["prediction"],
        labels=list(range(len(class_names))),
    )
    return pd.DataFrame(matrix, index=class_names, columns=class_names)


def dominant_confusions(
    predictions: pd.DataFrame, class_names: list[str]
) -> pd.DataFrame:
    incorrect = predictions[predictions["label"] != predictions["prediction"]].copy()
    if incorrect.empty:
        return pd.DataFrame(columns=["true_class", "predicted_class", "count"])
    incorrect["true_class"] = incorrect["label"].map(dict(enumerate(class_names)))
    incorrect["predicted_class"] = incorrect["prediction"].map(
        dict(enumerate(class_names))
    )
    return (
        incorrect.groupby(["true_class", "predicted_class"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["true_class", "count"], ascending=[True, False])
    )
