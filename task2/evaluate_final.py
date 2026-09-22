"""Final label-aware Task 2 evaluation.

Importing this module is harmless, but callers should construct the labeled
Sketch dataset only after validating the experiment lock.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from shared.pacs_protocol import SOURCE_DOMAINS
from task2.evaluation.class_analysis import (
    class_accuracy_changes,
    confusion_table,
    dominant_confusions,
    per_class_accuracy,
)
from task2.evaluation.domain_separability import (
    balanced_domain_separability,
    extract_features,
)
from task2.evaluation.metrics import classification_metrics, predict_labeled
from task2.experiment_io import validate_experiment_lock
from task2.train import load_selected_model


def _evaluation_loader(dataset, batch_size: int, workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def evaluate_locked_experiments(
    run_specs: list[dict],
    lock_path: str | Path,
    source_validation_datasets: dict,
    target_labeled_dataset,
    class_names: list[str],
    output_root: str | Path,
    device: torch.device,
) -> pd.DataFrame:
    """Evaluate every frozen checkpoint, then perform class-level comparisons."""
    lock_records = [
        {
            "run_name": spec["record"]["run_name"],
            "config_fingerprint": spec["record"]["config_fingerprint"],
            "checkpoint_path": spec["record"]["checkpoint_path"],
        }
        for spec in run_specs
    ]
    validate_experiment_lock(lock_path, lock_records)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_by_run: dict[str, pd.DataFrame] = {}
    summary_rows = []

    for spec in run_specs:
        config = spec["config"]
        record = spec["record"]
        run_name = record["run_name"]
        workers = int(config["training"]["num_workers"])
        batch_size = int(config["training"]["target_batch_size"])
        backbone, classifier, _, _ = load_selected_model(
            config, record["checkpoint_path"], device
        )
        source_loaders = {
            domain: _evaluation_loader(
                source_validation_datasets[domain], batch_size, workers
            )
            for domain in SOURCE_DOMAINS
        }
        target_loader = _evaluation_loader(target_labeled_dataset, batch_size, workers)

        source_metrics = {}
        source_feature_batches = []
        for domain, loader in source_loaders.items():
            predictions = predict_labeled(backbone, classifier, loader, device)
            source_metrics[domain] = classification_metrics(predictions)
            features, _ = extract_features(backbone, loader, device)
            source_feature_batches.append(features)

        target_predictions = predict_labeled(
            backbone, classifier, target_loader, device
        )
        target_predictions.insert(0, "run_name", run_name)
        target_predictions.to_csv(
            output_root / f"{run_name}_target_predictions.csv", index=False
        )
        predictions_by_run[run_name] = target_predictions
        target_metrics = classification_metrics(target_predictions)
        target_features, _ = extract_features(backbone, target_loader, device)
        probe = balanced_domain_separability(
            np.concatenate(source_feature_batches, axis=0),
            target_features,
            seed=int(config["seed"]),
            test_fraction=float(config["evaluation"]["domain_probe_test_fraction"]),
            c=float(config["evaluation"]["domain_probe_c"]),
        )

        row = {
            "run_name": run_name,
            "method": config["method"]["name"],
            "mmd_weight": config["method"].get("mmd_weight", np.nan),
            "best_epoch": record["best_epoch"],
            "mean_source_accuracy": float(
                np.mean(
                    [source_metrics[domain]["accuracy"] for domain in SOURCE_DOMAINS]
                )
            ),
            "mean_source_macro_f1": float(
                np.mean(
                    [source_metrics[domain]["macro_f1"] for domain in SOURCE_DOMAINS]
                )
            ),
            "target_accuracy": target_metrics["accuracy"],
            "target_macro_f1": target_metrics["macro_f1"],
            **probe,
        }
        for domain in SOURCE_DOMAINS:
            row[f"{domain}_validation_accuracy"] = source_metrics[domain]["accuracy"]
            row[f"{domain}_validation_macro_f1"] = source_metrics[domain]["macro_f1"]
        summary_rows.append(row)

        per_class_accuracy(target_predictions, class_names).to_csv(
            output_root / f"{run_name}_target_per_class.csv", index=False
        )
        confusion_table(target_predictions, class_names).to_csv(
            output_root / f"{run_name}_target_confusion.csv", index=True
        )
        dominant_confusions(target_predictions, class_names).to_csv(
            output_root / f"{run_name}_target_dominant_confusions.csv", index=False
        )
        del backbone, classifier
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = pd.DataFrame(summary_rows)
    baseline_accuracy = float(
        summary.loc[summary["run_name"] == "source_only", "target_accuracy"].iloc[0]
    )
    summary["target_accuracy_change_vs_source_only"] = (
        summary["target_accuracy"] - baseline_accuracy
    )
    summary.to_csv(output_root / "method_comparison.csv", index=False)

    baseline_predictions = predictions_by_run["source_only"]
    for run_name, predictions in predictions_by_run.items():
        if run_name == "source_only":
            continue
        class_accuracy_changes(baseline_predictions, predictions, class_names).to_csv(
            output_root / f"{run_name}_target_class_changes.csv", index=False
        )
    return summary
