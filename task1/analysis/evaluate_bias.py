"""Linear-head training, prediction metrics, and cue-bias summaries."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LinearHead(nn.Module):
    """The only trainable component in the frozen-backbone comparison."""

    def __init__(self, feature_dim: int, number_of_classes: int = 10):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, number_of_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


@dataclass
class HeadTrainingResult:
    head: LinearHead
    history: pd.DataFrame
    best_epoch: int
    best_validation_accuracy: float


def train_linear_head(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    device: torch.device,
    seed: int,
    batch_size: int = 128,
    max_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
) -> HeadTrainingResult:
    """Train with the exact optimizer, epoch cap, and early stopping protocol."""

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    head = LinearHead(train_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_features), torch.from_numpy(train_labels)
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_x = torch.from_numpy(validation_features).to(device)
    validation_y = torch.from_numpy(validation_labels).to(device)

    best_accuracy = -np.inf
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    history_rows = []

    for epoch in range(1, max_epochs + 1):
        head.train()
        total_loss, total_count = 0.0, 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total_count += len(labels)

        head.eval()
        with torch.inference_mode():
            validation_logits = head(validation_x)
            validation_loss = criterion(validation_logits, validation_y).item()
            validation_predictions = validation_logits.argmax(dim=1)
            validation_accuracy = (
                validation_predictions.eq(validation_y).float().mean().item()
            )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total_count,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )

        # Strict improvement retains the earliest checkpoint on exact ties.
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    head.load_state_dict(best_state)
    return HeadTrainingResult(
        head=head,
        history=pd.DataFrame(history_rows),
        best_epoch=best_epoch,
        best_validation_accuracy=float(best_accuracy),
    )


@torch.inference_mode()
def logits_from_head(head: LinearHead, features: np.ndarray, device: torch.device):
    """Evaluate a trained head on cached features without changing the backbone."""

    head.eval()
    logits = head(torch.from_numpy(features).to(device)).float().cpu().numpy()
    return logits


def logits_from_clip_zero_shot(
    image_features: np.ndarray, text_features: np.ndarray, logit_scale: float
) -> np.ndarray:
    """Compute scaled class similarities for the fixed CLIP prompt."""

    return logit_scale * image_features @ text_features.T


def predictions_frame(
    logits: np.ndarray,
    labels: np.ndarray,
    image_ids,
    predictor: str,
    condition: str,
) -> pd.DataFrame:
    """Return per-example predictions so summaries remain fully traceable."""

    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    return pd.DataFrame(
        {
            "image_id": list(image_ids),
            "label": labels.astype(int),
            "prediction": predictions.astype(int),
            "confidence": probabilities.max(axis=1),
            "predictor": predictor,
            "condition": condition,
        }
    )


def classification_summary(
    frame: pd.DataFrame, clean_predictions: pd.DataFrame | None = None
) -> dict:
    """Compute the manual-required accuracy, macro-F1, confidence, and consistency."""

    summary = {
        "accuracy": accuracy_score(frame.label, frame.prediction),
        "macro_f1": f1_score(frame.label, frame.prediction, average="macro"),
        "mean_max_confidence": frame.confidence.mean(),
    }
    if clean_predictions is not None:
        paired = frame.merge(
            clean_predictions[["image_id", "prediction"]],
            on="image_id",
            suffixes=("", "_clean"),
            validate="one_to_one",
        )
        summary["prediction_consistency"] = (
            paired.prediction.eq(paired.prediction_clean).mean()
        )
    return summary


def cue_conflict_summary(predictions: pd.DataFrame, conflict_manifest: pd.DataFrame):
    """Count shape, texture, and other decisions and compute bias plus coverage."""

    merged = predictions.merge(
        conflict_manifest[
            ["image_id", "content_label", "style_label", "direction"]
        ],
        on="image_id",
        validate="one_to_one",
    )
    merged["decision_type"] = np.select(
        [
            merged.prediction.eq(merged.content_label),
            merged.prediction.eq(merged.style_label),
        ],
        ["shape", "texture"],
        default="other",
    )
    counts = merged.decision_type.value_counts().reindex(
        ["shape", "texture", "other"], fill_value=0
    )
    covered = int(counts["shape"] + counts["texture"])
    total = int(counts.sum())
    summary = {
        "shape_count": int(counts["shape"]),
        "texture_count": int(counts["texture"]),
        "other_count": int(counts["other"]),
        "shape_bias": 100.0 * counts["shape"] / covered if covered else np.nan,
        "coverage": 100.0 * covered / total if total else np.nan,
    }
    return summary, merged


def stratified_paired_accuracy_difference(
    first: pd.DataFrame,
    second: pd.DataFrame,
    resamples: int,
    seed: int,
) -> dict:
    """Bootstrap the paired accuracy difference while preserving class counts."""

    paired = first[["image_id", "label", "prediction"]].merge(
        second[["image_id", "prediction"]],
        on="image_id",
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )
    rng = np.random.default_rng(seed)
    class_indices = {
        label: np.flatnonzero(paired.label.to_numpy() == label)
        for label in sorted(paired.label.unique())
    }
    differences = []
    for _ in range(resamples):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices.values()]
        )
        block = paired.iloc[sampled]
        first_accuracy = block.prediction_first.eq(block.label).mean()
        second_accuracy = block.prediction_second.eq(block.label).mean()
        differences.append(first_accuracy - second_accuracy)
    point = paired.prediction_first.eq(paired.label).mean() - paired.prediction_second.eq(
        paired.label
    ).mean()
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "accuracy_difference": float(point),
        "bootstrap_95ci_lower": float(lower),
        "bootstrap_95ci_upper": float(upper),
        "resamples": int(resamples),
    }
