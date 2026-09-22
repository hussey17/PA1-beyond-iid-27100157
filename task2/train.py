"""Common source-balanced training loop for all Task 2 methods.

This module never imports a labeled Sketch dataset.  Its target loader must
yield only images and identifiers.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from shared.pacs_protocol import SOURCE_DOMAINS, seed_everything
from task2.evaluation.metrics import classification_metrics, predict_labeled
from task2.experiment_io import (
    atomic_json_dump,
    atomic_torch_save,
    config_fingerprint,
)
from task2.methods.cdan import conditional_domain_loss
from task2.methods.dan import mmd_squared
from task2.methods.dann import domain_classification_loss, reversal_strength
from task2.methods.source_only import source_classification_loss
from task2.models.backbone import (
    ResNet18Backbone,
    set_training_mode_with_frozen_batchnorm,
)
from task2.models.classifier_head import ClassifierHead
from task2.models.domain_discriminator import DomainDiscriminator


class CyclingIterator:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iterator = iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def _loader(
    dataset, batch_size: int, shuffle: bool, workers: int, seed: int
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=workers > 0,
    )


def build_models(config: dict, device: torch.device):
    seed_everything(int(config["seed"]))
    backbone = ResNet18Backbone(pretrained=True).to(device)
    classifier = ClassifierHead(
        feature_dim=int(config["model"]["feature_dim"]),
        num_classes=int(config["model"]["num_classes"]),
    ).to(device)
    method = config["method"]["name"]
    discriminator = None
    if method in {"dann", "cdan"}:
        input_dim = int(config["model"]["feature_dim"])
        if method == "cdan":
            input_dim *= int(config["model"]["num_classes"])
        discriminator = DomainDiscriminator(
            input_dim=input_dim,
            hidden_dim=int(config["method"]["discriminator_hidden_dim"]),
            dropout=float(config["method"]["discriminator_dropout"]),
        ).to(device)
    return backbone, classifier, discriminator


def _grad_scaler(enabled: bool):
    if not enabled:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def _autocast(enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _optimizer_step(loss, optimizer, scaler) -> None:
    optimizer.zero_grad(set_to_none=True)
    if scaler is None:
        loss.backward()
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def _source_validation(backbone, classifier, loaders, device):
    metrics_by_domain = {}
    for domain, loader in loaders.items():
        metrics_by_domain[domain] = classification_metrics(
            predict_labeled(backbone, classifier, loader, device)
        )
    mean_macro_f1 = float(
        np.mean([item["macro_f1"] for item in metrics_by_domain.values()])
    )
    mean_accuracy = float(
        np.mean([item["accuracy"] for item in metrics_by_domain.values()])
    )
    return metrics_by_domain, mean_accuracy, mean_macro_f1


def run_training(
    config: dict,
    run_name: str,
    source_train_datasets: dict,
    source_validation_datasets: dict,
    target_unlabeled_dataset,
    output_root: str | Path,
    checkpoint_root: str | Path,
    device: torch.device,
    reuse_if_complete: bool = True,
) -> dict:
    """Train one method and select only by mean source-validation macro-F1."""
    output_dir = Path(output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_root) / f"{run_name}_seed{config['seed']}.pt"
    record_path = output_dir / "training_record.json"
    fingerprint = config_fingerprint(config)
    if reuse_if_complete and record_path.is_file() and checkpoint_path.is_file():
        import json

        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["config_fingerprint"] != fingerprint:
            raise RuntimeError(
                f"Existing run {run_name} has a different configuration. "
                "Use a new run name or remove only that run's outputs."
            )
        return record

    seed = int(config["seed"])
    seed_everything(seed)
    workers = int(config["training"]["num_workers"])
    source_batch_size = int(config["training"]["source_batch_size_per_domain"])
    source_loaders = {
        domain: _loader(
            source_train_datasets[domain],
            source_batch_size,
            shuffle=True,
            workers=workers,
            seed=seed + index,
        )
        for index, domain in enumerate(SOURCE_DOMAINS)
    }
    validation_loaders = {
        domain: _loader(
            source_validation_datasets[domain],
            batch_size=source_batch_size * 3,
            shuffle=False,
            workers=workers,
            seed=seed,
        )
        for domain in SOURCE_DOMAINS
    }
    method = config["method"]["name"]
    target_loader = None
    if method != "source_only":
        if target_unlabeled_dataset is None:
            raise ValueError(f"{method} requires an unlabeled target dataset")
        target_loader = _loader(
            target_unlabeled_dataset,
            batch_size=int(config["training"]["target_batch_size"]),
            shuffle=True,
            workers=workers,
            seed=seed + 100,
        )

    backbone, classifier, discriminator = build_models(config, device)
    parameters = list(backbone.parameters()) + list(classifier.parameters())
    if discriminator is not None:
        parameters += list(discriminator.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    amp_enabled = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    scaler = _grad_scaler(amp_enabled)
    steps_per_epoch = max(len(loader) for loader in source_loaders.values())
    max_epochs = int(config["training"]["max_epochs"])
    total_planned_steps = max(1, max_epochs * steps_per_epoch - 1)

    best_score = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history_rows = []
    global_step = 0

    for epoch in range(max_epochs):
        set_training_mode_with_frozen_batchnorm(backbone)
        classifier.train()
        if discriminator is not None:
            discriminator.train()
        source_iterators = {
            domain: CyclingIterator(loader) for domain, loader in source_loaders.items()
        }
        target_iterator = (
            CyclingIterator(target_loader) if target_loader is not None else None
        )
        epoch_totals = {
            "classification_loss": 0.0,
            "alignment_loss": 0.0,
            "weighted_alignment_loss": 0.0,
            "total_loss": 0.0,
        }
        epoch_domain_correct = 0
        epoch_domain_total = 0
        epoch_strength = 0.0

        for _ in range(steps_per_epoch):
            source_batches = [
                source_iterators[domain].next() for domain in SOURCE_DOMAINS
            ]
            source_images = torch.cat([batch[0] for batch in source_batches], dim=0).to(
                device, non_blocking=True
            )
            source_labels = torch.cat([batch[1] for batch in source_batches], dim=0).to(
                device, non_blocking=True
            )
            progress = min(1.0, global_step / total_planned_steps)
            strength = 0.0

            with _autocast(amp_enabled):
                source_features = backbone(source_images)
                source_logits = classifier(source_features)
                classification_loss = source_classification_loss(
                    source_logits, source_labels
                )
                alignment_loss = classification_loss.new_zeros(())
                domain_logits = domain_labels = None

                if method != "source_only":
                    target_images = target_iterator.next()[0].to(
                        device, non_blocking=True
                    )
                    target_features = backbone(target_images)
                if method == "dan":
                    alignment_loss = mmd_squared(
                        source_features.float(),
                        target_features.float(),
                        config["method"]["kernel_bandwidth_multipliers"],
                    )
                    weighted_alignment_loss = (
                        float(config["method"]["mmd_weight"]) * alignment_loss
                    )
                    total_loss = (
                        float(config["method"]["classification_loss_weight"])
                        * classification_loss
                        + weighted_alignment_loss
                    )
                elif method == "dann":
                    strength = reversal_strength(
                        progress,
                        maximum=float(config["method"]["gradient_reversal_max"]),
                        gamma=float(config["method"]["gradient_reversal_gamma"]),
                    )
                    alignment_loss, domain_logits, domain_labels = (
                        domain_classification_loss(
                            discriminator, source_features, target_features, strength
                        )
                    )
                    weighted_alignment_loss = (
                        float(config["method"]["domain_loss_weight"]) * alignment_loss
                    )
                    total_loss = classification_loss + weighted_alignment_loss
                elif method == "cdan":
                    strength = reversal_strength(
                        progress,
                        maximum=float(config["method"]["gradient_reversal_max"]),
                        gamma=float(config["method"]["gradient_reversal_gamma"]),
                    )
                    target_logits = classifier(target_features)
                    alignment_loss, domain_logits, domain_labels = (
                        conditional_domain_loss(
                            discriminator,
                            source_features,
                            source_logits,
                            target_features,
                            target_logits,
                            strength,
                        )
                    )
                    weighted_alignment_loss = (
                        float(config["method"]["domain_loss_weight"]) * alignment_loss
                    )
                    total_loss = classification_loss + weighted_alignment_loss
                else:
                    weighted_alignment_loss = alignment_loss
                    total_loss = classification_loss

            _optimizer_step(total_loss, optimizer, scaler)
            epoch_totals["classification_loss"] += float(classification_loss.detach())
            epoch_totals["alignment_loss"] += float(alignment_loss.detach())
            epoch_totals["weighted_alignment_loss"] += float(
                weighted_alignment_loss.detach()
            )
            epoch_totals["total_loss"] += float(total_loss.detach())
            epoch_strength += strength
            if domain_logits is not None:
                epoch_domain_correct += int(
                    (domain_logits.detach().argmax(dim=1) == domain_labels).sum().item()
                )
                epoch_domain_total += int(domain_labels.numel())
            global_step += 1

        validation, mean_accuracy, mean_macro_f1 = _source_validation(
            backbone, classifier, validation_loaders, device
        )
        row = {
            "epoch": epoch + 1,
            **{key: value / steps_per_epoch for key, value in epoch_totals.items()},
            "gradient_reversal_strength": epoch_strength / steps_per_epoch,
            "training_domain_accuracy": (
                epoch_domain_correct / epoch_domain_total
                if epoch_domain_total
                else np.nan
            ),
            "mean_source_validation_accuracy": mean_accuracy,
            "mean_source_validation_macro_f1": mean_macro_f1,
        }
        for domain, scores in validation.items():
            row[f"{domain}_validation_accuracy"] = scores["accuracy"]
            row[f"{domain}_validation_macro_f1"] = scores["macro_f1"]
        history_rows.append(row)

        if mean_macro_f1 > best_score:
            best_score = mean_macro_f1
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            atomic_torch_save(
                {
                    "run_name": run_name,
                    "epoch": best_epoch,
                    "selection_metric": "mean_source_validation_macro_f1",
                    "selection_score": best_score,
                    "config": config,
                    "config_fingerprint": fingerprint,
                    "backbone_state": backbone.state_dict(),
                    "classifier_state": classifier.state_dict(),
                    "discriminator_state": (
                        discriminator.state_dict()
                        if discriminator is not None
                        else None
                    ),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config["training"]["patience"]):
                break

    history = pd.DataFrame(history_rows)
    temporary_history = output_dir / "training_history.csv.tmp"
    history.to_csv(temporary_history, index=False)
    temporary_history.replace(output_dir / "training_history.csv")
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    record = {
        "run_name": run_name,
        "method": method,
        "config_fingerprint": fingerprint,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "best_epoch": best_epoch,
        "best_mean_source_validation_macro_f1": best_score,
        "epochs_completed": len(history),
        "steps_per_epoch": int(steps_per_epoch),
        "target_labels_used": False,
    }
    atomic_json_dump(record, record_path)
    return record


def load_selected_model(
    config: dict, checkpoint_path: str | Path, device: torch.device
):
    backbone, classifier, discriminator = build_models(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["config_fingerprint"] != config_fingerprint(config):
        raise RuntimeError("Checkpoint configuration fingerprint mismatch")
    backbone.load_state_dict(checkpoint["backbone_state"])
    classifier.load_state_dict(checkpoint["classifier_state"])
    if discriminator is not None and checkpoint["discriminator_state"] is not None:
        discriminator.load_state_dict(checkpoint["discriminator_state"])
    return backbone, classifier, discriminator, checkpoint
