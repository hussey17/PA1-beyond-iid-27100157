# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Task 1 — Inductive Biases and Feature Representations
#
# **Purpose.** This notebook implements the complete Task 1 protocol for STL-10. It compares frozen ResNet-50, ViT-B/16, and OpenCLIP ViT-B/32 representations under controlled color, cue-conflict, translation, and patch-structure interventions.
#
# The notebook deliberately separates three stages:
#
# 1. **Generate and freeze the data protocol.** Splits, the 500-image test subset, and transformed images are created once and saved.
# 2. **Train/evaluate models.** Every predictor receives the same saved pixels. Only a linear classifier head is trained; the backbones remain frozen.
# 3. **Analyze predictions and representations.** Metrics, per-example predictions, cosine stability, UMAP coordinates, and diagnostic figures are saved in machine-readable form.
#
# The notebook does **not** write report conclusions. It produces evidence for the repository owner to inspect and interpret.

# %% [markdown]
# ## 0. Colab bootstrap and dependencies
#
# When this notebook is uploaded directly to Colab, the first cell clones the public repository so that the versioned modules and configuration are available. Re-running the cell is safe.

# %%
from pathlib import Path
import os
import subprocess
import sys

REPOSITORY_URL = "https://github.com/hussey17/PA1-beyond-iid-27100157.git"
REPOSITORY_ROOT = Path("/content/PA1-beyond-iid-27100157")

if not REPOSITORY_ROOT.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1", REPOSITORY_URL, str(REPOSITORY_ROOT)],
        check=True,
    )
else:
    subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "pull", "--ff-only"],
        check=True,
    )
os.chdir(REPOSITORY_ROOT)
print(f"Working directory: {Path.cwd()}")

# %%
# Colab already supplies CUDA-enabled PyTorch and torchvision. Reinstalling them
# can break binary compatibility, so only the missing experiment dependencies
# are installed here.
%pip install -q open_clip_torch umap-learn pyyaml ipywidgets seaborn

# %% [markdown]
# ## 1. Imports, configuration, and reproducibility
#
# The YAML file is the frozen experimental specification. Printing it in the executed notebook makes the exact run configuration visible alongside the outputs.

# %%
import copy
import gc
import json
import platform
import shutil
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from IPython.display import HTML, clear_output, display
from PIL import Image
from torchvision.datasets import STL10

from task1.analysis.evaluate_bias import (
    LinearHead,
    classification_summary,
    cue_conflict_summary,
    logits_from_clip_zero_shot,
    logits_from_head,
    predictions_frame,
    stratified_paired_accuracy_difference,
    train_linear_head,
)
from task1.analysis.feature_similarity import (
    rowwise_cosine,
    stability_frame,
    summarize_stability,
)
from task1.analysis.representation import joint_umap_projection
from task1.data.make_cue_conflicts import (
    build_candidate_manifest,
    generate_candidates,
    prepare_adain,
    select_balanced_conflicts,
)
from task1.data.make_subset import (
    FileImageDataset,
    IndexedImageDataset,
    STL10_CLASSES,
    balanced_test_indices,
    common_geometry,
    export_clean_subset,
    save_split_manifest,
    seed_everything,
    stratified_train_val_indices,
)
from task1.data.transforms import (
    generate_patch_shuffle_set,
    generate_simple_intervention,
    generate_translation_sets,
    grayscale_rgb,
    rotate_hue,
)
from task1.models.backbones import (
    clip_text_classifier,
    extract_features,
    load_backbone,
)

CONFIG_PATH = REPOSITORY_ROOT / "task1/configs/task1.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as stream:
    CONFIG = yaml.safe_load(stream)

SEED = int(CONFIG["seed"])
seed_everything(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    raise RuntimeError(
        "This notebook is configured for a Colab NVIDIA GPU. "
        "Choose Runtime > Change runtime type > T4 GPU, then rerun."
    )

DATA_ROOT = REPOSITORY_ROOT / "data"
GENERATED_ROOT = REPOSITORY_ROOT / "task1/data/generated"
CACHE_ROOT = REPOSITORY_ROOT / "task1/cache"
RESULTS_ROOT = REPOSITORY_ROOT / "task1/results/run_seed6304"
FIGURES_ROOT = RESULTS_ROOT / "figures"
for directory in [DATA_ROOT, GENERATED_ROOT, CACHE_ROOT, RESULTS_ROOT, FIGURES_ROOT]:
    directory.mkdir(parents=True, exist_ok=True)

print(yaml.safe_dump(CONFIG, sort_keys=False))
print(f"PyTorch: {torch.__version__}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# %%
# Record software/hardware provenance before any experiment is run.
RUN_METADATA = {
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "seed": SEED,
    "python": platform.python_version(),
    "pytorch": torch.__version__,
    "cuda_device": torch.cuda.get_device_name(0),
    "configuration": CONFIG,
}
(RESULTS_ROOT / "run_metadata.json").write_text(
    json.dumps(RUN_METADATA, indent=2) + "\n", encoding="utf-8"
)
(RESULTS_ROOT / "environment_freeze.txt").write_text(
    subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    ),
    encoding="utf-8",
)

# %% [markdown]
# ## 2. Pre-registered hypotheses
#
# These statements were fixed before examining model outputs:
#
# 1. **Color:** grayscale and a +90° hue rotation should reduce accuracy by no more than five percentage points relative to each predictor's clean baseline.
# 2. **Shape versus texture:** among predictions selecting either intended cue-conflict class, the content/shape label should exceed the texture/style label (shape bias > 50%). Coverage and “other” decisions must be considered with shape bias.
# 3. **Translation:** accuracy and consistency should decline as displacement increases.
# 4. **Patch structure:** shuffling should cause a non-catastrophic decline because local evidence remains, although the ranking relative to translation is left empirical.
# 5. **Representations:** stronger interventions should generally reduce cosine stability, but prediction stability and representation stability need not agree example-by-example.
# 6. **CLIP:** the trained CLIP head should outperform fixed-prompt zero-shot CLIP on STL-10; the paired accuracy difference is accompanied by a class-stratified bootstrap interval.
#
# No ResNet-versus-ViT-versus-CLIP ranking is assumed because architecture, pretraining, supervision, augmentation, and capacity are confounded.

# %% [markdown]
# ## 3. STL-10 splits and common 500-image evaluation subset
#
# The official training partition is split 80/20 using stratification and seed 6304. The official test subset contains 50 examples from each of the ten classes. Original dataset indices are persisted so the exact samples can be reconstructed.

# %%
train_dataset = STL10(root=DATA_ROOT, split="train", download=True)
test_dataset = STL10(root=DATA_ROOT, split="test", download=True)

train_indices, validation_indices = stratified_train_val_indices(
    train_dataset.labels,
    validation_fraction=float(CONFIG["dataset"]["validation_fraction"]),
    seed=SEED,
)
test_indices = balanced_test_indices(
    test_dataset.labels,
    examples_per_class=int(CONFIG["dataset"]["test_examples_per_class"]),
    seed=SEED,
)

SPLIT_MANIFEST_PATH = RESULTS_ROOT / "stl10_splits_seed6304.json"
save_split_manifest(
    SPLIT_MANIFEST_PATH,
    train_indices,
    validation_indices,
    test_indices,
    SEED,
)

geometry = common_geometry(
    CONFIG["preprocessing"]["resize_shorter_edge"],
    CONFIG["preprocessing"]["crop_size"],
)
clean_manifest = export_clean_subset(
    test_dataset,
    test_indices,
    GENERATED_ROOT / "clean",
    geometry,
)
clean_manifest.to_csv(RESULTS_ROOT / "clean_test_manifest.csv", index=False)

display(
    clean_manifest.groupby(["label", "class_name"]).size().rename("count").reset_index()
)
assert len(clean_manifest) == 500
assert clean_manifest.groupby("class_name").size().eq(50).all()

# %% [markdown]
# ## 4. Generate all non-AdaIN interventions once
#
# Saving the transformed images before loading a backbone guarantees that all models receive identical pixels. A +90° hue rotation corresponds to torchvision's hue factor `0.25`; saturation and value are retained in HSV space apart from numerical conversion effects.

# %%
grayscale_manifest = generate_simple_intervention(
    clean_manifest,
    GENERATED_ROOT / "grayscale",
    "grayscale",
    grayscale_rgb,
)
hue_manifest = generate_simple_intervention(
    clean_manifest,
    GENERATED_ROOT / "hue_90",
    "hue_90",
    lambda image: rotate_hue(image, CONFIG["color"]["hue_degrees"]),
)
patch_manifest = generate_patch_shuffle_set(
    clean_manifest,
    GENERATED_ROOT / "patch_shuffle",
    seed=SEED,
    grid_size=CONFIG["patch_shuffle"]["grid_size"],
)
translation_manifest = generate_translation_sets(
    clean_manifest,
    GENERATED_ROOT / "translation",
    displacements=CONFIG["translation"]["displacements"],
    directions=CONFIG["translation"]["directions"],
)

for name, frame in {
    "grayscale": grayscale_manifest,
    "hue_90": hue_manifest,
    "patch_shuffle": patch_manifest,
    "translation": translation_manifest,
}.items():
    frame.to_csv(RESULTS_ROOT / f"{name}_manifest.csv", index=False)

print("Generated/cached intervention images:")
print({
    "grayscale": len(grayscale_manifest),
    "hue_90": len(hue_manifest),
    "patch_shuffle": len(patch_manifest),
    "translation": len(translation_manifest),
})

# %%
# Visual sanity check only; this cell does not select examples using a model.
sample_rows = clean_manifest.groupby("class_name", sort=False).head(1)
fig, axes = plt.subplots(len(sample_rows), 4, figsize=(10, 24))
for row_number, row in enumerate(sample_rows.itertuples(index=False)):
    paths = [
        row.path,
        grayscale_manifest.loc[grayscale_manifest.image_id.eq(row.image_id), "path"].iloc[0],
        hue_manifest.loc[hue_manifest.image_id.eq(row.image_id), "path"].iloc[0],
        patch_manifest.loc[patch_manifest.image_id.eq(row.image_id), "path"].iloc[0],
    ]
    for column, (title, path) in enumerate(zip(
        ["clean", "grayscale", "hue +90°", "patch shuffle"], paths
    )):
        axes[row_number, column].imshow(Image.open(path))
        axes[row_number, column].set_title(f"{row.class_name}: {title}")
        axes[row_number, column].axis("off")
plt.tight_layout()
plt.savefig(FIGURES_ROOT / "intervention_sanity_check.png", dpi=180, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Generate AdaIN cue-conflict candidates
#
# AdaIN aligns the channel-wise mean and standard deviation of content features with those of a style image. The selected `alpha=0.8` interpolates between the original content feature and the fully stylized feature: it gives the style strong influence while retaining 20% of the original content representation. It is a fixed compromise, not a value tuned from predictions.
#
# The notebook uses the MIT-licensed public PyTorch implementation by `naoto0804` and its released weights. The implementation is cloned at runtime and is not copied into this repository.

# %%
adain_vgg, adain_decoder, adain_repository = prepare_adain(
    CACHE_ROOT / "external", DEVICE
)

CUE_CANDIDATE_PATH = RESULTS_ROOT / "cue_conflict_review_manifest.csv"
candidate_manifest = build_candidate_manifest(
    clean_manifest=clean_manifest,
    class_pairs=CONFIG["cue_conflict"]["class_pairs"],
    candidates_per_direction=CONFIG["cue_conflict"]["candidates_per_direction"],
    seed=SEED,
    output_dir=GENERATED_ROOT / "cue_conflicts",
)

# Preserve review decisions if this cell is rerun.
if CUE_CANDIDATE_PATH.exists():
    previous_review = pd.read_csv(CUE_CANDIDATE_PATH, keep_default_na=False)
    previous_decisions = previous_review[
        ["candidate_id", "decision", "rejection_reason"]
    ]
    candidate_manifest = candidate_manifest.drop(
        columns=["decision", "rejection_reason"]
    ).merge(previous_decisions, on="candidate_id", how="left")
    candidate_manifest[["decision", "rejection_reason"]] = candidate_manifest[
        ["decision", "rejection_reason"]
    ].fillna("")

generate_candidates(
    candidate_manifest,
    adain_vgg,
    adain_decoder,
    device=DEVICE,
    alpha=float(CONFIG["cue_conflict"]["alpha"]),
    batch_size=8,
)
candidate_manifest.to_csv(CUE_CANDIDATE_PATH, index=False)

# Release AdaIN before loading the larger classification backbones.
del adain_vgg, adain_decoder
gc.collect()
torch.cuda.empty_cache()
print(f"Generated {len(candidate_manifest)} candidates for visual review.")

# %% [markdown]
# ## 6. Model-independent visual review gate
#
# Review **all** candidates before any classifier prediction is generated. Accept only when the content object remains identifiable, the image is not visibly corrupted, and the style effect is perceptible. Rejections must record one of those pre-registered reasons.
#
# The widget saves after every click, so work survives an interrupted Colab cell. The three panels show content, style, and generated conflict; no model output is available here.

# %%
import ipywidgets as widgets


class CueConflictReviewer:
    """Small persistent reviewer for the pre-registered visual rejection rule."""

    REASONS = [
        "content_unidentifiable",
        "visible_corruption",
        "style_imperceptible",
    ]

    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.frame = pd.read_csv(manifest_path, keep_default_na=False)
        self.output = widgets.Output()
        self.reason = widgets.Dropdown(options=self.REASONS, description="Reason")
        self.accept_button = widgets.Button(description="Accept", button_style="success")
        self.reject_button = widgets.Button(description="Reject", button_style="danger")
        self.accept_button.on_click(self._accept)
        self.reject_button.on_click(self._reject)
        display(widgets.HBox([self.accept_button, self.reject_button, self.reason]))
        display(self.output)
        self._show_next()

    def _pending_positions(self):
        return self.frame.index[~self.frame.decision.str.lower().isin(["accept", "reject"])].tolist()

    def _save(self):
        self.frame.to_csv(self.manifest_path, index=False)

    def _show_next(self):
        pending = self._pending_positions()
        with self.output:
            clear_output(wait=True)
            if not pending:
                display(HTML("<h4>Review complete.</h4>"))
                display(self.frame.decision.value_counts())
                return
            position = pending[0]
            row = self.frame.loc[position]
            completed = len(self.frame) - len(pending)
            print(f"Candidate {completed + 1}/{len(self.frame)}: {row.candidate_id}")
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for axis, title, path in zip(
                axes,
                [
                    f"Content/shape: {row.content_class}",
                    f"Style/texture: {row.style_class}",
                    "Generated conflict",
                ],
                [row.content_path, row.style_path, row.path],
            ):
                axis.imshow(Image.open(path))
                axis.set_title(title)
                axis.axis("off")
            plt.tight_layout()
            plt.show()

    def _accept(self, _):
        pending = self._pending_positions()
        if pending:
            self.frame.loc[pending[0], ["decision", "rejection_reason"]] = [
                "accept",
                "",
            ]
            self._save()
        self._show_next()

    def _reject(self, _):
        pending = self._pending_positions()
        if pending:
            self.frame.loc[pending[0], ["decision", "rejection_reason"]] = [
                "reject",
                self.reason.value,
            ]
            self._save()
        self._show_next()


reviewer = CueConflictReviewer(CUE_CANDIDATE_PATH)

# %% [markdown]
# ### Validate the review and freeze 200 balanced conflicts
#
# The next cell refuses to proceed if any decision is missing or any direction has fewer than 20 accepted images. If a direction is deficient, the following recovery cell deterministically generates the remaining candidates up to the 50 available class images; rerun the review widget afterward.

# %%
reviewed_manifest = pd.read_csv(CUE_CANDIDATE_PATH, keep_default_na=False)
valid_decisions = reviewed_manifest.decision.str.lower().isin(["accept", "reject"])
if not valid_decisions.all():
    raise RuntimeError(
        f"Review {int((~valid_decisions).sum())} remaining candidates before evaluation."
    )

review_counts = (
    reviewed_manifest.groupby(["direction", "decision"]).size().unstack(fill_value=0)
)
display(review_counts)

selected_conflicts = select_balanced_conflicts(
    reviewed_manifest,
    accepted_per_direction=CONFIG["cue_conflict"]["accepted_per_direction"],
)
selected_conflicts.to_csv(RESULTS_ROOT / "cue_conflicts_selected.csv", index=False)
assert len(selected_conflicts) == 200
assert selected_conflicts.groupby("direction").size().eq(20).all()
print("Cue-conflict set frozen: 200 images, 20 per direction.")

# %% [markdown]
# **Recovery only if the validation cell reports a deficient direction.** Change `GENERATE_EXTRA = True`, execute this cell once, and then rerun the reviewer and validation cells. Existing decisions are preserved. Do not use this cell after seeing any model prediction.

# %%
GENERATE_EXTRA = False
if GENERATE_EXTRA:
    extended_manifest = build_candidate_manifest(
        clean_manifest=clean_manifest,
        class_pairs=CONFIG["cue_conflict"]["class_pairs"],
        candidates_per_direction=50,
        seed=SEED,
        output_dir=GENERATED_ROOT / "cue_conflicts",
    )
    old_review = pd.read_csv(CUE_CANDIDATE_PATH, keep_default_na=False)[
        ["candidate_id", "decision", "rejection_reason"]
    ]
    extended_manifest = extended_manifest.drop(
        columns=["decision", "rejection_reason"]
    ).merge(old_review, on="candidate_id", how="left")
    extended_manifest[["decision", "rejection_reason"]] = extended_manifest[
        ["decision", "rejection_reason"]
    ].fillna("")
    adain_vgg, adain_decoder, _ = prepare_adain(CACHE_ROOT / "external", DEVICE)
    generate_candidates(
        extended_manifest,
        adain_vgg,
        adain_decoder,
        device=DEVICE,
        alpha=float(CONFIG["cue_conflict"]["alpha"]),
        batch_size=8,
    )
    extended_manifest.to_csv(CUE_CANDIDATE_PATH, index=False)
    del adain_vgg, adain_decoder
    gc.collect()
    torch.cuda.empty_cache()
    print("Additional candidates generated. Rerun the reviewer and validation cells.")

# %% [markdown]
# ## 7. Frozen-feature cache and linear-head helpers
#
# Feature caching makes notebook restarts inexpensive. Cache filenames include the backbone and condition, and saved identifiers allow alignment checks before any paired metric is computed.

# %%
BACKBONE_NAMES = ["resnet50", "vit_b16", "clip_vit_b32"]


def cached_features(cache_name, bundle, dataset):
    cache_path = CACHE_ROOT / f"{cache_name}.npz"
    if hasattr(dataset, "frame"):
        expected_identifiers = dataset.frame["image_id"].astype(str).tolist()
    else:
        expected_identifiers = [str(item) for item in dataset.indices]
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=False)
        cached_identifiers = payload["identifiers"].astype(str).tolist()
        if cached_identifiers == expected_identifiers:
            return payload["features"], payload["labels"], cached_identifiers
        print(f"Refreshing stale cache with changed identifiers: {cache_path.name}")
    features, labels, identifiers = extract_features(
        bundle,
        dataset,
        batch_size=int(CONFIG["training"]["feature_batch_size"]),
        workers=2,
    )
    np.savez_compressed(
        cache_path,
        features=features,
        labels=labels,
        identifiers=np.asarray(identifiers).astype(str),
    )
    return features, labels, list(np.asarray(identifiers).astype(str))


def manifest_features(bundle, frame, cache_name):
    dataset = FileImageDataset(frame, bundle.tensor_transform)
    return cached_features(cache_name, bundle, dataset)


def get_or_train_head(bundle, train_features, train_labels, val_features, val_labels):
    checkpoint_path = CACHE_ROOT / f"{bundle.name}_linear_head.pt"
    history_path = RESULTS_ROOT / f"{bundle.name}_head_training.csv"
    metadata_path = RESULTS_ROOT / f"{bundle.name}_head_metadata.json"
    head = LinearHead(bundle.feature_dim).to(DEVICE)
    if checkpoint_path.exists() and history_path.exists() and metadata_path.exists():
        head.load_state_dict(
            torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        )
        return head

    result = train_linear_head(
        train_features,
        train_labels,
        val_features,
        val_labels,
        device=DEVICE,
        seed=SEED,
        batch_size=int(CONFIG["training"]["head_batch_size"]),
        max_epochs=int(CONFIG["training"]["max_epochs"]),
        learning_rate=float(CONFIG["training"]["learning_rate"]),
        weight_decay=float(CONFIG["training"]["weight_decay"]),
        patience=int(CONFIG["training"]["patience"]),
    )
    torch.save(result.head.state_dict(), checkpoint_path)
    result.history.to_csv(history_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "best_epoch": result.best_epoch,
                "best_validation_accuracy": result.best_validation_accuracy,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result.head


def evaluate_logits(logits, labels, ids, predictor, condition, clean_frame=None):
    frame = predictions_frame(logits, labels, ids, predictor, condition)
    summary = classification_summary(frame, clean_frame)
    summary.update({"predictor": predictor, "condition": condition})
    return frame, summary

# %% [markdown]
# ## 8. Train the three heads and run common prediction evaluation
#
# For each backbone, the workflow is:
#
# 1. Load pretrained weights and freeze every backbone parameter.
# 2. Extract deterministic train/validation features once.
# 3. Train only a linear head with AdamW and early stopping on validation accuracy.
# 4. Evaluate clean, grayscale, hue, patch, translation, and accepted cue-conflict images.
# 5. For CLIP only, evaluate the same image features with the fixed zero-shot prompt.
#
# Keeping this as one backbone-at-a-time loop limits GPU memory use on a T4.

# %%
all_predictions = []
performance_rows = []
cue_summary_rows = []
cue_decision_frames = []
translation_rows = []
stability_frames = []
projection_frames = []

# Fixed visualization subset: 20 per class, deterministic within the 500 images.
visualization_manifest = pd.concat(
    [
        group.sample(
            n=CONFIG["representation"]["examples_per_class"],
            random_state=SEED,
        )
        for _, group in clean_manifest.groupby("label", sort=True)
    ],
    ignore_index=True,
)
visualization_ids = set(visualization_manifest.image_id)

for backbone_name in BACKBONE_NAMES:
    print(f"\n===== {backbone_name} =====")
    bundle = load_backbone(backbone_name, DEVICE)

    train_data = IndexedImageDataset(
        train_dataset, train_indices, bundle.tensor_transform, geometry
    )
    validation_data = IndexedImageDataset(
        train_dataset, validation_indices, bundle.tensor_transform, geometry
    )
    train_features, train_labels, _ = cached_features(
        f"{backbone_name}_train", bundle, train_data
    )
    validation_features, validation_labels, _ = cached_features(
        f"{backbone_name}_validation", bundle, validation_data
    )
    head = get_or_train_head(
        bundle,
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    )

    condition_manifests = {
        "clean": clean_manifest,
        "grayscale": grayscale_manifest,
        "hue_90": hue_manifest,
        "patch_shuffle": patch_manifest,
        "cue_conflict": selected_conflicts,
    }
    condition_features = {}
    for condition, manifest in condition_manifests.items():
        features, labels, ids = manifest_features(
            bundle, manifest, f"{backbone_name}_{condition}"
        )
        condition_features[condition] = (features, labels, [str(item) for item in ids])

    predictor = f"{backbone_name}_linear"
    clean_features, clean_labels, clean_ids = condition_features["clean"]
    clean_frame, clean_summary = evaluate_logits(
        logits_from_head(head, clean_features, DEVICE),
        clean_labels,
        clean_ids,
        predictor,
        "clean",
    )
    all_predictions.append(clean_frame)
    performance_rows.append(clean_summary)

    # Linear-head evaluation on non-translation interventions.
    for condition in ["grayscale", "hue_90", "patch_shuffle"]:
        features, labels, ids = condition_features[condition]
        frame, summary = evaluate_logits(
            logits_from_head(head, features, DEVICE),
            labels,
            ids,
            predictor,
            condition,
            clean_frame,
        )
        all_predictions.append(frame)
        performance_rows.append(summary)

    cue_features, cue_labels, cue_ids = condition_features["cue_conflict"]
    cue_frame, _ = evaluate_logits(
        logits_from_head(head, cue_features, DEVICE),
        cue_labels,
        cue_ids,
        predictor,
        "cue_conflict",
    )
    all_predictions.append(cue_frame)
    cue_summary, cue_decisions = cue_conflict_summary(
        cue_frame, selected_conflicts
    )
    cue_summary["predictor"] = predictor
    cue_summary_rows.append(cue_summary)
    cue_decision_frames.append(cue_decisions)

    # CLIP zero-shot uses the same cached normalized image embeddings.
    predictors_for_features = [(predictor, head, False)]
    if backbone_name == "clip_vit_b32":
        text_features, clip_scale = clip_text_classifier(bundle, STL10_CLASSES)
        zero_clean_frame, zero_clean_summary = evaluate_logits(
            logits_from_clip_zero_shot(clean_features, text_features, clip_scale),
            clean_labels,
            clean_ids,
            "clip_vit_b32_zeroshot",
            "clean",
        )
        all_predictions.append(zero_clean_frame)
        performance_rows.append(zero_clean_summary)
        for condition in ["grayscale", "hue_90", "patch_shuffle"]:
            features, labels, ids = condition_features[condition]
            frame, summary = evaluate_logits(
                logits_from_clip_zero_shot(features, text_features, clip_scale),
                labels,
                ids,
                "clip_vit_b32_zeroshot",
                condition,
                zero_clean_frame,
            )
            all_predictions.append(frame)
            performance_rows.append(summary)
        zero_cue_frame, _ = evaluate_logits(
            logits_from_clip_zero_shot(cue_features, text_features, clip_scale),
            cue_labels,
            cue_ids,
            "clip_vit_b32_zeroshot",
            "cue_conflict",
        )
        all_predictions.append(zero_cue_frame)
        zero_cue_summary, zero_cue_decisions = cue_conflict_summary(
            zero_cue_frame, selected_conflicts
        )
        zero_cue_summary["predictor"] = "clip_vit_b32_zeroshot"
        cue_summary_rows.append(zero_cue_summary)
        cue_decision_frames.append(zero_cue_decisions)
        predictors_for_features.append(("clip_vit_b32_zeroshot", None, True))

    # Translation prediction curves for the linear head and CLIP zero-shot.
    for displacement in CONFIG["translation"]["displacements"]:
        directions = ["none"] if displacement == 0 else CONFIG["translation"]["directions"]
        for direction in directions:
            group = translation_manifest[
                translation_manifest.displacement.eq(displacement)
                & translation_manifest.direction.eq(direction)
            ].reset_index(drop=True)
            features, labels, ids = manifest_features(
                bundle,
                group,
                f"{backbone_name}_translation_{displacement}_{direction}",
            )
            for predictor_name, predictor_head, is_zero_shot in predictors_for_features:
                logits = (
                    logits_from_clip_zero_shot(features, text_features, clip_scale)
                    if is_zero_shot
                    else logits_from_head(predictor_head, features, DEVICE)
                )
                matching_clean = (
                    zero_clean_frame if is_zero_shot else clean_frame
                )
                frame, summary = evaluate_logits(
                    logits,
                    labels,
                    [str(item) for item in ids],
                    predictor_name,
                    f"translation_{displacement}_{direction}",
                    matching_clean,
                )
                all_predictions.append(frame)
                summary.update(
                    {"displacement": displacement, "direction": direction}
                )
                translation_rows.append(summary)

            # Representations do not depend on which classifier reads them.
            if displacement > 0:
                similarities = rowwise_cosine(clean_features, features)
                stability_frames.append(
                    stability_frame(
                        clean_ids,
                        clean_labels,
                        similarities,
                        backbone_name,
                        "translation",
                        displacement=displacement,
                        direction=direction,
                    )
                )

    # Required paired representation stability for grayscale, patch shuffle,
    # cue conflicts, and translation (computed above for all deltas/directions).
    for condition in ["grayscale", "patch_shuffle"]:
        features, labels, ids = condition_features[condition]
        stability_frames.append(
            stability_frame(
                ids,
                labels,
                rowwise_cosine(clean_features, features),
                backbone_name,
                condition,
            )
        )

    clean_position = {image_id: position for position, image_id in enumerate(clean_ids)}
    cue_clean_positions = [
        clean_position[image_id]
        for image_id in selected_conflicts.content_image_id.astype(str)
    ]
    cue_clean_features = clean_features[cue_clean_positions]
    stability_frames.append(
        stability_frame(
            cue_ids,
            cue_labels,
            rowwise_cosine(cue_clean_features, cue_features),
            backbone_name,
            "cue_conflict",
        )
    )

    # UMAP uses the fixed 20-per-class visualization subset. For translation,
    # cardinal directions are assigned round-robin so every direction appears
    # equally without duplicating clean points.
    vis_positions = [clean_position[item] for item in visualization_manifest.image_id]
    vis_clean = clean_features[vis_positions]
    vis_labels = clean_labels[vis_positions]
    vis_ids = visualization_manifest.image_id.astype(str).tolist()

    for condition in ["grayscale", "patch_shuffle"]:
        transformed = condition_features[condition][0][vis_positions]
        projection_frames.append(
            joint_umap_projection(
                vis_clean,
                transformed,
                vis_labels,
                vis_ids,
                backbone_name,
                condition,
                seed=SEED,
                n_neighbors=CONFIG["representation"]["n_neighbors"],
                min_dist=CONFIG["representation"]["min_dist"],
                metric=CONFIG["representation"]["metric"],
            )
        )

    projection_frames.append(
        joint_umap_projection(
            cue_clean_features,
            cue_features,
            cue_labels,
            cue_ids,
            backbone_name,
            "cue_conflict",
            seed=SEED,
            n_neighbors=CONFIG["representation"]["n_neighbors"],
            min_dist=CONFIG["representation"]["min_dist"],
            metric=CONFIG["representation"]["metric"],
        )
    )

    translation_displacement = CONFIG["representation"]["translation_displacement"]
    translation_vis_rows = []
    for position, image_id in enumerate(vis_ids):
        direction = CONFIG["translation"]["directions"][position % 4]
        matching = translation_manifest[
            translation_manifest.image_id.eq(image_id)
            & translation_manifest.displacement.eq(translation_displacement)
            & translation_manifest.direction.eq(direction)
        ]
        translation_vis_rows.append(matching.iloc[0])
    translation_vis_manifest = pd.DataFrame(translation_vis_rows).reset_index(drop=True)
    translation_vis_features, _, _ = manifest_features(
        bundle,
        translation_vis_manifest,
        f"{backbone_name}_translation_visualization",
    )
    projection_frames.append(
        joint_umap_projection(
            vis_clean,
            translation_vis_features,
            vis_labels,
            vis_ids,
            backbone_name,
            "translation_32_balanced_directions",
            seed=SEED,
            n_neighbors=CONFIG["representation"]["n_neighbors"],
            min_dist=CONFIG["representation"]["min_dist"],
            metric=CONFIG["representation"]["metric"],
        )
    )

    del bundle, head, train_features, validation_features, condition_features
    gc.collect()
    torch.cuda.empty_cache()

# %% [markdown]
# ## 9. Save required tables and inspect aggregate results
#
# The primary CSV files contain per-example records as well as summaries. This makes every aggregate number traceable to a specific image, condition, and predictor.

# %%
predictions = pd.concat(all_predictions, ignore_index=True)
performance = pd.DataFrame(performance_rows)
cue_summaries = pd.DataFrame(cue_summary_rows)
cue_decisions = pd.concat(cue_decision_frames, ignore_index=True)
translation_results = pd.DataFrame(translation_rows)
stability = pd.concat(stability_frames, ignore_index=True)
projections = pd.concat(projection_frames, ignore_index=True)
projections["class_name"] = projections.label.map(
    {index: name for index, name in enumerate(STL10_CLASSES)}
)

predictions.to_csv(RESULTS_ROOT / "per_example_predictions.csv", index=False)
performance.to_csv(RESULTS_ROOT / "performance_summary.csv", index=False)
cue_summaries.to_csv(RESULTS_ROOT / "cue_conflict_summary.csv", index=False)
cue_decisions.to_csv(RESULTS_ROOT / "cue_conflict_decisions.csv", index=False)
translation_results.to_csv(RESULTS_ROOT / "translation_summary_by_direction.csv", index=False)
stability.to_csv(RESULTS_ROOT / "representation_stability_per_example.csv", index=False)
summarize_stability(stability).to_csv(
    RESULTS_ROOT / "representation_stability_summary.csv", index=False
)
projections.to_csv(RESULTS_ROOT / "umap_coordinates.csv", index=False)

display(
    performance[
        [
            "predictor",
            "condition",
            "accuracy",
            "macro_f1",
            "mean_max_confidence",
            "prediction_consistency",
        ]
    ].sort_values(["predictor", "condition"])
)
display(cue_summaries)
display(summarize_stability(stability))

# %%
# Express intervention accuracy relative to each predictor's own clean baseline.
clean_accuracy = (
    performance[performance.condition.eq("clean")]
    .set_index("predictor")["accuracy"]
)
performance["accuracy_change_from_clean"] = performance.apply(
    lambda row: row.accuracy - clean_accuracy[row.predictor], axis=1
)
performance.to_csv(RESULTS_ROOT / "performance_summary.csv", index=False)

color_patch_table = performance[
    performance.condition.isin(["clean", "grayscale", "hue_90", "patch_shuffle"])
][
    [
        "predictor",
        "condition",
        "accuracy",
        "accuracy_change_from_clean",
        "macro_f1",
        "mean_max_confidence",
        "prediction_consistency",
    ]
]
display(color_patch_table.sort_values(["predictor", "condition"]))

# %% [markdown]
# ### Paired uncertainty for intervention accuracy changes
#
# The same images appear in clean and transformed conditions, so a paired class-stratified bootstrap is more appropriate than treating the two accuracy estimates as independent. The interval quantifies evaluation-sample uncertainty only; it does not capture variation from training the head with other random seeds.

# %%
intervention_bootstrap_rows = []
for predictor_name in sorted(performance.predictor.unique()):
    clean_block = predictions[
        predictions.predictor.eq(predictor_name)
        & predictions.condition.eq("clean")
    ]
    for condition in ["grayscale", "hue_90", "patch_shuffle"]:
        transformed_block = predictions[
            predictions.predictor.eq(predictor_name)
            & predictions.condition.eq(condition)
        ]
        comparison = stratified_paired_accuracy_difference(
            transformed_block,
            clean_block,
            resamples=CONFIG["uncertainty"]["bootstrap_resamples"],
            seed=SEED,
        )
        comparison.update({"predictor": predictor_name, "condition": condition})
        intervention_bootstrap_rows.append(comparison)

intervention_bootstrap = pd.DataFrame(intervention_bootstrap_rows)
intervention_bootstrap.to_csv(
    RESULTS_ROOT / "intervention_accuracy_change_bootstrap.csv", index=False
)
display(intervention_bootstrap.sort_values(["predictor", "condition"]))

# %% [markdown]
# ## 10. CLIP trained-head versus zero-shot paired bootstrap
#
# “Significantly better” is evaluated as a paired, class-stratified bootstrap interval for the accuracy difference. If the 95% interval excludes zero, the observed direction is reliable under this resampling analysis; this is not a substitute for multiple training seeds.

# %%
clip_trained_clean = predictions[
    predictions.predictor.eq("clip_vit_b32_linear")
    & predictions.condition.eq("clean")
]
clip_zero_clean = predictions[
    predictions.predictor.eq("clip_vit_b32_zeroshot")
    & predictions.condition.eq("clean")
]
clip_comparison = stratified_paired_accuracy_difference(
    clip_trained_clean,
    clip_zero_clean,
    resamples=CONFIG["uncertainty"]["bootstrap_resamples"],
    seed=SEED,
)
(RESULTS_ROOT / "clip_trained_vs_zeroshot_bootstrap.json").write_text(
    json.dumps(clip_comparison, indent=2) + "\n", encoding="utf-8"
)
clip_comparison

# %% [markdown]
# ## 11. Translation curves
#
# Direction-level measurements are retained in CSV. The plot averages directions at each nonzero displacement, as required by the manual. At displacement zero there is only one unshifted condition.

# %%
translation_curve = (
    translation_results.groupby(["predictor", "displacement"], as_index=False)
    .agg(
        accuracy=("accuracy", "mean"),
        prediction_consistency=("prediction_consistency", "mean"),
    )
)
translation_curve.to_csv(RESULTS_ROOT / "translation_curve.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
sns.lineplot(
    data=translation_curve,
    x="displacement",
    y="accuracy",
    hue="predictor",
    marker="o",
    ax=axes[0],
)
sns.lineplot(
    data=translation_curve,
    x="displacement",
    y="prediction_consistency",
    hue="predictor",
    marker="o",
    ax=axes[1],
)
axes[0].set_title("Translation accuracy")
axes[1].set_title("Prediction consistency with clean image")
for axis in axes:
    axis.set_xticks(CONFIG["translation"]["displacements"])
    axis.set_ylim(0, 1)
axes[1].legend_.remove()
plt.tight_layout()
plt.savefig(FIGURES_ROOT / "translation_curves.png", dpi=220, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 12. UMAP visualizations
#
# Each panel was fitted independently to the combined clean and transformed features for one backbone/intervention. Colors indicate the ground-truth content class and marker style indicates condition. Absolute coordinates must not be compared between panels.

# %%
intervention_order = [
    "grayscale",
    "cue_conflict",
    "translation_32_balanced_directions",
    "patch_shuffle",
]
fig, axes = plt.subplots(
    len(BACKBONE_NAMES), len(intervention_order), figsize=(22, 15)
)
palette = sns.color_palette("tab10", n_colors=10)
for row, backbone_name in enumerate(BACKBONE_NAMES):
    for column, intervention in enumerate(intervention_order):
        axis = axes[row, column]
        subset = projections[
            projections.backbone.eq(backbone_name)
            & projections.intervention.eq(intervention)
        ]
        sns.scatterplot(
            data=subset,
            x="umap_1",
            y="umap_2",
            hue="class_name",
            style="condition",
            palette=palette,
            s=22,
            alpha=0.75,
            linewidth=0,
            ax=axis,
            legend=(row == 0 and column == len(intervention_order) - 1),
        )
        axis.set_title(f"{backbone_name} — {intervention}")
        axis.set_xticks([])
        axis.set_yticks([])
plt.tight_layout()
plt.savefig(FIGURES_ROOT / "umap_clean_vs_transformed.png", dpi=220, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 13. Cue-conflict agreements, disagreements, and failures
#
# This section identifies examples for human interpretation; it does not automatically claim why a model made a decision. Prefer cases that reveal agreement, differing cue selection, or “other” predictions rather than selecting only visually dramatic examples.

# %%
cue_wide = (
    cue_decisions.pivot_table(
        index="image_id",
        columns="predictor",
        values="decision_type",
        aggfunc="first",
    )
    .reset_index()
    .merge(
        selected_conflicts[
            ["image_id", "content_class", "style_class", "path"]
        ],
        on="image_id",
        validate="one_to_one",
    )
)
predictor_columns = [column for column in cue_wide if column.endswith(("linear", "zeroshot"))]
cue_wide["number_of_decision_types"] = cue_wide[predictor_columns].nunique(axis=1)
cue_wide["contains_other"] = cue_wide[predictor_columns].eq("other").any(axis=1)
cue_wide.to_csv(RESULTS_ROOT / "cue_conflict_case_table.csv", index=False)

informative = pd.concat(
    [
        cue_wide[cue_wide.number_of_decision_types.gt(1)].head(4),
        cue_wide[cue_wide.contains_other].head(2),
    ]
).drop_duplicates("image_id").head(6)
display(informative.drop(columns=["path"]))

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for axis, row in zip(axes.flat, informative.itertuples(index=False)):
    axis.imshow(Image.open(row.path))
    axis.set_title(f"shape={row.content_class}, texture={row.style_class}")
    axis.axis("off")
for axis in axes.flat[len(informative):]:
    axis.axis("off")
plt.tight_layout()
plt.savefig(FIGURES_ROOT / "cue_conflict_informative_cases.png", dpi=220, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 14. Evidence checklist and export
#
# Before using these outputs in the report, verify:
#
# - clean, grayscale, hue, and patch-shuffle results exist for all four predictors;
# - cue-conflict counts sum to 200 per predictor and include shape bias plus coverage;
# - translation curves contain 0, 8, 16, and 32 pixels;
# - representation stability includes grayscale, cue conflict, every nonzero translation, and patch shuffle for all three backbones;
# - UMAP settings and subset identifiers are saved;
# - candidate accepted/rejected counts and reasons are retained;
# - selected examples are inspected rather than treated as automatically explanatory.

# %%
expected_predictors = {
    "resnet50_linear",
    "vit_b16_linear",
    "clip_vit_b32_linear",
    "clip_vit_b32_zeroshot",
}
expected_conditions = {"clean", "grayscale", "hue_90", "patch_shuffle"}

assert expected_predictors.issubset(set(performance.predictor))
for predictor_name in expected_predictors:
    observed = set(performance[performance.predictor.eq(predictor_name)].condition)
    assert expected_conditions.issubset(observed), (predictor_name, observed)
assert cue_summaries[["shape_count", "texture_count", "other_count"]].sum(axis=1).eq(200).all()
assert set(translation_curve.displacement) == set(CONFIG["translation"]["displacements"])
assert set(stability.backbone) == set(BACKBONE_NAMES)
assert {"grayscale", "cue_conflict", "translation", "patch_shuffle"}.issubset(
    set(stability.intervention)
)
assert reviewed_manifest.decision.str.lower().isin(["accept", "reject"]).all()

RUN_METADATA["completed_utc"] = datetime.now(timezone.utc).isoformat()
RUN_METADATA["status"] = "complete"
(RESULTS_ROOT / "run_metadata.json").write_text(
    json.dumps(RUN_METADATA, indent=2) + "\n", encoding="utf-8"
)

archive_base = Path("/content/task1_results_seed6304")
archive_path = shutil.make_archive(
    str(archive_base),
    "zip",
    root_dir=RESULTS_ROOT,
)
print(f"Task 1 evidence checks passed. Download: {archive_path}")

# %% [markdown]
# ## 15. Interpretation prompts for the repository owner
#
# Use the saved evidence—not architectural stereotypes—to answer:
#
# 1. Do grayscale and hue rotation support the pre-registered “no substantial degradation” hypothesis for each predictor?
# 2. Is shape bias above 50%, and is coverage high enough for that number to be persuasive?
# 3. Which predictor changes most under translation and patch shuffling, relative to its own clean baseline?
# 4. Where do prediction consistency and cosine representation stability agree, and where do they diverge?
# 5. Does the trained CLIP head outperform zero-shot CLIP, and does the paired bootstrap interval exclude zero?
# 6. Which findings could plausibly reflect architecture, and which remain confounded by pretraining data, supervision, augmentation, or model capacity?
#
# These prompts organize inspection only. The final interpretation and report language should be written by the repository owner.
