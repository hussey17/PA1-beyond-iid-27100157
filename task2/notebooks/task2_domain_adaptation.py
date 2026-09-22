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
# # Task 2 — Unsupervised Domain Adaptation
#
# This notebook implements the complete PACS protocol for Source-only ERM, DAN,
# DANN, and CDAN, followed by the approved DAN alignment-strength study at
# `lambda_MMD = {0.1, 1, 10}`.
#
# The workflow enforces three boundaries:
#
# 1. Create and save the shared Task 2/Task 3 source splits.
# 2. Train and select every checkpoint using source labels and source-validation
#    macro-F1 only. Sketch is exposed only through an unlabeled dataset view.
# 3. Hash and lock every configuration/checkpoint before enabling final
#    label-aware target evaluation.
#
# The notebook produces evidence and machine-readable results; it does not write
# report conclusions.

# %% [markdown]
# ## 0. Colab bootstrap and dependencies

# %%
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_URL = "https://github.com/hussey17/PA1-beyond-iid-27100157.git"
REPOSITORY_ROOT = Path("/content/PA1-beyond-iid-27100157")

if Path("/content").exists():
    if not REPOSITORY_ROOT.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", REPOSITORY_URL, str(REPOSITORY_ROOT)],
            check=True,
        )
    else:
        subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "pull", "--ff-only"], check=True
        )
    os.chdir(REPOSITORY_ROOT)
else:
    REPOSITORY_ROOT = Path.cwd().resolve()

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-r",
        str(REPOSITORY_ROOT / "task2/requirements-colab.txt"),
    ],
    check=True,
)
CURRENT_COMMIT = subprocess.check_output(
    ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "--short", "HEAD"], text=True
).strip()
print(f"Working directory: {Path.cwd()}")
print(f"Repository commit: {CURRENT_COMMIT}")

# %% [markdown]
# ## 1. Imports, configuration, and reproducibility

# %%
import gc
import importlib
import inspect
import json
import platform
import shutil
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import yaml
from IPython.display import display

# Colab can retain imported modules even after `git pull` updates their files.
# Purge repository modules before importing them so this notebook cannot call a
# stale downloader or training implementation from an earlier runtime state.
for loaded_name in tuple(sys.modules):
    if loaded_name in {"shared", "task2"} or loaded_name.startswith(
        ("shared.", "task2.")
    ):
        del sys.modules[loaded_name]
importlib.invalidate_caches()

from shared.pacs import (
    PACS_CLASSES,
    PACS_PROVIDER_SIGNATURE,
    PACSLabeledDataset,
    PACSUnlabeledDataset,
    prepare_pacs_from_huggingface,
    scan_domain,
    scan_unlabeled_domain,
    validate_inventory,
)
from shared.pacs_protocol import (
    SOURCE_DOMAINS,
    TARGET_DOMAIN,
    build_transforms,
    load_split_manifest,
    make_source_splits,
    save_split_manifest,
    seed_everything,
)
from task2.evaluate_final import evaluate_locked_experiments
from task2.experiment_io import (
    load_config,
    lock_experiments,
    mark_target_label_access,
    validate_experiment_lock,
)
from task2.train import run_training

BASE_CONFIG_PATH = REPOSITORY_ROOT / "task2/configs/base.yaml"
CONFIG_ROOT = REPOSITORY_ROOT / "task2/configs"
with BASE_CONFIG_PATH.open("r", encoding="utf-8") as stream:
    BASE_CONFIG = yaml.safe_load(stream)

SEED = int(BASE_CONFIG["seed"])
seed_everything(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    raise RuntimeError(
        "Task 2 fine-tunes six ResNet-18 runs. Select a Colab T4 GPU runtime, then rerun."
    )

DATA_ROOT = REPOSITORY_ROOT / "data"
SPLIT_PATH = REPOSITORY_ROOT / "shared/splits/pacs_sketch_seed6304.json"
RESULTS_ROOT = REPOSITORY_ROOT / "task2/results/run_seed6304"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "task2/checkpoints"
FINAL_RESULTS_ROOT = RESULTS_ROOT / "final_evaluation"
FIGURES_ROOT = RESULTS_ROOT / "figures"
LOCK_PATH = RESULTS_ROOT / "experiment_lock.json"
for directory in [DATA_ROOT, RESULTS_ROOT, CHECKPOINT_ROOT, FIGURES_ROOT]:
    directory.mkdir(parents=True, exist_ok=True)

RUN_METADATA = {
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "seed": SEED,
    "python": platform.python_version(),
    "pytorch": torch.__version__,
    "cuda_device": torch.cuda.get_device_name(0),
    "target_labels_consulted_at_start": (
        json.loads(LOCK_PATH.read_text(encoding="utf-8")).get(
            "target_labels_consulted", False
        )
        if LOCK_PATH.is_file()
        else False
    ),
}
(RESULTS_ROOT / "run_metadata.json").write_text(
    json.dumps(RUN_METADATA, indent=2) + "\n", encoding="utf-8"
)
print(yaml.safe_dump(BASE_CONFIG, sort_keys=False))
print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 2. Pre-registered controlled-study hypothesis
#
# Increasing `lambda_MMD` is expected to make pooled source and target features
# harder to distinguish. Source-validation performance should remain relatively
# stable at weak or moderate alignment but may decline at the strongest setting.
# Target recognition is expected to be non-monotonic: moderate alignment may
# remove domain-specific nuisance information, while excessive marginal
# alignment may mix classes and cause negative transfer.
#
# The main comparison remains fixed at `lambda_MMD = 1`, irrespective of target
# results.

# %% [markdown]
# ## 3. Download, validate, and inventory PACS

# %%
active_loader_source = inspect.getsource(prepare_pacs_from_huggingface)
if "load_dataset" not in active_loader_source or "gdown" in active_loader_source:
    raise RuntimeError(
        "A stale PACS loader is still active. Use Runtime > Disconnect and delete "
        "runtime, reopen the notebook, and run from the first cell."
    )
print(f"PACS provider: {PACS_PROVIDER_SIGNATURE}")
PACS_ROOT = prepare_pacs_from_huggingface(DATA_ROOT)
samples_by_domain = {
    domain: scan_domain(PACS_ROOT, domain) for domain in SOURCE_DOMAINS
}
target_unlabeled_items = scan_unlabeled_domain(PACS_ROOT, TARGET_DOMAIN)
for samples in samples_by_domain.values():
    validate_inventory(samples)

inventory_rows = []
for domain, samples in samples_by_domain.items():
    for class_name in PACS_CLASSES:
        inventory_rows.append(
            {
                "domain": domain,
                "class_name": class_name,
                "count": sum(sample.class_name == class_name for sample in samples),
            }
        )
inventory = pd.DataFrame(inventory_rows)
inventory.to_csv(RESULTS_ROOT / "pacs_inventory.csv", index=False)
display(inventory.pivot(index="class_name", columns="domain", values="count"))
print(f"Unlabeled target images: {len(target_unlabeled_items)}")
assert inventory["count"].sum() + len(target_unlabeled_items) == 9991

# %% [markdown]
# ## 4. Freeze the shared Task 2/Task 3 source splits

# %%
if not SPLIT_PATH.is_file():
    split_identifiers = make_source_splits(
        {domain: samples_by_domain[domain] for domain in SOURCE_DOMAINS},
        validation_fraction=float(BASE_CONFIG["dataset"]["validation_fraction"]),
        seed=SEED,
    )
    save_split_manifest(
        SPLIT_PATH,
        split_identifiers,
        seed=SEED,
        validation_fraction=float(BASE_CONFIG["dataset"]["validation_fraction"]),
    )

source_splits = load_split_manifest(SPLIT_PATH, samples_by_domain, expected_seed=SEED)
shutil.copy2(SPLIT_PATH, RESULTS_ROOT / SPLIT_PATH.name)
split_summary = pd.DataFrame(
    [
        {
            "domain": domain,
            "train": len(source_splits[domain]["train"]),
            "validation": len(source_splits[domain]["validation"]),
        }
        for domain in SOURCE_DOMAINS
    ]
)
display(split_summary)

train_transform, evaluation_transform = build_transforms(
    resize_size=int(BASE_CONFIG["preprocessing"]["resize_size"]),
    crop_size=int(BASE_CONFIG["preprocessing"]["crop_size"]),
)
source_train_datasets = {
    domain: PACSLabeledDataset(source_splits[domain]["train"], train_transform)
    for domain in SOURCE_DOMAINS
}
source_validation_datasets = {
    domain: PACSLabeledDataset(
        source_splits[domain]["validation"], evaluation_transform
    )
    for domain in SOURCE_DOMAINS
}
target_unlabeled_dataset = PACSUnlabeledDataset(
    target_unlabeled_items, transform=train_transform
)

# Verify that the target training view exposes no labels.
target_item = target_unlabeled_dataset[0]
assert len(target_item) == 2 and isinstance(target_item[1], str)
del target_item

# %% [markdown]
# ## 5. Resolve all configurations before training
#
# Six runs are fixed in advance. The `dan` run at weight 1 is shared by the main
# comparison and controlled study, so only the 0.1 and 10 endpoints add work.

# %%
RUN_CONFIGS = {
    "source_only": load_config(BASE_CONFIG_PATH, CONFIG_ROOT / "source_only.yaml"),
    "dan_lambda_0p1": load_config(
        BASE_CONFIG_PATH,
        CONFIG_ROOT / "dan.yaml",
        {"method": {"mmd_weight": 0.1}},
    ),
    "dan": load_config(BASE_CONFIG_PATH, CONFIG_ROOT / "dan.yaml"),
    "dan_lambda_10": load_config(
        BASE_CONFIG_PATH,
        CONFIG_ROOT / "dan.yaml",
        {"method": {"mmd_weight": 10.0}},
    ),
    "dann": load_config(BASE_CONFIG_PATH, CONFIG_ROOT / "dann.yaml"),
    "cdan": load_config(BASE_CONFIG_PATH, CONFIG_ROOT / "cdan.yaml"),
}
assert RUN_CONFIGS["dan"]["method"]["mmd_weight"] == 1.0
for run_name, config in RUN_CONFIGS.items():
    resolved_path = RESULTS_ROOT / "planned_configs" / f"{run_name}.yaml"
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
display(
    pd.DataFrame(
        [
            {
                "run_name": name,
                "method": config["method"]["name"],
                "mmd_weight": config["method"].get("mmd_weight"),
            }
            for name, config in RUN_CONFIGS.items()
        ]
    )
)

# %% [markdown]
# ## 6. Train and source-select every checkpoint
#
# Existing complete runs are reused only when their resolved-configuration hash
# matches. A mismatched run aborts instead of silently overwriting results.

# %%
TRAINING_RECORDS = []
for run_name, config in RUN_CONFIGS.items():
    print(f"\n===== {run_name} =====")
    record = run_training(
        config=config,
        run_name=run_name,
        source_train_datasets=source_train_datasets,
        source_validation_datasets=source_validation_datasets,
        target_unlabeled_dataset=(
            None
            if config["method"]["name"] == "source_only"
            else target_unlabeled_dataset
        ),
        output_root=RESULTS_ROOT / "training",
        checkpoint_root=CHECKPOINT_ROOT,
        device=DEVICE,
        reuse_if_complete=True,
    )
    TRAINING_RECORDS.append(record)
    display(pd.DataFrame([record]))
    gc.collect()
    torch.cuda.empty_cache()

# %% [markdown]
# ## 7. Inspect source-only training diagnostics and lock experiments
#
# This stage plots only training losses and source-validation metrics. It then
# stores hashes of every resolved configuration and selected checkpoint. Once a
# different lock exists, the notebook refuses to replace it.

# %%
history_frames = []
for run_name in RUN_CONFIGS:
    history = pd.read_csv(RESULTS_ROOT / "training" / run_name / "training_history.csv")
    history.insert(0, "run_name", run_name)
    history_frames.append(history)
training_history = pd.concat(history_frames, ignore_index=True)
training_history.to_csv(RESULTS_ROOT / "all_training_history.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.lineplot(
    data=training_history,
    x="epoch",
    y="classification_loss",
    hue="run_name",
    ax=axes[0],
)
axes[0].set_title("Source classification loss")
sns.lineplot(
    data=training_history[training_history["run_name"] != "source_only"],
    x="epoch",
    y="alignment_loss",
    hue="run_name",
    ax=axes[1],
)
axes[1].set_title("MMD or domain loss (raw)")
fig.tight_layout()
fig.savefig(FIGURES_ROOT / "training_losses.png", dpi=180, bbox_inches="tight")
plt.show()

LOCK = lock_experiments(TRAINING_RECORDS, LOCK_PATH)
print(f"Locked {len(LOCK['runs'])} configurations/checkpoints at {LOCK_PATH}")
print("No target class label has been loaded by the training pipeline.")

# %% [markdown]
# ## 8. Explicit final target-evaluation gate
#
# Leave this flag `False` until all preceding runs and source-side diagnostics
# have completed and you accept the frozen configurations. Changing it to
# `True` records target-label access in the lock **before** constructing the
# labeled Sketch dataset. Target results are analysis-only and must not be used
# to revise a method, setting, or checkpoint.

# %%
RUN_FINAL_TARGET_EVALUATION = False  # Change once, only after accepting the lock above.

FINAL_SUMMARY = None
if not RUN_FINAL_TARGET_EVALUATION:
    print(
        "Final target evaluation is intentionally disabled. Confirm the experiment lock, "
        "set RUN_FINAL_TARGET_EVALUATION=True, and rerun from this cell."
    )
else:
    lock_records = [
        {
            "run_name": record["run_name"],
            "config_fingerprint": record["config_fingerprint"],
            "checkpoint_path": record["checkpoint_path"],
        }
        for record in TRAINING_RECORDS
    ]
    validate_experiment_lock(LOCK_PATH, lock_records)
    mark_target_label_access(LOCK_PATH)
    target_labeled_samples = scan_domain(PACS_ROOT, TARGET_DOMAIN)
    target_labeled_dataset = PACSLabeledDataset(
        target_labeled_samples, transform=evaluation_transform
    )
    run_specs = [
        {"record": record, "config": RUN_CONFIGS[record["run_name"]]}
        for record in TRAINING_RECORDS
    ]
    FINAL_SUMMARY = evaluate_locked_experiments(
        run_specs=run_specs,
        lock_path=LOCK_PATH,
        source_validation_datasets=source_validation_datasets,
        target_labeled_dataset=target_labeled_dataset,
        class_names=list(PACS_CLASSES),
        output_root=FINAL_RESULTS_ROOT,
        device=DEVICE,
    )
    display(FINAL_SUMMARY)

# %% [markdown]
# ## 9. Required comparison and controlled-study figures

# %%
if FINAL_SUMMARY is None:
    print(
        "Run the gated final-evaluation cell before generating target-result figures."
    )
else:
    main_order = ["source_only", "dan", "dann", "cdan"]
    main_comparison = (
        FINAL_SUMMARY[FINAL_SUMMARY["run_name"].isin(main_order)]
        .set_index("run_name")
        .loc[main_order]
        .reset_index()
    )
    main_comparison.to_csv(
        FINAL_RESULTS_ROOT / "required_method_comparison.csv", index=False
    )
    display(main_comparison)

    strength = FINAL_SUMMARY[FINAL_SUMMARY["method"] == "dan"].sort_values("mmd_weight")
    strength.to_csv(FINAL_RESULTS_ROOT / "dan_strength_study.csv", index=False)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, metric, title in [
        (axes[0], "domain_separability", "Domain separability"),
        (axes[1], "mean_source_macro_f1", "Mean source macro-F1"),
        (axes[2], "target_macro_f1", "Target macro-F1"),
    ]:
        sns.lineplot(data=strength, x="mmd_weight", y=metric, marker="o", ax=axis)
        axis.set_xscale("log")
        axis.set_title(title)
        axis.set_xlabel("lambda_MMD")
    figure.tight_layout()
    figure.savefig(
        FIGURES_ROOT / "dan_alignment_strength.png", dpi=180, bbox_inches="tight"
    )
    plt.show()

# %% [markdown]
# ## 10. Completeness audit and export
#
# The export contains compact results, figures, the exact shared split manifest,
# and the Source-only checkpoint required unchanged by Task 3. Other large
# checkpoints remain local and are intentionally excluded from Git.

# %%
if FINAL_SUMMARY is None:
    print("Export is available after the gated final evaluation.")
else:
    assert set(FINAL_SUMMARY["run_name"]) == set(RUN_CONFIGS)
    assert (RESULTS_ROOT / "all_training_history.csv").is_file()
    assert (FINAL_RESULTS_ROOT / "required_method_comparison.csv").is_file()
    assert (FINAL_RESULTS_ROOT / "dan_strength_study.csv").is_file()
    assert all(
        (FINAL_RESULTS_ROOT / f"{name}_target_predictions.csv").is_file()
        for name in RUN_CONFIGS
    )

    export_staging = Path("/content/task2_export_seed6304")
    if export_staging.exists():
        shutil.rmtree(export_staging)
    export_staging.mkdir(parents=True)
    shutil.copytree(RESULTS_ROOT, export_staging / "results")
    shutil.copy2(SPLIT_PATH, export_staging / SPLIT_PATH.name)
    source_checkpoint = Path(
        next(
            record["checkpoint_path"]
            for record in TRAINING_RECORDS
            if record["run_name"] == "source_only"
        )
    )
    shutil.copy2(source_checkpoint, export_staging / source_checkpoint.name)
    archive = shutil.make_archive(
        "/content/task2_results_seed6304", "zip", export_staging
    )
    print(f"Exported: {archive}")
    print(f"Task 3 ERM checkpoint: {source_checkpoint}")
