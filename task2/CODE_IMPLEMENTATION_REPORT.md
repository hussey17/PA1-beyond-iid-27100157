# Task 2 Code Implementation Reference

This document explains the implementation and its relationship to the assignment
requirements. It is a code reference, not submission-report prose and not an
interpretation of experimental results.

## 1. Experimental boundary

Task 2 is transductive unsupervised domain adaptation. Photo, Art Painting, and
Cartoon provide labeled source examples. Every Sketch image may participate in
adaptation, but its class label may not influence training, checkpoint selection,
hyperparameters, or method design.

The implementation makes this boundary structural:

- `PACSLabeledDataset` is used for sources and, later, final evaluation.
- Before the lock, Sketch is inventoried by `scan_unlabeled_domain`, which never
  parses its class-folder names. `PACSUnlabeledDataset` stores and returns only an image path and identifier.
  It has no label field in its internal items and returns no label from
  `__getitem__`.
- `task2/train.py` accepts only the unlabeled target view. It does not construct
  a labeled target loader or compute a target classification metric.
- `task2/evaluate_final.py` is a separate label-aware entry point.
- Before final evaluation, `experiment_lock.json` records SHA-256 hashes of all
  selected checkpoints and resolved configurations. The notebook marks target
  access before constructing the labeled Sketch dataset. A changed checkpoint,
  configuration, or run set then fails validation.

The lock cannot prevent a person from deliberately deleting evidence, but it
prevents accidental post-target tuning in the normal notebook workflow and
makes the intended chronology auditable.

## 2. PACS protocol shared with Task 3

`shared/pacs.py` discovers and validates the standard folder structure:

```text
PACS/
  photo/{dog,...,person}/
  art_painting/{dog,...,person}/
  cartoon/{dog,...,person}/
  sketch/{dog,...,person}/
```

The downloader first accepts any existing standard PACS folder unchanged. When
no such folder exists, it loads the public `flwrlabs/pacs` Hugging Face dataset
at pinned revision `394113073258ead631f617d2e13bb377c0715c4b`. It checks the
9,991-row count, required columns, seven-class order, and four domain names. The
original encoded image bytes are written into a temporary canonical folder tree
and moved into place only after full inventory validation. Hugging Face's cache
provides resumable content-addressed downloads, while the temporary tree keeps
an interrupted materialization from being mistaken for a ready dataset.

Colab retains Python modules across repeated notebook opens within one runtime.
After updating the Git checkout, the notebook explicitly removes all loaded
`shared.*` and `task2.*` modules before importing the experiment code. The data
cell prints the pinned Hugging Face provider and inspects the active loader
source, aborting with a clear stale-runtime message if an obsolete `gdown`
implementation is somehow still resident.

`shared/pacs_protocol.py` creates an independent stratified 80/20 split inside
each source domain with seed 6304. The manifest stores stable paths relative to
the PACS root, rather than machine-specific absolute paths. Loading the manifest
checks its seed, disjointness, missing identifiers, and exact domain coverage.
This same file and protocol are intended to be imported unchanged by Task 3.

The training transform is resize to 256x256, random 224x224 crop, random
horizontal flip, tensor conversion, and ImageNet V1 normalization. Validation
and evaluation replace the random crop with a 224x224 center crop and omit the
flip. Source and unlabeled target training images use the same stochastic
transform family.

## 3. Model components

### Backbone and classifier

`ResNet18Backbone` loads torchvision `ResNet18_Weights.IMAGENET1K_V1`, replaces
the ImageNet fully connected layer with identity, and returns the required
512-dimensional feature. `ClassifierHead` is one trainable linear map from 512
features to the seven PACS classes. The complete backbone and classifier are
fine-tuned for every method.

After each call to training mode, every BatchNorm module is immediately returned
to evaluation mode. Its ImageNet running mean and variance therefore remain
fixed, while its affine `weight` and `bias` parameters retain gradients. This
avoids introducing an uncontrolled source-target-mixture adaptation mechanism.

The experiment seed is reset before model construction for every run. Therefore
the pretrained backbone and newly initialized classifier begin identically
across methods. DANN/CDAN create their method-specific discriminator only after
the shared components have been initialized.

### Domain discriminator

Both adversarial methods use the required architecture:

```text
input -> Linear(256) -> ReLU -> Dropout(0.5) -> Linear(2)
```

DANN receives a 512-vector. CDAN receives a `7 x 512 = 3584`-dimensional
flattened class-probability/feature outer product.

## 4. Method objectives

### Source-only ERM

The three source mini-batches contain eight examples each. They are concatenated
and ordinary cross-entropy is averaged over the resulting 24 examples. Equal
per-domain batch counts give every source domain equal influence. No Sketch
image is loaded by this method. Its selected checkpoint is exported for direct,
unchanged reuse as Task 3 ERM.

### DAN

DAN minimizes

```text
classification_loss + lambda_MMD * MMD_squared(source_features, target_features)
```

All 24 pooled source features are compared with 24 target features at the
512-dimensional pre-classifier representation. The implementation computes the
biased empirical squared mean-embedding distance:

```text
mean(K_ss) + mean(K_tt) - 2 * mean(K_st)
```

Including the diagonal terms produces the direct empirical squared RKHS mean
distance and keeps the optimized estimate nonnegative. For every batch, the
base bandwidth is the detached median of positive off-diagonal squared distances
over the combined source-target batch. Three RBF matrices are evaluated with
bandwidths `0.5`, `1`, and `2` times that base and are summed, as required.

The main run uses weight 1. The controlled study adds weights 0.1 and 10. These
are all fixed before target evaluation; the weight-1 checkpoint serves both the
main table and middle study point.

### DANN

DANN concatenates source and target features and trains the domain discriminator
with binary cross-entropy. A custom autograd operation is identity in the
forward pass and multiplies the feature gradient by `-alpha` in the backward
pass. The schedule is

```text
alpha(p) = alpha_max * (2 / (1 + exp(-10 p)) - 1)
```

where `p` is optimizer-step progress over the planned 30-epoch budget and
`alpha_max = 1`. Domain-loss weight remains one. The discriminator therefore
receives its ordinary minimizing gradient; only the feature extractor receives
the scheduled reversed gradient. This avoids accidentally applying `alpha`
twice.

### CDAN

For feature `f` and softmax class probabilities `p`, CDAN forms the flattened
outer product `f outer p`. Neither input is detached, so domain gradients can
flow through both the feature extractor and classifier as the manual requires.
The conditional vector passes through the same discriminator and reversal
schedule as DANN. Entropy conditioning is intentionally absent.

## 5. Common optimization and checkpoint selection

All runs use AdamW with learning rate `1e-4`, weight decay `1e-4`, at most 30
epochs, and seed 6304. Mixed precision is enabled only on CUDA; MMD distance
calculations are promoted to float32.

An epoch is defined as the number of steps in the largest source-domain loader.
Each source loader and the target loader is independently cycled when exhausted.
Consequently every adaptation update always contains:

- 8 Photo examples;
- 8 Art Painting examples;
- 8 Cartoon examples;
- 24 unlabeled Sketch examples.

The source and target totals are both 24. Source-only uses the same source
iteration rule without loading Sketch.

At each epoch, accuracy and macro-F1 are computed separately for all three
source validation domains. Their unweighted mean macro-F1 selects the checkpoint.
Only a strictly greater score counts as improvement, so the earliest checkpoint
is retained on a tie. Training stops after five consecutive non-improving
epochs. Checkpoints and JSON metadata use temporary-file replacement to avoid
leaving a truncated file after interruption. A completed run is reused only if
its configuration fingerprint matches.

Training histories include classification loss, raw alignment/domain loss,
total loss, scheduled reversal strength, DANN/CDAN discriminator accuracy, and
each source validation metric. Raw and weighted MMD are distinguishable from the
resolved configuration: `alignment_loss` is raw MMD and `total_loss` contains
its configured multiplier.

## 6. Final evaluation and required evidence

After the lock, every selected model is evaluated on each source validation
domain and the complete labeled Sketch domain. Saved outputs include:

- per-source and mean source accuracy and macro-F1;
- target accuracy and macro-F1;
- target accuracy change from the exact Source-only checkpoint;
- per-example target prediction and confidence;
- per-class target accuracy and change from Source-only;
- complete target confusion matrices and dominant incorrect class pairs;
- training classification/alignment curves;
- the DAN strength table and three-panel strength plot.

For domain separability, each frozen backbone supplies all pooled source
validation features and all target features. Seed 6304 selects the same number
from each side, producing a balanced binary dataset. A stratified 70/30 split
then trains a logistic-regression probe with `C=1`; held-out accuracy is the
reported score, with 50% as chance. No feature standardization or extra probe
hyperparameter tuning is introduced because the manual specifies only the
balanced logistic model and `C`.

The training discriminator's accuracy is retained only as an optimization
diagnostic. It is not substituted for the post-hoc separability probe because a
near-chance adversarial discriminator could reflect successful confusion, weak
optimization, or feature collapse.

## 7. Files, reruns, and export

Small CSV/JSON/YAML results are written below `task2/results/run_seed6304`.
Checkpoints are written below the Git-ignored `task2/checkpoints` directory.
The final ZIP contains the results directory, exact split manifest, and selected
Source-only checkpoint. This checkpoint must be passed unchanged to Task 3.

The notebook is paired with a percent-format Python source. The `.py` file is
the reviewable source of truth for code changes; the `.ipynb` file is generated
from it for Colab execution. No raw PACS image, feature cache, or large model
checkpoint is committed to Git.

## 8. Intentional non-choices

The code does not tune on Sketch labels, alter the required main settings after
the controlled study, update BatchNorm running statistics, add entropy
conditioning to CDAN, detach CDAN inputs, use target pseudo-labels, standardize
the domain-probe features, or introduce a learning-rate schedule. Each would
change the assigned comparison rather than merely implement it.
