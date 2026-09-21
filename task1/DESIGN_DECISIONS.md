# Task 1 Design Decision Register

No experiment implementation should begin until the owner approves the decisions marked **Required**. Items marked **Implementation default** are documented for transparency but can be delegated without changing the research question.

## A. Required experimental-design decisions

### 1. Dataset

- Choose **STL-10** or **Oxford-IIIT Pets**.
- Recommended starting point: **STL-10**, because it has ten distinct classes, a naturally balanced 500-image test subset (50 per class), and lower compute and cue-conflict curation cost.
- Record the hypothesis for how the chosen class granularity may affect color, shape, and texture reliance.

### 2. Additional color intervention

Grayscale is mandatory. Choose exactly one additional intervention and its fixed parameters:

- **Fixed hue rotation**: cleanest controlled test of chromatic sensitivity; select one angle applied identically to all images.
- **Palette transfer**: more natural-looking but introduces a reference-palette choice and possible texture/contrast changes.
- **Class-swapped color statistics**: tests class-associated color cues directly but requires a deterministic class-pair/mapping rule and can introduce distribution artifacts.

Pre-register what the transformation changes and preserves, a directional hypothesis, and the primary metrics (accuracy change and prediction consistency are mandatory).

### 3. Cue-conflict construction

Approve all of the following before generation:

- Style-transfer method/implementation (AdaIN is the manual's practical default).
- At least five unordered class pairs.
- Sampling rule and target count per pair and direction, with at least 200 accepted conflicts total.
- Style strength and any other generator parameters.
- A model-independent visual rejection rule defined before evaluation.
- Human-review workflow for accepted/rejected images; model predictions must never determine retention.

Pre-register the expected ordering of shape bias and coverage across ResNet-50, ViT-B/16, and CLIP. Shape bias and coverage are co-primary metrics; raw shape, texture, and other counts must also be retained.

### 4. Representation visualization

- Choose **t-SNE** or **UMAP**.
- Choose projection settings (for example t-SNE perplexity/learning rate/iterations or UMAP neighbors/minimum distance/metric).
- Choose the fixed, class-balanced visualization subset size.
- Decide whether translation is visualized at every nonzero displacement or at one pre-registered displacement (32 px is the strongest diagnostic default).
- Decide figure organization: one panel per backbone/intervention or a smaller pre-registered comparison set.

Each backbone must fit its own 2D projection to combined clean and transformed features. Absolute coordinates must not be compared across separately fitted projections. Cosine representation stability remains the primary quantitative measure.

### 5. Hypotheses and decision criteria

Before seeing results, approve a short directional hypothesis for:

- dataset/class granularity;
- grayscale and the selected color transformation;
- cue-conflict shape bias and coverage;
- translation sensitivity as displacement increases;
- patch-shuffle sensitivity;
- representation/prediction agreement or mismatch;
- trained CLIP head versus zero-shot CLIP.

The manual fixes the central metrics. Decide whether to add uncertainty estimates (recommended: class-stratified bootstrap confidence intervals over evaluation examples) without changing the single-seed training protocol.

## B. Required operational choices not fixed by the manual

These affect reproducibility and should be approved once, then held constant:

- Execution platform/device and acceptable compute/storage budget.
- Linear-head training batch size.
- Common deterministic geometric preprocessing before model-specific normalization (recommended: resize shorter side to 256, center crop to 224, RGB conversion, with no stochastic test augmentation).
- Whether the linear-head training split uses any augmentation (recommended: none beyond deterministic preprocessing, to isolate the frozen representations).
- Loss for linear heads (recommended: multiclass cross-entropy) and no learning-rate scheduler unless explicitly approved.
- Confidence-interval policy, if any.
- Cue-conflict visual-QA mechanism and who performs the final accept/reject review.
- External style-transfer source and license/attribution.

## C. Implementation defaults that may be delegated

Unless the owner requests otherwise, these can be implemented deterministically and documented:

- Store split/subset identifiers and cue-conflict manifests as JSON/CSV.
- Cache frozen features with labels, image identifiers, model name, preprocessing version, and checksum metadata.
- Use the manual's fixed seed, pretrained weights, prompt, optimizer settings, epoch cap, and patience exactly as written.
- Select the best head checkpoint by validation accuracy; define improvement as strictly greater accuracy and keep the earliest checkpoint on ties.
- Apply model-specific normalization only after constructing the same 224x224 RGB clean/transformed image.
- Use the clean prediction for pairwise prediction consistency.
- Use each cue-conflict content image as its clean representation counterpart.
- Generate one deterministic, non-identity 4x4 patch permutation per image and reuse it for every model.
- Aggregate translation metrics equally across the four cardinal directions at each displacement.
- Save tidy, machine-readable per-example predictions and summary tables so every reported number is traceable.

## D. Manual-fixed choices (not open for redesign)

- Backbones: torchvision ResNet-50 ImageNet1K V2, torchvision ViT-B/16 ImageNet1K V1, and OpenCLIP ViT-B-32 (`openai`).
- Freeze every backbone; train only a linear head.
- Stratified 80/20 split of the official training partition, seed 6304.
- AdamW, learning rate `1e-3`, weight decay `1e-4`, at most 50 epochs, patience 5 on validation accuracy.
- Zero-shot prompt: `a photo of a {class}.`
- Class-balanced subset of 500 official test images, seed 6304, with saved identifiers.
- Common 224x224 RGB intervention image before model-specific normalization.
- Translation displacements 0, 8, 16, and 32 pixels in four cardinal directions, using reflection padding and shifted crop.
- Patch shuffle: 4x4 pixel-space grid, one non-identity permutation per image, seed 6304.
- Mandatory metrics/evidence and the four Task 1 research questions specified in the assignment manual.
