# Task 1 Design Decision Register

Status: **approved by the repository owner on 21 September 2026**. The choices below are now frozen for the first complete Task 1 run. Any later change must be recorded before rerunning the affected experiment.

## A. Required experimental-design decisions

### 1. Dataset - approved

- **STL-10**, selected to keep the Colab computation and cue-conflict review manageable.
- The official test subset contains exactly 50 examples per class (500 total), selected with seed 6304. The saved subset manifest is reused for every model and intervention.

### 2. Additional color intervention - approved

- **Fixed +90 degree hue rotation**, implemented as a `+0.25` hue-factor rotation in torchvision.
- This quarter-turn changes hue while retaining saturation and value in HSV space, apart from small numerical effects introduced by RGB-HSV conversion.
- Grayscale remains the mandatory common color-removal intervention.
- Primary metrics: accuracy change and prediction consistency relative to the paired clean image.

### 3. Cue-conflict construction - approved

- Method: AdaIN using the public `naoto0804/pytorch-AdaIN` implementation and its released encoder/decoder weights.
- Style strength: **alpha = 0.8**. In AdaIN, alpha linearly interpolates between the original content feature and the fully normalized content-style feature. A value of 0.8 is a pre-registered compromise: it supplies strong style evidence while retaining 20% of the original content feature to reduce destroyed or unrecognizable shapes. It is not a theoretically optimal value and will not be tuned using model predictions.
- Unordered pairs, following the published STL-10 class order and using every class once:
  - airplane <-> bird
  - car <-> cat
  - deer <-> dog
  - horse <-> monkey
  - ship <-> truck
- Generate both directions. Initially generate 30 candidates per direction (300 total), then retain the first 20 visually valid candidates per direction. This yields 200 balanced conflicts. If a direction has fewer than 20 valid candidates, generate from its remaining deterministic pairs until the quota is met.
- Reject only when the content object is no longer human-identifiable, the output is visibly corrupted, or the style effect is visually imperceptible. Record one reason for every rejection.
- Review occurs before prediction generation. Model outputs cannot be consulted while deciding retention.
- Report raw shape, texture, and other counts together with shape bias and coverage.

### 4. Representation visualization - approved

- **UMAP** with `n_neighbors=15`, `min_dist=0.1`, cosine distance, and seed 6304.
- Fixed class-balanced subset: 20 clean test examples per class.
- Translation visualization uses displacement 32 with the four directions assigned round-robin across the fixed visualization subset, so all directions are equally represented without duplicating clean points. Quantitative stability is computed for every nonzero displacement and direction.
- One projection is fitted to combined clean and transformed features for each backbone/intervention combination. Absolute coordinates are not compared between separately fitted projections.
- Figure organization: one row per backbone and one column per intervention. Cosine stability is the primary quantitative representation measure.

### 5. Pre-registered hypotheses - approved

These are hypotheses to test, not conclusions or assumptions about architecture.

1. **Color:** Removing chromatic information or rotating hue by 90 degrees will not substantially degrade classification relative to each predictor's clean baseline, suggesting that the learned decision functions are driven primarily by non-chromatic evidence. The notebook operationalizes "not substantially" as an absolute accuracy decrease of no more than five percentage points and reports the actual continuous change and prediction consistency.
2. **Shape versus texture:** Among cue-conflict predictions that select either intended label, the content/shape label will be chosen more often than the style/texture label (shape bias above 50%). Coverage and the number of "other" predictions will be interpreted jointly: a high shape-bias estimate with low coverage is weak evidence. Stylization may still cause a large representation shift or increase "other" predictions even if the content label dominates among covered decisions.
3. **Translation:** Accuracy and prediction consistency will decrease as displacement increases, indicating positional sensitivity introduced by finite image boundaries, stride/pooling, learned positional information, or the pretraining distribution.
4. **Patch structure:** Patch shuffling will cause a non-catastrophic decline because local appearance and short-range geometric evidence remain available even though global organization is disrupted. Performance is expected to remain above the 10% chance level; the magnitude relative to translation is treated as an empirical result rather than assumed in advance.
5. **Prediction versus representation:** Stronger interventions will generally reduce cosine representation stability, but prediction and representation stability will not correspond perfectly. Some examples may retain their predicted class despite substantial feature movement, while other small feature changes may cross a linear decision boundary.
6. **CLIP adaptation:** The trained CLIP linear head will outperform zero-shot CLIP on STL-10 accuracy and macro-F1 because the supervised head adapts the fixed image representation to the dataset's class boundaries. A class-stratified paired bootstrap interval for the accuracy difference will distinguish a reliable improvement from sampling variation.
7. **Architecture comparison:** No fixed ResNet-versus-ViT-versus-CLIP ordering is pre-registered. Architecture, pretraining data, supervision, augmentation, and capacity are confounded, so any observed ranking will be interpreted with that limitation.

Use 1,000 class-stratified bootstrap resamples for the principal paired accuracy comparisons without changing the manual's single-seed training protocol.

## B. Approved operational choices not fixed by the manual

These affect reproducibility and should be approved once, then held constant:

- Platform: Google Colab with an NVIDIA T4-class GPU. The notebook verifies CUDA availability before expensive steps.
- Linear-head batch size: 128. Frozen-feature extraction batch size: 64, chosen conservatively for ViT-B/16 on a 16 GB T4.
- Common preprocessing: convert to RGB, resize the shorter edge to 256, and center crop to 224 before model-specific normalization.
- No stochastic augmentation for frozen feature extraction or linear-head training.
- Linear-head loss: multiclass cross-entropy; no learning-rate scheduler.
- Confidence intervals: 1,000 class-stratified bootstrap resamples where implemented.
- Visual QA: an in-notebook review widget writes accepted/rejected decisions and reasons to a CSV manifest. The repository owner performs the review.
- External source: `naoto0804/pytorch-AdaIN`, MIT licensed; the repository and original Huang and Belongie paper must be cited in code attribution.

## C. Delegated implementation defaults

Unless the owner requests otherwise, these can be implemented deterministically and documented:

- Store split/subset identifiers and cue-conflict manifests as JSON/CSV.
- Cache frozen features with labels and image identifiers, encode the model/condition in each cache filename, and invalidate a cache when its identifier sequence changes.
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
