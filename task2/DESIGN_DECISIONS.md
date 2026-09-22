# Task 2 Design Decision Register

Status: **approved by the repository owner on 22 September 2026**. These choices are frozen before target-label evaluation.

## Controlled study

- Study DAN alignment strength with `lambda_MMD` in `{0.1, 1, 10}`.
- Retain `lambda_MMD = 1` for the required main-method comparison regardless of final target results.
- Reuse the main DAN run as the middle controlled-study point.
- Expected behavior: increasing alignment should generally reduce source-target domain separability. Source validation performance may decline at excessive strength. Target recognition is expected to be non-monotonic: moderate alignment may help, while excessive marginal alignment may mix semantic classes and cause negative transfer.

## Operational choices not numerically fixed by the manual

- An epoch contains `max(len(source_domain_loader))` updates. Shorter source loaders and the target loader are cycled so every update still contains 8 examples from each source domain and 24 target examples.
- MMD uses the biased, nonnegative squared empirical mean-embedding estimator. This matches the displayed squared RKHS-distance objective and avoids treating a negative unbiased estimate as an optimization target.
- Kernel bandwidth uses the detached median of positive off-diagonal squared distances in the current combined source-target batch. Each RBF is `exp(-distance_squared / bandwidth)` and the three kernels are summed.
- DANN/CDAN progress is optimizer-step progress over the planned 30-epoch budget. Varying gradient reversal would scale only the reversed feature gradient, not the discriminator's own gradient; the required comparison uses maximum strength 1.
- Validation improvement is strictly greater mean source-domain macro-F1; ties keep the earlier checkpoint.
- Target images receive the same stochastic training transform as source images during adaptation and the deterministic evaluation transform for diagnostics.
- The source-only, DAN, DANN, and CDAN feature extractor and classifier initialization is identical under seed 6304. Method-specific discriminator initialization occurs only after those shared parameters are constructed.
- Target final evaluation is guarded by a lock manifest containing configuration and checkpoint SHA-256 hashes. Changing either after locking invalidates the final-evaluation stage.

## Manual-fixed choices

- PACS sources: Photo, Art Painting, Cartoon; target: Sketch.
- Independently stratified 80/20 split in each source domain, seed 6304, shared with Task 3.
- Complete Sketch domain is unlabeled adaptation data and transductive evaluation data.
- ImageNet-pretrained torchvision ResNet-18 V1, seven-class linear head, full-network fine-tuning.
- Resize to 256x256; random 224x224 crop and horizontal flip for training; center crop for validation/evaluation; pretrained normalization.
- Freeze BatchNorm running mean/variance at ImageNet values while keeping affine scale and bias trainable.
- AdamW, learning rate `1e-4`, weight decay `1e-4`, at most 30 source epochs, patience 5 on mean source-validation macro-F1.
- Source-balanced batches of 8 per source domain and 24 target examples per adaptation update.
- DAN acts on the 512-dimensional pre-classifier feature with the three required median-relative RBF kernels.
- DANN/CDAN use the required 256-ReLU-dropout(0.5)-2 discriminator and logistic reversal schedule. CDAN uses the undetached feature-probability outer product and no entropy conditioning.
- Target class labels are used only after every configuration and checkpoint has been fixed.
