# Task 1 - Inductive Biases and Feature Representations

The experimental choices are frozen in `DESIGN_DECISIONS.md`. The Colab-ready notebook is `notebooks/task1_inductive_biases.ipynb`; its paired percent-format source is retained for readable version-control diffs.

The implemented layout follows the assignment boundary between intervention generation and model evaluation:

```text
configs/
data/
models/
analysis/
scripts/
notebooks/
results/
```

## Running on Colab

1. Upload or open `notebooks/task1_inductive_biases.ipynb` in Colab.
2. Select a T4 GPU runtime before executing the first cell.
3. Run cells in order. The notebook clones this repository and installs only dependencies not already supplied by Colab.
4. Complete the model-independent visual review of every AdaIN candidate before running model evaluation.
5. Download the final `task1_results_seed6304.zip` archive and share the executed notebook for result interpretation.

## Interrupted image writes

Generated PNGs are validated before reuse and written through an atomic temporary-file replacement. If Colab is interrupted during generation, rerunning the relevant generation cell automatically repairs missing, zero-byte, truncated, or incorrectly sized files. A manifest audit reports any remaining invalid path before feature extraction begins.

## External attribution

Cue conflicts use the MIT-licensed [`naoto0804/pytorch-AdaIN`](https://github.com/naoto0804/pytorch-AdaIN) implementation and its released pretrained weights. The method originates from Huang and Belongie, *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization* (ICCV 2017). External code is cloned at runtime and is not copied into this repository.
