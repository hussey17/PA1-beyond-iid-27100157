# PA1: Beyond IID (27100157)

Research code for Programming Assignment 1 in Advanced Topics in Machine Learning (Fall 2026). The assignment studies inductive biases, unsupervised domain adaptation, domain generalization, and open-set recognition.

## Status

- Repository scaffold: complete
- Task 1 design decisions: approved and frozen for the first run
- Task 1 implementation: Colab notebook prepared; execution pending
- Tasks 2-4: not started

## Reproducibility rules

- Use seed `6304` wherever the manual specifies it.
- Record every reported hyperparameter in configuration files.
- Keep generated interventions separate from model evaluation so every model receives identical inputs.
- Commit split indices and compact CSV/JSON results, but not raw datasets or unnecessary checkpoints.
- Treat target/unknown labels as evaluation-only wherever required by the assignment protocol.
- Attribute materially reused external code and pretrained assets here before submission.

## Repository layout

```text
common/                 Shared utilities only after genuine reuse is established
shared/                 PACS data protocol shared by Tasks 2 and 3
task1/                  Inductive biases and feature representations
task2/                  Unsupervised domain adaptation
task3/                  Domain generalization
task4/                  Open-set recognition
report/figures/         Final figures selected by the repository owner
```

Each task keeps notebooks separate from reusable Python modules. Notebooks document and orchestrate experiments; modules implement data preparation, models, transformations, metrics, and evaluation.

## Assignment ownership and AI-use boundary

Coding assistance is used to generate and explain implementation code. The repository owner is responsible for understanding every submitted line, approving experimental design choices, running or validating experiments, interpreting results, and writing the report. The workflow may organize required evidence and check completeness, but it does not generate report prose.

## Data and environment

Dataset downloads, environment setup, exact commands, and external-code attributions will be added with each task. Raw datasets and large model checkpoints are intentionally excluded from version control.
