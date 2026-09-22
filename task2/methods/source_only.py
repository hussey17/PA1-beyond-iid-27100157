"""Source-only empirical-risk objective."""

import torch
from torch.nn import functional as F


def source_classification_loss(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    return F.cross_entropy(logits, labels)
