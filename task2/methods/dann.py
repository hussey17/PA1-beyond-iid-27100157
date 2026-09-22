"""Gradient reversal and DANN domain objective."""

from __future__ import annotations

import math

import torch
from torch.autograd import Function
from torch.nn import functional as F


class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, strength: float):
        ctx.strength = float(strength)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradients: torch.Tensor):
        return -ctx.strength * gradients, None


def gradient_reverse(features: torch.Tensor, strength: float) -> torch.Tensor:
    return _GradientReversal.apply(features, strength)


def reversal_strength(
    progress: float, maximum: float = 1.0, gamma: float = 10.0
) -> float:
    if not 0.0 <= progress <= 1.0:
        raise ValueError("Training progress must lie in [0, 1]")
    return float(maximum) * (2.0 / (1.0 + math.exp(-float(gamma) * progress)) - 1.0)


def domain_classification_loss(
    discriminator,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    strength: float,
):
    features = torch.cat([source_features, target_features], dim=0)
    reversed_features = gradient_reverse(features, strength)
    logits = discriminator(reversed_features)
    labels = torch.cat(
        [
            torch.zeros(
                source_features.shape[0], dtype=torch.long, device=features.device
            ),
            torch.ones(
                target_features.shape[0], dtype=torch.long, device=features.device
            ),
        ]
    )
    return F.cross_entropy(logits, labels), logits, labels
