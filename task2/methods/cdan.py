"""Class-conditional adversarial input and loss for CDAN."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from task2.methods.dann import gradient_reverse


def conditional_outer_product(
    features: torch.Tensor, class_logits: torch.Tensor
) -> torch.Tensor:
    probabilities = F.softmax(class_logits, dim=1)
    joint = torch.bmm(features.unsqueeze(2), probabilities.unsqueeze(1))
    return joint.flatten(start_dim=1)


def conditional_domain_loss(
    discriminator,
    source_features: torch.Tensor,
    source_logits: torch.Tensor,
    target_features: torch.Tensor,
    target_logits: torch.Tensor,
    strength: float,
):
    features = torch.cat([source_features, target_features], dim=0)
    logits = torch.cat([source_logits, target_logits], dim=0)
    conditional = conditional_outer_product(features, logits)
    domain_logits = discriminator(gradient_reverse(conditional, strength))
    domain_labels = torch.cat(
        [
            torch.zeros(
                source_features.shape[0], dtype=torch.long, device=features.device
            ),
            torch.ones(
                target_features.shape[0], dtype=torch.long, device=features.device
            ),
        ]
    )
    return F.cross_entropy(domain_logits, domain_labels), domain_logits, domain_labels
