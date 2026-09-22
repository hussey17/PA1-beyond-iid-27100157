"""The shared ImageNet-pretrained ResNet-18 feature extractor."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Backbone(nn.Module):
    feature_dim = 512

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        network = resnet18(weights=weights)
        network.fc = nn.Identity()
        self.network = network

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def freeze_batchnorm_running_statistics(module: nn.Module) -> None:
    """Freeze only BN running statistics; affine parameters remain trainable."""
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()
            if child.affine:
                child.weight.requires_grad_(True)
                child.bias.requires_grad_(True)


def set_training_mode_with_frozen_batchnorm(module: nn.Module) -> None:
    module.train()
    freeze_batchnorm_running_statistics(module)
