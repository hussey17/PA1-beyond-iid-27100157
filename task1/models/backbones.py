"""Frozen ResNet-50, ViT-B/16, and OpenCLIP image encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import (
    ResNet50_Weights,
    ViT_B_16_Weights,
    resnet50,
    vit_b_16,
)
from tqdm.auto import tqdm


@dataclass
class BackboneBundle:
    name: str
    model: nn.Module
    feature_dim: int
    tensor_transform: transforms.Compose
    device: torch.device
    clip_tokenizer: Optional[object] = None
    clip_logit_scale: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return the manual-specified final representation."""

        images = images.to(self.device, non_blocking=True)
        use_amp = self.device.type == "cuda"
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if use_amp else torch.float32,
            enabled=use_amp,
        ):
            if self.name == "clip_vit_b32":
                features = self.model.encode_image(images, normalize=True)
            else:
                features = self.model(images)
        return features.float()


def _tensor_transform(mean, std):
    # Geometry has already been standardized to 224x224. Applying only tensor
    # conversion and normalization prevents model-specific crops from changing
    # the controlled intervention.
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def load_backbone(name: str, device: torch.device) -> BackboneBundle:
    """Load one frozen pretrained backbone with its required normalization."""

    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        preset = weights.transforms()
        tensor_transform = _tensor_transform(preset.mean, preset.std)
        tokenizer = None
        logit_scale = None
    elif name == "vit_b16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        feature_dim = model.heads.head.in_features
        model.heads = nn.Identity()
        preset = weights.transforms()
        tensor_transform = _tensor_transform(preset.mean, preset.std)
        tokenizer = None
        logit_scale = None
    elif name == "clip_vit_b32":
        import open_clip
        from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )
        feature_dim = int(model.visual.output_dim)
        tensor_transform = _tensor_transform(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        logit_scale = model.logit_scale.detach().exp().float().cpu()
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return BackboneBundle(
        name=name,
        model=model,
        feature_dim=feature_dim,
        tensor_transform=tensor_transform,
        device=device,
        clip_tokenizer=tokenizer,
        clip_logit_scale=logit_scale,
    )


@torch.inference_mode()
def extract_features(bundle: BackboneBundle, dataset, batch_size: int, workers: int = 2):
    """Extract features, labels, and stable identifiers in dataset order."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=bundle.device.type == "cuda",
    )
    feature_batches, label_batches, identifiers = [], [], []
    for images, labels, batch_ids in tqdm(loader, desc=f"Features: {bundle.name}"):
        feature_batches.append(bundle.encode(images).cpu())
        label_batches.append(torch.as_tensor(labels).long().cpu())
        if isinstance(batch_ids, torch.Tensor):
            identifiers.extend(batch_ids.cpu().tolist())
        else:
            identifiers.extend(list(batch_ids))
    return (
        torch.cat(feature_batches).numpy().astype(np.float32),
        torch.cat(label_batches).numpy().astype(np.int64),
        identifiers,
    )


@torch.inference_mode()
def clip_text_classifier(bundle: BackboneBundle, class_names: list[str]):
    """Encode the fixed zero-shot prompts and return text features and scale."""

    if bundle.name != "clip_vit_b32":
        raise ValueError("Zero-shot text classification is defined only for CLIP")
    prompts = [f"a photo of a {class_name}." for class_name in class_names]
    tokens = bundle.clip_tokenizer(prompts).to(bundle.device)
    text_features = bundle.model.encode_text(tokens, normalize=True).float().cpu().numpy()
    return text_features, float(bundle.clip_logit_scale.item())
