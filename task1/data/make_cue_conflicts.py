"""AdaIN cue-conflict generation and model-independent selection checks.

The external implementation is cloned at runtime instead of copied into this
repository.  This keeps attribution and provenance explicit.  Its released
weights are downloaded from the implementation's GitHub release.
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from task1.data.image_io import atomic_save_tensor, is_valid_image_file


ADAIN_REPOSITORY = "https://github.com/naoto0804/pytorch-AdaIN.git"
VGG_URL = (
    "https://github.com/naoto0804/pytorch-AdaIN/releases/download/"
    "v0.0.0/vgg_normalised.pth"
)
DECODER_URL = (
    "https://github.com/naoto0804/pytorch-AdaIN/releases/download/"
    "v0.0.0/decoder.pth"
)


def prepare_adain(install_root: Path, device: torch.device):
    """Clone the attributed implementation, download weights, and load models."""

    repository_dir = install_root / "pytorch-AdaIN"
    if not repository_dir.exists():
        install_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", ADAIN_REPOSITORY, str(repository_dir)],
            check=True,
        )
    models_dir = repository_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    vgg_path = models_dir / "vgg_normalised.pth"
    decoder_path = models_dir / "decoder.pth"
    if not vgg_path.exists():
        urllib.request.urlretrieve(VGG_URL, vgg_path)
    if not decoder_path.exists():
        urllib.request.urlretrieve(DECODER_URL, decoder_path)

    if str(repository_dir) not in sys.path:
        sys.path.insert(0, str(repository_dir))
    import net  # type: ignore

    decoder = net.decoder
    vgg = net.vgg
    decoder.load_state_dict(torch.load(decoder_path, map_location="cpu", weights_only=True))
    vgg.load_state_dict(torch.load(vgg_path, map_location="cpu", weights_only=True))
    vgg = torch.nn.Sequential(*list(vgg.children())[:31])
    return vgg.eval().to(device), decoder.eval().to(device), repository_dir


def _adain(content_feature: torch.Tensor, style_feature: torch.Tensor) -> torch.Tensor:
    """Match the channel-wise mean and standard deviation of style features."""

    def moments(feature: torch.Tensor, eps: float = 1e-5):
        batch, channels = feature.shape[:2]
        flattened = feature.reshape(batch, channels, -1)
        mean = flattened.mean(dim=2).reshape(batch, channels, 1, 1)
        std = (flattened.var(dim=2) + eps).sqrt().reshape(batch, channels, 1, 1)
        return mean, std

    content_mean, content_std = moments(content_feature)
    style_mean, style_std = moments(style_feature)
    normalized = (content_feature - content_mean) / content_std
    return normalized * style_std + style_mean


@torch.inference_mode()
def style_transfer_batch(
    vgg,
    decoder,
    content: torch.Tensor,
    style: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Create stylized images by interpolating content and AdaIN features."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("AdaIN alpha must be between 0 and 1")
    content_feature = vgg(content)
    style_feature = vgg(style)
    transferred = _adain(content_feature, style_feature)
    blended = alpha * transferred + (1.0 - alpha) * content_feature
    return decoder(blended).clamp(0.0, 1.0)


def build_candidate_manifest(
    clean_manifest: pd.DataFrame,
    class_pairs: Sequence[Sequence[str]],
    candidates_per_direction: int,
    seed: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Pair content and style examples deterministically in both directions."""

    rng = np.random.default_rng(seed)
    by_class = {
        class_name: group.iloc[rng.permutation(len(group))].reset_index(drop=True)
        for class_name, group in clean_manifest.groupby("class_name", sort=False)
    }
    rows = []
    for pair_index, pair in enumerate(class_pairs):
        class_a, class_b = pair
        for content_class, style_class in [(class_a, class_b), (class_b, class_a)]:
            content_rows = by_class[content_class]
            style_rows = by_class[style_class]
            if min(len(content_rows), len(style_rows)) < candidates_per_direction:
                raise ValueError("Not enough images to construct the requested candidates")
            direction_id = f"{content_class}_shape__{style_class}_texture"
            for rank in range(candidates_per_direction):
                content_row = content_rows.iloc[rank]
                # Offset the style order to avoid an arbitrary same-rank pairing pattern.
                style_row = style_rows.iloc[(rank + pair_index + 1) % len(style_rows)]
                candidate_id = f"{direction_id}__{rank:02d}"
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "pair_index": pair_index,
                        "direction": direction_id,
                        "candidate_rank": rank,
                        "content_image_id": content_row.image_id,
                        "content_path": content_row.path,
                        "content_label": int(content_row.label),
                        "content_class": content_class,
                        "style_image_id": style_row.image_id,
                        "style_path": style_row.path,
                        "style_label": int(style_row.label),
                        "style_class": style_class,
                        "path": str(output_dir / f"{candidate_id}.png"),
                        "decision": "",
                        "rejection_reason": "",
                    }
                )
    return pd.DataFrame(rows)


def generate_candidates(
    manifest: pd.DataFrame,
    vgg,
    decoder,
    device: torch.device,
    alpha: float,
    batch_size: int = 8,
) -> None:
    """Generate every cue-conflict candidate once and cache it as PNG."""

    to_tensor = transforms.ToTensor()
    for start in range(0, len(manifest), batch_size):
        batch = manifest.iloc[start : start + batch_size]
        # Existing-but-corrupt images are regenerated automatically. This is
        # important on Colab, where a runtime interruption can leave a partial
        # file whose pathname still exists.
        missing = [
            row
            for row in batch.itertuples(index=False)
            if not is_valid_image_file(row.path)
        ]
        if not missing:
            continue
        content = torch.stack(
            [to_tensor(Image.open(row.content_path).convert("RGB")) for row in missing]
        ).to(device)
        style = torch.stack(
            [to_tensor(Image.open(row.style_path).convert("RGB")) for row in missing]
        ).to(device)
        output = style_transfer_batch(vgg, decoder, content, style, alpha).cpu()
        for image_tensor, row in zip(output, missing):
            output_path = Path(row.path)
            atomic_save_tensor(image_tensor, output_path)


def select_balanced_conflicts(
    reviewed_manifest: pd.DataFrame, accepted_per_direction: int
) -> pd.DataFrame:
    """Select the first accepted candidates and enforce the balanced quota."""

    accepted = reviewed_manifest[
        reviewed_manifest["decision"].str.lower().eq("accept")
    ].copy()
    counts = accepted.groupby("direction").size()
    expected_directions = reviewed_manifest["direction"].unique().tolist()
    deficient = {
        direction: int(counts.get(direction, 0))
        for direction in expected_directions
        if counts.get(direction, 0) < accepted_per_direction
    }
    if deficient:
        raise ValueError(
            "Not enough accepted cue conflicts in these directions: " + str(deficient)
        )
    selected = (
        accepted.sort_values(["direction", "candidate_rank"])
        .groupby("direction", group_keys=False)
        .head(accepted_per_direction)
        .reset_index(drop=True)
    )
    selected["image_id"] = selected["candidate_id"]
    selected["label"] = selected["content_label"]
    selected["class_name"] = selected["content_class"]
    return selected
