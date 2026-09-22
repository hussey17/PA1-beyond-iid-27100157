"""DAN-style multi-kernel Maximum Mean Discrepancy."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def pairwise_squared_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_norm = left.square().sum(dim=1, keepdim=True)
    right_norm = right.square().sum(dim=1, keepdim=True).transpose(0, 1)
    return (left_norm + right_norm - 2.0 * left @ right.transpose(0, 1)).clamp_min(0.0)


def median_squared_bandwidth(
    features: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """Detached median of positive, off-diagonal combined-batch distances."""
    distances = pairwise_squared_distance(features, features).detach()
    mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)
    candidates = distances[mask]
    positive = candidates[candidates > 0]
    if positive.numel() == 0:
        return distances.new_tensor(epsilon)
    return positive.median().clamp_min(epsilon)


def multi_rbf_kernel(
    left: torch.Tensor,
    right: torch.Tensor,
    base_bandwidth: torch.Tensor,
    multipliers: Sequence[float] = (0.5, 1.0, 2.0),
) -> torch.Tensor:
    distances = pairwise_squared_distance(left, right)
    kernels = [
        torch.exp(-distances / (base_bandwidth * float(scale))) for scale in multipliers
    ]
    return torch.stack(kernels, dim=0).sum(dim=0)


def mmd_squared(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    multipliers: Sequence[float] = (0.5, 1.0, 2.0),
) -> torch.Tensor:
    """Biased nonnegative estimate of squared RKHS mean distance."""
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("MMD expects two [batch, feature] tensors")
    combined = torch.cat([source_features, target_features], dim=0)
    bandwidth = median_squared_bandwidth(combined)
    kernel_ss = multi_rbf_kernel(
        source_features, source_features, bandwidth, multipliers
    )
    kernel_tt = multi_rbf_kernel(
        target_features, target_features, bandwidth, multipliers
    )
    kernel_st = multi_rbf_kernel(
        source_features, target_features, bandwidth, multipliers
    )
    return (kernel_ss.mean() + kernel_tt.mean() - 2.0 * kernel_st.mean()).clamp_min(0.0)
