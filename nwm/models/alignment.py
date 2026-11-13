"""REPA/iREPA heads and losses for dense RAE navigation tokens.

The iREPA choices follow End2End-Diffusion/iREPA commit
99ad4ac234efe8de52ce157120f72856e836d09f (MIT): a 3x3 convolutional
projector and z-score normalization across spatial tokens.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def spatial_zscore(
    tokens: torch.Tensor,
    *,
    alpha: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize ``[B,T,D]`` features over the spatial/token dimension."""

    if tokens.ndim != 3:
        raise ValueError(f"Expected [B,T,D] tokens, got {tuple(tokens.shape)}")
    mean = tokens.mean(dim=1, keepdim=True)
    std = tokens.std(dim=1, keepdim=True)
    return (tokens - float(alpha) * mean) / (std + float(eps))


class RepresentationAlignmentHead(nn.Module):
    """Linear REPA or convolutional iREPA projector."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        mode: str,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        mode = str(mode).lower()
        if mode not in {"repa", "irepa"}:
            raise ValueError(f"Unsupported alignment mode: {mode}")
        if kernel_size % 2 != 1:
            raise ValueError("Alignment convolution kernel must be odd")
        self.mode = mode
        if mode == "repa":
            self.projector = nn.Linear(in_dim, out_dim)
        else:
            self.projector = nn.Conv2d(
                in_dim,
                out_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )

    def forward(
        self,
        tokens: torch.Tensor,
        hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if self.mode == "repa":
            return self.projector(tokens)
        batch, count, channels = tokens.shape
        if hw is None:
            side = math.isqrt(count)
            hw = (side, side)
        if hw[0] * hw[1] != count:
            raise ValueError(f"Alignment grid {hw} does not match {count} tokens")
        feature_map = tokens.reshape(batch, hw[0], hw[1], channels)
        feature_map = feature_map.permute(0, 3, 1, 2).contiguous()
        projected = self.projector(feature_map)
        return projected.permute(0, 2, 3, 1).reshape(batch, count, -1)


def representation_alignment_loss(
    projected: torch.Tensor,
    target_latents: torch.Tensor,
    *,
    mode: str,
    spatial_alpha: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return per-sample cosine alignment loss.

    ``target_latents`` is the clean RAE target in ``[B,C,H,W]`` format.
    iREPA spatially normalizes the target; standard REPA leaves it unchanged.
    """

    target = target_latents.flatten(2).transpose(1, 2)
    if str(mode).lower() == "irepa":
        target = spatial_zscore(target, alpha=spatial_alpha, eps=eps)
    if projected.shape != target.shape:
        raise ValueError(
            f"Projected/target shape mismatch: {projected.shape} vs {target.shape}"
        )
    projected = F.normalize(projected.float(), dim=-1, eps=eps)
    target = F.normalize(target.detach().float(), dim=-1, eps=eps)
    return 1.0 - (projected * target).sum(dim=-1).mean(dim=-1)


def clean_latent_prediction_loss(
    projected: torch.Tensor,
    target_latents: torch.Tensor,
) -> torch.Tensor:
    """Return per-sample MSE for an intermediate clean-latent predictor.

    The convolutional projector predicts the tokenizer's native latent, not a
    normalized teacher representation.  This keeps the weak prediction in the
    same coordinates as the main transport model and decoder.
    """

    if projected.ndim != 3 or target_latents.ndim != 4:
        raise ValueError(
            "Clean-latent prediction expects projected [B,T,C] and target "
            f"[B,C,H,W], got {tuple(projected.shape)} and "
            f"{tuple(target_latents.shape)}"
        )
    target = target_latents.flatten(2).transpose(1, 2)
    if projected.shape != target.shape:
        raise ValueError(
            f"Projected/clean-latent shape mismatch: {projected.shape} vs "
            f"{target.shape}"
        )
    return (projected.float() - target.detach().float()).square().mean(
        dim=(1, 2)
    )


def local_spatial_relation_loss(
    projected: torch.Tensor,
    teacher_latents: torch.Tensor,
    *,
    radius: int = 2,
    spatial_alpha: float = 0.6,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Match local patch relations while allowing different channel widths.

    ``projected`` is the weak native-latent prediction in ``[B,T,Cw]`` and
    ``teacher_latents`` is a multi-level frozen feature map ``[B,Ct,H,W]``.
    The loss compares cosine relations between neighboring tokens, so PAE's
    32-channel and V-JEPA's 96-channel predictions can both learn from their
    1024-channel multi-level teachers without another large projector.
    """

    if projected.ndim != 3 or teacher_latents.ndim != 4:
        raise ValueError(
            "Spatial relation loss expects projected [B,T,C] and teacher "
            f"[B,C,H,W], got {tuple(projected.shape)} and "
            f"{tuple(teacher_latents.shape)}"
        )
    radius = int(radius)
    if radius < 1:
        raise ValueError(f"Spatial relation radius must be positive, got {radius}")
    batch, token_count, _ = projected.shape
    teacher_batch, _, height, width = teacher_latents.shape
    if batch != teacher_batch or token_count != height * width:
        raise ValueError(
            "Projected/teacher spatial mismatch: "
            f"projected={tuple(projected.shape)}, "
            f"teacher={tuple(teacher_latents.shape)}"
        )
    if radius >= height or radius >= width:
        raise ValueError(
            f"Spatial relation radius {radius} does not fit grid {height}x{width}"
        )

    weak = spatial_zscore(
        projected.float(), alpha=spatial_alpha, eps=eps
    ).reshape(batch, height, width, -1)
    teacher = spatial_zscore(
        teacher_latents.detach().float().flatten(2).transpose(1, 2),
        alpha=spatial_alpha,
        eps=eps,
    ).reshape(batch, height, width, -1)
    weak = F.normalize(weak, dim=-1, eps=eps)
    teacher = F.normalize(teacher, dim=-1, eps=eps)

    offset_losses = []
    # One direction per undirected pair avoids counting every relation twice.
    for delta_y in range(0, radius + 1):
        for delta_x in range(-radius, radius + 1):
            if delta_y == 0 and delta_x <= 0:
                continue
            if delta_y == 0:
                left = (slice(None), slice(None), slice(0, width - delta_x))
                right = (slice(None), slice(None), slice(delta_x, width))
            elif delta_x < 0:
                left = (
                    slice(None),
                    slice(0, height - delta_y),
                    slice(-delta_x, width),
                )
                right = (
                    slice(None),
                    slice(delta_y, height),
                    slice(0, width + delta_x),
                )
            else:
                left = (
                    slice(None),
                    slice(0, height - delta_y),
                    slice(0, width - delta_x),
                )
                right = (
                    slice(None),
                    slice(delta_y, height),
                    slice(delta_x, width),
                )
            weak_relation = (weak[left] * weak[right]).sum(dim=-1)
            teacher_relation = (teacher[left] * teacher[right]).sum(dim=-1)
            offset_losses.append(
                F.smooth_l1_loss(
                    weak_relation,
                    teacher_relation,
                    reduction="none",
                ).mean(dim=(1, 2))
            )
    return torch.stack(offset_losses, dim=0).mean(dim=0)
