"""Import-safe held-out image metrics for NWM experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOImageDistance(nn.Module):
    """Per-sample DINOv2 patch-token cosine distance (lower is better)."""

    def __init__(
        self,
        device: torch.device | str,
        *,
        model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.model = (
            model
            if model is not None
            else torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vitb14",
            )
        )
        self.model = self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _features(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(device=self.device, dtype=torch.float32)
        images = F.interpolate(
            images,
            size=(224, 224),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        images = (images - self.mean.to(images)) / self.std.to(images)
        features = self.model.forward_features(images)["x_norm_patchtokens"]
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def forward(
        self,
        reference: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        if reference.ndim != 4 or prediction.ndim != 4:
            raise ValueError("DINO image distance expects [B,C,H,W] tensors")
        if reference.shape[0] != prediction.shape[0]:
            raise ValueError(
                "DINO image distance batch mismatch: "
                f"{reference.shape[0]} vs {prediction.shape[0]}"
            )
        reference_features = self._features(reference)
        prediction_features = self._features(prediction)
        return 1.0 - (
            reference_features * prediction_features
        ).sum(dim=-1).mean(dim=-1)
