from transformers import Dinov2WithRegistersModel
from torch import nn
import torch
from math import *
from typing import Optional, Sequence
from . import register_encoder


@register_encoder()
class Dinov2withNorm(nn.Module):
    def __init__(
        self,
        dinov2_path: str,
        normalize: bool = True,
        layer_indices: Optional[Sequence[int]] = None,
        layer_aggregation: str = "last",
        revision: Optional[str] = None,
    ):
        super().__init__()
        # Support both local paths and HuggingFace model IDs
        try:
            self.encoder = Dinov2WithRegistersModel.from_pretrained(
                dinov2_path,
                revision=revision,
                local_files_only=True,
            )
        except (OSError, ValueError, AttributeError):
            self.encoder = Dinov2WithRegistersModel.from_pretrained(
                dinov2_path,
                revision=revision,
                local_files_only=False,
            )
        self.encoder.requires_grad_(False)
        if normalize:
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size
        self.layer_indices = (
            None if layer_indices is None else tuple(int(i) for i in layer_indices)
        )
        self.layer_aggregation = str(layer_aggregation).lower()
        if self.layer_aggregation not in {"last", "mean"}:
            raise ValueError(
                f"Unsupported layer_aggregation={self.layer_aggregation!r}"
            )
        if self.layer_aggregation == "mean" and not self.layer_indices:
            raise ValueError("mean aggregation requires non-empty layer_indices")
        num_layers = int(self.encoder.config.num_hidden_layers)
        if self.layer_indices and any(i < 0 or i >= num_layers for i in self.layer_indices):
            raise ValueError(
                f"layer_indices={self.layer_indices} outside [0,{num_layers - 1}]"
            )
        
    def dinov2_forward_with_cls(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return configured patch features and the final frozen CLS token."""

        outputs = self.encoder(x, output_hidden_states=True)
        unused_token_num = 5  # 1 CLS + 4 register tokens
        if self.layer_aggregation == "last":
            patches = outputs.last_hidden_state[:, unused_token_num:]
        else:
            # Hugging Face hidden_states[0] is the embedding output, so block i
            # is hidden_states[i + 1]. RAEv2 normalizes every selected
            # intermediate layer before aggregating it.
            selected = []
            for layer_index in self.layer_indices:
                hidden = outputs.hidden_states[layer_index + 1]
                hidden = self.encoder.layernorm(hidden)
                selected.append(hidden[:, unused_token_num:])
            patches = torch.stack(selected, dim=0).mean(dim=0)
        final_cls = outputs.last_hidden_state[:, 0]
        return patches, final_cls

    def forward_with_multilayer_teacher(
        self,
        x: torch.Tensor,
        *,
        layer_indices: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ordinary final-layer state and a normalized layer mean.

        Both tensors come from one frozen DINO forward.  Keeping the ordinary
        final-layer patches as the RAE state makes the multi-layer tensor a
        training-only RAEv2 target instead of silently changing the tokenizer.
        """

        selected_indices = tuple(int(index) for index in layer_indices)
        if not selected_indices:
            raise ValueError("layer_indices must be non-empty")
        num_layers = int(self.encoder.config.num_hidden_layers)
        if any(index < 0 or index >= num_layers for index in selected_indices):
            raise ValueError(
                f"layer_indices={selected_indices} outside [0,{num_layers - 1}]"
            )

        outputs = self.encoder(x, output_hidden_states=True)
        unused_token_num = 5  # 1 CLS + 4 register tokens
        final_patches = outputs.last_hidden_state[:, unused_token_num:]
        selected = []
        for layer_index in selected_indices:
            hidden = outputs.hidden_states[layer_index + 1]
            hidden = self.encoder.layernorm(hidden)
            selected.append(hidden[:, unused_token_num:])
        teacher = torch.stack(selected, dim=0).mean(dim=0)
        return final_patches, teacher

    def dinov2_forward(self, x: torch.Tensor) -> torch.Tensor:
        patches, _ = self.dinov2_forward_with_cls(x)
        return patches

    def forward_with_cls(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dinov2_forward_with_cls(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dinov2_forward(x)
