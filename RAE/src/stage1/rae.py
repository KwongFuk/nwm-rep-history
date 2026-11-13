import torch
import torch.nn as nn
from .decoders import GeneralDecoder
from .encoders import ARCHS
from transformers import AutoConfig, AutoImageProcessor
from typing import Optional
from math import sqrt
from typing import Protocol

class Stage1Protocal(Protocol):
    # must have patch size attribute
    patch_size: int
    hidden_size: int 
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ...

class RAE(nn.Module):
    def __init__(self, 
        # ---- encoder configs ----
        encoder_cls: str = 'Dinov2withNorm',
        encoder_config_path: str = 'facebook/dinov2-base',
        encoder_revision: Optional[str] = None,
        encoder_input_size: int = 224,
        encoder_params: dict = {},
        # ---- decoder configs ----
        decoder_config_path: str = 'vit_mae-base',
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        # ---- noising, reshaping and normalization-----
        noise_tau: float = 0.8,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        encoder_cls = ARCHS[encoder_cls]
        self.encoder: Stage1Protocal = encoder_cls(**encoder_params)
        print(f"encoder_config_path: {encoder_config_path}")
        encoder_mean = getattr(self.encoder, "input_mean", None)
        encoder_std = getattr(self.encoder, "input_std", None)
        if encoder_mean is None or encoder_std is None:
            proc = AutoImageProcessor.from_pretrained(
                encoder_config_path,
                revision=encoder_revision,
                use_fast=False,
            )
            encoder_mean = proc.image_mean
            encoder_std = proc.image_std
        self.encoder_mean = torch.tensor(encoder_mean).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor(encoder_std).view(1, 3, 1, 1)
        encoder_config = AutoConfig.from_pretrained(
            encoder_config_path,
            revision=encoder_revision,
        )
        # see if the encoder has patch size attribute            
        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        assert self.encoder_input_size % self.encoder_patch_size == 0, f"encoder_input_size {self.encoder_input_size} must be divisible by encoder_patch_size {self.encoder_patch_size}"
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2 # number of patches of the latent
        
        # decoder
        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = self.latent_dim # set the hidden size of the decoder to be the same as the encoder's output
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches)) 
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)
        # load pretrained decoder weights
        if pretrained_decoder_path is not None:
            print(f"Loading pretrained decoder from {pretrained_decoder_path}")
            state_dict = torch.load(
                pretrained_decoder_path,
                map_location='cpu',
                weights_only=False,
            )
            if not isinstance(state_dict, dict) or not state_dict:
                raise TypeError("Decoder checkpoint must be a non-empty state dict")
            if not all(torch.is_tensor(value) for value in state_dict.values()):
                raise TypeError("Decoder checkpoint contains non-tensor state")
            try:
                self.decoder.load_state_dict(state_dict, strict=True)
            except RuntimeError as error:
                raise RuntimeError(
                    f"Decoder checkpoint is not exactly compatible: {error}"
                ) from error
        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d
        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location='cpu')
            self.latent_mean = stats.get('mean', None)
            self.latent_var = stats.get('var', None)
            expected_shape = (
                int(self.latent_dim),
                int(sqrt(self.base_patches)),
                int(sqrt(self.base_patches)),
            )
            if self.latent_mean is not None and tuple(self.latent_mean.shape) != expected_shape:
                raise ValueError(
                    f"Latent mean shape {tuple(self.latent_mean.shape)} != {expected_shape}"
                )
            if not torch.is_tensor(self.latent_var) or tuple(self.latent_var.shape) != expected_shape:
                actual = tuple(self.latent_var.shape) if torch.is_tensor(self.latent_var) else None
                raise ValueError(f"Latent variance shape {actual} != {expected_shape}")
            tensors = [self.latent_var]
            if self.latent_mean is not None:
                tensors.append(self.latent_mean)
            if any(not torch.isfinite(tensor).all() for tensor in tensors):
                raise ValueError("Latent normalization statistics contain non-finite values")
            if (self.latent_var < 0).any():
                raise ValueError("Latent variance contains negative values")
            self.do_normalization = True
            self.eps = eps
            print(f"Loaded normalization stats from {normalization_stat_path}")
        else:
            self.do_normalization = False
    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        noise = noise_sigma * torch.randn_like(x)
        return x + noise

    def _prepare_encoder_input(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = nn.functional.interpolate(x, size=(self.encoder_input_size, self.encoder_input_size), mode='bicubic', align_corners=False)
        return (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)

    def _finalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_tau > 0:
            z = self.noising(z)
        if self.reshape_to_2d:
            b, n, c = z.shape
            h = w = int(sqrt(n))
            z = z.transpose(1, 2).view(b, c, h, w)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        return z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self._prepare_encoder_input(x)
        return self._finalize_latent(self.encoder(normalized))

    def encode_with_teacher(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ordinary RAE state and final CLS from one DINO pass."""

        forward_with_cls = getattr(self.encoder, "forward_with_cls", None)
        if forward_with_cls is None:
            raise RuntimeError(
                "The configured RAE encoder does not expose forward_with_cls"
            )
        normalized = self._prepare_encoder_input(x)
        patches, final_cls = forward_with_cls(normalized)
        return self._finalize_latent(patches), final_cls

    def encode_with_multilayer_teacher(
        self,
        x: torch.Tensor,
        *,
        layer_indices: tuple[int, ...] = (2, 5, 8, 11),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode native RAE-B K1 and expose an RAEv2-style teacher map.

        The frozen encoder performs one forward: its final patches remain the
        ordinary RAE state while normalized intermediate patches are averaged
        for the spatial-relation auxiliary objective.
        """

        forward_multilayer = getattr(
            self.encoder,
            "forward_with_multilayer_teacher",
            None,
        )
        if forward_multilayer is None:
            raise RuntimeError(
                "The configured RAE encoder does not expose "
                "forward_with_multilayer_teacher"
            )
        normalized = self._prepare_encoder_input(x)
        patches, teacher_tokens = forward_multilayer(
            normalized,
            layer_indices=layer_indices,
        )
        latent = self._finalize_latent(patches)
        batch, token_count, channels = teacher_tokens.shape
        side = int(sqrt(token_count))
        if side * side != int(token_count):
            raise ValueError(
                f"RAE multi-layer teacher token count is not square: {token_count}"
            )
        teacher = teacher_tokens.transpose(1, 2).reshape(
            batch,
            channels,
            side,
            side,
        )
        return latent, teacher
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean
        if self.reshape_to_2d:
            b, c, h, w = z.shape
            n = h * w
            z = z.view(b, c, n).transpose(1, 2)
        output = self.decoder(z, drop_cls_token=False).logits
        x_rec = self.decoder.unpatchify(output)
        x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)
        return x_rec
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec
