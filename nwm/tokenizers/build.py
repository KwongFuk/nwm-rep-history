"""Common frozen stage-1 tokenizer construction for NWM training and inference."""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

# RAE stage-1 configs instantiate ``stage1.RAE``, which lives under RAE/src.
_RAE_SRC = str(Path(__file__).resolve().parents[2] / "RAE" / "src")
if _RAE_SRC not in sys.path:
    sys.path.append(_RAE_SRC)

from RAE.src.utils.model_utils import instantiate_from_config
from RAE.src.utils.train_utils import parse_configs
from nwm.tokenizers.external import (
    OfficialCosmosCITokenizer,
    OfficialVJEPASVAETokenizer,
)
from nwm.tokenizers.pae import PAELoadReport, load_official_pae


@dataclass(frozen=True)
class TokenizerInfo:
    tokenizer_type: str
    latent_channels: int
    source: str
    load_report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OfficialPAETokenizer(nn.Module):
    """Expose the official PAE through the RAE-compatible NWM interface."""

    def __init__(
        self,
        model: nn.Module,
        *,
        latent_channels: int,
        load_report: PAELoadReport,
    ) -> None:
        super().__init__()
        self.model = model
        self.latent_dim = int(latent_channels)
        self.load_report = load_report
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "OfficialPAETokenizer":
        super().train(False)
        self.model.eval()
        return self

    def _model_dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.model.encode(pixels.to(dtype=self._model_dtype()))

    def encode_with_multilayer_teacher(
        self,
        pixels: torch.Tensor,
        *,
        layer_indices: tuple[int, ...] = (11, 13, 15, 17, 19, 21, 23),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode PAE and return an RAEv2-style mean of normalized DINO layers.

        The official PAE DINOv2-L forward already materializes every hidden
        layer.  Reusing those tensors adds no second vision-encoder pass.  The
        default seven depths mirror RAEv2's registered K7 spacing for its
        24-block large encoder.
        """

        model = self.model
        pixels = pixels.to(dtype=self._model_dtype())
        _, _, image_height, image_width = pixels.shape
        if (
            image_height != int(model.encoder_input_size)
            or image_width != int(model.encoder_input_size)
        ):
            pixels = nn.functional.interpolate(
                pixels,
                size=(int(model.encoder_input_size), int(model.encoder_input_size)),
                mode="bicubic",
                align_corners=False,
            )
        normalized = (
            pixels - model.encoder_mean.to(pixels.device)
        ) / model.encoder_std.to(pixels.device)
        teacher_patches, teacher_aux = model.encoder(normalized)
        if (
            not isinstance(teacher_aux, (tuple, list))
            or len(teacher_aux) < 2
            or not teacher_aux[1]
        ):
            raise RuntimeError("Official PAE encoder did not expose hidden patch layers")
        hidden_layers = teacher_aux[1]
        if any(index < 0 or index >= len(hidden_layers) for index in layer_indices):
            raise ValueError(
                f"Teacher layer indices {layer_indices} exceed {len(hidden_layers)} layers"
            )
        layernorm = model.encoder.encoder.layernorm
        selected = [layernorm(hidden_layers[index]) for index in layer_indices]
        generalized_patches = torch.stack(selected, dim=0).mean(dim=0)

        aligned_patches = teacher_patches
        if bool(model.use_delta_encoder):
            aligned_patches = model.delta_encoder(normalized, aligned_patches)
        latent_tokens = model.latent_compressor[0](aligned_patches)
        batch, token_count, channels = latent_tokens.shape
        side = math.isqrt(int(token_count))
        if side * side != int(token_count):
            raise ValueError(f"PAE teacher token count is not square: {token_count}")
        latent_map = latent_tokens.transpose(1, 2).reshape(
            batch, channels, side, side
        )
        latent = model.latent_compressor[2](
            model.latent_compressor[1](latent_map)
        )
        teacher_map = generalized_patches.transpose(1, 2).reshape(
            batch,
            generalized_patches.shape[-1],
            side,
            side,
        )
        return latent, teacher_map

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.model.decode(latent.to(dtype=self._model_dtype()))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(pixels))


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _dtype_from_config(value: Any, *, default: torch.dtype) -> torch.dtype:
    if value is None:
        return default
    normalized = str(value).strip().lower().replace("torch.", "")
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported tokenizer dtype: {value!r}")
    return aliases[normalized]


def build_stage1_tokenizer(
    config: Mapping[str, Any],
    *,
    device: torch.device | str,
    project_root: str | Path,
) -> tuple[nn.Module, TokenizerInfo]:
    """Build and freeze the tokenizer selected by the resolved NWM config."""

    root = Path(project_root).resolve()
    tokenizer_config = dict(config.get("tokenizer", {}))
    tokenizer_type = str(
        tokenizer_config.get("type", config.get("tokenizer_type", "rae"))
    ).strip().lower()

    if tokenizer_type == "rae":
        config_path = _resolve_path(
            str(
                tokenizer_config.get(
                    "config_path",
                    config.get(
                        "config_path",
                        "RAE/configs/stage1/DINOv2-B.yaml",
                    ),
                )
            ),
            root,
        )
        rae_config, *_ = parse_configs(str(config_path))
        tokenizer = instantiate_from_config(rae_config).to(device).eval()
        tokenizer.requires_grad_(False)
        info = TokenizerInfo(
            tokenizer_type="rae",
            latent_channels=int(tokenizer.latent_dim),
            source=str(config_path),
        )
    elif tokenizer_type == "pae":
        required = (
            "source_root",
            "config_path",
            "checkpoint_path",
            "dino_model_path",
            "asset_manifest_path",
        )
        missing = [key for key in required if not tokenizer_config.get(key)]
        if missing:
            raise KeyError(f"PAE tokenizer config is missing: {missing}")
        dtype = _dtype_from_config(
            tokenizer_config.get("dtype"),
            default=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        source_root = _resolve_path(str(tokenizer_config["source_root"]), root)
        pae_model, load_report = load_official_pae(
            source_root=source_root,
            config_path=_resolve_path(str(tokenizer_config["config_path"]), root),
            checkpoint_path=_resolve_path(
                str(tokenizer_config["checkpoint_path"]), root
            ),
            dino_model_path=_resolve_path(
                str(tokenizer_config["dino_model_path"]), root
            ),
            asset_manifest_path=_resolve_path(
                str(tokenizer_config["asset_manifest_path"]), root
            ),
            device=device,
            dtype=dtype,
        )
        latent_channels = int(tokenizer_config.get("latent_channels", 32))
        tokenizer = OfficialPAETokenizer(
            pae_model,
            latent_channels=latent_channels,
            load_report=load_report,
        )
        info = TokenizerInfo(
            tokenizer_type="pae",
            latent_channels=latent_channels,
            source=str(source_root),
            load_report=load_report.to_dict(),
        )
    elif tokenizer_type == "cosmos_ci16x16":
        checkpoint_dir = tokenizer_config.get("checkpoint_dir")
        if not checkpoint_dir:
            raise KeyError("cosmos_ci16x16 tokenizer requires checkpoint_dir")
        dtype = _dtype_from_config(
            tokenizer_config.get("dtype"),
            default=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        checkpoint_root = _resolve_path(str(checkpoint_dir), root)
        tokenizer = OfficialCosmosCITokenizer.from_checkpoint_dir(
            checkpoint_root,
            device=device,
            dtype=dtype,
            input_size=int(tokenizer_config.get("input_size", 256)),
        )
        info = TokenizerInfo(
            tokenizer_type="cosmos_ci16x16",
            latent_channels=int(tokenizer.latent_dim),
            source=str(checkpoint_root),
            load_report={
                "release": "nvidia/Cosmos-0.1-Tokenizer-CI16x16",
                "latent_grid": [16, 16],
                "frozen": True,
            },
        )
    elif tokenizer_type == "vjepa2_1_svae96":
        required = (
            "vjepa_source_root",
            "semantic_wm_source_root",
            "vjepa_checkpoint_path",
            "adapter_checkpoint_path",
        )
        missing = [key for key in required if not tokenizer_config.get(key)]
        if missing:
            raise KeyError(f"vjepa2_1_svae96 tokenizer config is missing: {missing}")
        dtype = _dtype_from_config(
            tokenizer_config.get("dtype"),
            default=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        vjepa_source_root = _resolve_path(
            str(tokenizer_config["vjepa_source_root"]),
            root,
        )
        semantic_wm_source_root = _resolve_path(
            str(tokenizer_config["semantic_wm_source_root"]),
            root,
        )
        vjepa_checkpoint_path = _resolve_path(
            str(tokenizer_config["vjepa_checkpoint_path"]),
            root,
        )
        adapter_checkpoint_path = _resolve_path(
            str(tokenizer_config["adapter_checkpoint_path"]),
            root,
        )
        tokenizer = OfficialVJEPASVAETokenizer.from_official_assets(
            vjepa_source_root=vjepa_source_root,
            semantic_wm_source_root=semantic_wm_source_root,
            vjepa_checkpoint_path=vjepa_checkpoint_path,
            adapter_checkpoint_path=adapter_checkpoint_path,
            device=device,
            dtype=dtype,
            input_size=int(tokenizer_config.get("input_size", 256)),
            cast_encoder_to_dtype=bool(
                tokenizer_config.get("cast_encoder_to_dtype", False)
            ),
            compile_encoder=bool(tokenizer_config.get("compile_encoder", False)),
            compile_adapter=bool(tokenizer_config.get("compile_adapter", False)),
            load_pixel_decoder=bool(
                tokenizer_config.get("load_pixel_decoder", True)
            ),
        )
        info = TokenizerInfo(
            tokenizer_type="vjepa2_1_svae96",
            latent_channels=int(tokenizer.latent_dim),
            source=str(adapter_checkpoint_path),
            load_report={
                "encoder": "facebookresearch/vjepa2 ViT-L/16 2.1",
                "adapter_decoder": "Nilaksh404/semantic-wm adapter_vjepa_image_96.pt",
                "latent_grid": [16, 16],
                "frozen": True,
                "cast_encoder_to_dtype": bool(
                    tokenizer_config.get("cast_encoder_to_dtype", False)
                ),
                "compile_encoder": bool(
                    tokenizer_config.get("compile_encoder", False)
                ),
                "compile_adapter": bool(
                    tokenizer_config.get("compile_adapter", False)
                ),
                "pixel_decoder_loaded": bool(
                    tokenizer_config.get("load_pixel_decoder", True)
                ),
            },
        )
    else:
        raise ValueError(
            f"Unsupported tokenizer type {tokenizer_type!r}; expected "
            "'rae', 'pae', 'cosmos_ci16x16', or 'vjepa2_1_svae96'"
        )

    expected_channels = config.get("model_in_channels")
    if expected_channels is not None and int(expected_channels) != int(
        tokenizer.latent_dim
    ):
        raise ValueError(
            "Tokenizer/model channel mismatch: "
            f"tokenizer={tokenizer.latent_dim}, model_in_channels={expected_channels}"
        )
    return tokenizer, info
