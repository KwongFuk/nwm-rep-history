"""Frozen official tokenizer adapters for the representation comparison.

These wrappers only load released assets.  They never train or update the
tokenizer, S-VAE adapter, or pixel decoder.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def _require_file(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty {label}: {resolved}")
    return resolved


def _require_dir(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def _strip_compiled_prefix(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for raw_name, value in state.items():
        name = str(raw_name)
        while name.startswith("_orig_mod."):
            name = name.removeprefix("_orig_mod.")
        cleaned[name] = value
    return cleaned


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_semantic_wm_release(source_root: Path):
    """Load only the released S-VAE/decoder modules under an isolated alias."""

    package_name = "_frozen_semantic_wm_release"
    models_name = f"{package_name}.models"
    encoders_name = f"{models_name}.encoders"
    src_root = source_root / "src"
    required = (
        src_root / "models" / "base_autoencoder.py",
        src_root / "models" / "encoders" / "rae.py",
        src_root / "models" / "adapters.py",
        src_root / "models" / "pixel_decoder.py",
    )
    for path in required:
        _require_file(path, label="semantic-wm source file")

    if f"{models_name}.adapters" in sys.modules:
        return (
            sys.modules[f"{models_name}.adapters"],
            sys.modules[f"{models_name}.pixel_decoder"],
        )

    for name, package_path in (
        (package_name, src_root),
        (models_name, src_root / "models"),
        (encoders_name, src_root / "models" / "encoders"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(package_path)]
        sys.modules[name] = package

    _load_module(
        f"{models_name}.base_autoencoder",
        src_root / "models" / "base_autoencoder.py",
    )
    _load_module(
        f"{encoders_name}.rae",
        src_root / "models" / "encoders" / "rae.py",
    )
    adapters = _load_module(
        f"{models_name}.adapters",
        src_root / "models" / "adapters.py",
    )
    pixel_decoder = _load_module(
        f"{models_name}.pixel_decoder",
        src_root / "models" / "pixel_decoder.py",
    )
    return adapters, pixel_decoder


def _module_belongs_to_source(module: Any, source_root: Path) -> bool:
    """Return whether a loaded module was resolved from ``source_root``."""

    candidates: list[str] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        candidates.append(str(module_file))
    module_path = getattr(module, "__path__", None)
    if module_path:
        candidates.extend(str(path) for path in module_path)

    root = source_root.resolve()
    for candidate in candidates:
        try:
            Path(candidate).resolve().relative_to(root)
        except (OSError, ValueError):
            continue
        return True
    return False


def _drop_module_tree(package_name: str) -> None:
    for module_name in tuple(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)


def _install_source_package(package_name: str, package_path: Path) -> None:
    """Register an explicit package for an upstream namespace-only directory."""

    package = types.ModuleType(package_name)
    package.__file__ = str(package_path / "__init__.py")
    package.__package__ = package_name
    package.__path__ = [str(package_path)]
    spec = importlib.util.spec_from_loader(package_name, loader=None, is_package=True)
    if spec is not None:
        spec.submodule_search_locations = [str(package_path)]
    package.__spec__ = spec
    sys.modules[package_name] = package


def _import_vjepa2_1_vision_transformer(source_root: Path):
    """Import V-JEPA's legacy ``app``/``src`` packages from one pinned tree.

    Both V-JEPA and semantic-wm expose a top-level package named ``src``.
    Python otherwise reuses whichever one was imported first, which can route
    V-JEPA's ``src.masks`` import into semantic-wm.  Re-resolve both legacy
    namespaces from the pinned V-JEPA source and verify every critical module
    before constructing the encoder.
    """

    source_root = source_root.resolve()
    source_text = str(source_root)
    sys.path[:] = [entry for entry in sys.path if entry != source_text]
    sys.path.insert(0, source_text)

    critical_names = (
        "app.vjepa_2_1.models.vision_transformer",
        "src.masks.utils",
        "src.utils.tensors",
    )
    if all(
        name in sys.modules
        and _module_belongs_to_source(sys.modules[name], source_root)
        for name in critical_names
    ):
        return sys.modules["app.vjepa_2_1.models.vision_transformer"]

    # The upstream tree deliberately uses the generic top-level names ``app``
    # and ``src``.  Rebuild the complete package chain as one coherent source
    # tree instead of mixing cached parents and children from other projects.
    _drop_module_tree("app")
    _drop_module_tree("src")
    for package_name, package_path in (
        ("app", source_root / "app"),
        ("app.vjepa_2_1", source_root / "app" / "vjepa_2_1"),
        (
            "app.vjepa_2_1.models",
            source_root / "app" / "vjepa_2_1" / "models",
        ),
        (
            "app.vjepa_2_1.models.utils",
            source_root / "app" / "vjepa_2_1" / "models" / "utils",
        ),
        ("src", source_root / "src"),
        ("src.masks", source_root / "src" / "masks"),
        ("src.utils", source_root / "src" / "utils"),
    ):
        _install_source_package(package_name, package_path)

    importlib.invalidate_caches()
    modules = {
        "app.vjepa_2_1.models.vision_transformer": importlib.import_module(
            "app.vjepa_2_1.models.vision_transformer"
        ),
        "src.masks.utils": importlib.import_module("src.masks.utils"),
        "src.utils.tensors": importlib.import_module("src.utils.tensors"),
    }
    expected = {
        "app.vjepa_2_1.models.vision_transformer": (
            source_root / "app" / "vjepa_2_1" / "models" / "vision_transformer.py"
        ),
        "src.masks.utils": source_root / "src" / "masks" / "utils.py",
        "src.utils.tensors": source_root / "src" / "utils" / "tensors.py",
    }
    for module_name, module in modules.items():
        observed = Path(str(getattr(module, "__file__", ""))).resolve()
        required = expected[module_name].resolve()
        if observed != required:
            raise ImportError(
                f"V-JEPA module source mismatch for {module_name}: "
                f"observed={observed}, expected={required}"
            )
    return modules["app.vjepa_2_1.models.vision_transformer"]


def _load_vjepa2_1_encoder(
    *,
    source_root: Path,
    checkpoint_path: Path,
    device: torch.device | str,
    dtype: torch.dtype,
    cast_weights_to_dtype: bool = False,
    compile_encoder: bool = False,
) -> nn.Module:
    required = (
        source_root / "app" / "vjepa_2_1" / "models" / "vision_transformer.py",
        source_root / "src" / "masks" / "utils.py",
        source_root / "src" / "utils" / "tensors.py",
    )
    for path in required:
        _require_file(path, label="V-JEPA 2.1 source file")
    vit_module = _import_vjepa2_1_vision_transformer(source_root)
    if cast_weights_to_dtype:
        modules = importlib.import_module(
            "app.vjepa_2_1.models.utils.modules"
        )
        modules.rotate_queries_or_keys = _rotate_queries_or_keys_dtype_stable
    encoder = vit_module.vit_large(
        patch_size=16,
        img_size=(384, 384),
        num_frames=64,
        tubelet_size=2,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "ema_encoder" not in checkpoint or not isinstance(
        checkpoint["ema_encoder"], Mapping
    ):
        raise ValueError("V-JEPA 2.1 checkpoint is missing ema_encoder")
    cleaned = {
        str(name).replace("module.", "").replace("backbone.", ""): value
        for name, value in checkpoint["ema_encoder"].items()
    }
    encoder.load_state_dict(cleaned, strict=True)
    del checkpoint, cleaned
    # The released implementation normally keeps fp32 weights because RoPE
    # promotes q/k to fp32 while v remains under autocast.  The optional
    # dtype-stable RoPE fast path keeps all three tensors in ``dtype`` and
    # therefore permits persistent bf16 weights without an SDPA mismatch.
    encoder_dtype = dtype if cast_weights_to_dtype else torch.float32
    encoder = encoder.to(device=device, dtype=encoder_dtype).eval()
    encoder.requires_grad_(False)
    if compile_encoder:
        if not torch.cuda.is_available():
            raise RuntimeError("V-JEPA encoder compilation requires CUDA")
        encoder = torch.compile(
            encoder,
            dynamic=False,
            fullgraph=False,
            mode="reduce-overhead",
        )
    return encoder


def _rotate_queries_or_keys_dtype_stable(
    x: torch.Tensor,
    pos: torch.Tensor,
    n_registers: int,
    has_cls_first: bool,
) -> torch.Tensor:
    """V-JEPA RoPE with q/k output kept in the input tensor dtype.

    The upstream implementation multiplies bf16 frequencies by fp32 position
    tensors, silently promoting rotated q/k to fp32 while v remains bf16.
    Besides breaking persistent-bf16 encoder weights, that promotion adds many
    small fp32 kernels in every ViT block.  Casting positions once preserves
    the same RoPE formula and makes the SDPA dtype contract explicit.
    """

    _, _, token_count, head_dim = x.shape
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    class_tokens = 1 if has_cls_first else 0
    context_end = token_count - int(n_registers)
    x_cls = x[..., :class_tokens, :] if class_tokens else None
    x_ctx = x[..., class_tokens:context_end, :]
    x_reg = x[..., context_end:, :] if n_registers else None

    positions = pos.to(device=x.device, dtype=x.dtype)
    omega = torch.arange(
        head_dim // 2,
        dtype=x.dtype,
        device=x.device,
    )
    omega = 1.0 / (10000 ** (omega / (head_dim / 2.0)))
    frequencies = torch.einsum("..., f -> ... f", positions, omega)
    sin = frequencies.sin().repeat_interleave(2, dim=-1)
    cos = frequencies.cos().repeat_interleave(2, dim=-1)

    pairs = x_ctx.unflatten(-1, (-1, 2))
    first, second = pairs.unbind(dim=-1)
    rotated = torch.stack((-second, first), dim=-1).flatten(-2)
    out_ctx = x_ctx * cos + rotated * sin
    parts = ([x_cls] if class_tokens else []) + [out_ctx]
    if n_registers:
        parts.append(x_reg)
    return torch.cat(parts, dim=-2)


class OfficialCosmosCITokenizer(nn.Module):
    """NVIDIA Cosmos-0.1 CI16x16 as a frozen BCHW tokenizer."""

    latent_dim = 16

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        *,
        input_size: int = 256,
        source: str = "injected",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_size = int(input_size)
        self.source = str(source)
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_checkpoint_dir(
        cls,
        checkpoint_dir: str | Path,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        input_size: int = 256,
    ) -> "OfficialCosmosCITokenizer":
        root = _require_dir(checkpoint_dir, label="Cosmos CI16x16 checkpoint directory")
        encoder_path = _require_file(root / "encoder.jit", label="Cosmos encoder")
        decoder_path = _require_file(root / "decoder.jit", label="Cosmos decoder")
        encoder = torch.jit.load(str(encoder_path), map_location="cpu")
        decoder = torch.jit.load(str(decoder_path), map_location="cpu")
        encoder = encoder.to(device=device, dtype=dtype).eval()
        decoder = decoder.to(device=device, dtype=dtype).eval()
        return cls(
            encoder,
            decoder,
            input_size=input_size,
            source=str(root),
        )

    def train(self, mode: bool = True) -> "OfficialCosmosCITokenizer":
        super().train(False)
        self.encoder.eval()
        self.decoder.eval()
        return self

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError(f"Expected Cosmos pixels [B,3,H,W], got {tuple(pixels.shape)}")
        if pixels.shape[-2:] != (self.input_size, self.input_size):
            pixels = F.interpolate(
                pixels,
                size=(self.input_size, self.input_size),
                mode="bicubic",
                align_corners=False,
            )
        dtype = next(self.encoder.parameters()).dtype
        latent = self.encoder((pixels.clamp(0, 1) * 2.0 - 1.0).to(dtype=dtype))
        if isinstance(latent, (tuple, list)):
            latent = latent[0]
        if latent.ndim != 4 or tuple(latent.shape[1:]) != (16, 16, 16):
            raise ValueError(
                "Cosmos CI16x16 latent contract changed: "
                f"observed={tuple(latent.shape)}"
            )
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected Cosmos latent [B,16,H,W], got {tuple(latent.shape)}")
        dtype = next(self.decoder.parameters()).dtype
        pixels = self.decoder(latent.to(dtype=dtype))
        if isinstance(pixels, (tuple, list)):
            pixels = pixels[0]
        return ((pixels + 1.0) * 0.5).clamp(0.0, 1.0)


class OfficialVJEPASVAETokenizer(nn.Module):
    """Frozen V-JEPA 2.1 ViT-L + released S-VAE-96 + pixel decoder."""

    latent_dim = 96

    def __init__(
        self,
        encoder: nn.Module,
        adapter: nn.Module,
        pixel_decoder: nn.Module | None,
        *,
        input_size: int = 256,
        source: str = "injected",
        compile_adapter: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.pixel_decoder = pixel_decoder
        self._adapter_encode = adapter.encode
        if compile_adapter:
            if not torch.cuda.is_available():
                raise RuntimeError("V-JEPA adapter compilation requires CUDA")
            self._adapter_encode = torch.compile(
                adapter.encode,
                dynamic=False,
                fullgraph=False,
                mode="reduce-overhead",
            )
        self.input_size = int(input_size)
        self.source = str(source)
        self.register_buffer(
            "encoder_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "encoder_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_official_assets(
        cls,
        *,
        vjepa_source_root: str | Path,
        semantic_wm_source_root: str | Path,
        vjepa_checkpoint_path: str | Path,
        adapter_checkpoint_path: str | Path,
        device: torch.device | str,
        dtype: torch.dtype,
        input_size: int = 256,
        cast_encoder_to_dtype: bool = False,
        compile_encoder: bool = False,
        compile_adapter: bool = False,
        load_pixel_decoder: bool = True,
    ) -> "OfficialVJEPASVAETokenizer":
        vjepa_root = _require_dir(vjepa_source_root, label="V-JEPA 2.1 source")
        semantic_root = _require_dir(
            semantic_wm_source_root,
            label="semantic-wm source",
        )
        vjepa_checkpoint = _require_file(
            vjepa_checkpoint_path,
            label="V-JEPA 2.1 checkpoint",
        )
        adapter_checkpoint = _require_file(
            adapter_checkpoint_path,
            label="V-JEPA S-VAE checkpoint",
        )

        encoder = _load_vjepa2_1_encoder(
            source_root=vjepa_root,
            checkpoint_path=vjepa_checkpoint,
            device=device,
            dtype=dtype,
            cast_weights_to_dtype=cast_encoder_to_dtype,
            compile_encoder=compile_encoder,
        )
        adapters_module, decoder_module = _load_semantic_wm_release(semantic_root)
        released = torch.load(
            adapter_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        for key in (
            "adapter",
            "adapter_config",
            "encoder_config",
            "pixel_decoder",
            "pixel_decoder_config",
        ):
            if key not in released:
                raise ValueError(f"S-VAE checkpoint is missing {key}")
        encoder_config = dict(released["encoder_config"])
        if encoder_config.get("encoder_type") != "vjepa2":
            raise ValueError(
                "S-VAE encoder mismatch: "
                f"{encoder_config.get('encoder_type')!r} != 'vjepa2'"
            )
        adapter = adapters_module.create_adapter(
            dict(released["adapter_config"]),
            input_dim=1024,
        )
        pixel_decoder = None
        if load_pixel_decoder:
            pixel_decoder = decoder_module.create_pixel_decoder(
                dict(released["pixel_decoder_config"])
            )
        adapter.load_state_dict(
            _strip_compiled_prefix(released["adapter"]),
            strict=True,
        )
        if pixel_decoder is not None:
            pixel_decoder.load_state_dict(
                _strip_compiled_prefix(released["pixel_decoder"]),
                strict=True,
            )
        del released
        adapter = adapter.to(device=device, dtype=dtype).eval()
        if pixel_decoder is not None:
            pixel_decoder = pixel_decoder.to(device=device, dtype=dtype).eval()
        adapter.requires_grad_(False)
        if pixel_decoder is not None:
            pixel_decoder.requires_grad_(False)
        return cls(
            encoder,
            adapter,
            pixel_decoder,
            input_size=input_size,
            source=str(adapter_checkpoint),
            compile_adapter=compile_adapter,
        )

    def train(self, mode: bool = True) -> "OfficialVJEPASVAETokenizer":
        super().train(False)
        self.encoder.eval()
        self.adapter.eval()
        if self.pixel_decoder is not None:
            self.pixel_decoder.eval()
        return self

    def _prepare_encoder_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError(f"Expected V-JEPA pixels [B,3,H,W], got {tuple(pixels.shape)}")
        if pixels.shape[-2:] != (self.input_size, self.input_size):
            pixels = F.interpolate(
                pixels,
                size=(self.input_size, self.input_size),
                mode="bicubic",
                align_corners=False,
            )
        encoder_dtype = next(self.encoder.parameters()).dtype
        normalized = (
            pixels.clamp(0, 1) - self.encoder_mean.to(pixels)
        ) / self.encoder_std.to(pixels)
        return normalized.to(dtype=encoder_dtype)

    def _compress_final_tokens(self, encoded: torch.Tensor) -> torch.Tensor:
        adapter_dtype = next(self.adapter.parameters()).dtype
        if encoded.ndim != 3 or encoded.shape[1:] != (256, 1024):
            raise ValueError(
                "V-JEPA 2.1 image-token contract changed: "
                f"observed={tuple(encoded.shape)}"
            )
        features = encoded.to(dtype=adapter_dtype).reshape(
            encoded.shape[0], 1, 16, 16, 1024
        )
        compact = self._adapter_encode(features)
        if isinstance(compact, (tuple, list)):
            compact = compact[0]
        if compact.shape[1:] != (1, 16, 16, self.latent_dim):
            raise ValueError(
                "S-VAE-96 latent contract changed: "
                f"observed={tuple(compact.shape)}"
            )
        return compact[:, 0].permute(0, 3, 1, 2).contiguous()

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        normalized = self._prepare_encoder_pixels(pixels)
        adapter_dtype = next(self.adapter.parameters()).dtype
        use_autocast = normalized.is_cuda and adapter_dtype in {
            torch.bfloat16,
            torch.float16,
        }
        with torch.autocast(
            device_type="cuda",
            dtype=adapter_dtype,
            enabled=use_autocast,
        ):
            encoded = self.encoder(normalized.unsqueeze(2))
        return self._compress_final_tokens(encoded)

    def encode_with_multilayer_teacher(
        self,
        pixels: torch.Tensor,
        *,
        layer_indices: tuple[int, ...] = (5, 11, 17, 23),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return native S-VAE-96 latents and the official V-JEPA hierarchy.

        V-JEPA 2.1 ViT-L exposes normalized features at blocks 5/11/17/23
        through its ``return_hierarchical`` path.  The four chunks are read
        from one frozen encoder forward; the final chunk still feeds the
        released S-VAE adapter while the selected mean is used only as the
        detached spatial-relation teacher.
        """

        requested = tuple(int(index) for index in layer_indices)
        hierarchy = tuple(
            int(index)
            for index in getattr(
                self.encoder,
                "hierarchical_layers",
                (5, 11, 17, 23),
            )
        )
        if not requested or any(index not in hierarchy for index in requested):
            raise ValueError(
                f"Requested V-JEPA layers {requested} are not available in "
                f"the official hierarchy {hierarchy}"
            )
        if not hasattr(self.encoder, "return_hierarchical"):
            raise RuntimeError(
                "Pinned V-JEPA 2.1 encoder does not expose return_hierarchical"
            )

        normalized = self._prepare_encoder_pixels(pixels)
        adapter_dtype = next(self.adapter.parameters()).dtype
        use_autocast = normalized.is_cuda and adapter_dtype in {
            torch.bfloat16,
            torch.float16,
        }
        previous = bool(self.encoder.return_hierarchical)
        self.encoder.return_hierarchical = True
        try:
            with torch.autocast(
                device_type="cuda",
                dtype=adapter_dtype,
                enabled=use_autocast,
            ):
                hierarchical = self.encoder(normalized.unsqueeze(2))
        finally:
            self.encoder.return_hierarchical = previous

        if isinstance(hierarchical, (tuple, list)):
            chunks = tuple(hierarchical)
        elif (
            torch.is_tensor(hierarchical)
            and hierarchical.ndim == 3
            and hierarchical.shape[-1] == 1024 * len(hierarchy)
        ):
            chunks = tuple(hierarchical.chunk(len(hierarchy), dim=-1))
        else:
            observed = (
                tuple(hierarchical.shape)
                if torch.is_tensor(hierarchical)
                else type(hierarchical).__name__
            )
            raise ValueError(
                "V-JEPA hierarchical-token contract changed: "
                f"observed={observed}"
            )
        if len(chunks) != len(hierarchy) or any(
            chunk.ndim != 3 or chunk.shape[1:] != (256, 1024)
            for chunk in chunks
        ):
            raise ValueError(
                "V-JEPA hierarchy must contain four [B,256,1024] tensors"
            )

        selected = [chunks[hierarchy.index(index)] for index in requested]
        generalized = torch.stack(selected, dim=0).mean(dim=0)
        latent = self._compress_final_tokens(chunks[-1])
        teacher_map = generalized.transpose(1, 2).reshape(
            generalized.shape[0], 1024, 16, 16
        )
        return latent, teacher_map

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.pixel_decoder is None:
            raise RuntimeError(
                "Pixel decoder was intentionally omitted from this encode-only tokenizer"
            )
        if latent.ndim != 4 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected S-VAE latent [B,96,H,W], got {tuple(latent.shape)}")
        compact = latent.permute(0, 2, 3, 1).unsqueeze(1).contiguous()
        pixels = self.pixel_decoder(compact)
        if pixels.ndim != 5 or pixels.shape[1] != 1 or pixels.shape[-1] != 3:
            raise ValueError(
                "S-VAE pixel-decoder contract changed: "
                f"observed={tuple(pixels.shape)}"
            )
        return pixels[:, 0].permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0)
