"""Strict loading helpers for the pinned official PAE tokenizer."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


@dataclass(frozen=True)
class PAELoadReport:
    checkpoint: str
    checkpoint_sha256: str
    checkpoint_tensors: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    source_root: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_asset_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise TypeError("PAE asset manifest must contain a JSON object")
    return manifest


def validate_pae_assets(
    manifest: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    dino_model_path: str | Path,
) -> dict[str, str]:
    checkpoint = Path(checkpoint_path).resolve()
    dino = Path(dino_model_path).resolve()
    required = {
        "checkpoint": (
            checkpoint,
            int(manifest["checkpoint"]["bytes"]),
            str(manifest["checkpoint"]["sha256"]),
        ),
        "dino_config": (
            dino / "config.json",
            None,
            str(manifest["dino"]["config_sha256"]),
        ),
        "dino_preprocessor": (
            dino / "preprocessor_config.json",
            None,
            str(manifest["dino"]["preprocessor_sha256"]),
        ),
        "dino_model": (
            dino / "model.safetensors",
            None,
            str(manifest["dino"]["model_sha256"]),
        ),
    }
    hashes: dict[str, str] = {}
    for name, (path, expected_bytes, expected_hash) in required.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing PAE asset {name}: {path}")
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            raise ValueError(
                f"PAE asset size mismatch for {path}: "
                f"{path.stat().st_size} != {expected_bytes}"
            )
        observed = sha256_file(path)
        if observed != expected_hash:
            raise ValueError(
                f"PAE asset hash mismatch for {path}: {observed} != {expected_hash}"
            )
        hashes[name] = observed
    return hashes


def validate_pae_source(
    manifest: Mapping[str, Any],
    source_root: str | Path,
) -> dict[str, str]:
    root = Path(source_root).resolve()
    expected_files = dict(manifest["pae_source"]["patched_file_sha256"])
    expected_files.update(
        {
            "configs/pae/PAE_DINOv2L_d32.yaml": str(
                manifest["pae_source"]["config_sha256"]
            ),
            "configs/decoder/ViTL/config.json": str(
                manifest["pae_source"]["decoder_config_sha256"]
            ),
        }
    )
    observed: dict[str, str] = {}
    for relative, expected_hash in sorted(expected_files.items()):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned PAE source file: {path}")
        value = sha256_file(path)
        if value != expected_hash:
            raise ValueError(
                f"PAE source hash mismatch for {relative}: {value} != {expected_hash}"
            )
        observed[relative] = value
    return observed


def patch_transformers_dinov2_lazy_imports() -> None:
    """Repair stale lazy-import routing seen in the pinned evaluation environment."""

    import transformers
    from transformers.models.dinov2.modeling_dinov2 import (
        Dinov2Backbone,
        Dinov2ForImageClassification,
        Dinov2Model,
        Dinov2PreTrainedModel,
    )
    from transformers.models.dinov2_with_registers.modeling_dinov2_with_registers import (
        Dinov2WithRegistersBackbone,
        Dinov2WithRegistersForImageClassification,
        Dinov2WithRegistersModel,
        Dinov2WithRegistersPreTrainedModel,
    )

    direct = {
        "Dinov2Model": Dinov2Model,
        "Dinov2ForImageClassification": Dinov2ForImageClassification,
        "Dinov2Backbone": Dinov2Backbone,
        "Dinov2PreTrainedModel": Dinov2PreTrainedModel,
        "Dinov2WithRegistersModel": Dinov2WithRegistersModel,
        "Dinov2WithRegistersForImageClassification": (
            Dinov2WithRegistersForImageClassification
        ),
        "Dinov2WithRegistersBackbone": Dinov2WithRegistersBackbone,
        "Dinov2WithRegistersPreTrainedModel": Dinov2WithRegistersPreTrainedModel,
    }
    replacements = {
        name: (
            "models.dinov2_with_registers.modeling_dinov2_with_registers"
            if "WithRegisters" in name
            else "models.dinov2.modeling_dinov2"
        )
        for name in direct
    }
    class_map = getattr(transformers, "_class_to_module", None)
    if class_map is not None:
        class_map.update(replacements)
    for name, value in direct.items():
        setattr(transformers, name, value)


def _import_pae_class(source_root: Path):
    patch_transformers_dinov2_lazy_imports()
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = importlib.import_module("tokenizer.pae")
    return module.PAE


def _resolved_params(
    *,
    source_root: Path,
    config_path: Path,
    dino_model_path: Path,
) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    params = dict(config["stage_1"]["params"])
    params["encoder_config_path"] = str(dino_model_path)
    encoder_params = dict(params["encoder_params"])
    encoder_params["dinov2_path"] = str(dino_model_path)
    params["encoder_params"] = encoder_params
    decoder_config = Path(str(params["decoder_config_path"]))
    if not decoder_config.is_absolute():
        decoder_config = source_root / decoder_config
    params["decoder_config_path"] = str(decoder_config)
    return params


def load_official_pae(
    *,
    source_root: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
    dino_model_path: str | Path,
    asset_manifest_path: str | Path,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.nn.Module, PAELoadReport]:
    source = Path(source_root).resolve()
    config = Path(config_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    dino = Path(dino_model_path).resolve()
    manifest = load_asset_manifest(asset_manifest_path)
    validate_pae_source(manifest, source)
    asset_hashes = validate_pae_assets(
        manifest,
        checkpoint_path=checkpoint,
        dino_model_path=dino,
    )

    pae_class = _import_pae_class(source)
    model = pae_class(
        **_resolved_params(
            source_root=source,
            config_path=config,
            dino_model_path=dino,
        )
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if set(state) != {"model"} or not isinstance(state["model"], dict):
        raise ValueError("Official PAE checkpoint must contain only a model state")
    if not all(torch.is_tensor(value) and torch.isfinite(value).all() for value in state["model"].values()):
        raise ValueError("Official PAE checkpoint contains non-finite or non-tensor state")
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    allowed_unexpected = set(manifest["protocol"]["allowed_unexpected_checkpoint_keys"])
    if missing:
        raise RuntimeError(f"Official PAE load has missing keys: {missing}")
    if set(unexpected) != allowed_unexpected:
        raise RuntimeError(
            "Official PAE unexpected-key contract changed: "
            f"observed={sorted(unexpected)}, expected={sorted(allowed_unexpected)}"
        )
    model = model.to(device=device, dtype=dtype).eval()
    # Upstream PAE keeps these normalization tensors as plain attributes rather
    # than registered buffers, so Module.to(dtype=...) does not convert them.
    # Match the released model dtype explicitly to avoid promoting normalized
    # inputs back to fp32 before the bf16 delta encoder.
    model.encoder_mean = model.encoder_mean.to(device=device, dtype=dtype)
    model.encoder_std = model.encoder_std.to(device=device, dtype=dtype)
    report = PAELoadReport(
        checkpoint=str(checkpoint),
        checkpoint_sha256=asset_hashes["checkpoint"],
        checkpoint_tensors=len(state["model"]),
        missing_keys=tuple(missing),
        unexpected_keys=tuple(sorted(unexpected)),
        source_root=str(source),
    )
    del state
    return model, report
