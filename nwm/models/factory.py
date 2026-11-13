"""NWM construction with auditable paired initialization across latent spaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from nwm.models.cdit import CDiT_models


@dataclass(frozen=True)
class PairedInitializationReport:
    enabled: bool
    model_seed: int
    target_in_channels: int
    reference_in_channels: int
    copied_tensors: int
    target_only_tensors: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_kwargs_from_config(
    config: Mapping[str, Any],
    *,
    latent_channels: int,
    latent_size: int,
) -> dict[str, Any]:
    num_heads_value = config.get("num_heads")
    if num_heads_value is None:
        num_heads_value = {
            "CDiT-XL/2": 16,
            "CDiT-L/2": 16,
            "CDiT-B/2": 12,
            "CDiT-S/2": 6,
        }.get(str(config["model"]))
    model_kwargs = {
        "context_size": int(config["context_size"]),
        "input_size": int(latent_size),
        "in_channels": int(latent_channels),
        "learn_sigma": bool(config.get("learn_sigma", False)),
        "head_width": int(config.get("head_width", latent_channels)),
        "head_depth": int(config.get("head_depth", 2)),
        "head_num_heads": int(config.get("head_num_heads", 16)),
        "use_low_rank_adaln_head": bool(
            config.get("use_low_rank_adaln_head", False)
        ),
        "head_adaln_rank": (
            int(config["head_adaln_rank"])
            if config.get("head_adaln_rank") is not None
            else None
        ),
        "head_mlp_ratio": (
            float(config["head_mlp_ratio"])
            if config.get("head_mlp_ratio") is not None
            else None
        ),
        "self_attention_layout": config.get("self_attention_layout"),
        "self_mixer_layout": config.get("self_mixer_layout"),
        "head_attention_layout": config.get("head_attention_layout"),
        "window_size": int(config.get("window_size", 8)),
        "head_window_size": (
            int(config["head_window_size"])
            if config.get("head_window_size") is not None
            else None
        ),
        "linear_feature_scaling": bool(
            config.get("linear_feature_scaling", False)
        ),
        "mlp_ratio": float(config.get("mlp_ratio", 4.0)),
        "attention_mode": str(config.get("attention_mode", "softmax")),
        "backbone_mode": str(config.get("backbone_mode", "dit")),
        "gdn_softmax_every": int(config.get("gdn_softmax_every", 4)),
        "cross_gdn_layers_per_cycle": int(
            config.get("cross_gdn_layers_per_cycle", 3)
        ),
        "gdn_conv_kernel_size": int(config.get("gdn_conv_kernel_size", 4)),
        "gdn_backend": str(config.get("gdn_backend", "torch")),
        "gdn_cross_cache_backend": str(
            config.get("gdn_cross_cache_backend", "scan")
        ),
        "gdn_cross_query_backend": str(
            config.get("gdn_cross_query_backend", "torch")
        ),
        "gdn_state_dtype": str(
            config.get("gdn_state_dtype", "float32")
        ),
        "shifted_window_cross_mask": str(
            config.get("shifted_window_cross_mask", "legacy")
        ),
        "repa_mode": str(config.get("repa_mode", "none")),
        "repa_layer_depth": int(config.get("repa_layer_depth", 8)),
        "repa_kernel_size": int(config.get("repa_kernel_size", 3)),
        "repa_target_dim": int(config.get("repa_target_dim", latent_channels)),
    }
    # Custom model factories may own their own head-count default. Registered
    # production models always resolve above, while an explicit config value
    # remains authoritative without evaluating a missing fallback key.
    if num_heads_value is not None:
        model_kwargs["num_heads"] = int(num_heads_value)
    return model_kwargs


def build_nwm_model(
    config: Mapping[str, Any],
    *,
    latent_channels: int,
    latent_size: int,
) -> tuple[torch.nn.Module, PairedInitializationReport]:
    """Construct the target model and copy every compatible canonical tensor.

    Constructing C32 and C768 models from the same global seed is insufficient:
    the differently sized input projection advances the RNG by a different
    amount.  A canonical reference model makes every same-name, same-shape
    tensor bitwise identical across representation arms.
    """

    model_name = str(config["model"])
    kwargs = model_kwargs_from_config(
        config,
        latent_channels=latent_channels,
        latent_size=latent_size,
    )
    model_seed = int(config.get("model_init_seed", 0))
    reference_channels = int(
        config.get("paired_reference_in_channels", latent_channels)
    )
    reference_repa_mode = str(
        config.get(
            "paired_reference_repa_mode",
            config.get("repa_mode", "none"),
        )
    ).lower()
    reference_repa_target_dim = int(
        config.get(
            "paired_reference_repa_target_dim",
            config.get("repa_target_dim", reference_channels),
        )
    )
    reference_repa_kernel_size = int(
        config.get(
            "paired_reference_repa_kernel_size",
            config.get("repa_kernel_size", 3),
        )
    )
    reference_mlp_ratio = float(
        config.get(
            "paired_reference_mlp_ratio",
            config.get("mlp_ratio", 4.0),
        )
    )
    needs_reference = (
        reference_channels != int(latent_channels)
        or reference_repa_mode != str(config.get("repa_mode", "none")).lower()
        or reference_repa_target_dim
        != int(config.get("repa_target_dim", latent_channels))
        or reference_repa_kernel_size != int(config.get("repa_kernel_size", 3))
        or reference_mlp_ratio != float(config.get("mlp_ratio", 4.0))
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        target = CDiT_models[model_name](**kwargs)
        reference = None
        if needs_reference:
            reference_kwargs = dict(kwargs)
            reference_kwargs["in_channels"] = reference_channels
            reference_kwargs["repa_mode"] = reference_repa_mode
            reference_kwargs["repa_target_dim"] = reference_repa_target_dim
            reference_kwargs["repa_kernel_size"] = reference_repa_kernel_size
            reference_kwargs["mlp_ratio"] = reference_mlp_ratio
            torch.manual_seed(model_seed)
            reference = CDiT_models[model_name](**reference_kwargs)

    copied = 0
    target_only = 0
    if reference is not None:
        target_state = target.state_dict()
        reference_state = reference.state_dict()
        for name, value in target_state.items():
            reference_value = reference_state.get(name)
            if reference_value is not None and reference_value.shape == value.shape:
                value.copy_(reference_value)
                copied += 1
            else:
                target_only += 1
        target.load_state_dict(target_state, strict=True)
        del reference
    else:
        copied = len(target.state_dict())

    report = PairedInitializationReport(
        enabled="paired_reference_in_channels" in config,
        model_seed=model_seed,
        target_in_channels=int(latent_channels),
        reference_in_channels=reference_channels,
        copied_tensors=copied,
        target_only_tensors=target_only,
    )
    return target, report
