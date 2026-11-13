#!/usr/bin/env python3
"""Instantiate a dynamics model from a config and check its parameter budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nwm.utils.config import load_config  # noqa: E402
from nwm.models.cdit import CDiT_models  # noqa: E402
from nwm.models.factory import model_kwargs_from_config  # noqa: E402


def build_model(config):
    """Build the CDiT described by ``config`` (no checkpoint, no tokenizer)."""

    image_size = int(config["image_size"])
    latent_size = config.get("latent_size")
    if latent_size is None:
        if image_size % 14 != 0:
            raise ValueError(
                "Set latent_size explicitly when the encoder does not use "
                "the DINOv2 patch size 14"
            )
        latent_size = image_size // 14
    config = dict(config)
    config.setdefault("head_width", 2048)
    kwargs = model_kwargs_from_config(
        config,
        latent_channels=int(config.get("model_in_channels", 768)),
        latent_size=int(latent_size),
    )
    return CDiT_models[config["model"]](**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str)
    parser.add_argument("--json-output", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    import torch

    # Meta tensors: counting parameters does not need materialized weights.
    with torch.device("meta"):
        model = build_model(config)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    budget_value = config.get("parameter_budget_trainable")
    budget = int(budget_value) if budget_value is not None else None
    tolerance = float(config.get("parameter_tolerance", 0.005))
    relative_error = abs(trainable - budget) / budget if budget is not None else None
    report = {
        "config": str(Path(args.config).resolve()),
        "attention_mode": model.attention_mode,
        "backbone_mode": model.backbone_mode,
        "model_in_channels": int(config.get("model_in_channels", 768)),
        "block_layout": [
            {"self_mixer": block.self_mixer_type, "history_mixer": block.context_mixer_type}
            for block in model.blocks
        ],
        "head_layout": [block.attention_type for block in model.head_blocks],
        "repa_mode": model.repa_mode,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "budget": budget,
        "relative_error": relative_error,
        "tolerance": tolerance,
        "within_budget": relative_error <= tolerance if relative_error is not None else None,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["within_budget"] is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
