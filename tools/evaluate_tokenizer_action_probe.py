#!/usr/bin/env python3
"""Evaluate frozen-tokenizer navigation semantics with a fixed ridge probe.

This is the navigation-domain counterpart of the inverse-dynamics probe in
``semantic-wm``: a single, deliberately low-capacity readout predicts relative
SE(2) motion from the difference between pooled goal and observation latents.
The split, sample counts, regularization, and seed are identical for every
tokenizer arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAE_SRC = PROJECT_ROOT / "RAE" / "src"
for root in (PROJECT_ROOT, RAE_SRC):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from nwm.data import misc  # noqa: E402
from nwm.utils.config import load_config  # noqa: E402
from nwm.data.datasets import TrainingDataset  # noqa: E402
from nwm.tokenizers.build import build_stage1_tokenizer  # noqa: E402


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _make_dataset(config: dict, dataset_name: str, split: str, seed: int):
    data = config["datasets"][dataset_name]
    distance = data.get("distance", config["distance"])
    return TrainingDataset(
        data_folder=data["data_folder"],
        data_split_folder=data[split],
        dataset_name=dataset_name,
        image_size=config["image_size"],
        min_dist_cat=distance["min_dist_cat"],
        max_dist_cat=distance["max_dist_cat"],
        len_traj_pred=data.get("len_traj_pred", config["len_traj_pred"]),
        traj_stride=1,
        context_size=config["context_size"],
        transform=misc.transform,
        normalize=config["normalize"],
        goals_per_obs=int(data["goals_per_obs"]),
        deterministic_sampling=True,
        sample_seed=seed,
        index_cache_folder=str(
            Path(config["dataset_index_root"]) / dataset_name / split
        ),
        goal_offset_bins=(
            config.get("goal_offset_bins") if split == "train" else None
        ),
    )


def _indices(length: int, count: int, seed: int, label: str) -> list[int]:
    label_seed = int.from_bytes(
        hashlib.sha256(label.encode("utf-8")).digest()[:8], "little"
    )
    permutation = np.random.default_rng(seed ^ label_seed).permutation(length)
    return [int(value) for value in permutation[: min(length, count)]]


@torch.inference_mode()
def _encode(tokenizer, images: torch.Tensor, batch_size: int) -> torch.Tensor:
    device = next(tokenizer.parameters()).device
    outputs = []
    for start in range(0, len(images), batch_size):
        pixels = (
            images[start : start + batch_size].to(device) * 0.5 + 0.5
        ).clamp(0.0, 1.0)
        latent = tokenizer.encode(pixels).float()
        outputs.append(latent.mean(dim=(-2, -1)).cpu())
    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def _features_for_split(
    tokenizer,
    config: dict,
    split: str,
    observations_per_dataset: int,
    seed: int,
    encode_batch_size: int,
    dataset_names: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    all_features = []
    all_targets = []
    rows = []
    selected_datasets = dataset_names or sorted(config["datasets"])
    unknown = set(selected_datasets) - set(config["datasets"])
    if unknown:
        raise ValueError(f"Unknown probe datasets: {sorted(unknown)}")
    if len(set(selected_datasets)) != len(selected_datasets):
        raise ValueError("Probe dataset selection contains duplicates")
    context_size = int(config["context_size"])
    for dataset_name in selected_datasets:
        dataset = _make_dataset(config, dataset_name, split, seed)
        selected = _indices(
            len(dataset),
            observations_per_dataset,
            seed,
            f"{dataset_name}:{split}",
        )
        for item_index in selected:
            images, targets, relative_time, _relatives = dataset[item_index]
            goal_count = int(targets.shape[0])
            chosen = torch.cat(
                [
                    images[context_size - 1 : context_size],
                    images[context_size : context_size + goal_count],
                ],
                dim=0,
            )
            pooled = _encode(tokenizer, chosen, encode_batch_size)
            features = pooled[1:] - pooled[0]
            all_features.append(features)
            all_targets.append(targets.float().cpu())
            for goal_slot in range(goal_count):
                rows.append(
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "dataset_index": item_index,
                        "goal_slot": goal_slot,
                        "relative_time": float(relative_time[goal_slot]),
                    }
                )
    return torch.cat(all_features), torch.cat(all_targets), rows


def _pearson(prediction: torch.Tensor, target: torch.Tensor) -> list[float]:
    values = []
    for dim in range(target.shape[1]):
        x = prediction[:, dim] - prediction[:, dim].mean()
        y = target[:, dim] - target[:, dim].mean()
        denominator = torch.sqrt(x.square().sum() * y.square().sum())
        value = (x * y).sum() / denominator.clamp_min(1.0e-12)
        values.append(float(value))
    return values


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    error = prediction - target
    residual = error.square().sum()
    centered = target - target.mean(dim=0)
    total = centered.square().sum().clamp_min(1.0e-12)
    correlations = _pearson(prediction, target)
    return {
        "pairs": int(len(target)),
        "r2": float(1.0 - residual / total),
        "rmse": float(torch.sqrt(error.square().mean())),
        "pearson_per_dimension": correlations,
        "pearson_mean": float(sum(correlations) / len(correlations)),
    }


def _fit_and_evaluate(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, dict]:
    x_train = x_train.double()
    y_train = y_train.double()
    x_test = x_test.double()
    y_test = y_test.double()
    x_mean = x_train.mean(dim=0)
    x_std = x_train.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    y_mean = y_train.mean(dim=0)
    y_std = y_train.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    train_x = (x_train - x_mean) / x_std
    test_x = (x_test - x_mean) / x_std
    train_y = (y_train - y_mean) / y_std
    dimension = int(train_x.shape[1])
    gram = train_x.T @ train_x
    weights = torch.linalg.solve(
        gram + float(alpha) * torch.eye(dimension, dtype=torch.float64),
        train_x.T @ train_y,
    )
    prediction = (test_x @ weights) * y_std + y_mean
    return prediction, _metrics(prediction, y_test)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-observations-per-dataset", type=int, default=128)
    parser.add_argument("--test-observations-per-dataset", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated dataset subset for isolated formal units.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Tokenizer action probe requires CUDA")
    config = load_config(args.config, defaults_path="config/eval_config.yaml")
    dataset_names = (
        [value.strip() for value in args.datasets.split(",") if value.strip()]
        if args.datasets is not None
        else None
    )
    if dataset_names == []:
        raise ValueError("Probe dataset selection is empty")
    tokenizer, tokenizer_info = build_stage1_tokenizer(
        config,
        device=torch.device("cuda"),
        project_root=PROJECT_ROOT,
    )
    train_x, train_y, train_rows = _features_for_split(
        tokenizer,
        config,
        "train",
        args.train_observations_per_dataset,
        args.seed,
        args.encode_batch_size,
        dataset_names,
    )
    test_x, test_y, test_rows = _features_for_split(
        tokenizer,
        config,
        "test",
        args.test_observations_per_dataset,
        args.seed + 1,
        args.encode_batch_size,
        dataset_names,
    )
    prediction, overall = _fit_and_evaluate(
        train_x, train_y, test_x, test_y, args.ridge_alpha
    )
    per_dataset = {}
    per_goal_slot = {}
    for label, values in (
        ("dataset", sorted({row["dataset"] for row in test_rows})),
        ("goal_slot", sorted({row["goal_slot"] for row in test_rows})),
    ):
        destination = per_dataset if label == "dataset" else per_goal_slot
        for value in values:
            mask = torch.tensor([row[label] == value for row in test_rows])
            destination[str(value)] = _metrics(prediction[mask], test_y[mask])
    payload = {
        "schema_version": 1,
        "status": "completed",
        "protocol": "navigation_relative_se2_ridge_probe_v1",
        "paper_correspondence": (
            "Navigation-domain low-capacity analogue of the frozen-latent "
            "inverse-dynamics probe in arXiv:2605.06388; not BridgeV2 IDM."
        ),
        "tokenizer": tokenizer_info.to_dict(),
        "feature": "spatial_mean(goal_latent)-spatial_mean(observation_latent)",
        "target": "normalized relative SE(2) [x,y,yaw]",
        "seed": args.seed,
        "ridge_alpha": args.ridge_alpha,
        "train_pairs": int(len(train_y)),
        "test_pairs": int(len(test_y)),
        "latent_channels": int(train_x.shape[1]),
        "overall": overall,
        "per_dataset": per_dataset,
        "per_goal_slot": per_goal_slot,
        "datasets": sorted({row["dataset"] for row in test_rows}),
        "selection": {"train": train_rows, "test": test_rows},
    }
    finite_values = [
        overall["r2"], overall["rmse"], overall["pearson_mean"],
        *overall["pearson_per_dimension"],
    ]
    if any(not math.isfinite(value) for value in finite_values):
        raise ValueError(f"Non-finite probe metric: {finite_values}")
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps(overall, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
