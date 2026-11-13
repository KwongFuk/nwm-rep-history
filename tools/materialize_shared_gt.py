#!/usr/bin/env python3
"""Materialize decoded evaluation ground truth without reserving a GPU.

The model evaluator only copies transformed held-out frames when ``--gt 1``.
This utility performs that deterministic operation directly from a frozen
evaluation index, validates every written PNG, and emits the same marker
contract consumed by the GPU evaluation jobs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Iterable

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nwm.utils.config import load_config
from nwm.data import misc


DATASETS = ("sacson", "recon", "scand")


def image_size_pair(value: int | Iterable[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    pair = tuple(int(item) for item in value)
    if len(pair) != 2:
        raise ValueError(f"Expected scalar or two-dimensional image size: {pair}")
    return pair


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _transformed_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        tensor = misc.transform(image.convert("RGB"))
    pixels = (
        ((tensor * 0.5 + 0.5).clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255)
        .astype(np.uint8)
    )
    return Image.fromarray(pixels, mode="RGB")


def _atomic_png(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.tmp.{os.getpid()}{target.suffix}")
    image = _transformed_rgb(source)
    image.save(temporary, format="PNG")
    os.replace(temporary, target)


def _atomic_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        os.link(source, temporary)
    except FileExistsError:
        temporary.unlink()
        os.link(source, temporary)
    os.replace(temporary, target)


def _validate_pngs(paths: Iterable[Path], expected_size: tuple[int, int]) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != expected_size:
                raise ValueError(
                    f"Unexpected generated image contract: {path} "
                    f"mode={image.mode} size={image.size}"
                )


def materialize_sample(
    *,
    sample_id: int,
    row: tuple,
    data_root: Path,
    dataset: str,
    output_root: Path,
    future_frames: int,
    input_fps: int,
    rollout_fps: int,
    horizons: tuple[int, ...],
    image_size: tuple[int, int],
) -> int:
    trajectory, current_time, *_unused = row
    stride = input_fps // rollout_fps
    if stride < 1 or input_fps % rollout_fps:
        raise ValueError(
            f"rollout_fps={rollout_fps} must divide input_fps={input_fps}"
        )
    rollout_dir = output_root / dataset / f"rollout_{rollout_fps}fps" / f"id_{sample_id}"
    direct_dir = output_root / dataset / "time" / f"id_{sample_id}"
    rollout_paths = []
    direct_paths = []
    rollout_frame = 0
    direct_by_frame = {second * input_fps - 1: second for second in horizons}
    for future_index in range(future_frames):
        source = data_root / str(trajectory) / f"{int(current_time) + future_index + 1}.jpg"
        if not source.is_file():
            raise FileNotFoundError(source)
        if future_index % stride != stride - 1:
            continue
        rollout_target = rollout_dir / f"{rollout_frame}.png"
        if not rollout_target.is_file():
            _atomic_png(source, rollout_target)
        rollout_paths.append(rollout_target)
        if future_index in direct_by_frame:
            direct_target = direct_dir / f"{direct_by_frame[future_index]}.png"
            if not direct_target.is_file():
                _atomic_link(rollout_target, direct_target)
            direct_paths.append(direct_target)
        rollout_frame += 1
    if len(direct_paths) != len(horizons):
        raise ValueError(
            f"Direct horizon mismatch for sample {sample_id}: "
            f"observed={len(direct_paths)} expected={len(horizons)}"
        )
    expected_rollout = future_frames // stride
    if len(rollout_paths) != expected_rollout:
        raise ValueError(
            f"Rollout mismatch for sample {sample_id}: "
            f"observed={len(rollout_paths)} expected={expected_rollout}"
        )
    _validate_pngs((*rollout_paths, *direct_paths), image_size)
    return len(rollout_paths) + len(direct_paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--eval-index-dir", required=True, type=Path)
    parser.add_argument("--index-manifest-sha256", required=True)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--future-frames", required=True, type=int)
    parser.add_argument("--horizons", required=True)
    parser.add_argument("--input-fps", type=int, default=4)
    parser.add_argument("--rollout-fps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Materialize the deterministic first N samples from the verified "
            "frozen index without rewriting that index."
        ),
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--evaluation-series-id", required=True)
    args = parser.parse_args()

    if args.future_frames <= 0 or args.workers <= 0:
        raise ValueError("future-frames and workers must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    horizons = tuple(int(value) for value in args.horizons.split(",") if value)
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError(f"Invalid horizons: {horizons}")
    if max(horizons) * args.input_fps > args.future_frames:
        raise ValueError("Direct horizons exceed future-frame coverage")

    manifest_path = args.eval_index_dir / "manifest.json"
    if _sha256(manifest_path) != args.index_manifest_sha256:
        raise ValueError("Frozen index manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest["future_frames"]) != args.future_frames:
        raise ValueError("Frozen index future-frame mismatch")
    metadata = manifest["datasets"][args.dataset]
    index_path = args.eval_index_dir / metadata["openloop_file"]
    if _sha256(index_path) != metadata["openloop_sha256"]:
        raise ValueError("Frozen open-loop index SHA-256 mismatch")
    with index_path.open("rb") as handle:
        rows = pickle.load(handle)
    if len(rows) != int(metadata["selected_samples"]):
        raise ValueError("Frozen index sample-count mismatch")
    source_sample_count = len(rows)
    if args.max_samples is not None:
        rows = rows[: int(args.max_samples)]

    config = load_config(str(args.config), defaults_path="config/eval_config.yaml")
    data_root = Path(config["eval_datasets"][args.dataset]["data_folder"])
    image_size = image_size_pair(config["image_size"])
    if image_size != (224, 224):
        raise ValueError(f"Unexpected paper image size: {image_size}")

    def run(item: tuple[int, tuple]) -> int:
        sample_id, row = item
        return materialize_sample(
            sample_id=sample_id,
            row=row,
            data_root=data_root,
            dataset=args.dataset,
            output_root=args.output_root,
            future_frames=args.future_frames,
            input_fps=args.input_fps,
            rollout_fps=args.rollout_fps,
            horizons=horizons,
            image_size=image_size,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        counts = list(executor.map(run, enumerate(rows)))
    expected_per_sample = len(horizons) + args.future_frames // (
        args.input_fps // args.rollout_fps
    )
    if counts != [expected_per_sample] * len(rows):
        raise ValueError("Generated per-sample coverage mismatch")

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    for marker in (".time_complete", f".rollout_{args.rollout_fps}fps_complete"):
        _atomic_text(args.artifact_root / marker, "cpu_ground_truth_complete\n")
    payload = {
        "schema_version": 1,
        "status": "complete",
        "protocol": "cpu_materialized_shared_ground_truth_v1",
        "evaluation_series_id": args.evaluation_series_id,
        "arm": "shared_gt",
        "dataset": args.dataset,
        "hardware": "cpu",
        "index_manifest": str(manifest_path),
        "index_manifest_sha256": args.index_manifest_sha256,
        "evaluation_types": ["time", "rollout"],
        "direct_horizons_seconds": list(horizons),
        "future_frames": args.future_frames,
        "selected_samples": len(rows),
        "source_selected_samples": source_sample_count,
        "selection": (
            "deterministic_first_n_from_verified_frozen_index"
            if args.max_samples is not None
            else "all_verified_frozen_index_samples"
        ),
        "generated_png_references": int(sum(counts)),
        "unique_rollout_pngs": len(rows) * args.future_frames,
        "output_root": str(args.output_root),
    }
    _atomic_text(
        args.artifact_root / "run_manifest.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"shared_gt_complete dataset={args.dataset} samples={len(rows)} "
        f"png_references={sum(counts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
