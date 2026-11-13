#!/usr/bin/env python3
"""Generate paired tokenizer reconstructions for frozen time horizons."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nwm.data import misc
from nwm.utils.config import load_config
from nwm.data.datasets import EvalDataset
from nwm.data.eval_index import resolve_eval_index
from nwm.tokenizers.build import build_stage1_tokenizer


STOP_REQUESTED = False


def _request_stop(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_image(path: Path, image: torch.Tensor) -> None:
    pixels = (
        torch.nan_to_num(image.detach().cpu().float())
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )
    output = Image.fromarray((pixels * 255).astype("uint8"), mode="RGB")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    output.save(temporary, format="PNG")
    os.replace(temporary, path)


def _dataset(config: dict, dataset_name: str, index_path: Path) -> EvalDataset:
    data = config["eval_datasets"][dataset_name]
    return EvalDataset(
        data_folder=data["data_folder"],
        data_split_folder=data["test"],
        dataset_name=dataset_name,
        image_size=config["image_size"],
        min_dist_cat=config["eval_distance"]["eval_min_dist_cat"],
        max_dist_cat=config["eval_distance"]["eval_max_dist_cat"],
        len_traj_pred=config["eval_len_traj_pred"],
        traj_stride=config["traj_stride"],
        context_size=config["context_size"],
        normalize=config["normalize"],
        transform=misc.transform,
        goals_per_obs=4,
        predefined_index=index_path,
        traj_names="traj_names.txt",
    )


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--eval-index-dir", required=True, type=Path)
    parser.add_argument("--datasets", default="sacson,recon,scand")
    parser.add_argument("--seconds", default="4,16")
    parser.add_argument("--input-fps", type=int, default=4)
    parser.add_argument(
        "--eval-len-traj-pred",
        type=int,
        default=None,
        help="Override the config future-frame length for long-horizon suites.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-ids",
        type=int,
        default=None,
        help="Optional per-dataset cap used by smoke tests.",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    config_path = Path(args.config).resolve()
    config = load_config(
        str(config_path),
        defaults_path="config/eval_config.yaml",
    )
    if args.eval_len_traj_pred is not None:
        config["eval_len_traj_pred"] = int(args.eval_len_traj_pred)
    datasets = [
        value.strip() for value in args.datasets.split(",") if value.strip()
    ]
    seconds = sorted(
        {int(value.strip()) for value in args.seconds.split(",") if value.strip()}
    )
    if not seconds or min(seconds) <= 0:
        raise ValueError("seconds must contain positive integers")
    frame_indices = {
        second: second * int(args.input_fps) - 1 for second in seconds
    }
    if max(frame_indices.values()) >= int(config["eval_len_traj_pred"]):
        raise ValueError(
            "Requested tokenizer-floor horizon exceeds eval_len_traj_pred"
        )

    device = torch.device("cuda")
    tokenizer, info = build_stage1_tokenizer(
        config,
        device=device,
        project_root=Path(__file__).resolve().parents[1],
    )
    tokenizer.eval()
    output_root = args.output_dir.resolve()
    progress_path = output_root / "progress.json"
    progress = {
        "config": str(config_path),
        "tokenizer": info.to_dict(),
        "seconds": seconds,
        "datasets": {},
        "status": "running",
    }
    output_root.mkdir(parents=True, exist_ok=True)

    for dataset_name in datasets:
        index_path = resolve_eval_index(
            args.eval_index_dir.resolve(),
            dataset_name,
            "openloop",
        )
        dataset = _dataset(config, dataset_name, index_path)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=True,
            drop_last=False,
        )
        completed = 0
        for idxs, _obs_image, gt_image, _delta in loader:
            if args.max_ids is not None:
                remaining = int(args.max_ids) - completed
                if remaining <= 0:
                    break
                idxs = idxs[:remaining]
                gt_image = gt_image[:remaining]
            gt_image = gt_image.to(device, non_blocking=True)
            selected = torch.stack(
                [gt_image[:, frame_indices[second]] for second in seconds],
                dim=1,
            )
            batch, horizons = selected.shape[:2]
            pixels = (selected.flatten(0, 1) * 0.5 + 0.5).clamp(0, 1)
            with torch.amp.autocast(
                "cuda",
                enabled=True,
                dtype=torch.bfloat16,
            ):
                reconstruction = tokenizer.decode(tokenizer.encode(pixels))
            if reconstruction.shape[-2:] != pixels.shape[-2:]:
                reconstruction = F.interpolate(
                    reconstruction,
                    size=pixels.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
            reconstruction = reconstruction.unflatten(0, (batch, horizons))
            for row, sample_id in enumerate(idxs.reshape(-1).tolist()):
                sample_root = (
                    output_root
                    / dataset_name
                    / "time"
                    / f"id_{int(sample_id)}"
                )
                sample_root.mkdir(parents=True, exist_ok=True)
                for horizon_index, second in enumerate(seconds):
                    output_path = sample_root / f"{second}.png"
                    if not output_path.is_file():
                        _atomic_image(
                            output_path,
                            reconstruction[row, horizon_index],
                        )
            completed += int(batch)
            progress["datasets"][dataset_name] = {
                "completed": completed,
                "total": min(
                    len(dataset),
                    int(args.max_ids)
                    if args.max_ids is not None
                    else len(dataset),
                ),
            }
            _atomic_json(progress_path, progress)
            if STOP_REQUESTED:
                progress["status"] = "interrupted"
                _atomic_json(progress_path, progress)
                return 143

    progress["status"] = "completed"
    _atomic_json(progress_path, progress)
    print(json.dumps(progress, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
