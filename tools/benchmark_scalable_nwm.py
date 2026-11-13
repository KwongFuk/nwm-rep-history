#!/usr/bin/env python3
"""Model-core latency/memory benchmark for history mixers on one CUDA device.

Compares Global-Softmax, Cached-Linear and Balanced GDN while varying the
history length and the number of candidate queries that share one history.
The benchmark separates:
1. a single training update, where observed context must remain differentiable;
2. one-time history-summary preparation (cached history mixers only);
3. denoiser calls that reuse that summary across an NFE-step flow solve.

Tokenizer, decoder, data loading, DDP, and rollout quality are intentionally
outside this systems-only benchmark. ``image_size`` sets the synthetic latent
grid through the DINO/14 token contract (224 -> 16x16 tokens).

Paper configurations (H100, BF16, 120 queries per history):
  config/systems/global_softmax.yaml       Global-Softmax
  config/systems/cached_linear.yaml        Cached-Linear
  config/systems/balanced_gdn_proxy.yaml   Balanced GDN
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nwm.utils.config import load_config  # noqa: E402
from tools.check_model_budget import build_model  # noqa: E402


DEFAULT_VARIANTS = (
    ("global_softmax", "config/systems/global_softmax.yaml"),
    ("cached_linear", "config/systems/cached_linear.yaml"),
    ("balanced_gdn", "config/systems/balanced_gdn_proxy.yaml"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_scenarios(raw: str) -> list[tuple[int, int]]:
    scenarios = []
    for item in str(raw).split(","):
        item = item.strip().lower()
        if not item:
            continue
        try:
            image_size, context_frames = (int(value) for value in item.split("x"))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"scenario must be IMAGE_SIZE x CONTEXT_FRAMES, got {item!r}"
            ) from error
        if image_size <= 0 or image_size % 14 != 0 or context_frames <= 0:
            raise ValueError(
                f"invalid scenario {item!r}; image size must be divisible by 14 "
                "and context frames must be positive"
            )
        scenarios.append((image_size, context_frames))
    if not scenarios or len(scenarios) != len(set(scenarios)):
        raise ValueError("scenarios must be non-empty and unique")
    return scenarios


def parse_variants(raw: list[str] | None) -> list[tuple[str, Path]]:
    specs = []
    if raw:
        for item in raw:
            if "=" not in item:
                raise ValueError(f"variant must be LABEL=CONFIG, got {item!r}")
            label, config = item.split("=", 1)
            specs.append((label.strip(), config.strip()))
    else:
        specs = list(DEFAULT_VARIANTS)
    result = []
    seen = set()
    for label, raw_path in specs:
        if not label or label in seen:
            raise ValueError(f"duplicate or empty variant label {label!r}")
        seen.add(label)
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append((label, path.resolve()))
    return result


def scenario_config(config_path: Path, image_size: int, context_frames: int):
    config = load_config(str(config_path))
    if not bool(config.get("model_core_only", False)):
        raise ValueError(
            f"{config_path} must declare model_core_only=true; this benchmark "
            "must not be confused with end-to-end tokenizer support"
        )
    config["image_size"] = int(image_size)
    config["context_size"] = int(context_frames)
    return config


def variant_metadata(label: str, config_path: Path) -> dict[str, Any]:
    config = load_config(str(config_path))
    with torch.device("meta"):
        model = build_model(config)
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    budget = int(config["parameter_budget_trainable"])
    tolerance = float(config.get("parameter_tolerance", 0.0))
    relative_error = abs(trainable - budget) / float(budget)
    if relative_error > tolerance:
        raise ValueError(
            f"{label} parameter budget failed: {trainable} vs {budget}"
        )
    result = {
        "label": label,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "run_name": str(config["run_name"]),
        "model_in_channels": int(config["model_in_channels"]),
        "trainable_parameters": trainable,
        "parameter_budget": budget,
        "head_width": int(model.head_width),
        "head_depth": int(model.head_depth),
        "head_mlp_ratio": float(model.head_mlp_ratio),
        "backbone_mlp_ratio": float(model.mlp_ratio),
        "use_low_rank_adaln_head": bool(model.use_low_rank_adaln_head),
        "head_adaln_rank": model.head_adaln_rank,
        "attention_mode": str(model.attention_mode),
        "self_attention_layout": list(model.self_attention_layout),
        "head_attention_layout": list(model.head_attention_layout),
        "window_size": int(model.window_size),
        "points": [],
    }
    del model
    gc.collect()
    return result


def pairwise_mac(model) -> dict[str, int]:
    target_tokens = int(model.x_embedder.num_patches)
    context_tokens = target_tokens * int(model.context_size)
    grid = int(target_tokens**0.5)
    backbone_width = int(model.x_embedder.proj.out_channels)
    head_width = int(model.head_width)

    backbone_self = 0
    for block in model.blocks:
        if str(block.self_mixer_type) == "gdn":
            continue
        mixer = str(block.self_mixer_type)
        keys = (
            min(int(model.window_size), grid) ** 2
            if mixer in {"window", "shifted_window"}
            else target_tokens
        )
        backbone_self += 2 * target_tokens * keys * backbone_width

    backbone_cross = 0
    for block in model.blocks:
        mixer = str(block.context_mixer_type)
        if mixer in {"gdn", "cached_linear"}:
            # These mixers read a fixed-size state/summary.  Their D^2 work is
            # intentionally not mislabeled as token-pair MACs.
            continue
        if mixer == "shifted_window":
            keys = (
                int(model.context_size)
                * min(int(model.window_size), grid) ** 2
            )
        else:
            keys = context_tokens
        backbone_cross += 2 * target_tokens * keys * backbone_width

    ddt_self = 0
    for block in model.head_blocks:
        keys = (
            min(int(model.head_window_size), grid) ** 2
            if block.attention_type in {"window", "shifted_window"}
            else target_tokens
        )
        ddt_self += 2 * target_tokens * keys * head_width
    return {
        "backbone_self": int(backbone_self),
        "backbone_cross": int(backbone_cross),
        "ddt_self": int(ddt_self),
        "total": int(backbone_self + backbone_cross + ddt_self),
    }


def make_inputs(
    config,
    batch_size: int,
    device,
    seed: int,
    *,
    context_group_size: int,
    input_dtype: torch.dtype,
):
    if batch_size % context_group_size != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by "
            f"context_group_size={context_group_size}"
        )
    generator = torch.Generator(device=device).manual_seed(int(seed))
    channels = int(config["model_in_channels"])
    grid = int(config["image_size"]) // 14
    context_frames = int(config["context_size"])
    observations = batch_size // context_group_size
    base_context = torch.randn(
        observations,
        context_frames,
        channels,
        grid,
        grid,
        generator=generator,
        device=device,
        dtype=input_dtype,
    )
    return {
        "x": torch.randn(
            batch_size,
            channels,
            grid,
            grid,
            generator=generator,
            device=device,
            dtype=input_dtype,
        ),
        "target": torch.randn(
            batch_size,
            channels,
            grid,
            grid,
            generator=generator,
            device=device,
            dtype=input_dtype,
        ),
        "t": torch.rand(batch_size, generator=generator, device=device),
        "action": torch.randn(
            batch_size, 3, generator=generator, device=device
        ),
        "base_context": base_context,
        # Keep one physical history per observation.  Cached inference never
        # needs repeated context pixels/tokens; uncached inference and the
        # differentiable training path materialize a grouped view explicitly.
        "context": base_context,
        "context_group_size": int(context_group_size),
        "relative_time": torch.rand(
            batch_size, generator=generator, device=device
        ),
    }


def timed_cuda(callable_, *, warmup: int, timed: int, device):
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize(device)
    durations = []
    for _ in range(timed):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        callable_()
        end.record()
        end.synchronize()
        durations.append(float(start.elapsed_time(end)))
    return {
        "mean_ms": statistics.fmean(durations),
        "median_ms": statistics.median(durations),
        "min_ms": min(durations),
        "max_ms": max(durations),
    }


def tensor_storage_bytes(value) -> int:
    """Count unique tensor storages in a nested cache structure."""

    seen = set()

    def visit(item) -> int:
        if isinstance(item, torch.Tensor):
            key = (
                item.untyped_storage().data_ptr(),
                item.untyped_storage().nbytes(),
            )
            if key in seen:
                return 0
            seen.add(key)
            return int(item.untyped_storage().nbytes())
        if isinstance(item, dict):
            return sum(visit(child) for child in item.values())
        if isinstance(item, (list, tuple)):
            return sum(visit(child) for child in item)
        return 0

    return visit(value)


def run_inference(
    model,
    inputs,
    *,
    warmup,
    timed,
    device,
    repeated_queries,
    full_generation_timed,
    measure_full_generation,
):
    model.eval()
    uses_memory = model.uses_cached_context
    memory_holder = {"value": None}
    inference_context = (
        inputs["context"]
        if uses_memory
        else inputs["context"].repeat_interleave(
            inputs["context_group_size"], dim=0
        )
    )

    with torch.inference_mode(), torch.amp.autocast(
        "cuda", dtype=torch.bfloat16
    ):
        prepare = None
        if uses_memory:
            torch.cuda.reset_peak_memory_stats(device)

            def prepare_call():
                base_memory = model.prepare_rollout_memory(
                    inputs["base_context"]
                )
                memory_holder["value"] = model.expand_rollout_memory(
                    base_memory,
                    repeats=inputs["context_group_size"],
                )

            prepare = timed_cuda(
                prepare_call,
                warmup=warmup,
                timed=timed,
                device=device,
            )
            prepare["peak_allocated_gib"] = (
                torch.cuda.max_memory_allocated(device) / 1024**3
            )
            prepare["compact_cache_gib"] = tensor_storage_bytes(
                memory_holder["value"]
            ) / 1024**3

        torch.cuda.reset_peak_memory_stats(device)

        def forward_call():
            return model(
                inputs["x"],
                inputs["t"],
                inputs["action"],
                inference_context,
                inputs["relative_time"],
                memory=memory_holder["value"],
            )

        forward = timed_cuda(
            forward_call,
            warmup=warmup,
            timed=timed,
            device=device,
        )
        forward["peak_allocated_gib"] = (
            torch.cuda.max_memory_allocated(device) / 1024**3
        )

        repeated = {}
        for count in repeated_queries:
            torch.cuda.reset_peak_memory_stats(device)

            def repeated_call(count=count):
                output = None
                for _ in range(int(count)):
                    output = forward_call()
                return output

            timing = timed_cuda(
                repeated_call,
                warmup=1,
                timed=max(1, min(int(timed), 3)),
                device=device,
            )
            timing["peak_allocated_gib"] = (
                torch.cuda.max_memory_allocated(device) / 1024**3
            )
            repeated[str(count)] = timing

        full_generation = None
        if measure_full_generation:
            torch.cuda.reset_peak_memory_stats(device)

            def full_generation_call():
                if uses_memory:
                    base_memory = model.prepare_rollout_memory(
                        inputs["base_context"]
                    )
                    memory_holder["value"] = model.expand_rollout_memory(
                        base_memory,
                        repeats=inputs["context_group_size"],
                    )
                output = None
                for _ in range(50):
                    output = forward_call()
                return output

            full_generation = timed_cuda(
                full_generation_call,
                warmup=1,
                timed=int(full_generation_timed),
                device=device,
            )
            full_generation["peak_allocated_gib"] = (
                torch.cuda.max_memory_allocated(device) / 1024**3
            )

    prepare_ms = 0.0 if prepare is None else float(prepare["mean_ms"])
    forward_ms = float(forward["mean_ms"])
    return {
        "context_cache_used": uses_memory,
        "prepare_context": prepare,
        "cached_denoiser": forward,
        "repeated_queries": repeated,
        "full_generation_nfe50": full_generation,
        "solve_estimate_ms": {
            str(nfe): prepare_ms + nfe * forward_ms for nfe in (50, 16, 8, 4)
        },
    }


def run_training(model, inputs, *, warmup, timed, device):
    model.train()
    training_context = inputs["context"].repeat_interleave(
        inputs["context_group_size"], dim=0
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0e-4,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model_kwargs = {}
            if model.uses_cached_context:
                model_kwargs["context_group_size"] = inputs[
                    "context_group_size"
                ]
            output = model(
                inputs["x"],
                inputs["t"],
                inputs["action"],
                training_context,
                inputs["relative_time"],
                **model_kwargs,
            )
            loss = F.mse_loss(output, inputs["target"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss

    torch.cuda.reset_peak_memory_stats(device)
    timing = timed_cuda(step, warmup=warmup, timed=timed, device=device)
    timing["peak_allocated_gib"] = (
        torch.cuda.max_memory_allocated(device) / 1024**3
    )
    timing["samples_per_second"] = (
        inputs["x"].shape[0] * 1000.0 / float(timing["mean_ms"])
    )
    del optimizer
    return timing


def run_point(
    *,
    config_path,
    image_size,
    context_frames,
    batch_size,
    context_group_size,
    input_dtype,
    warmup,
    timed,
    include_training,
    device,
    seed,
    repeated_queries,
    full_generation_timed,
    measure_full_generation,
):
    config = scenario_config(config_path, image_size, context_frames)
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    model = build_model(config).to(device)
    inputs = make_inputs(
        config,
        batch_size,
        device,
        seed,
        context_group_size=context_group_size,
        input_dtype=input_dtype,
    )
    point = {
        "status": "ok",
        "image_size": int(image_size),
        "grid_size": int(image_size) // 14,
        "target_tokens": (int(image_size) // 14) ** 2,
        "context_frames": int(context_frames),
        "context_tokens": int(context_frames) * (int(image_size) // 14) ** 2,
        "batch_size": int(batch_size),
        "unique_observations": int(batch_size) // int(context_group_size),
        "context_group_size": int(context_group_size),
        "pairwise_mac": pairwise_mac(model),
        "inference": run_inference(
            model,
            inputs,
            warmup=warmup,
            timed=timed,
            device=device,
            repeated_queries=repeated_queries,
            full_generation_timed=full_generation_timed,
            measure_full_generation=measure_full_generation,
        ),
        "training": None,
    }
    if include_training:
        point["training"] = run_training(
            model,
            inputs,
            warmup=warmup,
            timed=timed,
            device=device,
        )
    del inputs, model
    gc.collect()
    torch.cuda.empty_cache()
    return point


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "cuda"), default="cuda")
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument(
        "--scenarios",
        default="224x4,224x8,224x16,224x32",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-group-size", type=int, default=1)
    parser.add_argument(
        "--input-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=5)
    parser.add_argument(
        "--repeated-queries",
        default="1,5,10,20,50",
        help="Comma-separated repeated-query counts measured directly.",
    )
    parser.add_argument("--full-generation-timed", type=int, default=2)
    parser.add_argument(
        "--measure-full-generation", type=int, choices=(0, 1), default=1
    )
    parser.add_argument("--include-training", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="history-mixer-systems")
    args = parser.parse_args()
    if (
        args.batch_size <= 0
        or args.context_group_size <= 0
        or args.warmup_steps <= 0
        or args.timed_steps <= 0
    ):
        raise ValueError("batch, warmup, and timed counts must be positive")
    if args.batch_size % args.context_group_size != 0:
        raise ValueError(
            "batch-size must be divisible by context-group-size"
        )
    input_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.input_dtype]
    repeated_queries = tuple(
        int(value) for value in str(args.repeated_queries).split(",")
    )
    if (
        not repeated_queries
        or any(value <= 0 for value in repeated_queries)
        or len(set(repeated_queries)) != len(repeated_queries)
        or args.full_generation_timed <= 0
    ):
        raise ValueError("repeated-query counts and full-generation-timed must be positive")

    variants = parse_variants(args.variant)
    scenarios = parse_scenarios(args.scenarios)
    output = args.output.resolve()
    payload = {
        "schema_version": 1,
        "run_id": str(args.run_id),
        "status": "validating",
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "project_root": str(PROJECT_ROOT),
        "mode": str(args.mode),
        "batch_size": int(args.batch_size),
        "unique_observations": (
            int(args.batch_size) // int(args.context_group_size)
        ),
        "context_group_size": int(args.context_group_size),
        "input_dtype": str(args.input_dtype),
        "warmup_steps": int(args.warmup_steps),
        "timed_steps": int(args.timed_steps),
        "repeated_queries": list(repeated_queries),
        "full_generation_timed": int(args.full_generation_timed),
        "measure_full_generation": bool(args.measure_full_generation),
        "precision": "bf16_autocast",
        "include_training": bool(args.include_training),
        "scenarios": [
            {"image_size": image_size, "context_frames": context}
            for image_size, context in scenarios
        ],
        "comparison_scope": [
            "synthetic_model_core_training_step",
            "context_preparation",
            "cached_denoiser_forward",
        ],
        "excluded_scope": [
            "tokenizer",
            "decoder",
            "data_io",
            "ddp",
            "rollout_quality",
        ],
        "spatial_resolution_contract": (
            "The benchmark fixes a 16x16=256-token spatial grid and varies "
            "context length only. Tokenizer and decoder execution are excluded."
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "variants": [
            variant_metadata(label, config_path)
            for label, config_path in variants
        ],
    }
    atomic_json(payload, output)
    if args.mode == "validate":
        payload["status"] = "validated"
        payload["finished_at_utc"] = utc_now()
        atomic_json(payload, output)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "CUDA mode requires exactly one visible GPU, got "
            f"{torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    payload["environment"]["gpu"] = properties.name
    payload["environment"]["gpu_total_memory_gib"] = (
        properties.total_memory / 1024**3
    )
    payload["status"] = "running"
    atomic_json(payload, output)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    for metadata, (_, config_path) in zip(payload["variants"], variants):
        for image_size, context_frames in scenarios:
            try:
                point = run_point(
                    config_path=config_path,
                    image_size=image_size,
                    context_frames=context_frames,
                    batch_size=args.batch_size,
                    context_group_size=args.context_group_size,
                    input_dtype=input_dtype,
                    warmup=args.warmup_steps,
                    timed=args.timed_steps,
                    include_training=bool(args.include_training),
                    device=device,
                    seed=args.seed,
                    repeated_queries=repeated_queries,
                    full_generation_timed=args.full_generation_timed,
                    measure_full_generation=bool(args.measure_full_generation),
                )
            except torch.OutOfMemoryError as error:
                point = {
                    "status": "oom",
                    "image_size": image_size,
                    "context_frames": context_frames,
                    "error": str(error).splitlines()[0],
                }
            gc.collect()
            torch.cuda.empty_cache()
            metadata["points"].append(point)
            atomic_json(payload, output)

    payload["status"] = "completed"
    payload["finished_at_utc"] = utc_now()
    atomic_json(payload, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
