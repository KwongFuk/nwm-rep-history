"""Atomic, full-state checkpoint helpers for exact training resume."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import re
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
import torch


REQUIRED_TRAINING_KEYS = {
    "model",
    "ema",
    "opt",
    "scheduler",
    "epoch",
    "batch_in_epoch",
    "train_steps",
    "rng_states_by_rank",
    "config",
    "config_sha256",
}

REQUIRED_EVALUATION_KEYS = {
    "ema",
    "epoch",
    "train_steps",
    "config",
    "config_sha256",
}


def canonical_config(config: Mapping[str, Any]) -> str:
    """Serialize a resolved config in a stable, hashable form."""

    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def config_fingerprint(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_config(config).encode("utf-8")).hexdigest()


def capture_rng_state() -> Dict[str, Any]:
    """Capture all process-local RNG streams used by this trainer."""

    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_cpu_state = state["torch_cpu"]
    if not torch.is_tensor(torch_cpu_state):
        raise TypeError("Saved torch_cpu RNG state is not a tensor")
    # Training checkpoints are loaded with ``map_location=device`` so model
    # tensors land directly on the GPU. That mapping also moves the CPU RNG
    # byte vector, but torch.set_rng_state strictly requires a CPU ByteTensor.
    torch.set_rng_state(torch_cpu_state.detach().to(device="cpu", dtype=torch.uint8))
    if "torch_cuda_all" in state and torch.cuda.is_available():
        cuda_states = state["torch_cuda_all"]
        if not isinstance(cuda_states, (list, tuple)) or not all(
            torch.is_tensor(value) for value in cuda_states
        ):
            raise TypeError("Saved torch_cuda_all RNG state is not a tensor sequence")
        torch.cuda.set_rng_state_all(
            [
                value.detach().to(device="cpu", dtype=torch.uint8)
                for value in cuda_states
            ]
        )


def validate_training_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_TRAINING_KEYS.difference(checkpoint))
    if missing:
        raise ValueError(
            "Checkpoint is not exactly resumable; missing keys: " + ", ".join(missing)
        )


def validate_evaluation_snapshot(snapshot: Mapping[str, Any]) -> None:
    """Validate an inference-only EMA snapshot.

    Evaluation snapshots deliberately omit optimizer, scheduler, scaler, and
    RNG state.  They must never be advertised as resumable checkpoints.
    """

    missing = sorted(REQUIRED_EVALUATION_KEYS.difference(snapshot))
    if missing:
        raise ValueError(
            "Evaluation snapshot is incomplete; missing keys: "
            + ", ".join(missing)
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def write_complete_checkpoint(
    checkpoint: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Atomically write a checkpoint and only then publish its marker."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        torch.save(dict(checkpoint), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)

    marker = destination.with_name(destination.name + ".complete")
    payload = {
        "checkpoint_name": destination.name,
        "kind": "training",
        "train_steps": int(checkpoint["train_steps"]),
        "config_sha256": str(checkpoint["config_sha256"]),
    }
    _atomic_write_text(marker, json.dumps(payload, sort_keys=True) + "\n")
    return destination


def write_complete_evaluation_snapshot(
    snapshot: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Atomically publish an EMA-only evaluation snapshot and marker."""

    validate_evaluation_snapshot(snapshot)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        torch.save(dict(snapshot), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)

    marker = destination.with_name(destination.name + ".complete")
    payload = {
        "snapshot_name": destination.name,
        "kind": "evaluation",
        "epoch": int(snapshot["epoch"]),
        "train_steps": int(snapshot["train_steps"]),
        "config_sha256": str(snapshot["config_sha256"]),
    }
    _atomic_write_text(marker, json.dumps(payload, sort_keys=True) + "\n")
    return destination


def publish_checkpoint_alias(
    step_checkpoint: str | Path,
    alias_name: str,
) -> Path:
    """Atomically point a named alias at a complete training checkpoint."""

    step_path = Path(step_checkpoint)
    if not is_complete_checkpoint(step_path):
        raise ValueError(f"Cannot publish incomplete checkpoint: {step_path}")
    alias = step_path.parent / str(alias_name)
    temporary = alias.with_name(f".{alias.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    os.symlink(step_path.name, temporary)
    os.replace(temporary, alias)
    _fsync_directory(alias.parent)

    step_marker = json.loads(
        step_path.with_name(step_path.name + ".complete").read_text(encoding="utf-8")
    )
    step_marker["checkpoint_name"] = alias.name
    step_marker["target"] = step_path.name
    _atomic_write_text(
        alias.with_name(alias.name + ".complete"),
        json.dumps(step_marker, sort_keys=True) + "\n",
    )
    return alias


def publish_latest_checkpoint(step_checkpoint: str | Path) -> Path:
    """Atomically point ``latest.pth.tar`` at a complete step checkpoint."""

    return publish_checkpoint_alias(step_checkpoint, "latest.pth.tar")


def is_complete_checkpoint(path: str | Path) -> bool:
    candidate = Path(path)
    marker = candidate.with_name(candidate.name + ".complete")
    if not candidate.is_file() or not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("kind", "training") != "training":
        return False
    if payload.get("checkpoint_name") != candidate.name:
        return False
    target = payload.get("target")
    if target is not None and not (candidate.parent / str(target)).is_file():
        return False
    return True


def is_complete_evaluation_snapshot(path: str | Path) -> bool:
    candidate = Path(path)
    marker = candidate.with_name(candidate.name + ".complete")
    if not candidate.is_file() or not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("kind") == "evaluation"
        and payload.get("snapshot_name") == candidate.name
        and isinstance(payload.get("epoch"), int)
        and isinstance(payload.get("train_steps"), int)
    )


def latest_complete_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """Return latest complete checkpoint, ignoring partial newer files."""

    directory = Path(checkpoint_dir)
    latest = directory / "latest.pth.tar"
    if is_complete_checkpoint(latest):
        return latest

    candidates = []
    pattern = re.compile(r"checkpoint_step_(\d+)\.pth\.tar$")
    for candidate in directory.glob("checkpoint_step_*.pth.tar"):
        match = pattern.match(candidate.name)
        if match and is_complete_checkpoint(candidate):
            candidates.append((int(match.group(1)), candidate))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def latest_readable_complete_checkpoint(
    checkpoint_dir: str | Path,
    *,
    validator: Optional[Callable[[Mapping[str, Any]], None]] = validate_training_checkpoint,
) -> Optional[Path]:
    """Return the newest readable complete checkpoint, falling back safely.

    A complete marker proves that the writer reached its publish boundary, but
    it cannot protect a file from later storage corruption.  Candidates are
    therefore deserialized and validated from newest to oldest.
    """

    directory = Path(checkpoint_dir)
    pattern = re.compile(r"checkpoint_step_(\d+)\.pth\.tar$")
    candidates = []
    latest = directory / "latest.pth.tar"
    if is_complete_checkpoint(latest):
        target = latest.resolve() if latest.is_symlink() else latest
        match = pattern.match(target.name)
        candidates.append((int(match.group(1)) if match else 2**63 - 1, latest))
    for candidate in directory.glob("checkpoint_step_*.pth.tar"):
        match = pattern.match(candidate.name)
        if match and is_complete_checkpoint(candidate):
            candidates.append((int(match.group(1)), candidate))

    seen = set()
    for _, candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
        identity = str(candidate.resolve())
        if identity in seen:
            continue
        seen.add(identity)
        try:
            state = torch.load(candidate, map_location="cpu", weights_only=False)
            if not isinstance(state, Mapping):
                raise TypeError("checkpoint root is not a mapping")
            if validator is not None:
                validator(state)
        except (OSError, RuntimeError, TypeError, ValueError, EOFError, pickle.UnpicklingError):
            continue
        return candidate
    return None
