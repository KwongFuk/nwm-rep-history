"""Audited model-checkpoint loading shared by evaluation entrypoints.

The released RAE-NWM checkpoint and local training checkpoints may store the
same state under ``ema`` or ``model`` and may carry ``torch.compile`` prefixes.
This module normalizes only those known wrappers and rejects learned tensor
gaps instead of silently leaving parameters randomly initialized.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple, Union

import torch
from torch import nn


KNOWN_STATE_PREFIXES: Tuple[str, ...] = ("_orig_mod.", "module.")


@dataclass(frozen=True)
class CheckpointLoadReport:
    """Machine-readable account of one model state load."""

    state_key: str
    tensor_count: int
    missing_parameters: Tuple[str, ...]
    missing_buffers: Tuple[str, ...]
    unexpected_keys: Tuple[str, ...]
    adapted_keys: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_checkpoint_file(path: Union[str, Path]) -> Any:
    """Load a checkpoint on CPU, using memory mapping when supported."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def _strip_known_prefixes(name: str) -> str:
    normalized = str(name)
    changed = True
    while changed:
        changed = False
        for prefix in KNOWN_STATE_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
    return normalized


def normalize_state_dict_keys(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove known DDP/compile wrappers and reject normalization collisions."""

    normalized: Dict[str, Any] = {}
    for raw_name, value in state.items():
        name = _strip_known_prefixes(raw_name)
        if name in normalized:
            raise ValueError(
                f"Checkpoint key collision after prefix normalization: {raw_name!r} -> {name!r}"
            )
        normalized[name] = value
    return normalized


def extract_model_state(
    checkpoint: Any,
    *,
    preferred_state_key: str = "ema",
) -> Tuple[Dict[str, Any], str]:
    """Extract a tensor state dict from a training checkpoint or raw state dict."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint).__name__}")

    candidates = [preferred_state_key]
    candidates.extend(key for key in ("ema", "model", "state_dict") if key not in candidates)
    for key in candidates:
        value = checkpoint.get(key)
        if isinstance(value, Mapping) and value:
            state = normalize_state_dict_keys(value)
            if not all(torch.is_tensor(item) for item in state.values()):
                raise TypeError(f"Checkpoint field {key!r} is not a tensor state dict")
            return state, key

    if checkpoint and all(torch.is_tensor(item) for item in checkpoint.values()):
        return normalize_state_dict_keys(checkpoint), "raw"
    raise KeyError(
        f"Checkpoint has no non-empty model state among {tuple(candidates)}"
    )


def _matches_prefix(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == prefix.rstrip(".") or name.startswith(prefix) for prefix in prefixes)


def audit_incompatible_keys(
    model: nn.Module,
    *,
    missing_keys: Sequence[str],
    unexpected_keys: Sequence[str],
    allowed_missing_trainable_prefixes: Sequence[str] = (),
    allowed_missing_buffer_names: Sequence[str] = (),
    allowed_unexpected_prefixes: Sequence[str] = (),
    state_key: str = "unknown",
    tensor_count: int = 0,
    adapted_keys: Sequence[str] = (),
) -> CheckpointLoadReport:
    """Separate declared trainable additions from unsafe learned-key gaps."""

    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    missing_parameters = tuple(sorted(key for key in missing_keys if key in parameters))
    missing_buffers = tuple(sorted(key for key in missing_keys if key in buffers))
    unclassified_missing = tuple(
        sorted(key for key in missing_keys if key not in parameters and key not in buffers)
    )

    unsafe_parameters = []
    for key in missing_parameters:
        explicitly_allowed = _matches_prefix(key, allowed_missing_trainable_prefixes)
        if not explicitly_allowed or not parameters[key].requires_grad:
            unsafe_parameters.append(key)

    allowed_buffers = set(allowed_missing_buffer_names)
    unsafe_buffers = [key for key in missing_buffers if key not in allowed_buffers]
    unsafe_unexpected = [
        key for key in unexpected_keys if not _matches_prefix(key, allowed_unexpected_prefixes)
    ]
    if unsafe_parameters or unsafe_buffers or unclassified_missing or unsafe_unexpected:
        raise RuntimeError(
            "Unsafe checkpoint incompatibility: "
            f"missing_learned={unsafe_parameters}, missing_buffers={unsafe_buffers}, "
            f"unclassified_missing={list(unclassified_missing)}, "
            f"unexpected={unsafe_unexpected}"
        )

    return CheckpointLoadReport(
        state_key=state_key,
        tensor_count=int(tensor_count),
        missing_parameters=missing_parameters,
        missing_buffers=missing_buffers,
        unexpected_keys=tuple(sorted(unexpected_keys)),
        adapted_keys=tuple(sorted(adapted_keys)),
    )


def load_model_checkpoint(
    model: nn.Module,
    checkpoint: Union[str, Path, Mapping[str, Any]],
    *,
    preferred_state_key: str = "ema",
    allowed_missing_trainable_prefixes: Sequence[str] = (),
    allowed_missing_buffer_names: Sequence[str] = (),
    allowed_unexpected_prefixes: Sequence[str] = (),
    adapt_context_pos_embed: bool = False,
) -> CheckpointLoadReport:
    """Load a model state and fail on every undeclared learned-key mismatch."""

    checkpoint_object = (
        load_checkpoint_file(checkpoint)
        if isinstance(checkpoint, (str, Path))
        else checkpoint
    )
    state, state_key = extract_model_state(
        checkpoint_object,
        preferred_state_key=preferred_state_key,
    )
    adapted_keys = []
    if adapt_context_pos_embed and "pos_embed" in state:
        model_state = model.state_dict()
        checkpoint_pos = state["pos_embed"]
        model_pos = model_state.get("pos_embed")
        parameter = dict(model.named_parameters()).get("pos_embed")
        shape_can_adapt = (
            model_pos is not None
            and checkpoint_pos.ndim == 3
            and model_pos.ndim == 3
            and checkpoint_pos.shape[1:] == model_pos.shape[1:]
            and checkpoint_pos.shape[0] != model_pos.shape[0]
        )
        if shape_can_adapt:
            if parameter is None or parameter.requires_grad:
                raise RuntimeError(
                    "Context pos_embed adaptation is restricted to a frozen parameter"
                )
            if not torch.equal(
                checkpoint_pos,
                checkpoint_pos[:1].expand_as(checkpoint_pos),
            ):
                raise RuntimeError(
                    "Cannot adapt context pos_embed with frame-dependent values"
                )
            state = dict(state)
            state["pos_embed"] = model_pos
            adapted_keys.append("pos_embed")
    try:
        incompatible = model.load_state_dict(state, strict=False)
    except RuntimeError as error:
        raise RuntimeError(
            f"Checkpoint tensors are shape-incompatible with the instantiated model: {error}"
        ) from error
    return audit_incompatible_keys(
        model,
        missing_keys=incompatible.missing_keys,
        unexpected_keys=incompatible.unexpected_keys,
        allowed_missing_trainable_prefixes=allowed_missing_trainable_prefixes,
        allowed_missing_buffer_names=allowed_missing_buffer_names,
        allowed_unexpected_prefixes=allowed_unexpected_prefixes,
        state_key=state_key,
        tensor_count=len(state),
        adapted_keys=adapted_keys,
    )
