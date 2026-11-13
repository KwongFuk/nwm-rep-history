"""Configuration loading shared by training and evaluation entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _expand_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_paths(item) for item in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def _load_with_base(path: Path, seen: Set[Path]) -> Dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Config inheritance cycle detected at {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    base_ref = current.pop("base_config", None)
    if base_ref is None:
        merged = current
    else:
        base_path = Path(os.path.expandvars(os.path.expanduser(str(base_ref))))
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = _deep_merge(_load_with_base(base_path, seen), current)
    seen.remove(path)
    return merged


def load_config(path: str, defaults_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML with optional inheritance, deep merge, and env expansion."""

    user_config = _load_with_base(Path(path), set())
    if defaults_path is not None:
        defaults = _load_with_base(Path(defaults_path), set())
        user_config = _deep_merge(defaults, user_config)
    return _expand_paths(user_config)


def assert_config_not_deprecated(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
) -> None:
    """Fail closed when an explicitly deprecated route is selected.

    Deprecation metadata is deliberately kept in the YAML rather than inferred
    from a filename such as ``*-scratch.yaml``. Some historical scratch paths
    contain pinned upstream assets and remain valid.
    """

    lifecycle_value = config.get("lifecycle", {})
    lifecycle = (
        dict(lifecycle_value) if isinstance(lifecycle_value, Mapping) else {}
    )
    status = str(lifecycle.get("status", "active")).strip().lower()
    if status != "deprecated":
        return

    deprecation_id = str(
        lifecycle.get("deprecation_id", "unspecified-deprecation")
    )
    reason = str(lifecycle.get("reason", "no reason recorded"))
    replacement = str(lifecycle.get("replacement", "none recorded"))
    raise RuntimeError(
        f"Deprecated config refused: {Path(config_path)} "
        f"[{deprecation_id}]. Reason: {reason}. "
        f"Replacement: {replacement}."
    )
