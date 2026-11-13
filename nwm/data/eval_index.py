"""Helpers for campaign-local, immutable evaluation indices."""

from __future__ import annotations

from pathlib import Path


def resolve_eval_index(
    eval_index_dir: str | None,
    dataset_name: str,
    eval_kind: str,
) -> str | None:
    """Resolve a frozen per-dataset index, or return ``None`` when disabled."""

    if not eval_index_dir:
        return None
    if eval_kind not in {"openloop", "planning"}:
        raise ValueError(f"Unsupported evaluation index kind: {eval_kind!r}")
    path = (
        Path(eval_index_dir).expanduser().resolve()
        / f"{dataset_name}_{eval_kind}.pkl"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen evaluation index does not exist: {path}"
        )
    return str(path)
