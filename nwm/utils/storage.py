"""Optional runtime checks for heavy-artifact output paths.

Checkpoints and evaluation outputs can be large. On shared clusters it is
useful to refuse writing them to a quota-limited home directory. Set
``RAENWM_DURABLE_ROOTS`` (``os.pathsep`` separated) to the allowed storage
roots to enable the check; when it is unset, every path is accepted.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def durable_roots(extra_roots: Iterable[str | Path] = ()) -> tuple[Path, ...]:
    configured = [item for item in os.environ.get("RAENWM_DURABLE_ROOTS", "").split(os.pathsep) if item]
    values = [*(Path(item) for item in configured), *(Path(item) for item in extra_roots)]
    roots: list[Path] = []
    for value in values:
        resolved = value.expanduser().resolve()
        if resolved not in roots:
            roots.append(resolved)
    return tuple(roots)


def is_durable_path(path: str | Path, *, roots: Iterable[str | Path] = ()) -> bool:
    allowed = durable_roots(roots)
    if not allowed:
        return True
    candidate = Path(path).expanduser().resolve()
    return any(candidate == root or candidate.is_relative_to(root) for root in allowed)


def require_durable_path(
    path: str | Path,
    *,
    label: str,
    roots: Iterable[str | Path] = (),
) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not is_durable_path(candidate, roots=roots):
        allowed = ", ".join(str(root) for root in durable_roots(roots))
        raise ValueError(f"{label} must be under a durable root ({allowed}), got {candidate}")
    return candidate
