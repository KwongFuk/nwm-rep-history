#!/usr/bin/env python3
"""Apply and verify the two runtime-compatibility edits to the pinned PAE source tree.

Usage: python tools/patch_pae_source.py --source-root $PAE_SOURCE_ROOT --asset-manifest config/assets/pae.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


REGISTRY_OLD = "\n".join(
    (
        "# Import modules that perform registration on import.",
        "from . import dinov2" + "  ",
        "from . import dinov3",
        "from . import siglip2",
        "from . import mae",
        "from . import internvit",
        "from .delta import DeltaEncoder",
    )
)

REGISTRY_NEW = """# Import modules that perform registration on import. Keep optional encoders
# best-effort so a broken local Transformers lazy import for unused backbones
# does not prevent DINOv2 evaluation.
for _module in ("dinov2", "dinov3", "siglip2", "mae", "internvit"):
    try:
        __import__(f"{__name__}.{_module}", fromlist=[_module])
    except ImportError as exc:
        print(f"[PAE encoder registry] skipped optional encoder {_module}: {exc}")
from .delta import DeltaEncoder
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--asset-manifest", required=True)
    args = parser.parse_args()

    root = Path(args.source_root).resolve()
    manifest = json.loads(
        Path(args.asset_manifest).resolve().read_text(encoding="utf-8")
    )

    registry = root / "tokenizer/encoders/__init__.py"
    registry_text = registry.read_text(encoding="utf-8")
    if REGISTRY_OLD not in registry_text:
        raise RuntimeError(f"Pinned PAE registry source no longer matches: {registry}")
    atomic_write(registry, registry_text.replace(REGISTRY_OLD, REGISTRY_NEW, 1))

    dinov2 = root / "tokenizer/encoders/dinov2.py"
    dinov2_text = dinov2.read_text(encoding="utf-8")
    old_import = "from transformers import Dinov2WithRegistersModel"
    new_import = (
        "from transformers.models.dinov2_with_registers."
        "modeling_dinov2_with_registers import Dinov2WithRegistersModel"
    )
    if dinov2_text.count(old_import) != 1:
        raise RuntimeError(f"Pinned PAE DINOv2 import no longer matches: {dinov2}")
    dinov2_text = dinov2_text.replace(old_import, new_import, 1)
    if not dinov2_text.endswith("\n\n"):
        raise RuntimeError(f"Pinned PAE DINOv2 ending no longer matches: {dinov2}")
    atomic_write(dinov2, dinov2_text[:-1])

    expected = manifest["pae_source"]["patched_file_sha256"]
    observed = {
        relative: sha256_file(root / relative)
        for relative in sorted(expected)
    }
    mismatched = {
        relative: {"expected": expected[relative], "observed": value}
        for relative, value in observed.items()
        if value != expected[relative]
    }
    if mismatched:
        raise RuntimeError(f"Patched PAE source hash mismatch: {mismatched}")
    print(json.dumps({"pae_runtime_compat": observed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
