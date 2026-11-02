"""One config file per run: everything needed to reproduce it (best practice: no hidden state)."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from rtod import coco


def _git_sha() -> str:
    if sha := os.environ.get("RTOD_SHA"):  # set on the pod, where there is no checkout
        return sha
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return done.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_manifest(out_dir: Path, args: dict[str, Any], subset: str) -> Path:
    import torch

    ids_file = coco.LISTS / f"{subset}.txt"
    manifest = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in args.items()},
        "git_sha": _git_sha(),
        "id_list": {"name": subset, "sha256": hashlib.sha256(ids_file.read_bytes()).hexdigest()},
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    path = out_dir / "run_config.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
