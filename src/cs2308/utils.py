from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def stage_manifest(stage_dir: Path, *, config: dict[str, Any], inputs: dict[str, Any], outputs: dict[str, Any], status: str = "complete") -> None:
    write_json(stage_dir / "manifest.json", {
        "status": status,
        "created_at_unix": time.time(),
        "config_hash": stable_hash(json.dumps(config, sort_keys=True)),
        "inputs": inputs,
        "outputs": outputs,
    })


def is_complete(stage_dir: Path) -> bool:
    manifest = stage_dir / "manifest.json"
    return manifest.exists() and read_json(manifest).get("status") == "complete"


def require_inside(root: Path, path: Path) -> Path:
    resolved_root, resolved_path = root.resolve(), path.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError(f"Path escapes project root: {resolved_path}")
    return resolved_path


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
