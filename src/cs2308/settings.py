from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def artifacts(self) -> Path: return self.root / "artifacts"
    @property
    def datasets(self) -> Path: return self.artifacts / "datasets"
    @property
    def models(self) -> Path: return self.artifacts / "models"
    @property
    def indexes(self) -> Path: return self.artifacts / "indexes"
    @property
    def runs(self) -> Path: return self.root / "runs"
    @property
    def reports(self) -> Path: return self.root / "reports"
    @property
    def cache(self) -> Path: return self.root / ".cache"

    def ensure(self) -> None:
        for path in (self.artifacts, self.datasets, self.models, self.indexes, self.runs, self.reports, self.cache):
            path.mkdir(parents=True, exist_ok=True)

    def configure_environment(self) -> None:
        cache = self.cache
        values = {
            "HF_HOME": cache / "huggingface",
            "HF_DATASETS_CACHE": cache / "huggingface" / "datasets",
            "TRANSFORMERS_CACHE": cache / "huggingface" / "hub",
            "TORCH_HOME": cache / "torch",
            "TRITON_CACHE_DIR": cache / "triton",
            "XDG_CACHE_HOME": cache / "xdg",
            "TMPDIR": cache / "tmp",
            "TEMP": cache / "tmp",
            "TMP": cache / "tmp",
        }
        for key, value in values.items():
            value.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(value)
        # The base Runpod image enables hf_transfer globally, but the project does
        # not depend on it. Disable the inherited switch for portable downloads.
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


def load_config(path: str | Path) -> tuple[dict[str, Any], Paths]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = config_path.parents[1]
    paths = Paths(root)
    paths.ensure()
    paths.configure_environment()
    return config, paths
