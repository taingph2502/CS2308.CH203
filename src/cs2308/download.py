from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm import tqdm

from .settings import Paths
from .utils import stage_manifest


def download_all(config: dict, paths: Paths, force: bool = False) -> Path:
    stage = paths.runs / config["project"]["run_name"] / "01_download"
    if (stage / "manifest.json").exists() and not force:
        return stage
    stage.mkdir(parents=True, exist_ok=True)
    sources = config["sources"]
    entries = [
        ("finetune_dataset", "dataset", paths.datasets / "raw" / "vietnamese_medical_qa"),
        ("rag_dataset", "dataset", paths.datasets / "raw" / "vihealthqa"),
        ("base_model", "model", paths.models / "qwen2_5_3b_instruct"),
        ("embedder", "model", paths.models / "bge_m3"),
        ("reranker", "model", paths.models / "bge_reranker_v2_m3"),
        ("phobert", "model", paths.models / "phobert_base_v2"),
    ]
    results = {}
    for key, repo_type, destination in tqdm(entries, desc="HF snapshots", unit="repo"):
        source = sources[key]
        destination.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=source["repo_id"], repo_type=repo_type, revision=source["revision"],
            local_dir=destination, local_dir_use_symlinks=False,
        )
        results[key] = {"repo_id": source["repo_id"], "revision": source["revision"], "path": str(destination),
                        "files": sum(1 for item in destination.rglob("*") if item.is_file())}
    stage_manifest(stage, config=config, inputs={"sources": sources}, outputs=results)
    return stage
