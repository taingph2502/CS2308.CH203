from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import py_vncorenlp
from bert_score import BERTScorer
from transformers import AutoTokenizer

from .retrieval import _embed
from .settings import Paths
from .utils import stage_manifest


def _rows(path: Path) -> list[dict]: return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

def _bootstrap(values: np.ndarray, count: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed); means = np.empty(count, dtype=np.float32)
    for index in range(count): means[index] = values[rng.integers(0, len(values), len(values))].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _load_phobert_segmenter(paths: Paths):
    """Load one PhoBERT-compatible VnCoreNLP word segmenter for this process."""
    root = paths.models / "vncorenlp"
    files = {
        "VnCoreNLP-1.2.jar": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.2.jar",
        "models/wordsegmenter/vi-vocab": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab",
        "models/wordsegmenter/wordsegmenter.rdr": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr",
    }
    for relative, url in files.items():
        destination = root / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(url, destination)
    return py_vncorenlp.VnCoreNLP(annotators=["wseg"], max_heap_size="-Xmx2g", save_dir=str(root))


def _phobert_segment(texts: list[str], segmenter) -> list[str]:
    return [" ".join(segmenter.word_segment(text)) for text in texts]


def _truncate_for_phobert(texts: list[str], paths: Paths) -> list[str]:
    """BERTScore does not apply encoder truncation; enforce PhoBERT's 256-token limit."""
    # Keep this tokenizer identical to BERTScore's tokenizer below.  The fast
    # implementation is also the only one for which we pre-validated all ids.
    tokenizer = AutoTokenizer.from_pretrained(paths.models / "phobert_base_v2", local_files_only=True, use_fast=True)
    return [
        tokenizer.decode(
            tokenizer(text, truncation=True, max_length=256, add_special_tokens=True)["input_ids"],
            skip_special_tokens=True,
        )
        for text in texts
    ]

def score_generation(config: dict, paths: Paths, force: bool = False) -> Path:
    run = paths.runs / config["project"]["run_name"]; stage = run / "07_generation_scores"
    if (stage / "manifest.json").exists() and not force: return stage
    stage.mkdir(parents=True, exist_ok=True)
    summary = {}
    segmenter = _load_phobert_segmenter(paths)
    # PhoBERT's tokenizer config has an effectively unlimited model_max_length.
    # BERTScore otherwise re-tokenizes the already-truncated strings without a
    # positional limit.  Use one scorer with its tokenizer explicitly bounded.
    phobert_scorer = BERTScorer(
        model_type=str(paths.models / "phobert_base_v2"),
        num_layers=config["evaluation"]["phobert_num_layers"], lang="vi",
        batch_size=8, device="cuda", use_fast_tokenizer=True,
    )
    phobert_scorer._tokenizer.model_max_length = 256
    for arm in ("qwen", "qwen_rag", "qwen_pissa", "qwen_pissa_rag"):
        rows = _rows(run / "06_generation" / f"{arm}.jsonl")
        pred, ref = [x["prediction"] for x in rows], [x["reference"] for x in rows]
        segmented = _phobert_segment(pred + ref, segmenter)
        segmented_pred, segmented_ref = segmented[:len(pred)], segmented[len(pred):]
        segmented_pred = _truncate_for_phobert(segmented_pred, paths)
        segmented_ref = _truncate_for_phobert(segmented_ref, paths)
        _, _, f1 = phobert_scorer.score(segmented_pred, segmented_ref, batch_size=8, verbose=True)
        pvec = _embed(pred, paths.models / "bge_m3", config["retrieval"]["embedding_max_length"])
        rvec = _embed(ref, paths.models / "bge_m3", config["retrieval"]["embedding_max_length"])
        cosine = np.sum(pvec * rvec, axis=1)
        values = {"phobert_bertscore_f1": f1.cpu().numpy(), "bge_m3_cosine": cosine}
        scored = [{**row, **{key: float(vector[index]) for key, vector in values.items()}} for index, row in enumerate(rows)]
        (stage / f"{arm}_scored.jsonl").write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in scored) + "\n", encoding="utf-8")
        summary[arm] = {key: {"mean": float(vector.mean()), "ci95": _bootstrap(vector, config["evaluation"]["bootstrap_samples"], config["project"]["seed"])} for key, vector in values.items()}
    (stage / "generation_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    stage_manifest(stage, config=config, inputs={"arms": 4}, outputs={**summary, "phobert_preprocessing": "VnCoreNLP_RDRSegmenter_then_256_token_truncation"})
    return stage
