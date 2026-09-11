from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from .settings import Paths
from .utils import set_seed, stable_hash, stage_manifest, write_json


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


def record_id(split: str, row: dict) -> str:
    payload = normalize(row["question"]) + "\n" + normalize(row["answer"])
    return f"{split}:{stable_hash(payload)[:20]}"


def _load_local(paths: Paths) -> tuple[Dataset, DatasetDict]:
    ft_files = list((paths.datasets / "raw" / "vietnamese_medical_qa").rglob("*.parquet"))
    if not ft_files:
        raise FileNotFoundError("Fine-tune parquet snapshot is missing")
    finetune = load_dataset("parquet", data_files=[str(item) for item in ft_files], split="train")
    rag_root = paths.datasets / "raw" / "vihealthqa"
    csvs = list(rag_root.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError("ViHealthQA CSV snapshot is missing")
    mapping = {}
    for item in csvs:
        lower = item.name.lower()
        if "train" in lower: mapping["train"] = str(item)
        elif "valid" in lower or "val" in lower or "dev" in lower: mapping["validation"] = str(item)
        elif "test" in lower: mapping["test"] = str(item)
    if set(mapping) != {"train", "validation", "test"}:
        raise ValueError(f"Cannot identify ViHealthQA splits from {[x.name for x in csvs]}")
    return finetune, load_dataset("csv", data_files=mapping)


def _near_duplicate_indices(train_questions: list[str], test_questions: list[str], threshold: float) -> set[int]:
    """Conservative char n-gram cosine candidate filter; exact duplicates are handled separately."""
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, norm="l2", dtype=np.float32)
    test_matrix = vectorizer.fit_transform(test_questions)
    train_matrix = vectorizer.transform(train_questions)
    flagged: set[int] = set()
    for start in tqdm(range(0, train_matrix.shape[0], 256), desc="Near-duplicate audit", unit="batch"):
        similarity = train_matrix[start:start + 256] @ test_matrix.T
        rows = np.where(np.asarray(similarity.max(axis=1).todense()).ravel() >= threshold)[0]
        flagged.update(start + int(row) for row in rows)
    return flagged


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare_data(config: dict, paths: Paths, force: bool = False) -> Path:
    stage = paths.runs / config["project"]["run_name"] / "02_prepare_data"
    if (stage / "manifest.json").exists() and not force:
        return stage
    stage.mkdir(parents=True, exist_ok=True)
    set_seed(config["project"]["seed"])
    finetune, rag = _load_local(paths)
    test_rows = [{**row, "question": normalize(row["question"]), "answer": normalize(row["answer"])} for row in rag["test"]]
    ft_rows = [{"question": normalize(row["question"]), "answer": normalize(row["answer"])} for row in finetune]
    ft_rows = [row for row in ft_rows if row["question"] and row["answer"]]
    test_q = {row["question"].casefold() for row in test_rows}
    exact = {index for index, row in enumerate(ft_rows) if row["question"].casefold() in test_q}
    near = _near_duplicate_indices([row["question"] for row in ft_rows], [row["question"] for row in test_rows], config["data"]["near_duplicate_jaccard"])
    clean = [row for index, row in enumerate(ft_rows) if index not in exact and index not in near]
    rng = np.random.default_rng(config["project"]["seed"])
    order = rng.permutation(len(clean))
    train_size, validation_size = config["data"]["train_size"], config["data"]["finetune_validation_size"]
    if len(clean) < train_size:
        raise RuntimeError(f"Only {len(clean)} decontaminated rows remain; required {train_size}")
    train_rows = [clean[int(i)] for i in order[:train_size]]
    validation_rows = [clean[int(i)] for i in order[train_size:train_size + validation_size]]
    evaluation_rows = [test_rows[int(i)] for i in rng.permutation(len(test_rows))[:config["data"]["evaluation_size"]]]
    documents: list[dict] = []
    for split, dataset in rag.items():
        for row in dataset:
            answer = normalize(row["answer"])
            question = normalize(row["question"])
            documents.append({"doc_id": record_id(split, {"question": question, "answer": answer}), "split": split,
                              "source_id": int(row["id"]), "text": answer, "link": row.get("link", ""), "content_hash": stable_hash(answer)})
    gold = {(normalize(row["question"]), normalize(row["answer"])): record_id("test", row) for row in test_rows}
    for row in evaluation_rows:
        row["example_id"] = record_id("test", row)
        row["gold_doc_id"] = gold[(row["question"], row["answer"])]
    _write_jsonl(stage / "finetune_train.jsonl", train_rows)
    _write_jsonl(stage / "finetune_validation.jsonl", validation_rows)
    _write_jsonl(stage / "evaluation.jsonl", evaluation_rows)
    _write_jsonl(stage / "documents.jsonl", documents)
    write_json(stage / "contamination_report.json", {"raw_finetune_rows": len(ft_rows), "exact_question_matches": len(exact), "near_question_matches": len(near), "clean_rows": len(clean), "test_rows": len(test_rows)})
    stage_manifest(stage, config=config, inputs={"finetune_rows": len(finetune), "vihealthqa_splits": {key: len(value) for key, value in rag.items()}}, outputs={"train": len(train_rows), "validation": len(validation_rows), "evaluation": len(evaluation_rows), "documents": len(documents)})
    return stage
