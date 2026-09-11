from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from .settings import Paths
from .utils import chunks, stage_manifest


def _rows(path: Path) -> list[dict]: return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

def _segment(text: str) -> list[str]:
    return text.casefold().split()

def _embed(texts: list[str], model_path: Path, max_length: int, batch_size: int = 32) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16).cuda().eval()
    vectors = []
    with torch.inference_mode():
        for batch in tqdm(list(chunks(texts, batch_size)), desc="BGE-M3 embeddings", unit="batch"):
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
            vector = model(**encoded).last_hidden_state[:, 0]
            vector = torch.nn.functional.normalize(vector.float(), p=2, dim=1)
            vectors.append(vector.cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(vectors, axis=0)

def build_index(config: dict, paths: Paths, force: bool = False) -> Path:
    run = paths.runs / config["project"]["run_name"]
    stage = run / "04_build_index"
    if (stage / "manifest.json").exists() and not force: return stage
    stage.mkdir(parents=True, exist_ok=True)
    docs = _rows(run / "02_prepare_data" / "documents.jsonl")
    texts = [row["text"] for row in docs]
    tokens = [_segment(text) for text in tqdm(texts, desc="BM25 tokenization", unit="doc")]
    bm25 = BM25Okapi(tokens)
    with (paths.indexes / "vihealthqa_bm25.pkl").open("wb") as handle: pickle.dump(bm25, handle)
    (paths.indexes / "vihealthqa_documents.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in docs) + "\n", encoding="utf-8")
    embeddings = _embed(texts, paths.models / "bge_m3", config["retrieval"]["embedding_max_length"])
    np.save(paths.indexes / "vihealthqa_bge_m3_embeddings.npy", embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1]); index.add(embeddings)
    faiss.write_index(index, str(paths.indexes / "vihealthqa_bge_m3.faiss"))
    stage_manifest(stage, config=config, inputs={"documents": len(docs)}, outputs={"embedding_shape": list(embeddings.shape), "faiss": "exact_IndexFlatIP", "bm25_tokenizer": "unicode_whitespace"})
    return stage

def _rank(scores: np.ndarray) -> list[int]: return np.argsort(-scores, kind="stable").astype(int).tolist()

def _metrics(ranking: list[int], gold: int) -> dict:
    rank = ranking.index(gold) + 1
    return {"rank": rank, "rr": 1.0 / rank, **{f"hit@{k}": float(rank <= k) for k in (1, 3, 5, 10)}}

def _rerank(query: str, candidate_ids: list[int], docs: list[dict], tokenizer, model) -> list[int]:
    scores = []
    with torch.inference_mode():
        for group in chunks(candidate_ids, 16):
            encoded = tokenizer([(query, docs[idx]["text"]) for idx in group], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            values = model(**encoded).logits.reshape(-1).float().cpu().tolist()
            scores.extend(zip(group, values))
    return [idx for idx, _ in sorted(scores, key=lambda pair: -pair[1])]

def evaluate_retrieval(config: dict, paths: Paths, force: bool = False) -> Path:
    run = paths.runs / config["project"]["run_name"]
    stage = run / "05_retrieval_eval"
    if (stage / "manifest.json").exists() and not force: return stage
    stage.mkdir(parents=True, exist_ok=True)
    docs = _rows(paths.indexes / "vihealthqa_documents.jsonl")
    doc_to_idx = {row["doc_id"]: idx for idx, row in enumerate(docs)}
    eval_rows = _rows(run / "02_prepare_data" / "evaluation.jsonl")
    with (paths.indexes / "vihealthqa_bm25.pkl").open("rb") as handle: bm25 = pickle.load(handle)
    dense_docs = np.load(paths.indexes / "vihealthqa_bge_m3_embeddings.npy")
    query_vectors = _embed([row["question"] for row in eval_rows], paths.models / "bge_m3", config["retrieval"]["embedding_max_length"])
    rerank_tok = AutoTokenizer.from_pretrained(paths.models / "bge_reranker_v2_m3", local_files_only=True)
    reranker = AutoModelForSequenceClassification.from_pretrained(paths.models / "bge_reranker_v2_m3", local_files_only=True, torch_dtype=torch.bfloat16).cuda().eval()
    results = {key: [] for key in ("bm25", "bge_m3", "hybrid_rrf", "hybrid_reranker")}
    per_query = {key: [] for key in results}
    k = config["retrieval"]["rrf_k"]; rerank_depth = config["retrieval"]["rerank_depth"]
    for row, vector in tqdm(zip(eval_rows, query_vectors), total=len(eval_rows), desc="Retrieval ablation", unit="query"):
        bm_rank = _rank(np.asarray(bm25.get_scores(_segment(row["question"]))))
        dense_rank = _rank(dense_docs @ vector)
        rrf = np.zeros(len(docs), dtype=np.float32)
        for rank, idx in enumerate(bm_rank, 1): rrf[idx] += 1.0 / (k + rank)
        for rank, idx in enumerate(dense_rank, 1): rrf[idx] += 1.0 / (k + rank)
        hybrid = _rank(rrf)
        reranked = _rerank(row["question"], hybrid[:rerank_depth], docs, rerank_tok, reranker)
        reranked_set = set(reranked)
        reranked += [idx for idx in hybrid if idx not in reranked_set]
        gold = doc_to_idx[row["gold_doc_id"]]
        for name, ranking in (("bm25", bm_rank), ("bge_m3", dense_rank), ("hybrid_rrf", hybrid), ("hybrid_reranker", reranked)):
            metric = _metrics(ranking, gold); metric["example_id"] = row["example_id"]
            per_query[name].append(metric)
            results[name].append({"example_id": row["example_id"], "ranked_doc_ids": [docs[idx]["doc_id"] for idx in ranking[:100]], **metric})
    del reranker; torch.cuda.empty_cache()
    summary = {}
    for name, values in per_query.items(): summary[name] = {key: float(np.mean([item[key] for item in values])) for key in ("rr", "hit@1", "hit@3", "hit@5", "hit@10")}
    for name, values in results.items(): (stage / f"{name}_rankings.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in values) + "\n", encoding="utf-8")
    (stage / "retrieval_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    stage_manifest(stage, config=config, inputs={"evaluation_rows": len(eval_rows), "documents": len(docs)}, outputs=summary)
    return stage
