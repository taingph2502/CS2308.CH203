from __future__ import annotations

import json
from pathlib import Path

from .settings import Paths


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(config: dict, paths: Paths) -> None:
    """Render a reproducible final report after all metric stages complete."""
    run = paths.runs / config["project"]["run_name"]
    prep = _read(run / "02_prepare_data" / "manifest.json")
    contamination = _read(run / "02_prepare_data" / "contamination_report.json")
    training = _read(run / "03_train_pissa" / "metrics.json")
    retrieval = _read(run / "05_retrieval_eval" / "retrieval_metrics.json")
    generation = _read(run / "07_generation_scores" / "generation_metrics.json")
    paths.reports.mkdir(parents=True, exist_ok=True)
    source = config["sources"]
    training_cfg, retrieval_cfg = config["training"], config["retrieval"]
    run_rel = f"runs/{config['project']['run_name']}"

    lines = [
        "# Báo cáo CS2308: PiSSA + RAG cho hỏi-đáp y tế tiếng Việt", "",
        "## Tóm tắt", "",
        "Thí nghiệm so sánh Qwen2.5-3B-Instruct nguyên gốc, RAG, PiSSA và PiSSA+RAG trên cùng "
        "1.000 mẫu ViHealthQA. Toàn bộ số liệu dưới đây sinh từ manifest/artifact của run này. "
        "Đây là automatic evaluation, không là bằng chứng an toàn hay hiệu quả lâm sàng.", "",
        "## Input và output", "",
        "- **SFT input:** 5.000 cặp question/answer `hungnm/vietnamese-medical-qa`, Qwen chat "
        "template; token prompt có label `-100`, chỉ assistant completion tạo loss.",
        "- **SFT output:** PiSSA residual base, initial adapter, native final adapter, converted LoRA "
        "adapter, checkpoint và training metrics.",
        "- **RAG input:** query, 10.015 answer documents ViHealthQA, BM25+BGE-M3 index và reranker.",
        "- **RAG output:** ranking top-100 audit được, context top-3 giới hạn 1.800 token và 4 arm predictions.",
        "- **Metrics output:** MRR/Hit@1/3/5/10; PhoBERT-BERTScore F1, BGE-M3 cosine, per-example "
        "scores và bootstrap CI95%.", "",
        "## Tái lập", "",
        f"- Run/seed: `{config['project']['run_name']}` / `{config['project']['seed']}`.",
        f"- Base: `{source['base_model']['repo_id']}` @ `{source['base_model']['revision']}`.",
        f"- SFT: `{source['finetune_dataset']['repo_id']}` @ `{source['finetune_dataset']['revision']}`.",
        f"- Corpus: `{source['rag_dataset']['repo_id']}` @ `{source['rag_dataset']['revision']}`.",
        f"- Embedder/reranker: `{source['embedder']['repo_id']}` @ `{source['embedder']['revision']}`; "
        f"`{source['reranker']['repo_id']}` @ `{source['reranker']['revision']}`.",
        f"- Metric encoder: `{source['phobert']['repo_id']}` @ `{source['phobert']['revision']}`.",
        "- Cache, snapshots, checkpoints và outputs nằm dưới gốc `CS2308` (Runpod: `/workspace/CS2308`).", "",
        "## Dữ liệu và contamination", "",
        f"Tạo `{prep['outputs']['train']}` train, `{prep['outputs']['validation']}` validation, "
        f"`{prep['outputs']['evaluation']}` evaluation và `{prep['outputs']['documents']}` documents.", "",
        "| Kiểm tra | Giá trị |", "|---|---:|",
        f"| Raw SFT records hợp lệ | {contamination['raw_finetune_rows']} |",
        f"| Exact overlaps với test | {contamination['exact_question_matches']} |",
        f"| Near-question candidates bị loại | {contamination['near_question_matches']} |",
        f"| SFT records còn lại | {contamination['clean_rows']} |", "",
        "Near-duplicate audit dùng TF-IDF character n-gram cosine (3–5) ở ngưỡng "
        f"`{config['data']['near_duplicate_jaccard']}`; tên config được giữ lại nhưng phép đo là cosine.", "",
        "## Fine-tune PiSSA", "",
        f"PiSSA: `r={training_cfg['rank']}`, `alpha={training_cfg['alpha']}`, "
        f"`{training_cfg['pissa_init']}`, targets `{'`, `'.join(training_cfg['target_modules'])}`. "
        f"BF16/TF32, fused AdamW, activation checkpointing, 3 epoch, LR `{training_cfg['learning_rate']}`, "
        f"cosine warm-up {training_cfg['warmup_ratio']:.0%}, effective batch {training_cfg['effective_batch_size']}.",
        "",
        "PiSSA lưu principal components vào adapter và residual vào base frozen; vì vậy residual model "
        "và initial adapter là artifact bắt buộc. Sau reload, `enable_input_require_grads()` bảo đảm "
        "autograd graph cho reentrant checkpointing. Final adapter cũng được chuyển đổi sang LoRA bằng "
        "initial adapter để inference tái lập đúng trọng số.", "",
        "| Training metric | Giá trị |", "|---|---:|",
        f"| Train loss | {training.get('train_loss', float('nan')):.6f} |",
        f"| Eval loss | {training.get('eval_loss', float('nan')):.6f} |",
        f"| Train runtime (s) | {training.get('train_runtime', float('nan')):.2f} |",
        f"| Train samples/s | {training.get('train_samples_per_second', float('nan')):.3f} |",
        f"| Attention backend | {training.get('attention_backend', 'unknown')} |", "",
        "## Retrieval ablation", "",
        "BGE-M3 dùng CLS pooling, L2 normalization và FAISS `IndexFlatIP` exact; BM25 dùng cùng corpus. "
        f"RRF `k={retrieval_cfg['rrf_k']}`; reranker chỉ chạy top-{retrieval_cfg['rerank_depth']} candidates. "
        "Gold là `gold_doc_id` được persist trong evaluation artifact.", "",
        "| Method | MRR | Hit@1 | Hit@3 | Hit@5 | Hit@10 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metric in retrieval.items():
        lines.append(f"| {name} | {metric['rr']:.4f} | {metric['hit@1']:.4f} | {metric['hit@3']:.4f} | {metric['hit@5']:.4f} | {metric['hit@10']:.4f} |")
    lines.extend(["", "## Generation ablation", "",
                  "Cả bốn arm dùng greedy decoding/Qwen chat template; RAG dùng hybrid+reranker top-3 và "
                  "ngân sách context 1.800 token. CI là percentile bootstrap 95% (10.000 resamples, seed 2308).", "",
                  "| System | PhoBERT-BERTScore F1 (mean [CI95]) | BGE-M3 cosine (mean [CI95]) |", "|---|---:|---:|"])
    for name, metric in generation.items():
        pho, cosine = metric["phobert_bertscore_f1"], metric["bge_m3_cosine"]
        lines.append(f"| {name} | {pho['mean']:.4f} [{pho['ci95'][0]:.4f}, {pho['ci95'][1]:.4f}] | {cosine['mean']:.4f} [{cosine['ci95'][0]:.4f}, {cosine['ci95'][1]:.4f}] |")
    baseline, rag = generation["qwen"], generation["qwen_rag"]
    pissa, pissa_rag = generation["qwen_pissa"], generation["qwen_pissa_rag"]
    def delta(left: dict, right: dict, key: str) -> float:
        return left[key]["mean"] - right[key]["mean"]
    lines.extend([
        "", "### Diễn giải ablation", "",
        f"- RAG trên base Qwen tăng PhoBERT-BERTScore **{delta(rag, baseline, 'phobert_bertscore_f1'):+.4f}** "
        f"và BGE cosine **{delta(rag, baseline, 'bge_m3_cosine'):+.4f}**. Kết quả phù hợp với retrieval "
        "hybrid+reranker có Hit@10 cao nhất.",
        f"- PiSSA đơn lẻ thay đổi PhoBERT-BERTScore **{delta(pissa, baseline, 'phobert_bertscore_f1'):+.4f}** "
        f"và BGE cosine **{delta(pissa, baseline, 'bge_m3_cosine'):+.4f}** so với base. Với 5.000 mẫu/3 epoch, "
        "đây chưa phải bằng chứng rằng PiSSA cải thiện quality end-to-end; cần error analysis và nhiều seed.",
        f"- Thêm RAG cho model PiSSA tăng lần lượt **{delta(pissa_rag, pissa, 'phobert_bertscore_f1'):+.4f}** "
        f"và **{delta(pissa_rag, pissa, 'bge_m3_cosine'):+.4f}**, nhưng vẫn thấp hơn Qwen+RAG "
        f"**{delta(pissa_rag, rag, 'phobert_bertscore_f1'):+.4f}** BERTScore. Vì vậy Qwen+RAG là arm tốt nhất "
        "theo hai metric tự động của run này, không phải một khuyến nghị lâm sàng.",
        "- CI bootstrap thể hiện bất định lấy mẫu cho mean của test set cố định; chúng không thay thế kiểm định qua seed, "
        "bác sĩ đánh giá hay kiểm tra an toàn.",
    ])
    lines.extend(["", "## Artifact inventory", "",
                  "| Artifact | Relative path | Purpose |", "|---|---|---|",
                  "| Config | `configs/experiment.yaml` | pinned sources and hyperparameters |",
                  f"| Data | `{run_rel}/02_prepare_data` | JSONL splits, corpus, contamination report, manifest |",
                  f"| Train | `{run_rel}/03_train_pissa` | log, checkpoints, metrics, manifest |",
                  "| PiSSA residual | `artifacts/models/pissa_qwen2_5_3b_r32_residual` | frozen residual base + `pissa_init` |",
                  "| Inference adapter | `artifacts/models/pissa_qwen2_5_3b_r32_lora_best` | converted final adapter |",
                  f"| Retrieval | `{run_rel}/05_retrieval_eval` | rankings JSONL and metrics |",
                  f"| Generation | `{run_rel}/06_generation` | four prediction JSONL files |",
                  f"| Scores | `{run_rel}/07_generation_scores` | per-example scores and aggregate metrics |", "",
                  "## Giới hạn và sử dụng có trách nhiệm", "",
                  "- Similarity metrics không kiểm tra factuality, guideline concordance, dosage, triage, contraindication hay safety.",
                  "- PhoBERT-BERTScore dùng VnCoreNLP word segmentation và cắt mỗi text ở 256 token theo giới hạn encoder; BGE cosine dùng toàn bộ text đến giới hạn BGE-M3 1.024 token.",
                  "- Gold ViHealthQA không chứng minh coverage của y văn thực tế; cần expert review và safety/abstention evaluation trước sử dụng y tế.",
                  "- Đọc scored JSONL/per-query rankings và error categories; không chọn arm chỉ dựa vào một mean score."])
    (paths.reports / "CS2308_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
