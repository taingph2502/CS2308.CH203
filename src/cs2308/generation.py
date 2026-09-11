from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .pissa import SYSTEM_PROMPT
from .settings import Paths
from .utils import chunks, stage_manifest


def _rows(path: Path) -> list[dict]: 
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _prompt(question: str, context: list[dict] | None) -> list[dict]:
    if not context:
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    evidence = "\n\n".join(f"[{i + 1}] {item['text']}" for i, item in enumerate(context))
    instruction = f"{SYSTEM_PROMPT} Chỉ dùng ngữ cảnh dưới đây; nếu thiếu thông tin, hãy nói rõ.\n\nNGỮ CẢNH:\n{evidence}"
    return [{"role": "system", "content": instruction}, {"role": "user", "content": question}]


def _bounded_context(context: list[dict], tokenizer, token_budget: int) -> list[dict]:
    """Preserve ranked evidence order while enforcing the configured RAG budget."""
    kept, used = [], 0
    for item in context:
        token_ids = tokenizer(item["text"], add_special_tokens=False)["input_ids"]
        remaining = token_budget - used
        if remaining <= 0:
            break
        clone = dict(item)
        if len(token_ids) > remaining:
            clone["text"] = tokenizer.decode(token_ids[:remaining], skip_special_tokens=True)
            kept.append(clone)
            break
        kept.append(clone)
        used += len(token_ids)
    return kept


def _generate(model, tokenizer, rows: list[dict], contexts: dict[str, list[dict]], use_rag: bool, config: dict,
              checkpoint_path: Path | None = None) -> list[dict]:
    results = _rows(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else []
    completed = {item["example_id"] for item in results}
    pending = [row for row in rows if row["example_id"] not in completed]
    tokenizer.padding_side = "left"
    for group in tqdm(list(chunks(pending, config["generation"]["batch_size"])), desc="Generation", unit="batch"):
        prompts, used = [], []
        for row in group:
            context = _bounded_context(
                contexts[row["example_id"]], tokenizer, config["retrieval"]["rag_context_token_budget"]
            ) if use_rag else []
            prompts.append(tokenizer.apply_chat_template(_prompt(row["question"], context), tokenize=False, add_generation_prompt=True))
            used.append(context)
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=config["generation"]["max_input_tokens"]).to("cuda")
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=config["generation"]["max_new_tokens"], do_sample=False, use_cache=True,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        width = encoded.input_ids.shape[1]
        decoded = tokenizer.batch_decode(output[:, width:], skip_special_tokens=True)
        for row, prediction, context in zip(group, decoded, used):
            results.append({"example_id": row["example_id"], "question": row["question"], "reference": row["answer"], "prediction": prediction.strip(),
                            "context_doc_ids": [item["doc_id"] for item in context], "rag": use_rag})
        if checkpoint_path:
            checkpoint_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in results) + "\n", encoding="utf-8")
    return results


def generate_all(config: dict, paths: Paths, force: bool = False) -> Path:
    run = paths.runs / config["project"]["run_name"]
    stage = run / "06_generation"
    if (stage / "manifest.json").exists() and not force: 
        return stage
    stage.mkdir(parents=True, exist_ok=True)
    
    rows = _rows(run / "02_prepare_data" / "evaluation.jsonl")
    docs = {row["doc_id"]: row for row in _rows(paths.indexes / "vihealthqa_documents.jsonl")}
    ranks = {row["example_id"]: row["ranked_doc_ids"] for row in _rows(run / "05_retrieval_eval" / "hybrid_reranker_rankings.jsonl")}
    contexts = {example_id: [docs[doc_id] for doc_id in doc_ids[:config["retrieval"]["rag_context_k"]] if doc_id in docs] for example_id, doc_ids in ranks.items()}
    
    # Tự động chọn float16 cho Colab T4 hoặc bfloat16 nếu GPU hỗ trợ
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # 1. Trỏ vào model trong artifacts/models/qwen2.5-3b-pissa-medical
    custom_model_dir = paths.models / "qwen2.5-3b-pissa-medical"
    tokenizer_source = custom_model_dir if (custom_model_dir / "tokenizer_config.json").exists() else (paths.models / "qwen2_5_3b_instruct")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Xử lý 2 arm Base (qwen và qwen_rag): Chỉ nạp base model nếu file chưa tồn tại hoặc chạy --force
    base_arms = [("qwen", False), ("qwen_rag", True)]
    needed_base_arms = [(name, rag) for name, rag in base_arms if not (stage / f"{name}.jsonl").exists() or force]
    
    if needed_base_arms:
        base = AutoModelForCausalLM.from_pretrained(
            paths.models / "qwen2_5_3b_instruct", 
            torch_dtype=dtype, 
            local_files_only=True, 
            attn_implementation="sdpa"
        ).cuda().eval()
        for name, rag in needed_base_arms:
            output_path = stage / f"{name}.jsonl"
            partial_path = output_path.with_suffix(".partial.jsonl")
            outputs = _generate(base, tokenizer, rows, contexts, rag, config, partial_path)
            output_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in outputs) + "\n", encoding="utf-8")
            partial_path.unlink(missing_ok=True)
        del base
        torch.cuda.empty_cache()

    # 3. Nạp trực tiếp mô hình fine-tune mới từ artifacts/models/ và sinh cho 2 arm (qwen_pissa và qwen_pissa_rag)
    adapted = AutoModelForCausalLM.from_pretrained(
        custom_model_dir,
        torch_dtype=dtype,
        local_files_only=True,
        attn_implementation="sdpa",
    ).cuda().eval()

    for name, rag in (("qwen_pissa", False), ("qwen_pissa_rag", True)):
        output_path = stage / f"{name}.jsonl"
        if output_path.exists() and not force:
            continue
        partial_path = output_path.with_suffix(".partial.jsonl")
        outputs = _generate(adapted, tokenizer, rows, contexts, rag, config, partial_path)
        output_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in outputs) + "\n", encoding="utf-8")
        partial_path.unlink(missing_ok=True)
    
    del adapted
    torch.cuda.empty_cache()

    stage_manifest(
        stage, 
        config=config, 
        inputs={"evaluation_rows": len(rows), "retriever": "hybrid_reranker"}, 
        outputs={"arms": 4, "records_per_arm": len(rows), "backend": f"transformers_sdpa_{dtype}"}
    )
    return stage
