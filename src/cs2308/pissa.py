from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .settings import Paths
from .utils import set_seed, stage_manifest


SYSTEM_PROMPT = "Bạn là trợ lý thông tin y khoa tiếng Việt. Trả lời rõ ràng, thận trọng và không bịa thông tin."


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class CompletionCollator:
    def __init__(self, pad_token_id: int): self.pad_token_id = pad_token_id
    def __call__(self, features):
        maximum = max(len(item["input_ids"]) for item in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            padding = maximum - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append([1] * len(item["input_ids"]) + [0] * padding)
            batch["labels"].append(item["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def _tokenize(rows: list[dict], tokenizer, max_length: int) -> Dataset:
    def one(row: dict) -> dict:
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["question"]},
        ], tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ], tokenize=True, add_generation_prompt=False)
        full = full[:max_length]
        labels = full.copy()
        labels[:min(len(prompt), len(labels))] = [-100] * min(len(prompt), len(labels))
        return {"input_ids": full, "labels": labels}
    return Dataset.from_list([one(row) for row in rows])


def train_pissa(config: dict, paths: Paths, force: bool = False) -> Path:
    run = paths.runs / config["project"]["run_name"]
    stage = run / "03_train_pissa"
    if (stage / "manifest.json").exists() and not force:
        return stage
    stage.mkdir(parents=True, exist_ok=True)
    set_seed(config["project"]["seed"])
    source = paths.models / "qwen2_5_3b_instruct"
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_rows = _load_jsonl(run / "02_prepare_data" / "finetune_train.jsonl")
    validation_rows = _load_jsonl(run / "02_prepare_data" / "finetune_validation.jsonl")
    max_length = config["training"]["max_length_default"]
    train_set, validation_set = _tokenize(train_rows, tokenizer, max_length), _tokenize(validation_rows, tokenizer, max_length)
    model_kwargs = {"torch_dtype": torch.bfloat16, "local_files_only": True, "low_cpu_mem_usage": True}
    try:
        model = AutoModelForCausalLM.from_pretrained(source, attn_implementation="flash_attention_2", **model_kwargs).cuda()
        attention_backend = "flash_attention_2"
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(source, attn_implementation="sdpa", **model_kwargs).cuda()
        attention_backend = "sdpa"
    model.config.use_cache = False
    train_cfg = config["training"]
    peft_cfg = LoraConfig(r=train_cfg["rank"], lora_alpha=train_cfg["alpha"], lora_dropout=0.0,
        target_modules=train_cfg["target_modules"], bias="none", task_type="CAUSAL_LM", init_lora_weights=train_cfg["pissa_init"])
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    residual_dir = paths.models / "pissa_qwen2_5_3b_r32_residual"
    init_dir = residual_dir / "pissa_init"
    if not init_dir.exists():
        model.peft_config["default"].init_lora_weights = True
        model.save_pretrained(init_dir, safe_serialization=True)
        residual = model.unload()
        residual.save_pretrained(residual_dir, safe_serialization=True)
        del residual, model
        model = AutoModelForCausalLM.from_pretrained(residual_dir, torch_dtype=torch.bfloat16, local_files_only=True)
        model = PeftModel.from_pretrained(model, init_dir, is_trainable=True)
        # Reentrant activation checkpointing needs an autograd-connected input.
        # PEFT adapters alone do not guarantee this after reloading the residual model.
        model.enable_input_require_grads()
        model.config.use_cache = False
    else:
        del model
        model = AutoModelForCausalLM.from_pretrained(residual_dir, torch_dtype=torch.bfloat16, local_files_only=True)
        model = PeftModel.from_pretrained(model, init_dir, is_trainable=True)
        model.enable_input_require_grads()
        model.config.use_cache = False
    args = TrainingArguments(
        output_dir=str(stage / "checkpoints"), num_train_epochs=train_cfg["epochs"], learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"], lr_scheduler_type="cosine", per_device_train_batch_size=2,
        per_device_eval_batch_size=2, gradient_accumulation_steps=16, gradient_checkpointing=True,
        bf16=True, tf32=True, optim="adamw_torch_fused", weight_decay=0.0, max_grad_norm=1.0,
        logging_steps=10, eval_strategy="steps", eval_steps=train_cfg["eval_steps"], save_strategy="steps",
        save_steps=train_cfg["save_steps"], save_total_limit=3, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, report_to=["tensorboard"],
        dataloader_num_workers=4, dataloader_pin_memory=True, remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_set, eval_dataset=validation_set,
                      data_collator=CompletionCollator(tokenizer.pad_token_id))
    result = trainer.train(resume_from_checkpoint=None)
    trainer.save_state()
    native = paths.models / "pissa_qwen2_5_3b_r32_native_best"
    converted = paths.models / "pissa_qwen2_5_3b_r32_lora_best"
    trainer.model.save_pretrained(native, safe_serialization=True)
    trainer.model.save_pretrained(converted, safe_serialization=True, path_initial_model_for_weight_conversion=str(init_dir))
    tokenizer.save_pretrained(native)
    tokenizer.save_pretrained(converted)
    metrics = {**result.metrics, **trainer.evaluate(), "attention_backend": attention_backend, "max_length": max_length,
               "train_rows": len(train_rows), "validation_rows": len(validation_rows)}
    (stage / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    stage_manifest(stage, config=config, inputs={"base_model": str(source), "train_rows": len(train_rows)}, outputs={
        "native_adapter": str(native), "converted_lora_adapter": str(converted), "residual_model": str(residual_dir), "metrics": metrics})
    return stage
