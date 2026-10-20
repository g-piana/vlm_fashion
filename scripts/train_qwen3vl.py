"""
Fine-tune Qwen3-VL-8B with QLoRA for fashion product analysis.

Trains ONE adapter on all categories — the per-category vocabulary
is embedded in each training example's system prompt.

Usage:
    python scripts/train_qwen3vl.py --config configs/model.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from PIL import Image
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel
from unsloth.trainer import is_bfloat16_supported

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(cfg: dict):
    model, tokenizer = FastVisionModel.from_pretrained(
        cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
        dtype=getattr(torch, cfg["model"].get("dtype", "float16")),
    )
    lora_cfg = cfg["lora"]
    model = FastVisionModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg.get("target_modules"),
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
        random_state=cfg["training"].get("seed", 42),
    )
    return model, tokenizer


def load_dataset_from_parquet(path: str, max_samples: int | None = None) -> Dataset:
    import pandas as pd
    df = pd.read_parquet(path)
    if max_samples:
        df = df.head(max_samples)
    records = []
    for _, row in df.iterrows():
        convs = json.loads(row["conversations"])
        records.append({"conversations": convs, "image_path": row["image_path"]})
    return Dataset.from_list(records)


def format_sample(sample: dict) -> dict:
    convs = sample["conversations"]
    image_path = sample["image_path"]

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        log.warning("Failed to open %s: %s", image_path, e)
        image = Image.new("RGB", (512, 512), (128, 128, 128))

    messages = []
    for msg in convs:
        if msg["role"] in ("system", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
        elif msg["role"] == "user":
            content = []
            for part in msg["content"]:
                if part["type"] == "image":
                    content.append({"type": "image", "image": image})
                else:
                    content.append(part)
            messages.append({"role": "user", "content": content})
    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    log.info("Loading model: %s", cfg["model"]["name"])
    model, tokenizer = load_model_and_tokenizer(cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("Trainable: %s / %s (%.2f%%)",
             f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    train_ds = load_dataset_from_parquet(data_cfg["train_path"], data_cfg.get("max_samples"))
    val_ds = load_dataset_from_parquet(data_cfg["val_path"])
    train_ds = train_ds.map(format_sample, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(format_sample, remove_columns=val_ds.column_names)
    log.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

    sft_config = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        eval_strategy=train_cfg.get("eval_strategy", "epoch"),
        save_total_limit=train_cfg["save_total_limit"],
        seed=train_cfg.get("seed", 42),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=cfg["model"]["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=FastVisionModel.get_vision_data_collator(tokenizer),
    )

    log.info("Starting training...")
    trainer.train()

    final_dir = Path(train_cfg["output_dir"]) / "final"
    log.info("Saving LoRA adapter to %s", final_dir)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info("Done.")


if __name__ == "__main__":
    main()
