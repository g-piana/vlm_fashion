"""
Fine-tune Qwen3-VL-8B with QLoRA for fashion product analysis.
"""

from __future__ import annotations

# Unsloth must be imported first
import unsloth

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
import pandas as pd
from datasets import Dataset
from PIL import Image
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator, is_bfloat16_supported

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
        use_gradient_checkpointing=lora_cfg.get(
            "use_gradient_checkpointing", "unsloth"
        ),
        random_state=cfg["training"].get("seed", 42),
    )
    return model, tokenizer


def build_messages(row: dict) -> list[dict]:
    """Convert a parquet row into a messages list with PIL image."""
    convs_raw = row["conversations"]
    convs = json.loads(convs_raw) if isinstance(convs_raw, str) else convs_raw

    try:
        image = Image.open(row["image_path"]).convert("RGB")
    except Exception:
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

    return messages


def load_dataset_as_list(path: str, max_samples: int | None = None) -> list[dict]:
    """Load parquet and convert each row to a messages dict."""
    df = pd.read_parquet(path)
    if max_samples:
        df = df.head(max_samples)

    samples = []
    for _, row in df.iterrows():
        messages = build_messages(row)
        samples.append({"messages": messages})
    return samples


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
    log.info(
        "Trainable: %s / %s (%.2f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )

    log.info("Loading and converting training data...")
    train_samples = load_dataset_as_list(
        data_cfg["train_path"], data_cfg.get("max_samples")
    )
    val_samples = load_dataset_as_list(data_cfg["val_path"])
    log.info("Train: %d, Val: %d", len(train_samples), len(val_samples))

    sft_config = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_steps=5,
        weight_decay=train_cfg["weight_decay"],
        fp16=True,
        bf16=False,
        logging_steps=train_cfg["logging_steps"],
        save_strategy="epoch",
        save_total_limit=2,
        seed=train_cfg.get("seed", 42),
        dataloader_num_workers=0,  # 0 to avoid pickling PIL images across processes
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=cfg["model"]["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_samples,
        eval_dataset=val_samples,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
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