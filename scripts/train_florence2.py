"""
Fine-tune Florence-2-large with LoRA for structured fashion tag extraction.

Produces a lightweight adapter that generates JSON tags from product images.

Usage:
    python scripts/train_florence2.py --config configs/florence2_lora.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class FlorenceFashionDataset(torch.utils.data.Dataset):
    """Dataset for Florence-2 fashion tag training."""

    def __init__(self, df: pd.DataFrame, processor, max_length: int = 1024):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (512, 512), (128, 128, 128))

        prompt = row["prompt"]
        answer = row["tags_json"]

        # Florence-2 uses a task-prefix approach
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # Tokenize the target
        labels = self.processor.tokenizer(
            answer,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # Squeeze batch dim (DataLoader will re-batch)
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0),
        }


def collate_fn(batch):
    """Stack tensors from batch items."""
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0].keys()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/florence2_lora.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model ──
    log.info("Loading Florence-2: %s", cfg["model"]["name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(
        cfg["model"]["name"],
        trust_remote_code=True,
    )

    # Apply LoRA
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(device)

    # ── Data ──
    train_df = pd.read_parquet(data_cfg["train_path"])
    val_df = pd.read_parquet(data_cfg["val_path"])

    train_dataset = FlorenceFashionDataset(train_df, processor)
    val_dataset = FlorenceFashionDataset(val_df, processor)

    bs = train_cfg["per_device_train_batch_size"]
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True,
        collate_fn=collate_fn, num_workers=train_cfg.get("dataloader_num_workers", 4),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False,
        collate_fn=collate_fn, num_workers=train_cfg.get("dataloader_num_workers", 4),
    )

    # ── Optimizer & scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    num_epochs = train_cfg["num_train_epochs"]
    accum_steps = train_cfg["gradient_accumulation_steps"]
    total_steps = (len(train_loader) // accum_steps) * num_epochs
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──
    log.info(
        "Training for %d epochs, %d total steps (accum=%d)",
        num_epochs, total_steps, accum_steps,
    )
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / accum_steps
            loss.backward()
            epoch_loss += loss.item() * accum_steps

            if step % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{epoch_loss / step:.4f}")

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        val_loss /= len(val_loader)
        log.info("Epoch %d — train_loss: %.4f, val_loss: %.4f",
                 epoch, epoch_loss / len(train_loader), val_loss)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = output_dir / "best"
            model.save_pretrained(str(save_path))
            processor.save_pretrained(str(save_path))
            log.info("Saved best model (val_loss=%.4f) to %s", val_loss, save_path)

    # Save final
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    processor.save_pretrained(str(final_path))
    log.info("Saved final adapter to %s", final_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
