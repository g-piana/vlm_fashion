"""
Prepare fashion product dataset for fine-tuning.

Unified single-model approach: each training example contains both a short
description and structured JSON tags as the assistant response.

Input format (labels.json) — list of dicts:
  {
    "image": "BG53049-01.jpg",
    "description": "Stiletto-heel open-toe sandal in black ...",
    "tags": {
      "category": "sandal",
      "subcategory": "stiletto sandal",
      "toe_shape": "open-toe",
      "heel_type": "stiletto",
      ...
    }
  }

Usage:
    python scripts/prepare_dataset.py \
        --input data/raw/labels.json \
        --images-dir data/raw/images \
        --output-dir data/splits \
        --val-split 0.1 \
        --test-split 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_system_prompt(config_path: str = "configs/qwen3vl_qlora.yaml") -> str:
    """Load the system prompt from the training config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["system_prompt"].strip()


def format_assistant_response(description: str, tags: dict | None) -> str:
    """Build the combined DESCRIPTION + TAGS assistant response."""
    parts = [f"DESCRIPTION:\n{description.strip()}"]
    if tags:
        tags_json = json.dumps(tags, indent=2, ensure_ascii=False)
        parts.append(f"TAGS:\n{tags_json}")
    return "\n\n".join(parts)


def build_conversation(
    image_path: str, description: str, tags: dict | None, system_prompt: str
) -> list[dict]:
    """Format a single example as a Qwen3-VL chat conversation."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Analyze this fashion product."},
            ],
        },
        {
            "role": "assistant",
            "content": format_assistant_response(description, tags),
        },
    ]


def load_labels(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    log.info("Loaded %d labelled samples from %s", len(data), path)
    return data


def validate_images(samples: list[dict], images_dir: Path) -> list[dict]:
    valid, missing = [], 0
    for s in samples:
        img = images_dir / s["image"]
        if img.exists():
            s["_abs_image_path"] = str(img.resolve())
            valid.append(s)
        else:
            missing += 1
    if missing:
        log.warning("Skipped %d samples with missing images", missing)
    log.info("Validated %d samples with images", len(valid))
    return valid


def split_data(samples, val_split, test_split, seed=42):
    train_val, test = train_test_split(samples, test_size=test_split, random_state=seed)
    relative_val = val_split / (1.0 - test_split)
    train, val = train_test_split(train_val, test_size=relative_val, random_state=seed)
    log.info("Split: train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test


def build_dataframe(samples, system_prompt):
    rows, skipped = [], 0
    for s in samples:
        desc = s.get("description", "")
        if not desc:
            skipped += 1
            continue
        tags = s.get("tags")
        conv = build_conversation(s["_abs_image_path"], desc, tags, system_prompt)
        rows.append({
            "image_path": s["_abs_image_path"],
            "conversations": json.dumps(conv, ensure_ascii=False),
        })
    if skipped:
        log.warning("Skipped %d samples without descriptions", skipped)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare fashion VLM dataset")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--config", type=str, default="configs/qwen3vl_qlora.yaml")
    parser.add_argument("--val-split", type=float, default=0.10)
    parser.add_argument("--test-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = load_system_prompt(args.config)

    samples = load_labels(args.input)
    samples = validate_images(samples, args.images_dir)
    if not samples:
        log.error("No valid samples found.")
        return

    train, val, test = split_data(samples, args.val_split, args.test_split, args.seed)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        df = build_dataframe(split, system_prompt)
        path = args.output_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        log.info("Wrote %s (%d rows)", path, len(df))


if __name__ == "__main__":
    main()
