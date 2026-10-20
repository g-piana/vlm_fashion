"""
Prepare fashion product dataset for fine-tuning.

Reads all label JSON files from data/raw/labels/, applies the correct
per-category system prompt, and produces combined train/val/test splits.

Each label file is a list of dicts:
  {
    "image": "BG53049-01.jpg",
    "category_config": "shoes",       # which YAML to use for prompt
    "image_type": "product",          # "product" or "look"
    "description": "Stiletto-heel open-toe sandal...",
    "tags": { ... }                   # single dict for product, list for look
  }

Usage:
    python scripts/prepare_dataset.py \
        --labels-dir data/raw/labels \
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
from sklearn.model_selection import train_test_split

from category_config import build_system_prompt, get_user_message

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def format_assistant_response(
    description: str, tags, image_type: str = "product"
) -> str:
    """Build combined DESCRIPTION + TAGS response."""
    parts = [f"DESCRIPTION:\n{description.strip()}"]
    if tags is not None:
        tags_json = json.dumps(tags, indent=2, ensure_ascii=False)
        parts.append(f"TAGS:\n{tags_json}")
    return "\n\n".join(parts)


def build_conversation(sample: dict, system_prompt: str) -> list[dict]:
    image_type = sample.get("image_type", "product")
    user_msg = get_user_message()
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["_abs_image_path"]},
                {"type": "text", "text": user_msg},
            ],
        },
        {
            "role": "assistant",
            "content": format_assistant_response(
                sample["description"], sample.get("tags"), image_type
            ),
        },
    ]


def load_all_labels(labels_dir: Path) -> list[dict]:
    """Load and merge all JSON label files."""
    all_samples = []
    for json_file in sorted(labels_dir.glob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        for s in data:
            s["_source_file"] = json_file.name
        all_samples.extend(data)
        log.info("Loaded %d samples from %s", len(data), json_file.name)
    log.info("Total: %d samples from %d files",
             len(all_samples), len(list(labels_dir.glob("*.json"))))
    return all_samples


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
    log.info("Validated %d samples", len(valid))
    return valid


def build_dataframe(samples: list[dict]) -> pd.DataFrame:
    rows, skipped = [], 0
    for s in samples:
        desc = s.get("description", "")
        if not desc:
            skipped += 1
            continue

        cat_config = s.get("category_config", "shoes")
        image_type = s.get("image_type", "product")
        system_prompt = build_system_prompt(cat_config, image_type)
        conv = build_conversation(s, system_prompt)

        rows.append({
            "image_path": s["_abs_image_path"],
            "category_config": cat_config,
            "image_type": image_type,
            "conversations": json.dumps(conv, ensure_ascii=False),
        })
    if skipped:
        log.warning("Skipped %d samples without descriptions", skipped)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", type=Path, default=Path("data/raw/labels"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/raw/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--val-split", type=float, default=0.10)
    parser.add_argument("--test-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_all_labels(args.labels_dir)
    samples = validate_images(samples, args.images_dir)
    if not samples:
        log.error("No valid samples found.")
        return

    # Stratified split by category_config to ensure each category in all splits
    categories = [s.get("category_config", "shoes") for s in samples]
    train_val, test = train_test_split(
        samples, test_size=args.test_split, random_state=args.seed,
        stratify=categories if len(set(categories)) > 1 else None,
    )
    cat_tv = [s.get("category_config", "shoes") for s in train_val]
    relative_val = args.val_split / (1.0 - args.test_split)
    train, val = train_test_split(
        train_val, test_size=relative_val, random_state=args.seed,
        stratify=cat_tv if len(set(cat_tv)) > 1 else None,
    )
    log.info("Split: train=%d, val=%d, test=%d", len(train), len(val), len(test))

    for name, split in [("train", train), ("val", val), ("test", test)]:
        df = build_dataframe(split)
        path = args.output_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        log.info("Wrote %s (%d rows)", path, len(df))

        # Log category distribution
        if not df.empty:
            dist = df["category_config"].value_counts().to_dict()
            log.info("  %s distribution: %s", name, dist)


if __name__ == "__main__":
    main()
