"""
Prepare fashion product dataset for fine-tuning.

Reads configs/dataset_manifest.yaml which lists all labelled batches
(each with a labels JSON and an images directory), merges them,
applies per-category system prompts, and produces train/val/test splits.

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --manifest configs/dataset_manifest.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from category_config import build_system_prompt, get_user_message

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def format_assistant_response(description: str, tags, image_type: str = "product") -> str:
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


def load_batch(batch_cfg: dict) -> list[dict]:
    """Load one batch: read labels JSON, resolve image paths."""
    labels_path = Path(batch_cfg["labels"])
    images_dir = Path(batch_cfg["images_dir"])
    batch_name = batch_cfg.get("name", labels_path.stem)

    if not labels_path.exists():
        log.warning("Labels file not found, skipping: %s", labels_path)
        return []
    if not images_dir.exists():
        log.warning("Images dir not found, skipping: %s", images_dir)
        return []

    with open(labels_path, encoding="utf-8") as f:
        data = json.load(f)

    valid, missing = [], 0
    for s in data:
        img = images_dir / s["image"]
        if img.exists():
            s["_abs_image_path"] = str(img.resolve())
            s["_batch"] = batch_name
            valid.append(s)
        else:
            missing += 1

    if missing:
        log.warning("  %s: skipped %d samples with missing images", batch_name, missing)
    log.info("  %s: %d valid samples from %s", batch_name, len(valid), labels_path.name)
    return valid


def build_dataframe(samples: list[dict]) -> pd.DataFrame:
    rows, skipped = [], 0
    for s in samples:
        desc = s.get("description", "")
        if not desc:
            skipped += 1
            continue

        cat_config = s.get("category_config", "shoes_women")
        image_type = s.get("image_type", "product")
        system_prompt = build_system_prompt(cat_config, image_type)
        conv = build_conversation(s, system_prompt)

        rows.append({
            "image_path": s["_abs_image_path"],
            "category_config": cat_config,
            "image_type": image_type,
            "batch": s.get("_batch", "unknown"),
            "conversations": json.dumps(conv, ensure_ascii=False),
        })
    if skipped:
        log.warning("Skipped %d samples without descriptions", skipped)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset from manifest")
    parser.add_argument("--manifest", type=Path,
                        default=Path("configs/dataset_manifest.yaml"))
    args = parser.parse_args()

    if not args.manifest.exists():
        log.error("Manifest not found: %s", args.manifest)
        return

    with open(args.manifest) as f:
        manifest = yaml.safe_load(f)

    batches_cfg = manifest.get("batches", [])
    splits_cfg = manifest.get("splits", {})
    output_dir = Path(manifest.get("output_dir", "data/splits"))
    output_dir.mkdir(parents=True, exist_ok=True)

    val_ratio = splits_cfg.get("val_ratio", 0.10)
    test_ratio = splits_cfg.get("test_ratio", 0.05)
    seed = splits_cfg.get("seed", 42)

    # Load all batches
    log.info("Loading %d batches from manifest:", len(batches_cfg))
    all_samples = []
    for batch_cfg in batches_cfg:
        samples = load_batch(batch_cfg)
        all_samples.extend(samples)

    if not all_samples:
        log.error("No valid samples found across all batches.")
        return

    log.info("Total: %d valid samples from %d batches", len(all_samples), len(batches_cfg))

    # Stratified split by category_config
    categories = [s.get("category_config", "unknown") for s in all_samples]
    can_stratify = len(set(categories)) > 1

    train_val, test = train_test_split(
        all_samples, test_size=test_ratio, random_state=seed,
        stratify=categories if can_stratify else None,
    )
    cat_tv = [s.get("category_config", "unknown") for s in train_val]
    relative_val = val_ratio / (1.0 - test_ratio)
    train, val = train_test_split(
        train_val, test_size=relative_val, random_state=seed,
        stratify=cat_tv if can_stratify else None,
    )
    log.info("Split: train=%d, val=%d, test=%d", len(train), len(val), len(test))

    for name, split in [("train", train), ("val", val), ("test", test)]:
        df = build_dataframe(split)
        path = output_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        log.info("Wrote %s (%d rows)", path, len(df))

        if not df.empty:
            cat_dist = df["category_config"].value_counts().to_dict()
            batch_dist = df["batch"].value_counts().to_dict()
            log.info("  categories: %s", cat_dist)
            log.info("  batches: %s", batch_dist)


if __name__ == "__main__":
    main()
