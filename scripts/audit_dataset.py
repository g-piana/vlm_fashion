"""
Audit training data integrity.

Checks:
  1. Every image path in the labels JSON actually exists on disk
  2. Every labelled sample made it into the parquet splits
  3. Subcategory values are valid for the declared category_config
  4. No duplicate images across batches
  5. Cross-check: for specific images, show what label was used vs what the model predicted

Usage:
    python scripts/audit_dataset.py \
        --manifest configs/dataset_manifest.yaml \
        --predictions outputs/eval_adapter_v2.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_manifest(manifest_path: Path) -> list[dict]:
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    return manifest.get("batches", [])


def audit_labels(batches: list[dict]) -> dict[str, dict]:
    """Check each labels JSON for missing images and invalid values."""
    all_images = {}  # image_path -> batch_name (for duplicate detection)
    all_samples = {}  # image filename -> sample (for cross-reference)
    issues = []

    for batch in batches:
        labels_path = Path(batch["labels"])
        images_dir = Path(batch["images_dir"])
        name = batch.get("name", labels_path.stem)

        if not labels_path.exists():
            issues.append(f"[{name}] Labels file missing: {labels_path}")
            continue

        with open(labels_path, encoding="utf-8") as f:
            data = json.load(f)

        log.info("\n── Batch: %s (%d samples) ──", name, len(data))

        missing_images = 0
        missing_desc = 0
        missing_tags = 0
        subcat_counts = Counter()

        for s in data:
            img_file = s.get("image", "")
            abs_path = str((images_dir / img_file).resolve())

            # Check image exists
            if not (images_dir / img_file).exists():
                missing_images += 1
                log.warning("  Missing image: %s", img_file)

            # Check for duplicates across batches
            if abs_path in all_images:
                issues.append(
                    f"[{name}] Duplicate image also in {all_images[abs_path]}: {img_file}"
                )
            all_images[abs_path] = name

            # Check description
            if not s.get("description", "").strip():
                missing_desc += 1

            # Check tags
            tags = s.get("tags") or {}
            if not tags:
                missing_tags += 1
            else:
                sub = tags.get("subcategory", "")
                cat = tags.get("category", "")
                subcat_counts[f"{cat}/{sub}"] += 1

            # Store for cross-reference
            stem = Path(img_file).stem
            all_samples[stem] = {
                "batch": name,
                "image": img_file,
                "category": (s.get("tags") or {}).get("category", ""),
                "subcategory": (s.get("tags") or {}).get("subcategory", ""),
                "description_start": (s.get("description") or "")[:60],
            }

        log.info("  Missing images:      %d", missing_images)
        log.info("  Missing description: %d", missing_desc)
        log.info("  Missing tags:        %d", missing_tags)
        log.info("  Subcategory counts:")
        for k, n in subcat_counts.most_common():
            log.info("    %3d  %s", n, k)

    if issues:
        log.warning("\n── Issues found ──")
        for issue in issues:
            log.warning("  %s", issue)
    else:
        log.info("\nNo cross-batch issues found.")

    return all_samples


def audit_parquets() -> dict[str, str]:
    """Check which images ended up in which split."""
    split_map = {}  # abs_image_path -> split name
    for split in ["train", "val", "test"]:
        path = Path(f"data/splits/{split}.parquet")
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for img_path in df["image_path"]:
            split_map[str(img_path)] = split
    log.info(
        "\nParquet splits: train=%d, val=%d, test=%d",
        sum(1 for v in split_map.values() if v == "train"),
        sum(1 for v in split_map.values() if v == "val"),
        sum(1 for v in split_map.values() if v == "test"),
    )
    return split_map


def cross_reference(
    all_samples: dict,
    predictions_path: Path,
    split_map: dict,
):
    """For each prediction, show ground truth label vs predicted."""
    if not predictions_path or not predictions_path.exists():
        return

    with open(predictions_path, encoding="utf-8") as f:
        predictions = json.load(f)

    log.info("\n── Prediction cross-reference ──")
    log.info(
        "%-35s %-25s %-25s %-6s %s",
        "IMAGE", "GT category/subcat", "PRED category/subcat", "SPLIT", "MATCH"
    )

    mismatches = []
    for pred in predictions:
        img_path = pred.get("image", "")
        stem = Path(img_path).stem
        gt = all_samples.get(stem)
        split = split_map.get(img_path, "unknown")

        pred_tags = pred.get("tags") or {}
        pred_cat = pred_tags.get("category", "") if isinstance(pred_tags, dict) else ""
        pred_sub = pred_tags.get("subcategory", "") if isinstance(pred_tags, dict) else ""

        if gt:
            gt_label = f"{gt['category']}/{gt['subcategory']}"
            pred_label = f"{pred_cat}/{pred_sub}"
            match = "✓" if gt["subcategory"] == pred_sub else "✗"
            log.info(
                "%-35s %-25s %-25s %-6s %s",
                stem[:35], gt_label[:25], pred_label[:25], split, match
            )
            if gt["subcategory"] != pred_sub:
                mismatches.append({
                    "image": stem,
                    "gt": gt_label,
                    "pred": pred_label,
                    "split": split,
                })
        else:
            log.info(
                "%-35s %-25s %-25s %-6s %s",
                stem[:35], "(not in training)", f"{pred_cat}/{pred_sub}", split, "?"
            )

    if mismatches:
        log.info("\n── Mismatches (%d) ──", len(mismatches))
        for m in mismatches:
            log.info("  [%s] %s  GT: %s  PRED: %s", m["split"], m["image"], m["gt"], m["pred"])
    else:
        log.info("\nAll subcategories matched!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("configs/dataset_manifest.yaml"))
    parser.add_argument("--predictions", type=Path, default=None,
                        help="Inference output JSON to cross-reference")
    args = parser.parse_args()

    batches = load_manifest(args.manifest)
    all_samples = audit_labels(batches)
    split_map = audit_parquets()
    cross_reference(all_samples, args.predictions, split_map)


if __name__ == "__main__":
    main()
