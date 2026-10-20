"""
Evaluate model outputs against ground truth, with per-category breakdown.

Usage:
    python scripts/evaluate.py \
        --predictions outputs/results.json \
        --ground-truth data/splits/test.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def parse_assistant_text(text: str) -> dict:
    desc = ""
    desc_match = re.search(r"DESCRIPTION:\s*\n(.+?)(?:\n\nTAGS:|\Z)", text, re.DOTALL)
    if desc_match:
        desc = desc_match.group(1).strip()
    tags = None
    tags_match = re.search(r"TAGS:\s*\n([\[\{].+)", text, re.DOTALL)
    if tags_match:
        try:
            tags = json.loads(tags_match.group(1))
        except json.JSONDecodeError:
            pass
    return {"description": desc, "tags": tags}


def build_gt_map(gt_df: pd.DataFrame) -> dict:
    gt_map = {}
    for _, row in gt_df.iterrows():
        convs = json.loads(row["conversations"])
        for msg in convs:
            if msg["role"] == "assistant":
                parsed = parse_assistant_text(msg["content"])
                gt_map[row["image_path"]] = {
                    **parsed,
                    "category_config": row.get("category_config", "unknown"),
                }
                break
    return gt_map


def evaluate_descriptions(predictions, gt_map):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    by_category: dict[str, list[float]] = {}
    overall = []

    for pred in predictions:
        img = pred.get("image", "")
        desc = pred.get("description", "")
        if not desc or img not in gt_map:
            continue
        gt = gt_map[img]
        score = scorer.score(gt["description"], desc)["rougeL"].fmeasure
        overall.append(score)
        cat = gt.get("category_config", "unknown")
        by_category.setdefault(cat, []).append(score)

    if not overall:
        return

    print(f"\n── Description Evaluation ({len(overall)} samples) ──")
    print(f"  Overall ROUGE-L:  {sum(overall)/len(overall):.4f}")
    for cat, scores in sorted(by_category.items()):
        avg = sum(scores) / len(scores)
        print(f"  {cat:20s} ROUGE-L: {avg:.4f} (n={len(scores)})")


def evaluate_tags(predictions, gt_map):
    all_string_fields = set()
    all_list_fields = set()

    # Discover fields from ground truth
    for gt in gt_map.values():
        tags = gt.get("tags")
        if isinstance(tags, dict):
            for k, v in tags.items():
                if isinstance(v, list):
                    all_list_fields.add(k)
                elif v is not None:
                    all_string_fields.add(k)

    field_correct = {f: 0 for f in all_string_fields}
    field_total = {f: 0 for f in all_string_fields}
    list_f1 = {f: [] for f in all_list_fields}
    matched = 0

    for pred in predictions:
        img = pred.get("image", "")
        tags = pred.get("tags")
        if not isinstance(tags, dict) or img not in gt_map:
            continue
        gt_tags = gt_map[img].get("tags")
        if not isinstance(gt_tags, dict):
            continue
        matched += 1

        for f in all_string_fields:
            if f in gt_tags and gt_tags[f] is not None:
                field_total[f] += 1
                if str(tags.get(f, "")).lower().strip() == str(gt_tags[f]).lower().strip():
                    field_correct[f] += 1

        for f in all_list_fields:
            if f in gt_tags and gt_tags[f]:
                pred_set = set(str(t).lower() for t in (tags.get(f) or []))
                gt_set = set(str(t).lower() for t in gt_tags[f])
                if pred_set or gt_set:
                    tp = len(pred_set & gt_set)
                    p = tp / len(pred_set) if pred_set else 0
                    r = tp / len(gt_set) if gt_set else 0
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                    list_f1[f].append(f1)

    print(f"\n── Tag Evaluation ({matched} matched) ──")
    for f in sorted(all_string_fields):
        if field_total[f]:
            acc = field_correct[f] / field_total[f]
            print(f"  {f:25s} accuracy: {acc:.3f} ({field_correct[f]}/{field_total[f]})")
    for f in sorted(all_list_fields):
        if list_f1[f]:
            avg = sum(list_f1[f]) / len(list_f1[f])
            print(f"  {f:25s} F1 (avg): {avg:.3f} (n={len(list_f1[f])})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    args = parser.parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)

    gt_df = pd.read_parquet(args.ground_truth)
    gt_map = build_gt_map(gt_df)

    evaluate_descriptions(predictions, gt_map)
    evaluate_tags(predictions, gt_map)


if __name__ == "__main__":
    main()
