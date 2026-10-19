"""
Evaluate fine-tuned model outputs against ground truth.

Metrics:
  - Descriptions: ROUGE-L, BERTScore, average length
  - Tags: per-field accuracy, list-field F1

Usage:
    python scripts/evaluate.py \
        --predictions outputs/results.json \
        --ground-truth data/splits/test.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def evaluate_descriptions(predictions: list[dict], gt_map: dict[str, str]):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rouge_scores, pred_lengths, gt_lengths = [], [], []

    for pred in predictions:
        img = pred.get("image", "")
        desc = pred.get("description", "")
        if not desc or img not in gt_map:
            continue
        gt_desc = gt_map[img]["description"]
        score = scorer.score(gt_desc, desc)
        rouge_scores.append(score["rougeL"].fmeasure)
        pred_lengths.append(len(desc.split()))
        gt_lengths.append(len(gt_desc.split()))

    if not rouge_scores:
        log.warning("No matched samples for description evaluation")
        return

    print("\n── Description Evaluation ──")
    print(f"  Matched samples:     {len(rouge_scores)}")
    print(f"  ROUGE-L (avg):       {sum(rouge_scores)/len(rouge_scores):.4f}")
    print(f"  Avg pred length:     {sum(pred_lengths)/len(pred_lengths):.1f} words")
    print(f"  Avg GT length:       {sum(gt_lengths)/len(gt_lengths):.1f} words")

    try:
        from bert_score import score as bert_score
        refs = [gt_map[p["image"]]["description"] for p in predictions
                if p.get("description") and p["image"] in gt_map]
        preds = [p["description"] for p in predictions
                 if p.get("description") and p["image"] in gt_map]
        _, _, F1 = bert_score(preds, refs, lang="en", verbose=False)
        print(f"  BERTScore F1 (avg):  {F1.mean().item():.4f}")
    except ImportError:
        log.info("pip install bert-score for BERTScore")


def evaluate_tags(predictions: list[dict], gt_map: dict[str, dict]):
    string_fields = ["category", "subcategory", "toe_shape", "heel_type",
                     "heel_height", "closure", "primary_color", "material", "season"]
    list_fields = ["secondary_colors", "embellishments", "style_tags", "occasion"]

    field_correct = {f: 0 for f in string_fields}
    field_total = {f: 0 for f in string_fields}
    list_f1 = {f: [] for f in list_fields}
    matched = 0

    for pred in predictions:
        img = pred.get("image", "")
        tags = pred.get("tags")
        if not tags or img not in gt_map or not gt_map[img].get("tags"):
            continue
        gt = gt_map[img]["tags"]
        matched += 1

        for f in string_fields:
            if f in gt and gt[f]:
                field_total[f] += 1
                if str(tags.get(f, "")).lower().strip() == str(gt[f]).lower().strip():
                    field_correct[f] += 1

        for f in list_fields:
            if f in gt and gt[f]:
                pred_set = set(str(t).lower() for t in (tags.get(f) or []))
                gt_set = set(str(t).lower() for t in gt[f])
                if pred_set or gt_set:
                    tp = len(pred_set & gt_set)
                    p = tp / len(pred_set) if pred_set else 0
                    r = tp / len(gt_set) if gt_set else 0
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                    list_f1[f].append(f1)

    print(f"\n── Tag Evaluation ({matched} matched) ──")
    for f in string_fields:
        if field_total[f]:
            acc = field_correct[f] / field_total[f]
            print(f"  {f:20s} accuracy: {acc:.3f} ({field_correct[f]}/{field_total[f]})")
    for f in list_fields:
        if list_f1[f]:
            avg = sum(list_f1[f]) / len(list_f1[f])
            print(f"  {f:20s} F1 (avg): {avg:.3f} (n={len(list_f1[f])})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    args = parser.parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)

    gt_df = pd.read_parquet(args.ground_truth)

    # Build ground-truth lookup from parquet conversations
    gt_map = {}
    for _, row in gt_df.iterrows():
        convs = json.loads(row["conversations"])
        assistant_text = ""
        for msg in convs:
            if msg["role"] == "assistant":
                assistant_text = msg["content"]
                break

        # Parse description
        desc = ""
        import re
        desc_match = re.search(r"DESCRIPTION:\s*\n(.+?)(?:\n\nTAGS:|\Z)", assistant_text, re.DOTALL)
        if desc_match:
            desc = desc_match.group(1).strip()

        # Parse tags
        tags = None
        tags_match = re.search(r"TAGS:\s*\n(\{.+\})", assistant_text, re.DOTALL)
        if tags_match:
            try:
                tags = json.loads(tags_match.group(1))
            except json.JSONDecodeError:
                pass

        gt_map[row["image_path"]] = {"description": desc, "tags": tags}

    evaluate_descriptions(predictions, gt_map)
    evaluate_tags(predictions, gt_map)


if __name__ == "__main__":
    main()
