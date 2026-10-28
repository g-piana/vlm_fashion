"""
Score model predictions against a reviewed ground-truth labels file, with a
per-subcategory report, and compare two runs (e.g. adapter v1 vs v2).

Ground truth = the labels JSON exported by review_labels.py
(items with image, category_config, description, tags).
Predictions  = the JSON written by infer.py.
Items are matched by image FILENAME, so folder layouts do not matter.

Usage
    # score one run and save the report
    python scripts/eval_heldout.py \
        --predictions outputs/heldout_bags_v1.json \
        --ground-truth data/heldout/bags_women/bags_women_heldout01.json \
        --out-report outputs/reports/heldout_bags_v1.json

    # score a new run and compare with the previous report
    python scripts/eval_heldout.py \
        --predictions outputs/heldout_bags_v2.json \
        --ground-truth data/heldout/bags_women/bags_women_heldout01.json \
        --out-report outputs/reports/heldout_bags_v2.json \
        --compare outputs/reports/heldout_bags_v1.json

Extra options
    --errors-csv FILE              one row per image with a wrong subcategory/field
    --target-per-subcategory N     coverage check: which subcategories need more images
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PLACEHOLDER_PREFIX = "[held-out"
CORE = ("category", "subcategory")


def fname(p) -> str:
    return Path(str(p)).name


def as_set(v) -> set[str]:
    if v is None or v == "":
        return set()
    if isinstance(v, (list, tuple, set)):
        return {str(x).strip().lower() for x in v if x not in (None, "")}
    return {str(v).strip().lower()}


def f1_sets(pred: set, gt: set) -> float:
    if not pred and not gt:
        return 1.0
    tp = len(pred & gt)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gt)
    return 2 * p * r / (p + r)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def known_subcategories(gt_items: list[dict]) -> list[str]:
    """Full list of subcategories of the config(s) used, when the project is importable."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from taxonomy_utils import get_spec
        subs: list[str] = []
        for cfg in {g.get("category_config") for g in gt_items if g.get("category_config")}:
            for s in get_spec(cfg).taxonomy.values():
                subs.extend(s)
        return subs
    except Exception:
        return []


def score(preds: list[dict], gts: list[dict]) -> dict:
    gt_by = {}
    for g in gts:
        if fname(g.get("image", "")) in gt_by:
            print(f"WARNING: duplicate ground-truth filename {fname(g['image'])}")
        gt_by[fname(g.get("image", ""))] = g
    pred_by = {fname(p.get("image", "")): p for p in preds}

    matched = sorted(set(gt_by) & set(pred_by))
    rep: dict = {
        "n_ground_truth": len(gt_by),
        "n_predictions": len(pred_by),
        "n_matched": len(matched),
        "missing_predictions": sorted(set(gt_by) - set(pred_by)),
        "extra_predictions": sorted(set(pred_by) - set(gt_by)),
    }

    # ----- taxonomy ------------------------------------------------------
    cat_ok = sub_ok = sub_ok_given_cat = 0
    per_image: dict[str, dict] = {}
    conf: Counter = Counter()
    gt_n: Counter = Counter()
    pred_n: Counter = Counter()
    tp: Counter = Counter()
    field_names: set[str] = set()
    for g in gt_by.values():
        if isinstance(g.get("tags"), dict):
            field_names |= {k for k in g["tags"] if k not in CORE}

    field_stats = {f: {"exact": 0, "present": 0, "exact_present": 0, "f1_sum": 0.0, "f1_n": 0}
                   for f in field_names}

    compliance = Counter()
    for img in matched:
        gt, pr = gt_by[img], pred_by[img]
        gt_tags = gt.get("tags") if isinstance(gt.get("tags"), dict) else {}
        pr_tags = pr.get("tags") if isinstance(pr.get("tags"), dict) else {}
        gc, gs = gt_tags.get("category"), gt_tags.get("subcategory")
        pc, ps = pr_tags.get("category"), pr_tags.get("subcategory")

        c_ok, s_ok = (pc == gc), (ps == gs)
        cat_ok += c_ok
        sub_ok += s_ok
        sub_ok_given_cat += c_ok and s_ok
        gt_n[gs] += 1
        pred_n[ps] += 1
        tp[gs] += s_ok
        if not s_ok:
            conf[(gs, ps)] += 1

        wrong_fields = []
        for f in field_names:
            g_set, p_set = as_set(gt_tags.get(f)), as_set(pr_tags.get(f))
            st = field_stats[f]
            same = g_set == p_set
            st["exact"] += same
            if g_set:
                st["present"] += 1
                st["exact_present"] += same
            if g_set or p_set:
                st["f1_sum"] += f1_sets(p_set, g_set)
                st["f1_n"] += 1
            if not same:
                wrong_fields.append(f)

        per_image[img] = {"gt_category": gc, "gt_subcategory": gs, "pred_category": pc,
                          "pred_subcategory": ps, "ok": bool(s_ok), "wrong_fields": sorted(wrong_fields)}

        val = pr.get("validation")
        if isinstance(val, dict):
            compliance["with_validation_record"] += 1
            compliance["invalid_after_repair"] += not val.get("ok", True)
            compliance["category_repaired"] += bool(val.get("repaired"))
            compliance["any_issue"] += bool(val.get("issues"))

    n = len(matched) or 1
    rep["category_accuracy"] = cat_ok / n
    rep["subcategory_accuracy"] = sub_ok / n
    rep["subcategory_accuracy_given_category_right"] = (sub_ok_given_cat / cat_ok) if cat_ok else None

    per_sub = {}
    for s in sorted(set(gt_n) | set(pred_n), key=lambda x: str(x)):
        support, predicted, hit = gt_n[s], pred_n[s], tp[s]
        prec = hit / predicted if predicted else None
        rec = hit / support if support else None
        f1 = (2 * prec * rec / (prec + rec)) if prec and rec else (0.0 if support else None)
        per_sub[str(s)] = {"support": support, "predicted": predicted, "correct": hit,
                           "precision": prec, "recall": rec, "f1": f1}
    rep["per_subcategory"] = per_sub
    with_support = [v for v in per_sub.values() if v["support"]]
    rep["macro_f1"] = sum(v["f1"] or 0 for v in with_support) / len(with_support) if with_support else None
    rep["macro_recall"] = sum(v["recall"] or 0 for v in with_support) / len(with_support) if with_support else None
    rep["confusions"] = [{"truth": str(a), "predicted": str(b), "count": c} for (a, b), c in conf.most_common()]

    rep["fields"] = {}
    for f, st in sorted(field_stats.items()):
        rep["fields"][f] = {
            "exact_all": st["exact"] / n,
            "gt_present": st["present"],
            "exact_when_present": (st["exact_present"] / st["present"]) if st["present"] else None,
            "f1": (st["f1_sum"] / st["f1_n"]) if st["f1_n"] else None,
        }
    rep["compliance"] = dict(compliance)
    rep["per_image"] = per_image

    # ----- descriptions (rough: GT descriptions are corrected model output) --
    try:
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        vals = []
        for img in matched:
            gd = (gt_by[img].get("description") or "").strip()
            pdsc = (pred_by[img].get("description") or "").strip()
            if gd and pdsc and not gd.startswith(PLACEHOLDER_PREFIX):
                vals.append(sc.score(gd, pdsc)["rougeL"].fmeasure)
        rep["description_rougeL"] = (sum(vals) / len(vals)) if vals else None
    except Exception:
        rep["description_rougeL"] = None
    return rep


def pct(x) -> str:
    return "  n/a" if x is None else f"{x:5.1%}"


def print_report(rep: dict, target: int | None, gts: list[dict]) -> None:
    print(f"\nMatched {rep['n_matched']} / {rep['n_ground_truth']} ground-truth images "
          f"({rep['n_predictions']} predictions)")
    if rep["missing_predictions"]:
        print(f"  no prediction for: {rep['missing_predictions'][:8]}{' ...' if len(rep['missing_predictions']) > 8 else ''}")
    print("\n── Taxonomy ──")
    print(f"  category accuracy:                    {pct(rep['category_accuracy'])}")
    print(f"  subcategory accuracy:                 {pct(rep['subcategory_accuracy'])}")
    print(f"  subcategory accuracy | category ok:   {pct(rep['subcategory_accuracy_given_category_right'])}")
    print(f"  macro recall / macro F1 (subcats):    {pct(rep['macro_recall'])} / {pct(rep['macro_f1'])}")
    if rep["compliance"]:
        c = rep["compliance"]
        print(f"  model output needed category repair:  {c.get('category_repaired', 0)}   "
              f"invalid after repair: {c.get('invalid_after_repair', 0)}   any issue: {c.get('any_issue', 0)}")

    print("\n── Per subcategory ──  (support = images in ground truth)")
    print(f"  {'subcategory':18s} {'support':>7s} {'recall':>7s} {'prec':>7s} {'F1':>7s}")
    for s, v in sorted(rep["per_subcategory"].items(), key=lambda kv: (-kv[1]["support"], kv[0])):
        flag = "  <-- predicted but never in ground truth" if not v["support"] else ""
        print(f"  {s:18s} {v['support']:7d} {pct(v['recall']):>7s} {pct(v['precision']):>7s} {pct(v['f1']):>7s}{flag}")

    if rep["confusions"]:
        print("\n── Most frequent subcategory confusions (truth -> predicted) ──")
        for c in rep["confusions"][:12]:
            print(f"  {c['count']:3d} x {c['truth']} -> {c['predicted']}")

    print("\n── Attributes ──  (exact set match; F1 where either side has values)")
    print(f"  {'field':22s} {'exact(all)':>10s} {'exact|present':>14s} {'F1':>7s} {'n present':>10s}")
    for f, v in rep["fields"].items():
        print(f"  {f:22s} {pct(v['exact_all']):>10s} {pct(v['exact_when_present']):>14s} {pct(v['f1']):>7s} {v['gt_present']:10d}")
    if rep.get("description_rougeL") is not None:
        print(f"\n  description ROUGE-L vs corrected text: {rep['description_rougeL']:.3f} "
              "(optimistic: ground truth started from the model's own text)")

    if target:
        full = known_subcategories(gts)
        counts = {s: v["support"] for s, v in rep["per_subcategory"].items()}
        names = sorted(set(full) | {s for s, n in counts.items() if n})
        need = [(s, target - counts.get(s, 0)) for s in names if counts.get(s, 0) < target]
        print(f"\n── Coverage (target {target} images per subcategory) ──")
        if need:
            for s, k in sorted(need, key=lambda x: -x[1]):
                print(f"  {s:18s} has {counts.get(s, 0)}, needs {k} more")
        else:
            print("  every subcategory has enough images")


def print_comparison(new: dict, old: dict) -> None:
    def d(a, b):
        return "  n/a" if a is None or b is None else f"{(a - b) * 100:+5.1f}"
    print("\n══ Comparison with previous report ══")
    for key, label in (("category_accuracy", "category accuracy"), ("subcategory_accuracy", "subcategory accuracy"),
                       ("macro_recall", "macro recall"), ("macro_f1", "macro F1")):
        print(f"  {label:22s} {pct(old.get(key))} -> {pct(new.get(key))}   ({d(new.get(key), old.get(key))} pts)")

    fixed, broken = [], []
    for img, v in new["per_image"].items():
        o = old["per_image"].get(img)
        if o and o["ok"] != v["ok"]:
            (fixed if v["ok"] else broken).append((img, v, o))
    print(f"\n  images fixed (wrong -> right): {len(fixed)}    broken (right -> wrong): {len(broken)}")
    for img, v, o in fixed:
        print(f"    + {img}: {o['pred_subcategory']} -> {v['pred_subcategory']}")
    for img, v, o in broken:
        print(f"    - {img}: {o['pred_subcategory']} -> {v['pred_subcategory']} (truth {v['gt_subcategory']})")
    if len(new["per_image"]) < 60:
        print("  note: with so few images, a swing of 2-3 images is within noise; look at WHICH images flipped.")

    print("\n  attribute F1 change:")
    for f, v in new["fields"].items():
        o = old["fields"].get(f)
        if o:
            print(f"    {f:22s} {pct(o['f1'])} -> {pct(v['f1'])}   ({d(v['f1'], o['f1'])} pts)")
    rows = []
    for s, v in new["per_subcategory"].items():
        o = old["per_subcategory"].get(s)
        if o and v["support"] and o["f1"] is not None and v["f1"] is not None and abs(v["f1"] - o["f1"]) >= 0.2:
            rows.append((s, o["f1"], v["f1"], v["support"]))
    if rows:
        print("\n  subcategories whose F1 moved by 20+ points:")
        for s, a, b, sup in sorted(rows, key=lambda r: r[2] - r[1]):
            print(f"    {s:18s} {pct(a)} -> {pct(b)}   (support {sup})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--ground-truth", type=Path, required=True)
    ap.add_argument("--out-report", type=Path)
    ap.add_argument("--compare", type=Path, help="report JSON from a previous run")
    ap.add_argument("--errors-csv", type=Path)
    ap.add_argument("--target-per-subcategory", type=int)
    args = ap.parse_args()

    preds, gts = load_json(args.predictions), load_json(args.ground_truth)
    rep = score(preds, gts)
    print_report(rep, args.target_per_subcategory, gts)

    if args.compare:
        print_comparison(rep, load_json(args.compare))
    if args.out_report:
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nreport saved: {args.out_report}")
    if args.errors_csv:
        args.errors_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.errors_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "truth_subcategory", "predicted_subcategory", "wrong_fields"])
            for img, v in rep["per_image"].items():
                if not v["ok"] or v["wrong_fields"]:
                    w.writerow([img, v["gt_subcategory"], v["pred_subcategory"], " ".join(v["wrong_fields"])])
        print(f"errors csv saved: {args.errors_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
