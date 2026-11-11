"""
Themed review queues across ALL batches, and merge-back routed to the right batch by file name.

Build (one drafts file per category_config, sorted by subcategory so similar items sit together):
    python scripts/review_queue.py build --name mules \
        --subcategories "flat mule,heeled mule,mule sandal,flat sandal,heeled sandal,slider,slip on,other flat shoes"
    python scripts/review_queue.py build --name straps --attr model_details=ankle-strap,ankle-wrap
    python scripts/review_queue.py build --name disagree --predictions outputs/pred_v5/*.json \
        --compare subcategory,model_details

Review each outputs/queues/<name>_<config>.json with:
    python scripts/review_labels.py --drafts outputs/queues/mules_shoes_women.json --category shoes_women \
        --export-dir outputs/reviewed --batch-name mules

Merge (finds each image's batch; refuses names that exist in more than one batch):
    python scripts/review_queue.py merge --exported outputs/reviewed/shoes_women_mules.json [--dry-run]
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LIST_FIELDS_HINT = "subcategory,category,model_details,closure,..."

# --rules shoes: label consistency with the agreed shoe guidelines (heel rule, closed-toe mules)
FLAT_SIDE = {"ballerina", "boat shoe", "d'orsay flat", "flat lace up", "flat mule", "loafer", "slip on",
             "espadrille", "other flat shoes", "flat sandal", "slider"}
HEELED_SIDE = {"pump", "mary jane pump", "d'orsay pump", "loafer pump", "slingback", "heeled lace up",
               "heeled mule", "heeled sandal"}


def rule_issues(tags: dict) -> list[str]:
    sub, hh, ht = tags.get("subcategory"), tags.get("heel_height"), tags.get("heel_type")
    out = []
    if sub in FLAT_SIDE and (hh in ("mid-heel", "high-heel") or ht == "kitten-heel"):
        out.append(f"heel rule: flat-side {sub!r} with {hh or ''} {ht or ''}".strip())
    if sub in HEELED_SIDE and hh in ("flat-heel", "low-heel") and ht != "kitten-heel":
        out.append(f"heel rule: heeled-side {sub!r} with {hh}")
    if ht == "kitten-heel" and hh in ("flat-heel", "low-heel"):
        out.append(f"kitten heel labelled {hh} (kitten = mid-heel)")
    if sub in ("flat mule", "heeled mule") and tags.get("toe_shape") == "peep-toe":
        out.append(f"open toe on {sub!r}: mule sandal?")
    if sub == "desert" and tags.get("category") != "boots-and-booties-women" and "women" in str(tags.get("category")):
        out.append("desert moved to boots: category to update")
    return out


def load_batches(manifest: Path) -> list[dict]:
    m = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    out = []
    for b in m.get("batches", []):
        path = ROOT / b["labels"]
        if not path.exists():
            print(f"missing labels file, skipped: {path}")
            continue
        out.append({**b, "path": path, "data": json.loads(path.read_text(encoding="utf-8"))})
    return out


def norm(v) -> frozenset | str | None:
    if isinstance(v, list):
        return frozenset(str(x) for x in v)
    return None if v in (None, "", "none", "null") else str(v)


def show(v) -> str:
    return "[" + ", ".join(sorted(v)) + "]" if isinstance(v, frozenset) else repr(v)


def build(a) -> int:
    batches = load_batches(a.manifest)
    subs = {s.strip() for s in (a.subcategories or "").split(",") if s.strip()}
    wanted = {Path(s.strip()).name for s in (a.images or "").split(",") if s.strip()}
    attr_filters = []
    for f in a.attr or []:
        field, _, vals = f.partition("=")
        attr_filters.append((field.strip(), {v.strip() for v in vals.split(",") if v.strip()}))

    preds: dict[str, dict] = {}
    for p in a.predictions or []:
        for r in json.loads(Path(p).read_text(encoding="utf-8")):
            if isinstance(r, dict) and isinstance(r.get("tags"), dict):
                preds[Path(str(r.get("image", ""))).name] = r["tags"]
    compare = [c.strip() for c in (a.compare or "subcategory").split(",") if c.strip()]
    if a.predictions and not preds:
        sys.exit("no usable predictions found")

    queues: dict[str, list] = defaultdict(list)
    for b in batches:
        for s in b["data"]:
            tags = s.get("tags")
            if not isinstance(tags, dict):
                continue
            reasons = []
            if wanted and s["image"] in wanted:
                reasons.append("listed")
            if subs and tags.get("subcategory") in subs:
                reasons.append(f"subcategory {tags.get('subcategory')!r}")
            for field, vals in attr_filters:
                v = tags.get(field)
                hit = set(v) & vals if isinstance(v, list) else ({v} & vals)
                if hit:
                    reasons.append(f"{field} {sorted(hit)}")
            if a.rules:
                reasons += rule_issues(tags)
            if preds:
                pt = preds.get(s["image"])
                if pt is not None:
                    for c in compare:
                        if norm(tags.get(c)) != norm(pt.get(c)):
                            reasons.append(f"{c}: label {show(norm(tags.get(c)))} vs model {show(norm(pt.get(c)))}")
            if not reasons:
                continue
            q = copy.deepcopy(s)
            q["image"] = str((ROOT / b["images_dir"] / s["image"]).resolve())
            q["_review"] = [f"[{b['name']}] " + "; ".join(reasons)]
            queues[s.get("category_config") or "shoes_women"].append(q)

    out_dir = ROOT / "outputs" / "queues"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not queues:
        print("nothing matched")
        return 0
    for cfg, items in sorted(queues.items()):
        items.sort(key=lambda q: (str(q["tags"].get("category")), str(q["tags"].get("subcategory")), q["image"]))
        out = out_dir / f"{a.name}_{cfg}.json"
        if out.exists() and not a.force:
            print(f"{out} exists: not overwritten (use --force)")
            continue
        out.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        counts = defaultdict(int)
        for q in items:
            counts[q["tags"].get("subcategory")] += 1
        print(f"{out}: {len(items)} items  {dict(sorted(counts.items(), key=lambda x: -x[1]))}")
        print(f"  -> python scripts/review_labels.py --drafts {out.relative_to(ROOT)} --category {cfg} "
              f"--export-dir outputs/reviewed --batch-name {a.name}")
    return 0


def merge(a) -> int:
    exported = json.loads(a.exported.read_text(encoding="utf-8"))
    if not isinstance(exported, list):
        sys.exit("exported file must be a JSON list")
    batches = load_batches(a.manifest)
    where: dict[str, list[int]] = defaultdict(list)
    for bi, b in enumerate(batches):
        for s in b["data"]:
            where[s["image"]].append(bi)

    plan: dict[int, dict[str, dict]] = defaultdict(dict)
    problems = []
    for r in exported:
        img = Path(str(r.get("image", ""))).name
        hits = where.get(img, [])
        if len(hits) == 1:
            plan[hits[0]][img] = {**r, "image": img}
        elif not hits:
            problems.append(f"{img}: not in any batch")
        else:
            problems.append(f"{img}: in several batches {[batches[i]['name'] for i in hits]}")
    if problems:
        print("NOT merged, fix first:\n  " + "\n  ".join(problems[:30]))
        return 1

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    for bi, repl in plan.items():
        b = batches[bi]
        before = len(b["data"])
        new = [repl.get(s["image"], s) for s in b["data"]]
        assert len(new) == before
        print(f"{b['name']}: {len(repl)} samples replaced ({before} total)")
        if not a.dry_run:
            b["path"].with_name(b["path"].name + f".bak_{stamp}").write_text(
                b["path"].read_text(encoding="utf-8"), encoding="utf-8")
            tmp = b["path"].with_name(b["path"].name + ".tmp")
            tmp.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            tmp.replace(b["path"])
    print("DRY RUN: nothing written" if a.dry_run else f"merged (backups *.bak_{stamp})")
    print("Skipped items are not in the export, so their labels are unchanged.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "configs/dataset_manifest.yaml")
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--name", required=True, help="queue name, used in file names")
    b.add_argument("--subcategories", help="comma-separated subcategory names")
    b.add_argument("--images", help="comma-separated image file names (spot checks)")
    b.add_argument("--attr", action="append", help="field=value1,value2 (repeatable)")
    b.add_argument("--predictions", nargs="+", help="infer.py output file(s) for a disagreement queue")
    b.add_argument("--compare", help=f"fields compared with predictions (default subcategory): {LIST_FIELDS_HINT}")
    b.add_argument("--rules", action="store_true", help="flag labels breaking the shoe guideline rules")
    b.add_argument("--force", action="store_true")
    m = sp.add_parser("merge")
    m.add_argument("--exported", type=Path, required=True)
    m.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build" and not (a.subcategories or a.attr or a.predictions or a.rules or a.images):
        ap.error("build needs --images, --subcategories, --attr, --rules and/or --predictions")
    return build(a) if a.cmd == "build" else merge(a)


if __name__ == "__main__":
    sys.exit(main())
