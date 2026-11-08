"""
Migrate label files to the aligned vocabulary and queue the judgement calls for review.

Mechanical fixes (applied automatically):
  bags   closure: buckle -> removed          model_details: zip-pocket -> external-pocket
  shoes  closure: zip-closure -> zip-up      closure: button-closure -> removed (review)
         toe_shape 'cap-toe' -> moved to model_details; toe_shape re-guessed from the
         description (round/square/pointed/almond/peep) or left empty (review)
         subcategory 'mule' -> the single *mule* subcategory of the sample's category (review)
         subcategory 'sneaker' -> 'low-top sneaker' (review); model_details 'quilted' -> removed
Review only (nothing changed): filename prefix vs subcategory, heeled+flat-heel, flat shoe+mid/high heel,
  bag whose closure became empty.

    python scripts/migrate_labels.py                  # dry run: report only
    python scripts/migrate_labels.py --apply          # write labels (+ .bak_migrate) and outputs/review_<batch>.json
    # review each outputs/review_<batch>.json with review_labels.py, export, then:
    python scripts/migrate_labels.py --merge <exported.json> --batch <batch_name>
"""
from __future__ import annotations

import argparse, copy, json, re, sys, unicodedata
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from category_config import load_category_config  # noqa: E402

VALUE_MAP = {
    "bags": {"closure": {"buckle": None}, "model_details": {"zip-pocket": "external-pocket"}},
    "shoes": {"closure": {"zip-closure": "zip-up", "button-closure": None},
              "model_details": {"quilted": None}},
}
# old subcategory -> new one; applied only when the old name is gone and the new one exists under the same category
SUBCATEGORY_MAP = {"sneaker": "low-top sneaker"}
TOE_WORDS = {"almond-toe": r"\balmond", "round-toe": r"\bround(ed)?\b", "square-toe": r"\bsquare(d)?\b",
             "pointed-toe": r"\bpoint(ed|y)\b", "peep-toe": r"\bpeep"}


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def as_list(v) -> list:
    return list(v) if isinstance(v, list) else ([v] if v else [])


def guess_toe(desc: str, allowed: set[str]) -> str | None:
    hits = [v for v, p in TOE_WORDS.items() if v in allowed and re.search(p, (desc or "").lower())]
    return hits[0] if len(hits) == 1 else None


def migrate_sample(s: dict, cfgs: dict) -> tuple[list[str], list[str]]:
    """Mutates s. Returns (changes, review reasons)."""
    changes: list[str] = []
    review: list[str] = []
    tags = s.get("tags")
    if not isinstance(tags, dict):
        return changes, review
    cfg_name = s.get("category_config") or "shoes_women"
    if cfg_name not in cfgs:
        cfgs[cfg_name] = load_category_config(cfg_name)
    cfg = cfgs[cfg_name]
    macro = cfg_name.split("_")[0]

    for attr, mapping in VALUE_MAP.get(macro, {}).items():
        val = tags.get(attr)
        if isinstance(val, list):
            new: list = []
            for v in val:
                nv = mapping[v] if v in mapping else v
                if nv is not None and nv not in new:
                    new.append(nv)
            if new != val:
                tags[attr] = new
                changes.append(f"{attr}: {val} -> {new}")
                if not new:
                    review.append(f"{attr} became empty (was {val})")
        elif isinstance(val, str) and val in mapping:
            tags[attr] = mapping[val]
            changes.append(f"{attr}: {val!r} -> {mapping[val]!r}")
            if mapping[val] is None:
                review.append(f"{attr} '{val}' dropped")

    if macro == "shoes" and tags.get("toe_shape") == "cap-toe":
        md = as_list(tags.get("model_details"))
        if "cap-toe" not in md:
            md.append("cap-toe")
        tags["model_details"] = md
        spec = cfg.get("attributes", {}).get("toe_shape", {})
        allowed = {str(v) for v in spec.get("values", [])} if isinstance(spec, dict) else set()
        shape = guess_toe(s.get("description", ""), allowed)
        tags["toe_shape"] = shape
        changes.append(f"toe_shape: 'cap-toe' -> model_details + toe_shape={shape!r}")
        review.append(f"toe_shape {'guessed from description: ' + shape if shape else 'unknown, set it'}")

    taxonomy = cfg.get("taxonomy", {})
    cat, sub = tags.get("category"), tags.get("subcategory")
    if sub == "mule" and "mule" not in taxonomy.get(cat, []):
        cands = [x for x in taxonomy.get(cat, []) if "mule" in x]
        if len(cands) == 1:
            tags["subcategory"] = cands[0]
            changes.append(f"subcategory: 'mule' -> {cands[0]!r}")
            review.append("mule: check closed vs open toe (category may need to change)")
        else:
            review.append(f"mule: no single mule subcategory under category '{cat}'")

    cur_sub = tags.get("subcategory")
    if cur_sub in SUBCATEGORY_MAP and cur_sub not in taxonomy.get(cat, []):
        new = SUBCATEGORY_MAP[cur_sub]
        if new in taxonomy.get(cat, []):
            tags["subcategory"] = new
            changes.append(f"subcategory: {cur_sub!r} -> {new!r}")
            review.append(f"'{cur_sub}' -> '{new}': confirm it is not a tennis shoe / other sneaker type")
        else:
            review.append(f"subcategory '{cur_sub}' is gone and '{new}' is not under '{cat}'")

    img = s.get("image", "")
    if "__" in img:
        pre, cur = img.split("__")[0], slug(tags.get("subcategory") or "")
        if pre != cur:
            review.append(f"filename says '{pre}', subcategory is '{tags.get('subcategory')}'")
    hh, cat = tags.get("heel_height"), str(tags.get("category", ""))
    if "heeled" in cat and hh == "flat-heel":
        review.append("heeled category but heel_height=flat-heel")
    if "flat-shoes" in cat and hh in ("mid-heel", "high-heel"):
        review.append(f"flat category but heel_height={hh}")
    return changes, review


def write_json(path: Path, data, backup_suffix: str) -> None:
    bak = path.with_suffix(backup_suffix)
    if not bak.exists():
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def merge(manifest: dict, batch: str, exported: Path) -> int:
    b = next((x for x in manifest["batches"] if x.get("name") == batch), None)
    if not b:
        print(f"batch '{batch}' not in manifest")
        return 1
    path = ROOT / b["labels"]
    data = json.loads(path.read_text(encoding="utf-8"))
    by_img = {Path(r["image"]).name: r for r in json.loads(exported.read_text(encoding="utf-8"))}
    out, n = [], 0
    for s in data:
        r = by_img.pop(s["image"], None)
        out.append(r if r else s)
        n += bool(r)
    write_json(path, out, ".json.bak_premerge")
    print(f"{batch}: replaced {n} samples from {exported.name}")
    if by_img:
        print(f"  WARNING {len(by_img)} exported images not in the batch: {sorted(by_img)[:10]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "configs/dataset_manifest.yaml")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--merge", type=Path, help="exported review_labels JSON to merge back")
    ap.add_argument("--batch", help="batch name for --merge")
    a = ap.parse_args()
    manifest = yaml.safe_load(a.manifest.read_text(encoding="utf-8")) or {}

    if a.merge:
        if not a.batch:
            ap.error("--merge needs --batch")
        return merge(manifest, a.batch, a.merge)

    cfgs: dict = {}
    for b in manifest.get("batches", []):
        path = ROOT / b["labels"]
        if not path.exists():
            print(f"missing: {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        images_dir = ROOT / b["images_dir"]
        rules, queue = Counter(), []
        for s in data:
            changes, review = migrate_sample(s, cfgs)
            rules.update(c.split(":")[0] for c in changes)
            if review:
                q = copy.deepcopy(s)
                q["image"] = str((images_dir / s["image"]).resolve())
                q["_review"] = review
                queue.append(q)
        print(f"\n{b['name']}: {len(data)} samples, auto-fixed {dict(rules) or 'nothing'}, {len(queue)} to review")
        for q in queue[:40]:
            print(f"    {Path(q['image']).name}: {'; '.join(q['_review'])}")
        if a.apply:
            write_json(path, data, ".json.bak_migrate")
            if queue:
                out = ROOT / "outputs" / f"review_{b['name']}.json"
                out.parent.mkdir(exist_ok=True)
                if out.exists():  # a second --apply would see already-fixed labels and shrink the queue
                    print(f"  {out.name} already exists: NOT overwritten (delete it to rebuild)")
                else:
                    out.write_text(json.dumps(queue, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    cat = queue[0].get("category_config", "shoes_women")
                    print(f"  -> python scripts/review_labels.py --drafts {out} --category {cat} --port 8501")
    print("\nWRITTEN" if a.apply else "\nDRY RUN: nothing written (use --apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
