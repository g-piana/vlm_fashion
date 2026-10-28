"""
Use a folder-per-subcategory image collection with the (flat) pipeline.

The tools expect ONE flat folder per batch: infer.py lists only the files directly
inside --images-dir, the review tool exports the bare filename, and prepare_dataset.py
looks for  images_dir/<filename>.  So:

  1. flatten:  copy every image to the flat batch folder as  <subcategory-slug>__<name>
               (your organised folders stay untouched)
  2. preset:   after infer.py, pre-select category/subcategory in the drafts from that
               prefix, so the review only has to CONFIRM it (and fix the descriptions)

    python scripts/folder_labels.py flatten --src data/raw/images/bags_women_batch03_src \
        --dst data/raw/images/bags_women_batch03 --config bags_women          # dry run
    python scripts/folder_labels.py flatten ... --write                       # do it
    python scripts/folder_labels.py preset --drafts outputs/bags_women_batch03_drafts.json \
        --config bags_women --output outputs/bags_women_batch03_preset.json

Folder names are matched ignoring case, accents, spaces and apostrophes
("Minaudière", "doctor's bag", "Top handle bag" all work). Subcategory names that
exist under two categories (e.g. shoes "lace up") are reported, not guessed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy_utils import get_spec  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".webp"}
SEP = "__"


def slugify(text: str) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")


def sub_index(config: str) -> dict[str, list[tuple[str, str]]]:
    """slug -> [(category, subcategory)]"""
    idx: dict[str, list] = defaultdict(list)
    for cat, subs in get_spec(config).taxonomy.items():
        for s in subs:
            idx[slugify(s)].append((cat, s))
    return idx


def cmd_flatten(a) -> int:
    idx = sub_index(a.config)
    plan, problems = [], []
    for folder in sorted(p for p in a.src.iterdir() if p.is_dir()):
        slug = slugify(folder.name)
        hits = idx.get(slug, [])
        if len(hits) != 1:
            problems.append(f"folder '{folder.name}': " + ("no matching subcategory" if not hits
                            else f"ambiguous, belongs to {[c for c, _ in hits]}"))
            continue
        for f in sorted(folder.rglob("*")):
            if f.is_file() and f.suffix.lower() in EXT:
                # import pdb; pdb.set_trace()
                # clean_name = f.name.replace('GUCCI','BRANDS')
                plan.append((f, a.dst / f"{slug}{SEP}{re.sub(r'[^A-Za-z0-9._-]+', '_', f.name)}", hits[0][1]))
    loose = [f for f in sorted(a.src.iterdir()) if f.is_file() and f.suffix.lower() in EXT]
    for f in loose:
        plan.append((f, a.dst / f.name, None))

    targets = [t for _, t, _ in plan]
    dupes = {t for t in targets if targets.count(t) > 1}
    per_sub: dict = defaultdict(int)
    for _, _, sub in plan:
        per_sub[sub or "(loose, no folder)"] += 1
    print(f"{len(plan)} images to {'move' if a.move else 'copy'} into {a.dst}")
    for k, n in sorted(per_sub.items()):
        print(f"  {n:4d}  {k}")
    for p in problems:
        print("  ! SKIPPED", p)
    if dupes:
        print("  ! name collisions (rename these before running):", sorted(t.name for t in dupes)[:8])
        return 1
    if not a.write:
        print("\nDry run only. Add --write to apply.")
        return 0
    a.dst.mkdir(parents=True, exist_ok=True)
    for src, dst, _ in plan:
        if dst.exists():
            print("  ! exists, skipped:", dst.name)
            continue
        (shutil.move if a.move else shutil.copy2)(src, dst)
    print("done")
    return 0


def cmd_preset(a) -> int:
    idx = sub_index(a.config)
    items = json.loads(a.drafts.read_text(encoding="utf-8"))
    set_n = changed = missing = 0
    for it in items:
        name = Path(str(it.get("image", ""))).name
        if SEP not in name:
            missing += 1
            continue
        hits = idx.get(name.split(SEP, 1)[0], [])
        if len(hits) != 1:
            missing += 1
            continue
        cat, sub = hits[0]
        tags = it.get("tags") if isinstance(it.get("tags"), dict) else {}
        if tags.get("subcategory") != sub:
            changed += 1
        tags["category"], tags["subcategory"] = cat, sub
        it["tags"] = tags
        set_n += 1
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{set_n} drafts preset from the filename prefix ({changed} differed from the model's guess); "
          f"{missing} without a usable prefix -> {a.output}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("flatten")
    f.add_argument("--src", type=Path, required=True)
    f.add_argument("--dst", type=Path, required=True)
    f.add_argument("--config", required=True)
    f.add_argument("--move", action="store_true", help="move instead of copy")
    f.add_argument("--write", action="store_true")
    p = sub.add_parser("preset")
    p.add_argument("--drafts", type=Path, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    return cmd_flatten(a) if a.cmd == "flatten" else cmd_preset(a)


if __name__ == "__main__":
    sys.exit(main())
