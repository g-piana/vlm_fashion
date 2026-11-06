"""
Apply consolidation/decisions.yaml (filled in after reconcile_taxonomy.py).

Always prints the plan:
  * app side   values to add to base/<file>.yaml, app rows to rename
  * VLM side   config value lists to edit
  * labels     how many label values change, per file
With --apply it rewrites the label files (a .bak copy of each changed file is kept).

Usage:
    python scripts/migrate_label_values.py                      # plan only
    python scripts/migrate_label_values.py --apply
    python scripts/migrate_label_values.py --labels "data/raw/labels/*.json" "outputs/inference/*_drafts.json"
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIELD_FILE = {"colours": "colours.yaml", "material": "materials.yaml", "finishing": "finishings.yaml",
              "pattern": "patterns.yaml", "embellishments": "embellishments.yaml", "style": "styles.yaml"}


def build_rules(decisions: dict):
    """value rules {vlm_attr: {old: new|None}}, subcategory rules {(cat, old): new|None}, plan lines."""
    values: dict[str, dict] = defaultdict(dict)
    subcats: dict[tuple, str | None] = {}
    app_add, app_rename, vlm_edit = defaultdict(list), [], defaultdict(list)

    for group, entries in decisions.items():
        field = group.split("[")[0]
        for key, d in (entries or {}).items():
            std, app_now = d.get("standard"), d.get("app_now")
            attrs = d.get("vlm_attributes") or []
            if field == "subcategory":
                cat, _, old = key.partition("/")
                if not old:
                    vlm_edit["category"].append(f"{key}: not in app — fix the VLM hierarchy or add it to the app")
                    continue
                if std != old:
                    subcats[(cat, old)] = std
                    vlm_edit["taxonomy"].append(f"{cat}: {old} -> {std}")
                if app_now is None and std:
                    app_add["category_hierarchy.yaml"].append(f"{cat}: add subcategory '{std}'")
                elif app_now and std and app_now != std:
                    app_rename.append(f"category_hierarchy: {cat}/{app_now} -> {std}")
                elif app_now == std:
                    app_rename.append(f"category_hierarchy: {cat}/{std}: fix slug (see note: {d.get('note')})")
                continue

            if std != key:
                for a in attrs:
                    values[a][key] = std
                vlm_edit[", ".join(attrs)].append(f"{key} -> {std if std else '(remove)'}")
            if std is None:
                continue
            target = f"macro_modeldetails.yaml [{group[group.find('[') + 1:-1]}]" if field == "mod_details" \
                else FIELD_FILE.get(field, f"{field}.yaml")
            if app_now is None:
                app_add[target].append(std)
            elif app_now != std:
                app_rename.append(f"{target}: {app_now} -> {std}")
    return values, subcats, (app_add, app_rename, vlm_edit)


def migrate_item(tags: dict, values: dict, subcats: dict, counts: Counter) -> None:
    for attr, rules in values.items():
        v = tags.get(attr)
        if isinstance(v, list):
            new = []
            for x in v:
                y = rules.get(x, x)
                if y != x:
                    counts[f"{attr}: {x} -> {y}"] += 1
                if y is not None and y not in new:
                    new.append(y)
            tags[attr] = new
        elif v in rules:
            counts[f"{attr}: {v} -> {rules[v]}"] += 1
            tags[attr] = rules[v]
    key = (tags.get("category"), tags.get("subcategory"))
    if key in subcats:
        counts[f"subcategory: {key[0]}/{key[1]} -> {subcats[key]}"] += 1
        tags["subcategory"] = subcats[key]


def migrate_file(path: Path, values, subcats, apply: bool) -> Counter:
    with open(path, encoding="utf-8") as f:
        samples = json.load(f)
    counts: Counter = Counter()
    for s in samples if isinstance(samples, list) else []:
        tags = s.get("tags")
        for item in tags if isinstance(tags, list) else [tags]:
            if isinstance(item, dict):
                migrate_item(item, values, subcats, counts)
    if apply and counts:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=Path, default=PROJECT_ROOT / "consolidation" / "decisions.yaml")
    ap.add_argument("--labels", nargs="*", default=[str(PROJECT_ROOT / "data/raw/labels/*.json")])
    ap.add_argument("--apply", action="store_true", help="rewrite label files (keeps .bak copies)")
    args = ap.parse_args()

    with open(args.decisions, encoding="utf-8") as f:
        decisions = (yaml.safe_load(f) or {}).get("decisions") or {}
    values, subcats, (app_add, app_rename, vlm_edit) = build_rules(decisions)

    print("── App side (edit base/*.yaml by hand) ──")
    for target, vals in app_add.items():
        print(f"  add to {target}: {', '.join(map(str, vals))}")
    for line in app_rename:
        print(f"  rename {line}   (rename the DB row first, then the YAML: see notes)")
    print("\n── VLM configs (edit by hand) ──")
    for attrs, lines in vlm_edit.items():
        print(f"  {attrs}: {'; '.join(lines)}")

    print(f"\n── Labels ({'APPLY' if args.apply else 'dry run'}) ──")
    total = Counter()
    files = sorted({p for pattern in args.labels for p in glob.glob(pattern)})
    for p in files:
        c = migrate_file(Path(p), values, subcats, args.apply)
        if c:
            print(f"  {p}: {sum(c.values())} change(s)")
            total.update(c)
    for change, n in total.most_common():
        print(f"    {n:4d} x {change}")
    if not total:
        print("  no label values to change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
