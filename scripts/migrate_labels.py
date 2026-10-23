"""
Bring existing label files in line with the current taxonomy/vocabularies.

Deterministic fixes (always applied):
  * attributes declared `type: list` that hold a single string  ->  ["string"]
  * key `macrocategory` (redundant, comes from the category_config)  ->  dropped
  * key `secondary_color`  ->  merged into `secondary_colors` when that is empty
Explicit fixes (you choose):
  --map OLD=NEW     rename a vocabulary value everywhere it appears
                    (material, secondary_materials, ...). `--map OLD=` removes it.
  --drop-keys K ... also drop these unknown keys (look at them first!)

Default is a dry run that only reports. Use --write to apply (a .bak copy of
each file is kept).

Usage:
    python scripts/migrate_labels.py --manifest configs/dataset_manifest.yaml
    python scripts/migrate_labels.py --manifest configs/dataset_manifest.yaml \
        --map synthetic-leather=<new-slug> --write
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy_utils import DEFAULT_CONFIG, get_spec, validate_tags  # noqa: E402


def migrate_tags(spec, tags: dict, value_map: dict, drop_keys: set, log: Counter) -> None:
    # key clean-up
    tags.pop("macrocategory", None) and log.update(["dropped key macrocategory"])
    if "secondary_color" in tags:
        old = tags.pop("secondary_color")
        log.update(["merged secondary_color -> secondary_colors"])
        if old and not tags.get("secondary_colors"):
            tags["secondary_colors"] = [old] if isinstance(old, str) else list(old)
    for k in drop_keys:
        if k in tags:
            del tags[k]
            log.update([f"dropped key {k}"])

    for key, value in list(tags.items()):
        if key in ("category", "subcategory") or value is None:
            continue
        info = spec.attrs.get(key)
        # value renames
        if value_map:
            if isinstance(value, list):
                new = []
                for v in value:
                    if str(v) in value_map:
                        log.update([f"{key}: {v} -> {value_map[str(v)] or '(removed)'}"])
                        if value_map[str(v)]:
                            new.append(value_map[str(v)])
                    else:
                        new.append(v)
                tags[key] = value = new
            elif str(value) in value_map:
                log.update([f"{key}: {value} -> {value_map[str(value)] or '(removed)'}"])
                tags[key] = value = value_map[str(value)] or None
        # single string in a list-typed attribute
        if info and info["multi"] and isinstance(value, str):
            tags[key] = [value]
            log.update([f"{key}: string -> list"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--map", action="append", default=[], metavar="OLD=NEW")
    ap.add_argument("--drop-keys", nargs="*", default=[])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    value_map = {}
    for m in args.map:
        old, _, new = m.partition("=")
        value_map[old] = new
    drop_keys = set(args.drop_keys)

    with open(args.manifest, encoding="utf-8") as f:
        batches = (yaml.safe_load(f) or {}).get("batches", [])

    for b in batches:
        path = Path(b["labels"])
        if not path.exists():
            continue
        samples = json.loads(path.read_text(encoding="utf-8"))
        log: Counter = Counter()
        before = after = 0
        leftovers: Counter = Counter()
        for s in samples:
            tags = s.get("tags")
            if not isinstance(tags, dict):
                continue
            spec = get_spec(s.get("category_config") or DEFAULT_CONFIG)
            before += bool(validate_tags(spec, tags))
            migrate_tags(spec, tags, value_map, drop_keys, log)
            issues = validate_tags(spec, tags)
            after += bool(issues)
            for lv, msg in issues:
                leftovers[f"{lv}: {msg}"] += 1
                if "subsubcategory" in msg:
                    leftovers[f"    {s.get('image')}: subsubcategory = {tags.get('subsubcategory')!r}"] += 1

        print(f"\n{b.get('name', path.stem)}: samples with issues {before} -> {after}")
        for what, n in log.most_common():
            print(f"   {n:4d} x {what}")
        for what, n in leftovers.most_common(12):
            print(f"   remaining {n:3d} x {what}")
        if args.write:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
            path.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"   written (backup {path.name}.bak)")
    if not args.write:
        print("\nDry run only. Add --write to apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
