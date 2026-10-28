"""
Bring existing label files in line with the current taxonomy/vocabularies.

Deterministic fixes (always applied):
  * attributes declared `type: list` that hold a single string  ->  ["string"]
  * key `macrocategory` (redundant, comes from the category_config)  ->  dropped
  * key `secondary_color`  ->  merged into `secondary_colors` when that is empty
Explicit fixes (you choose):
  --map OLD=NEW     rename a vocabulary value everywhere it appears
                    (material, secondary_materials, ...). `--map OLD=` removes it.
                    `--map ATTR:OLD=NEW` limits it to one attribute, e.g. `--map embellishments:none=`
  --move SRC:VAL=DST[:NEWVAL]
                    move a value to another attribute, e.g. a value that changed
                    kind in the taxonomy:  --move material:canvas=finishing:canvas
                    (skipped, with a message, if DST is not defined in the config)
  --fallback ATTR=VALUE
                    value to put in a scalar attribute emptied by --move
                    (without it the attribute becomes null)
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


def _lookup(value_map: dict, key: str, v):
    for k in (f"{key}:{v}", str(v)):
        if k in value_map:
            return True, value_map[k]
    return False, None


def apply_moves(spec, tags: dict, moves: list, fallbacks: dict, log: Counter) -> bool:
    moved = False
    for src, val, dst, new_val in moves:
        cur = tags.get(src)
        if isinstance(cur, list):
            hit = val in [str(x) for x in cur]
        else:
            hit = cur is not None and str(cur) == val
        if not hit:
            continue
        if dst not in spec.attrs:
            log.update([f"SKIPPED move {src}:{val} -> {dst}: '{dst}' is not an attribute of this config"])
            continue
        # remove from source
        if isinstance(cur, list):
            rest = [x for x in cur if str(x) != val]
            if not rest and not spec.attrs.get(src, {}).get("multi"):
                tags[src] = fallbacks.get(src)          # single-valued attribute emptied
            else:
                tags[src] = rest
        else:
            tags[src] = fallbacks.get(src)
        # add to destination
        dcur = tags.get(dst)
        if spec.attrs[dst]["multi"]:
            lst = list(dcur) if isinstance(dcur, list) else ([dcur] if dcur else [])
            if new_val not in lst:
                lst.append(new_val)
            tags[dst] = lst
        elif dcur in (None, ""):
            tags[dst] = new_val
        else:
            log.update([f"{dst} already set ({dcur!r}); {val!r} not copied"])
        log.update([f"{src}: {val} -> {dst}: {new_val}"])
        moved = True
    return moved


def migrate_tags(spec, tags: dict, value_map: dict, drop_keys: set, log: Counter,
                 moves=(), fallbacks=None, unresolved=None) -> bool:
    moved = apply_moves(spec, tags, list(moves), fallbacks or {}, log)
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

    for key in list(tags):
        if key in ("category", "subcategory"):
            continue
        value = tags[key]
        info = spec.attrs.get(key)
        if value is None:
            if info and info["multi"]:
                tags[key] = []                      # list fields use [] when nothing applies
            continue
        # value renames ("OLD=NEW" everywhere, or "ATTR:OLD=NEW" for one attribute)
        if value_map:
            if isinstance(value, list):
                new = []
                for v in value:
                    found, repl = _lookup(value_map, key, v)
                    if found:
                        log.update([f"{key}: {v} -> {repl or '(removed)'}"])
                        if repl:
                            new.append(repl)
                    else:
                        new.append(v)
                tags[key] = value = new
            else:
                found, repl = _lookup(value_map, key, value)
                if found:
                    log.update([f"{key}: {value} -> {repl or '(removed)'}"])
                    tags[key] = value = repl or None
        # shape: list-declared fields hold lists, enum-declared fields hold one value
        if info and info["multi"]:
            if value is None:
                tags[key] = []
            elif isinstance(value, str):
                tags[key] = [value]
                log.update([f"{key}: string -> list"])
        elif info and isinstance(value, list):
            if not value:
                tags[key] = None
                log.update([f"{key}: [] -> null"])
            elif len(value) == 1:
                tags[key] = value[0]
                log.update([f"{key}: [x] -> x"])
            elif key == "material" and "secondary_materials" in spec.attrs:
                tags[key] = value[0]
                sec = tags.get("secondary_materials")
                sec = list(sec) if isinstance(sec, list) else ([sec] if sec else [])
                tags["secondary_materials"] = sec + [v for v in value[1:] if v not in sec]
                log.update([f"material: {len(value)} values -> first kept, rest moved to secondary_materials"])
            elif unresolved is not None:
                unresolved.append((key, value))     # needs a decision (see summary)
    return moved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--map", action="append", default=[], metavar="OLD=NEW")
    ap.add_argument("--move", action="append", default=[], metavar="SRC:VAL=DST[:NEWVAL]")
    ap.add_argument("--fallback", action="append", default=[], metavar="ATTR=VALUE")
    ap.add_argument("--drop-keys", nargs="*", default=[])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    value_map = {}
    for m in args.map:
        old, _, new = m.partition("=")
        value_map[old] = new
    drop_keys = set(args.drop_keys)
    moves = []
    for m in args.move:
        left, _, right = m.partition("=")
        src, _, val = left.partition(":")
        dst, _, new_val = right.partition(":")
        moves.append((src, val, dst, new_val or val))
    fallbacks = dict(f.partition("=")[::2] for f in args.fallback)

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
        moved_images: list[str] = []
        multi_examples: list = []
        for s in samples:
            tags = s.get("tags")
            if not isinstance(tags, dict):
                continue
            spec = get_spec(s.get("category_config") or DEFAULT_CONFIG)
            before += bool(validate_tags(spec, tags))
            unres: list = []
            if migrate_tags(spec, tags, value_map, drop_keys, log, moves, fallbacks, unres):
                moved_images.append(s.get("image", "?"))
            for k, v in unres:
                multi_examples.append((s.get("image", "?"), k, v))
            issues = validate_tags(spec, tags)
            after += bool(issues)
            for lv, msg in issues:
                leftovers[f"{lv}: {msg}"] += 1
                if "subsubcategory" in msg:
                    leftovers[f"    {s.get('image')}: subsubcategory = {tags.get('subsubcategory')!r}"] += 1

        print(f"\n{b.get('name', path.stem)}: samples with issues {before} -> {after}")
        for what, n in log.most_common():
            print(f"   {n:4d} x {what}")
        if moved_images:
            print(f"   moved in {len(moved_images)} samples (check them): " + ", ".join(moved_images[:12])
                  + (" ..." if len(moved_images) > 12 else ""))
        if multi_examples:
            by_field = Counter(k for _, k, _ in multi_examples)
            print("   several values in a single-valued field (decide: declare it `type: list`, or fix by hand): "
                  + ", ".join(f"{k} x{n}" for k, n in by_field.items()))
            for img, k, v in multi_examples[:6]:
                print(f"      {img}: {k} = {v}")
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
