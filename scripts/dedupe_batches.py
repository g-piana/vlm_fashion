"""
Find images that appear in more than one batch of the manifest (same file name), compare the
files (md5) and the labels, and optionally drop the later copies from the labels JSON.
The first batch in manifest order keeps the sample; image files are never deleted.

    python scripts/dedupe_batches.py            # report only
    python scripts/dedupe_batches.py --apply    # remove later duplicates (+ .bak_dedupe backup)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def md5(p: Path) -> str | None:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.is_file() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "configs/dataset_manifest.yaml")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    batches = (yaml.safe_load(a.manifest.read_text(encoding="utf-8")) or {}).get("batches", [])
    data, where = {}, defaultdict(list)
    for b in batches:
        path = ROOT / b["labels"]
        data[b["name"]] = (path, json.loads(path.read_text(encoding="utf-8")), ROOT / b["images_dir"])
        for s in data[b["name"]][1]:
            where[s["image"]].append(b["name"])

    dupes = {img: names for img, names in where.items() if len(names) > 1}
    if not dupes:
        print("no duplicate image names across batches")
        return 0
    drop: dict[str, set[str]] = defaultdict(set)
    for img, names in sorted(dupes.items()):
        keep, others = names[0], names[1:]
        k_path, k_data, k_dir = data[keep]
        k_tags = next(s.get("tags") for s in k_data if s["image"] == img)
        k_md5 = md5(k_dir / img)
        notes = []
        for o in others:
            o_path, o_data, o_dir = data[o]
            o_tags = next(s.get("tags") for s in o_data if s["image"] == img)
            same_file = md5(o_dir / img) == k_md5 and k_md5 is not None
            notes.append(f"{o}: file {'identical' if same_file else 'DIFFERENT or missing'}, "
                         f"labels {'identical' if o_tags == k_tags else 'differ'}")
            if same_file or o == keep:
                drop[o].add(img)
        print(f"{img}: keep in {keep}; " + "; ".join(notes))
    print("\nDIFFERENT files with the same name are not removed: rename one of them instead.")
    if not a.apply:
        print(f"report only: {sum(len(v) for v in drop.values())} duplicate samples would be removed (use --apply)")
        return 0
    for name, imgs in drop.items():
        path, d, _ = data[name]
        path.with_name(path.name + ".bak_dedupe").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        new = [s for s in d if s["image"] not in imgs]
        path.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{name}: removed {len(d) - len(new)} duplicates ({len(new)} left)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
