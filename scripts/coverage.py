"""
How many labelled images does each subcategory have, and how many are missing
to reach a target?  Run it before collecting images so you gather exactly what
is needed in ONE round.

    python scripts/coverage.py --manifest configs/dataset_manifest.yaml --target 8
    python scripts/coverage.py --manifest configs/dataset_manifest.yaml --target 8 --config bags_women
    python scripts/coverage.py --manifest configs/dataset_manifest.yaml --min-side 700 --small-csv outputs/small_images.csv

Counts come from the reviewed labels listed in the manifest; the subcategory
list comes from the generated config, so empty subcategories show up too.

The "small" column counts labelled images whose SHORTER side is below --min-side
pixels (default 700): thumbnails lose fine texture. --small-csv writes the list of
those images (batch, file, subcategory, size) so you can replace only the ones that
matter. A size summary per batch is printed at the end.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

try:
    from PIL import Image
except ImportError:  # sizes are optional
    Image = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy_utils import DEFAULT_CONFIG, get_spec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("configs/dataset_manifest.yaml"))
    ap.add_argument("--target", type=int, default=8, help="desired labelled images per subcategory")
    ap.add_argument("--config", help="only this category_config")
    ap.add_argument("--min-side", type=int, default=700, help="shorter side below this counts as small")
    ap.add_argument("--small-csv", type=Path, help="write the list of small images here")
    args = ap.parse_args()

    counts: dict[str, Counter] = defaultdict(Counter)
    small: dict[str, Counter] = defaultdict(Counter)
    small_rows: list[list] = []
    buckets: dict[str, Counter] = defaultdict(Counter)
    for b in (yaml.safe_load(open(args.manifest, encoding="utf-8")) or {}).get("batches", []):
        p = Path(b["labels"])
        if not p.exists():
            continue
        for s in json.load(open(p, encoding="utf-8")):
            tags = s.get("tags")
            if isinstance(tags, dict):
                cfg_name = s.get("category_config") or DEFAULT_CONFIG
                counts[cfg_name][tags.get("subcategory")] += 1
                side = None
                img = Path(b.get("images_dir", "")) / s.get("image", "")
                if Image is not None and img.exists():
                    with Image.open(img) as im:
                        side = min(im.size)
                if side is None:
                    buckets[b.get("name", p.stem)]["unknown"] += 1
                    continue
                buckets[b.get("name", p.stem)]["<400" if side < 400 else "400-699" if side < 700 else "700-999" if side < 1000 else ">=1000"] += 1
                if side < args.min_side:
                    small[cfg_name][tags.get("subcategory")] += 1
                    small_rows.append([b.get("name", p.stem), s.get("image"), tags.get("subcategory"), side])

    configs = [args.config] if args.config else sorted(counts)
    grand_missing = 0
    for cfg in configs:
        spec = get_spec(cfg)
        rows, missing_total = [], 0
        for cat, subs in spec.taxonomy.items():
            for s in subs:
                n = counts[cfg][s]
                miss = max(0, args.target - n)
                missing_total += miss
                rows.append((cat, s, n, miss))
        total = sum(counts[cfg].values())
        print(f"\n{cfg}: {total} labelled images, {len(rows)} subcategories, target {args.target} each")
        print(f"  {'category':22s} {'subcategory':20s} {'have':>5s} {'need':>5s} {'small':>6s}")
        for cat, s, n, miss in sorted(rows, key=lambda r: (-r[3], r[0], r[1])):
            print(f"  {cat:22s} {s:20s} {n:5d} {('+%d' % miss) if miss else 'ok':>5s} {small[cfg][s] or '':>6}")
        print(f"  -> {missing_total} more images needed for {cfg}")
        grand_missing += missing_total
    print(f"\nTOTAL still to collect: {grand_missing}")
    print(f"\nshorter side of the labelled images (small = under {args.min_side} px):")
    for batch, c in buckets.items():
        print(f"  {batch:24s} " + "  ".join(f"{k}: {c.get(k, 0)}" for k in ("<400", "400-699", "700-999", ">=1000", "unknown") if c.get(k, 0) or k != "unknown"))
    if args.small_csv:
        args.small_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.small_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["batch", "image", "subcategory", "shorter_side"])
            w.writerows(sorted(small_rows, key=lambda r: (r[2] or "", r[3])))
        print(f"{len(small_rows)} small images listed in {args.small_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
