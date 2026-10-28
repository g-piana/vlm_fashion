"""
How many labelled images does each subcategory have, and how many are missing
to reach a target?  Run it before collecting images so you gather exactly what
is needed in ONE round.

    python scripts/coverage.py --manifest configs/dataset_manifest.yaml --target 8
    python scripts/coverage.py --manifest configs/dataset_manifest.yaml --target 8 --config bags_women

Counts come from the reviewed labels listed in the manifest; the subcategory
list comes from the generated config, so empty subcategories show up too.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy_utils import DEFAULT_CONFIG, get_spec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("configs/dataset_manifest.yaml"))
    ap.add_argument("--target", type=int, default=8, help="desired labelled images per subcategory")
    ap.add_argument("--config", help="only this category_config")
    args = ap.parse_args()

    counts: dict[str, Counter] = defaultdict(Counter)
    for b in (yaml.safe_load(open(args.manifest, encoding="utf-8")) or {}).get("batches", []):
        p = Path(b["labels"])
        if not p.exists():
            continue
        for s in json.load(open(p, encoding="utf-8")):
            tags = s.get("tags")
            if isinstance(tags, dict):
                counts[s.get("category_config") or DEFAULT_CONFIG][tags.get("subcategory")] += 1

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
        print(f"  {'category':22s} {'subcategory':20s} {'have':>5s} {'need':>5s}")
        for cat, s, n, miss in sorted(rows, key=lambda r: (-r[3], r[0], r[1])):
            print(f"  {cat:22s} {s:20s} {n:5d} {('+%d' % miss) if miss else 'ok':>5s}")
        print(f"  -> {missing_total} more images needed for {cfg}")
        grand_missing += missing_total
    print(f"\nTOTAL still to collect: {grand_missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
