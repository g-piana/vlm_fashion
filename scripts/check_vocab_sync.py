"""
Check that the controlled vocabularies used in the prompts match their
source-of-truth files.

    data/raw/materials.yaml   ->  configs/categories/_base.yaml : attributes.material
    data/raw/colours.yaml     ->  ...                           : attributes.primary_color
    (optional) embellishments / patterns / styles files if present

The prompt, the review tool and taxonomy_utils.py all read _base.yaml, so they
are only as current as that file. Updating it is a manual paste from
extract_vocabularies.py; this script tells you whether that step was done.

Usage:
    python scripts/check_vocab_sync.py
    python scripts/check_vocab_sync.py --source pattern=data/raw/patterns.yaml
Exit code 1 when any vocabulary is out of sync.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "configs" / "categories" / "_base.yaml"

# values added to _base.yaml by hand on purpose (the extractor never produces them)
IGNORE_EXTRA = {"none"}

# attribute in _base.yaml -> default source file
DEFAULT_SOURCES = {
    "material": ROOT / "data/raw/materials.yaml",
    "primary_color": ROOT / "data/raw/colours.yaml",
    "embellishments": ROOT / "data/raw/embellishments.yaml",
    "pattern": ROOT / "data/raw/patterns.yaml",
    "style_tags": ROOT / "data/raw/styles.yaml",
    "finishing": ROOT / "data/raw/finishings.yaml",
}


def _load(path: Path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def source_slugs(data) -> list[str]:
    """Entries may be {key: {slug: ...}}, {key: value}, or a plain list."""
    if isinstance(data, dict):
        return [str(v["slug"]) if isinstance(v, dict) and "slug" in v else str(k)
                for k, v in data.items()]
    return [str(x["slug"]) if isinstance(x, dict) and "slug" in x else str(x) for x in data]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", default=[], metavar="ATTR=PATH",
                    help="override/add a source file for an attribute")
    args = ap.parse_args()

    sources = dict(DEFAULT_SOURCES)
    for item in args.source:
        attr, _, path = item.partition("=")
        sources[attr] = Path(path)

    attrs = (_load(BASE).get("attributes") or {})
    bad = False
    print(f"Comparing sources with {BASE.relative_to(ROOT)}\n")
    for attr, path in sources.items():
        if attr not in attrs or not isinstance(attrs[attr].get("values"), list):
            continue
        if not path.exists():
            print(f"  {attr:15s} skipped (source not found: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path})")
            continue
        src = set(source_slugs(_load(path)))
        base_vals = [str(v) for v in attrs[attr]["values"] if v is not None]
        base = set(base_vals)
        if attr == "embellishments":
            pass
        missing, stale = sorted(src - base), sorted(base - src - IGNORE_EXTRA)
        dups = sorted(v for v, c in Counter(base_vals).items() if c > 1)
        if not missing and not stale:
            print(f"  {attr:15s} OK ({len(base)} values)" + (f"   [duplicates in list: {dups}]" if dups else ""))
            continue
        bad = True
        print(f"  {attr:15s} OUT OF SYNC")
        if missing:
            print(f"      in source, missing from _base.yaml: {missing}")
        if stale:
            print(f"      in _base.yaml, not in source:       {stale}")
    # per-macrocategory lists in configs/attributes/*.yaml must be subsets of the source
    print("\nPer-macrocategory lists (configs/attributes/*.yaml):")
    for tmpl in sorted((ROOT / "configs" / "attributes").glob("*.yaml")):
        spec = _load(tmpl)
        for attr, path in sources.items():
            vals = (spec.get(attr) or {}).get("values") if isinstance(spec.get(attr), dict) else None
            if not isinstance(vals, list):
                continue
            if not path.exists():
                print(f"  {tmpl.name}: {attr} NOT CHECKED (source file not found: {path})")
                continue
            unknown = sorted(str(v) for v in vals if v is not None and str(v) not in set(source_slugs(_load(path))))
            if unknown:
                bad = True
                print(f"  {tmpl.name}: {attr} has values not in the source file: {unknown}")
            else:
                print(f"  {tmpl.name}: {attr} OK ({len(vals)} values)")
    if bad:
        print("\nRe-run scripts/extract_vocabularies.py and paste the block into _base.yaml.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
