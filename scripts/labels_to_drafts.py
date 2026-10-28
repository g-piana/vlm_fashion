"""
Re-open already reviewed labels in the review tool under the CURRENT schema.

Use after you changed the attribute list (new field such as `finishing`, a
dropped field such as `season`): keys that no longer exist are removed, new
keys are added empty ([] for list fields, null otherwise), and everything you
already reviewed stays preselected. You only add what is new.

    python scripts/labels_to_drafts.py \
        --labels data/raw/labels/shoes_women_batch01.json \
        --images-dir data/raw/images/shoes_women_batch01 \
        --output outputs/shoes_women_batch01_redraft.json
    python scripts/review_labels.py --drafts outputs/shoes_women_batch01_redraft.json \
        --category shoes_women --batch-name batch01

(Exporting from the review tool with the same category/batch name rewrites
data/raw/labels/shoes_women_batch01.json; a .bak copy of the old file is made here.)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy_utils import DEFAULT_CONFIG, get_spec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    items = json.loads(args.labels.read_text(encoding="utf-8"))
    out, added, dropped = [], {}, {}
    for it in items:
        cfg = it.get("category_config") or DEFAULT_CONFIG
        spec = get_spec(cfg)
        tags = dict(it.get("tags") or {})
        for k in list(tags):
            if k not in spec.allowed_keys:
                dropped[k] = dropped.get(k, 0) + 1
                del tags[k]
        for attr, info in spec.attrs.items():
            if attr not in tags:
                tags[attr] = [] if info["multi"] else None
                added[attr] = added.get(attr, 0) + 1
            elif info["multi"]:
                if tags[attr] is None:
                    tags[attr] = []
                elif isinstance(tags[attr], str):
                    tags[attr] = [tags[attr]]
            elif isinstance(tags[attr], list) and len(tags[attr]) <= 1:
                tags[attr] = tags[attr][0] if tags[attr] else None
        out.append({
            "image": str(args.images_dir / it["image"]),
            "category_config": cfg,
            "image_type": it.get("image_type", "product"),
            "description": it.get("description", ""),
            "tags": tags,
        })
    shutil.copy2(args.labels, args.labels.with_suffix(args.labels.suffix + ".bak"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} drafts written to {args.output}")
    print("  keys added (new in schema):  ", added or "none")
    print("  keys dropped (not in schema):", dropped or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
