"""
Turn model predictions into drafts for BLIND review of a held-out set.

The review tool preselects whatever the draft contains, and people tend to
accept it. For a test set that biases the ground truth towards the model and
inflates its score. This script blanks category/subcategory (and optionally the
description, which often names the bag type) so you must choose them yourself.

    python scripts/make_blind_drafts.py \
        --predictions outputs/heldout_bags_v1.json \
        --output outputs/heldout_bags_blind_drafts.json [--hide-description]

Keep the original predictions file: it is the baseline you score later.
Then review with:
    python scripts/review_labels.py --drafts outputs/heldout_bags_blind_drafts.json \
        --category bags_women --batch-name heldout01 --export-dir data/heldout/bags_women
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PLACEHOLDER = "[held-out review: description not used for scoring]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--hide-description", action="store_true",
                    help="replace the description with a placeholder (it often names the subcategory)")
    ap.add_argument("--blind-all", action="store_true",
                    help="blank every tag (maximum independence, much more work)")
    args = ap.parse_args()

    items = json.loads(args.predictions.read_text(encoding="utf-8"))
    out = []
    for it in items:
        it = dict(it)
        it.pop("validation", None)
        tags = dict(it.get("tags") or {})
        if args.blind_all:
            tags = {k: ([] if isinstance(v, list) else None) for k, v in tags.items()}
        else:
            tags["category"] = None
            tags["subcategory"] = None
        it["tags"] = tags
        if args.hide_description or not it.get("description"):
            it["description"] = PLACEHOLDER
        out.append(it)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} blind drafts written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
