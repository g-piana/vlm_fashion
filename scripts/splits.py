"""
Sticky, subcategory-stratified train/val/test assignment.

Why: a random 5% test split leaves most subcategories with no test image at all,
and every new batch would reshuffle it. Here
  * every (category_config, subcategory) group gets `test_per_subcategory` test images
    (default 2) as soon as it has at least `min_images_for_test` images (default 5),
    always leaving >= `min_train_per_subcategory` for training;
  * the assignment is stored in a registry file (configs/split_registry.json): an image
    that was once train/val/test stays there, so the test set is frozen and only NEW
    images can ever be added to it;
  * the rest of the new images go to val (about `val_ratio`, deterministic by hash) or train.

Commit configs/split_registry.json: it is your frozen test set.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

log = logging.getLogger(__name__)


def sample_id(s: dict) -> str:
    return f'{s.get("_batch", "unknown")}/{s["image"]}'


def group_key(s: dict) -> tuple[str, str]:
    tags = s.get("tags") if isinstance(s.get("tags"), dict) else {}
    return (s.get("category_config") or "unknown", str(tags.get("subcategory") or "?"))


def assign_splits(samples: list[dict], registry_path: Path, test_per_subcategory: int = 2,
                  min_images_for_test: int = 5, min_train_per_subcategory: int = 3,
                  val_ratio: float = 0.10, seed: int = 42, resplit: bool = False) -> dict:
    registry: dict[str, str] = {}
    if registry_path.exists() and not resplit:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    elif registry_path.exists() and resplit:
        backup = registry_path.with_suffix(".json.bak")
        backup.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
        log.warning("--resplit: the frozen test set is being rebuilt (old registry saved as %s). "
                    "Scores of earlier adapters are no longer comparable.", backup)

    by_id = {sample_id(s): s for s in samples}
    split_of = {i: registry[i] for i in by_id if i in registry}

    groups: dict[tuple, list[str]] = defaultdict(list)
    for i in sorted(by_id):
        groups[group_key(by_id[i])].append(i)

    # test quota: only images that have never been assigned can become test images
    for key, ids in sorted(groups.items()):
        new_here = [i for i in ids if i not in split_of]
        if not new_here:
            continue
        total = len(ids)
        existing_test = sum(1 for i in ids if split_of.get(i) == "test")
        if total < min_images_for_test:
            k = 0
        else:
            k = max(0, min(test_per_subcategory - existing_test, len(new_here),
                           total - existing_test - min_train_per_subcategory))
        rng = random.Random(f"{seed}-{key[0]}-{key[1]}")
        for i in (rng.sample(new_here, k) if k else []):
            split_of[i] = "test"

    # everything else that is new: val (deterministic by hash) or train
    for i in sorted(by_id):
        if i not in split_of:
            h = int(hashlib.md5(f"{seed}:{i}".encode()).hexdigest(), 16) % 10_000 / 10_000
            split_of[i] = "val" if h < val_ratio else "train"

    registry.update(split_of)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(dict(sorted(registry.items())), indent=0), encoding="utf-8")

    parts = {"train": [], "val": [], "test": []}
    for i in sorted(by_id):
        parts[split_of[i]].append(by_id[i])

    # report
    log.info("Split registry: %s (commit it: it is your frozen test set)", registry_path)
    per_cfg: dict[str, Counter] = defaultdict(Counter)
    test_n: Counter = Counter()
    for name, items in parts.items():
        for s in items:
            per_cfg[group_key(s)[0]][name] += 1
            if name == "test":
                test_n[group_key(s)] += 1
    for cfg, c in sorted(per_cfg.items()):
        log.info("  %s: train=%d val=%d test=%d", cfg, c["train"], c["val"], c["test"])
    no_test = [f"{k[1]} ({len(v)})" for k, v in sorted(groups.items()) if test_n[k] == 0]
    if no_test:
        log.info("  subcategories with NO test image yet (too few images; they train but are not scored): %s",
                 ", ".join(no_test))
    return parts


def write_test_files(test_samples: list[dict], output_dir: Path) -> None:
    """test_manifest.csv (for infer.py --manifest) and test_labels.json (ground truth for eval_heldout.py)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "test_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "category_config", "image_type"])
        for s in test_samples:
            w.writerow([s["_abs_image_path"], s.get("category_config", "shoes_women"), s.get("image_type", "product")])
    labels = [{"image": s["image"], "category_config": s.get("category_config", "shoes_women"),
               "image_type": s.get("image_type", "product"), "description": s.get("description", ""),
               "tags": s.get("tags")} for s in test_samples]
    (output_dir / "test_labels.json").write_text(json.dumps(labels, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s and %s (%d test images)", output_dir / "test_manifest.csv",
             output_dir / "test_labels.json", len(test_samples))
