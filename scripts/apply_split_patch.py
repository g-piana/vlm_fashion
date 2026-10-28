"""
Replace the random 5% split in prepare_dataset.py with the sticky,
subcategory-stratified one from splits.py.

    python scripts/apply_split_patch.py      (from the project root, scripts/splits.py in place)

Adds  --resplit  (rebuild the frozen test set; normally never needed) and writes
data/splits/test_manifest.csv + data/splits/test_labels.json on every run.
Optional keys in the manifest's `splits:` section:
    test_per_subcategory: 2     min_images_for_test: 5
A .bak5 copy of prepare_dataset.py is kept; nothing is written if an anchor is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "prepare_dataset.py"

NEW_BLOCK = '''    # Sticky split: stratified by (category_config, subcategory); test images are frozen
    # in configs/split_registry.json and only NEW images can ever join the test set.
    parts = assign_splits(
        all_samples,
        Path(manifest.get("split_registry", "configs/split_registry.json")),
        test_per_subcategory=splits_cfg.get("test_per_subcategory", 2),
        min_images_for_test=splits_cfg.get("min_images_for_test", 5),
        val_ratio=val_ratio, seed=seed, resplit=args.resplit,
    )
    train, val, test = parts["train"], parts["val"], parts["test"]
    log.info("Split: train=%d, val=%d, test=%d", len(train), len(val), len(test))
    write_test_files(test, output_dir)
'''


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "assign_splits" in text:
        print("already patched")
        return 0
    problems = []

    start = "    # Stratified split by category_config"
    end = '    log.info("Split: train=%d, val=%d, test=%d", len(train), len(val), len(test))\n'
    if text.count(start) != 1 or text.count(end) != 1:
        problems.append(f"split block markers found {text.count(start)}x / {text.count(end)}x (expected 1 / 1)")
    else:
        a, b = text.index(start), text.index(end) + len(end)
        text = text[:a] + NEW_BLOCK + text[b:]

    for old, new, label in (
        ("from taxonomy_utils import validate_samples",
         "from taxonomy_utils import validate_samples\nfrom splits import assign_splits, write_test_files", "import"),
        ("args = parser.parse_args()",
         'parser.add_argument("--resplit", action="store_true",\n'
         '                        help="rebuild the frozen test set (normally never needed)")\n'
         "    args = parser.parse_args()", "--resplit option"),
    ):
        if text.count(old) != 1:
            problems.append(f"{label}: anchor found {text.count(old)}x (expected 1): {old[:50]!r}")
        else:
            text = text.replace(old, new, 1)

    if problems:
        print("NOTHING WRITTEN:")
        for p in problems:
            print("  -", p)
        return 1
    TARGET.with_suffix(".py.bak5").write_text(TARGET.read_text(encoding="utf-8"), encoding="utf-8")
    TARGET.write_text(text, encoding="utf-8")
    print("prepare_dataset.py patched (backup: prepare_dataset.py.bak5)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
