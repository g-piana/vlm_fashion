"""
Snapshot / compare system prompts for every config.

Use it to prove that a code or taxonomy change did not alter any prompt
the trained adapter has already seen (e.g. shoes).

    python scripts/snapshot_prompts.py --out /tmp/prompts_before.json
    # ... make changes, regenerate configs ...
    python scripts/snapshot_prompts.py --out /tmp/prompts_after.json
    python scripts/snapshot_prompts.py --compare /tmp/prompts_before.json /tmp/prompts_after.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from category_config import build_system_prompt, list_configs  # noqa: E402


def snapshot() -> dict[str, str]:
    snap = {}
    for name in list_configs():
        for image_type in ("product", "look"):
            key = f"{name}|{image_type}"
            try:
                prompt = build_system_prompt(name, image_type)
                snap[key] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
            except Exception as e:  # keep going; report as error
                snap[key] = f"ERROR: {type(e).__name__}"
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args()

    if args.compare:
        a = json.loads(args.compare[0].read_text())
        b = json.loads(args.compare[1].read_text())
        changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        unchanged = len(set(a) | set(b)) - len(changed)
        print(f"unchanged: {unchanged}   changed: {len(changed)}")
        for k in changed:
            print("  CHANGED", k)
    elif args.out:
        args.out.write_text(json.dumps(snapshot(), indent=2))
        print(f"wrote {args.out}")
    else:
        ap.error("use --out or --compare")


if __name__ == "__main__":
    main()
