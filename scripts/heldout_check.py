"""
Leakage guard: make sure no held-out / candidate image is also a TRAINING image.

Catches: same filename, byte-identical files, and near-duplicates (same picture
resized/re-encoded) via a perceptual hash. It cannot catch the same product in
another colourway or angle: look at product codes yourself.

    python scripts/heldout_check.py \
        --candidates data/heldout/bags_women/images \
        --manifest configs/dataset_manifest.yaml

Run it (a) when you pick the held-out images and (b) before every training run.
Use --also DIR for extra folders to compare against (repeatable).
Exit code 1 when anything suspicious is found.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml
from PIL import Image

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def dhash(p: Path, size: int = 8) -> int:
    with Image.open(p) as im:
        im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        px = im.tobytes()
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | (px[row * (size + 1) + col] > px[row * (size + 1) + col + 1])
    return bits


def files(d: Path) -> list[Path]:
    return sorted(f for f in d.rglob("*") if f.is_file() and f.suffix.lower() in EXT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=Path("configs/dataset_manifest.yaml"))
    ap.add_argument("--also", type=Path, action="append", default=[])
    ap.add_argument("--max-distance", type=int, default=5, help="max Hamming distance (of 64) for near-duplicates")
    args = ap.parse_args()

    train_dirs = [Path(b["images_dir"]) for b in (yaml.safe_load(open(args.manifest)) or {}).get("batches", [])]
    train_dirs += args.also
    train = [f for d in train_dirs if d.exists() for f in files(d)]
    cand = files(args.candidates)
    print(f"{len(cand)} candidate images vs {len(train)} reference images from {len(train_dirs)} folders")

    t_names = {f.name: f for f in train}
    t_md5 = {}
    t_hash = []
    for f in train:
        t_md5.setdefault(md5(f), f)
        try:
            t_hash.append((dhash(f), f))
        except Exception:
            pass

    problems = 0
    for c in cand:
        reasons = []
        if c.name in t_names:
            reasons.append(f"same filename as {t_names[c.name]}")
        h5 = md5(c)
        if h5 in t_md5:
            reasons.append(f"byte-identical to {t_md5[h5]}")
        try:
            hc = dhash(c)
            best = min(((bin(hc ^ h).count("1"), f) for h, f in t_hash), default=None, key=lambda x: x[0])
            if best and best[0] <= args.max_distance:
                reasons.append(f"near-duplicate of {best[1]} (distance {best[0]})")
        except Exception as e:
            reasons.append(f"could not hash ({e})")
        if reasons:
            problems += 1
            print(f"  ! {c.name}: " + "; ".join(reasons))

    # duplicates inside the candidate set itself
    seen: dict[int, Path] = {}
    for c in cand:
        try:
            hc = dhash(c)
        except Exception:
            continue
        for h, other in seen.items():
            if bin(hc ^ h).count("1") <= args.max_distance:
                problems += 1
                print(f"  ! {c.name}: near-duplicate of {other.name} inside the candidate set")
                break
        seen[hc] = c

    print("OK: no leakage found" if not problems else f"{problems} problem(s) found")
    print("(same product in another colour/angle cannot be detected here: compare product codes)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
