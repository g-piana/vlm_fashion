"""Find near-duplicate images in a new batch, within the batch and against all earlier batches.

Product shots are mostly plain background, so a plain image hash matches different shoes
with a similar outline. This version:
  1. crops each image to the product (trims the near-uniform background),
  2. compares a fine 16x16 perceptual hash of the crop (shape and detail),
  3. AND compares the average colour of the crop (rejects a navy shoe vs a tan shoe).
A pair is reported only when both agree.

Usage:
    python scripts/check_duplicates.py data/raw/images/shoes_women_batch04
    python scripts/check_duplicates.py data/raw/images/bags_women_batch04 --max-dist 60
    python scripts/check_duplicates.py DIR --only-new        # skip earlier batches
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imagehash
import numpy as np
import yaml
from PIL import Image, ImageChops

EXT = {".jpg", ".jpeg", ".png", ".webp"}


def crop_to_product(img: Image.Image, tol: int = 18) -> Image.Image:
    """Trim the background: everything close to the corner colour."""
    rgb = img.convert("RGB")
    a = np.asarray(rgb).astype(int)
    corners = np.array([a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]])
    bg = np.median(corners, axis=0)
    mask = (np.abs(a - bg).max(axis=2) > tol)
    if mask.sum() < 0.002 * mask.size:          # nothing found: keep the whole image
        return rgb
    ys, xs = np.where(mask)
    pad = 4
    box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
           min(xs.max() + pad, a.shape[1]), min(ys.max() + pad, a.shape[0]))
    return rgb.crop(box)


def signature(path: Path):
    crop = crop_to_product(Image.open(path))
    ph = imagehash.phash(crop, hash_size=16)               # 256-bit, shape/detail
    small = np.asarray(crop.resize((32, 32))).astype(float)
    colour = small.reshape(-1, 3).mean(axis=0)             # average RGB of the product
    aspect = crop.width / max(crop.height, 1)
    return ph, colour, aspect


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("new_dir", type=Path)
    ap.add_argument("--manifest", type=Path, default=Path("configs/dataset_manifest.yaml"))
    ap.add_argument("--max-dist", type=int, default=48, help="max hash distance out of 256 (default 48; unrelated shoes score ~100+)")
    ap.add_argument("--max-colour", type=float, default=12.0, help="max average-colour difference (0-441)")
    ap.add_argument("--max-aspect", type=float, default=0.08, help="max relative difference in crop shape")
    ap.add_argument("--only-new", action="store_true", help="compare the new folder with itself only")
    a = ap.parse_args()

    dirs = [a.new_dir]
    if not a.only_new and a.manifest.exists():
        m = yaml.safe_load(open(a.manifest))
        dirs += [Path(b["images_dir"]) for b in m.get("batches", [])]
    dirs = list(dict.fromkeys(d.resolve() for d in dirs if d.exists()))
    new_dir = a.new_dir.resolve()

    sig = {}
    for d in dirs:
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in EXT:
                try:
                    sig[p] = signature(p)
                except Exception as e:
                    print("skip", p.name, e)

    new = [p for p in sig if p.parent == new_dir]
    order = {p: i for i, p in enumerate(new)}
    rows, flagged = [], set()
    for x in new:
        hx, cx, ax = sig[x]
        for y, (hy, cy, ay) in sig.items():
            if y == x or (y.parent == new_dir and order[y] <= order[x]):
                continue
            dist = hx - hy
            if dist > a.max_dist:
                continue
            cdiff = float(np.linalg.norm(cx - cy))
            adiff = abs(ax - ay) / max(ax, ay)
            if cdiff > a.max_colour or adiff > a.max_aspect:
                continue
            where = "same batch" if y.parent == new_dir else y.parent.name
            rows.append((dist, cdiff, x.name, y.name, where))
            flagged.add(x.name)

    for dist, cdiff, xn, yn, where in sorted(rows):
        print(f"hash {dist:3d}  colour {cdiff:5.1f}   {xn}  ~  {yn}  ({where})")
    print(f"\n{len(rows)} pairs, {len(flagged)} images in {a.new_dir.name} have a near-duplicate")
    print("hash < 40 and colour < 5: very likely the same photo; above that, open both before dropping")


if __name__ == "__main__":
    main()
