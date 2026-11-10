"""
VLM -> app bridge: turns VLM results into an import file for the app's
`import_content_s3` management command (--format json).

Input: either the labels JSON exported by review_labels.py (reviewed), or the raw
infer.py output (drafts: needs --allow-unreviewed and is always imported unpublished).

    python scripts/export_for_app.py \
        --input outputs/reviewed/bags_women_ext01.json \
        --images-dir /mnt/e/vlm_images/bags_ext01 \
        --meta configs/app_export/bags_ext01.yaml \
        --client base \
        --out exports/bags_ext01_import.json

Gates (any failure -> nothing is written; --drop-invalid excludes the failing items instead):
  1. training gate   taxonomy_utils.validate_samples(strict): same check as prepare_dataset.py
  2. app gate        category, subcategory and every attribute value must exist in the app
                     vocabulary of --client (inflowe-taxonomy), after the `app: rename/skip` rules
  3. metadata        only whitelisted importer columns (never id or image_type)
  4. files           image must exist; duplicate slugs; existing --out needs --force
Output: {"_meta": {client, taxonomy_fingerprint, source_sha256, ...}, "items": [row, ...]}
The patched importer checks the fingerprint against its own taxonomy before importing.

Meta YAML:
    defaults:                 # any whitelisted importer column, applied to every item
      brand: Gucci
      reference_codes: [GUCCI_BAGS_02]
      season: spring
      year: "2019"
      published: false        # default false: check in the app, then publish
      gender: women           # optional ASSERTIONS: must match the category config, never override it
      macrocategory: bags
    per_image_csv: path.csv   # optional: key column `slug` (item slug) or `image` (file name)
                              # + whitelisted columns; overrides defaults. List values comma-separated.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from category_config import load_category_config  # noqa: E402
from reconcile_taxonomy import IGNORED, app_values, field_of  # noqa: E402
from taxonomy_utils import validate_samples  # noqa: E402
from inflowe_taxonomy import ConfigurationLoader, Vocabulary  # noqa: E402

FORMAT = "inflowe-import/1"
# app field (reconcile_taxonomy naming) -> importer column
COLUMN = {"colours": "colour", "material": "material", "pattern": "pattern", "finishing": "finishing",
          "embellishments": "embellishments", "mod_details": "mod_details", "style": "style"}
# metadata columns the bridge may pass through (never id, image_type, render_size, slug, name)
META_COLUMNS = {"brand", "reference_codes", "published", "season", "year", "collection", "people",
                "people_group", "icon", "icon_group", "style", "show_exit", "article", "title",
                "product_purpose", "line", "decade"}
# allowed in metadata only as a cross-check: they always come from the category config
CHECK_COLUMNS = {"gender", "macrocategory"}


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def clean_text(s, limit: int = 4000) -> str:
    s = re.sub(r"<[^>]*>", "", str(s or ""))                      # no markup reaches the app
    s = "".join(ch for ch in s if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return re.sub(r"[ \t]+", " ", s).strip()[:limit]


def as_values(v) -> list[str]:
    vals = v if isinstance(v, list) else [v]
    return [str(x) for x in vals if x not in IGNORED and str(x) not in IGNORED]


def load_meta(path: Path | None) -> tuple[dict, dict[str, dict]]:
    """Returns (defaults, per-item overrides keyed by item slug)."""
    if not path:
        return {}, {}
    allowed = META_COLUMNS | CHECK_COLUMNS
    meta = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = meta.get("defaults") or {}
    bad = set(defaults) - allowed
    if bad:
        sys.exit(f"meta defaults: columns not allowed: {sorted(bad)} (allowed: {sorted(allowed)})")
    per_item: dict[str, dict] = {}
    if meta.get("per_image_csv"):
        p = Path(meta["per_image_csv"])
        p = p if p.is_absolute() else path.parent / p
        with open(p, newline="", encoding="utf-8-sig") as f:
            for n, r in enumerate(csv.DictReader(f), start=2):
                key = r.pop("slug", None) or Path(r.pop("image", None) or "").stem
                r.pop("image", None)
                bad = set(r) - allowed
                if bad:
                    sys.exit(f"{p.name}: columns not allowed: {sorted(bad)}")
                if not key:
                    sys.exit(f"{p.name} line {n}: needs a `slug` or `image` value")
                k = slug(key)
                if k in per_item:
                    sys.exit(f"{p.name} line {n}: duplicate key {k!r}")
                per_item[k] = {c: v.strip() for c, v in r.items() if v not in (None, "") and v.strip()}
    return defaults, per_item


class Bridge:
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab
        self.cfgs: dict[str, dict] = {}
        self.trees: dict[tuple, dict] = {}
        self.allowed: dict[tuple, set] = {}

    def cfg(self, name: str) -> dict:
        if name not in self.cfgs:
            self.cfgs[name] = load_category_config(name)
        return self.cfgs[name]

    def tree(self, gender: str, macro: str) -> dict:
        if (gender, macro) not in self.trees:
            self.trees[(gender, macro)] = self.vocab.taxonomy(gender, macro)
        return self.trees[(gender, macro)]

    def app_set(self, field: str, macro: str | None) -> set:
        key = (field, macro if field == "mod_details" else None)
        if key not in self.allowed:
            self.allowed[key] = app_values(self.vocab, field, key[1])
        return self.allowed[key]

    def row(self, item: dict, cfg_name: str, meta: dict, images_dir: Path, auto_title: bool) -> tuple[dict, list[str]]:
        errs: list[str] = []
        cfg = self.cfg(cfg_name)
        gender, macro = cfg.get("gender"), cfg.get("macrocategory")
        tags = item.get("tags")
        image = Path(str(item.get("image", ""))).name
        stem = Path(image).stem
        row: dict = {"name": stem, "slug": slug(stem), "gender": gender, "macrocategory": macro}

        # taxonomy: category slug + subcategory SLUG resolved through the app tree
        tree = self.tree(gender, macro)
        cat, sub = tags.get("category"), tags.get("subcategory")
        if cat not in tree:
            errs.append(f"category {cat!r} not in app for {gender}/{macro}")
        else:
            row["category"] = cat
            subs = {e.name: e.slug for e in tree[cat]}
            if subs:
                if sub in subs:
                    row["subcategory"] = subs[sub]
                else:
                    errs.append(f"subcategory {sub!r} not under {cat!r} in app")
            elif sub:
                errs.append(f"category {cat!r} has no subcategories in app, got {sub!r}")

        # attributes -> importer columns, via the same mapping the reconciliation used
        cols: dict[str, list[str]] = {}
        for attr, spec in (cfg.get("attributes") or {}).items():
            values = as_values(tags.get(attr))
            if not values:
                continue
            field = field_of(spec, attr)
            if field is None:          # e.g. bag_size: not stored in the app
                continue
            col = COLUMN.get(field)
            if col is None:
                sys.exit(f"config error: app field {field!r} of attribute {attr!r} has no importer column")
            app = spec.get("app") if isinstance(spec, dict) and isinstance(spec.get("app"), dict) else {}
            rename, skip = app.get("rename") or {}, set(app.get("skip") or [])
            allowed = self.app_set(field, macro)
            for v in values:
                if v in skip:
                    continue
                v = str(rename.get(v, v))
                if v not in allowed:
                    errs.append(f"{attr}={v!r} not in app {field}" + (f"[{macro}]" if field == "mod_details" else ""))
                elif v not in cols.setdefault(col, []):
                    cols[col].append(v)
        primary = tags.get("primary_color")
        if primary in cols.get("colour", []):
            cols["colour"].remove(primary)
            cols["colour"].insert(0, primary)
            row["dominant_colour"] = [primary]
        row.update({k: v for k, v in cols.items() if v})

        desc = clean_text(item.get("description"))
        if not desc:
            errs.append("empty description")
        row["description"] = desc

        meta = dict(meta)
        for k in CHECK_COLUMNS & set(meta):
            if str(meta.pop(k)).strip().lower() != str(row[k]).lower():
                errs.append(f"metadata {k} does not match the category config ({row[k]!r})")
        row.update(meta)
        if "published" in row:
            row["published"] = str(row["published"]).strip().lower() in ("true", "1", "yes", "y", "t")
        else:
            row["published"] = False
        if not row.get("title") and auto_title:
            label = f"{(primary or '').replace('-', ' ')} {sub or cat or ''}".strip()
            row["title"] = clean_text(label.title(), 120)

        path = images_dir / image
        if not path.is_file():
            errs.append(f"image not found: {path}")
        else:
            try:
                with Image.open(path) as im:
                    row["img_width"], row["img_height"] = im.size
            except Exception as e:  # corrupt / not an image
                errs.append(f"unreadable image: {e}")
        row["image"] = image
        return row, errs


def main() -> int:
    ap = argparse.ArgumentParser(description="VLM results -> app import JSON")
    ap.add_argument("--input", type=Path, required=True, help="review_labels export or infer.py output")
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--meta", type=Path, help="metadata YAML (brand, season, ...)")
    ap.add_argument("--client", default="base", help="inflowe-taxonomy client (base, nillab, ...)")
    ap.add_argument("--data", help="taxonomy data dir (default: inflowe-taxonomy package)")
    ap.add_argument("--category", help="category config for items without category_config")
    ap.add_argument("--adapter", help="adapter tag recorded in _meta, e.g. qwen3vl_v5")
    ap.add_argument("--allow-unreviewed", action="store_true", help="accept raw infer.py drafts (forced unpublished)")
    ap.add_argument("--allow-warnings", action="store_true", help="vocabulary warnings do not block (errors still do)")
    ap.add_argument("--drop-invalid", action="store_true", help="exclude failing items instead of aborting")
    ap.add_argument("--no-auto-title", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite --out")
    a = ap.parse_args()

    if a.out.exists() and not a.force:
        sys.exit(f"{a.out} exists (use --force)")
    raw = a.input.read_bytes()
    items = json.loads(raw)
    if not isinstance(items, list):
        sys.exit("input must be a JSON list")
    drafts = any(isinstance(i, dict) and ("validation" in i or "error" in i) for i in items)
    if drafts and not a.allow_unreviewed:
        sys.exit("input looks like raw infer.py drafts: review it first, or pass --allow-unreviewed")
    defaults, per_item = load_meta(a.meta)
    used_keys: set[str] = set()

    vocab = Vocabulary(a.client, ConfigurationLoader(a.data))
    bridge = Bridge(vocab)
    rows, failed, samples = [], {}, []
    for it in items:
        image = Path(str(it.get("image", ""))).name
        cfg_name = it.get("category_config") or a.category
        problems = []
        if it.get("error") or not isinstance(it.get("tags"), dict):
            problems.append(f"no usable tags ({it.get('error') or 'look or unparsed output'})")
        elif not cfg_name:
            problems.append("no category_config (pass --category)")
        elif drafts and isinstance(it.get("validation"), dict) and not it["validation"].get("ok", True):
            problems.append(f"infer validation failed: {it['validation'].get('issues')}")
        if problems:
            failed[image] = problems
            continue
        meta = {**defaults, **per_item.get(slug(Path(image).stem), {})}
        used_keys.add(slug(Path(image).stem))
        if drafts:
            meta["published"] = False
        row, errs = bridge.row(it, cfg_name, meta, a.images_dir, not a.no_auto_title)
        if errs:
            failed[image] = errs
            continue
        rows.append(row)
        samples.append({"image": image, "category_config": cfg_name, "image_type": it.get("image_type", "product"),
                        "description": row["description"], "tags": it["tags"], "_batch": a.input.stem})

    # training gate on everything that passed the app gate
    if samples and not validate_samples(samples, strict=not a.allow_warnings):
        sys.exit("training-gate validation failed (see report above); nothing written")
    dupes = sorted({r["slug"] for r in rows if sum(x["slug"] == r["slug"] for x in rows) > 1})
    if dupes:
        sys.exit(f"duplicate slugs: {dupes[:10]}")

    unused = sorted(set(per_item) - used_keys)
    if unused:
        print(f"WARNING {len(unused)} metadata rows match no item: {unused[:10]}")
    print(f"{len(items)} items in, {len(rows)} ok, {len(failed)} failing")
    for img, errs in list(failed.items())[:50]:
        print(f"  x {img}: {'; '.join(errs)}")
    if failed and not a.drop_invalid:
        print("nothing written: fix the items above, or pass --drop-invalid to exclude them")
        return 1
    if not rows:
        print("nothing to write")
        return 1

    out = {"_meta": {"format": FORMAT, "client": a.client, "taxonomy_fingerprint": vocab.fingerprint,
                     "source": a.input.name, "source_sha256": hashlib.sha256(raw).hexdigest(),
                     "reviewed": not drafts, "adapter": a.adapter, "count": len(rows),
                     "excluded": sorted(failed), "published_any": any(r["published"] for r in rows),
                     "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
           "items": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(a.out.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(a.out)
    print(f"wrote {a.out} ({len(rows)} items, fingerprint {vocab.fingerprint}, "
          f"{'reviewed' if not drafts else 'UNREVIEWED drafts, unpublished'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
