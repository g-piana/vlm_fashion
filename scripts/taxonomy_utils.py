"""
Taxonomy and vocabulary enforcement — one source of truth for every stage.

The generated category configs (configs/categories/generated/*.yaml, merged
with _base.yaml and the attribute template by category_config.py) define:
  * the taxonomy        {category: [subcategories]}
  * the vocabularies    {attribute: allowed values}

This module checks (and, where it is safe, repairs) a `tags` object against
that definition. It is used by:
  * prepare_dataset.py   -> validate_samples()        (gate before training)
  * infer.py             -> apply_taxonomy_rules()    (repair after generation)
  * evaluate.py          -> evaluate_taxonomy()       (violation + top-down metrics)
  * generate_category_configs.py -> find_duplicate_subcategories()
  * the CLI below        -> audit label files / the whole manifest

Severity levels
  error    category/subcategory missing, unknown, or not a valid pair
           (training on these teaches the model the wrong taxonomy)
  warning  unknown key, value outside the controlled vocabulary, list/scalar
           shape mismatch for attributes declared `type: list`

Usage
    python scripts/taxonomy_utils.py --manifest configs/dataset_manifest.yaml
    python scripts/taxonomy_utils.py --labels data/raw/labels/bags_women_batch01.json \
        --config bags_women
    python scripts/taxonomy_utils.py --manifest configs/dataset_manifest.yaml --strict
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from category_config import load_category_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

CORE_KEYS = ("category", "subcategory")
DEFAULT_CONFIG = "shoes_women"  # same fallback prepare_dataset.py uses


# ── Spec ─────────────────────────────────────────────────────────────────

class ConfigSpec:
    """Taxonomy + vocabularies of one category config, ready for checking."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.taxonomy: dict[str, list[str]] = {
            c: [str(s) for s in (subs or [])]
            for c, subs in config.get("taxonomy", {}).items()
        }
        # subcategory -> categories that contain it
        self.sub_index: dict[str, list[str]] = defaultdict(list)
        for cat, subs in self.taxonomy.items():
            for s in subs:
                self.sub_index[s].append(cat)

        attrs = config.get("attributes", {}) or {}
        self.attrs: dict[str, dict] = {}
        for attr, spec in attrs.items():
            if not isinstance(spec, dict):
                continue
            self.attrs[attr] = {
                "multi": spec.get("type") == "list",
                "values": self._values_of(attrs, attr),
            }
        self.allowed_keys = set(CORE_KEYS) | set(self.attrs)

    @staticmethod
    def _values_of(attrs: dict, name: str, seen: tuple = ()) -> set[str] | None:
        spec = attrs.get(name)
        if not isinstance(spec, dict):
            return None
        vals = spec.get("values")
        if isinstance(vals, list):
            return {str(v) for v in vals if v is not None}
        if isinstance(vals, str) and vals.startswith("same_as:"):
            ref = vals.split(":", 1)[1].strip()
            if ref == name or ref in seen:
                return None
            return ConfigSpec._values_of(attrs, ref, seen + (name,))
        return None  # free-text attribute: no vocabulary to enforce

    def duplicates(self) -> dict[str, list[str]]:
        """Subcategory names that appear under more than one category."""
        return {s: c for s, c in self.sub_index.items() if len(c) > 1}


_SPEC_CACHE: dict[str, ConfigSpec] = {}


def get_spec(config_name: str) -> ConfigSpec:
    if config_name not in _SPEC_CACHE:
        _SPEC_CACHE[config_name] = ConfigSpec(config_name, load_category_config(config_name))
    return _SPEC_CACHE[config_name]


def find_duplicate_subcategories(taxonomy: dict[str, list[str]]) -> dict[str, list[str]]:
    """For the generator: subcategory names shared by several categories."""
    index: dict[str, list[str]] = defaultdict(list)
    for cat, subs in taxonomy.items():
        for s in subs or []:
            index[str(s)].append(cat)
    return {s: c for s, c in index.items() if len(c) > 1}


# ── Validation ───────────────────────────────────────────────────────────

Issue = tuple[str, str]  # (level, message)


def validate_tags(spec: ConfigSpec, tags) -> list[Issue]:
    """Return a list of (level, message). Empty list = fully valid."""
    if not isinstance(tags, dict):
        return [("error", "tags is not an object")]

    issues: list[Issue] = []
    cat, sub = tags.get("category"), tags.get("subcategory")

    # category / subcategory pair
    if cat in (None, ""):
        issues.append(("error", "category is missing"))
    elif not isinstance(cat, str) or cat not in spec.taxonomy:
        issues.append(("error", f"unknown category {cat!r}"))
    else:
        subs = spec.taxonomy[cat]
        if subs:
            if sub in (None, ""):
                issues.append(("error", f"subcategory is missing (category {cat!r} requires one)"))
            elif not isinstance(sub, str) or sub not in subs:
                owners = spec.sub_index.get(sub) if isinstance(sub, str) else None
                if owners:
                    issues.append((
                        "error",
                        f"subcategory {sub!r} belongs to {owners}, not to {cat!r}",
                    ))
                else:
                    issues.append(("error", f"unknown subcategory {sub!r}"))
        elif sub not in (None, ""):
            issues.append(("error", f"category {cat!r} has no subcategories but subcategory is {sub!r}"))

    # attributes
    for key, value in tags.items():
        if key in CORE_KEYS:
            continue
        if key not in spec.attrs:
            issues.append(("warning", f"unknown key {key!r}"))
            continue
        if value is None:
            continue
        info = spec.attrs[key]
        if info["multi"] and not isinstance(value, list):
            issues.append(("warning", f"{key}: expected a list, got {type(value).__name__}"))
        elif not info["multi"] and isinstance(value, list):
            issues.append(("warning", f"{key}: holds a list but is declared `type: enum` "
                                      "(declare `type: list`, or keep a single value)"))
        if info["values"] is None:
            continue
        for item in (value if isinstance(value, list) else [value]):
            if item is not None and str(item) not in info["values"]:
                issues.append(("warning", f"{key}: {item!r} not in vocabulary"))
    return issues


def repair_tags(spec: ConfigSpec, tags, strip_unknown: bool = False):
    """
    Deterministic, safe repairs. Returns (new_tags, changes).

    The subcategory is the more visual decision, so when it is a valid name
    belonging to exactly one category, the category is derived from it.
    Nothing is invented: an invalid subcategory is left alone (and will be
    reported by validate_tags).
    """
    if not isinstance(tags, dict):
        return tags, []
    t = copy.deepcopy(tags)
    changes: list[str] = []

    sub, cat = t.get("subcategory"), t.get("category")
    owners = spec.sub_index.get(sub) if isinstance(sub, str) else None
    if owners:
        if len(owners) == 1 and cat != owners[0]:
            changes.append(f"category {cat!r} -> {owners[0]!r} (derived from subcategory {sub!r})")
            t["category"] = owners[0]
        elif len(owners) > 1 and cat not in owners:
            changes.append(f"subcategory {sub!r} is ambiguous {owners}; category {cat!r} kept")

    # list-declared fields are always lists: null -> [], "x" -> ["x"]  (safe direction only)
    # single-valued fields never keep a one-element list: ["x"] -> "x", [] -> null
    for key, info in spec.attrs.items():
        if key not in t:
            continue
        if info["multi"]:
            if t[key] is None:
                t[key] = []
            elif isinstance(t[key], str):
                changes.append(f"{key}: {t[key]!r} -> [{t[key]!r}]")
                t[key] = [t[key]]
        elif isinstance(t[key], list) and len(t[key]) <= 1:
            changes.append(f"{key}: {t[key]!r} -> single value")
            t[key] = t[key][0] if t[key] else None

    if strip_unknown:
        for key in list(t):
            if key not in spec.allowed_keys:
                changes.append(f"dropped unknown key {key!r}")
                del t[key]
    return t, changes


def apply_taxonomy_rules(result: dict, config_name: str, repair: bool = True) -> dict:
    """
    Post-process one infer.py result. Adds a "validation" entry:
        {"ok": bool, "issues": [...], "repaired": [...]}
    `result["tags"]` is repaired in place when repair=True.
    Look configs (tags is an array) are left untouched.
    """
    tags = result.get("tags")
    if isinstance(tags, list):
        return result
    try:
        spec = get_spec(config_name)
    except FileNotFoundError:
        return result

    if not isinstance(tags, dict):
        result["validation"] = {"ok": False, "issues": ["error: tags missing or unparseable"], "repaired": []}
        return result

    repaired: list[str] = []
    if repair:
        result["tags"], repaired = repair_tags(spec, tags)
    issues = validate_tags(spec, result["tags"])
    result["validation"] = {
        "ok": not any(level == "error" for level, _ in issues),
        "issues": [f"{level}: {msg}" for level, msg in issues],
        "repaired": repaired,
    }
    return result


# ── Dataset gate (prepare_dataset.py) ────────────────────────────────────

def _report(rows: list[dict]) -> None:
    for r in rows:
        print(f"  {r['name']}: {r['n']} samples, {r['errors']} with errors, {r['warnings']} with warnings")
        for msg, count in r["counter"].most_common(8):
            print(f"      {count:4d} x {msg}")
        for fname, msgs in r["examples"][:5]:
            print(f"      e.g. {fname}: {msgs[0]}")


def validate_samples(samples: list[dict], strict: bool = False) -> bool:
    """
    Validate loaded label samples (dicts with image, tags, category_config,
    optional _batch). Prints a summary. Returns True when training may proceed.
    """
    by_batch: dict[str, dict] = {}
    missing_cfg = Counter()
    total_err = total_warn = 0

    for s in samples:
        if not s.get("description"):
            continue  # prepare_dataset skips these anyway
        batch = s.get("_batch", "unknown")
        row = by_batch.setdefault(
            batch, {"name": batch, "n": 0, "errors": 0, "warnings": 0,
                    "counter": Counter(), "examples": []},
        )
        row["n"] += 1

        cfg = s.get("category_config")
        if not cfg:
            missing_cfg[batch] += 1
            cfg = DEFAULT_CONFIG
        tags = s.get("tags")
        if isinstance(tags, list):          # look configs: not checked here
            continue
        if tags is None:
            row["warnings"] += 1
            row["counter"]["warning: no tags"] += 1
            total_warn += 1
            continue
        try:
            spec = get_spec(cfg)
        except FileNotFoundError:
            row["errors"] += 1
            row["counter"][f"error: unknown category_config {cfg!r}"] += 1
            total_err += 1
            continue

        issues = validate_tags(spec, tags)
        if not issues:
            continue
        has_err = any(level == "error" for level, _ in issues)
        has_warn = any(level == "warning" for level, _ in issues)
        row["errors"] += has_err
        row["warnings"] += has_warn
        total_err += has_err
        total_warn += has_warn
        for level, msg in issues:
            row["counter"][f"{level}: {msg}"] += 1
        row["examples"].append((s.get("image", "?"), [f"{lv}: {m}" for lv, m in issues]))

    print("\n── Taxonomy / vocabulary validation ──")
    _report(list(by_batch.values()))
    for batch, n in missing_cfg.items():
        print(f"  NOTE {batch}: {n} samples have no category_config and would use '{DEFAULT_CONFIG}'")

    if total_err:
        log.error("%d samples have taxonomy errors — fix the labels (or use --no-validate)", total_err)
        return False
    if total_warn:
        if strict:
            log.error("%d samples have vocabulary warnings and --strict was given", total_warn)
            return False
        log.warning("%d samples have vocabulary warnings (use --strict to make them fatal)", total_warn)
    else:
        log.info("Validation passed: no issues")
    return True


# ── Evaluation (evaluate.py) ─────────────────────────────────────────────

def evaluate_taxonomy(predictions: list[dict], gt_map: dict) -> None:
    """
    Taxonomy compliance and top-down accuracy.

    Reports how often the model produced an invalid/mismatched pair (before
    repair) and, against ground truth, category accuracy, subcategory
    accuracy, subcategory accuracy given the right category, and the most
    frequent subcategory confusions.
    """
    n = n_invalid = n_repaired = n_issue = 0
    for p in predictions:
        tags = p.get("tags")
        if not isinstance(tags, dict):
            continue
        n += 1
        val = p.get("validation")
        if val is None:  # predictions produced before apply_taxonomy_rules existed
            try:
                spec = get_spec(p.get("category_config") or DEFAULT_CONFIG)
            except FileNotFoundError:
                continue
            issues = validate_tags(spec, tags)
            n_invalid += any(lv == "error" for lv, _ in issues)
            n_issue += bool(issues)
        else:
            n_invalid += not val.get("ok", True)
            n_repaired += bool(val.get("repaired"))
            n_issue += bool(val.get("issues"))

    if n:
        print(f"\n── Taxonomy compliance ({n} predictions with tags) ──")
        print(f"  invalid/mismatched category-subcategory (after repair): {n_invalid} ({n_invalid / n:.1%})")
        print(f"  repaired by derivation from subcategory:                 {n_repaired} ({n_repaired / n:.1%})")
        print(f"  any vocabulary/key issue remaining:                      {n_issue} ({n_issue / n:.1%})")

    cat_ok = sub_ok = sub_ok_given_cat = cat_total = 0
    confusions: Counter = Counter()
    for p in predictions:
        tags, gt = p.get("tags"), (gt_map.get(p.get("image", "")) or {}).get("tags")
        if not isinstance(tags, dict) or not isinstance(gt, dict):
            continue
        gc, gs = gt.get("category"), gt.get("subcategory")
        if gc is None:
            continue
        cat_total += 1
        c_ok = tags.get("category") == gc
        s_ok = tags.get("subcategory") == gs
        cat_ok += c_ok
        sub_ok += s_ok
        sub_ok_given_cat += c_ok and s_ok
        if not s_ok:
            confusions[(gs, tags.get("subcategory"))] += 1

    if cat_total:
        print(f"\n── Top-down accuracy ({cat_total} samples) ──")
        print(f"  category accuracy:                      {cat_ok / cat_total:.3f}")
        print(f"  subcategory accuracy (overall):         {sub_ok / cat_total:.3f}")
        if cat_ok:
            print(f"  subcategory accuracy | category right:  {sub_ok_given_cat / cat_ok:.3f}")
        if confusions:
            print("  top subcategory confusions (truth -> predicted):")
            for (g, pr), c in confusions.most_common(10):
                print(f"      {c:3d} x {g} -> {pr}")


# ── CLI ──────────────────────────────────────────────────────────────────

def _load_manifest_samples(manifest: Path) -> list[dict]:
    with open(manifest, encoding="utf-8") as f:
        batches = (yaml.safe_load(f) or {}).get("batches", [])
    samples = []
    for b in batches:
        path = Path(b["labels"])
        if not path.exists():
            log.warning("Labels file not found, skipping: %s", path)
            continue
        with open(path, encoding="utf-8") as f:
            for s in json.load(f):
                s["_batch"] = b.get("name", path.stem)
                samples.append(s)
    return samples


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate labels against taxonomy + vocabularies")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=Path)
    src.add_argument("--labels", type=Path)
    ap.add_argument("--config", help="category_config to assume when a sample has none (with --labels)")
    ap.add_argument("--strict", action="store_true", help="vocabulary warnings are fatal too")
    args = ap.parse_args()

    if args.manifest:
        samples = _load_manifest_samples(args.manifest)
    else:
        with open(args.labels, encoding="utf-8") as f:
            samples = json.load(f)
        for s in samples:
            s["_batch"] = args.labels.stem
            if args.config and not s.get("category_config"):
                s["category_config"] = args.config
    return 0 if validate_samples(samples, strict=args.strict) else 1


if __name__ == "__main__":
    sys.exit(main())
