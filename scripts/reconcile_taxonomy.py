"""
One-off (and re-runnable) consolidation of the VLM vocabularies with the app taxonomy.

Compares every value the VLM can produce with the resolved app taxonomy of a client
(inflowe-taxonomy package) and writes:
  consolidation/report.md        what matches, what differs, what the app has that the VLM never produces
  consolidation/decisions.yaml   one entry per mismatch, pre-filled with a suggestion

In decisions.yaml you set `standard:` for each entry, the single name both sides will use:
  standard == app_now          the VLM adopts the app name   -> labels migrated
  app_now is null              the app adopts the VLM name   -> add it to base/<file>.yaml
  standard differs from both   both converge on a new name   -> rename the app row + migrate labels
  standard: null               the VLM stops producing it    -> value removed from labels
`app_now` is pre-filled only for confident matches; fuzzy candidates are listed under `maybe`.

Usage:
    python scripts/reconcile_taxonomy.py --client base
    python scripts/reconcile_taxonomy.py --client base --out consolidation --force
"""

from __future__ import annotations

import argparse
import difflib
from collections import defaultdict
from pathlib import Path

import yaml

from inflowe_taxonomy import Vocabulary, slugify

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IGNORED = {None, "none", "null", ""}

# VLM attribute -> app import field. Overridden by an `app: {field: ...}` block on the attribute.
DEFAULT_FIELDS = {
    "primary_color": "colours", "secondary_colors": "colours",
    "material": "material", "secondary_materials": "material",
    "embellishments": "embellishments", "pattern": "pattern",
    "finishing": "finishing", "style_tags": "style",
    "model_details": "mod_details", "handle_type": "mod_details",
    "bag_closure": "mod_details", "hardware_color": "mod_details",
}
# suggestion hints only (the decision is yours)
HINTS = {"hardware_color": "{v}-hardware"}
# attribute template file -> app macrocategory slug
TEMPLATE_MACRO = {"ready_to_wear": "ready-to-wear"}


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_values(attrs: dict, name: str) -> list[str]:
    spec = attrs.get(name) or {}
    vals = spec.get("values") if isinstance(spec, dict) else None
    if isinstance(vals, str) and vals.startswith("same_as:"):
        return resolve_values(attrs, vals.split(":", 1)[1])
    return [str(v) for v in vals or [] if v not in IGNORED and str(v) not in IGNORED]


def field_of(spec: dict, attr: str) -> str | None:
    app = spec.get("app") if isinstance(spec, dict) else None
    if isinstance(app, dict) and "field" in app:
        return app["field"]
    return DEFAULT_FIELDS.get(attr)


def suggest(value: str, attr: str, candidates: set[str]) -> tuple[str | None, list[str], str]:
    """(confident app match or None, fuzzy alternatives, note). Only confident matches pre-fill."""
    if attr in HINTS:
        hinted = HINTS[attr].format(v=value)
        if hinted in candidates:
            return hinted, [], "naming rule"
        return None, [], f"naming rule: app should get '{hinted}'"
    if slugify(value) in candidates:
        return slugify(value), [], "same value, different spelling"
    longer = sorted(c for c in candidates if value in c.split("-"))
    if len(longer) == 1:
        return longer[0], [], "app name extends the VLM name"
    close = difflib.get_close_matches(value, sorted(candidates), n=3, cutoff=0.8)
    if close:
        return None, close, "similar app value(s) listed in maybe: check before choosing"
    return None, [], "not in the app"


def collect_vlm(configs: Path) -> tuple[dict, dict, list[tuple[str, dict]]]:
    """VLM values per (app field, macro) -> {value: set(vlm attrs)}, unmapped attrs, generated configs."""
    base_attrs = (load_yaml(configs / "categories" / "_base.yaml").get("attributes") or {})
    values: dict[tuple, dict] = defaultdict(lambda: defaultdict(set))
    unmapped: dict[str, list[str]] = defaultdict(list)

    def add(attrs: dict, macro: str | None, where: str) -> None:
        lookup = {**base_attrs, **attrs}  # same_as: may point at a base attribute
        for attr, spec in attrs.items():
            field = field_of(spec, attr)
            if field is None:
                if not (isinstance(spec, dict) and isinstance(spec.get("app"), dict)):
                    unmapped[where].append(attr)
                continue
            key = (field, macro if field == "mod_details" else None)
            for v in resolve_values(lookup, attr):
                values[key][v].add(attr)

    add(base_attrs, None, "_base")
    for f in sorted((configs / "attributes").glob("*.yaml")):
        if " " in f.stem:  # e.g. "bags copy.yaml"
            continue
        add(load_yaml(f), TEMPLATE_MACRO.get(f.stem, f.stem), f.name)

    generated = [(f.stem, load_yaml(f)) for f in sorted((configs / "categories" / "generated").glob("*.yaml"))]
    return values, unmapped, generated


def app_values(v: Vocabulary, field: str, macro: str | None) -> set[str]:
    if field == "mod_details":
        return {e.slug for e in v.mod_details(macro)}
    attr = next((a for a, s in v.loader.registry.items() if s.get("import_field") == field), None)
    return v.slugs(attr) if attr else set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", default="base")
    ap.add_argument("--configs", type=Path, default=PROJECT_ROOT / "configs")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "consolidation")
    ap.add_argument("--data", help="taxonomy data dir (default: inflowe-taxonomy package)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing decisions.yaml")
    args = ap.parse_args()

    from inflowe_taxonomy import ConfigurationLoader
    vocab = Vocabulary(args.client, ConfigurationLoader(args.data))
    vlm, unmapped, generated = collect_vlm(args.configs)

    decisions: dict[str, dict] = {}
    report = [f"# Taxonomy reconciliation — client `{args.client}` (fingerprint {vocab.fingerprint})", ""]

    # attribute vocabularies
    report += ["## Attribute values", "",
               "| App field | VLM values | Match | To decide | App-only |", "|---|---|---|---|---|"]
    for (field, macro), values in sorted(vlm.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        label = f"{field}[{macro}]" if macro else field
        app = app_values(vocab, field, macro)
        todo = {}
        for value, attrs in sorted(values.items()):
            if value in app:
                continue
            attr = sorted(attrs)[0]
            match, maybe, note = suggest(value, attr, app)
            standard = match or (HINTS[attr].format(v=value) if attr in HINTS else value)
            entry = {"standard": standard, "app_now": match, "vlm_attributes": sorted(attrs), "note": note}
            if maybe:
                entry["maybe"] = maybe
            todo[value] = entry
        if todo:
            decisions[label] = todo
        app_only = sorted(app - set(values))
        report.append(f"| {label} | {len(values)} | {len(values) - len(todo)} | {len(todo)} | {len(app_only)} |")
    report.append("")

    # category tree
    report += ["## Categories and subcategories", ""]
    for name, cfg in generated:
        gender, macro = cfg.get("gender"), cfg.get("macrocategory")
        app_tree = vocab.taxonomy(gender, macro)
        todo = {}
        for cat, subs in (cfg.get("taxonomy") or {}).items():
            if cat not in app_tree:
                todo[f"{cat}"] = {"standard": None, "app_now": None, "vlm_attributes": ["category"],
                                  "note": f"category not in app for {gender}/{macro}"}
                continue
            app_names = {s.name: s.slug for s in app_tree[cat]}
            for sub in subs or []:
                if sub in app_names and app_names[sub] == slugify(sub):
                    continue
                if sub in app_names:
                    note, closest = f"app slug '{app_names[sub]}' should be '{slugify(sub)}'", sub
                else:
                    close = difflib.get_close_matches(sub, list(app_names), n=1, cutoff=0.8)
                    closest = close[0] if close else None
                    note = (f"app has '{closest}' (slug '{app_names[closest]}')" if closest
                            else "not in the app")
                todo[f"{cat}/{sub}"] = {"standard": sub, "app_now": closest,
                                        "vlm_attributes": ["subcategory"], "note": note}
            vlm_names = set(subs or [])
            skipped = sorted(s.name for s in app_tree[cat] if s.name not in vlm_names)
            if skipped:
                report.append(f"- `{cat}` app-only subcategories (VLM never predicts): {', '.join(skipped)}")
        if todo:
            decisions[f"subcategory[{name}]"] = todo
    report.append("")

    if unmapped:
        report += ["## VLM attributes without an app field", "",
                   "Add `app: {field: <import field>}` or `app: {field: null}` to each:", ""]
        report += [f"- {where}: {', '.join(attrs)}" for where, attrs in unmapped.items()]
        report.append("")

    n = sum(len(v) for v in decisions.values())
    report.insert(1, f"\n**{n} values to decide** in `decisions.yaml`.\n")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text("\n".join(report), encoding="utf-8")
    target = args.out / "decisions.yaml"
    if target.exists() and not args.force:
        target = args.out / "decisions.new.yaml"
    header = ("# Set `standard` for each entry (see reconcile_taxonomy.py docstring).\n"
              "# Remove `note` freely; keep `app_now` and `vlm_attributes`.\n"
              f"client: {args.client}\nfingerprint: {vocab.fingerprint}\n")
    with open(target, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump({"decisions": decisions}, f, allow_unicode=True, sort_keys=False, width=120)
    print(f"{n} values to decide -> {target}\nreport -> {args.out / 'report.md'}")
    for line in report:
        if line.startswith(("|", "-")):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
