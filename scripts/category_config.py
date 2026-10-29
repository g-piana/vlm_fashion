"""
Category config loader.

All configuration is in YAML — no hardcoded vocabularies in Python.

Files:
  configs/categories/_base.yaml             — shared attributes (color, material, ...)
  configs/attributes/<macrocat>.yaml        — macrocategory-specific attributes
  configs/categories/generated/<name>.yaml  — taxonomy (auto-generated from hierarchy)

Usage:
    from category_config import build_system_prompt, list_configs
    prompt = build_system_prompt("shoes_women")
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs" / "categories"
GENERATED_DIR = CONFIGS_DIR / "generated"
ATTRIBUTES_DIR = PROJECT_ROOT / "configs" / "attributes"
GUIDELINES_DIR = PROJECT_ROOT / "configs" / "guidelines"

def load_value_definitions() -> dict:
    return _load_yaml(GUIDELINES_DIR / "attribute_values.yaml")


def _format_value_definitions(attrs: dict, defs: dict) -> str:
    lines = []
    for attr, vals in (defs or {}).items():
        spec = attrs.get(attr)
        allowed = spec.get("values") if isinstance(spec, dict) else None
        if not isinstance(allowed, list):
            continue
        items = [f"      {v}: {vals[v]}" for v in allowed if v in vals]
        if items:
            lines.append(f"    {attr}:")
            lines.extend(items)
    return "\n".join(lines)


def load_guidelines(macrocat: str) -> dict:
    return _load_yaml(GUIDELINES_DIR / f"{macrocat}.yaml")

def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_base_config() -> dict:
    return _load_yaml(CONFIGS_DIR / "_base.yaml")


def load_attribute_template(template_name: str) -> dict:
    """Load macrocategory-specific attributes from configs/attributes/<name>.yaml."""
    path = ATTRIBUTES_DIR / f"{template_name}.yaml"
    return _load_yaml(path)


def load_category_config(name: str) -> dict:
    """Load a category config by name, merging base + attributes + taxonomy."""
    # Find the config file
    gen_path = GENERATED_DIR / f"{name}.yaml"
    base_path = CONFIGS_DIR / f"{name}.yaml"

    if gen_path.exists():
        config = _load_yaml(gen_path)
    elif base_path.exists():
        config = _load_yaml(base_path)
    else:
        raise FileNotFoundError(
            f"No config found for '{name}' in {GENERATED_DIR} or {CONFIGS_DIR}"
        )

    # Start with base attributes (color, material, etc.)
    base = load_base_config()
    merged_attrs = copy.deepcopy(base.get("attributes", {}))

    # Add macrocategory-specific attributes from YAML template
    template_name = config.get("attribute_template", "")
    if template_name:
        template_attrs = load_attribute_template(template_name)
        merged_attrs.update(template_attrs)

    # Also merge any inline attributes from the config itself
    merged_attrs.update(config.get("attributes", {}))

    result = {**base, **config}
    result["attributes"] = merged_attrs

    macrocat = config.get("macrocategory")
    if macrocat and not config.get("includes_all_generated"):
        guidelines = load_guidelines(macrocat)
        if guidelines:
            result["guidelines"] = guidelines

    return result


def _format_taxonomy(taxonomy: dict) -> str:
    lines = []
    for cat_slug, subcats in taxonomy.items():
        if subcats:
            subcat_str = ", ".join(str(s) for s in subcats)
            lines.append(f"    {cat_slug}: [{subcat_str}]")
        else:
            lines.append(f"    {cat_slug}: (no subcategories)")
    return "\n".join(lines)

def _format_guidelines(taxonomy: dict, guidelines: dict) -> str:
    if not guidelines:
        return ""
    lines = []
    for subcats in taxonomy.values():
        for s in subcats:
            if s in guidelines:
                lines.append(f"    {s}: {guidelines[s]}")
    return "\n".join(lines)


def _format_attributes(attrs: dict) -> str:
    lines = []
    for name, spec in attrs.items():
        if isinstance(spec, dict):
            vals = spec.get("values")
            if vals and not str(vals).startswith("same_as"):
                vals_str = ", ".join(str(v) for v in vals)
                lines.append(f"    {name}: [{vals_str}]")
            elif spec.get("description"):
                lines.append(f"    {name}: {spec['description']}")
            elif str(vals or "").startswith("same_as:"):
                ref = str(vals).replace("same_as:", "")
                lines.append(f"    {name}: list, same values as {ref}")
    return "\n".join(lines)


def build_system_prompt(
    config_name: str,
    image_type: str = "product",
    gender: str | None = None,
) -> str:
    config = load_category_config(config_name)

    templates = config.get("prompt_templates", {})
    template = templates.get(image_type, templates.get("product", ""))

    gender = gender or config.get("gender", "unknown")
    macrocat = config.get("macrocategory", "unknown")

    taxonomy_str = _format_taxonomy(config.get("taxonomy", {}))
    attrs_str = _format_attributes(config.get("attributes", {}))

    g_str = _format_guidelines(config.get("taxonomy", {}), config.get("guidelines", {}))
    guidelines_block = (
        f"\n\nSubcategory guidelines (decisive visual markers):\n{g_str}" if g_str else ""
    )
    defs_str = _format_value_definitions(config.get("attributes", {}), load_value_definitions())
    definitions_block = f"\n\nValue definitions (what each value means):\n{defs_str}" if defs_str else ""


    prompt = f"""{template}

Gender context: {gender}
Macrocategory: {macrocat}

You MUST classify this item using EXACTLY one category and one subcategory
from the taxonomy below. Output the exact names as shown.

Taxonomy (category → subcategories):
{taxonomy_str}{guidelines_block}

Additional attributes (use controlled vocabulary):
{attrs_str}{definitions_block}"""

    return prompt.strip()


def get_user_message() -> str:
    base = load_base_config()
    return base.get("prompt_templates", {}).get(
        "user_message", "Analyze this fashion product."
    )


def list_configs() -> list[str]:
    configs = set()
    for d in [GENERATED_DIR, CONFIGS_DIR]:
        if d.exists():
            for p in d.glob("*.yaml"):
                if p.stem not in ("_base", "_registry"):
                    configs.add(p.stem)
    return sorted(configs)


if __name__ == "__main__":
    import sys
    configs = list_configs()
    print(f"Available configs ({len(configs)}):")
    for c in configs:
        print(f"  {c}")

    if len(sys.argv) > 1:
        name = sys.argv[1]
        print(f"\n{'='*60}")
        print(f"System prompt for: {name}")
        print(f"{'='*60}")
        print(build_system_prompt(name))
