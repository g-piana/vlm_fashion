"""
Category config loader.

Reads _base.yaml + a category YAML, merges attributes, and builds
the full system prompt with controlled vocabulary for that category.

Usage:
    from category_config import build_system_prompt, load_category_config
    prompt = build_system_prompt("shoes")           # product image
    prompt = build_system_prompt("shoes", is_look=True)  # look image
    prompt = build_system_prompt("look")            # look with all vocabs
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "categories"


def _load_yaml(name: str) -> dict:
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Category config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_category_config(category: str) -> dict:
    """Load a category config, merging with _base.yaml."""
    base = _load_yaml("_base")
    if category == "_base":
        return base

    cat = _load_yaml(category)

    # Merge attributes: category-specific override/extend base
    merged_attrs = copy.deepcopy(base.get("attributes", {}))
    merged_attrs.update(cat.get("attributes", {}))

    result = {**base, **cat}
    result["attributes"] = merged_attrs

    # For looks that include multiple categories, merge those too
    if "includes" in cat:
        for inc_name in cat["includes"]:
            inc = _load_yaml(inc_name)
            for k, v in inc.get("attributes", {}).items():
                if k not in result["attributes"]:
                    result["attributes"][k] = v
            # Merge category_values into a combined list
            if "category_values" in inc:
                existing = result.get("category_values", [])
                result["category_values"] = existing + inc["category_values"]

    return result


def _format_vocabulary(config: dict) -> str:
    """Format the controlled vocabulary block for the system prompt."""
    lines = []

    # Category values
    if "category_values" in config:
        vals = ", ".join(config["category_values"])
        lines.append(f"  category: [{vals}]")
        lines.append(f"  subcategory: free text, be specific")

    # All attributes
    for name, spec in config["attributes"].items():
        if isinstance(spec, dict):
            if spec.get("values") and not str(spec["values"]).startswith("same_as"):
                vals = ", ".join(str(v) for v in spec["values"])
                lines.append(f"  {name}: [{vals}]")
            elif spec.get("description"):
                lines.append(f"  {name}: {spec['description']}")
            elif str(spec.get("values", "")).startswith("same_as:"):
                ref = str(spec["values"]).replace("same_as:", "")
                lines.append(f"  {name}: list, same values as {ref}")
        else:
            lines.append(f"  {name}: {spec}")

    # Look-level attributes
    if "look_attributes" in config:
        lines.append("")
        lines.append("  # Look-level attributes (apply to overall image):")
        for name, spec in config["look_attributes"].items():
            if isinstance(spec, dict):
                if spec.get("values") and not str(spec["values"]).startswith("same_as"):
                    vals = ", ".join(str(v) for v in spec["values"])
                    lines.append(f"  {name}: [{vals}]")
                elif spec.get("description"):
                    lines.append(f"  {name}: {spec['description']}")

    return "\n".join(lines)


def build_system_prompt(
    category: str,
    image_type: str = "product",
) -> str:
    """Build the full system prompt for a category and image type.

    Args:
        category: Category name (e.g. "shoes", "bags", "look")
        image_type: "product" (single item) or "look" (multi-item)

    Returns:
        Complete system prompt string with vocabulary.
    """
    config = load_category_config(category)

    # Pick the right prompt template
    templates = config.get("prompt_templates", {})
    template = templates.get(image_type, templates.get("product", ""))

    vocabulary = _format_vocabulary(config)

    prompt = f"""{template}
Controlled vocabulary:
{vocabulary}"""

    return prompt.strip()


def get_user_message(category: str = "_base") -> str:
    """Get the user message template."""
    config = load_category_config(category)
    return config.get("prompt_templates", {}).get(
        "user_message", "Analyze this fashion product."
    )


def list_categories() -> list[str]:
    """List all available category configs (excluding _base)."""
    return sorted(
        p.stem for p in CONFIGS_DIR.glob("*.yaml")
        if p.stem != "_base"
    )


if __name__ == "__main__":
    # Quick test: print all available prompts
    for cat in list_categories():
        print(f"\n{'='*60}")
        print(f"Category: {cat}")
        print(f"{'='*60}")
        print(build_system_prompt(cat))
        print()
