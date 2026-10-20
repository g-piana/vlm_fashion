"""
Category config loader.

Loads configs from two locations:
  1. configs/categories/          — hand-written base + attribute templates
  2. configs/categories/generated/ — auto-generated from taxonomy

The generated configs reference an attribute_template (e.g. "shoes")
which maps to hand-written attribute definitions in the base configs.

Usage:
    from category_config import build_system_prompt, list_configs
    prompt = build_system_prompt("shoes_women")
    prompt = build_system_prompt("shoes_women", image_type="look")
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "categories"
GENERATED_DIR = CONFIGS_DIR / "generated"

# ── Attribute templates ──────────────────────────────────────────────────
# Hand-written attribute sets per macrocategory type. These define the
# additional structured fields beyond the base (color, material, etc.)

ATTRIBUTE_TEMPLATES = {
    "shoes": {
        "toe_shape": {"type": "enum", "values": [
            "round", "pointed", "square", "almond", "peep-toe", "open-toe", "cap-toe"]},
        "heel_type": {"type": "enum", "values": [
            "flat", "kitten", "block", "stiletto", "wedge", "platform",
            "cone", "spool", "stacked", "sculpted", "none"]},
        "heel_height": {"type": "enum", "values": [
            "flat", "low", "mid", "high", "ultra-high"]},
        "heel_finish": {"type": "enum", "values": [
            "wrapped", "metallic", "wooden", "lucite", "lacquered", "natural", "null"]},
        "sole_type": {"type": "enum", "values": [
            "leather", "rubber", "crepe", "rope", "platform", "lug", "commando", "null"]},
        "closure": {"type": "enum", "values": [
            "slip-on", "lace-up", "buckle", "zip", "side-zip", "back-zip",
            "velcro", "elastic", "tie", "button", "monk-strap", "none"]},
        "shaft_height": {"type": "enum", "values": [
            "ankle", "mid-calf", "knee", "over-knee", "null"]},
    },
    "bags": {
        "handle_type": {"type": "enum", "values": [
            "top-handle", "shoulder-strap", "chain-strap", "wrist-strap",
            "crossbody-strap", "detachable-strap", "drawstring", "none"]},
        "bag_closure": {"type": "enum", "values": [
            "zip", "flap", "drawstring", "clasp", "magnetic-snap", "turn-lock",
            "frame", "open-top", "buckle", "toggle", "none"]},
        "hardware_color": {"type": "enum", "values": [
            "gold", "silver", "gunmetal", "rose-gold", "antique-brass", "black", "none"]},
        "bag_size": {"type": "enum", "values": [
            "micro", "mini", "small", "medium", "large", "oversized"]},
    },
    "ready_to_wear": {
        "fit": {"type": "enum", "values": [
            "slim", "regular", "relaxed", "oversized", "tailored", "cropped",
            "flared", "straight", "skinny", "wide-leg", "A-line", "bodycon"]},
        "length": {"type": "enum", "values": [
            "cropped", "hip", "mid-thigh", "knee", "midi", "maxi", "floor-length", "null"]},
        "neckline": {"type": "enum", "values": [
            "crew", "v-neck", "scoop", "boat", "turtleneck", "mock-neck", "cowl",
            "off-shoulder", "halter", "strapless", "collared", "mandarin",
            "sweetheart", "square", "keyhole", "henley", "null"]},
        "collar": {"type": "enum", "values": [
            "point", "spread", "button-down", "band", "peter-pan", "shawl",
            "notched-lapel", "peak-lapel", "mandarin", "stand", "none", "null"]},
        "sleeve_length": {"type": "enum", "values": [
            "sleeveless", "cap", "short", "elbow", "three-quarter", "long", "null"]},
        "closure": {"type": "enum", "values": [
            "button", "zip", "snap", "hook", "tie", "wrap", "pullover",
            "toggle", "double-breasted", "none"]},
    },
    "luggage": {
        "luggage_type": {"type": "enum", "values": [
            "hardside", "softside", "wheeled", "duffle", "garment-bag", "null"]},
    },
    "accessories": {},
    "eyewear": {
        "frame_shape": {"type": "enum", "values": [
            "aviator", "wayfarer", "round", "cat-eye", "rectangular",
            "oversized", "shield", "square", "oval", "geometric", "null"]},
    },
    "jewelry": {
        "jewelry_material": {"type": "enum", "values": [
            "gold", "silver", "rose-gold", "platinum", "stainless-steel",
            "brass", "copper", "leather", "fabric", "bead", "pearl", "null"]},
    },
    "watches": {
        "dial_shape": {"type": "enum", "values": [
            "round", "square", "rectangular", "tonneau", "oval", "null"]},
        "strap_type": {"type": "enum", "values": [
            "leather", "metal-bracelet", "rubber", "nato", "mesh", "fabric", "null"]},
    },
    "generic": {},
}


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_base_config() -> dict:
    return _load_yaml(CONFIGS_DIR / "_base.yaml")


def load_category_config(name: str) -> dict:
    """Load a generated or hand-written category config by name.

    Tries generated/ first, then the base configs/ dir.
    """
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

    # Merge with base
    base = load_base_config()
    merged_attrs = copy.deepcopy(base.get("attributes", {}))

    # Add macrocategory-specific attributes from template
    template_key = config.get("attribute_template", "generic")
    template_attrs = ATTRIBUTE_TEMPLATES.get(template_key, {})
    merged_attrs.update(template_attrs)

    # Also merge any hand-written attributes in the config itself
    merged_attrs.update(config.get("attributes", {}))

    result = {**base, **config}
    result["attributes"] = merged_attrs

    return result


def _format_taxonomy(taxonomy: dict) -> str:
    """Format taxonomy tree for the system prompt."""
    lines = []
    for cat_slug, subcats in taxonomy.items():
        if subcats:
            subcat_str = ", ".join(str(s) for s in subcats)
            lines.append(f"    {cat_slug}: [{subcat_str}]")
        else:
            lines.append(f"    {cat_slug}: (no subcategories)")
    return "\n".join(lines)


def _format_attributes(attrs: dict) -> str:
    """Format attribute vocabulary for the system prompt."""
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
    """Build the full system prompt for a config and image type.

    Args:
        config_name: Config name (e.g. "shoes_women", "bags_men")
        image_type: "product" or "look"
        gender: Override gender (otherwise taken from config)
    """
    config = load_category_config(config_name)

    templates = config.get("prompt_templates", {})
    template = templates.get(image_type, templates.get("product", ""))

    gender = gender or config.get("gender", "unknown")
    macrocat = config.get("macrocategory", "unknown")

    taxonomy_str = _format_taxonomy(config.get("taxonomy", {}))
    attrs_str = _format_attributes(config.get("attributes", {}))

    prompt = f"""{template}

Gender context: {gender}
Macrocategory: {macrocat}

You MUST classify this item using EXACTLY one category and one subcategory
from the taxonomy below. Output the exact names as shown.

Taxonomy (category → subcategories):
{taxonomy_str}

Additional attributes (use controlled vocabulary):
{attrs_str}"""

    return prompt.strip()


def get_user_message() -> str:
    base = load_base_config()
    return base.get("prompt_templates", {}).get(
        "user_message", "Analyze this fashion product."
    )


def list_configs() -> list[str]:
    """List all available configs (generated + hand-written, excluding _base)."""
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
