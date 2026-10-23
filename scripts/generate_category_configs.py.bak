"""
Generate per-macrocategory VLM config files from the master taxonomy.

Reads category_hierarchy.yaml and produces one config per
(gender, macrocategory) pair under configs/categories/generated/.

Each generated config embeds the exact category→subcategory taxonomy
tree so the VLM system prompt constrains output to valid values.

Usage:
    python scripts/generate_category_configs.py \
        --taxonomy data/raw/category_hierarchy.yaml \
        --output-dir configs/categories/generated

    # Regenerate after taxonomy changes:
    python scripts/generate_category_configs.py --taxonomy data/raw/category_hierarchy.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Macrocategories that need specific attribute sets
# (mapped to the hand-written attribute template in _base_attributes/)
ATTRIBUTE_TEMPLATES = {
    "shoes": "shoes",
    "bags": "bags",
    "ready-to-wear": "ready_to_wear",
    "luggage": "luggage",
    "accessories": "accessories",
    "eyewear": "eyewear",
    "jewelry": "jewelry",
    "watches": "watches",
}


def extract_taxonomy_tree(macrocat_data: dict) -> dict[str, list[str]]:
    """Extract {category_slug: [subcategory_names]} from a macrocategory."""
    tree = {}
    categories = macrocat_data.get("categories", {})
    if not categories:
        return tree

    for cat_key, cat_data in categories.items():
        cat_slug = cat_data.get("slug", cat_key)
        subcats = cat_data.get("subcategories", {})
        if isinstance(subcats, dict) and subcats:
            subcat_names = [
                sc_data.get("name", sc_key)
                for sc_key, sc_data in subcats.items()
            ]
        else:
            subcat_names = []
        tree[cat_slug] = subcat_names

    return tree


def build_config(
    gender: str,
    macrocat_slug: str,
    macrocat_data: dict,
) -> dict:
    """Build a config dict for one (gender, macrocategory) pair."""
    taxonomy = extract_taxonomy_tree(macrocat_data)
    attr_template = ATTRIBUTE_TEMPLATES.get(macrocat_slug, "generic")

    config = {
        "# Auto-generated from category_hierarchy.yaml — do not edit manually.": None,
        "# Re-generate with: python scripts/generate_category_configs.py": None,
        "inherits": "_base",
        "gender": gender,
        "macrocategory": macrocat_slug,
        "attribute_template": attr_template,
        "taxonomy": {},
    }

    # Build taxonomy section
    for cat_slug, subcats in taxonomy.items():
        if subcats:
            config["taxonomy"][cat_slug] = subcats
        else:
            # Categories with no subcategories
            config["taxonomy"][cat_slug] = []

    return config


def write_config(config: dict, path: Path) -> None:
    """Write config to YAML, handling the comment pseudo-keys."""
    lines = []

    # Write comments
    for key in list(config.keys()):
        if key.startswith("#"):
            lines.append(key)
            del config[key]

    lines.append("")

    # Write the rest as standard YAML
    yaml_str = yaml.dump(
        config, default_flow_style=False, allow_unicode=True, sort_keys=False,
        width=120,
    )
    lines.append(yaml_str)

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Generate VLM category configs from master taxonomy"
    )
    parser.add_argument(
        "--taxonomy", type=Path,
        default=Path("data/raw/category_hierarchy.yaml"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("configs/categories/generated"),
    )
    args = parser.parse_args()

    if not args.taxonomy.exists():
        log.error("Taxonomy file not found: %s", args.taxonomy)
        return

    with open(args.taxonomy, encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for gender, gender_data in taxonomy.items():
        macrocategories = gender_data.get("macrocategories", {})
        for macro_slug, macro_data in macrocategories.items():
            # Skip empty macrocategories (no categories defined)
            cats = macro_data.get("categories", {})
            if not cats:
                log.info("Skipping %s/%s (no categories)", gender, macro_slug)
                continue

            config_name = f"{macro_slug}_{gender}"
            config = build_config(gender, macro_slug, macro_data)
            out_path = args.output_dir / f"{config_name}.yaml"
            write_config(config, out_path)
            generated.append(config_name)

            # Count for logging
            n_cats = len(config["taxonomy"])
            n_subcats = sum(len(v) for v in config["taxonomy"].values())
            log.info(
                "Generated %-30s  %2d categories, %3d subcategories",
                config_name, n_cats, n_subcats,
            )

    # Generate a look config that includes all macrocategories
    look_config = {
        "# Auto-generated — includes all taxonomy for multi-item images": None,
        "inherits": "_base",
        "image_type": "look",
        "includes_all_generated": True,
        "taxonomy": {},
    }
    for gender, gender_data in taxonomy.items():
        for macro_slug, macro_data in gender_data.get("macrocategories", {}).items():
            tree = extract_taxonomy_tree(macro_data)
            for cat_slug, subcats in tree.items():
                # Prefix with gender to avoid collisions
                look_config["taxonomy"][cat_slug] = subcats

    look_path = args.output_dir / "look.yaml"
    write_config(look_config, look_path)
    generated.append("look")
    log.info("Generated look config with %d total categories",
             len(look_config["taxonomy"]))

    # Write a registry file listing all generated configs
    registry = {
        "generated_configs": sorted(generated),
        "source": str(args.taxonomy),
    }
    registry_path = args.output_dir / "_registry.yaml"
    with open(registry_path, "w") as f:
        yaml.dump(registry, f, default_flow_style=False)

    log.info("Done. %d configs written to %s", len(generated), args.output_dir)


if __name__ == "__main__":
    main()
