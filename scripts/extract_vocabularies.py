"""
Extract vocabulary lists from taxonomy YAML files for use in _base.yaml.

Reads colours.yaml, materials.yaml (+ material_groups.yaml) and prints
formatted lists ready to paste into configs/categories/_base.yaml.

Usage:
    python scripts/extract_vocabularies.py \
        --colours data/raw/colours.yaml \
        --materials data/raw/materials.yaml \
        --material-groups data/raw/material_groups.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def extract_colours(path: Path) -> None:
    data = load_yaml(path)
    all_slugs = [v["slug"] for v in data.values()]
    primary_slugs = [v["slug"] for v in data.values() if v.get("primary")]

    print("# ── Colours (all) ──")
    print(f"# {len(all_slugs)} values")
    print(format_yaml_list("primary_color", all_slugs))
    print()
    print("# ── Colours (primary only) ──")
    print(f"# {len(primary_slugs)} values")
    print(format_yaml_list("primary_color_primary_only", primary_slugs))
    print()

def extract_embellishments(path: Path) -> None:
    data = load_yaml(path)
    all_values = data 

    print("# ── Embellishments (all) ──")
    print(f"# {len(all_values)} values")
    print(format_yaml_list("embellishments", all_values))
    print()

def extract_patterns(path: Path) -> None:
    data = load_yaml(path)
    all_values = data 

    print("# ── Patterns (all) ──")
    print(f"# {len(all_values)} values")
    print(format_yaml_list("patterns", all_values))
    print()

def extract_styles(path: Path) -> None:
    data = load_yaml(path)
    all_values = data 

    print("# ── Styles (all) ──")
    print(f"# {len(all_values)} values")
    print(format_yaml_list("styless", all_values))
    print()

def extract_materials(mat_path: Path, groups_path: Path | None) -> None:
    materials = load_yaml(mat_path)

    # Group materials by their group field
    by_group: dict[str, list[str]] = {}
    all_slugs = []
    for entry in materials.values():
        slug = entry["slug"]
        group = entry.get("group", "other")
        by_group.setdefault(group, []).append(slug)
        all_slugs.append(slug)

    # Load group names if available
    group_names = {}
    if groups_path and groups_path.exists():
        groups = load_yaml(groups_path)
        group_names = {v["slug"]: v["name"] for v in groups.values()}

    print("# ── Materials (all, flat list) ──")
    print(f"# {len(all_slugs)} values")
    print(format_yaml_list("material", all_slugs))
    print()

    print("# ── Materials (by group) ──")
    for group_key in sorted(by_group.keys()):
        label = group_names.get(group_key, group_key)
        slugs = by_group[group_key]
        print(f"# {label} ({len(slugs)}): {', '.join(slugs)}")
    print()


def format_yaml_list(field_name: str, values: list[str], width: int = 80) -> str:
    """Format a list of values as a YAML field ready for _base.yaml."""
    lines = [f"  {field_name}:"]
    lines.append("    type: enum")
    lines.append("    values:")

    # Build wrapped lines
    current_line = "      ["
    for i, v in enumerate(values):
        addition = v if i == 0 else f", {v}"
        if len(current_line) + len(addition) > width:
            lines.append(current_line + ",")
            current_line = f"       {v}"
        else:
            current_line += addition

    current_line += "]"
    lines.append(current_line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract vocabulary lists from taxonomy YAMLs"
    )
    parser.add_argument("--colours", type=Path,
                        default=Path("data/raw/colours.yaml"))
    parser.add_argument("--materials", type=Path,
                        default=Path("data/raw/materials.yaml"))
    parser.add_argument("--material-groups", type=Path,
                        default=Path("data/raw/material_groups.yaml"))
    parser.add_argument("--embellishments", type=Path,
                    default=Path("data/raw/embellishments.yaml"))
    parser.add_argument("--patterns", type=Path,
                    default=Path("data/raw/patterns.yaml"))
    parser.add_argument("--styles", type=Path,
                    default=Path("data/raw/styles.yaml"))


    args = parser.parse_args()

    if args.colours.exists():
        extract_colours(args.colours)
    else:
        print(f"# colours file not found: {args.colours}")

    if args.materials.exists():
        extract_materials(args.materials, args.material_groups)
    else:
        print(f"# materials file not found: {args.materials}")

    if args.embellishments.exists():
        extract_embellishments(args.embellishments)
    else:
        print(f"# embellishments file not found: {args.embellishments}")
    if args.patterns.exists():
        extract_patterns(args.patterns)
    else:
        print(f"# patterns file not found: {args.patterns}")
    if args.styles.exists():
        extract_styles(args.styles)
    else:
        print(f"# styles file not found: {args.styles}")



if __name__ == "__main__":
    main()
