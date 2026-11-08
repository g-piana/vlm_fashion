from pathlib import Path

p = Path("scripts/generate_category_configs.py")
s = p.read_text(encoding="utf-8")
s = s.replace(
    'default=Path("data/raw/category_hierarchy.yaml"),',
    'default=Path("inflowe-taxonomy/data/base/category_hierarchy.yaml"),',
)
s = s.replace(
    "        --taxonomy data/raw/category_hierarchy.yaml \\",
    "        --taxonomy inflowe-taxonomy/data/base/category_hierarchy.yaml \\",
    1
)
s = s.replace(
    "    python scripts/generate_category_configs.py --taxonomy data/raw/category_hierarchy.yaml",
    "    python scripts/generate_category_configs.py",
    1
)
p.write_text(s, encoding="utf-8")
print("generate_category_configs.py patched")

p = Path("scripts/extract_vocabularies.py")
s = p.read_text(encoding="utf-8")
for old, new in [
    ('default=Path("data/raw/colours.yaml"))', 'default=Path("inflowe-taxonomy/data/base/colours.yaml"))'),
    ('default=Path("data/raw/materials.yaml"))', 'default=Path("inflowe-taxonomy/data/base/materials.yaml"))'),
    ('default=Path("data/raw/material_groups.yaml"))', 'default=Path("inflowe-taxonomy/data/base/material_groups.yaml"))'),
    ('default=Path("data/raw/embellishments.yaml"))', 'default=Path("inflowe-taxonomy/data/base/embellishments.yaml"))'),
    ('default=Path("data/raw/patterns.yaml"))', 'default=Path("inflowe-taxonomy/data/base/patterns.yaml"))'),
    ('default=Path("data/raw/styles.yaml"))', 'default=Path("inflowe-taxonomy/data/base/styles.yaml"))'),
]:
    s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("extract_vocabularies.py patched")

p = Path("configs/categories/generated/_registry.yaml")
s = p.read_text(encoding="utf-8")
s = s.replace(
    "source: data/raw/category_hierarchy.yaml",
    "source: inflowe-taxonomy/data/base/category_hierarchy.yaml",
)
p.write_text(s, encoding="utf-8")
print("_registry.yaml patched")