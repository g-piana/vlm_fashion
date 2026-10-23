"""
Apply the taxonomy-enforcement edits to existing scripts.

Run from the project root, after copying taxonomy_utils.py into scripts/:

    python scripts/apply_taxonomy_patches.py            # apply
    python scripts/apply_taxonomy_patches.py --dry-run  # only check anchors

Safety:
  * every edit is anchored on one exact line that must occur exactly once;
  * if ANY anchor is missing/ambiguous in ANY file, nothing is written;
  * originals are saved as <file>.bak;
  * files already patched are detected and skipped.

Files touched:
  scripts/review_labels.py              linked category/subcategory select, list-typed
                                        attributes as chips, export gate, strip unknown keys
  scripts/prepare_dataset.py            validation gate (--strict / --no-validate)
  scripts/infer.py                      derive category from subcategory + validation record
  scripts/generate_category_configs.py  warn when subcategory names are not unique

evaluate.py is not patched (two manual lines, see the README note).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

# NOTE: the review_labels.py JavaScript lives inside a Python f-string, so every
# literal brace is doubled ({{ }}) and "\n" is written "\\n" in the replacements.

NEW_TAXONOMY_SELECT_JS = r"""function buildTaxonomySelect(cat, sub) {{
  // One select for category + subcategory: picking a subcategory sets its category,
  // so an invalid pair cannot be created. Categories without subcategories are
  // listed as plain options.
  const tax = VOCABS._taxonomy || {{}};
  const cur = (cat && tax[cat] && (sub ? tax[cat].includes(sub) : tax[cat].length === 0))
    ? cat + '||' + (sub || '') : '';
  let opts = '<option value="">(none)</option>';
  for (const [c, subs] of Object.entries(tax)) {{
    if (subs.length === 0) {{
      const v = c + '||';
      opts += '<option value="' + escHtml(v) + '"' + (v === cur ? ' selected' : '') + '>' + escHtml(c) + '</option>';
    }} else {{
      opts += '<optgroup label="' + escHtml(c) + '">';
      for (const s of subs) {{
        const v = c + '||' + s;
        opts += '<option value="' + escHtml(v) + '"' + (v === cur ? ' selected' : '') + '>' + escHtml(s) + '</option>';
      }}
      opts += '</optgroup>';
    }}
  }}
  const warn = ((cat || sub) && !cur)
    ? '<div style="color:#e67e22;font-size:12px;margin-top:4px">Draft had ' + escHtml(String(cat)) + ' / ' +
      escHtml(String(sub)) + ' (not a valid pair): pick the subcategory.</div>'
    : '';
  return '<div class="field-group"><label>category / subcategory</label>' +
         '<select id="taxonomy_select">' + opts + '</select>' + warn + '</div>';
}}

function buildSelect(name, value, options) {{"""

EXPORT_GATE_JS = r"""const allowedKeys = new Set(VOCABS._allowed_keys || []);
  const exportable = DRAFTS.filter(d => !d._skip && d.description);
  exportable.forEach(d => {{
    if (d.tags && allowedKeys.size) {{
      Object.keys(d.tags).forEach(k => {{ if (!allowedKeys.has(k)) delete d.tags[k]; }});
    }}
  }});
  const badItems = exportable.filter(d => {{
    const t = d.tags || {{}};
    const subs = (VOCABS._taxonomy || {{}})[t.category];
    if (!subs) return true;
    if (subs.length === 0) return !!t.subcategory;
    return !subs.includes(t.subcategory);
  }});
  if (badItems.length) {{
    alert('Cannot export: missing or invalid category/subcategory in ' + badItems.length +
          ' item(s):\\n' + badItems.slice(0, 15).map(d => d.image.split('/').pop()).join('\\n'));
    return;
  }}
  const labels = DRAFTS"""

EDITS: dict[str, dict] = {
    "review_labels.py": {
        "marker": "buildTaxonomySelect",
        "edits": [
            (
                'vocabs["_subcategories"] = sorted(set(all_subcategories))',
                'vocabs["_subcategories"] = sorted(set(all_subcategories))\n'
                "    # list-typed attributes -> multi-select chips; allowed keys -> strip typos on export\n"
                '    vocabs["_multi"] = [\n'
                '        n for n, s in config.get("attributes", {}).items()\n'
                '        if isinstance(s, dict) and s.get("type") == "list"\n'
                "    ]\n"
                '    vocabs["_allowed_keys"] = ["category", "subcategory"] + list(config.get("attributes", {}).keys())',
            ),
            (
                "html += buildSelect('category', tags.category, VOCABS._categories);",
                "html += buildTaxonomySelect(tags.category, tags.subcategory);",
            ),
            (
                "const subcats = tags.category ? (VOCABS._taxonomy[tags.category] || []) : VOCABS._subcategories;",
                "// category + subcategory are chosen together (see buildTaxonomySelect)",
            ),
            (
                "html += buildSelect('subcategory', tags.subcategory, subcats);",
                "// (single linked select above)",
            ),
            (
                "if (Array.isArray(val)) {{",
                "if (Array.isArray(val) || (VOCABS._multi || []).includes(name)) {{",
            ),
            (
                "html += buildMultiSelect(name, val, spec);",
                "html += buildMultiSelect(name, Array.isArray(val) ? val : (val ? [val] : []), spec);",
            ),
            (
                "function buildSelect(name, value, options) {{",
                NEW_TAXONOMY_SELECT_JS,
            ),
            (
                "item.tags[name] = sel.value || null;",
                "if (sel.id === 'taxonomy_select') {{\n"
                "      const parts = (sel.value || '').split('||');\n"
                "      item.tags.category = parts[0] || null;\n"
                "      item.tags.subcategory = parts[1] || null;\n"
                "      return;\n"
                "    }}\n"
                "    item.tags[name] = sel.value || null;",
            ),
            ("const labels = DRAFTS", EXPORT_GATE_JS),
        ],
    },
    "prepare_dataset.py": {
        "marker": "taxonomy_utils",
        "edits": [
            (
                "from category_config import build_system_prompt, get_user_message",
                "from category_config import build_system_prompt, get_user_message\n"
                "from taxonomy_utils import validate_samples",
            ),
            (
                "args = parser.parse_args()",
                'parser.add_argument("--strict", action="store_true",\n'
                '                        help="treat vocabulary warnings as fatal too")\n'
                '    parser.add_argument("--no-validate", action="store_true",\n'
                '                        help="skip taxonomy/vocabulary validation")\n'
                "    args = parser.parse_args()",
            ),
            (
                'log.info("Total: %d valid samples from %d batches", len(all_samples), len(batches_cfg))',
                'log.info("Total: %d valid samples from %d batches", len(all_samples), len(batches_cfg))\n'
                "\n"
                "    if not args.no_validate:\n"
                "        if not validate_samples(all_samples, strict=args.strict):\n"
                "            return",
            ),
        ],
    },
    "infer.py": {
        "marker": "taxonomy_utils",
        "edits": [
            (
                "from category_config import build_system_prompt, get_user_message, list_configs",
                "from category_config import build_system_prompt, get_user_message, list_configs\n"
                "from taxonomy_utils import apply_taxonomy_rules",
            ),
            (
                "return parse_model_output(text)",
                "result = parse_model_output(text)\n"
                "        # derive category from subcategory + record taxonomy/vocabulary issues\n"
                "        return apply_taxonomy_rules(result, category)",
            ),
        ],
    },
    "generate_category_configs.py": {
        "marker": "find_duplicate_subcategories",
        "edits": [
            (
                "import yaml",
                "import yaml\n\nfrom taxonomy_utils import find_duplicate_subcategories",
            ),
            (
                'n_subcats = sum(len(v) for v in config["taxonomy"].values())',
                'n_subcats = sum(len(v) for v in config["taxonomy"].values())\n'
                '            dups = find_duplicate_subcategories(config["taxonomy"])\n'
                "            if dups:\n"
                '                log.warning("%s: subcategory names shared by several categories %s '
                '(category cannot be derived from them)", config_name, dups)',
            ),
        ],
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--scripts-dir", type=Path, default=SCRIPTS)
    args = ap.parse_args()

    if not (args.scripts_dir / "taxonomy_utils.py").exists():
        print("taxonomy_utils.py must be in the scripts directory first.")
        return 1

    problems: list[str] = []
    plans: dict[Path, str] = {}

    for fname, spec in EDITS.items():
        path = args.scripts_dir / fname
        if not path.exists():
            problems.append(f"{fname}: file not found")
            continue
        text = path.read_text(encoding="utf-8")
        if spec["marker"] in text:
            print(f"{fname}: already patched, skipping")
            continue
        new = text
        for anchor, replacement in spec["edits"]:
            n = new.count(anchor)
            if n != 1:
                problems.append(f"{fname}: anchor found {n}x (expected 1): {anchor[:70]!r}")
                continue
            new = new.replace(anchor, replacement, 1)
        plans[path] = new

    if problems:
        print("NOTHING WRITTEN. Problems:")
        for p in problems:
            print("  -", p)
        return 1

    for path, new in plans.items():
        if args.dry_run:
            print(f"{path.name}: anchors OK (dry run)")
            continue
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        path.write_text(new, encoding="utf-8")
        print(f"{path.name}: patched (backup: {path.name}.bak)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
