"""
Make the review tool follow the DECLARED attribute type (list vs enum) instead of
the shape of whatever the draft happened to contain.

Bug this fixes
  A draft with a list in a single-valued field (e.g. material: ["cotton", "leather"])
  was drawn as chips, so two materials could be selected as primary, and even after
  you deselected one the export kept a one-element list (material: ["cotton"]).

After the patch
  * enum-declared field  -> always a single-choice select (first draft value is
                            preselected; if the draft had several, an orange note says so)
  * list-declared field  -> always chips
  * export normalises:   enum + one-element list -> the value; list + scalar/null -> list

Run from the project root (after the earlier review_labels patches):
    python scripts/apply_review_shape_patch.py
A .bak4 copy is kept; nothing is written if an anchor is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "review_labels.py"

RENDER_NOTE = r"""html += buildSelect(name, Array.isArray(val) ? (val[0] ?? null) : val, spec) +
        ((Array.isArray(val) && val.length > 1)
          ? '<div style="color:#e67e22;font-size:12px;grid-column:1/-1">' + escHtml(name) +
            ': the draft had several values (' + escHtml(val.join(', ')) +
            '); only the first is kept. Check the secondary fields.</div>'
          : '');"""

EXPORT_SHAPE = r"""Object.keys(d.tags).forEach(k => {{ if (!allowedKeys.has(k)) delete d.tags[k]; }});
      const multiKeys = new Set(VOCABS._multi || []);
      Object.keys(d.tags).forEach(k => {{
        if (k === 'category' || k === 'subcategory') return;
        const v = d.tags[k];
        if (multiKeys.has(k)) {{
          if (v === null || v === undefined) d.tags[k] = [];
          else if (!Array.isArray(v)) d.tags[k] = [v];
        }} else if (Array.isArray(v) && v.length <= 1) {{
          d.tags[k] = v.length ? v[0] : null;
        }}
      }});"""

EDITS = [
    ("if (Array.isArray(val) || (VOCABS._multi || []).includes(name)) {{",
     "if ((VOCABS._multi || []).includes(name)) {{", "chips only for list-declared fields"),
    ("html += buildSelect(name, val, spec);", RENDER_NOTE, "single select for enum-declared fields"),
    ("Object.keys(d.tags).forEach(k => {{ if (!allowedKeys.has(k)) delete d.tags[k]; }});",
     EXPORT_SHAPE, "export shape normalisation"),
]


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "multiKeys" in text:
        print("already patched")
        return 0
    problems, new = [], text
    for old, repl, label in EDITS:
        n = new.count(old)
        if n != 1:
            problems.append(f"{label}: anchor found {n}x (expected 1): {old[:60]!r}")
        else:
            new = new.replace(old, repl, 1)
    if problems:
        print("NOTHING WRITTEN:")
        for p in problems:
            print("  -", p)
        return 1
    TARGET.with_suffix(".py.bak4").write_text(text, encoding="utf-8")
    TARGET.write_text(new, encoding="utf-8")
    print("review_labels.py patched (backup: review_labels.py.bak4)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
