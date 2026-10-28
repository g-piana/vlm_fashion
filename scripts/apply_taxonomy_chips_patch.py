"""
Second patch for review_labels.py: show category / subcategory as text titles
with the subcategories as single-choice chips (everything visible at once)
instead of one dropdown.

Run AFTER apply_taxonomy_patches.py, from the project root:

    python scripts/apply_taxonomy_chips_patch.py

Also fixes the Python 3.13 SyntaxWarning (invalid escape '\\.') in the template.
A .bak2 copy of review_labels.py is kept.
"""

from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "review_labels.py"

# JavaScript inside a Python f-string: literal braces are doubled.
NEW_JS = r"""function pickTaxonomy(el) {{
  const box = document.getElementById('taxonomy_chips');
  box.querySelectorAll('.tax-chip').forEach(c => c.classList.remove('selected'));
  box.querySelectorAll('.tax-cat').forEach(t => t.classList.toggle('active', t.dataset.cat === el.dataset.cat));
  el.classList.add('selected');
  box.dataset.value = el.dataset.value;
}}

function buildTaxonomySelect(cat, sub) {{
  // Categories as titles, subcategories as single-choice chips. Picking a chip
  // sets category + subcategory together, so an invalid pair cannot be created.
  if (!document.getElementById('taxStyle')) {{
    const st = document.createElement('style');
    st.id = 'taxStyle';
    st.textContent =
      '.tax-cat {{ font-size: 12px; font-weight: 600; color: #7f8ea3; text-transform: uppercase; letter-spacing: 0.5px; margin: 12px 0 4px; }}' +
      '.tax-cat.active {{ color: #a8d8ea; }}' +
      '.tax-chips {{ display: flex; flex-wrap: wrap; gap: 4px; }}' +
      '.tax-chip {{ padding: 3px 10px; border-radius: 12px; font-size: 12px; cursor: pointer; border: 1px solid #444; background: #16213e; color: #aaa; user-select: none; transition: all 0.15s; }}' +
      '.tax-chip.selected {{ background: #4a9eff; color: #fff; border-color: #4a9eff; }}';
    document.head.appendChild(st);
  }}
  const tax = VOCABS._taxonomy || {{}};
  let cur = '';
  let msg = '';
  if (sub) {{
    const owners = Object.keys(tax).filter(c => tax[c].includes(sub));
    if (cat && tax[cat] && tax[cat].includes(sub)) {{
      cur = cat + '||' + sub;
    }} else if (owners.length === 1) {{
      cur = owners[0] + '||' + sub;
      if (cat) msg = 'Draft category ' + escHtml(String(cat)) + ' was corrected from the subcategory.';
    }}
  }} else if (cat && tax[cat] && tax[cat].length === 0) {{
    cur = cat + '||';
  }}
  if (!cur && (cat || sub)) {{
    msg = 'Draft had ' + escHtml(String(cat)) + ' / ' + escHtml(String(sub)) + ' (not a valid pair): pick the subcategory.';
  }}
  let html = '<div class="field-group" style="grid-column: 1 / -1"><label>category / subcategory</label>' +
             '<div id="taxonomy_chips" data-value="' + escHtml(cur) + '">';
  for (const [c, subs] of Object.entries(tax)) {{
    const active = cur.split('||')[0] === c ? ' active' : '';
    const title = c.replace(/-(women|men)$/, '').replace(/-/g, ' ');
    html += '<div class="tax-cat' + active + '" data-cat="' + escHtml(c) + '">' + escHtml(title) + '</div><div class="tax-chips">';
    for (const s of (subs.length ? subs : [''])) {{
      const v = c + '||' + s;
      html += '<span class="tax-chip' + (v === cur ? ' selected' : '') + '" data-cat="' + escHtml(c) +
              '" data-value="' + escHtml(v) + '" onclick="pickTaxonomy(this)">' + escHtml(s || '(no subcategory)') + '</span>';
    }}
    html += '</div>';
  }}
  html += '</div>' + (msg ? '<div style="color:#e67e22;font-size:12px;margin-top:6px">' + msg + '</div>' : '') + '</div>';
  return html;
}}

"""

SAVE_HOOK = r"""// Category / subcategory (single-choice chips)
  const taxBox = document.getElementById('taxonomy_chips');
  if (taxBox) {{
    const parts = (taxBox.dataset.value || '').split('||');
    item.tags.category = parts[0] || null;
    item.tags.subcategory = parts[1] || null;
  }}

  // Multi-selects (chips)"""


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "pickTaxonomy" in text:
        print("already patched")
        return 0

    start_marker = "function buildTaxonomySelect(cat, sub) {{"
    end_marker = "function buildSelect(name, value, options) {{"
    hook_anchor = "// Multi-selects (chips)"
    problems = []
    if text.count(start_marker) != 1:
        problems.append("buildTaxonomySelect not found: run apply_taxonomy_patches.py first")
    if text.count(end_marker) != 1:
        problems.append(f"anchor found {text.count(end_marker)}x: {end_marker}")
    if text.count(hook_anchor) != 1:
        problems.append(f"anchor found {text.count(hook_anchor)}x: {hook_anchor}")
    if problems:
        print("NOTHING WRITTEN:")
        for p in problems:
            print("  -", p)
        return 1

    a, b = text.index(start_marker), text.index(end_marker)
    if a > b:
        print("NOTHING WRITTEN: unexpected order of functions")
        return 1
    new = text[:a] + NEW_JS + text[b:]
    new = new.replace(hook_anchor, SAVE_HOOK, 1)

    # silence the Python 3.13 SyntaxWarning (regex escape inside the f-string)
    bad, good = r".replace(/\.[^.]+$/, '')", r".replace(/\\.[^.]+$/, '')"
    if new.count(bad) == 1:
        new = new.replace(bad, good, 1)
        print("fixed the '\\.' SyntaxWarning")
    else:
        print("note: '\\.' regex line not found once; SyntaxWarning left as is")

    TARGET.with_suffix(".py.bak2").write_text(text, encoding="utf-8")
    TARGET.write_text(new, encoding="utf-8")
    print("review_labels.py patched (backup: review_labels.py.bak2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
