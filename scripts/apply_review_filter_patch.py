"""
Add a filter to the review tool: show only some items while reviewing.

    python scripts/apply_review_filter_patch.py       (from the project root)

In the page header there is now a filter box and an "unreviewed" checkbox:
  * type words separated by commas and press Enter, e.g.   pump, d'orsay, ballerina
    -> only items whose subcategory, category or filename contains one of the words
  * a word starting with "-" excludes:                     pump, -wedge
  * "unreviewed" hides the items you have already visited
  * Prev/Next (and the arrow keys) move only through the visible items; the counter shows
    "3 / 12 (of 194)"
  * clear the box to see everything again
You can also open the page with the filter already set:  http://localhost:8501/?filter=pump,d'orsay

Export, Save Progress and the saved state still cover ALL items: items you do not visit
keep their current labels, so reviewing a subset never overwrites the rest.

A .bak6 copy of review_labels.py is kept; nothing is written if an anchor is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "review_labels.py"

# JavaScript lives inside a Python f-string: literal braces are doubled.
FILTER_JS = r"""// ── Filter: only some items are visited; Export / Save still cover ALL items ──
function applyFilter() {{
  saveCurrent();
  const raw = (document.getElementById('filterInput').value || '').toLowerCase();
  const newOnly = document.getElementById('filterNew').checked;
  const terms = raw.split(',').map(s => s.trim()).filter(Boolean);
  const inc = terms.filter(t => !t.startsWith('-'));
  const exc = terms.filter(t => t.startsWith('-')).map(t => t.slice(1).trim()).filter(Boolean);
  const active = inc.length > 0 || exc.length > 0 || newOnly;
  visible = [];
  if (active) {{
    DRAFTS.forEach((d, i) => {{
      const t = d.tags || {{}};
      const hay = [t.subcategory, t.category, String(d.image || '').split('/').pop()].join(' ').toLowerCase();
      if (newOnly && reviewed.has(i)) return;
      if (exc.some(x => hay.includes(x))) return;
      if (inc.length && !inc.some(x => hay.includes(x))) return;
      visible.push(i);
    }});
    if (visible.length && !visible.includes(currentIdx)) {{
      currentIdx = visible.find(i => i >= currentIdx);
      if (currentIdx === undefined) currentIdx = visible[0];
    }}
  }}
  render();
  if (active && !visible.length) {{
    document.getElementById('counter').textContent = '0 matches';
  }}
}}

function navigate(delta) {{"""

NAV_NEW = r"""const vis = visible.length ? visible : DRAFTS.map((_, i) => i);
  let pos = vis.indexOf(currentIdx);
  if (pos < 0) pos = 0;
  currentIdx = vis[Math.max(0, Math.min(vis.length - 1, pos + delta))];"""

COUNTER_NEW = r"""(visible.length
      ? `${{visible.indexOf(currentIdx) + 1}} / ${{visible.length}} (of ${{DRAFTS.length}})`
      : `${{currentIdx + 1}} / ${{DRAFTS.length}}`);"""

HEADER_NEW = """<div class="nav">
    <input id="filterInput" type="text" placeholder="filter: pump, d'orsay, -wedge" onchange="applyFilter()"
           style="width:210px;padding:6px 8px;border-radius:6px;border:1px solid #444;background:#0f1a30;color:#eee;font-size:12px;" />
    <label style="font-size:12px;color:#aaa;white-space:nowrap;margin-right:8px;">
      <input type="checkbox" id="filterNew" onchange="applyFilter()" /> unreviewed</label>"""

INIT_NEW = r"""// optional: open the page with ?filter=pump,d'orsay
try {{
  const f = new URLSearchParams(window.location.search).get('filter');
  if (f) {{ document.getElementById('filterInput').value = f; applyFilter(); }}
}} catch(e) {{}}

// Keyboard navigation"""

EDITS = [
    ("let currentIdx = 0;", "let currentIdx = 0;\nlet visible = [];", "state variable"),
    ("function navigate(delta) {{", FILTER_JS, "filter function"),
    ("currentIdx = Math.max(0, Math.min(DRAFTS.length - 1, currentIdx + delta));", NAV_NEW, "navigate within visible items"),
    ("`${{currentIdx + 1}} / ${{DRAFTS.length}}`;", COUNTER_NEW, "counter"),
    ('<div class="nav">', HEADER_NEW, "header controls"),
    ("// Keyboard navigation", INIT_NEW, "?filter= parameter"),
]


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "applyFilter" in text:
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
    TARGET.with_suffix(".py.bak6").write_text(text, encoding="utf-8")
    TARGET.write_text(new, encoding="utf-8")
    print("review_labels.py patched (backup: review_labels.py.bak6)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
