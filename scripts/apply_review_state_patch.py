"""
Fix the "description from another batch" bug in review_labels.py.

ROOT CAUSE
  Saved review progress was keyed only by category (+ export dir), never by the
  drafts file, and was restored BY POSITION. Reviewing bags_women batch02 after
  batch01 therefore laid batch01's saved descriptions/tags over batch02's
  images (same category, same key, item 0 -> item 0 ...).
  The Python side had a second flaw: an unconditional loop after the
  "different batch - ignoring" message that applied the old progress anyway.

FIX
  Browser (localStorage)
    * key = category + hash of THIS drafts file's image list
    * restore matches items by image filename, never by index
    * the saved record carries its image name
    * ?fresh=1 in the URL ignores and deletes the saved state
  Server (outputs/<drafts-name>_review_progress.json)
    * one progress file per drafts file
    * restore matches by image filename; the stray unconditional loop is gone

Run from the project root (after the earlier patches):
    python scripts/apply_review_state_patch.py
A .bak3 copy is kept; nothing is written if an anchor is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "review_labels.py"

# ---- JavaScript (inside a Python f-string => literal braces doubled) -------
NEW_JS_BLOCK = r"""// Saved progress belongs to ONE drafts file: the key contains a hash of its image
// list, and items are matched by image filename, never by position. Another batch
// (or another run with different images) can therefore never pick this state up.
const imgName = (d) => String((d && d.image) || '').split('/').pop();
function hashStr(s) {{
  let h = 5381;
  for (let i = 0; i < s.length; i++) {{ h = ((h << 5) + h + s.charCodeAt(i)) | 0; }}
  return (h >>> 0).toString(36);
}}
const draftSig = hashStr(DRAFTS.map(imgName).join('|'));
const savedKey = 'review2_' + CATEGORY + '_' + draftSig;
try {{
  if (new URLSearchParams(window.location.search).has('fresh')) {{
    localStorage.removeItem(savedKey);
    console.log('fresh start: saved progress for this drafts file deleted');
  }} else {{
    const saved = localStorage.getItem(savedKey);
    if (saved) {{
      const byImage = new Map(JSON.parse(saved).filter(s => s.image).map(s => [s.image, s]));
      let restored = 0;
      DRAFTS.forEach((d, i) => {{
        const s = byImage.get(imgName(d));
        if (!s) return;
        d.description = s.description ?? d.description;
        d.tags = s.tags ?? d.tags;
        d._skip = s._skip ?? false;
        if (s._reviewed) reviewed.add(i);
        restored++;
      }});
      console.log('restored ' + restored + ' / ' + DRAFTS.length + ' items from saved progress');
    }}
  }}
}} catch(e) {{}}

"""

# ---- Python side -------------------------------------------------------------
NEW_PY_BLOCK = '''    # Saved progress belongs to ONE drafts file (progress file named after it) and
    # is matched to items by image filename, so it can never land on other images.
    progress_path = Path("outputs") / f"{Path(args.drafts).stem}_review_progress.json"
    ReviewHandler.progress_path = progress_path
    if progress_path.exists():
        with open(progress_path, encoding="utf-8") as f:
            saved = json.load(f)
        by_image = {Path(s.get("image", "")).name: s for s in saved if s.get("image")}
        restored = 0
        for i, d in enumerate(drafts):
            s = by_image.get(Path(d.get("image", "")).name)
            if s:
                drafts[i] = s
                restored += 1
        log.info("Restored %d/%d items from %s", restored, len(drafts), progress_path)

'''


def replace_between(text: str, start: str, end: str, new: str, label: str, problems: list):
    if text.count(start) != 1 or text.count(end) != 1:
        problems.append(f"{label}: markers found {text.count(start)}x / {text.count(end)}x (expected 1 / 1)")
        return text
    a, b = text.index(start), text.index(end)
    if a > b:
        problems.append(f"{label}: markers in unexpected order")
        return text
    return text[:a] + new + text[b:]


def replace_once(text: str, old: str, new: str, label: str, problems: list):
    if text.count(old) != 1:
        problems.append(f"{label}: anchor found {text.count(old)}x (expected 1): {old[:60]!r}")
        return text
    return text.replace(old, new, 1)


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "draftSig" in text:
        print("already patched")
        return 0
    problems: list[str] = []

    new = replace_between(
        text,
        "// Load saved progress from localStorage",
        "function navigate(delta) {{",
        NEW_JS_BLOCK,
        "localStorage restore",
        problems,
    )
    new = replace_once(new, "_reviewed: reviewed.has(i),",
                       "_reviewed: reviewed.has(i),\n    image: imgName(d),", "saveProgress", problems)
    new = replace_between(
        new,
        "    # Load saved progress if it exists",
        "    vocabs = load_vocabularies(args.category)",
        NEW_PY_BLOCK,
        "server progress restore",
        problems,
    )
    new = replace_once(
        new,
        'save_path = Path("outputs") / f"{self.category}_review_progress.json"',
        "save_path = self.progress_path",
        "/save handler",
        problems,
    )
    # class attribute on ReviewHandler: use the first anchor line that occurs exactly once
    # ('batch_name: str = "batch01"' alone is NOT unique in the real file)
    for anchor in ('html_content: str = ""',
                   'export_dir: Path = Path("data/raw/labels")',
                   'category: str = "shoes_women"'):
        if new.count(anchor) == 1:
            new = new.replace(
                anchor,
                anchor + '\n    progress_path: Path = Path("outputs") / "review_progress.json"',
                1,
            )
            break
    else:
        problems.append("handler attribute: no unique anchor line found in class ReviewHandler")

    if problems:
        print("NOTHING WRITTEN:")
        for p in problems:
            print("  -", p)
        return 1
    TARGET.with_suffix(".py.bak3").write_text(text, encoding="utf-8")
    TARGET.write_text(new, encoding="utf-8")
    print("review_labels.py patched (backup: review_labels.py.bak3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())