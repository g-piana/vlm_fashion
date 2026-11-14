"""
Attribute gallery: one self-contained HTML page showing every labelled image grouped by the value
of one field (or a combination), so outliers stand out at a glance. Click images to select them,
then "Copy" gives the file names for:  review_queue.py build --images "<names>"

    python scripts/attribute_gallery.py --config shoes_women --group heel_height+sole_type
    python scripts/attribute_gallery.py --config shoes_women --group finishing
    python scripts/attribute_gallery.py --config shoes_women --group subcategory \
        --subcategories "mary jane flat,d'orsay flat,mary jane pump,d'orsay pump"
    explorer.exe outputs/gallery/<name>.html        # open it in the Windows browser from WSL

List fields (finishing, model_details, ...): an image appears under each of its values.
Images are embedded as small thumbnails, so the page works anywhere.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
NONE = "(none)"


def values(tags: dict, field: str) -> list[str]:
    v = tags.get(field)
    if isinstance(v, list):
        return [str(x) for x in v] or [NONE]
    return [NONE if v in (None, "", "none", "null") else str(v)]


def thumb(path: Path, size: int) -> str:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=72)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="category config, e.g. shoes_women")
    ap.add_argument("--group", required=True, help="field, or fields joined by + (heel_height+sole_type)")
    ap.add_argument("--subcategories", help="only these subcategories (comma-separated)")
    ap.add_argument("--manifest", type=Path, default=ROOT / "configs/dataset_manifest.yaml")
    ap.add_argument("--size", type=int, default=200, help="thumbnail size in px")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    fields = [f.strip() for f in a.group.split("+") if f.strip()]
    subs = {s.strip() for s in (a.subcategories or "").split(",") if s.strip()}
    groups: dict[str, list[str]] = defaultdict(list)
    images: dict[str, dict] = {}
    for b in (yaml.safe_load(a.manifest.read_text(encoding="utf-8")) or {}).get("batches", []):
        for s in json.loads((ROOT / b["labels"]).read_text(encoding="utf-8")):
            t = s.get("tags")
            if not isinstance(t, dict) or s.get("category_config", "shoes_women") != a.config:
                continue
            if subs and t.get("subcategory") not in subs:
                continue
            name = s["image"]
            if name not in images:
                info = " | ".join(f"{k}: {', '.join(values(t, k))}" for k in
                                  ["subcategory"] + [f for f in fields if f != "subcategory"])
                images[name] = {"src": thumb(ROOT / b["images_dir"] / name, a.size),
                                "info": f"{name} [{b['name']}] | {info}"}
            if len(fields) == 1:
                keys = values(t, fields[0])
            else:  # combination of fields: one group per combination
                keys = [" + ".join(", ".join(values(t, f)) for f in fields)]
            for k in keys:
                groups[k].append(name)

    out = a.out or ROOT / "outputs" / "gallery" / f"{a.config}_{'+'.join(fields)}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    order = sorted(groups, key=lambda k: (k == NONE, -len(groups[k]), k))
    sections = []
    for k in order:
        cells = "".join(
            f'<figure data-n="{html.escape(n)}" title="{html.escape(images[n]["info"])}">'
            f'<img loading="lazy" data-k="{html.escape(n)}"><figcaption>{html.escape(n)}</figcaption></figure>'
            for n in groups[k])
        sections.append(f'<section><h2>{html.escape(k)} <small>({len(groups[k])})</small></h2>'
                        f'<div class="grid">{cells}</div></section>')
    srcs = json.dumps({n: v["src"] for n, v in images.items()})
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(a.config)} by {html.escape(a.group)}</title>
<style>
:root{{--bg:#111;--fg:#eee;--muted:#999;--sel:#4da3ff;--card:#1c1c1c}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}}
header{{position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:2;display:flex;gap:8px;flex-wrap:wrap;align-items:center}}
h1{{font-size:18px;margin:0 12px 0 0}} h2{{font-size:16px;margin:24px 0 8px}} small{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax({a.size}px,1fr));gap:8px}}
figure{{margin:0;background:var(--card);padding:4px;border:3px solid transparent;cursor:pointer;border-radius:6px}}
figure.on{{border-color:var(--sel)}} img{{width:100%;aspect-ratio:1;object-fit:contain;background:#fff}}
figcaption{{font-size:11px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
textarea{{width:100%;height:48px;background:var(--card);color:var(--fg);border:1px solid #333}}
button{{padding:6px 12px}}
</style></head><body>
<header><h1>{html.escape(a.config)} — {html.escape(a.group)} ({len(images)} images)</h1>
<button onclick="copySel()">Copy selection</button><button onclick="clearSel()">Clear</button>
<span id="count">0 selected</span><textarea id="sel" readonly placeholder="selected file names appear here"></textarea></header>
{''.join(sections)}
<script>
const SRC={srcs};
document.querySelectorAll('img[data-k]').forEach(i=>i.src=SRC[i.dataset.k]||'');
const sel=new Set();
function render(){{document.querySelectorAll('figure').forEach(f=>f.classList.toggle('on',sel.has(f.dataset.n)));
 document.getElementById('sel').value=[...sel].join(',');document.getElementById('count').textContent=sel.size+' selected';}}
document.querySelectorAll('figure').forEach(f=>f.onclick=()=>{{sel.has(f.dataset.n)?sel.delete(f.dataset.n):sel.add(f.dataset.n);render();}});
function copySel(){{const t=document.getElementById('sel');t.select();try{{navigator.clipboard.writeText(t.value)}}catch(e){{document.execCommand('copy')}}}}
function clearSel(){{sel.clear();render();}}
</script></body></html>"""
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}: {len(images)} images in {len(groups)} groups")
    for k in order:
        print(f"  {len(groups[k]):4d}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
