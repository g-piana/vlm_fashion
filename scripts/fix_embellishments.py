import json, sys, shutil
from pathlib import Path
MAP = {"logo-element":"logo","metal-plate":"logo","braid":"trim","charme":"charm"}
DROP = {"hardware","zipper","o-ring","pin","shell","none"}
for f in sys.argv[1:]:
    p = Path(f); items = json.loads(p.read_text(encoding="utf-8")); n = 0
    for it in items:
        t = it.get("tags")
        if not isinstance(t, dict) or t.get("embellishments") is None:
            continue
        e = t["embellishments"]; e = [e] if isinstance(e, str) else e
        new = list(dict.fromkeys(MAP.get(v, v) for v in e if v not in DROP))
        if new != t["embellishments"]:
            t["embellishments"] = new; n += 1
    shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
    p.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(p, n, "items changed")