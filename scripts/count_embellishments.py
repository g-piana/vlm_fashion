import json, glob, collections
c = collections.Counter()
for f in glob.glob("data/raw/labels/*.json"):
    for it in json.load(open(f, encoding="utf-8")):
        v = (it.get("tags") or {}).get("embellishments") or []
        c.update([v] if isinstance(v, str) else v)
print(c.most_common())
