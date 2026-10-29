import json, glob, collections
c = collections.Counter()
for f in glob.glob("data/raw/labels/*.json"):
    for it in json.load(open(f, encoding="utf-8")):
        c.update((it.get("tags") or {}).get("embellishments") or [])
print(c.most_common())
