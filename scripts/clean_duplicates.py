import json, shutil, datetime

path = "data/raw/labels/bags_women_batch04.json"   # batch that holds the duplicate
 # file names of the copies to remove
drop = {
         "BAGS_DOCTORS_B_004_001.webp",       
         "BAGS_MESSENGER_B_004_003.webp",  
         "BAGS_DUFFLE_B_004_001.webp",  
        }

stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
shutil.copy(path, f"{path}.bak_dedup_{stamp}")
data = json.load(open(path, encoding="utf-8"))
seen, out = set(), []
for s in data:
    if s["image"] in drop or s["image"] in seen:   # also drops a second same-name copy
        continue
    seen.add(s["image"]); out.append(s)
json.dump(out, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"{len(data)} -> {len(out)} samples")
