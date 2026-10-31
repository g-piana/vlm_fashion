import json

pred = {s["image"].split("/")[-1]: s for s in json.load(open("outputs/test_v3.json"))}
gt = {s["image"]: s for s in json.load(open("data/splits/test_labels.json"))}

for name, s in pred.items():
    if "saffiano" in (s["tags"].get("finishing") or []):
        print(name, "| label finishing:", gt[name]["tags"].get("finishing"))