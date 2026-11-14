import yaml, json; from PIL import Image; from collections import Counter

for b in yaml.safe_load(open("configs/dataset_manifest.yaml"))["batches"]:
    sizes = Counter()
    for s in json.load(open(b["labels"])):
        w, h = Image.open(f"{b['images_dir']}/{s['image']}").size
        sizes["<=1024" if max(w, h) <= 1024 else ">1024"] += 1
        if max(w, h) < 400: sizes["<400 (low-res)"] += 1
    print(b["name"], dict(sizes))