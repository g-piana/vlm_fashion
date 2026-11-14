"""
Measure real token lengths of training sequences (system prompt + image + user message + answer)
against model.max_seq_length, to check whether training truncated the answers.

    python scripts/measure_sequence.py            # 20 samples per category config
    python scripts/measure_sequence.py --n 100
Reads the base model name from configs/model.yaml and samples real images and labels from the manifest.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import yaml
from PIL import Image
from transformers import AutoProcessor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from category_config import build_system_prompt, get_user_message  # noqa: E402


def model_name(cfg: dict) -> str:
    m = cfg.get("model", {})
    for key in ("name", "model_name", "base_model", "base", "id", "path", "pretrained"):
        if isinstance(m.get(key), str):
            return m[key]
    for v in m.values():  # fall back: first value that looks like a repo id or a path
        if isinstance(v, str) and ("/" in v):
            return v
    sys.exit(f"could not find the model name in configs/model.yaml 'model' section: {m}")


def user_message(image_type: str) -> str:
    try:
        return get_user_message(image_type)
    except TypeError:
        return get_user_message()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="samples per category config")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/model.yaml")
    ap.add_argument("--manifest", type=Path, default=ROOT / "configs/dataset_manifest.yaml")
    a = ap.parse_args()

    mcfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    name, max_len = model_name(mcfg), mcfg["model"]["max_seq_length"]
    print(f"model: {name}   max_seq_length: {max_len}")
    proc = AutoProcessor.from_pretrained(name)

    by_cfg: dict[str, list] = {}
    for b in (yaml.safe_load(a.manifest.read_text(encoding="utf-8")) or {}).get("batches", []):
        for s in json.loads((ROOT / b["labels"]).read_text(encoding="utf-8")):
            if isinstance(s.get("tags"), dict):
                by_cfg.setdefault(s.get("category_config", "shoes_women"), []).append((b, s))

    random.seed(0)
    for cfg_name, samples in sorted(by_cfg.items()):
        system = build_system_prompt(cfg_name, "product")
        totals, over = [], 0
        for b, s in random.sample(samples, min(a.n, len(samples))):
            img = Image.open(ROOT / b["images_dir"] / s["image"]).convert("RGB")
            answer = f"DESCRIPTION:\n{s.get('description', '')}\n\nTAGS:\n{json.dumps(s['tags'], ensure_ascii=False)}"
            msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": user_message(s.get("image_type", "product"))}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]}]
            text = proc.apply_chat_template(msgs, tokenize=False)
            n = len(proc(text=[text], images=[img], return_tensors="pt")["input_ids"][0])
            totals.append(n)
            over += n > max_len
        sys_tokens = len(proc.tokenizer(system)["input_ids"])
        print(f"{cfg_name:14s} system prompt {sys_tokens:5d} tok | full sequence median {statistics.median(totals):6.0f}"
              f"  max {max(totals):6d} | over {max_len}: {over}/{len(totals)}")
    print("Anything over max_seq_length was truncated in training (the end of the answer = the last tag fields).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
