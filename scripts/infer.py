"""
Unified inference pipeline: Qwen3-VL for descriptions + structured tags.

Processes a single image or batch-processes a directory.

Usage:
    # Single image
    python scripts/infer.py --image path/to/shoe.jpg --adapter models/qwen3vl/final

    # Batch
    python scripts/infer.py --images-dir data/raw/images --adapter models/qwen3vl/final

    # Zero-shot (no adapter, base model only)
    python scripts/infer.py --image path/to/shoe.jpg
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_system_prompt(config_path: str = "configs/qwen3vl_qlora.yaml") -> str:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["system_prompt"].strip()


def parse_model_output(text: str) -> dict:
    """Parse the combined DESCRIPTION + TAGS output into structured data."""
    result = {"description": "", "tags": None, "raw_output": text}

    # Extract description
    desc_match = re.search(r"DESCRIPTION:\s*\n(.+?)(?:\n\nTAGS:|\Z)", text, re.DOTALL)
    if desc_match:
        result["description"] = desc_match.group(1).strip()

    # Extract tags JSON
    tags_match = re.search(r"TAGS:\s*\n(\{.+\})", text, re.DOTALL)
    if tags_match:
        try:
            result["tags"] = json.loads(tags_match.group(1))
        except json.JSONDecodeError:
            log.warning("Failed to parse tags JSON")

    return result


class FashionAnalyzer:
    """Generates descriptions + tags using Qwen3-VL."""

    def __init__(self, adapter_path: str | None = None, config_path: str = "configs/qwen3vl_qlora.yaml"):
        from unsloth import FastVisionModel

        model_name = adapter_path or "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
        log.info("Loading model from %s", model_name)

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name, load_in_4bit=True,
        )
        FastVisionModel.for_inference(self.model)
        self.system_prompt = load_system_prompt(config_path)

    def analyze(self, image: Image.Image) -> dict:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Analyze this fashion product."},
                ],
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=512, temperature=0.3,
                top_p=0.9, do_sample=True,
            )

        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return parse_model_output(text)


def process_image(image_path: Path, analyzer: FashionAnalyzer) -> dict:
    image = Image.open(image_path).convert("RGB")
    result = analyzer.analyze(image)
    result["image"] = str(image_path)
    del result["raw_output"]  # drop from saved results
    return result


def main():
    parser = argparse.ArgumentParser(description="Fashion product analysis")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path)
    group.add_argument("--images-dir", type=Path)
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter (omit for zero-shot)")
    parser.add_argument("--config", type=str, default="configs/qwen3vl_qlora.yaml")
    parser.add_argument("--output", type=Path, default=Path("outputs/results.json"))
    parser.add_argument("--extensions", nargs="+",
                        default=[".jpg", ".jpeg", ".png", ".webp"])
    args = parser.parse_args()

    analyzer = FashionAnalyzer(args.adapter, args.config)

    if args.image:
        image_paths = [args.image]
    else:
        image_paths = sorted(
            p for p in args.images_dir.iterdir()
            if p.suffix.lower() in args.extensions
        )

    log.info("Processing %d images", len(image_paths))

    results = []
    for path in tqdm(image_paths, desc="Analyzing"):
        try:
            result = process_image(path, analyzer)
            results.append(result)
        except Exception as e:
            log.error("Failed on %s: %s", path, e)
            results.append({"image": str(path), "error": str(e)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Saved %d results to %s", len(results), args.output)

    # Print first result
    if results and "error" not in results[0]:
        print("\n── Description ──")
        print(results[0]["description"])
        if results[0].get("tags"):
            print("\n── Tags ──")
            print(json.dumps(results[0]["tags"], indent=2))


if __name__ == "__main__":
    main()
