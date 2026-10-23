"""
Inference pipeline with category-based prompt routing.

The category can be provided explicitly or detected by an upstream
classifier (e.g. ResNet). The correct vocabulary is loaded from the
category YAML and injected into the system prompt.

Usage:
    # Single image, explicit category
    python scripts/infer.py --image shoe.jpg --category shoes_women

    # Single image, look/runway
    python scripts/infer.py --image runway.jpg --category look

    # Batch with a manifest CSV (image_path, category_config, image_type)
    python scripts/infer.py --manifest data/raw/batch_manifest.csv

    # Batch directory (defaults to "shoes" category)
    python scripts/infer.py --images-dir data/raw/images --category shoes_women

    # With fine-tuned adapter
    python scripts/infer.py --image shoe.jpg --category shoes_women \
        --adapter models/qwen3vl/final
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from category_config import build_system_prompt, get_user_message, list_configs
from taxonomy_utils import apply_taxonomy_rules

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def parse_model_output(text: str) -> dict:
    """Parse combined DESCRIPTION + TAGS output."""
    result = {"description": "", "tags": None}

    desc_match = re.search(r"DESCRIPTION:\s*\n(.+?)(?:\n\nTAGS:|\Z)", text, re.DOTALL)
    if desc_match:
        result["description"] = desc_match.group(1).strip()

    tags_match = re.search(r"TAGS:\s*\n([\[\{].+)", text, re.DOTALL)
    if tags_match:
        try:
            result["tags"] = json.loads(tags_match.group(1))
        except json.JSONDecodeError:
            log.warning("Failed to parse tags JSON from output")
            result["tags_raw"] = tags_match.group(1).strip()

    return result


class FashionAnalyzer:
    """Generates descriptions + tags using Qwen3-VL with category routing."""

    def __init__(self, adapter_path: str | None = None):
        from unsloth import FastVisionModel

        model_name = adapter_path or "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
        log.info("Loading model from %s", model_name)
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name, load_in_4bit=True,
        )
        FastVisionModel.for_inference(self.model)
        self._prompt_cache: dict[tuple, str] = {}

    def _get_prompt(self, category: str, image_type: str) -> str:
        key = (category, image_type)
        if key not in self._prompt_cache:
            self._prompt_cache[key] = build_system_prompt(category, image_type)
        return self._prompt_cache[key]

    def analyze(
        self,
        image: Image.Image,
        category: str = "shoes",
        image_type: str = "product",
    ) -> dict:
        system_prompt = self._get_prompt(category, image_type)
        user_msg = get_user_message()

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_msg},
                ],
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=1024, temperature=0.3,
                top_p=0.9, do_sample=True,
            )

        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        result = parse_model_output(text)
        # derive category from subcategory + record taxonomy/vocabulary issues
        return apply_taxonomy_rules(result, category)


def load_manifest(path: Path) -> list[dict]:
    """Load a CSV manifest with columns: image_path, category_config, image_type."""
    items = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append({
                "image_path": Path(row["image_path"]),
                "category": row.get("category_config", "shoes"),
                "image_type": row.get("image_type", "product"),
            })
    return items


def main():
    parser = argparse.ArgumentParser(description="Fashion product analysis")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path)
    group.add_argument("--images-dir", type=Path)
    group.add_argument("--manifest", type=Path,
                       help="CSV with image_path, category_config, image_type")
    parser.add_argument("--category", type=str, default="shoes",
                        choices=list_configs(),
                        help="Category config to use (ignored with --manifest)")
    parser.add_argument("--image-type", type=str, default="product",
                        choices=["product", "look"])
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/results.json"))
    parser.add_argument("--extensions", nargs="+",
                        default=[".jpg", ".jpeg", ".png", ".webp"])
    args = parser.parse_args()

    analyzer = FashionAnalyzer(args.adapter)

    # Build work items
    if args.manifest:
        items = load_manifest(args.manifest)
    elif args.image:
        items = [{"image_path": args.image, "category": args.category,
                  "image_type": args.image_type}]
    else:
        items = [
            {"image_path": p, "category": args.category,
             "image_type": args.image_type}
            for p in sorted(args.images_dir.iterdir())
            if p.suffix.lower() in args.extensions
        ]

    log.info("Processing %d images", len(items))

    results = []
    for item in tqdm(items, desc="Analyzing"):
        try:
            image = Image.open(item["image_path"]).convert("RGB")
            result = analyzer.analyze(
                image, category=item["category"], image_type=item["image_type"]
            )
            result["image"] = str(item["image_path"])
            result["category_config"] = item["category"]
            result["image_type"] = item["image_type"]
            results.append(result)
        except Exception as e:
            log.error("Failed on %s: %s", item["image_path"], e)
            results.append({"image": str(item["image_path"]), "error": str(e)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Saved %d results to %s", len(results), args.output)

    # Print first result
    if results and "error" not in results[0]:
        print(f"\n── Description ({results[0].get('category_config')}) ──")
        print(results[0]["description"])
        if results[0].get("tags"):
            print("\n── Tags ──")
            print(json.dumps(results[0]["tags"], indent=2))


if __name__ == "__main__":
    main()
