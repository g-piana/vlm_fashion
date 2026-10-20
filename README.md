# Local VLM — Fashion Product Descriptions & Tags

Single-model pipeline using **Qwen3-VL-8B** to generate concise descriptions
and structured JSON tags from fashion product images. Supports multiple
categories (shoes, bags, ready-to-wear) and multi-item runway looks.

## Architecture

```
Image → ResNet classifier → category / image_type
                ↓
        Route to category prompt config
                ↓
       Qwen3-VL (single LoRA adapter) → description + tags
```

One fine-tuned model, many category vocabularies. Each category has its own
YAML config defining the relevant attributes and controlled vocabulary. The
system prompt is built dynamically at inference time based on the category.

## Quick start

```bash
# 1. Prepare data (merges all label files, applies per-category prompts)
python scripts/prepare_dataset.py

# 2. Fine-tune (one adapter, all categories)
python scripts/train_qwen3vl.py --config configs/model.yaml

# 3. Inference
python scripts/infer.py --image shoe.jpg --category shoes
python scripts/infer.py --image runway.jpg --category look
python scripts/infer.py --manifest batch.csv --adapter models/qwen3vl/final

# 4. Evaluate
python scripts/evaluate.py \
    --predictions outputs/results.json \
    --ground-truth data/splits/test.parquet
```

## Project structure

```
local-vlm/
├── configs/
│   ├── model.yaml                # Shared: model, LoRA, training params
│   └── categories/
│       ├── _base.yaml            # Shared attributes (color, material, ...)
│       ├── shoes.yaml            # + heel_type, toe_shape, sole_type, ...
│       ├── bags.yaml             # + handle_type, bag_closure, hardware, ...
│       ├── ready_to_wear.yaml    # + fit, neckline, collar, sleeve, ...
│       └── look.yaml             # Multi-item: merges all category vocabs
├── data/
│   ├── raw/
│   │   ├── labels/               # JSON files per category/batch
│   │   └── images/               # Symlink to image storage
│   └── splits/                   # train/val/test parquet
├── scripts/
│   ├── category_config.py        # Config loader & prompt builder
│   ├── prepare_dataset.py        # Merge labels, build prompts, split
│   ├── train_qwen3vl.py          # QLoRA fine-tuning
│   ├── infer.py                  # Inference with category routing
│   └── evaluate.py               # Per-category metrics
├── models/                       # LoRA adapters (git-ignored)
└── outputs/                      # Results (git-ignored)
```

## Label format

Each JSON file in `data/raw/labels/` is a list of samples:

```json
[
  {
    "image": "BG53049-01.jpg",
    "category_config": "shoes",
    "image_type": "product",
    "description": "Stiletto-heel open-toe sandal in black...",
    "tags": {
      "category": "sandal",
      "subcategory": "stiletto sandal",
      "toe_shape": "open-toe",
      "heel_type": "stiletto",
      "primary_color": "black",
      "material": "patent leather",
      ...
    }
  }
]
```

For looks (runway/editorial), `image_type` is `"look"` and `tags` is an
**array** of items.

## Adding a new category

1. Create `configs/categories/your_category.yaml` with `inherits: _base`
2. Define category-specific attributes and values
3. Add labelled data to `data/raw/labels/your_category_batch01.json`
4. Re-run `prepare_dataset.py` and `train_qwen3vl.py`

The model improves across all categories with each new batch of data.
