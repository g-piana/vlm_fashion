# Local VLM — Fashion Product Descriptions & Tags

Single-model pipeline using **Qwen3-VL-8B** to generate both concise
descriptions and structured JSON tags from fashion product images.
Runs entirely on local hardware (RTX 3090, 128 GB RAM).

| Task | Output | Example |
|---|---|---|
| Description | 2-3 sentence natural language | "Stiletto-heel open-toe sandal in black..." |
| Tags | Structured JSON with controlled vocabulary | `{"category": "sandal", "heel_type": "stiletto", ...}` |

## Quick start

```bash
# 1. Prepare data
python scripts/prepare_dataset.py \
    --input data/raw/labels.json \
    --images-dir data/raw/images \
    --output-dir data/splits

# 2. Fine-tune
python scripts/train_qwen3vl.py --config configs/qwen3vl_qlora.yaml

# 3. Inference (with adapter)
python scripts/infer.py --image path/to/shoe.jpg --adapter models/qwen3vl/final

# 4. Zero-shot (no fine-tuning, base model)
python scripts/infer.py --image path/to/shoe.jpg

# 5. Evaluate
python scripts/evaluate.py \
    --predictions outputs/results.json \
    --ground-truth data/splits/test.parquet
```

## Project structure

```
local-vlm/
├── configs/
│   └── qwen3vl_qlora.yaml       # Model, LoRA, training, and controlled vocabulary
├── data/
│   ├── raw/                      # Original images + labels.json
│   │   └── images/               # (symlink to E: drive)
│   └── splits/                   # train/val/test parquet files
├── scripts/
│   ├── prepare_dataset.py        # Data prep + splits
│   ├── train_qwen3vl.py          # QLoRA fine-tuning via Unsloth
│   ├── infer.py                  # Inference (single or batch)
│   └── evaluate.py               # ROUGE-L, BERTScore, tag accuracy/F1
├── models/                       # Saved LoRA adapters (git-ignored)
├── outputs/                      # Inference results (git-ignored)
├── requirements.txt
└── README.md
```

## Data format

Input `labels.json`:

```json
[
  {
    "image": "BG53049-01.jpg",
    "description": "Stiletto-heel open-toe sandal in black features a tall heel with metallic finish...",
    "tags": {
      "category": "sandal",
      "subcategory": "stiletto sandal",
      "toe_shape": "open-toe",
      "heel_type": "stiletto",
      "heel_height": "high",
      "closure": "slip-on",
      "primary_color": "black",
      "secondary_colors": ["silver"],
      "material": "patent leather",
      "embellishments": ["fringe", "raw edges"],
      "style_tags": ["strappy", "cage", "avant-garde"],
      "occasion": ["evening", "editorial"],
      "season": "all-season"
    }
  }
]
```

## Output format

The model produces:

```
DESCRIPTION:
Stiletto-heel open-toe sandal in black features a tall, slender heel
with a metallic finish. Multiple wide straps wrap around the foot and
ankle in a crisscrossing construction. Raw fringe edges outline each
strap, creating texture throughout the design.

TAGS:
{
  "category": "sandal",
  "subcategory": "stiletto sandal",
  "toe_shape": "open-toe",
  ...
}
```

## Controlled vocabulary

All tag values are constrained to a fixed vocabulary defined in
`configs/qwen3vl_qlora.yaml`. This ensures consistent, machine-readable
output suitable for database ingestion and search/filter UIs.
