for ct in shoes bags; do
    python scripts/infer.py --images-dir data/raw/images/${ct}_women_batch04 \
      --category ${ct}_women --adapter models/qwen3vl_v9/checkpoint-168 \
      --output outputs/drafts/${ct}_women_batch04_drafts.json
done

for ckpt in checkpoint-168 checkpoint-252; do
    python scripts/infer.py --manifest data/splits/test_manifest.csv \
    --adapter models/qwen3vl_v9/${ckpt} \
    --output outputs/test_v10_${ckpt}.json
done