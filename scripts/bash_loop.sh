for ct in shoes bags; do
    python scripts/infer.py --images-dir data/raw/images/${ct}_women_batch04 \
      --category ${ct}_women --adapter models/qwen3vl_v9/checkpoint-168 \
      --output outputs/drafts/${ct}_women_batch04_drafts.json
done

for ckpt in checkpoint-168 checkpoint-252; do
    python scripts/infer.py --manifest data/splits/test_manifest.csv \
      --adapter models/qwen3vl_v10/${ckpt} \
      --output outputs/test_v10_${ckpt}.json
done


for ck in checkpoint-168 checkpoint-252; do
  python scripts/eval_heldout.py --predictions outputs/test_v11_$ck.json \
    --ground-truth data/splits/test_labels.json \
    --out-report outputs/reports/test_v11_$ck.json \
    --errors-csv outputs/reports/errors_v11_$ck.csv \
    --compare outputs/reports/baseline_v10ep3_bagsfix.json
done

for ad in qwen3vl_v12/checkpoint-267 qwen3vl_v13/checkpoint-178 qwen3vl_v13/checkpoint-267; do
  tag=$(echo $ad | tr '/' '_')
  python scripts/infer.py --manifest data/splits/test_manifest.csv \
    --adapter models/$ad --output outputs/test196_$tag.json
done

tags="qwen3vl_v12_checkpoint-267 $(ls outputs/test_qwen3vl_v13_checkpoint-*.json | sed 's#outputs/test_##; s#\.json##' | sort -t- -k2 -n)"
for tag in $tags; do
  python scripts/eval_heldout.py --predictions outputs/test_$tag.json \
    --ground-truth data/splits/test_labels.json \
    --out-report outputs/reports/r13_$tag.json \
    --errors-csv outputs/reports/errors_r13_$tag.csv \
    $( [ $tag != qwen3vl_v12_checkpoint-267 ] && echo --compare outputs/reports/r13_qwen3vl_v12_checkpoint-267.json )
done

mkdir -p docs && cp /mnt/c/Users/Giulio/Downloads/pipeline_runbook.md docs/ && git add docs && git commit -m "docs: pipeline runbook" && git push
