#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=outputs/logs/run_$(date +%Y%m%d_%H%M).log; mkdir -p outputs/logs
exec > >(tee -a "$LOG") 2>&1
NEW=qwen3vl_v14; CFG=configs/model_shoes_bags.yaml
TEST=data/splits/test_manifest.csv; EVAL=outputs/ext01_eval_manifest.csv

echo "== $(date) training $NEW"
python scripts/train_qwen3vl.py --config "$CFG"
for ck in $(ls -d models/$NEW/checkpoint-* | sort -t- -k2 -n | tail -n 2); do
  tag=${NEW}_$(basename "$ck")
  echo "== $(date) inference $ck"
  python scripts/infer.py --manifest "$TEST" --adapter "$ck" --output "outputs/test_${tag}.json"
  python scripts/infer.py --manifest "$EVAL" --adapter "$ck" --output "outputs/ext01eval_${tag}.json"
done
echo "== $(date) v13 with its frozen prompt"
python ../vlm_shoes_bags_v1/scripts/infer.py --manifest "$TEST" --adapter models/qwen3vl_v13/checkpoint-267 --output outputs/test_v13_frozen_204.json
python ../vlm_shoes_bags_v1/scripts/infer.py --manifest "$EVAL" --adapter models/qwen3vl_v13/checkpoint-267 --output outputs/ext01eval_v13_frozen.json
echo "== $(date) done"