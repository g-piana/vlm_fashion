#!/usr/bin/env bash
# Overnight: train v13, then run inference on the test set with v12 (baseline) and every v13 checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=outputs/logs/run_v13_$(date +%Y%m%d_%H%M).log
mkdir -p outputs/logs
exec > >(tee -a "$LOG") 2>&1

echo "== $(date) training v13"
python scripts/train_qwen3vl.py --config configs/model.yaml

adapters=("qwen3vl_v12/checkpoint-267")
for ck in $(ls -d models/qwen3vl_v13/checkpoint-* | sort -t- -k2 -n); do
  adapters+=("${ck#models/}")
done

for ad in "${adapters[@]}"; do
  tag=$(echo "$ad" | tr '/' '_')
  echo "== $(date) inference $ad"
  python scripts/infer.py --manifest data/splits/test_manifest.csv \
    --adapter "models/$ad" --output "outputs/test_$tag.json"
done
echo "== $(date) done"