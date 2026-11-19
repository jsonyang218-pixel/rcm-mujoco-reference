#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${1:-}"
if [[ -z "$output_root" ]]; then
  echo "usage: bash scripts/run_experience_ratio_experiments.sh OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$output_root"
cd "$project_root"
for ratio in r0_natural r10 r20; do
  python lerobot_training/run_model_pipeline.py \
    --config "configs/state_attention_experience_ratio_${ratio}_20261118.json" \
    --output-dir "$output_root/$ratio"
done
