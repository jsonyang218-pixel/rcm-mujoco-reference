#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${1:-}"
if [[ -z "$output_root" ]]; then
  echo "usage: bash scripts/run_experience_replay_abc.sh OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$output_root"
cd "$project_root"
for mode in a_original b_static c_balanced; do
  python lerobot_training/run_model_pipeline.py \
    --config "configs/state_attention_experience_replay_${mode}_20261118.json" \
    --output-dir "$output_root/$mode"
done
