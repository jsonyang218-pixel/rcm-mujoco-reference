#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
exec python lerobot_training/run_model_pipeline.py \
  --config configs/frontiers2022_ddpg_pipeline.json "$@"
