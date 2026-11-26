#!/bin/bash
set -u
cd /home/him1/mujoco_bc/rcm_4090_deploy/runs/20261125/output_head_sanity_code_v1/rcm_4090_deploy || exit 1
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH
source /home/him1/mujoco_bc/rcm_4090_deploy/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 RCM_DRIVER_VARIANT=g10
export RCM_RUNTIME_ROOT=/home/him1/mujoco_bc/rcm_4090_deploy
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_DISABLE_NATIVE_JIT=1
export PYTHONDONTWRITEBYTECODE=1
printf 'Plan 8: 35 minute cap; logs and reports: /home/him1/mujoco_bc/rcm_4090_deploy/runs/20261125/output_head_sanity_v1/\n'
python -E -B -u -m scripts.research.output_head_sanity.workflow run --output-dir /home/him1/mujoco_bc/rcm_4090_deploy/runs/20261125/output_head_sanity_v1
result=$?
printf '\nPlan 8 exited %s; report.md, metrics.json, publication_receipt.json are in the output directory.\n' "$result"
exec /bin/bash --noprofile --norc
