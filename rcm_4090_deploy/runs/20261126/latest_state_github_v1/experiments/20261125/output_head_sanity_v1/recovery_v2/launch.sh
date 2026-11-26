#!/bin/bash
set -u
cd /home/him1/mujoco_bc/rcm_4090_deploy/runs/20261125/output_head_sanity_code_v1/rcm_4090_deploy || exit 1
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH
source /home/him1/mujoco_bc/rcm_4090_deploy/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 RCM_DRIVER_VARIANT=g10
export RCM_RUNTIME_ROOT=/home/him1/mujoco_bc/rcm_4090_deploy
export RCM_OUTPUT_HEAD_SANITY_DIR=/home/him1/mujoco_bc/rcm_4090_deploy/runs/20261125/output_head_sanity_v1/recovery_v2
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_DISABLE_NATIVE_JIT=1 PYTHONDONTWRITEBYTECODE=1
printf 'Plan 8 cache repair recovery; 35min cap; output: %s\n' "$RCM_OUTPUT_HEAD_SANITY_DIR"
python -E -B -u -m scripts.research.output_head_sanity.workflow run --output-dir "$RCM_OUTPUT_HEAD_SANITY_DIR"
result=$?
printf '\nRecovery exited %s. Reports, logs and publication receipt: %s\n' "$result" "$RCM_OUTPUT_HEAD_SANITY_DIR"
exec /bin/bash --noprofile --norc
