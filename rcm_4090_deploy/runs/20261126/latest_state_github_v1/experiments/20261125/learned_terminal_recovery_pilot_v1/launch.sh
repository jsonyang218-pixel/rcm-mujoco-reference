#!/bin/bash
set -uo pipefail
cd /home/him1/mujoco_bc/rcm_4090_deploy || exit 1
out=runs/20261125/learned_terminal_recovery_pilot_v1
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 RCM_DRIVER_VARIANT=g10
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 TORCH_DISABLE_NATIVE_JIT=1
printf 'learned terminal recovery: one ridge fit, 3 train targets, E43 isolated, six local rollouts. 15 minute cap.\n'
timeout --signal=TERM --kill-after=10s 930s python -E -B -u "$out/pilot.py" run --output-dir "$PWD/$out" 2>&1 | tee "$out/terminal.log"
code=${PIPESTATUS[0]}
if [ "$code" -ne 0 ]; then
    python -E -B -u "$out/pilot.py" terminal_failure --output-dir "$PWD/$out" --exit-code "$code"
fi
printf '\nExit %s. Results: %s/%s/{workflow_status.json,report.md,summary.json,local_metrics.json}\n' "$code" "$PWD" "$out"
exec /bin/bash --noprofile --norc
