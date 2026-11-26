#!/bin/bash
set -uo pipefail
cd /home/him1/mujoco_bc/rcm_4090_deploy || exit 1
out=runs/20261125/learned_terminal_recovery_pilot_v1
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH
source .venv/bin/activate
export DISPLAY=:1 XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 RCM_DRIVER_VARIANT=g10
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 TORCH_DISABLE_NATIVE_JIT=1
printf 'Authorized infrastructure recovery. Frozen recipe unchanged. One actual experiment only.\n'
printf 'Original failed artifacts remain in %s. New results: %s/recovery_v1/run\n' "$out" "$out"
timeout --signal=TERM --kill-after=10s 930s python -E -B -u "$out/recovery_entry_v1.py" _run --output-dir "$PWD/$out" 2>&1 | tee "$out/recovery_v1/terminal.log"
code=${PIPESTATUS[0]}
python -E -B -u "$out/recovery_entry_v1.py" _terminal_exit --output-dir "$PWD/$out" --exit-code "$code"
printf '\nExit %s. Results: %s/%s/recovery_v1/run/{workflow_status.json,report.md,summary.json,local_metrics.json}\n' "$code" "$PWD" "$out"
exec /bin/bash --noprofile --norc
