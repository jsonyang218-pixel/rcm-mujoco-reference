#!/usr/bin/env bash
set -uo pipefail
cd /home/him1/mujoco_bc/rcm_4090_deploy
source .venv/bin/activate
unset PYTHONHOME PYTHONPATH
export DISPLAY=:1 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus XDG_RUNTIME_DIR=/run/user/1000
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 RCM_DRIVER_VARIANT=g10 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TORCH_DISABLE_NATIVE_JIT=1
out=runs/20261126/original_neural_terminal_pilot_v1
.venv/bin/python -E -B -u "$out/pilot.py" run 2>&1 | tee "$out/terminal.log"
rc=${PIPESTATUS[0]}
printf '\nPilot exit=%s; result: %s/summary.json ; failures: %s/failure.json\n' "$rc" "$out" "$out"
printf 'Terminal retained for review. Press Enter to close.\n'
read -r _terminal_close
exit "$rc"
