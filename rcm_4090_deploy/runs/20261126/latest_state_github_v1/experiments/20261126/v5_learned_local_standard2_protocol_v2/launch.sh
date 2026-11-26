#!/usr/bin/env bash
set -uo pipefail
cd /home/him1/mujoco_bc/rcm_4090_deploy
source .venv/bin/activate
unset PYTHONHOME PYTHONPATH
export DISPLAY=:1 XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 RCM_DRIVER_VARIANT=g10 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
out=runs/20261126/v5_learned_local_standard2_protocol_v2
.venv/bin/python -E -B -u "$out/standard2.py" supervise 2>&1 | tee "$out/supervisor.terminal.log"
rc=${PIPESTATUS[0]}
printf '\nSupervisor exit=%s; results: %s/summary.json\n' "$rc" "$out"
printf 'Terminal retained for review. Press Enter to close.\n'
read -r _terminal_close
exit "$rc"
