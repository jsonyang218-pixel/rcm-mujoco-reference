# Local OCT AdaLN preflight, 2026-11-25

Code: `runs/20261125/local_oct_adaln_code_v1`, based on Git commit
`a9d6ed37ae1af76658ba4a85928fb545c91196df`. Synthetic inputs only; no dataset,
training, closed-loop run, or performance comparison.

- `python -B -m unittest -v scripts.verification.test_local_oct_adaln`: three
  contract checks passed on CPU. They test mocked conditioning, not policy
  performance.
- GPU preflight loaded the historical `baseline_task2_step4000` checkpoint
  twice on RTX 4090, with the branch disabled and enabled at zero output
  initialization. On one synthetic 224x224 input, the predicted velocity was
  bit-for-bit equal; maximum absolute difference was 0. The branch output
  projection had finite, nonzero gradient norm 6.6960.
- Initial GPU backward failed because `torch` was imported before the code set
  `TORCH_DISABLE_NATIVE_JIT=1`; Triton then tried to compile and failed because
  `/usr/include/python3.12/Python.h` is absent. Setting this environment
  variable before process start allowed the synthetic preflight to pass.
- The specified base commit omits `lerobot_training/navigation_overlay.py`,
  although training imports it. The existing deployment file was copied into
  the isolated branch to restore this dependency; baseline compatibility still
  requires full workflow preflight.
- Formal two-batch smoke, training, probes, evaluation, reporting and push have
  not run. A compliant multi-stage orchestration with time and gate handling is
  still required.

## Final launch preflight

6 contract/gate/report tests passed; train/val first samples, baseline CUDA loader and a real NVIDIA EGL scene frame passed. Full evidence: `launch_preflight.json`. No experiment training/closed-loop result yet.
