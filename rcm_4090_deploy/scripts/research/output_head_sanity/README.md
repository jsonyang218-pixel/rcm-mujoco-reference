# Plan 8 deterministic output-head sanity check

Run only through the bounded workflow, from this isolated deployment checkout.
Use the physical deployment `.venv`, clear AppImage Python/library variables,
and set CUDA_VISIBLE_DEVICES=0, MUJOCO_GL=egl, TORCH_DISABLE_NATIVE_JIT=1.

The input card is `/home/him1/mujoco_bc/规划8.txt`. Runtime outputs are fixed to
`/home/him1/mujoco_bc/rcm_4090_deploy/runs/20261125/output_head_sanity_v1/`.
Code branches from `9fef16d`; historical code/data/checkpoints remain intact.

The baseline is `baseline_task2_step4000` with the exact C-only overlay and 8%
replacement correction sampler. Each new head has 315696 parameters, batch16,
AdamW lr1e-4 (the general learning rate in baseline config), decay1e-4,
4000 updates, matching initial seed and data draw sequence. Encoder is frozen
in eval mode. Pure F_geom_cond is cached; image-free extraction is checked
against the actual model's full condition. No RGB fusion exists in this global
condition. Keep the original RGB/state observer and controller during rollout.

R2 uses the specified invertible action symlog and 0.1 cosine direction loss.
The latter compares physical actions divided by the fixed per-axis scales,
on each nonzero future action whose independent training error is <80µm.
Truth is never passed into learned-policy inference.

True training error is reconstructed from stored encoder state and explicitly
resolved targets. Source offsets and terminal audits take priority over ambiguous
ancestry indices. Thirty-eight legacy episodes require their archived raw
control report: converted and raw state arrays must match exactly. Vector FK is
checked against scalar FK at 1e-10mm and MuJoCo site at 1e-6mm (the measured
preflight maximum site difference is 0.000711µm). Report threshold-border counts.
Baseline dataset E40 and the archived v5 replay E40 are distinct; audit both.

Paired replay reuses the exact archived 22/26-cycle prefix, RGB hashes,
encoder history and target. It reruns teacher and the true baseline, and reports
the archived v5 A4000 separately. It injects a new head only after the prefix.
The conditional fast5 gate requires both paired finals below both baselines.
If >=15min remain, run both qualifying arms together (10 target workers);
with 5–15min, run the better arm (5 workers); otherwise preserve a time skip.
Pool limit is12. Full1600 raw trajectories remain local; no dev20 is authorized.

A 1960-second computation alarm reserves140s for reporting and publishing within
35min. Failure/interruption records partial evidence, stops dependent work and
publishes lightweight results. There are no experiment retries. Weights and raw
trajectories are never staged. Publication is on the new experiment branch only.

Preparation repairs and failed preflight checks are retained and indexed in
TEST_LOG.md; they are not training runs. `preflight` permits only the dedicated
preflight_checks directory to preexist; `run` refuses an existing workflow.log.
Launch exactly once in a new GNOME Terminal; do not poll after launching.

Commands:

```bash
python -E -B -u -m scripts.research.output_head_sanity.workflow preflight --output-dir /home/him1/mujoco_bc/rcm_4090_deploy/runs/20261125/output_head_sanity_v1
python -E -B -u -m scripts.research.output_head_sanity.workflow run --output-dir /home/him1/mujoco_bc/rcm_4090_deploy/runs/20261125/output_head_sanity_v1
```

Report workflow completion, final/min20µm, sustained200 cycles, strict0.35µm
success and temporal versus repeated-observation flips separately. All results
concern this software/MuJoCo simulation; no hardware/clinical claim is supported.
