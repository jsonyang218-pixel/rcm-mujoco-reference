# Restricted OCT Perception Experiment

This repository now contains the restricted-perception implementation and
frozen data interfaces described in the revised experiment plan. The Stage 0
acceptance gate passes. Stage 1 has passed data construction, offline causal
screening, and the required paired closed-loop gate.

## Implemented

- Independent coarse RGB-D and fine OCT measurement models with per-episode
  registration bias, coarse AR(1) drift, distance-dependent noise, fine FOV
  and range limits, delay, and dropout.
- Continuous coarse/fine target-space blending with a `0.1 s` low-pass weight,
  plus a hard-switch control mode.
- `FINE_TRACKING`, `FINE_COAST`, `COARSE_FALLBACK`, and `PRECISION_HOLD` state
  transitions. Fine loss coasts for at most two frames; the safety path holds
  when coarse precision is insufficient and requests a bounded needle-axis
  retraction after `0.5 s`.
- Delayed fine measurements retain the measurement-time kinematic tip and are
  aligned to the current tip without using target truth.
- A frozen policy allowlist. Coarse/fine audit biases and simulator truth are
  not present in `RestrictedObservation.policy_features()`.
- A frozen target split with separate training, DAgger, development, and final
  test targets, plus a deterministic 20-target median-evaluation subset.
- Restricted sidecars for the current v7 training and validation datasets.
- One-factor image degradation tooling and four trained restricted-policy
  variants: full 224, 112, 112 plus Gaussian blur, and 112 plus blur plus
  center occlusion.
- Online restricted-policy evaluation, closed-loop smoke evaluation, paired
  target-cluster bootstrap, exact McNemar, and statistics utilities.

## Frozen configuration

The configuration is [configs/restricted_perception.json](configs/restricted_perception.json).
The externally visible camera rotation is a snapshot of the current v9
`eye_internal` camera and is covered by a unit test.

The experiment declares coarse switching at `2 mm` with a `+/-0.2 mm`
transition band, selected after scanning `1/2/3 mm`. The coarsest passing
precision level is `20 um`; `5 um` and `10 um` fail the deterministic
closed-loop success requirement.

## Stage 0 evidence

The machine-readable report is
`outputs/restricted_perception_stage0_20261106/stage0_report.json`.

- Hard-vs-continuous switching passes at all `1/2/3 mm` distances with the
  default `0.2 mm` half-width. Per-event P95 step reduction is approximately
  `90.6%-91.7%`, above the required `80%`.
- Fused near-target perception P95 is `4.00 um`.
- Deterministic closed-loop success is `99.5%` at `20 um`, `89%` at `10 um`,
  and `65%` at `5 um`.
- Blended P50 arrival/stop time is `7.475 s` versus `7.600 s` for hard
  switching, so the `0.5 s` non-inferiority rule passes.

The report's config SHA-256 is
`74C966EBE685F0AEAFAA4A2A12932CAF9EC2B4EC7FDF0BE176CF226A200F9FE5`.

## Target isolation

`configs/target_split_manifest.json` contains 254 unique training targets, 59
unique DAgger targets, 40 development targets, and 60 final-test targets.
Final targets were generated before any model result was inspected using seed
`20261215`.

The minimum distances are:

| Pair | Minimum distance |
| --- | ---: |
| final to final | 1.000167 mm |
| final to training | 1.001074 mm |
| final to DAgger | 1.104997 mm |
| final to development | 1.018658 mm |

The original strict posterior cap (`z >= 0.9`) admitted zero additional
targets at `1 mm` separation. The final set therefore uses the predeclared
extended posterior region `z >= 0.8`; this distribution change is recorded in
the manifest and was not selected from model results.

## Restricted sidecars

Both sidecars pass schema, row alignment, effective-bit, fallback-state, and
truth-omission checks.

| Split | Frames | Episodes | Verification |
| --- | ---: | ---: | --- |
| training | 195,578 | 608 | PASS |
| validation | 29,000 | 100 | PASS |

Each verified metadata and `verification.json` file is under the corresponding
dataset's `restricted_sidecar/` directory.

## Stage 1 status

The one-factor ladder is recorded in
`outputs/restricted_stage1_causal_ladder_20261106.json`.

Offline near-target action errors show that the restricted vector's causal
effect grows after degradation:

| Input | Zero-vector increase | Mismatch-vector increase |
| --- | ---: | ---: |
| 224 | 4.84% | 7.89% |
| 112 | 15.32% | 31.04% |
| 112 + blur 1.5 | 15.19% | 33.49% |
| 112 + blur 1.5 + center occlusion 25% | 17.73% | 38.12% |

The paired closed-loop run started each of the 40 development targets at a
`100 um` error and used a `200`-frame budget.

| Variant | Successes | Success rate | Final-error P95 |
| --- | ---: | ---: | ---: |
| restricted candidate | 37/40 | 92.5% | 22.97 um |
| zero restricted vector | 11/40 | 27.5% | 228.77 um |
| mismatched restricted vector | 12/40 | 30.0% | 294.51 um |
| v4 no-DAgger diffusion baseline | 13/40 | 32.5% | 110.00 um |

Zeroing or mismatching the restricted vector reduced success by 65 and 62.5
percentage points respectively. Both exact McNemar tests are significant, so
the image causal gate passes. Against the v4 no-DAgger baseline, the candidate
success-rate difference is 60 percentage points with a target-clustered
bootstrap 95% interval of `[47.83, 73.08]`; exact McNemar is `p=8.05e-7`.
Final-error P95 improves by `79.1%`. No variant had a `>30 mm` uncontrolled
episode, and all safety-limited counts are zero.

The machine-readable result is
`outputs/restricted_stage1_closed_loop_20261106/stage1_summary.json`.

## Stage 2 status

Stage 2 collected 120 new restricted DAgger episodes (2,160 frames) over 60
fresh training targets and two stratified starts per target. Four sequential
single-factor policies were trained with recovery-data replay:

| Variant | Successes | Final-error P95 |
| --- | ---: | ---: |
| restricted DAgger only | 38/40 | 20.00 um |
| add velocity | 40/40 | 11.99 um |
| add distance-scaled residual cap (`10 um` scale) | 40/40 | 13.06 um |
| add stop head | 40/40 | 15.51 um |

The velocity input improved over DAgger-only. After correcting the cap scale
to `10 um` and applying clipping only at inference, both later variants reached
40/40 without safety violations. Their final-error P95 values were still worse
than the velocity-only model, so those two features were rejected from the
selected policy. Stage 2 passes with the velocity-only variant selected.

### Corrected extrinsic audit

A later sign audit found that the original restricted configuration used the
legacy `Reference` camera pose, while closed-loop RGB-D/OCT measurement uses
`RCMSceneV9.eye_internal`. The two pose matrices differed by up to `1.0918`.
Stage 0 was rerun and passed after replacing the extrinsic. The training and
validation restricted sidecars were rebuilt under `restricted_sidecar_v2`,
Stage 1 was retrained, and all 120 Stage 2 DAgger episodes were recollected.

The corrected Stage 1 result remains a PASS:

| Variant | Successes | Final-error P95 |
| --- | ---: | ---: |
| normal restricted vector | 39/40 | 18.07 um |
| zero vector | 10/40 | 100.11 um |
| same-distance mismatch vector | 9/40 | 82.71 um |

The corrected Stage 2 ranking changed. DAgger-only reached 40/40 with final
error P95 `16.67 um` and was selected. Velocity reached 34/40, residual-cap
38/40, and residual-cap-plus-stop 38/40. The corrected Stage 2 report is
`outputs/restricted_stage2_acceptance_v5_20261106/summary.json`.

## Stage 3 status

Stage 3 uses the frozen protocol in
`configs/stage34_protocol.json`, seed `20270301`, all 40 development targets,
and a `100 um` start error.

| Variant | Successes | Final-error P95 |
| --- | ---: | ---: |
| selected restricted student | 40/40 | 16.67 um |
| guided no-DAgger action-chunk MLP | 39/40 | 17.01 um |
| old diffusion structure + DAgger, 7-median | 39/40 | 10.46 um |
| guided OCT regression | 40/40 | 11.69 um |
| raw diffusion, 7-median | 0/40 | 6520.55 um |

The geometric restricted-observation teacher was compared under one-factor
noise, image occlusion, OCT dropout, and delay. The student reached 40/40
under default, doubled noise, and 20% OCT dropout; 39/40 under delay;
occlusion was a failure case at 0/40. The geometry teacher reached 8/40,
11/40, 7/40, 3/40, and 11/40 respectively. The student had a target-clustered
bootstrap success-difference lower bound above zero in four scenarios, so the
learning-value gate passes.

The deterministic single-step head is a bounded residual on the selected
student, trained from 2,270 on-policy training-target states labelled by the
seven-sample median of the final diffusion teacher. It reached 40/40 on the
development targets versus 39/40 for the seven-sample teacher and passes the
predeclared five-percentage-point non-inferiority margin. No development or
final-test target was used to generate distillation labels.

Stage 3 PASS evidence is
`outputs/restricted_stage3_summary_20261106.json`. Latency is reported in
`outputs/restricted_stage3_latency_20261106.json`; on CUDA, the distilled
single-step head is `0.384 ms` P50 / `0.413 ms` P95 at batch 1, versus
`288.7 ms` P50 / `343.3 ms` P95 for seven diffusion samples at batch 1.

## Stage 4 final test

Stage 4 used all 60 frozen final-test targets, three seeds per target, and the
remaining budget of 1600 frames. Resource-only exceptions from concurrent
renderer memory exhaustion were rerun without changing the policy or protocol;
the final files contain no resource exceptions.

The final-test result does not support the intended cross-target precision
claim:

| Threshold | Distilled single-step | Seven-sample median subset |
| --- | ---: | ---: |
| 20 um | 20/180 (11.1%) | 18/60 (30.0%) |
| 10 um | 14/180 (7.8%) | 17/60 (28.3%) |
| 5 um | 7/180 (3.9%) | 6/60 (10.0%) |

On the hash-selected paired 60 episodes, the 20 um success-rate difference was
`-15.0` percentage points with target-cluster bootstrap 95% interval
`[-30.8, 0.0]`. At 10 um the difference was `-18.3` points with interval
`[-30.6, -5.6]`. The distilled head is much faster, but it is not non-inferior
to the seven-sample diffusion control on final targets.

The final test is therefore **COMPLETE / FAILED PERFORMANCE**, not a success.
It establishes an out-of-distribution generalization gap rather than a
deployment-ready result. The machine-readable report is
`outputs/restricted_stage4_summary_20261106.json`.

### Post-hoc failure analysis

The frozen final targets are strongly separated from development targets in
eye-surface location. Development targets span normalized posterior coordinate
`0.9006–0.9996` and polar angle `1.71–25.76 deg`; final targets span
`0.8001–0.8822` and `28.10–36.86 deg`. The two distributions have KS statistic
`1.0` for both coordinates. All targets are effectively on the ellipsoid wall,
so wall clearance is not a confounder.

Fine fixed-bias norms are not shifted: development P95 is `2.65 um`, final-test
P95 is `2.86 um`, and the final-test maximum is `3.31 um`. No episode has fine
bias above `20 um`. Stage 0 fused near-target perception P95 is `3.73 um`, so
the `20 um` threshold is not intrinsically impossible because of fine bias.

Of the 180 final-test episodes:

- 20 succeed at 20 um;
- 51 reach 20 um at least once but never satisfy the three-frame settled hold;
- 8 reach 20–30 um, 41 reach 30–50 um, and 45 reach 50–100 um;
- 6 never approach from the 100 um start;
- 9 fail at initialization because the B-axis encoder is outside the
  calibrated safe range.

The 51 hold failures have median minimum error `5.54 um`, median final error
`96.29 um`, and fine-channel availability in `99.52%` of frames. Coarse
fallback never occurs. The primary learned-policy problem is therefore
terminal stability rather than fine-channel availability or bias.

The full report is
`outputs/restricted_stage4_failure_analysis_20261107.json`; the narrative is
`STAGE4_FAILURE_ANALYSIS.md`.

## Commands

```powershell
$env:PYTHONPATH='.'
python -m unittest -v tests.test_restricted_perception tests.test_restricted_perception_experiments tests.test_restricted_sidecar tests.test_restricted_policy tests.test_target_split tests.test_causal_and_degradation tests.test_statistics
python run_restricted_perception_stage0.py --switch-episodes 100 --closed-loop-seeds-per-target 10
python build_target_split_manifest.py
python verify_restricted_sidecar.py --dataset <dataset.npz> --sidecar <restricted_sidecar.npz>
python train_restricted_policy.py --help
python evaluate_restricted_policy.py --help
```

## Scope limitations

All results are MuJoCo simulation results. Camera calibration, registration
bias, OCT FOV, noise, dropout, and delay are simulation assumptions. The project
does not establish real OCT accuracy, clinical precision, tissue contact, or
hardware safety.
