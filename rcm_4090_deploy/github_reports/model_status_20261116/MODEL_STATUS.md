# RCM MuJoCo Model Status and Evaluation Summary

Updated: 2026-11-16  
Scope: MuJoCo simulation only. These results are not evidence of physical OCT,
hardware accuracy, or clinical safety.

## Current conclusion

The best five-class milestone model is **U-Net Flow + covariate-shift DAgger
round 2**. It improves the previous Flow baseline substantially, but still
reaches `0/20` development targets at the strict `0.1 um` gate. The best
closed-loop median among the strict Flow experiments is `448.589 um`, still
far above the final target.

The current DiT + Rectified Flow line is a separate architecture experiment.
DAgger round 1 reduced the pure-DiT final-error mean from `4080.069 um` to
`2334.450 um` and removed the 15-16 mm failure tail. It remains worse than the
U-Net Flow + DAgger round-2 median and has not passed strict closure.

## Evaluation threshold warning

Older v7-v12 Diffusion Policy results used a `20 um` settled gate. Later Flow,
residual, and DiT results use `0.1 um` or `0.35 um`. Absolute endpoint error can
be compared, but success counts under different gates cannot be treated as
equivalent.

## Model registry

| Stage | Model / data | Completion | Gate | Success | Final mean / median / max (um) | Main conclusion |
|---|---|---|---:|---:|---:|---|
| v7 | Guided Diffusion + DAgger, cached features | Trained and evaluated | 20 um | 20/20 repeat | 9.840 / 10.160 / 11.298 | First stable simulated near-target baseline |
| v8 | Full-unfreeze Diffusion | Trained and evaluated | 20 um | 19/20 | 11.848 / 11.117 / 26.791 | Lower offline MAE did not replace v7 |
| v9 | Terminal DAgger Diffusion | Trained and evaluated | 20 um | 19/20 | 11.159 / 10.865 / 23.146 | Reduced failure tail, not median precision |
| v10 | Four-frame temporal Diffusion | Trained and evaluated | 20 um | 19/20 | 12.464 / 10.418 / 45.648 | Temporal context did not improve closure |
| v11 H4 | Four-step action horizon | Trained and evaluated | 20 um | 19/20 | 11.459 / 10.807 / 27.691 | Better tail than H8, did not beat v9 |
| v12 OCT | OCT-aware Diffusion | Trained and evaluated | 20 um | 0/20 | 22495 / 22815 / 23697 | Wrong-direction motion and A/B saturation |
| v12 OCT DAgger1 | OCT Diffusion + strict DAgger | Trained and evaluated | 20 um | 0/20 | 16097 / 18034 / 23974 | DAgger helped, ordinary Diffusion remained inadequate |
| Flow | Temporal U-Net + 4-step Flow Matching | Trained and evaluated | 0.1 um | 0/20 | 1731.480 / 1010.959 / 15219.883 | Major improvement, one outlier remained |
| Flow DAgger2 | Covariate-shift DAgger round 2 | Trained and evaluated | 0.1 um | 0/20 | 986.731 / 448.589 / 5065.026 | Best Flow checkpoint; near-target plateau remains |
| Flow + residual | OCT proportional + bounded residual | Trained and evaluated | 0.1 um | 0/20 | 4800.836 / 1606.998 / 14571.062 | Repeated terminal solving amplified failures |
| Flow + reachable | Frozen reachable target + terminal lock | Trained and evaluated | 0.35 um | 0/20 | 4890.596 / 2609.548 / 14399.986 | Reachability was not the dominant blocker |
| DiT/RF | Pure DiT + Rectified Flow | Trained and evaluated | 0.35 um | 0/20 | 4080.069 / 2230.331 / 16439.474 | Fast inference, poor closed-loop stability |
| DiT/RF DAgger1 | DiT/RF + online DAgger round 1 | Trained and evaluated | 0.35 um | 0/20 | 2334.450 / 1546.263 / 4406.408 | Strong tail improvement, terminal hold still fails |
| DiT/RF DAgger2 | DiT/RF + near-target DAgger round 2 | Collection and training complete; closure pending | 0.35 um | pending | pending | Best val loss 0.158247; only 1/80 expert episode failed |

Raw metric definitions for the newer strict evaluations are under `evidence/`.
The full chronological audit is in `../TEST_LOG.md`.

## Dataset milestones

| Dataset | Episodes | Frames | Expert quality | Role |
|---|---:|---:|---|---|
| `lerobot_rcm_train_guided_dagger1_full_224` | 608 | 187440 | Earlier guided teacher data | Base v12 training set |
| `lerobot_v12_oct_dagger1_fixed_224` | 646 | 200943 | Strict filtered DAgger round 1 | First strict-zero OCT DAgger base |
| `lerobot_v12_oct_dagger2_covshift_verified_224` | 701 | 209045 | 701/701 strict | Flow DAgger round-2 base |
| `lerobot_v12_oct_dit_rf_dagger1_round1_verified_224` | 760 | 218433 | 760/760 strict | DiT/RF DAgger round-1 base |
| `lerobot_v12_oct_dit_rf_dagger2_near_target_verified_224` | 839 | 234316 | 839/839 strict | DiT/RF DAgger round-2 base |

The near-target DiT round 2 requested 80 episodes. It produced 79 strict
successes and one `LOW_QUALITY_VISUAL_GEOMETRY` failure at `90.637375 um`.
The failed episode is preserved as evidence and excluded from training.

## Checkpoint status

| Model | Checkpoint | SHA-256 |
|---|---|---|
| v8 | `diffusion_policy_v8_full_unfreeze_b96_lr1e4_20261112/best_model.pth` | `e93337c065ce1692a7e500530d8b34f341b7345ea4878893cd7070b63508b3ff` |
| v9 | `diffusion_policy_v9_terminal_dagger_20261113/best_model.pth` | `fc6d35240f9543171929c2ad4ad5ab1ded0155e6f05f515954b89e91a493f0dd` |
| v10 | `diffusion_policy_v10_temporal_20261113/best_model.pth` | `f451aaadfbbb7a95efac6214a9a16c69243f9f732960c550d192e4eaf9c7b5c4` |
| v11 H4 | `diffusion_policy_v11_h4_20261113/best_model.pth` | `4223375c5f20546b700022f378fc59061990c81601ebb880a21c0ea4c1dab4b7` |
| v12 OCT | `diffusion_policy_v12_oct_20261114_retry1/best_model.pth` | `146555bd69b91c50447c55b67426564e7d8a11537b5ff03f4aedbba976406bb9` |
| v12 OCT DAgger1 | `diffusion_policy_v12_oct_dagger1_fixed_20261115/best_model.pth` | `966fa042d80b692f7117b3061b9e8305683f117706ebbb41afb2a3598fdd5175` |
| Flow | `flow_temporal_unet_v12_zeroerror_20261115_retry3/best_model.pth` | `85c38dfbffaa2c15b3e677f4c419940f014a66ec92ae1a250ae0837eeeb7a87b` |
| Flow DAgger2 | `flow_temporal_unet_v12_oct_dagger2_covshift_20261115/best_model.pth` | `675a9360d9f971d9d962ceae647c09d708cf74f408dcf3d1e6d5fb7d9dce5fea` |
| DiT/RF | `dit_rectified_flow_v12_oct_dagger2_20261116/best_model.pth` | `4533a7f428f93011a5eb485d1a38f498641c1e3211f17cd21be7ff140650ff81` |
| DiT/RF DAgger1 | `dit_rectified_flow_v12_oct_dit_rf_dagger1_20261116/best_model.pth` | `3ad95651e66c4cb62adbe65ce765fd4408f59748a1619919d3c664319dd85d07` |
| DiT/RF DAgger2 | `dit_rectified_flow_v12_oct_dit_rf_dagger2_near_target_20261116/best_model.pth` | `686b78ae7744884ce34369e545f760253251ac0ccfa3c30691528cadf3f15424` |

Checkpoint binaries are intentionally excluded from Git. They can be published
as GitHub Release assets, Git LFS objects, or an external model registry.

## Failure interpretation

The strict Flow and DiT evaluations converge from roughly 20 mm toward
`40-500 um`, then lose progress near the target. In the DiT DAgger round-1
evaluation:

- All 20 episodes reached a per-episode minimum below `350 um`.
- Only one episode reached below `100 um`.
- 15/20 reached their minimum around steps `655-730`.
- Median action norm fell from about `0.043` before step 650 to `0.015`
  after step 750.
- The fraction of steps reducing error fell from nearly 100% before step 650
  to 20-32% afterward.
- Estimated OCT navigation error differed from simulated truth by a mean of
  only `0.97 um`, so target perception is not the primary blocker.

This is a terminal-control and action-distribution problem, not a missing OCT
target problem.

## Recommended next step

Do not claim zero-error completion from expert-label zero error alone. The
learned policy must pass the independent development closed loop. The next
architectural candidate should combine:

1. A reliable coarse approach policy.
2. OCT/Jacobian proportional base correction near the target.
3. A bounded learned residual with temporal smoothing.
4. A debounced terminal lock derived from online-estimated target state.
5. Closed-loop acceptance using an independent target split.

