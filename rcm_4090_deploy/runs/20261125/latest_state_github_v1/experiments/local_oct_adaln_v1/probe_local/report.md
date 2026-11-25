# Step 4.0 zero-training near-target blindness probe

No policy weights, drivers, training scripts, or existing run directories were modified. No policy retraining or closed-loop simulation rerun was performed.

## Probe 1: exact inputs, normalization, and execution horizon

| Item | Confirmed value |
|---|---|
| Checkpoint | `/home/him1/mujoco_bc/rcm_4090_deploy/runs/20261125/local_oct_adaln_v1/checkpoints/arm_B_local_oct/step_004000.pth` |
| Policy / architecture | `dit_rectified_flow` / `state_conditioned_temporal_attention_v1` |
| Stored / policy image size | 224x224 / 224x224 RGB, no crop |
| Native render size | [1920, 1440] |
| RGB normalization | ImageNet mean/std; bilinear resize only |
| OCT input | 3D target-minus-tip vector in mm + validity; no OCT image |
| OCT symlog | `False` |
| `n_obs_steps` / `n_action_steps` | 2 / 16 |
| fast5 `exec_steps` | **1** |
| fast5 samples / flow steps | 1 / 4 |
| fast5 action aggregation | `single_sample` |
| Camera FOV / target distance | 45.0 deg / 40.447 mm |
| Retinal-plane scale | 23.269 um/px native, 149.586 um/px at 224 |

| Offset | Native pixels | 224-policy pixels | Below one 224 pixel? |
|---|---:|---:|---|
| 20um | 0.860 | 0.134 | True |
| 30um | 1.289 | 0.201 | True |

The visual-only signal is sub-pixel at 224x224, but the policy also receives the continuous OCT 3D vector. Therefore sub-pixel RGB alone does not establish input blindness.

| Channel vector | Mean | Std |
|---|---|---|
| `state_mean` | `[0.06349938362836838, 0.363604873418808, 12.981619834899902]` | `n/a` |
| `state_std` | `[6.787475109100342, 5.752523899078369, 6.081380844116211]` | `[6.787475109100342, 5.752523899078369, 6.081380844116211]` |
| `oct_mean` | `[0.7278906106948853, -0.026162389665842056, 6.579880237579346, 0.970044732093811]` | `n/a` |
| `oct_std` | `[1.216712474822998, 0.9585331082344055, 5.981095314025879, 0.17052701115608215]` | `[1.216712474822998, 0.9585331082344055, 5.981095314025879, 0.17052701115608215]` |
| `velocity_mean` | `[0.004034513141959906, -0.003496601479128003, 0.4751315414905548]` | `n/a` |
| `velocity_std` | `[0.6506950259208679, 0.4250059723854065, 0.24601227045059204]` | `[0.6506950259208679, 0.4250059723854065, 0.24601227045059204]` |
| `previous_action_mean` | `[0.00024240043421741575, -0.0001381530601065606, 0.033337727189064026]` | `n/a` |
| `previous_action_std` | `[0.045525968074798584, 0.029723668470978737, 0.01701800711452961]` | `[0.045525968074798584, 0.029723668470978737, 0.01701800711452961]` |
| `action_mean` | `[0.00018426710448693484, -7.851731788832694e-05, 0.03335476294159889]` | `n/a` |
| `action_std` | `[0.04544081911444664, 0.02966109849512577, 0.016713595017790794]` | `[0.04544081911444664, 0.02966109849512577, 0.016713595017790794]` |

| OCT offset | Normalized XYZ delta | Normalized norm |
|---|---|---|
| 20um | `[0.016437737274707823, 0.020865215638549522, 0.0033438691326485464]` | 0.02677 |
| 30um | `[0.024656605912061737, 0.03129782345782428, 0.005015803698972819]` | 0.04016 |

With `oct_symlog_enabled=False`, the OCT channel is ordinary `(value-oct_mean)/oct_std`; a 20-30 um XYZ offset corresponds to 0.02-0.03 mm divided by each OCT std channel. No explicit channel noise field is logged, so std is the available scale reference.

## Probe 2: 5-fold grouped linear readout

Source frames: `/home/him1/mujoco_bc/rcm_4090_deploy/data/homog_clean_replay_20261123_mixed_1600_600_12_hold_v2/data/dataset.npz`. Rows were filtered by `norm(truth_navigation_mm) < 60 um`, balanced across approach/recede, and folds were grouped by source episode.

| Band | Feature | Frames | Sign accuracy X/Y/Z | Mean cosine | Median cosine | P10 cosine |
|---|---|---:|---|---:|---:|---:|
| under_60 | F_vis | 317 | 0.483/0.590/0.568 | 0.089 | 0.136 | -0.719 |
| under_60 | F_cond | 317 | 0.530/0.609/0.555 | 0.164 | 0.276 | -0.630 |
| under_60 | F_oct | 317 | 1.000/0.997/0.965 | 0.999 | 1.000 | 0.999 |
| under_40 | F_vis | 250 | 0.504/0.544/0.480 | 0.046 | -0.010 | -0.770 |
| under_40 | F_cond | 250 | 0.556/0.516/0.508 | 0.132 | 0.204 | -0.734 |
| under_40 | F_oct | 250 | 1.000/0.996/0.984 | 0.999 | 1.000 | 0.999 |

Folds are grouped by source episode. `F_vis` is the full flattened ResNet token embedding; `F_cond` is the full flattened condition-token tensor plus global condition entering the diffusion head; `F_oct` is the deployable 3D OCT vector and serves as the reference upper bound.

## Probe 3: diffusion sampling variance

- Observations: 30 balanced under-60 um frames.
- Samples per observation: K=8.
- fast5 executes 1 step; action[0] and action[0:exec_steps] are identical.

| Axis | Sign agreement action0 | First-16 majority agreement | Obs. with sign flip action0 | std/|mean| median | std/|mean| p90 |
|---|---:|---:|---:|---:|---:|
| A | 0.667 | 0.659 | 0.967 | 2.667 | 9.622 |
| B | 0.688 | 0.666 | 0.967 | 2.535 | 9.546 |
| C | 0.812 | 0.746 | 0.600 | 1.206 | 6.111 |

## Receding-horizon check

- Status: `skipped_runtime_guard`.
- fast5 evaluator already uses exec_steps=1, so the requested reference (a) and replan-every-step (b) are identical. Condition (c), 2 targets x 1600 steps x K=4, cannot fit the remaining <=10 minute sub-budget based on the completed 1600-step fast5 runs.
- No full 1600-step simulation was launched.

## Binary verdict

**Selected Stage 4 change: A: Input Representation / Scaling**

F_cond fails the near-target direction criterion while raw OCT direction remains strong.

Read the measured per-axis K=8 values above; the original baseline conclusion is not reused for this checkpoint.
