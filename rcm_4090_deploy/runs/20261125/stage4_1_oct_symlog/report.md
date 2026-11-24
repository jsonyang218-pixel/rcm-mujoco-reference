# Stage 4.1 controlled OCT-symlog experiment

Status: `in_progress`

Single changed training variable: `oct_symlog=False -> True`. Dataset, initialization, seed, architecture, replay ratio, optimizer, driver, control period, and 4,000-step budget were preserved.

## Train/eval parity

- Status: `passed`.
- 20 um normalized OCT XYZ delta: `[0.428520530462265, 0.44339120388031006, 0.3500848710536957]`.
- Training and evaluation call the same `normalize_oct_symlog` method.

## Probe 2/3 before vs after

| Metric | Baseline | OCT symlog |
|---|---:|---:|
| F_cond cosine <60 | 0.1468 | 0.1619 |
| F_cond sign X <60 | 0.5237 | 0.5268 |
| F_cond sign Y <60 | 0.5994 | 0.6404 |
| K=8 sign flip A | 0.9667 | 0.9667 |
| K=8 sign flip B | 0.9667 | 0.9667 |

## fast5 1600

| Model/Target | min um | final um | entered 20 | max consecutive <=20 (cycles/s) | hold max um | late A flip | late B flip | drift um | 200-cycle hold |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| Baseline/E00 | 41.954 | 50.906 | False | 0/0.00 | None | 0.372 | 0.441 | 12.876 | False |
| Baseline/E02 | 35.244 | 58.895 | False | 0/0.00 | None | 0.447 | 0.424 | 0.962 | False |
| Baseline/E06 | 27.311 | 43.136 | False | 0/0.00 | None | 0.466 | 0.424 | 29.273 | False |
| Baseline/E08 | 35.114 | 63.557 | False | 0/0.00 | None | 0.432 | 0.417 | 7.342 | False |
| Baseline/E10 | 23.202 | 111.354 | False | 0/0.00 | None | 0.417 | 0.392 | 62.612 | False |
| OCT symlog/E00 | 31.593 | 75.755 | False | 0/0.00 | None | 0.401 | 0.454 | 22.796 | False |
| OCT symlog/E02 | 32.100 | 85.744 | False | 0/0.00 | None | 0.451 | 0.434 | 1.927 | False |
| OCT symlog/E06 | 35.065 | 84.865 | False | 0/0.00 | None | 0.446 | 0.457 | 45.987 | False |
| OCT symlog/E08 | 34.492 | 76.784 | False | 0/0.00 | None | 0.436 | 0.427 | 16.947 | False |
| OCT symlog/E10 | 19.984 | 142.970 | True | 1/0.05 | 19.98362740626262 | 0.436 | 0.409 | 91.348 | False |

- Baseline aggregate: `{'completed_targets': 5, 'final_mean_um': 65.56928808053874, 'min_mean_um': 32.56482497246049, 'entered_20_um_count': 0, 'strict_hold_200_count': 0, 'late_ab_sign_flip_rate_mean': {'A': 0.42704507512520873, 'B': 0.4196994991652755, 'C': 0.2734557595993322}}`
- OCT-symlog aggregate: `{'completed_targets': 5, 'final_mean_um': 93.22356138945617, 'min_mean_um': 30.646513110169376, 'entered_20_um_count': 1, 'strict_hold_200_count': 0, 'late_ab_sign_flip_rate_mean': {'A': 0.43372287145242067, 'B': 0.43639398998330553, 'C': 0.28380634390651077}}`

## K=4 mean inference-only check

`{'status': 'completed', 'targets': {'E06': {'episode_index': 6, 'min_error_um': 35.064502150079065, 'final_error_um': 84.86455812214449, 'entered_20_um': False, 'max_consecutive_le_20_cycles': 0, 'max_consecutive_le_20_seconds': 0.0, 'hold_segment_max_error_um': None, 'hold_start_step': None, 'hold_end_step': None, 'late_ab_sign_flip_rate': {'A': 0.44574290484140233, 'B': 0.4574290484140234, 'C': 0.2704507512520868}, 'absolute_drift_1000_to_1600_um': 45.9872601081076, 'strict_hold_200_cycles': False, 'raw_result': '/home/him1/mujoco_bc/rcm_4090_deploy/runs/20261125/stage4_1_oct_symlog/eval_k4_mean/shards/tasks/eval_E06/raw/evaluation_results.json'}, 'E10': {'episode_index': 10, 'min_error_um': 19.98362740626262, 'final_error_um': 142.97002163275488, 'entered_20_um': True, 'max_consecutive_le_20_cycles': 1, 'max_consecutive_le_20_seconds': 0.05, 'hold_segment_max_error_um': 19.98362740626262, 'hold_start_step': 839, 'hold_end_step': 839, 'late_ab_sign_flip_rate': {'A': 0.4357262103505843, 'B': 0.4090150250417362, 'C': 0.3121869782971619}, 'absolute_drift_1000_to_1600_um': 91.3484132873603, 'strict_hold_200_cycles': False, 'raw_result': '/home/him1/mujoco_bc/rcm_4090_deploy/runs/20261125/stage4_1_oct_symlog/eval_k4_mean/shards/tasks/eval_E10/raw/evaluation_results.json'}}}`

## dev20

`{'status': 'skipped_gate_or_time', 'gate_passed': False, 'remaining_minutes': 123.10277178287507}`

## Verdict

OCT symlog did not cross the conditional fast5 gate; no dev20 run was started.
Targets now enter <=20 um, but sustained 200-cycle holds remain the decision point for Stage 5 terminal hold/braking data.
