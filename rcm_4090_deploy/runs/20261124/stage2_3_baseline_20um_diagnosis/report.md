# Stage 2 + Stage 3 pre-check: why baseline stays near 20-60 um

Read-only analysis of completed evaluation JSON. No training, simulation rollout, model loading, or result overwrite was performed.

## Exact JSON schema observed

- Top level: `['num_episodes', 'success_rate', 'avg_steps', 'avg_final_error_mm', 'median_final_error_mm', 'evaluation_contract', 'episodes']`
- Episode: `['success', 'steps', 'final_error_mm', 'max_error_mm', 'final_state_A_deg_B_deg_C_mm', 'final_velocity_deg_deg_mm_per_s', 'final_tip_world_mm', 'target_world_mm', 'motor_summary', 'motor_records', 'ground_truth_used_for_control', 'ground_truth_used_for_success', 'execution_backend', 'evaluation_contract_version', 'controller_variant', 'state_action_units', 'oct_navigation_used', 'safety_limited_steps', 'last_safety_error', 'early_stop_reason', 'terminal_success', 'policy_control_mode_counts', 'last_terminal_refinement', 'terminal_locked_target_world_mm', 'trajectory', 'episode_index', 'target_id', 'seed', 'target_robot_mm', 'nominal_target_robot_mm', 'target_shift_um', 'reachable_encoder_counts_ABC', 'target_manifest_used']`
- Trajectory: `['raw_actions', 'control_targets', 'states', 'actions', 'errors', 'velocities', 'low_speed', 'encoder_settled', 'oct_navigation_mm', 'oct_navigation_norm_mm', 'oct_navigation_valid', 'simulation_time_s', 'policy_inference_ms']`
- Motor record: `['driver', 'step_index', 'start_time_s', 'end_time_s', 'duration_s', 'integration_steps', 'tau_s', 'requested_task_rad_rad_m', 'commanded_task_rad_rad_m', 'requested_robot_ABCE', 'handles_ABCE', 'quantized_target_robot_ABCE', 'actual_before_robot_ABCE', 'actual_after_robot_ABCE', 'actual_before_task_rad_rad_m', 'actual_after_task_rad_rad_m', 'actual_velocity_rad_rad_m_per_s', 'tip_rcm_radius_before_mm', 'tip_rcm_radius_after_mm', 'tip_displacement_um', 'status', 'fault']`
- Control target: `['controller_variant', 'scale', 'accepted_delta_deg_deg_mm']`
- Terminal success: `['error_threshold_mm', 'required_consecutive_frames', 'consecutive_frames_at_exit', 'low_speed_at_exit', 'encoder_settled_at_exit', 'terminal_frame_pass', 'terminal_condition']`
- Available measurements: OCT target vector, OCT norm and validity, A/B/C state, velocity, accepted control target, motor execution before/after task state, requested/commanded target, tip displacement.
- The logged OCT vector is in `camera_optical` coordinates. The JSON does not retain the camera rotation, so only its norm can be compared directly with world-space true error. Per-axis camera components are reported separately and are not interpreted as world-space bias.
- Absent from logs: raw RGB/depth arrays, explicit predicted tip-world trajectory, explicit true-error vector, and explicit named pending-execution-residual field. The true vector direction is therefore derived statically from recorded A/B/C plus target pose, then scaled to the recorded ground-truth scalar error.

## Baseline per-target bottleneck table

| Target | t_min | min um | final um | OCT distance error @min | OCT-world component delta* @min | deadzone <1um near | deadzone <1um <60 | toward before | toward after | lag @min | classification |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E00 | 1454 | 41.95 | 50.81 | 0.02 | 83.04 | 0.297 | 0.284 | 0.588 | 0.490 | 0.0013 | Direction Reversal |
| E02 | 661 | 35.24 | 56.03 | 0.01 | 56.97 | 0.089 | 0.238 | 0.980 | 0.353 | 0.0002 | Direction Reversal |
| E06 | 1466 | 27.31 | 43.21 | 0.07 | 54.34 | 0.208 | 0.230 | 0.569 | 0.471 | 0.0040 | Direction Reversal |
| E08 | 1572 | 35.11 | 66.42 | 0.03 | 65.68 | 0.128 | 0.219 | 0.784 | 0.464 | 0.0039 | Direction Reversal |
| E10 | 1118 | 23.20 | 112.98 | 0.05 | 37.95 | 0.228 | 0.222 | 0.510 | 0.373 | 0.0005 | Direction Reversal |

Columns `toward before/after` are the fraction of non-zero raw policy commands whose world-space induced displacement has positive cosine with the ground-truth target vector. `lag @min` is command-to-actual task residual in mixed A-deg/B-deg/C-mm units and is magnitude only.
The `OCT-world component delta*` column is an explicitly non-gated cross-frame contrast included for auditability; it is not a perception error metric.

## Perception statistics

| Target | norm bias near mean/median/p90 um | norm bias <60 mean/median/p90 um | OCT camera XYZ @min um | A/B/C state residual @min |
|---|---|---|---|---|
| E00 | 0.03/0.03/0.05 | 0.02/0.02/0.05 | 62.09/-16.55/52.60* | 7.61/9.05/2.56 |
| E02 | 0.03/0.03/0.06 | 0.03/0.03/0.06 | 29.19/-13.99/46.87* | 6.93/13.50/2.27 |
| E06 | 0.03/0.04/0.07 | 0.04/0.04/0.06 | 49.08/-7.61/22.06* | 4.24/0.11/3.48 |
| E08 | 0.02/0.02/0.05 | 0.04/0.05/0.07 | 65.65/-2.01/0.22* | 3.78/18.09/2.04 |
| E10 | 0.02/0.03/0.07 | 0.02/0.02/0.07 | 27.76/-7.72/24.70* | 3.14/-4.60/3.76 |

The final camera XYZ column at `t_min` is shown only as the available camera-frame observation. Because camera rotation is absent, it is not reported as world-axis bias. A/B/C state residual is recorded separately and uses different physical units.

### Per-axis available-field distributions under 60 um

| Target | OCT camera X mean/median/p90 | Y | Z | A state residual mean/median/p90 | B | C |
|---|---|---|---|---|---|---|
| E00 | 40.89/41.76/51.34 | -5.23/-5.92/11.28 | 27.92/31.05/39.08 | 7.59/7.60/7.63 | 9.05/9.04/9.07 | 2.56/2.56/2.57 |
| E02 | 36.14/35.65/51.54 | -10.44/-11.15/17.04 | 29.64/30.90/45.33 | 6.88/6.89/6.93 | 13.45/13.45/13.47 | 2.28/2.28/2.29 |
| E06 | 35.18/33.98/49.08 | 0.77/0.84/9.84 | 24.01/21.56/44.96 | 4.23/4.23/4.26 | 0.11/0.11/0.13 | 3.50/3.49/3.51 |
| E08 | 26.28/25.50/38.87 | -16.32/-15.73/21.55 | 37.40/40.27/53.08 | 3.82/3.83/3.86 | 18.10/18.10/18.11 | 2.08/2.08/2.09 |
| E10 | 31.53/30.47/47.44 | 3.61/3.00/11.87 | 17.74/15.85/40.77 | 3.10/3.10/3.14 | -4.61/-4.61/4.63 | 3.77/3.77/3.79 |

These are component distributions in their recorded frames, not world-space measurement bias.

## Direction statistics

| Target | policy cosine median before/after/late | policy toward before/after/late | policy sign X/Y/Z before | policy sign X/Y/Z after | policy sign X/Y/Z late | actual sign X/Y/Z after |
|---|---|---|---|---|---|---|
| E00 | 0.168/-0.066/-0.022 | 0.588/0.490/0.490 | 0.45/0.55/0.65 | 0.47/0.41/0.55 | 0.47/0.51/0.51 | 0.45/0.45/0.53 |
| E02 | 0.995/-0.393/-0.000 | 0.980/0.353/0.498 | 0.94/0.98/0.98 | 0.20/0.69/0.49 | 0.51/0.51/0.47 | 0.25/0.47/0.43 |
| E06 | 0.109/-0.089/-0.034 | 0.569/0.471/0.480 | 0.51/0.55/0.63 | 0.49/0.53/0.43 | 0.50/0.52/0.51 | 0.45/0.37/0.45 |
| E08 | 0.636/-0.356/0.035 | 0.784/0.464/0.522 | 0.65/0.63/0.82 | 0.46/0.46/0.14 | 0.52/0.49/0.49 | 0.14/0.50/0.11 |
| E10 | 0.004/-0.326/-0.075 | 0.510/0.373/0.472 | 0.47/0.49/0.65 | 0.33/0.59/0.43 | 0.46/0.49/0.49 | 0.35/0.53/0.41 |

## Near-target command and displacement distribution

| Target | policy command median A/B/C | policy command p90 |abs| A/B/C | actual displacement median X/Y/Z um | actual displacement p90 |abs| X/Y/Z um |
|---|---|---|---|---|
| E00 | 0.00083/-0.00026/0.00026 | 0.00924/0.00512/0.00239 | -0.048/0.013/-0.163 | 4.004/2.159/1.738 |
| E02 | 0.00278/0.00116/0.00342 | 0.01421/0.00497/0.04271 | -0.366/1.916/-2.045 | 8.098/13.198/26.814 |
| E06 | 0.00083/0.00013/-0.00005 | 0.00885/0.00576/0.00375 | 0.000/0.093/-0.050 | 4.311/2.390/2.353 |
| E08 | -0.00086/0.00017/0.00037 | 0.00966/0.00539/0.00423 | 0.109/0.468/-0.398 | 3.860/2.163/3.014 |
| E10 | 0.00133/0.00043/0.00010 | 0.00848/0.00660/0.00276 | -0.000/0.071/0.018 | 4.329/2.276/2.457 |

## Stage 3 matched-error comparison: Baseline 20-50 um

| State | Frames | policy toward | policy cosine median | command norm median | actual toward | error median um | lag median |
|---|---:|---:|---:|---:|---:|---:|---:|
| Approaching | 500 | 0.760 | 0.474 | 0.00625 | 0.864 | 43.24 | 0.0019 |
| Receding | 529 | 0.178 | -0.497 | 0.00664 | 0.113 | 44.00 | 0.0020 |
| High lag/pending | 306 | 0.474 | -0.077 | 0.00821 | 0.533 | 43.90 | 0.0018 |

Pooled within each target's 20-50 um frames; high state is top quartile of max(command-to-actual lag, pending target residual) inside that target's matched band. Groups can overlap.
Approaching frames are 0.760 toward-target versus 0.178 for receding frames. High execution-lag/pending frames remain mixed (0.474 toward), but the absolute logged lag is small. This supports a direction/behavior failure more strongly than a large actuator lag.

## Optional Fallback F raw check

| Target | t_min | min um | final um | OCT distance error @min | OCT-world component delta* @min | deadzone <1um near | deadzone <1um <60 | toward before | toward after | lag @min | classification |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E00 | 1594 | 45.65 | 56.81 | 0.02 | 91.20 | 0.339 | 0.291 | 0.529 | 0.000 | 0.0028 | Direction Reversal |
| E02 | 1506 | 43.81 | 55.37 | 0.04 | 87.52 | 0.228 | 0.291 | 0.608 | 0.412 | 0.0003 | Direction Reversal |
| E06 | 1182 | 26.52 | 36.51 | 0.04 | 50.73 | 0.188 | 0.212 | 0.647 | 0.451 | 0.0036 | Direction Reversal |
| E08 | 1575 | 33.27 | 57.55 | 0.05 | 63.36 | 0.160 | 0.206 | 0.765 | 0.240 | 0.0007 | Direction Reversal |
| E10 | 1153 | 16.99 | 63.66 | 0.03 | 33.45 | 0.188 | 0.236 | 0.627 | 0.431 | 0.0010 | Direction Reversal |

### Fallback F matched-error comparison

| State | Frames | policy toward | policy cosine median | command norm median | actual toward | error median um | lag median |
|---|---:|---:|---:|---:|---:|---:|---:|
| Approaching | 371 | 0.784 | 0.437 | 0.00624 | 0.892 | 43.98 | 0.0019 |
| Receding | 370 | 0.170 | -0.549 | 0.00641 | 0.132 | 43.85 | 0.0020 |
| High lag/pending | 219 | 0.521 | 0.060 | 0.00784 | 0.575 | 45.65 | 0.0015 |

### Fallback F E10 <=20 um hold

- Exact maximum consecutive cycles: **4**
- Maximum error within the longest hold segment: **19.282 um**
- Hold start/end: `1152` / `1155`
- Interpretation: E10 entered the <=20 um band for a short 4-cycle segment; it did not sustain a 20 um hold.

## Verdict

1. Dominant factor: **Direction Reversal**; perception-distance flags are 0/5, deadzone<1um is 0.190, and mean after-t_min toward fraction is 0.430.
2. Execution lag at t_min averages 0.0020 mixed A-deg/B-deg/C-mm units, so actuator lag alone is too small to explain the 23-42 um floor.
3. Fix target direction/policy behavior before stage 4; Stage 4 should not add execution-target residual as the first response to this static evidence.
