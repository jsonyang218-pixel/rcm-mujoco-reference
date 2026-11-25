# Local OCT AdaLN v1

Workflow: completed

Original OCT is normalized linearly with joint state, velocity and previous_action. Geometry enters both per-observation geometry tokens (cross-attention alongside vision) and a separate global MLP (AdaLN and temporal attention bias). Vision does not enter the global MLP. The previously quoted 37,632 is flattened visual scalar dimensionality, not a token count; probe feature_dimensions records the measured dimensions. A poor full F_cond ridge readout alone cannot establish visual drowning. The local 7D signed OCT MLP residual enters AdaLN and final modulation only; geometry tokens and temporal-bias conditioning retain the original global vector. stage4 symlog adds residuals to the original linear geometry/global channels.

Historical baseline is a reference, not a provenance-matched reusable evaluation. Same targets/seeds/protocol are recorded in preflight.json; full evidence is in metrics.json.

Stages: {'step0_existing_hold': {'status': 'completed', 'started_at': '2026-11-25T18:04:06+08:00', 'ended_at': '2026-11-25T18:04:07+08:00'}, 'step0_probe_baseline': {'status': 'completed', 'started_at': '2026-11-25T18:04:07+08:00', 'ended_at': '2026-11-25T18:04:12+08:00'}, 'step0_probe_stage4_symlog': {'status': 'completed', 'started_at': '2026-11-25T18:04:12+08:00', 'ended_at': '2026-11-25T18:04:17+08:00'}, 'two_batch_smoke': {'status': 'completed', 'started_at': '2026-11-25T18:04:17+08:00', 'ended_at': '2026-11-25T18:04:24+08:00'}, 'matched_train4000': {'status': 'completed', 'started_at': '2026-11-25T18:04:24+08:00', 'ended_at': '2026-11-25T18:07:09+08:00'}, 'step1_probe_local': {'status': 'completed', 'started_at': '2026-11-25T18:07:09+08:00', 'ended_at': '2026-11-25T18:07:13+08:00'}, 'step1_fast5': {'status': 'completed', 'started_at': '2026-11-25T18:07:13+08:00', 'ended_at': '2026-11-25T18:20:59+08:00'}, 'step2_k4_mean': {'status': 'completed', 'started_at': '2026-11-25T18:21:01+08:00', 'ended_at': '2026-11-25T18:34:47+08:00'}, 'step3_dev20': {'status': 'skipped_gate_or_time', 'passed': False, 'remaining_minutes': 119.30845147371292, 'rule': 'complete fast5 AND ((final<58 AND min<25) OR entered20>=2); remaining>=40'}}

Failure/partial reason: none

## Probe 2 and 3

| Model | Band | Feature | cosine | sign XYZ |
|---|---|---|---:|---|
| baseline | under_60 | F_cond | 0.1468190777385545 | {'X': 0.5236593059936908, 'Y': 0.5993690851735016, 'Z': 0.555205047318612} |
| baseline | under_60 | F_geom_cond | 0.7291478700327403 | {'X': 0.9116719242902208, 'Y': 0.8359621451104101, 'Z': 0.7066246056782335} |
| baseline | under_40 | F_cond | 0.1643359136404177 | {'X': 0.564, 'Y': 0.572, 'Z': 0.512} |
| baseline | under_40 | F_geom_cond | 0.6490567096679382 | {'X': 0.888, 'Y': 0.828, 'Z': 0.624} |

baseline K=8: `{'A': {'mean_sign_agreement_action0': 0.7083333333333334, 'mean_first16_majority_agreement': 0.6575520833333334, 'fraction_observations_with_sign_flip_action0': 0.9666666666666667, 'std_over_abs_mean_action0': {'count': 30, 'median': 1.9370088577270508, 'p90': 8.165264368057262, 'mean': 5.477091666062673}}, 'B': {'mean_sign_agreement_action0': 0.6875, 'mean_first16_majority_agreement': 0.665625, 'fraction_observations_with_sign_flip_action0': 0.9666666666666667, 'std_over_abs_mean_action0': {'count': 30, 'median': 2.25675630569458, 'p90': 15.578431797027601, 'mean': 12.391594589749973}}, 'C': {'mean_sign_agreement_action0': 0.8041666666666667, 'mean_first16_majority_agreement': 0.7059895833333333, 'fraction_observations_with_sign_flip_action0': 0.6666666666666666, 'std_over_abs_mean_action0': {'count': 30, 'median': 1.02242711186409, 'p90': 5.682482528686524, 'mean': 4.335493611792724}}}`

| stage4_symlog | under_60 | F_cond | 0.16190466334394582 | {'X': 0.526813880126183, 'Y': 0.6403785488958991, 'Z': 0.5488958990536278} |
| stage4_symlog | under_60 | F_geom_cond | 0.7724480796295856 | {'X': 0.8990536277602523, 'Y': 0.8580441640378549, 'Z': 0.7381703470031545} |
| stage4_symlog | under_40 | F_cond | 0.1116495549013548 | {'X': 0.572, 'Y': 0.54, 'Z': 0.532} |
| stage4_symlog | under_40 | F_geom_cond | 0.7050023906422545 | {'X': 0.908, 'Y': 0.824, 'Z': 0.644} |

stage4_symlog K=8: `{'A': {'mean_sign_agreement_action0': 0.6666666666666666, 'mean_first16_majority_agreement': 0.6578125, 'fraction_observations_with_sign_flip_action0': 0.9666666666666667, 'std_over_abs_mean_action0': {'count': 30, 'median': 2.678001284599304, 'p90': 9.198804473876955, 'mean': 4.0012403706709545}}, 'B': {'mean_sign_agreement_action0': 0.6916666666666667, 'mean_first16_majority_agreement': 0.6666666666666666, 'fraction_observations_with_sign_flip_action0': 0.9666666666666667, 'std_over_abs_mean_action0': {'count': 30, 'median': 2.4888049364089966, 'p90': 10.44534673690796, 'mean': 4.924605520566304}}, 'C': {'mean_sign_agreement_action0': 0.8125, 'mean_first16_majority_agreement': 0.7427083333333333, 'fraction_observations_with_sign_flip_action0': 0.6, 'std_over_abs_mean_action0': {'count': 30, 'median': 1.2244434356689453, 'p90': 6.080082321167011, 'mean': 4.2565890610218045}}}`

| local_oct_adaln | under_60 | F_cond | 0.16384247479375716 | {'X': 0.5299684542586751, 'Y': 0.6088328075709779, 'Z': 0.555205047318612} |
| local_oct_adaln | under_60 | F_geom_cond | 0.7452860552270306 | {'X': 0.9211356466876972, 'Y': 0.8391167192429022, 'Z': 0.7066246056782335} |
| local_oct_adaln | under_40 | F_cond | 0.13168345955816976 | {'X': 0.556, 'Y': 0.516, 'Z': 0.508} |
| local_oct_adaln | under_40 | F_geom_cond | 0.6645249619754627 | {'X': 0.896, 'Y': 0.828, 'Z': 0.628} |

local_oct_adaln K=8: `{'A': {'mean_sign_agreement_action0': 0.6666666666666666, 'mean_first16_majority_agreement': 0.659375, 'fraction_observations_with_sign_flip_action0': 0.9666666666666667, 'std_over_abs_mean_action0': {'count': 30, 'median': 2.667089343070984, 'p90': 9.622337150573731, 'mean': 4.4223720073699955}}, 'B': {'mean_sign_agreement_action0': 0.6875, 'mean_first16_majority_agreement': 0.66640625, 'fraction_observations_with_sign_flip_action0': 0.9666666666666667, 'std_over_abs_mean_action0': {'count': 30, 'median': 2.534727454185486, 'p90': 9.546367931365968, 'mean': 4.738502417008082}}, 'C': {'mean_sign_agreement_action0': 0.8125, 'mean_first16_majority_agreement': 0.74609375, 'fraction_observations_with_sign_flip_action0': 0.6, 'std_over_abs_mean_action0': {'count': 30, 'median': 1.2058624029159546, 'p90': 6.11069674491884, 'mean': 3.235722216963768}}}`


## Closed-loop metrics

| Model | N | final mean um | min mean um | entered20 | sustained200 | strict |
|---|---:|---:|---:|---:|---:|---:|
| local4000_k1 | 5 | 98.34491249456755 | 33.39202951254059 | 0 | 0 | 0 |
| historical_baseline_fast5 | 5 | 65.56928808053874 | 32.56482497246049 | 0 | 0 | None |
| historical_stage4_fast5 | 5 | 93.22356138945617 | 30.646513110169376 | 1 | 0 | None |
| selected_k4_mean | 5 | 99.4121540538059 | 62.92832523273213 | 0 | 0 | 0 |

## Per-target hold and direction metrics

| Model/target | status | min um | final um | entries20 | max cycles/s | hold200 | last200 max um | speed overall/late um/s | late ABC flip | toward<60 |
|---|---|---:|---:|---:|---|---|---:|---|---|---|
| local4000_k1/E00 | completed | 32.7480163198273 | 95.07415008575508 | 0 | 0/0.0 | False | 107.2289200428512 | 274.3016288670699/44.264192282259465 | {'A': 0.3823038397328882, 'B': 0.4607679465776294, 'C': 0.3055091819699499} | 0.5062388591800356 |
| local4000_k1/E02 | completed | 38.214635275830844 | 99.20962061574494 | 0 | 0/0.0 | False | 98.45596055292447 | 273.83588131411284/42.47691302452733 | {'A': 0.44073455759599334, 'B': 0.4273789649415693, 'C': 0.2921535893155259} | 0.48 |
| local4000_k1/E06 | completed | 41.26282310280054 | 94.8401407958104 | 0 | 0/0.0 | False | 104.32468254374045 | 276.0894580620624/45.265842745987264 | {'A': 0.46243739565943237, 'B': 0.4607679465776294, 'C': 0.2687813021702838} | 0.5109034267912772 |
| local4000_k1/E08 | completed | 30.54926180191695 | 71.66478975510908 | 0 | 0/0.0 | False | 75.70021283852017 | 274.07664099108257/46.19908582691712 | {'A': 0.42237061769616024, 'B': 0.4240400667779633, 'C': 0.25375626043405675} | 0.4982698961937716 |
| local4000_k1/E10 | completed | 24.18541106232728 | 130.93586122041822 | 0 | 0/0.0 | False | 154.06144416792807 | 277.5948954245554/50.63274990654594 | {'A': 0.41903171953255425, 'B': 0.39565943238731216, 'C': 0.2654424040066778} | 0.5276752767527675 |
| historical_baseline_fast5/E00 | completed | 41.95423511717145 | 50.90558715953609 | 0 | 0/0.0 | False | 61.52210673156619 | 273.1590027285127/43.18457562755102 | {'A': 0.3722871452420701, 'B': 0.44073455759599334, 'C': 0.29716193656093487} | 0.4957627118644068 |
| historical_baseline_fast5/E02 | completed | 35.24362875568662 | 58.89450626945718 | 0 | 0/0.0 | False | 72.58011695260166 | 274.2559438514699/42.41976165975194 | {'A': 0.44741235392320533, 'B': 0.4240400667779633, 'C': 0.2921535893155259} | 0.4673913043478261 |
| historical_baseline_fast5/E06 | completed | 27.311038456880812 | 43.13566854474498 | 0 | 0/0.0 | False | 71.29500788773778 | 276.6956601393883/47.61671278885024 | {'A': 0.4657762938230384, 'B': 0.4240400667779633, 'C': 0.2687813021702838} | 0.47954545454545455 |
| historical_baseline_fast5/E08 | completed | 35.1135925222468 | 63.55687479946394 | 0 | 0/0.0 | False | 80.18314523929305 | 274.71885571924963/47.78300469582904 | {'A': 0.4323873121869783, 'B': 0.41736227045075125, 'C': 0.24040066777963273} | 0.5027027027027027 |
| historical_baseline_fast5/E10 | completed | 23.20163001031677 | 111.35380362949151 | 0 | 0/0.0 | False | 129.46843887150206 | 278.2272921148625/51.652922586679296 | {'A': 0.41736227045075125, 'B': 0.39232053422370616, 'C': 0.2687813021702838} | 0.4897119341563786 |
| historical_stage4_fast5/E00 | completed | 31.592551719993192 | 75.75511179591602 | 0 | 0/0.0 | False | 85.74112664477093 | 273.63789735337065/43.38278186090377 | {'A': 0.4006677796327212, 'B': 0.4540901502504174, 'C': 0.2654424040066778} | 0.48484848484848486 |
| historical_stage4_fast5/E02 | completed | 32.10003131946855 | 85.74432912555933 | 0 | 0/0.0 | False | 81.65433719732796 | 274.34564443523004/42.20420661911823 | {'A': 0.4507512520868113, 'B': 0.4340567612687813, 'C': 0.32554257095158595} | 0.4875283446712018 |
| historical_stage4_fast5/E06 | completed | 35.064502150079065 | 84.86455812214449 | 0 | 0/0.0 | False | 101.16722024241164 | 276.3868731794905/47.69522609245025 | {'A': 0.44574290484140233, 'B': 0.4574290484140234, 'C': 0.2704507512520868} | 0.5117370892018779 |
| historical_stage4_fast5/E08 | completed | 34.49185295504347 | 76.78378627090609 | 0 | 0/0.0 | False | 85.40888654601778 | 274.3175788193094/43.51608422447349 | {'A': 0.4357262103505843, 'B': 0.4273789649415693, 'C': 0.24540901502504173} | 0.5120967741935484 |
| historical_stage4_fast5/E10 | completed | 19.98362740626262 | 142.97002163275488 | 1 | 1/0.05 | False | 154.69688326629807 | 278.131090749082/52.523507183740094 | {'A': 0.4357262103505843, 'B': 0.4090150250417362, 'C': 0.3121869782971619} | 0.45910290237467016 |
| selected_k4_mean/E00 | completed | 160.42446648891095 | 180.030003699133 | 0 | 0/0.0 | False | 186.81149625394173 | 258.74853931547113/14.409186063410948 | {'A': 0.39398998330550916, 'B': 0.38898163606010017, 'C': 0.327212020033389} | None |
| selected_k4_mean/E02 | completed | 40.01703282212109 | 58.71510536517016 | 0 | 0/0.0 | False | 63.465308718549174 | 255.83978629055846/17.37308251531108 | {'A': 0.327212020033389, 'B': 0.41569282136894825, 'C': 0.31385642737896496} | 0.3724832214765101 |
| selected_k4_mean/E06 | completed | 25.18972944518165 | 80.27152636661114 | 0 | 0/0.0 | False | 81.66855751834848 | 258.9333125938223/13.79844682129026 | {'A': 0.34891485809682804, 'B': 0.4073455759599332, 'C': 0.24373956594323873} | 0.32680722891566266 |
| selected_k4_mean/E08 | completed | 50.70766649963625 | 74.99357908377532 | 0 | 0/0.0 | False | 81.7353046619534 | 256.1721664543184/12.555022557332645 | {'A': 0.34223706176961605, 'B': 0.32220367278797996, 'C': 0.3005008347245409} | 0.32272727272727275 |
| selected_k4_mean/E10 | completed | 38.302730907810734 | 103.0505557543399 | 0 | 0/0.0 | False | 111.25045865364623 | 261.14778930531173/22.433885405913106 | {'A': 0.3038397328881469, 'B': 0.39398998330550916, 'C': 0.2888146911519199} | 0.2863636363636364 |

## Decision and next data

{'metric_improvement_vs_historical': {'final_mean_um_delta': 32.77562441402881, 'min_mean_um_delta': 0.8272045400800963}, 'fast5_gate_pass': False, 'entered20': 0, 'sustained200': 0, 'strict_success': 0, 'ab_sample_flip_change': {'A': 0.0, 'B': 0.0}}

For entries without 200-cycle hold, next data must cover the last 200–600 cycles of approach, braking and repeated exit/re-entry: signed OCT, velocity, previous_action, accepted ABC commands and motor execution speed. Inspect per-target max hold and last200 maximum before selecting any new training.

Step0 full historical hold table follows. Missing/short raw stays explicit.

# Unified 20 um sustained-hold metrics

Read-only metrics from completed raw trajectories. No extrapolation.

| Model | Target | status | min um | final um | entries <=20 | max <=20 cycles | max <=20 s | sustained 200 | last-200 max um | speed overall um/s | speed last-200 um/s |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| baseline_task2_step4000_fast5 | E00 | completed | 41.954 | 50.906 | 0 | 0 | 0.00 | False | 61.522 | 273.159 | 43.185 |
| baseline_task2_step4000_fast5 | E02 | completed | 35.244 | 58.895 | 0 | 0 | 0.00 | False | 72.580 | 274.256 | 42.420 |
| baseline_task2_step4000_fast5 | E06 | completed | 27.311 | 43.136 | 0 | 0 | 0.00 | False | 71.295 | 276.696 | 47.617 |
| baseline_task2_step4000_fast5 | E08 | completed | 35.114 | 63.557 | 0 | 0 | 0.00 | False | 80.183 | 274.719 | 47.783 |
| baseline_task2_step4000_fast5 | E10 | completed | 23.202 | 111.354 | 0 | 0 | 0.00 | False | 129.468 | 278.227 | 51.653 |
| baseline_task2_step4000_dev20 | E00 | completed | 41.954 | 50.906 | 0 | 0 | 0.00 | False | 61.522 | 273.159 | 43.185 |
| baseline_task2_step4000_dev20 | E01 | completed | 34.963 | 63.556 | 0 | 0 | 0.00 | False | 88.844 | 275.694 | 49.078 |
| baseline_task2_step4000_dev20 | E02 | completed | 35.244 | 58.895 | 0 | 0 | 0.00 | False | 72.580 | 274.256 | 42.420 |
| baseline_task2_step4000_dev20 | E03 | completed | 32.249 | 43.470 | 0 | 0 | 0.00 | False | 86.334 | 274.964 | 45.691 |
| baseline_task2_step4000_dev20 | E04 | completed | 32.684 | 66.078 | 0 | 0 | 0.00 | False | 84.677 | 276.660 | 48.731 |
| baseline_task2_step4000_dev20 | E05 | completed | 29.668 | 87.923 | 0 | 0 | 0.00 | False | 83.690 | 278.604 | 48.710 |
| baseline_task2_step4000_dev20 | E06 | completed | 27.311 | 43.136 | 0 | 0 | 0.00 | False | 71.295 | 276.696 | 47.617 |
| baseline_task2_step4000_dev20 | E07 | completed | 21.465 | 58.251 | 0 | 0 | 0.00 | False | 90.100 | 278.365 | 55.071 |
| baseline_task2_step4000_dev20 | E08 | completed | 35.114 | 63.557 | 0 | 0 | 0.00 | False | 80.183 | 274.719 | 47.783 |
| baseline_task2_step4000_dev20 | E09 | completed | 31.981 | 77.182 | 0 | 0 | 0.00 | False | 80.016 | 275.744 | 43.049 |
| baseline_task2_step4000_dev20 | E10 | completed | 23.202 | 111.354 | 0 | 0 | 0.00 | False | 129.468 | 278.227 | 51.653 |
| baseline_task2_step4000_dev20 | E11 | completed | 26.020 | 42.918 | 0 | 0 | 0.00 | False | 62.792 | 277.246 | 48.254 |
| baseline_task2_step4000_dev20 | E12 | completed | 34.778 | 74.906 | 0 | 0 | 0.00 | False | 74.883 | 275.372 | 44.195 |
| baseline_task2_step4000_dev20 | E13 | completed | 39.024 | 44.941 | 0 | 0 | 0.00 | False | 87.859 | 277.269 | 49.950 |
| baseline_task2_step4000_dev20 | E14 | completed | 31.559 | 52.137 | 0 | 0 | 0.00 | False | 111.487 | 277.723 | 47.186 |
| baseline_task2_step4000_dev20 | E15 | completed | 29.391 | 46.174 | 0 | 0 | 0.00 | False | 58.620 | 279.289 | 49.594 |
| baseline_task2_step4000_dev20 | E16 | completed | 30.988 | 59.332 | 0 | 0 | 0.00 | False | 63.442 | 277.256 | 48.998 |
| baseline_task2_step4000_dev20 | E17 | completed | 19.155 | 28.557 | 1 | 1 | 0.05 | False | 74.150 | 279.433 | 50.376 |
| baseline_task2_step4000_dev20 | E18 | completed | 29.827 | 59.940 | 0 | 0 | 0.00 | False | 65.556 | 277.974 | 49.962 |
| baseline_task2_step4000_dev20 | E19 | completed | 30.815 | 44.523 | 0 | 0 | 0.00 | False | 62.943 | 278.206 | 49.438 |
| v4_fallback_F_step4000_fast5 | E00 | completed | 45.647 | 56.718 | 0 | 0 | 0.00 | False | 66.561 | 272.554 | 40.909 |
| v4_fallback_F_step4000_fast5 | E02 | completed | 43.815 | 57.302 | 0 | 0 | 0.00 | False | 70.602 | 272.268 | 40.779 |
| v4_fallback_F_step4000_fast5 | E06 | completed | 26.522 | 36.798 | 0 | 0 | 0.00 | False | 71.663 | 276.015 | 49.723 |
| v4_fallback_F_step4000_fast5 | E08 | completed | 33.273 | 58.161 | 0 | 0 | 0.00 | False | 81.158 | 273.195 | 41.140 |
| v4_fallback_F_step4000_fast5 | E10 | completed | 16.992 | 62.590 | 1 | 4 | 0.20 | False | 74.950 | 276.679 | 51.310 |
| v4_fallback_F_step4000_dev20 | E00 | completed | 45.647 | 56.718 | 0 | 0 | 0.00 | False | 66.561 | 272.554 | 40.909 |
| v4_fallback_F_step4000_dev20 | E01 | completed | 29.454 | 75.891 | 0 | 0 | 0.00 | False | 102.811 | 274.221 | 45.516 |
| v4_fallback_F_step4000_dev20 | E02 | completed | 43.815 | 57.302 | 0 | 0 | 0.00 | False | 70.602 | 272.268 | 40.779 |
| v4_fallback_F_step4000_dev20 | E03 | completed | 26.867 | 47.121 | 0 | 0 | 0.00 | False | 79.800 | 275.110 | 46.148 |
| v4_fallback_F_step4000_dev20 | E04 | completed | 29.752 | 57.217 | 0 | 0 | 0.00 | False | 70.860 | 275.331 | 44.895 |
| v4_fallback_F_step4000_dev20 | E05 | completed | 29.790 | 57.306 | 0 | 0 | 0.00 | False | 58.011 | 277.104 | 46.575 |
| v4_fallback_F_step4000_dev20 | E06 | completed | 26.522 | 36.798 | 0 | 0 | 0.00 | False | 71.663 | 276.015 | 49.723 |
| v4_fallback_F_step4000_dev20 | E07 | completed | 25.269 | 51.554 | 0 | 0 | 0.00 | False | 58.123 | 276.703 | 50.451 |
| v4_fallback_F_step4000_dev20 | E08 | completed | 33.273 | 58.161 | 0 | 0 | 0.00 | False | 81.158 | 273.195 | 41.140 |
| v4_fallback_F_step4000_dev20 | E09 | completed | 37.066 | 53.673 | 0 | 0 | 0.00 | False | 77.089 | 275.075 | 42.198 |
| v4_fallback_F_step4000_dev20 | E10 | completed | 16.992 | 62.590 | 1 | 4 | 0.20 | False | 74.950 | 276.679 | 51.310 |
| v4_fallback_F_step4000_dev20 | E11 | completed | 31.617 | 38.589 | 0 | 0 | 0.00 | False | 80.007 | 276.603 | 47.531 |
| v4_fallback_F_step4000_dev20 | E12 | completed | 34.842 | 57.151 | 0 | 0 | 0.00 | False | 66.075 | 274.982 | 45.957 |
| v4_fallback_F_step4000_dev20 | E13 | completed | 38.096 | 42.057 | 0 | 0 | 0.00 | False | 92.893 | 275.380 | 46.367 |
| v4_fallback_F_step4000_dev20 | E14 | completed | 26.324 | 38.451 | 0 | 0 | 0.00 | False | 80.150 | 277.318 | 43.888 |
| v4_fallback_F_step4000_dev20 | E15 | completed | 28.616 | 45.298 | 0 | 0 | 0.00 | False | 63.144 | 277.557 | 48.412 |
| v4_fallback_F_step4000_dev20 | E16 | completed | 31.852 | 59.544 | 0 | 0 | 0.00 | False | 62.919 | 276.909 | 49.044 |
| v4_fallback_F_step4000_dev20 | E17 | completed | 34.791 | 38.657 | 0 | 0 | 0.00 | False | 105.658 | 278.488 | 48.276 |
| v4_fallback_F_step4000_dev20 | E18 | completed | 30.599 | 59.940 | 0 | 0 | 0.00 | False | 65.556 | 277.173 | 50.103 |
| v4_fallback_F_step4000_dev20 | E19 | completed | 26.613 | 37.607 | 0 | 0 | 0.00 | False | 75.493 | 276.887 | 49.595 |
| stage4_1_oct_symlog_fast5 | E00 | completed | 31.593 | 75.755 | 0 | 0 | 0.00 | False | 85.741 | 273.638 | 43.383 |
| stage4_1_oct_symlog_fast5 | E02 | completed | 32.100 | 85.744 | 0 | 0 | 0.00 | False | 81.654 | 274.346 | 42.204 |
| stage4_1_oct_symlog_fast5 | E06 | completed | 35.065 | 84.865 | 0 | 0 | 0.00 | False | 101.167 | 276.387 | 47.695 |
| stage4_1_oct_symlog_fast5 | E08 | completed | 34.492 | 76.784 | 0 | 0 | 0.00 | False | 85.409 | 274.318 | 43.516 |
| stage4_1_oct_symlog_fast5 | E10 | completed | 19.984 | 142.970 | 1 | 1 | 0.05 | False | 154.697 | 278.131 | 52.524 |
| v5_A_step2000_fast5 | E00 | completed | 45.725 | 50.906 | 0 | 0 | 0.00 | False | 69.161 | 273.069 | 44.264 |
| v5_A_step2000_fast5 | E02 | completed | 39.490 | 59.741 | 0 | 0 | 0.00 | False | 72.729 | 274.229 | 42.715 |
| v5_A_step2000_fast5 | E06 | completed | 25.818 | 27.135 | 0 | 0 | 0.00 | False | 73.089 | 276.955 | 47.620 |
| v5_A_step2000_fast5 | E08 | completed | 33.173 | 62.225 | 0 | 0 | 0.00 | False | 82.124 | 273.971 | 44.697 |
| v5_A_step2000_fast5 | E10 | completed | 24.088 | 85.448 | 0 | 0 | 0.00 | False | 94.255 | 278.186 | 51.477 |
| v5_A_step4000_fast5 | E00 | completed | 37.823 | 53.144 | 0 | 0 | 0.00 | False | 60.344 | 273.058 | 42.554 |
| v5_A_step4000_fast5 | E02 | completed | 39.612 | 59.136 | 0 | 0 | 0.00 | False | 74.930 | 274.573 | 43.558 |
| v5_A_step4000_fast5 | E06 | completed | 31.972 | 33.922 | 0 | 0 | 0.00 | False | 97.774 | 276.951 | 48.606 |
| v5_A_step4000_fast5 | E08 | completed | 32.855 | 57.026 | 0 | 0 | 0.00 | False | 76.374 | 274.763 | 47.778 |
| v5_A_step4000_fast5 | E10 | completed | 20.520 | 67.881 | 0 | 0 | 0.00 | False | 95.074 | 278.130 | 52.537 |
| v5_B_step4000_fast5 | E00 | completed | 52.732 | 63.735 | 0 | 0 | 0.00 | False | 85.283 | 269.871 | 38.433 |
| v5_B_step4000_fast5 | E02 | completed | 54.971 | 60.756 | 0 | 0 | 0.00 | False | 98.117 | 270.805 | 37.881 |
| v5_B_step4000_fast5 | E06 | completed | 50.784 | 50.518 | 0 | 0 | 0.00 | False | 109.258 | 274.242 | 44.956 |
| v5_B_step4000_fast5 | E08 | completed | 38.817 | 74.759 | 0 | 0 | 0.00 | False | 113.609 | 272.048 | 41.747 |
| v5_B_step4000_fast5 | E10 | completed | 26.717 | 43.444 | 0 | 0 | 0.00 | False | 51.922 | 275.280 | 46.944 |


Software/MuJoCo simulation only; no hardware accuracy claim.
