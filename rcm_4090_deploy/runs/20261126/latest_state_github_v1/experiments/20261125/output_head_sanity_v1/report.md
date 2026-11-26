# Plan 8: deterministic output-head and action-scale sanity check

Workflow status: **failed**. Elapsed: 0.08 minutes.

Frozen baseline: baseline_task2_step4000, SHA256 185a6ef02caea4ea74ba752d67618d459c906450f5b315cd8010c3c5f1703f1b.
Training: same effective baseline dataset, C-only overlay, 8% correction replacement sampling, batch16, 4000 updates per arm.
Head: 384→384→384→48, LayerNorm/SiLU, AdamW lr1e-4 (baseline config general lr), decay1e-4, AMP fp16, seed20261118.
Encoder eval/frozen; exact F_geom_cond cache parity, no global vision fusion. Ground truth only audit/training mask/post-hoc metrics.
R2 cosine uses physical action divided by fixed per-axis symlog scales; masked per future training frame <80µm and nonzero.

## Action-scale audit

### all

```json
{"frames": 319697, "physical_abs_median_p90_max_ABC": [[0.0036001205444335938, 0.0012090152595192194, 0.041536808013916016], [0.014400005340576172, 0.0073757171630859375, 0.042461395263671875], [0.28980064392089844, 0.14434337615966797, 0.042713165283203125]], "normalized_abs_median_p90_max_ABC": [[0.08328167763413816, 0.04243063730797207, 0.5448637658372973], [0.32095091438231677, 0.24895310697218903, 1.9609742202282994], [6.381595153358333, 4.869035441387064, 4.551260703866013]], "fraction_total_frames": 1.0, "fraction_total_normalized_energy": 1.0}
```

### true_lt60

```json
{"frames": 93859, "physical_abs_median_p90_max_ABC": [[0.001154190395027399, 0.0008510205079801381, 0.0030605902429670095], [0.0049262306652963165, 0.0023866714909672737, 0.006100260186940432], [0.06120000034570694, 0.044992879033088684, 0.023377083241939545]], "normalized_abs_median_p90_max_ABC": [[0.02774758975182832, 0.027675889088895688, 1.8212309497458108], [0.10913057200467341, 0.0825481284759637, 1.9956665759876555], [1.3427516147441487, 1.5195457564858437, 3.3943532868392583]], "fraction_total_frames": 0.2935873655367426, "fraction_total_normalized_energy": 0.3369042285214194}
```

### baseline_training_E40_lt60

```json
{"frames": 27, "physical_abs_median_p90_max_ABC": [[0.0, 3.24249267578125e-05, 0.00084686279296875], [0.012599992752075195, 0.005788993835449221, 0.00786247253417969], [0.014400005340576172, 0.02250385284423828, 0.011201858520507812]], "normalized_abs_median_p90_max_ABC": [[0.0040551008559692155, 0.0026471480110970857, 1.9449974774443854], [0.27647307900401846, 0.19464182598700064, 1.9956665759876555], [0.31284071267037833, 0.7613463866093019, 1.9956665759876555]], "fraction_total_frames": 8.445496829810727e-05, "fraction_total_normalized_energy": 9.813021716224497e-05}
```

### paired_v5_training_E40_lt60

```json
{"frames": 600, "dataset": "/home/him1/mujoco_bc/rcm_4090_deploy/data/homog_label_fix_20261124_v5_A_t3_w025/data/dataset.npz", "sha256": "67cd04157444fab90ff526f0d62871664139a7fd254ceb131288e894ac9a1ea6", "physical_abs_median_p90_max_ABC": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.008224266581237316, 0.006266583222895861, 0.0056366282515227795]], "normalized_abs_median_p90_max_ABC": [[0.0040551008559692155, 0.0026471480110970857, 1.9956665759876555], [0.0040551008559692155, 0.0026471480110970857, 1.9956665759876555], [0.1850436204626204, 0.21391994439541348, 2.023790513081794]]}
```

Dataset energy counts one action per stored frame, after C-overlay; normalized action energy is not realized Flow MSE.
Baseline training episode40 and v5 paired replay episode40 are different datasets/targets and are reported separately.

## Paired replay

| Start | Controller | min µm | final µm |
|---|---|---:|---:|

Archived 63.975/59.622µm belongs to v5 A4000, not baseline_task2. Exact source prefix/RGB/history verified before each new branch.

## fast5 hold and late flips

| Arm | E | min µm | final µm | entries ≤20 | max hold cycles | last200 max µm | late raw flip A/B/C |
|---|---|---:|---:|---:|---:|---:|---|

Repeated-identical-observation action flips are zero by deterministic construction; temporal closed-loop flips are separately measured above.
Fast5 gate: both paired finals must beat both the rerun baseline_task2 and the archived v5 reference. No retries or parameter changes. Below15min prioritize best eligible paired final mean.

## Conclusions

{}

Error/skip reason: AssertionError('Tensor-likes are not close!\n\nMismatched elements: 361 / 384 (94.0%)\nGreatest absolute difference: 0.0003998279571533203 at index (0, 245) (up to 1e-06 allowed)\nGreatest relative difference: 0.048562709242105484 at index (0, 3) (up to 1e-05 allowed)')

A touch below20µm, final convergence, sustained200 cycles and strict0.35µm/3-frame settled success are separate outcomes.
No dev20/final-test inference. These are software/MuJoCo simulation results, not hardware/clinical accuracy.
