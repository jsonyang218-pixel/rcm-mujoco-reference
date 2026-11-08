# Stage 4 failure analysis

## Question

Was the final-test failure caused by:

1. target locations outside the development distribution,
2. fine-channel bias larger than the `20 um` precision threshold, or
3. policy instability after reaching the target?

The machine-readable report is
`outputs/restricted_stage4_failure_analysis_20261107.json`.

## Target distribution

The robot-coordinate bounding boxes overlap only partially. The more important
difference is the location on the compiled eye ellipsoid.

| Metric | Development | Final test |
| --- | ---: | ---: |
| normalized posterior-axis coordinate | 0.9006–0.9996 | 0.8001–0.8822 |
| polar angle from posterior axis | 1.71–25.76 deg | 28.10–36.86 deg |
| distance from eye center | 11.93–12.00 mm | 11.83–12.00 mm |
| P95 distance to ellipsoid wall | 0.000498 um | 0.000502 um |

All targets are effectively on the eye wall. The sets do not overlap in either
normalized posterior coordinate or polar angle: two-sample KS statistic is
`1.0` for both (`p=1.45e-28`). The closest development-to-final target distance
is `1.0187 mm`.

This is an out-of-distribution target-location test: the final set lies in a
more peripheral posterior band than every development target. Within the final
set, mean target success at 20 um falls monotonically:

| Final-target polar angle | Targets | Mean target success at 20 um |
| --- | ---: | ---: |
| 28–31 deg | 22 | 30.3% |
| 31–33 deg | 7 | 0% |
| 33–35 deg | 9 | 0% |
| 35–38 deg | 22 | 0% |

Across the 60 final targets, Spearman correlation is `rho=-0.460`
(`p=2.16e-4`) between polar angle and target success, and `rho=0.527`
(`p=1.52e-5`) between polar angle and median minimum error.

## Restricted-vector bias

The fixed coarse and fine biases were reconstructed from the exact episode
seeds and simulator episode IDs used in evaluation.

| Bias norm | Development P50/P95/max | Final-test P50/P95/max |
| --- | ---: | ---: |
| coarse | 299.2 / 420.9 / 466.2 um | 285.0 / 412.4 / 477.3 um |
| fine | 2.12 / 2.65 / 3.24 um | 2.00 / 2.86 / 3.31 um |

The development-versus-final KS tests are not significant for coarse bias
(`D=0.133`, `p=0.566`) or fine bias (`D=0.186`, `p=0.185`). No final-test
episode has fine fixed-bias norm above `20 um`.

Stage 0 reports near-target fused perception P95 `3.73 um`, fine P95
`4.98 um`, and deterministic closed-loop success `96.5%` at `20 um`. Since the
precision gate uses `2 x P95`, the measured perception floor supports a
`20 um` threshold; the threshold is not intrinsically unreachable because of
fine-channel bias.

The coarse bias is hundreds of micrometres, but it belongs to the far-range
channel. Near the target, the fusion selects the fine channel. Coarse bias
therefore does not explain the observed 20 um failures.

## Failure modes across 180 episodes

| Failure mode | Episodes |
| --- | ---: |
| success at 20 um | 20 |
| reached at least once inside 20 um but did not hold | 51 |
| minimum error 20–30 um | 8 |
| minimum error 30–50 um | 41 |
| minimum error 50–100 um | 45 |
| no approach from 100 um start | 6 |
| unsafe target initialization | 9 |

The 51 hold failures reached a median minimum error of `5.54 um`, but their
final error reached a median of `96.29 um`. Their fine channel was valid for a
median `99.52%` of control frames; median precision-hold fraction was only
`0.48%`, and coarse fallback never occurred.

This separates the mechanisms:

- Perception was available and accurate enough for many episodes.
- The policy reached the target region but could not generate a stable
  three-frame settled hold.
- Continued commands then drove the tip away from the target.

Three final targets, across all three seeds, fail before the first control
frame because the initial B-axis encoder lies outside the calibrated safe
range. Two additional targets, also all three seeds, never approach the target
and remain around the `100 um` start error. The final split generator did not
perform scene reachability validation when these targets were created.

## Conclusion

The evidence supports distribution shift as a primary explanation, not
fine-channel bias:

1. The final set is entirely outside the development set in normalized
   posterior location and polar angle.
2. Fine bias is below `20 um` for every episode and its distribution is not
   shifted relative to development.
3. Most failed episodes either reach the target and fail to hold or stop at a
   lateral/peripheral near-miss.
4. Nine episodes are invalid for policy-capability claims because their
   initial poses violate calibrated encoder limits.

The original final set cannot be reused as an untouched confirmatory set after
this analysis. A corrected protocol should:

- reject targets whose 100 um start poses violate encoder limits;
- stratify new final targets across the development and extended posterior
  bands rather than placing all final targets outside the development band;
- add an explicit stable-hold objective or hysteresis-based terminal
  controller;
- report success with unsafe-initialization failures separated from model
  failures.
