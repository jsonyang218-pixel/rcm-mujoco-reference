# Trained policy navigation demonstration

- [Video with needle-tip trajectory](trained_policy_with_trajectory.mp4)
- [3D needle-tip trajectory](needle_tip_trajectory_3d.png)
- [Trajectory coordinates](needle_tip_trajectory.csv)

This single-target MuJoCo simulation used the trained DiT rectified-flow
state-conditioned temporal attention policy, epoch 7, from
`outputs/homog_label_fix_20261124_v5/homog_label_fix_20261124_v5_phase1_B/best_model.pth`.
The checkpoint and training dataset are not included. Observations came from
strict RGB-D and actions were executed through the simulated MotorBackend.
The video shows the accumulated needle-tip path projected from simulation ground
truth; it was not used to control the policy.

The episode ran 1,600 steps (80 seconds of simulation) with seed 42. The
final error was 55.905 micrometres, so it did not satisfy the 20 micrometre
error and three-consecutive-settled-frames success condition. This is a
single-target simulation demonstration, not clinical or hardware validation.
