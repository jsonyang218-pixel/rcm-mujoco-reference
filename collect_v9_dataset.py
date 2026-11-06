"""Collect current-v9 RGB/state/action episodes through MotorBackend.

Each episode starts with an explicit simulator reset, then all motion is sent
as cumulative protocol handles. No target-pose inverse enters visual control.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from rcm_reference import HERE
from rcm_scene_v9 import RCMSceneV9
from rcm_fundus import make_random_fundus_targets
from rcm_visual_servo import VisualServoController, visual_servo_settings
from rcm_waypoint_dataset import EpisodeWriter, make_policy_task, write_manifest
from run_waypoint_comparison import guard_scene_goal_inverse, perception_record

UNITS = np.array([180 / np.pi, 180 / np.pi, 1000.])

# The ordinary dataset starts every episode at the same home pose.  Recovery
# collection instead starts near the sampled target so the expert contains
# both braking and reverse corrections around the goal.
RECOVERY_START_PROFILES = (
    ("under_0p02mm", (0.0, 0.0, -0.02)),
    ("under_0p05mm", (0.0, 0.0, -0.05)),
    ("under_0p10mm", (0.0, 0.0, -0.10)),
    ("under_0p20mm", (0.0, 0.0, -0.20)),
    ("under_0p40mm", (0.0, 0.0, -0.40)),
    ("under_0p80mm", (0.0, 0.0, -0.80)),
    ("over_0p02mm", (0.0, 0.0, 0.02)),
    ("over_0p05mm", (0.0, 0.0, 0.05)),
    ("over_0p10mm", (0.0, 0.0, 0.10)),
    ("over_0p20mm", (0.0, 0.0, 0.20)),
    ("over_0p40mm", (0.0, 0.0, 0.40)),
    ("over_0p80mm", (0.0, 0.0, 0.80)),
    ("lateral_A_pos", (0.20, 0.0, 0.0)),
    ("lateral_A_neg", (-0.20, 0.0, 0.0)),
    ("lateral_B_pos", (0.0, 0.20, 0.0)),
    ("lateral_B_neg", (0.0, -0.20, 0.0)),
    ("combined_A_pos_under", (0.20, 0.0, -0.20)),
    ("combined_A_neg_over", (-0.20, 0.0, 0.20)),
    ("combined_B_pos_under", (0.0, 0.20, -0.20)),
    ("combined_B_neg_over", (0.0, -0.20, 0.20)),
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def write_jsonl(path, rows):
    with Path(path).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def recovery_start_task(target_task, index):
    """Return a deterministic near-target start and its named perturbation."""
    target_task = np.asarray(target_task, dtype=np.float64)
    if target_task.shape != (3,) or not np.all(np.isfinite(target_task)):
        raise ValueError("target_task must be a finite A/B/C vector")
    name, delta = RECOVERY_START_PROFILES[int(index) % len(RECOVERY_START_PROFILES)]
    return target_task + np.asarray(delta, dtype=np.float64), name


def collect_episode(
    config,
    out,
    dataset,
    index,
    seed,
    start_task,
    random_target,
    max_iterations,
    start_task_mode="fixed",
):
    episode_dir = out / f'episode_{index:06d}'
    episode_dir.mkdir()
    # Initial placement is explicitly a simulator reset, before the first
    # dataset observation. It is not represented as a motor action.
    scene = RCMSceneV9({**copy.deepcopy(config), 'motor_execution': {'enabled': False}})
    scene.set_task(np.zeros(3))
    if random_target:
        targets = make_random_fundus_targets(scene, np.random.default_rng(seed), 1)
        target_robot = targets['targets'][0]['point_mm']
        scene.set_target_world(scene.robot_to_world(target_robot))
    if start_task_mode == "recovery":
        start_task, start_task_profile = recovery_start_task(
            scene.target_task(validate=True) * UNITS,
            index,
        )
    else:
        start_task = np.asarray(start_task, dtype=np.float64)
        start_task_profile = "fixed"
    if start_task.shape != (3,) or not np.all(np.isfinite(start_task)):
        scene.close()
        raise ValueError("start_task must be a finite A/B/C vector")
    scene.set_task(start_task / UNITS)
    # Capture the goal only for audit attribution before strict visual
    # isolation is enabled. It must never enter the controller or policy data.
    goal_task_abc = scene.target_task(validate=True) * UNITS
    target_robot = scene.world_to_robot(scene.site_mm('lesion')).tolist()
    scene.enable_motor_execution(config.get('motor_execution'))
    motor = scene.motor_execution
    settings = visual_servo_settings(scene)
    settings.update(max_iterations=max_iterations, save_diagnostics=True,
                    position_feedback='geometry', occlusion_handling='stop',
                    perception_backend='strict_rgbd')
    if not np.isclose(settings['sample_period_s'], motor.control_period_s):
        scene.close()
        raise ValueError('Motor and visual control sample periods must match')
    guard_scene_goal_inverse(scene)
    controller = VisualServoController(scene, settings=settings, output_dir=episode_dir)
    writer = EpisodeWriter(dataset, index, make_policy_task('visual_current_target', []),
                           'visual_servo_motor', seed, ['P1'])
    audits = []
    pending = None
    started = time.monotonic()

    def store(sample, action, terminal=False, transition=None):
        writer.add(sample['time_s'], sample['state'], action, 'P1', sample['rgb'],
                   terminal=terminal, phase='hold' if terminal else 'visual_servo',
                   visibility=sample['visibility'], perception=sample['perception'])
        audits.append({
            'frame_index': len(audits), 'timestamp_s': sample['time_s'],
            'robot_ABCE': sample['robot'], 'state_A_deg_B_deg_C_mm': sample['state'].tolist(),
            'rgb_pixels_sha256': hashlib.sha256(sample['rgb'].tobytes()).hexdigest(),
            'motor_transition_index': transition, 'terminal': terminal,
        })

    def capture(iteration, observation):
        nonlocal pending
        if pending is not None:
            step = motor.last_record
            if step is None or not np.isclose(step['start_time_s'], pending['time_s']):
                raise RuntimeError('Motor/image transition timing mismatch')
            store(pending, np.asarray(step['commanded_task_rad_rad_m']) * UNITS,
                  transition=step['step_index'])
        pending = {
            'time_s': motor.time_s, 'state': scene.current_task() * UNITS,
            'robot': list(scene.motor_robot()), 'rgb': observation['rgb'].copy(),
            'visibility': controller.sensor.visibility_metrics(observation['segmentation']),
            'perception': {**perception_record(scene, observation, 'visual_servo'),
                           'control_frame_index': iteration},
        }
        if iteration % 50 == 0:
            print(json.dumps({'episode': index, 'iteration': iteration,
                'simulation_s': motor.time_s, 'wall_s': round(time.monotonic()-started, 1),
                'error_um': scene.metrics()['tip_target_error_um'],
                'actual_C_mm': pending['state'][2]}, ensure_ascii=False), flush=True)
        return True

    controller.iteration_callback = capture
    try:
        report = controller.run()
        if pending is not None:
            store(pending, pending['state'], terminal=True)
        # _run_geometry() may break before calling iteration_callback (e.g.
        # LOW_QUALITY_VISUAL_GEOMETRY, SINGULAR_ACTUATION_JACOBIAN) after the
        # motor advance at the top of the loop already appended a record, but
        # before a frame was captured for it. That leaves one orphaned trailing
        # motor step with no observation. Drop it so len(records)==len(audits)-1
        # and motor.time_s matches the last frame timestamp.
        _expected = max(0, len(audits) - 1)
        if len(motor.records) > _expected:
            _orphaned = len(motor.records) - _expected
            motor.records = motor.records[:_expected]
            motor.last_record = motor.records[-1] if motor.records else None
            motor.time_s -= _orphaned * motor.control_period_s
        report.update(scene_backend='RCMSceneV9',
            reset_task_A_deg_B_deg_C_mm=np.asarray(start_task).tolist(),
            reset_task_profile=start_task_profile,
            target_task_A_deg_B_deg_C_mm=goal_task_abc.tolist(),
            target_robot_mm=target_robot,
            collection_wall_s=time.monotonic()-started)
        controller.write(report, episode_dir)
        write_jsonl(episode_dir/'motor_steps.jsonl', motor.records)
        write_jsonl(episode_dir/'frame_audit.jsonl', audits)
        episode = writer.finish(report['result'], stop_constraint_summary={
            'pass': bool(report['terminal_stop_validated']), 'attempted_targets': 1,
            'targets_with_validated_stop': int(report['terminal_stop_validated'])})
        episode.update(motor_steps=(episode_dir/'motor_steps.jsonl').relative_to(out).as_posix(),
            frame_audit=(episode_dir/'frame_audit.jsonl').relative_to(out).as_posix(),
            servo_report=(episode_dir/'visual_servo_report.json').relative_to(out).as_posix(),
            initial_task_A_deg_B_deg_C_mm=np.asarray(start_task).tolist(),
            initial_task_profile=start_task_profile,
            final_error_um=report['final_tip_target_error_um'],
            motor_execution=motor.summary())
        return episode
    finally:
        controller.sensor.close()
        scene.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'config.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20261030)
    parser.add_argument('--start-task', nargs=3, type=float, default=[0.,0.,0.], metavar=('A_DEG','B_DEG','C_MM'))
    parser.add_argument(
        '--start-task-mode',
        choices=('fixed', 'recovery'),
        default='fixed',
        help='fixed uses --start-task; recovery cycles deterministic perturbations around each target',
    )
    parser.add_argument('--random-targets', action='store_true', help='One seeded posterior-cap target per independent episode')
    parser.add_argument('--max-iterations', type=int, default=1600)
    parser.add_argument('--resume', action='store_true',
                        help='Continue an existing dataset up to the requested total episode count')
    args = parser.parse_args()
    if args.episodes < 1 or args.max_iterations < 1:
        parser.error('Episodes and max-iterations must be positive')
    if args.start_task_mode == 'recovery' and args.start_task != [0., 0., 0.]:
        parser.error('--start-task cannot be combined with --start-task-mode recovery')
    out = args.output_dir.resolve()
    dataset = out/'dataset'
    config = json.loads(args.config.read_text())
    config['motor_execution'] = {**config.get('motor_execution', {}), 'enabled': True}
    config['navigation']['visual_servo']['perception_backend'] = 'strict_rgbd'
    source_hashes = {n: hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in (
        'collect_v9_dataset.py','rcm_motor_execution.py','rcm_reference.py','rcm_scene_v9.py',
        'rcm_visual_servo.py','rcm_visual_geometry.py','rcm_waypoint_dataset.py',
        'mujoco_model/motor_backend.py','mujoco_model/model_sync.py','v9_eye_scene.xml',
        'rcm_strict_rgbd.py','rcm_rgb_features.py')}
    provenance = {'scene': 'RCMSceneV9', 'execution': 'MotorBackend consume/step -> feedback -> qpos',
        'scope': 'Single-target independent episodes; no RRT/PRM/multitarget motor validation or hardware data',
        'seed': args.seed, 'requested_episodes': args.episodes, 'random_targets': args.random_targets,
        'start_task_mode': args.start_task_mode,
        'start_task': args.start_task,
        'recovery_start_profiles': (
            [{'name': name, 'delta_A_deg_B_deg_C_mm': list(delta)}
             for name, delta in RECOVERY_START_PROFILES]
            if args.start_task_mode == 'recovery' else []
        ),
        'source_sha256': source_hashes,
        'motor_parameters_are_identified': False, 'clock': 'simulation; not realtime acquisition'}
    episodes = []
    if out.exists():
        if not args.resume:
            parser.error('Output directory already exists; pass --resume to continue it')
        manifest_path = dataset/'manifest.json'
        if not manifest_path.is_file():
            parser.error('Cannot resume without dataset/manifest.json')
        previous_protocol = json.loads((out/'collection_protocol.json').read_text())
        for key in ('scene', 'execution', 'seed', 'requested_episodes',
                    'random_targets', 'start_task_mode', 'start_task',
                    'recovery_start_profiles'):
            if previous_protocol.get(key) != provenance.get(key):
                parser.error(f'Cannot resume: collection protocol field {key!r} changed')
        previous_config = json.loads((out/'config_snapshot.json').read_text())
        if previous_config != config:
            parser.error('Cannot resume: effective config changed')
        previous_manifest = json.loads(manifest_path.read_text())
        episodes = list(previous_manifest.get('episodes', []))
        if [entry.get('episode_index') for entry in episodes] != list(range(len(episodes))):
            parser.error('Cannot resume: episode indexes are not contiguous')
        if len(episodes) > args.episodes:
            parser.error('Existing dataset has more episodes than requested')
        source_hash_changed = (
            previous_protocol.get('source_sha256') != provenance.get('source_sha256'))
        # A completed dataset can be reopened for validation without generating
        # new samples. Preserve the original collection hashes and record the
        # verifier hashes separately; partial datasets still require exact
        # source compatibility before appending data.
        if source_hash_changed and len(episodes) != args.episodes:
            parser.error('Cannot resume: collection protocol field \'source_sha256\' changed')
        next_index = len(episodes)
        if (out/f'episode_{next_index:06d}').exists():
            parser.error('Cannot resume: an uncommitted episode directory already exists')
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/'config_snapshot.json', config)
        write_json(out/'collection_protocol.json', provenance)
        next_index = 0
    for index in range(next_index, args.episodes):
        episode = collect_episode(config, out, dataset, index, args.seed+index,
            args.start_task, args.random_targets, args.max_iterations,
            args.start_task_mode)
        episodes.append(episode)
        manifest = write_manifest(dataset, episodes, [1440,1920,3],
            float(config['navigation']['visual_servo']['sample_period_s']), 'visual_current_target')
        manifest.update(source='Current v9 offline MotorBackend first-order simulation; no hardware measurements',
            execution_backend='MotorBackend',
            perception_backend='strict_rgbd',
            visibility_semantics='Occlusion cause is unknown; navigation uses RGB-derived masks and normal visible depth without simulator IDs or counterfactual rendering',
            action_semantics='Quantized protocol target as absolute logical A/B/C task position for the following control interval; never the following actual state',
            state_semantics='Actual encoder feedback decoded into logical A/B/C at the same simulation instant as raw RGB',
            audit_path_base='Parent of this dataset directory',
            reset_semantics='Explicit simulator initialization before each episode; never labelled as an executed action')
        write_json(dataset/'manifest.json', manifest)
        print(json.dumps({'episode_complete': index, 'result': episode['result'], 'frames': episode['frames']},ensure_ascii=False),flush=True)
    from verify_v9_dataset import verify
    verification = verify(out)
    verification['collection_source_sha256'] = provenance.get('source_sha256')
    verification['verification_source_sha256'] = source_hashes
    verification['resume_source_hash_changed'] = (
        out.exists() and
        locals().get('source_hash_changed', False))
    write_json(out/'dataset_verification.json', verification)
    print(json.dumps(verification,ensure_ascii=False,indent=2))
    return 0 if verification['result']=='PASS' and all(e['result']=='PASS' for e in episodes) else 1


if __name__=='__main__':
    raise SystemExit(main())
