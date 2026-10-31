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


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def write_jsonl(path, rows):
    with Path(path).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def collect_episode(config, out, dataset, index, seed, start_task, random_target, max_iterations):
    episode_dir = out / f'episode_{index:06d}'
    episode_dir.mkdir()
    # Initial placement is explicitly a simulator reset, before the first
    # dataset observation. It is not represented as a motor action.
    scene = RCMSceneV9({**copy.deepcopy(config), 'motor_execution': {'enabled': False}})
    scene.set_task(np.asarray(start_task) / UNITS)
    if random_target:
        targets = make_random_fundus_targets(scene, np.random.default_rng(seed), 1)
        target_robot = targets['targets'][0]['point_mm']
        scene.set_target_world(scene.robot_to_world(target_robot))
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
        report.update(scene_backend='RCMSceneV9',
            reset_task_A_deg_B_deg_C_mm=list(start_task), target_robot_mm=target_robot,
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
            initial_task_A_deg_B_deg_C_mm=list(start_task),
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
    parser.add_argument('--random-targets', action='store_true', help='One seeded posterior-cap target per independent episode')
    parser.add_argument('--max-iterations', type=int, default=1600)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_iterations < 1:
        parser.error('Episodes and max-iterations must be positive')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    dataset = out/'dataset'
    config = json.loads(args.config.read_text())
    config['motor_execution'] = {**config.get('motor_execution', {}), 'enabled': True}
    config['navigation']['visual_servo']['perception_backend'] = 'strict_rgbd'
    write_json(out/'config_snapshot.json', config)
    provenance = {'scene': 'RCMSceneV9', 'execution': 'MotorBackend consume/step -> feedback -> qpos',
        'scope': 'Single-target independent episodes; no RRT/PRM/multitarget motor validation or hardware data',
        'seed': args.seed, 'requested_episodes': args.episodes, 'random_targets': args.random_targets,
        'source_sha256': {n: hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in (
            'collect_v9_dataset.py','rcm_motor_execution.py','rcm_reference.py','rcm_scene_v9.py',
            'rcm_visual_servo.py','rcm_visual_geometry.py','rcm_waypoint_dataset.py',
            'mujoco_model/motor_backend.py','mujoco_model/model_sync.py','v9_eye_scene.xml',
            'rcm_strict_rgbd.py','rcm_rgb_features.py')},
        'motor_parameters_are_identified': False, 'clock': 'simulation; not realtime acquisition'}
    write_json(out/'collection_protocol.json', provenance)
    episodes = []
    for index in range(args.episodes):
        episode = collect_episode(config, out, dataset, index, args.seed+index,
            args.start_task, args.random_targets, args.max_iterations)
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
    write_json(out/'dataset_verification.json', verification)
    print(json.dumps(verification,ensure_ascii=False,indent=2))
    return 0 if verification['result']=='PASS' and all(e['result']=='PASS' for e in episodes) else 1


if __name__=='__main__':
    raise SystemExit(main())
