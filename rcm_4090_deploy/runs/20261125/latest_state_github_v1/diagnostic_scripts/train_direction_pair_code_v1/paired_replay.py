"""Matched approach/recession on one training target; branch controllers for 20 ticks."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path('/home/him1/mujoco_bc/rcm_4090_deploy')
sys.path.insert(0, str(ROOT))
from paired_select import select_pair
from lerobot_training.eval_policy import PolicyEvaluator, StrictRGBDSensor, visual_servo_settings
from rcm.scene_v9 import RCMSceneV9
from scripts.research.homog_replay_common import QUALITY, TARGETS, make_teacher
from lerobot_training.navigation_overlay import file_sha256

OUT = ROOT / 'runs/20261125/train_direction_pair_v1'
MODEL = ROOT / 'outputs/homog_label_fix_20261124_v5/homog_label_fix_20261124_v5_phase1_A/step_004000.pth'
META = ROOT / 'data/homog_label_fix_20261124_v5_A_t3_w025/metadata.json'
MODEL_SHA = '6cc389df3786f1f34c43cf65d6dc17fe910ab342fae8d84c8e37cea6a7db6c11'
TRAIN_EP = 40
PREFIX = 48
HORIZON = 20
SEED = 43001
UNIT = np.array([180 / np.pi, 180 / np.pi, 1000.])
METRIC_KEYS = ('requested_robot_ABCE', 'quantized_target_robot_ABCE', 'actual_after_robot_ABCE')


def save(name, value):
    path = OUT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False,
                              default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()))
    tmp.replace(path)


def record(stage, result):
    with (ROOT / 'TEST_LOG.md').open('a') as f:
        f.write(f'\n- {datetime.now().astimezone().isoformat()}: train_direction_pair_v1 {stage}: {result}; '
                f'config/output={OUT}; train episode 40, v5 A4000, seed {SEED}, 20 cycles.\n')


def training_target():
    row = json.loads(META.read_text())['episodes'][TRAIN_EP]
    assert row['episode_index'] == TRAIN_EP and row['regime'] == 'T3'
    point = np.asarray(row['target_robot_mm'])
    dev = json.loads(Path(TARGETS).read_text())['targets']
    coords = [t.get('reachable_target_robot_mm', t['point_mm']) for t in dev]
    return point, float(1000 * np.linalg.norm(np.asarray(coords) - point, axis=1).min())


def initial_scene(point):
    scene = RCMSceneV9()
    world = np.asarray(scene.robot_to_world(point))
    scene.set_target_world(world)
    q = scene.target_task(validate=True).copy()
    q[2] -= 0.000090
    scene.set_task(q)
    return scene, world


def preflight():
    import torch
    point, gap = training_target()
    assert gap > 100 and file_sha256(MODEL) == MODEL_SHA
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
    scene, world = initial_scene(point)
    try:
        initial_um = float(1000 * np.linalg.norm(scene.site_mm('needle_tip') - world))
        assert 75 < initial_um < 105, initial_um
    finally:
        scene.close()
    out = dict(checkpoint=str(MODEL), checkpoint_sha256=MODEL_SHA, training_metadata=str(META),
               training_source_sha256=file_sha256(META), train_episode=TRAIN_EP,
               target_robot_mm=point, nearest_dev_um=gap, initial_error_um=initial_um,
               code_sha256=file_sha256(Path(__file__)), selection_code_sha256=file_sha256(Path(__file__).with_name('paired_select.py')),
               prefix='24 forward and 24 reverse C commands, 3 um per tick; setup only',
               selection='one pair, pre-action true tip and signed OCT; no future errors or policy outputs',
               controls=['v5 A4000 alone','strict RGBD teacher'], control_period_s=.05, steps=HORIZON,
               k=1, flow_steps=4, exec_steps=1, common_policy_seed=SEED,
               outputs='per-step actual encoder positions, tip movement, independent 3D error, raw/accepted commands',
               scope='One deliberately induced axial history pair on a training target. No generalization or hold test.')
    save('card.json', out)
    save('workflow_status.json', {'status':'prepared'})
    record('preflight passed', f'known checkpoint, training target, nearest development target {gap:.2f} um, initial error {initial_um:.2f} um')
    print(json.dumps({'preflight':'passed','nearest_dev_um':gap,'initial_um':initial_um}))


def run_episode(evaluator, point, control, start=PREFIX):
    scene, world = initial_scene(point)
    sensor = teacher = None
    original = evaluator.get_action
    tips, frames, extra = [], [], []
    latest = {}
    try:
        settings = visual_servo_settings(scene)
        settings.update(perception_backend='strict_rgbd', rgbd_render_quality=QUALITY.copy())
        sensor = StrictRGBDSensor(scene, settings)
        if control == 'teacher':
            teacher = make_teacher(scene)
        observe = sensor.observe

        def observe_and_keep():
            obs = observe()
            latest['obs'] = obs
            return obs

        sensor.observe = observe_and_keep

        def command(rgb, state, oct_feature):
            i = len(tips)
            tips.append(scene.site_mm('needle_tip').copy())
            frames.append(hashlib.sha256(np.ascontiguousarray(rgb)).hexdigest())
            if i < start:
                # Run the production observer to fill the actual RGB/state history.
                original(rgb, state, oct_feature)
                return np.array([0., 0., .003 if i < PREFIX // 2 else -.003])
            if control == 'policy':
                if i == start:
                    evaluator.action_generator.manual_seed(SEED)
                return original(rgb, state, oct_feature)
            if control != 'teacher':
                raise RuntimeError('unexpected release during setup')
            q = scene.current_task().copy()
            action = teacher.geometry_action_from_observation(
                q, latest['obs'], previous_velocity=scene.motor_execution.actual_velocity.copy(), command_q=q)
            if not action['valid']:
                raise ValueError('invalid teacher observation or action')
            raw = np.asarray(action['action_deg_deg_mm'], dtype=float)
            extra.append(raw.tolist())
            return np.clip(raw, [-.2, -.1, -.03], [.2, .1, .03])

        evaluator.get_action = command
        steps = start if control == 'setup' else start + HORIZON
        ep = evaluator.evaluate_episode(scene, world, sensor=sensor, seed=3820, max_steps=steps,
            stop_on_success=False, control_period_s=.05, error_threshold_mm=.00035,
            max_consecutive_safety_limited=None)
        assert ep['early_stop_reason'] is None and len(ep['motor_records']) == steps
        ep.update(tips_pre_world_mm=tips, frame_sha256=frames, teacher_raw_before_caps=extra)
        return ep
    finally:
        evaluator.get_action = original
        if teacher is not None:
            teacher.sensor.close()
        if sensor is not None:
            sensor.close()
        scene.close()


def summarize(source, ep, start, control):
    a, b = source['trajectory'], ep['trajectory']
    for field in ('states','velocities','errors','oct_navigation_mm','simulation_time_s','oct_navigation_valid'):
        np.testing.assert_allclose(a[field][:start+1], b[field][:start+1], atol=1e-8, rtol=0)
    for field in ('raw_actions','actions'):
        np.testing.assert_allclose(a[field][:start], b[field][:start], atol=1e-10, rtol=0)
    assert source['frame_sha256'][:start+1] == ep['frame_sha256'][:start+1]
    for m, n in zip(source['motor_records'][:start], ep['motor_records'][:start]):
        assert all(m[k] == n[k] for k in METRIC_KEYS)
    tips = np.vstack([ep['tips_pre_world_mm'],ep['final_tip_world_mm']])
    delta = (tips - ep['target_world_mm']) * 1000
    error = np.linalg.norm(delta, axis=1)
    np.testing.assert_allclose(error, 1000*np.r_[b['errors'],ep['final_error_mm']], atol=1e-7, rtol=0)
    rows = []
    for i in range(start, start+HORIZON):
        m = ep['motor_records'][i]
        q_before = np.asarray(b['states'][i])
        q_after = np.asarray(m['actual_after_task_rad_rad_m'])*UNIT
        actuated = np.asarray(m['commanded_task_rad_rad_m'])*UNIT
        rows.append(dict(cycle=i-start+1, error_um=error[i+1], tip_delta_xyz_um=delta[i+1]-delta[i],
             raw_action_deg_deg_mm=ep['teacher_raw_before_caps'][i-start] if control=='teacher' else b['raw_actions'][i],
             accepted_action_deg_deg_mm=b['actions'][i], quantized_increment_deg_deg_mm=actuated-q_before,
             actual_increment_deg_deg_mm=q_after-q_before, actual_encoder_ABCE=m['actual_after_robot_ABCE'],
             actual_velocity_before_deg_deg_mm_s=b['velocities'][i]))
    return dict(prefix_verified=True, first_rgb_and_history_equal=True, step=start, initial_error_um=error[start],
        first_actual_velocity=b['velocities'][start],
        first_policy_velocity_from_encoder_history=(np.asarray(b['states'][start],dtype=np.float32)-np.asarray(b['states'][start-1],dtype=np.float32))/.05,
        previous_accepted=b['actions'][start-1],
        first_signed_oct_mm=b['oct_navigation_mm'][start], first=rows[0], error_1s_um=error[-1],
        error_change_1s_um=error[-1]-error[start], tip_displacement_1s_xyz_um=delta[-1]-delta[start],
        motionless_A_cycles=sum(abs(r['actual_increment_deg_deg_mm'][0])<1e-9 for r in rows), cycles=rows)


def replay():
    card=json.loads((OUT/'card.json').read_text())
    for path,key in [(MODEL,'checkpoint_sha256'),(META,'training_source_sha256'),(Path(__file__),'code_sha256'),
                     (Path(__file__).with_name('paired_select.py'),'selection_code_sha256')]:
        assert file_sha256(path)==card[key]
    point, _ = training_target()
    model = PolicyEvaluator(str(MODEL), device='cuda', num_action_samples=1, n_action_execute_steps=1,
                            flow_integration_steps=4, controller_variant='feedback', rgbd_render_quality=QUALITY.copy())
    assert model.fine_evaluator is None and model.n_obs_steps==2 and model.include_velocity and model.include_previous_action
    source = run_episode(model, point, 'setup')
    save('setup.json',source)
    pair = select_pair(source, tips_world_mm=np.asarray(source['tips_pre_world_mm']),horizon=0)
    save('pair.json',pair)
    if pair['status']!='matched':
        save('workflow_status.json',{'status':'no_pair','diagnosis':'No pair met the prespecified criteria','details':pair})
        record('selection gate failed', 'no matched pair; no branches executed')
        return
    record('selection passed', f"approach step {pair['approach']['step']}, recede step {pair['recede']['step']}")
    results={}
    for direction in ('approach','recede'):
        start=int(pair[direction]['step'])
        results[direction]={}
        for control in ('policy','teacher'):
            ep=run_episode(model,point,control,start)
            receipt=summarize(source,ep,start,control)
            save(direction+'_'+control+'.json',receipt)
            results[direction][control]={k:v for k,v in receipt.items() if k!='cycles'}
            record(direction+' '+control+' passed', f"exact prefix; error {receipt['initial_error_um']:.2f} -> {receipt['error_1s_um']:.2f} um")
    save('summary.json', {'pair':pair,'branches':results,'policy_seed':SEED,'steps':HORIZON})
    save('workflow_status.json',{'status':'completed','result':str(OUT/'summary.json')})
    record('completed','4 verified 1-second branches saved to summary.json')
    print('Results:',OUT/'summary.json')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=('prepare','run'))
    stage=p.parse_args().stage
    if stage=='prepare':
        OUT.mkdir(exist_ok=False)
    try:
        if stage=='prepare':
            preflight()
        else:
            assert json.loads((OUT/'workflow_status.json').read_text())['status']=='prepared'
            save('workflow_status.json',{'status':'running'})
            replay()
    except Exception as exc:
        save('workflow_status.json',{'status':'failed','stage':stage,'error':str(exc),'traceback':traceback.format_exc()})
        record(stage+' failed',str(exc))
        raise
