"""One authorized paired-data / balanced-first-window original-v5 diagnostic.

All executable additions and artifacts stay in this directory. Production model,
teacher, scene and motor modules are imported unchanged. No neural closed loop.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import csv
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/him1/mujoco_bc/rcm_4090_deploy')
OUT = Path(__file__).resolve().parent
OLD = ROOT/'runs/20261126/original_neural_terminal_pilot_v1'
AUDIT = ROOT/'runs/20261126/original_neural_terminal_data_audit_v1'
SOURCE = ROOT/'data/homog_clean_replay_20261123_mixed_1600_600_12_hold_v2'
REF = Path(json.loads((OLD/'experiment_card.json').read_text())['init_checkpoint'])
REF_SHA = '6cc389df3786f1f34c43cf65d6dc17fe910ab342fae8d84c8e37cea6a7db6c11'
sys.path.insert(0, str(ROOT))
DESKTOP = dict(DISPLAY=':1', DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus', XDG_RUNTIME_DIR='/run/user/1000')
KEYS = ('observation.image', 'observation.state', 'observation.oct', 'observation.velocity', 'observation.previous_action')
CLASSES = ('approaching', 'receding', 'settled')
SEEDS = (92001, 92002, 92003, 92004)
FLOW_SEEDS = (91001, 91002, 91003, 91004)
SEED = 20261126
CHILDREN = []
DEADLINE = None
RUN_STARTED = None
PRIOR_BUDGET = 0.
PROGRESS = dict(data_gate='not_run', training='not_started', offline_improvement='not_evaluated',
    optimizer_updates=0, closed_loop_started=False, closed_loop_authorized=False)


class DataGateRejected(Exception):
    pass


def now():
    return datetime.now().astimezone().isoformat()


def default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value, update=False):
    path = Path(path)
    if not path.is_absolute():
        path = OUT/path
    assert path.is_relative_to(OUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=default)+'\n'
    if update:
        temp = path.with_suffix(path.suffix+'.tmp')
        temp.write_text(encoded)
        temp.replace(path)
    else:
        with path.open('x') as f:
            f.write(encoded)


def append(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=default)+'\n')
        f.flush()


def record(stage, result):
    row = dict(time=now(), experiment=OUT.name, stage=stage, result=result, command=sys.argv,
        card=str(OUT/'experiment_card.json'), original_model_sha256=REF_SHA,
        split='E40 train; E41 development holdout already used for selection', output=str(OUT))
    line = ('\n- '+json.dumps(row, ensure_ascii=False, allow_nan=False, default=default)+'\n').encode()
    receipts = []
    for path in (OUT/'TEST_LOG.md', ROOT/'TEST_LOG.md'):
        with path.open('a+b') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0, 2)
            offset = f.tell()
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
            f.seek(offset)
            assert f.read(len(line)) == line, f'TEST_LOG write/readback failed: {path}'
        receipts.append(dict(path=str(path), offset=offset, bytes=len(line), exact_readback=True))
    with (OUT/'test_log_receipts.jsonl').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(dict(time=row['time'], stage=stage, receipts=receipts))+'\n')
    print(json.dumps(row, ensure_ascii=False, default=default), flush=True)


def status(stage, state='running', **extra):
    row = dict(time=now(), status=state, stage=stage, pid=os.getpid(), **PROGRESS)
    row.update(extra)
    save('task_status.json', row, update=True)
    append(OUT/'status_history.jsonl', row)


def original():
    spec = importlib.util.spec_from_file_location('unchanged_original_neural_pilot', OLD/'pilot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.REF == REF and module.REF_SHA == REF_SHA and tuple(module.INPUT_KEYS) == KEYS
    return module


def plans():
    result = []
    for target in (40, 41):
        for side in ('below', 'above'):
            side_sign = -1 if side == 'below' else 1
            for level in (30, 50, 70):
                for motion in ('approaching', 'receding'):
                    direction = -side_sign if motion == 'approaching' else side_sign
                    # Fixed two real 3um ticks. 4.4um is only a registered setup
                    # offset estimate, not feedback compensation or a label.
                    initial_um = level + 4.4 if motion == 'approaching' else level - 4.4
                    result.append(dict(id=f'E{target}_{side}_{level}_{motion}', target=f'E{target}',
                        side=side, level_um=level, requested_incoming=motion,
                        initial_C_offset_m=side_sign*initial_um*1e-6,
                        prefix_actions_deg_deg_mm=[[0., 0., direction*.003]]*2,
                        teacher_steps=260 if level == 30 and motion == 'approaching' else 32,
                        designated_anchor_teacher_step=0))
    return result


def pools_for(meta):
    pools = {}
    for cls in CLASSES:
        pools[cls] = {}
        for side in ('below', 'above'):
            episodes = defaultdict(list)
            for i, row in enumerate(meta):
                if row['input_class'] == cls and row['side'] == side:
                    episodes[row['episode']].append(i)
            assert episodes, (cls, side, 'empty sampling stratum')
            pools[cls][side] = dict(sorted(episodes.items()))
    return pools


def draw_batch(pools, generator, size=4):
    import torch
    def choose(values):
        return values[int(torch.randint(len(values), (1,), generator=generator))]
    selected = []
    for _ in range(size):
        cls = choose(CLASSES)
        side = choose(('below', 'above'))
        episode = choose(tuple(pools[cls][side]))
        selected.append(choose(pools[cls][side][episode]))
    return selected


def prepare():
    started = time.monotonic()
    assert not (OUT/'experiment_card.json').exists(), 'Prepared already; never overwrite or relaunch'
    assert not (OUT/'actual_start_receipt.json').exists()
    ast.parse(Path(__file__).read_text())
    syntax=subprocess.run(['bash','-n',str(OUT/'launch.sh')],capture_output=True,text=True,timeout=5)
    assert syntax.returncode==0,syntax.stderr
    save('syntax_check.json',dict(status='passed',time=now(),Python='ast.parse(pilot.py)',shell='bash -n launch.sh',
        production_modules_imported=False,model_forward_calls=0))
    record('syntax_check_passed',read(OUT/'syntax_check.json'))
    import numpy as np
    import torch
    assert sha(REF) == REF_SHA
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
    for name in ('report.md', 'prefix_history_evidence.json', 'recommended_collection_spec_not_executed.json'):
        assert (AUDIT/name).exists()
    registry = read(SOURCE/'shards/contract.json')['groups']['collection']['tasks']
    old_card = read(OLD/'experiment_card.json')
    targets = []
    for target in (40, 41):
        task = read(SOURCE/f'shards/tasks/collection_E{target}/E{target}.json')['task']
        assert task == registry[target] == old_card['targets'][target-40]['task']
        assert task['id'] == f'E{target}' and task['regime'] == 'T3'
        targets.append(dict(task=task, role='training' if target == 40 else 'development_holdout_previously_used_for_selection'))
    assert len(plans()) == 24 and sum(p['teacher_steps'] for p in plans()) == 1680
    # Test only index selection with synthetic metadata, no model/scene/teacher.
    synthetic = [dict(input_class=c, side=s, episode=f'{c}_{s}_{ep}') for c in CLASSES for s in ('below','above') for ep in range(2)]
    pools = pools_for(synthetic)
    g1 = torch.Generator().manual_seed(SEED)
    g2 = torch.Generator().manual_seed(SEED)
    draws1 = [draw_batch(pools, g1) for _ in range(129)]
    draws2 = [draw_batch(pools, g2) for _ in range(129)]
    assert draws1 == draws2 and all(0 <= i < len(synthetic) for b in draws1 for i in b)
    prior=dict(true_error_um=50.,signed_OCT_mm=[0.,0.,.05,1.],encoder_ABCE=[0,0,0,0],state_deg_deg_mm=[0.,0.,20.])
    approaching=dict(true_error_um=49.,signed_OCT_mm=[0.,0.,.049,1.],encoder_ABCE=[0,0,1,0],state_deg_deg_mm=[0.,0.,20.001],actual_velocity_si=[0.,0.,.00002])
    receding=dict(approaching,true_error_um=51.,signed_OCT_mm=[0.,0.,.051,1.],actual_velocity_si=[0.,0.,-.00002])
    stationary=dict(prior,true_error_um=10.,actual_velocity_si=[0.,0.,0.])
    ambiguous=dict(approaching,signed_OCT_mm=prior['signed_OCT_mm'])
    assert input_class(approaching,prior)['input_class']=='approaching'
    assert input_class(receding,prior)['input_class']=='receding'
    assert input_class(stationary,dict(stationary))['input_class']=='settled'
    assert input_class(ambiguous,prior)['input_class']=='ambiguous'
    record('pure_index_and_motion_definition_checks_passed',dict(seed=SEED,sampler_replay_batches=129,
        class_side_episode_window_indices_in_bounds=True,synthetic_motion_checks=4,model_calls=0,simulation_cycles=0))
    display = subprocess.run(['xdpyinfo', '-display', ':1'], env={**os.environ, **DESKTOP}, capture_output=True, text=True, timeout=8)
    assert display.returncode == 0, display.stderr
    assert Path('/run/user/1000/bus').exists() and shutil.which('gnome-terminal')
    assert shutil.disk_usage(OUT).free > 15*1024**3
    card = dict(name=OUT.name, time=now(), question='After paired coverage and balanced first-window sampling changes, does one128-update original neural candidate improve physical first actions?',
        scope='diagnostic only; never a20um neural success claim', init_checkpoint=str(REF), init_sha256=REF_SHA,
        original_pilot=str(OLD/'pilot.py'), old_audit=str(AUDIT/'report.md'), targets=targets,
        split='E40 training / E41 whole-target DEVELOPMENT holdout already used for selection; no dev5/dev20/final test collection',
        branches=plans(), max_teacher_branches=24, max_collection_workers=4,
        setup='Fresh branch only: exact target task, fixed signed C initial offset,2 actual3um G10 ticks at50ms. Record from initial state; never reset velocity/history/previous command within branch. No adaptive search.',
        setup_offset_estimate_um=4.4, preferred_anchor_distances_um=[30,50,70],
        teacher='Unchanged homog_replay_common.make_teacher + original PolicyEvaluator.evaluate_episode feedback acceptance/scales, .2/.1/.03 teacher caps, actual previous velocity',
        observation='Original real RGB2/signed measured OCT/ABC/float32 encoder finite-difference velocity/previous ACCEPTED command. XYZ only setup, metadata, offline scores/groups.',
        direction_definition=dict(actual_C_velocity_min_abs_mm_s=1e-6, radial_deadband_um_s=.02,
            measured='(norm(OCT_t)-norm(OCT_tminus1))*1000/.05',
            independent='(norm(siteXYZ_t-targetXYZ)-norm(siteXYZ_tminus1-targetXYZ))*1000/.05',
            moving='actual|vC|>1e-6 AND measured and independent radial rates both beyond .02um/s in same direction',
            settled='ALL actual encoder counts unchanged over preceding real interval; actual state delta exactly0; independent distance<=20um',
            ambiguous='Everything else: save but do not select as FIRST window. Such actions remain future labels.',
            crossing='Measured signedOCT Z sign change pre/post saved separately as action outcome; never defines incoming motion'),
        data_gate=dict(pairs_per_target_side_min=3, pairing='Only predesignated teacher step0, same target/side/30,50,70level opposite-direction independent branches; no nearest-neighbor search',
            OCT_vector_L2_max_um=5., true_and_measured_anchor_distance_um=[20.,100.],
            opposite_actual_C_signs=True, complete_real_history_min_frames=2, future_contiguous_teacher_actions=16,
            physical_hold='Each target at least200 consecutive real intervals, pre/post independent error<=20um and ALL actual encoder counts unchanged; after that200-run at least16 REAL future teacher actions',
            hold_encoder_count_threshold=0, hold_actual_state_displacement_threshold_rad_rad_m=[0.,0.,0.],
            strict_hold_separate='abs(accepted_C)<=1e-9mm and abs(actual_vC)<=1e-6mm/s; nonzero-command settled separately counted',
            alignments='RGB and lastpost observation real; t/t-1/accepted label and next motor interval; ABC units and actual encoder difference; independent XYZ norm; full16 unpadded same-episode teacher horizon',
            required_classes_per_target=['approaching','receding','settled'], failure='Stop without training; preserve evidence; never loosen gate'),
        windows='All same-episode teacher starts with2 actual images and16 true future labels; ambiguous anchors saved but unselected; no padding/concatenation. Target and episode fully isolated between train/development.',
        train=dict(candidates=1, seed=SEED, optimizer_updates=128, max_attempts=144, AMP_skips_not_updates=True,
            optimizer='AdamW', lr=3e-5, weight_decay=1e-4, batch=4, clip=1., AMP='original FP16 autocast/dynamic GradScaler defaults',
            architecture='Original v5 state_conditioned_temporal_attention_v1 DiT rectified flow, no changes',
            loss='Original rectified_flow_loss all16x3 normalized velocity MSE',
            vision='frozen/eval, all62 tensors unchanged; GroupNorm no BN drift', nonvision_parameters=23164611,
            sampler='Each slot: uniform3classes, uniform2measured target sides, uniform episode within side/class, uniform independent window within episode; replacement, original CPU seed. Long settled trajectories cannot dominate by count.',
            interpretation='Data amount, coverage and sampling distribution change together; not a single-factor causal experiment'),
        offline=dict(K=1, flow_steps=4, exec_steps=1, batch=1, precision='FP32/no autocast',
            seeds=list(SEEDS), flow_loss_seeds=list(FLOW_SEEDS), all_admitted_windows=True,
            groups=['all','approaching','receding','settled','teacher_C_nonzero','teacher_C_zero','below','above','settled_nonzero_command','strict_hold'],
            moving_aggregation='approaching and receding group means weighted equally, regardless of independent window counts',
            production_caps_deg_deg_mm=old_card['model_caps_deg_deg_mm'], flow_loss_secondary_only=True,
            promotion=['E41 equal-motion-weighted C_MAE strictly decreases','E41 receding C_MAE strictly decreases',
                'E41 all nonzero-teacher-C direction accuracy does not decrease','E41 all-window A_MAE does not increase',
                'E41 all-window B_MAE does not increase','E41 settled C_MAE does not increase'],
            tolerance='numeric zero tolerance; all4seeds retained; training reported separately'),
        budgets_seconds=dict(total=900, collection_including_data_gate=600, training=120, offline_before_plus_after=180),
        no_closed_loop=True, no_dev5=True, no_dev20=True, no_publication=True,
        reused_parity=[str(OLD/'short_gate/first_input_parity.json'),str(OLD/'step0_parity.json'),str(OLD/'recovery_report_v1/checkpoint_training_audit.json')],
        new_parity='Only one fresh-data production get_action versus standalone first-action check, no action executed', desktop=DESKTOP)
    save('experiment_card.json', card)
    input_files = {Path(p) for p in read(AUDIT/'input_hashes.json')['files']}
    input_files |= {p for p in AUDIT.iterdir() if p.is_file()}
    input_files |= {Path(__file__), OUT/'launch.sh', OUT/'experiment_card.json', ROOT/'docs/EXPERIMENT_SOP.md'}
    files = {str(p):sha(p) for p in sorted(input_files)}
    # The original audit already enumerated its original-pilot tree; protect it again.
    for p,row in read(AUDIT/'input_hashes.json')['files'].items():
        assert files[p] == row['sha256']
    from scripts.tools.experiment_workflow import scene_and_code
    save('input_fingerprints.json', dict(time=now(), files=files, runtime=scene_and_code()))
    save('preflight.json', dict(status='passed', time=now(), checkpoint_sha256=REF_SHA,
        GPU=torch.cuda.get_device_name(0), executable=sys.executable, DISPLAY_probe_returncode=display.returncode,
        duplicate_execution=False, fixed_branches=24, sampler_index_selftest='passed129x4 exact duplicate CPU-seed replay; no model/scene constructed',
        reviewed_audit='Initial away windows4/100 WERE preserved; original outcome classes misrepresented incoming motion. Tail15 excluded only as anchors, still future labels.',
        preparation_elapsed_s=time.monotonic()-started, no_model_forward_in_preparation=True))
    status('prepared', 'prepared')
    record('preflight_passed', read(OUT/'preflight.json'))


def identity_check():
    proof = read(OUT/'input_fingerprints.json')
    for p,h in proof['files'].items():
        assert sha(p) == h, f'Frozen input changed: {p}'
    from scripts.tools.experiment_workflow import scene_and_code
    assert proof['runtime'] == scene_and_code(), 'Production/scene identity changed'


def collect(spec):
    import numpy as np
    from PIL import Image
    from rcm.scene_v9 import RCMSceneV9
    from lerobot_training.eval_policy import PolicyEvaluator, TASK_TO_DEG_DEG_MM
    from scripts.research.homog_replay_common import make_teacher
    unit = np.asarray([np.pi/180,np.pi/180,.001])
    folder = OUT/'data'/spec['id']
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'rgb').mkdir()
    scene = None
    sensor = None
    rows = []
    actions = []
    previous = np.zeros(3)
    started = time.monotonic()
    try:
        save(folder/'task_status.json', dict(status='running', stage='fresh_scene_setup', pid=os.getpid(), branch=spec))
        scene = RCMSceneV9()
        task = next(t['task'] for t in read(OUT/'experiment_card.json')['targets'] if t['task']['id']==spec['target'])
        world = np.asarray(scene.robot_to_world(np.asarray(task['target_robot_mm'])))
        np.testing.assert_allclose(world, task['target_world_mm'], atol=1e-9, rtol=0)
        scene.set_target_world(world)
        q = scene.target_task(validate=True).copy()
        q[2] += spec['initial_C_offset_m']
        scene.set_task(q)
        scene.enable_motor_execution()
        scene.reset_motor_time(0.)  # Once at FRESH branch setup, never in flight.
        teacher = make_teacher(scene)
        sensor = teacher.sensor
        real_observe = sensor.observe
        cache = {}

        def capture():
            tick_time = float(scene.motor_execution.time_s)
            if cache.get('time') == tick_time:
                return cache['obs'], cache['row']
            sensor.observation_time_s = tick_time
            obs = real_observe()
            rgb = np.asarray(obs['rgb'], np.uint8)
            oct_feature = np.zeros(4, np.float32)
            navigation = obs.get('oct_navigation_depth')
            delta = None if navigation is None else navigation.get('delta_xyz_m')
            if delta is not None and len(delta)==3 and np.isfinite(delta).all():
                oct_feature[:3] = np.asarray(delta,dtype=np.float64)*1000
                oct_feature[3] = 1.
            image_path = folder/'rgb'/f'{len(rows):04d}.png'
            Image.fromarray(rgb).save(image_path, compress_level=1)
            tip = scene.site_mm('needle_tip').copy()
            row = dict(index=len(rows), time_s=tick_time, rgb_path=str(image_path.relative_to(OUT)), rgb_sha256=sha(image_path),
                RGB_shape=list(rgb.shape), RGB_dtype=str(rgb.dtype), actual_sensor_observe_call=len(rows)+1,
                OCT_metadata=obs.get('oct_metadata'),RGB_depth_source=obs.get('oct_depth_source'),
                state_deg_deg_mm=np.asarray(scene.current_task())*TASK_TO_DEG_DEG_MM,
                signed_OCT_mm=oct_feature, actual_velocity_si=np.asarray(scene.motor_execution.actual_velocity).copy(),
                previous_accepted_deg_deg_mm=previous.copy(), encoder_ABCE=list(scene.motor_execution.backend.robot),
                needle_site_world_mm=tip, target_world_mm=world.copy(), true_error_um=float(np.linalg.norm(tip-world)*1000))
            row = json.loads(json.dumps(row, default=default))
            rows.append(row)
            append(folder/'observations.jsonl', row)
            cache.update(time=tick_time, obs=obs, row=row)
            return obs, row

        # One real observation per distinct motor timestamp. The post observation
        # is also the next pre observation, not a second/duplicated frame.
        sensor.observe = lambda: capture()[0]
        capture()
        for raw in spec['prefix_actions_deg_deg_mm']:
            _,pre = capture()
            action = np.asarray(raw)
            scene.apply_prepared_motor(scene.motor_execution.preview(scene.current_task()+action*unit, .05))
            previous = action.copy()
            _,post = capture()
            row = dict(phase='prefix', global_action=len(actions), teacher_step=None,
                pre_observation=pre['index'], post_observation=post['index'], raw_deg_deg_mm=action,
                clipped_deg_deg_mm=action, accepted_deg_deg_mm=action, safety_scale=1., motor_record=scene.motor_execution.last_record)
            row = json.loads(json.dumps(row,default=default))
            actions.append(row)
            append(folder/'actions.jsonl',row)
        prefix_previous = previous.copy()
        pending = {}

        class TeacherEvaluator(PolicyEvaluator):
            def __init__(self):
                self.use_oct = True
                self.measure_inference = False
                self.controller_variant = 'feedback'
                self.fine_evaluator = None
                self.last_terminal_solution = None
                self.terminal_locked_target_world_mm = None

            def reset_buffer(self, seed=None):
                self.previous = prefix_previous.copy()
                self.control_mode_counts = {'teacher_diagnostic':0}

            def record_applied_action(self, action):
                nonlocal previous
                previous = np.asarray(action).copy()
                self.previous = previous.copy()

            def invalidate_action_chunk(self):
                pass

            def get_action(self, rgb, state_abc, oct_feature):
                obs,pre = capture()
                assert oct_feature[3]>.5
                np.testing.assert_array_equal(oct_feature,pre['signed_OCT_mm'])
                np.testing.assert_array_equal(state_abc,pre['state_deg_deg_mm'])
                result = teacher.geometry_action_from_observation(np.asarray(state_abc)*unit, obs,
                    previous_velocity=scene.motor_execution.actual_velocity.copy(), command_q=np.asarray(state_abc)*unit)
                assert result['valid'], f'Teacher observation invalid: {spec["id"]}'
                pending.update(pre=pre['index'], raw=np.asarray(result['action_deg_deg_mm']).copy())
                self.control_mode_counts['teacher_diagnostic'] += 1
                return pending['raw']

        e = TeacherEvaluator()
        def observer(event):
            _,post = capture()  # Real final post RGB/OCT/XYZ is saved as well.
            row = dict(phase='teacher', global_action=len(actions), teacher_step=event['step'],
                pre_observation=pending['pre'], post_observation=post['index'], raw_deg_deg_mm=pending['raw'],
                clipped_deg_deg_mm=event['clipped_action_deg_deg_mm'], accepted_deg_deg_mm=event['accepted_action_deg_deg_mm'],
                safety_scale=event['safety_scale'], motor_record=event['motor_record'])
            row = json.loads(json.dumps(row,default=default))
            actions.append(row)
            append(folder/'actions.jsonl',row)
        e.step_observer = observer
        ep = e.evaluate_episode(scene, world, max_steps=spec['teacher_steps'], error_threshold_mm=.00035,
            control_period_s=.05, action_limit_deg_deg_mm=(.2,.1,.03), max_consecutive_safety_limited=None,
            seed=434300+int(spec['target'][1:]), sensor=sensor, stop_on_success=False, progress_every=80)
        save(folder/'teacher_trajectory.json',ep)
        assert ep['early_stop_reason'] is None and len(actions)==2+spec['teacher_steps']
        assert len(rows)==3+spec['teacher_steps']
        result = dict(status='completed', time=now(), branch=spec, observations=len(rows), teacher_actions=spec['teacher_steps'],
            real_sensor_calls=len(rows), final_true_error_um=rows[-1]['true_error_um'], elapsed_s=time.monotonic()-started,
            final_post_RGB_OCT_saved=True, no_midbranch_resets=True, neural_action_calls=0)
        save(folder/'summary.json',result)
        save(folder/'task_status.json',result,update=True)
        record('teacher_branch_completed',result)
    except BaseException as error:
        failure = dict(status='technical_failure', time=now(), branch=spec, error=repr(error), traceback=traceback.format_exc(),
            saved_observations=len(rows), saved_actions=len(actions), scientific_negative=False)
        save(folder/'failure.json',failure)
        save(folder/'task_status.json',failure,update=True)
        record('teacher_branch_failed',failure)
        raise
    finally:
        if sensor is not None:
            sensor.close()
        if scene is not None:
            scene.close()


def stop_children():
    for child,stream,name in CHILDREN:
        if child.poll() is None:
            try:
                os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
    for child,stream,name in CHILDREN:
        try:
            child.wait(timeout=4)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL)
            child.wait(timeout=3)
        if not stream.closed:
            stream.close()


def collect_all():
    queue = sorted(plans(),key=lambda p:(-p['teacher_steps'],p['id']))
    active = []
    complete = []
    try:
        while queue or active:
            while queue and len(active)<4:
                spec = queue.pop(0)
                stream = (OUT/f'worker_{spec["id"]}.log').open('x')
                cmd = [sys.executable,'-E','-B','-u',str(Path(__file__)),'worker','--branch',spec['id']]
                append(OUT/'commands.jsonl',dict(time=now(),command=cmd,branch=spec['id']))
                child = subprocess.Popen(cmd,cwd=ROOT,env=os.environ.copy(),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                item = (child,stream,spec['id'])
                CHILDREN.append(item)
                active.append(item)
            for item in list(active):
                child,stream,name = item
                rc = child.poll()
                if rc is not None:
                    stream.close()
                    active.remove(item)
                    assert rc==0, f'Collection worker failed {name}, exit{rc}; retained worker log/data, no retry'
                    complete.append(read(OUT/'data'/name/'summary.json'))
            if active:
                time.sleep(.1)
    finally:
        stop_children()
    result = dict(status='completed', branches=complete, count=len(complete), max_parallel=4, retries=0)
    save('collection_summary.json',result)
    record('collection_completed',dict(branches=len(complete),teacher_actions=sum(r['teacher_actions'] for r in complete)))


def input_class(pre, prior):
    import numpy as np
    error_delta = (pre['true_error_um']-prior['true_error_um'])/.05
    oct_delta = (np.linalg.norm(np.asarray(pre['signed_OCT_mm'])[:3])-np.linalg.norm(np.asarray(prior['signed_OCT_mm'])[:3]))*1000/.05
    same_encoder = pre['encoder_ABCE']==prior['encoder_ABCE']
    same_state = bool(np.array_equal(pre['state_deg_deg_mm'],prior['state_deg_deg_mm']))
    vC = pre['actual_velocity_si'][2]*1000
    if same_encoder and same_state and pre['true_error_um']<=20:
        cls = 'settled'
    elif abs(vC)>1e-6 and error_delta<-.02 and oct_delta<-.02:
        cls = 'approaching'
    elif abs(vC)>1e-6 and error_delta>.02 and oct_delta>.02:
        cls = 'receding'
    else:
        cls = 'ambiguous'
    return dict(input_class=cls, actual_C_velocity_mm_s=vC, incoming_XYZ_radial_um_s=error_delta,
        incoming_OCT_radial_um_s=float(oct_delta), unchanged_encoders=same_encoder, unchanged_actual_state=same_state)


def data_audit():
    import numpy as np
    unit = np.asarray([np.pi/180,np.pi/180,.001])
    episode_data = {}
    evidence = []
    checks = []
    candidates = []
    holds = []
    def check(name,a,b,tolerance,unit_name):
        a=np.asarray(a);b=np.asarray(b)
        diff=np.abs(a.astype(float)-b.astype(float))
        checks.append(dict(name=name,max_abs_difference=float(diff.max()) if diff.size else 0.,tolerance=tolerance,
            mismatched_elements=int((diff>tolerance).sum()),unit=unit_name,passed=bool(np.all(diff<=tolerance))))
    for spec in plans():
        name=spec['id'];folder=OUT/'data'/name
        obs=[json.loads(s) for s in (folder/'observations.jsonl').read_text().splitlines()]
        actions=[json.loads(s) for s in (folder/'actions.jsonl').read_text().splitlines()]
        assert len(obs)==len(actions)+1==spec['teacher_steps']+3
        assert len([a for a in actions if a['phase']=='teacher'])==spec['teacher_steps']
        check(name+'/clock',np.diff([o['time_s'] for o in obs]),np.full(len(actions),.05),1e-12,'s')
        for j,o in enumerate(obs):
            error=float(np.linalg.norm(np.asarray(o['needle_site_world_mm'])-o['target_world_mm'])*1000)
            check(name+f'/o{j}/XYZ_error',[error],[o['true_error_um']],1e-10,'um')
            assert o['actual_sensor_observe_call']==j+1
            if j:
                check(name+f'/o{j}/previous_command',o['previous_accepted_deg_deg_mm'],actions[j-1]['accepted_deg_deg_mm'],0.,'deg,deg,mm')
                check(name+f'/o{j}/encoder_velocity',o['actual_velocity_si'],
                    (np.asarray(o['state_deg_deg_mm'])-obs[j-1]['state_deg_deg_mm'])*unit/.05,1e-11,'rad,rad,m/s')
            path=OUT/o['rgb_path']
            assert path.exists() and sha(path)==o['rgb_sha256']
        selected=[]
        for j,a in enumerate(actions):
            pre=obs[a['pre_observation']];post=obs[a['post_observation']];m=a['motor_record']
            assert a['global_action']==j and a['pre_observation']==j and a['post_observation']==j+1
            assert m['driver']['variant']=='g10' and m['duration_s']==.05
            check(name+f'/a{j}/start_end',[pre['time_s'],post['time_s']],[m['start_time_s'],m['end_time_s']],0.,'s')
            check(name+f'/a{j}/actual_pre',np.asarray(pre['state_deg_deg_mm'])*unit,m['actual_before_task_rad_rad_m'],1e-11,'rad,rad,m')
            check(name+f'/a{j}/actual_post',np.asarray(post['state_deg_deg_mm'])*unit,m['actual_after_task_rad_rad_m'],1e-11,'rad,rad,m')
            check(name+f'/a{j}/encoder_pre',pre['encoder_ABCE'],m['actual_before_robot_ABCE'],0.,'integer counts')
            check(name+f'/a{j}/encoder_post',post['encoder_ABCE'],m['actual_after_robot_ABCE'],0.,'integer counts')
            check(name+f'/a{j}/post_actual_velocity',post['actual_velocity_si'],m['actual_velocity_rad_rad_m_per_s'],0.,'rad,rad,m/s')
            check(name+f'/a{j}/accepted_request',(np.asarray(pre['state_deg_deg_mm'])+a['accepted_deg_deg_mm'])*unit,m['requested_task_rad_rad_m'],1e-11,'rad,rad,m')
            check(name+f'/a{j}/accepted_scale',np.asarray(a['clipped_deg_deg_mm'])*a['safety_scale'],a['accepted_deg_deg_mm'],1e-12,'deg,deg,mm')
            if a['phase']!='teacher':
                continue
            k=a['teacher_step'];history=j>=1;future=spec['teacher_steps']-k
            cls=input_class(pre,obs[j-1])
            measured_side='below' if pre['signed_OCT_mm'][2]>0 else 'above' if pre['signed_OCT_mm'][2]<0 else 'zero'
            reason='insufficient_future' if future<16 else 'ambiguous_incoming_motion' if cls['input_class']=='ambiguous' else ''
            row=dict(episode=name,target=spec['target'],split='train' if spec['target']=='E40' else 'development_E41',
                teacher_step=k,pre_observation=j,post_observation=j+1,time_s=pre['time_s'],side=measured_side,
                registered_side=spec['side'],level_um=spec['level_um'],designated_anchor=k==0,
                signed_OCT_mm=pre['signed_OCT_mm'],measured_error_um=float(np.linalg.norm(pre['signed_OCT_mm'][:3])*1000),
                true_error_um=pre['true_error_um'],post_true_error_um=post['true_error_um'],
                previous_accepted=pre['previous_accepted_deg_deg_mm'],accepted_label=a['accepted_deg_deg_mm'],raw_teacher=a['raw_deg_deg_mm'],
                raw_minus_accepted=(np.asarray(a['raw_deg_deg_mm'])-a['accepted_deg_deg_mm']).tolist(),
                true_post_minus_pre_um=post['true_error_um']-pre['true_error_um'],
                measured_crossing_outcome=pre['signed_OCT_mm'][2]*post['signed_OCT_mm'][2]<0,
                actual_execution_delta=(np.asarray(post['state_deg_deg_mm'])-pre['state_deg_deg_mm']).tolist(),
                history_complete=history,future_teacher_actions=future,eligible_first_window=not bool(reason),excluded_reason=reason,
                zero_teacher_C=abs(a['accepted_deg_deg_mm'][2])<=1e-9,
                strict_hold=abs(a['accepted_deg_deg_mm'][2])<=1e-9 and abs(cls['actual_C_velocity_mm_s'])<=1e-6,
                settled_nonzero_command=cls['input_class']=='settled' and abs(a['accepted_deg_deg_mm'][2])>1e-9,**cls)
            evidence.append(row);selected.append(row)
            if row['eligible_first_window']:
                assert history and all(o['signed_OCT_mm'][3]>.5 for o in obs[j-1:j+1])
                assert all(actions[z]['phase']=='teacher' for z in range(j,j+16))
                candidates.append(row)
        # Physical hold is on execution intervals, independently of label value.
        run=0;longest=0;first200=None
        for k,a in enumerate(actions[2:]):
            pre=obs[a['pre_observation']];post=obs[a['post_observation']]
            physical=pre['true_error_um']<=20 and post['true_error_um']<=20 and pre['encoder_ABCE']==post['encoder_ABCE'] and pre['state_deg_deg_mm']==post['state_deg_deg_mm']
            run=run+1 if physical else 0
            longest=max(longest,run)
            if run==200 and first200 is None:
                first200=dict(start_teacher_step=k-199,end_teacher_step=k,future_actions_after_200=spec['teacher_steps']-k-1)
        holds.append(dict(episode=name,target=spec['target'],longest_physical_hold=longest,first200=first200,
            passed=bool(first200 and first200['future_actions_after_200']>=16),
            encoder_count_threshold=0, actual_state_threshold=0))
        episode_data[name]=dict(spec=spec,obs=obs,actions=actions,evidence=selected)
    save('raw_state_evidence.json',evidence)
    write_csv('raw_state_evidence.csv',evidence)
    save('all_alignment_checks.json',checks)
    save('hold_gate_evidence.json',holds)
    pair_rows=[]
    for target in ('E40','E41'):
        for side in ('below','above'):
            for level in (30,50,70):
                left=next(r for r in evidence if r['episode']==f'{target}_{side}_{level}_approaching' and r['teacher_step']==0)
                right=next(r for r in evidence if r['episode']==f'{target}_{side}_{level}_receding' and r['teacher_step']==0)
                vector_gap=float(np.linalg.norm(np.asarray(left['signed_OCT_mm'][:3])-right['signed_OCT_mm'][:3])*1000)
                tests=dict(opposite_input_classes=left['input_class']=='approaching' and right['input_class']=='receding',
                    same_measured_side=left['side']==right['side']==side,
                    vector_L2_le5um=vector_gap<=5.,both_valid_first=left['eligible_first_window'] and right['eligible_first_window'],
                    both_true_and_measured_distances_20_100=all(20<=r[key]<=100 for r in (left,right) for key in ('true_error_um','measured_error_um')),
                    opposite_actual_C=left['actual_C_velocity_mm_s']*right['actual_C_velocity_mm_s']<0,
                    real_C_velocity_nonzero=all(abs(r['actual_C_velocity_mm_s'])>1e-6 for r in (left,right)))
                pair_rows.append(dict(target=target,side=side,level_um=level,OCT_vector_L2_um=vector_gap,
                    approaching=left,receding=right,checks=tests,passed=all(tests.values())))
    save('paired_anchor_evidence.json',pair_rows)
    write_csv('paired_anchor_evidence.csv',[dict(target=r['target'],side=r['side'],level_um=r['level_um'],OCT_vector_L2_um=r['OCT_vector_L2_um'],
        approaching_episode=r['approaching']['episode'],receding_episode=r['receding']['episode'],
        approaching_error_um=r['approaching']['true_error_um'],receding_error_um=r['receding']['true_error_um'],
        approaching_vC_mm_s=r['approaching']['actual_C_velocity_mm_s'],receding_vC_mm_s=r['receding']['actual_C_velocity_mm_s'],
        approaching_time_s=r['approaching']['time_s'],receding_time_s=r['receding']['time_s'],passed=r['passed'],checks=r['checks']) for r in pair_rows])
    counts={target:dict(Counter(r['input_class'] for r in candidates if r['target']==target)) for target in ('E40','E41')}
    tests=dict(all_alignments=all(c['passed'] for c in checks),all12_registered_pairs_pass=all(r['passed'] for r in pair_rows),
        pairs3_each_target_side=all(sum(r['passed'] and r['target']==target and r['side']==side for r in pair_rows)>=3 for target in ('E40','E41') for side in ('below','above')),
        hold200_plus16_each_target=all(any(r['target']==target and r['passed'] for r in holds) for target in ('E40','E41')),
        each_class_and_side_nonempty=all(any(r['target']==t and r['side']==s and r['input_class']==c for r in candidates) for t in ('E40','E41') for s in ('below','above') for c in CLASSES),
        whole_target_split=True, no_cross_episode_horizons=True, no_padding=True, real_lastpost_observation=True)
    result=dict(status='passed' if all(tests.values()) else 'failed',time=now(),checks=tests,counts=counts,
        teacher_states=len(evidence),eligible_windows=len(candidates),exclusions=dict(Counter(r['excluded_reason'] for r in evidence if r['excluded_reason'])),
        paired_anchors=len(pair_rows),pair_evidence=str(OUT/'paired_anchor_evidence.json'),hold_evidence=str(OUT/'hold_gate_evidence.json'),
        alignment_check_count=len(checks),alignment_mismatches=sum(c['mismatched_elements'] for c in checks),
        class_counts_include_repeated_draws=False)
    save('raw_data_gate.json',result)
    record('raw_data_gate_'+result['status'],result)
    return result,episode_data,candidates


def write_csv(name,rows):
    with (OUT/name).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['empty'])
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False,default=default) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def build_windows(e,episode_data,candidates):
    import numpy as np
    import torch
    from PIL import Image
    samples=dict(train=[],development_E41=[])
    metas=dict(train=[],development_E41=[])
    max_velocity_difference=np.zeros(3)
    image_checks=0
    for name,data in episode_data.items():
        obs=data['obs'];actions=data['actions'];images={}
        selected=[r for r in candidates if r['episode']==name]
        wanted=sorted({j for row in selected for j in (row['pre_observation']-1,row['pre_observation'])})
        for j in wanted:
            with Image.open(OUT/obs[j]['rgb_path']) as image:
                rgb=np.asarray(image.convert('RGB')).copy()
            assert list(rgb.shape)==obs[j]['RGB_shape'] and rgb.dtype==np.uint8
            tensor=e.preprocess_image(rgb)
            image=Image.fromarray(rgb)
            if not e.direct_224_preprocessing:
                image=image.resize((640,480),Image.Resampling.BILINEAR)
            image=image.resize((224,224),Image.Resampling.BILINEAR)
            independent=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255.
            independent=(independent-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
            torch.testing.assert_close(tensor,independent,rtol=0,atol=0)
            image_checks+=1;images[j]=tensor
        for row in selected:
            j=row['pre_observation'];split=row['split']
            state=torch.tensor([obs[j-1]['state_deg_deg_mm'],obs[j]['state_deg_deg_mm']],dtype=torch.float32)
            velocity=(state[-1]-state[-2])/.05
            sample={KEYS[0]:torch.stack([images[j-1],images[j]]),KEYS[1]:state,
                KEYS[2]:torch.tensor([obs[j-1]['signed_OCT_mm'],obs[j]['signed_OCT_mm']],dtype=torch.float32),
                KEYS[3]:velocity,KEYS[4]:torch.tensor(obs[j]['previous_accepted_deg_deg_mm'],dtype=torch.float32),
                'action':torch.tensor([a['accepted_deg_deg_mm'] for a in actions[j:j+16]],dtype=torch.float32)}
            assert sample['action'].shape==(16,3) and torch.isfinite(sample[KEYS[0]]).all()
            actual=np.asarray(obs[j]['actual_velocity_si'])/np.asarray([np.pi/180,np.pi/180,.001])
            max_velocity_difference=np.maximum(max_velocity_difference,np.abs(velocity.numpy()-actual))
            meta=dict(row,window_index=len(samples[split]),history_observation_indices=[j-1,j],
                action_teacher_steps_inclusive=[row['teacher_step'],row['teacher_step']+15],
                image_paths=[obs[j-1]['rgb_path'],obs[j]['rgb_path']],padding=0)
            samples[split].append(sample);metas[split].append(meta)
    batches={s:{k:torch.stack([r[k] for r in rows]) for k in rows[0]} for s,rows in samples.items()}
    save('window_manifest.json',dict(train=metas['train'],development_E41=metas['development_E41'],
        whole_target_isolation=True,overlap_between_splits=False,all_eligible_anchors=True))
    torch.save(batches,OUT/'real_input_windows.pt')
    save('preprocessed_input_gate.json',dict(status='passed',images_reprocessed_bit_exact=image_checks,
        max_float32_velocity_minus_actual_deg_deg_mm_s=max_velocity_difference,
        normalization='Original checkpoint statistics unchanged; production model applies normalization internally',
        dataset_sha256=sha(OUT/'real_input_windows.pt'),split_window_counts={s:len(metas[s]) for s in metas}))
    record('production_preprocessing_passed',read(OUT/'preprocessed_input_gate.json'))
    return batches,metas


def new_input_parity(e,batch,meta,episode_data):
    import numpy as np
    import torch
    old=original();data=episode_data[meta['episode']];j=meta['pre_observation'];obs=data['obs']
    def real_row(k):
        from PIL import Image
        with Image.open(OUT/obs[k]['rgb_path']) as image:
            rgb=np.asarray(image.convert('RGB')).copy()
        return dict(rgb=rgb,state=obs[k]['state_deg_deg_mm'],oct=obs[k]['signed_OCT_mm'])
    prior=real_row(j-1);current=real_row(j)
    inputs=old.arguments(e,prior,current,obs[j]['previous_accepted_deg_deg_mm'])
    for key,value in zip(KEYS,inputs):
        torch.testing.assert_close(value.cpu(),batch[key][:1],rtol=0,atol=0)
    e.model.eval();e.reset_buffer(seed=SEEDS[0])
    old.seed_history(e,[prior],obs[j]['previous_accepted_deg_deg_mm'])
    action=e.get_action(current['rgb'],np.asarray(current['state']),np.asarray(current['oct'],np.float32))
    with torch.no_grad():
        expected=e.model.sample(*inputs,steps=4,generator=torch.Generator(device='cuda').manual_seed(SEEDS[0]))[0,0].cpu().numpy()
    np.testing.assert_array_equal(action,expected)
    result=dict(status='passed',time=now(),episode=meta['episode'],teacher_step=meta['teacher_step'],sample_seed=SEEDS[0],
        input_max_abs_difference=0.,first_action_max_abs_difference=0.,physical_first_action_deg_deg_mm=action,
        production_clipped_first_action_deg_deg_mm=np.clip(action,-np.asarray(read(OUT/'experiment_card.json')['offline']['production_caps_deg_deg_mm']),np.asarray(read(OUT/'experiment_card.json')['offline']['production_caps_deg_deg_mm'])),
        original_production_entry='PolicyEvaluator.get_action; unchanged graph/K1/flow4/exec1',
        neural_samples=2, executed_motor_commands=0,closed_loop_cycles=0,
        reused_evidence=read(OUT/'experiment_card.json')['reused_parity'])
    save('new_data_input_first_action_parity.json',result)
    record('new_data_parity_passed',result)


def configure_and_parameters(e):
    import torch
    old=original();old.configure_training(e.model)
    rows=[dict(name=n,shape=list(p.shape),count=p.numel(),trainable=p.requires_grad) for n,p in e.model.named_parameters()]
    assert sum(r['count'] for r in rows if r['trainable'])==23164611
    assert sum(r['count'] for r in rows if not r['trainable'])==11373504
    assert not e.model.vision.training
    assert sum(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in e.model.vision.modules())==0
    result=dict(parameters=rows,trainable_count=23164611,frozen_count=11373504,trainable_tensors=sum(r['trainable'] for r in rows),
        vision_eval=True,vision_BN_count=0,vision_GroupNorm_count=sum(isinstance(m,torch.nn.GroupNorm) for m in e.model.vision.modules()))
    save('parameters.json',result)
    record('original_parameter_scope_passed',dict(trainable_parameters=23164611,frozen_parameters=11373504,
        trainable_tensors=result['trainable_tensors'],vision_eval=True,vision_BN_count=0))
    e.model.eval()


def train_once(e,batch,meta):
    import numpy as np
    import torch
    old=original();old.configure_training(e.model)
    model=e.model
    frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith('vision.')}
    buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
    active=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(active,lr=3e-5,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=True)
    pools=pools_for(meta)
    save('training_sampling_pools.json',dict(pools=pools,probability_each_class=1/3,probability_each_side_conditional=.5,
        class_counts={c:dict(windows=sum(r['input_class']==c for r in meta),episodes=len({r['episode'] for r in meta if r['input_class']==c})) for c in CLASSES}))
    rng=torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    updates=0;trace=[];counts=Counter();episodes=Counter()
    PROGRESS.update(training='running',optimizer_updates=0)
    for attempt in range(144):
        if updates==128:
            break
        indices=draw_batch(pools,rng)
        b={k:v[indices].cuda() for k,v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            value=old.loss(model,b)
        assert torch.isfinite(value),'Nonfinite RF loss: technical failure'
        scaler.scale(value).backward();scaler.unscale_(optimizer)
        gradient=torch.nn.utils.clip_grad_norm_(active,1.)
        scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update()
        updated=scaler.get_scale()>=scale_before
        updates+=int(updated)
        PROGRESS['optimizer_updates']=updates
        if updated:
            for k in indices:
                counts[meta[k]['input_class']]+=1;episodes[meta[k]['episode']]+=1
        row=dict(time=now(),attempt=attempt+1,optimizer_updates=updates,optimizer_updated=updated,
            flow_loss=float(value.detach()),gradient_norm=float(gradient) if torch.isfinite(gradient) else None,
            grad_scale=scaler.get_scale(),sample_indices=indices,
            sample_classes=[meta[k]['input_class'] for k in indices],sample_episodes=[meta[k]['episode'] for k in indices])
        append(OUT/'training.jsonl',row);trace.append(row)
        assert not updated or torch.isfinite(gradient),'AMP accepted nonfinite gradient'
        if attempt==0 or updated and updates%16==0:
            print('TRAIN',json.dumps(row),flush=True)
            status('training',optimizer_updates=updates,attempt=attempt+1)
    assert updates==128,'128 optimizer updates not completed within144 attempts; no extension'
    for k,v in model.state_dict().items():
        if k in frozen:
            torch.testing.assert_close(v.cpu(),frozen[k],rtol=0,atol=0)
    for k,v in model.named_buffers():
        torch.testing.assert_close(v.cpu(),buffers[k],rtol=0,atol=0)
    checkpoint={k:v for k,v in e.checkpoint.items() if k not in ('optimizer_state_dict','scheduler_state_dict','scaler_state_dict','rng_state','model_state_dict')}
    params=read(OUT/'parameters.json')
    freeze=dict(vision_frozen=True,nonvision_all_trainable=True,trainable_count=23164611,
        trainable_parameters=[r['name'] for r in params['parameters'] if r['trainable']],
        frozen_parameters=[r['name'] for r in params['parameters'] if not r['trainable']])
    checkpoint.update(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),scaler_state_dict=scaler.state_dict(),
        scheduler_state_dict=None,training_module=OUT.name,train_loss=trace[-1]['flow_loss'],val_loss=None,
        dataset_sha256=read(OUT/'preprocessed_input_gate.json')['dataset_sha256'],epoch=0,
        progress=dict(global_step=128,optimizer_updates=128),freeze_config=freeze,
        sampling_contract=dict(mode='class_side_episode_window_uniform_replacement',batch_size=4,seed=SEED,train_target='E40',development_target='E41'),
        config=copy.deepcopy(e.checkpoint['config']))
    checkpoint['config'].update(require_temporal_bias_only=False,freeze_config=freeze,lr=3e-5,backbone_lr=3e-5,attention_lr=3e-5,
        lr_schedule='constant',seed=SEED,init_from=str(REF),output_dir=str(OUT),pilot_optimizer_updates=128,pilot_dataset=str(OUT/'real_input_windows.pt'))
    torch.save(checkpoint,OUT/'after.pth')
    steps=[int(v['step']) for v in optimizer.state.values()]
    assert min(steps)==max(steps)==128
    # Reconstruct recorded draw indices once from the independent seed stream.
    replay=torch.Generator().manual_seed(SEED)
    assert all(draw_batch(pools,replay)==r['sample_indices'] for r in trace)
    result=dict(status='completed',time=now(),optimizer_updates=128,attempts=len(trace),AMP_skips=len(trace)-128,
        optimizer_state_steps_min=min(steps),optimizer_state_steps_max=max(steps),seed=SEED,
        before_sha256=REF_SHA,after_sha256=sha(OUT/'after.pth'),frozen_vision_tensors_bit_exact=len(frozen),
        all_normalization_buffers_unchanged=True,successful_draw_class_counts=dict(counts),
        successful_draw_class_proportions={c:counts[c]/512 for c in CLASSES},successful_draw_episode_counts=dict(episodes),
        independent_collected_windows=len(meta),draws_are_not_independent_new_data=True,sampler_trace_replay='exact',
        initial_flow_loss=trace[0]['flow_loss'],final_flow_loss=trace[-1]['flow_loss'])
    save('training_result.json',result)
    PROGRESS.update(training='completed',optimizer_updates=128)
    record('training_completed',result)
    model.eval()


def offline(e,batch,meta,label,split):
    import numpy as np
    import torch
    old=original()
    e.model.eval()
    folder=OUT/'offline'/label
    folder.mkdir(parents=True,exist_ok=True)
    teacher=batch['action'][:,0].numpy()
    predictions=[];flows=[];rows=[]
    caps=np.asarray(read(OUT/'experiment_card.json')['offline']['production_caps_deg_deg_mm'])
    with torch.no_grad():
        for j,window in enumerate(meta):
            b={k:v[j:j+1].cuda() for k,v in batch.items()}
            draws=[];flow=[]
            for seed in SEEDS:
                p=e.model.sample(*(b[k] for k in KEYS),steps=4,generator=torch.Generator(device='cuda').manual_seed(seed))[0,0].cpu().numpy()
                assert np.isfinite(p).all()
                draws.append(p)
                clipped=np.clip(p,-caps,caps)
                rows.append(dict(window=j,episode=window['episode'],teacher_step=window['teacher_step'],input_class=window['input_class'],
                    side=window['side'],seed=seed,raw_A_deg=float(p[0]),raw_B_deg=float(p[1]),raw_C_mm=float(p[2]),
                    production_A_deg=float(clipped[0]),production_B_deg=float(clipped[1]),production_C_mm=float(clipped[2]),
                    teacher_A_deg=float(teacher[j,0]),teacher_B_deg=float(teacher[j,1]),teacher_C_mm=float(teacher[j,2])))
            for seed in FLOW_SEEDS:
                torch.manual_seed(seed)
                flow.append(float(old.loss(e.model,b)))
            predictions.append(draws);flows.append(flow)
            if j%128==0:
                print('OFFLINE',label,split,j,'/',len(meta),flush=True)
    raw=np.asarray(predictions);clipped=np.clip(raw,-caps,caps)
    masks=dict(all=np.ones(len(meta),bool))
    masks.update({c:np.asarray([m['input_class']==c for m in meta]) for c in CLASSES})
    masks.update({s:np.asarray([m['side']==s for m in meta]) for s in ('below','above')})
    masks['teacher_C_nonzero']=np.abs(teacher[:,2])>1e-9
    masks['teacher_C_zero']=~masks['teacher_C_nonzero']
    masks['settled_nonzero_command']=masks['settled']&masks['teacher_C_nonzero']
    masks['strict_hold']=np.asarray([m['strict_hold'] for m in meta])
    groups={}
    for name,mask in masks.items():
        signs=mask&masks['teacher_C_nonzero']
        groups[name]=dict(windows=int(mask.sum()),nonzero_teacher_C_windows=int(signs.sum()),
            physical_first_action_MAE_deg_deg_um=(np.abs(clipped[mask]-teacher[mask,None]).mean((0,1))*[1,1,1000]).tolist() if mask.any() else None,
            raw_MAE_deg_deg_um=(np.abs(raw[mask]-teacher[mask,None]).mean((0,1))*[1,1,1000]).tolist() if mask.any() else None,
            C_direction_accuracy=float((np.sign(clipped[signs,:,2])==np.sign(teacher[signs,None,2])).mean()) if signs.any() else None,
            per_seed_physical_MAE_deg_deg_um=(np.abs(clipped[mask]-teacher[mask,None]).mean(0)*[1,1,1000]).tolist() if mask.any() else None,
            per_seed_C_direction_accuracy=(np.sign(clipped[signs,:,2])==np.sign(teacher[signs,None,2])).mean(0).tolist() if signs.any() else None,
            independent_episodes=len({meta[i]['episode'] for i in np.flatnonzero(mask)}))
    weighted=(np.asarray(groups['approaching']['physical_first_action_MAE_deg_deg_um'])+groups['receding']['physical_first_action_MAE_deg_deg_um'])/2
    np.savez_compressed(folder/f'{split}_predictions.npz',raw=raw,production_clipped=clipped,teacher=teacher,
        flow_loss=np.asarray(flows),seeds=np.asarray(SEEDS),flow_seeds=np.asarray(FLOW_SEEDS))
    write_csv(f'offline_{label}_{split}_per_window_seed.csv',rows)
    result=dict(status='completed',time=now(),model=label,split=split,groups=groups,
        equal_motion_weighted_MAE_deg_deg_um=weighted.tolist(),flow_loss=float(np.mean(flows)),
        equal_motion_weighted_C_direction_accuracy=(groups['approaching']['C_direction_accuracy']+groups['receding']['C_direction_accuracy'])/2,
        sample_seeds=list(SEEDS),K=1,flow_steps=4,exec_steps=1,batch=1,precision='FP32',all_seeds_retained=True)
    save(folder/f'{split}_metrics.json',result)
    record('offline_completed',dict(model=label,split=split,windows=len(meta),equal_motion_weighted_MAE=weighted,
        flow_loss=result['flow_loss'],metrics=str(folder/f'{split}_metrics.json')))
    return result


def promotion(before,after):
    checks=[]
    def add(name,x,y,strict=False,higher=False):
        passed=y>x if higher and strict else y>=x if higher else y<x if strict else y<=x
        checks.append(dict(name=name,before=x,after=y,passed=bool(passed),required='after > before' if higher and strict else 'after >= before' if higher else 'after < before' if strict else 'after <= before'))
    add('E41_equal_motion_C_MAE_um',before['equal_motion_weighted_MAE_deg_deg_um'][2],after['equal_motion_weighted_MAE_deg_deg_um'][2],strict=True)
    add('E41_receding_C_MAE_um',before['groups']['receding']['physical_first_action_MAE_deg_deg_um'][2],after['groups']['receding']['physical_first_action_MAE_deg_deg_um'][2],strict=True)
    add('E41_nonzero_teacher_C_direction',before['groups']['teacher_C_nonzero']['C_direction_accuracy'],after['groups']['teacher_C_nonzero']['C_direction_accuracy'],higher=True)
    for i,axis in enumerate(('A','B')):
        add(f'E41_all_{axis}_MAE_deg',before['groups']['all']['physical_first_action_MAE_deg_deg_um'][i],after['groups']['all']['physical_first_action_MAE_deg_deg_um'][i])
    add('E41_settled_C_MAE_um',before['groups']['settled']['physical_first_action_MAE_deg_deg_um'][2],after['groups']['settled']['physical_first_action_MAE_deg_deg_um'][2])
    result=dict(status='passed' if all(c['passed'] for c in checks) else 'failed',time=now(),checks=checks,
        numeric_tolerance=0.,E41_is_development_not_independent_test=True,closed_loop_allowed=False,flow_loss_not_gate=True)
    save('offline_gate.json',result)
    PROGRESS['offline_improvement']=result['status']
    record('offline_gate_'+result['status'],result)
    return result


def alarm(signum,frame):
    raise TimeoutError('Registered stage or total runtime budget exhausted; no retry/extension')


@contextmanager
def budget(stage,seconds):
    started=time.monotonic()
    allowed=min(seconds,DEADLINE-started)
    if allowed<=0:
        raise TimeoutError('Total900s budget exhausted')
    signal.setitimer(signal.ITIMER_REAL,allowed)
    try:
        yield
    finally:
        elapsed=time.monotonic()-started
        append(OUT/'budget_ledger.jsonl',dict(time=now(),stage=stage,elapsed_s=elapsed,registered_cap_s=seconds))
        left=DEADLINE-time.monotonic()
        signal.setitimer(signal.ITIMER_REAL,left if left>0 else 0.)


def finish(state_name,**extra):
    result=dict(status=state_name,time=now(),**PROGRESS,diagnostic_only=True,neural_20um_claim=False,
        development_not_independent_test=True,distribution_and_data_both_changed=True,unrequested_retry=False)
    result.update(extra)
    result['artifacts']={name:str(OUT/name) for name in ('data_gate.json','raw_data_gate.json','training_result.json',
        'offline_gate.json','input_hash_verification.json') if (OUT/name).exists()}
    if (OUT/'raw_data_gate.json').exists():
        result['actual_data_counts']=read(OUT/'raw_data_gate.json')['counts']
    if (OUT/'training_result.json').exists():
        result['training_result']=read(OUT/'training_result.json')
    if RUN_STARTED is not None:
        result['total_runtime_including_preflight_s']=PRIOR_BUDGET+time.monotonic()-RUN_STARTED
    if (OUT/'budget_ledger.jsonl').exists():
        result['budget_accounting']=[json.loads(line) for line in (OUT/'budget_ledger.jsonl').read_text().splitlines()]
    save('summary.json',result)
    lines=['# 配对末端数据与原神经最小训练验收','',
        f'**数据门：{result["data_gate"]}；训练：{result["training"]}（真实optimizer updates={result["optimizer_updates"]}）；离线改善：{result["offline_improvement"]}。**','',
        f'流程状态：{state_name}。仅诊断；没有执行神经闭环、dev5/dev20或20µm神经成功验收。',
        'E40为训练登记，E41为已经参与选择的整目标开发留出，不是全新独立测试。原网络、教师、驱动未改；初始化只用6cc389df原权重。','',
        '上轮history假设已否定：初始4/100远离输入被保留，旧outcome分类掩盖输入方向；尾15步不能作为首窗但仍为未来监督。本轮分别记录输入实际方向与执行后结果。','',
        '配对仅使用预注册24条真实前缀的首个教师时刻，30/50/70µm；门禁不放宽、不搜索。每时间戳真实调用原传感器；静止画面像素可以相同，但没有复制帧充当未来。',
        '物理保持是独立XYZ误差≤20µm且所有实际编码器不动；非零受理C的停稳单列，不能冒称原strict hold，也不改教师标签。','',
        '训练首窗按接近/远离/停稳各1/3，再按侧、episode均衡，有放回。重复抽样不计作独立采集。数据量、覆盖和采样均发生改变，不能宣称单因素因果。','']
    if 'actual_data_counts' in result:
        lines+=['实际有效首动作窗数量：'+json.dumps(result['actual_data_counts'],ensure_ascii=False),'']
    if 'reason' in result:
        lines+=['停止原因：'+result['reason'],'']
    if (OUT/'offline_gate.json').exists():
        gate=read(OUT/'offline_gate.json')
        lines+=['|E41开发留出门禁|before|after|通过|','|---|---:|---:|---|']
        for c in gate['checks']:
            lines.append(f'|{c["name"]}|{c["before"]:.10g}|{c["after"]:.10g}|{c["passed"]}|')
        lines+=['','训练及开发各组ABC MAE/C方向准确率、所有4seed及flow loss独立存于offline；flow loss不作为晋级门。']
    if 'error' in result:
        lines+=['','技术/预算失败：'+result['error'],'不把工程失败写成神经科学负结果；保留failure.json及逐步证据，不重试。']
    lines+=['','## 产物','',
        '- 数据/时间/配对/保持：data/、raw_state_evidence.csv/json、paired_anchor_evidence.csv/json、hold_gate_evidence.json、all_alignment_checks.json。',
        '- 窗口与身份：window_manifest.json、real_input_windows.pt、input_fingerprints.json、input_hash_verification.json。',
        '- 神经链路：new_data_input_first_action_parity.json、parameters.json、before.pth、after.pth、training.jsonl、training_result.json。',
        '- 离线：offline/及offline_*_per_window_seed.csv；空组显式0/null。',
        '- 启动与日志：experiment_card.json、gnome_launch_receipt.json、actual_start_receipt.json、terminal.log、TEST_LOG.md、test_log_receipts.jsonl。','',
        '失败阶段后未生成的产物保持缺失；以task_status/summary真实状态为准。每个报告阶段的TEST_LOG写入都在锁内fsync并立即按字节读回。',
        '只验证软件与MuJoCo，不代表临床或真实硬件结论。离线通过也只建议下一轮，不执行闭环；离线负结果不证明原架构不可能。','',
        '机器可读完整结果：','```json',json.dumps(result,ensure_ascii=False,indent=2,default=default),'```']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    record('workflow_finished',dict(status=state_name,**PROGRESS,summary=str(OUT/'summary.json'),report=str(OUT/'report.md')))
    status('finished','completed' if not state_name.startswith(('technical','budget','interrupted')) else state_name)
    print('FINAL_RESULT_PATH',OUT/'summary.json',flush=True)


def send_start_receipt(receipt):
    try:
        fd=os.open(OUT/'startup_ack.fifo',os.O_WRONLY|os.O_NONBLOCK)
        os.write(fd,(json.dumps(receipt)+'\n').encode());os.close(fd)
    except OSError:
        pass


def run():
    global DEADLINE,RUN_STARTED,PRIOR_BUDGET
    with (OUT/'.execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/'actual_start_receipt.json').exists(),'Refuse duplicate execution'
        receipt=dict(time=now(),status='actual_process_started',pid=os.getpid(),ppid=os.getppid(),stage='identity_preflight',
            argv=sys.argv,host=os.uname().nodename,desktop={k:os.environ.get(k) for k in DESKTOP},
            GNOME_TERMINAL_SCREEN=os.environ.get('GNOME_TERMINAL_SCREEN'),not_scientific_success=True)
        save('actual_start_receipt.json',receipt)
        send_start_receipt(receipt)
        prior=read(OUT/'preparation_budget.json')['elapsed_s']+read(OUT/'prelaunch_budget.json')['elapsed_s']
        PRIOR_BUDGET=prior
        RUN_STARTED=time.monotonic()
        DEADLINE=time.monotonic()+900-prior
        signal.signal(signal.SIGALRM,alarm)
        signal.signal(signal.SIGTERM,lambda s,f: (_ for _ in ()).throw(InterruptedError('SIGTERM')))
        signal.setitimer(signal.ITIMER_REAL,max(.001,900-prior))
        stage='identity'
        try:
            status(stage)
            identity_check()
            import torch
            torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
            assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
            record('actual_execution_started',receipt)
            stage='collection_and_data_gate';status(stage)
            with budget(stage,600):
                collect_all()
                raw_gate,episode_data,candidates=data_audit()
                if raw_gate['status']!='passed':
                    PROGRESS['data_gate']='failed'
                    save('data_gate.json',dict(status='failed',time=now(),raw=raw_gate,preprocessing='not_run',new_entry_parity='not_run'))
                    raise DataGateRejected('Preregistered raw pair/hold/alignment gate failed; no training')
                PROGRESS['data_gate']='raw_passed_input_validation_running'
                status('data_input_validation')
                e=original().evaluator(REF)
                assert e.n_obs_steps==2 and e.n_action_steps==16 and e.is_dit_rectified_flow
                assert not e.fine_evaluator and not e.guided_oct and not e.model.oct_symlog_enabled and not e.model.use_local_oct_branch
                batches,meta=build_windows(e,episode_data,candidates)
                new_input_parity(e,batches['train'],meta['train'][0],episode_data)
                configure_and_parameters(e)
                shutil.copyfile(REF,OUT/'before.pth')
                assert sha(OUT/'before.pth')==REF_SHA
                data_gate=dict(status='passed',time=now(),raw=raw_gate,preprocessing=read(OUT/'preprocessed_input_gate.json'),
                    new_entry_parity=read(OUT/'new_data_input_first_action_parity.json'),all_evidence_saved=True)
                save('data_gate.json',data_gate)
                PROGRESS['data_gate']='passed'
                record('data_gate_passed',dict(windows=raw_gate['counts'],data_gate=str(OUT/'data_gate.json')))
                del episode_data
            stage='offline_before';status(stage)
            offline_started=time.monotonic()
            with budget(stage,180):
                before={s:offline(e,batches[s],meta[s],'before',s) for s in batches}
            offline_used=time.monotonic()-offline_started
            stage='training';status(stage)
            with budget(stage,120):
                train_once(e,batches['train'],meta['train'])
            del e;torch.cuda.empty_cache()
            stage='offline_after';status(stage)
            with budget(stage,180-offline_used):
                e=original().evaluator(OUT/'after.pth')
                assert e.model.vision.training is False
                save('after_production_loader_receipt.json',dict(status='passed',time=now(),model_sha256=sha(OUT/'after.pth'),
                    original_PolicyEvaluator=True,original_checkpoint_schema=True,K=1,flow4=True,exec1=True))
                after={s:offline(e,batches[s],meta[s],'after',s) for s in batches}
            gate=promotion(before['development_E41'],after['development_E41'])
            stage='final_identity';status(stage)
            identity_check()
            save('input_hash_verification.json',dict(status='passed',time=now(),all_frozen_files_unchanged=True,
                input_files=len(read(OUT/'input_fingerprints.json')['files']),production_teacher_driver_and_prior_results_unchanged=True))
            finish('completed_offline_passed' if gate['status']=='passed' else 'completed_offline_failed',
                offline_gate=gate,teacher_20um_is_data_acceptance_only=True,next_step='只建议下一轮；本轮不做闭环')
            return 0
        except DataGateRejected as error:
            signal.setitimer(signal.ITIMER_REAL,0.)
            finish('completed_data_gate_failed',reason=str(error),raw_data_gate=read(OUT/'raw_data_gate.json'))
            return 0
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL,0.)
            stop_children()
            if PROGRESS['data_gate']!='passed':
                PROGRESS['data_gate']='incomplete_technical_failure'
            failure=dict(time=now(),stage=stage,error=repr(error),traceback=traceback.format_exc(),
                scientific_negative=False,no_unconditional_retry=True,**PROGRESS)
            save('failure.json',failure)
            record('workflow_failure',failure)
            finish('budget_failure' if isinstance(error,TimeoutError) else 'interrupted' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'technical_failure',
                failed_stage=stage,error=repr(error),scientific_negative=False)
            traceback.print_exc()
            return 1
        finally:
            signal.setitimer(signal.ITIMER_REAL,0.)


def launch():
    launch_started=time.monotonic()
    with (OUT/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/'gnome_launch_request.json').exists() and not (OUT/'actual_start_receipt.json').exists(),'Refuse second launch'
        assert read(OUT/'preflight.json')['status']=='passed'
        identity_check()
        os.mkfifo(OUT/'startup_ack.fifo',0o600)
        fd=os.open(OUT/'startup_ack.fifo',os.O_RDWR|os.O_NONBLOCK)
        cmd=['gnome-terminal','--window','--title=Paired terminal original-v5 minifit E40 E41 DEVELOPMENT 128updates',
            '--working-directory='+str(ROOT),'--','bash',str(OUT/'launch.sh')]
        save('gnome_launch_request.json',dict(time=now(),command=cmd,desktop=DESKTOP))
        save('prelaunch_budget.json',dict(elapsed_s=time.monotonic()-launch_started+1.,
            includes_launch_IPC_and_process_start_reserve_s=1.,overlapping_startup_ack_wait_not_double_counted=True))
        try:
            result=subprocess.run(cmd,env={**os.environ,**DESKTOP},capture_output=True,text=True,timeout=15)
            receipt=dict(time=now(),command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
            save('gnome_launch_receipt.json',receipt)
            if result.returncode:
                status('launch_failed','technical_failure',launch_error=result.stderr)
                record('gnome_launch_failed',dict(**receipt,no_background_fallback=True))
                finish('technical_failure',failed_stage='gnome_launch',error=result.stderr,no_background_fallback=True)
                return 1
            selector=selectors.DefaultSelector();selector.register(fd,selectors.EVENT_READ)
            ready=selector.select(20)
            message=os.read(fd,65536).decode() if ready else ''
            selector.close()
            ack=dict(time=now(),one_startup_ack_only=True,no_phase_polling=True,
                actual_start=json.loads(message.splitlines()[0]) if message else None,timed_out=not bool(message))
            save('launch_startup_acknowledgement.json',ack)
            record('gnome_actual_start_acknowledgement',ack)
            print('ACTUAL_START_RECEIPT',json.dumps(ack,ensure_ascii=False),flush=True)
            return 0 if message else 1
        except BaseException as error:
            save('launch_failure.json',dict(time=now(),error=repr(error),traceback=traceback.format_exc(),no_background_fallback=True))
            record('gnome_launch_exception',dict(error=repr(error),no_background_fallback=True))
            if not (OUT/'summary.json').exists():
                finish('technical_failure',failed_stage='gnome_launch',error=repr(error),no_background_fallback=True)
            raise
        finally:
            os.close(fd)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('prepare','launch','run','worker'))
    p.add_argument('--branch')
    a=p.parse_args()
    if a.mode=='prepare':
        started=time.monotonic()
        try:
            prepare()
        except BaseException as error:
            status('preparation_failed','technical_failure',error=repr(error))
            record('preparation_failed',dict(error=repr(error),traceback=traceback.format_exc()))
            raise
        finally:
            # Include earlier shell resource/file preflight conservatively as1s.
            save('preparation_budget.json',dict(elapsed_s=time.monotonic()-started+1.,includes_earlier_shell_preflight_reserve_s=1.))
        return 0
    if a.mode=='launch':
        return launch()
    if a.mode=='run':
        return run()
    import torch
    torch.set_num_threads(1)
    collect(next(r for r in plans() if r['id']==a.branch))
    return 0


if __name__=='__main__':
    sys.exit(main())
