"""One original-v5 neural terminal pilot. No architecture/loss/controller search."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path('/home/him1/mujoco_bc/rcm_4090_deploy')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
REF = ROOT/'outputs/homog_label_fix_20261124_v5/homog_label_fix_20261124_v5_phase1_A/step_004000.pth'
REF_SHA = '6cc389df3786f1f34c43cf65d6dc17fe910ab342fae8d84c8e37cea6a7db6c11'
SOURCE = ROOT/'data/homog_clean_replay_20261123_mixed_1600_600_12_hold_v2'
OLD = ROOT/'runs/20261125/learned_terminal_recovery_pilot_v1/recovery_v1/run'
OLD_CODE = ROOT/'runs/20261125/learned_terminal_recovery_pilot_v1/pilot.py'
KINDS = ('below_recede', 'above_recede')
SEEDS = (92001, 92002, 92003, 92004)
FLOW_SEEDS = (91001, 91002, 91003, 91004)
TRAIN_SEED = 20261126
DESKTOP = dict(DISPLAY=':1', DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus', XDG_RUNTIME_DIR='/run/user/1000')
INPUT_KEYS = ('observation.image', 'observation.state', 'observation.oct', 'observation.velocity', 'observation.previous_action')


def now():
    return datetime.now().astimezone().isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    if not path.is_absolute():
        path = OUT/path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=default)+'\n')
    tmp.replace(path)


def record(stage, result):
    row = dict(time=now(), experiment=OUT.name, stage=stage, result=result, argv=sys.argv, output=str(OUT))
    line = '\n- '+json.dumps(row, ensure_ascii=False, allow_nan=False, default=default)+'; MuJoCo simulation only.\n'
    for path in (OUT/'TEST_LOG.md', ROOT/'TEST_LOG.md'):
        with path.open('ab') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line.encode())
    print(json.dumps(row, ensure_ascii=False, default=default), flush=True)


def state(stage, **kwargs):
    save('workflow_status.json', dict(time=now(), pid=os.getpid(), stage=stage, **kwargs))


def ack(value):
    save('startup_first_action_receipt.json', value)
    try:
        fd = os.open(OUT/'startup_ack.fifo', os.O_WRONLY | os.O_NONBLOCK)
        os.write(fd, (json.dumps(value, default=default)+'\n').encode())
        os.close(fd)
    except OSError:
        pass


def evaluator(path=REF):
    from lerobot_training.eval_policy import PolicyEvaluator
    return PolicyEvaluator(str(path), device='cuda', num_action_samples=1,
        n_action_execute_steps=1, flow_integration_steps=4, controller_variant='feedback')


def frozen_sources():
    from scripts.tools.experiment_workflow import scene_and_code
    return scene_and_code()


def prepare():
    import numpy as np
    import torch
    assert not (OUT/'experiment_card.json').exists(), 'Existing card: refuse duplicate preparation'
    assert sha(REF) == REF_SHA
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
    spec = read(SOURCE/'shards/contract.json')['groups']['collection']
    meta = read(SOURCE/'metadata.json')
    split = read(ROOT/'configs/target_split_manifest.json')
    forbidden = np.asarray([r['point_mm'] for k in ('development_targets', 'final_test_targets') for r in split[k]])
    dev = np.asarray([r.get('reachable_target_robot_mm', r['point_mm']) for r in read(ROOT/'configs/v12_reachable_development_targets_0p35um.json')['targets']])
    targets = []
    files = [REF, SOURCE/'metadata.json', SOURCE/'shards/contract.json', OLD_CODE,
        ROOT/'runs/20261125/terminal_tinyfit_code_v1/tinyfit.py',
        ROOT/'runs/20261125/train_direction_pair_code_v1/paired_replay.py',
        ROOT/'runs/20261125/train_direction_pair_code_v1/paired_select.py',
        ROOT/'configs/target_split_manifest.json', ROOT/'configs/v12_reachable_development_targets_0p35um.json',
        ROOT/'lerobot_training/eval_policy.py', ROOT/'lerobot_training/dit_rectified_flow.py',
        ROOT/'lerobot_training/dit_state_attention.py', ROOT/'lerobot_training/dit_experience_replay.py',
        ROOT/'rcm/visual_servo.py', ROOT/'rcm/motor_execution.py', ROOT/'rcm/scene_v9.py',
        ROOT/'scripts/research/homog_replay_common.py', Path(__file__), OUT/'launch.sh']
    for i in (40, 41):
        path = SOURCE/f'shards/tasks/collection_E{i:02d}/E{i:02d}.json'
        t = read(path)['task']
        assert t == spec['tasks'][i] and t['id'] == f'E{i:02d}' and t['regime'] == 'T3'
        np.testing.assert_array_equal(t['target_robot_mm'], meta['episodes'][i]['target_robot_mm'])
        point = np.asarray(t['target_robot_mm'])
        gap = float(np.linalg.norm(forbidden-point, axis=1).min()*1000)
        dev_gap = float(np.linalg.norm(dev-point, axis=1).min()*1000)
        assert min(gap, dev_gap)>1
        targets.append(dict(task=t, role='train' if i == 40 else 'whole_target_heldout_local_development',
            registry=str(SOURCE/'shards/contract.json')+'#groups.collection.tasks',
            nearest_legacy_dev_or_final_um=gap, nearest_reachable_dev_um=dev_gap))
        files.append(path)
    with np.load(OLD/'training_arrays.npz', allow_pickle=False) as a:
        ridge = {k:list(a[k].shape) for k in a.files}
    old_rows = []
    for i in (40, 41):
        for kind in KINDS:
            path = OLD/f'trajectories/train_E{i:02d}_{kind}.json'
            # E40 below_recede came from an older 20-tick paired continuation.
            if not path.exists():
                path = ROOT/'runs/20261125/output_head_sanity_v1/recovery_v2/paired/recede_teacher_raw.json'
            row = read(path)
            old_rows.append(dict(path=str(path), sha256=sha(path), keys=list(row), trajectory_keys=list(row['trajectory']),
                rgb_payload_or_paths_present=any(k in row or k in row['trajectory'] for k in ('rgb', 'images', 'image_paths', 'observation.image')),
                usable_as_complete_neural_sample=False, reason='No two-frame RGB payload/history; first RGB hash cannot reconstruct pixels'))
            files.append(path)
    assert not any(r['rgb_payload_or_paths_present'] for r in old_rows)
    e = evaluator()
    assert e.n_obs_steps == 2 and e.n_action_steps == 16
    assert e.is_dit_rectified_flow and not e.fine_evaluator and not e.guided_oct
    assert e.model.oct_symlog_enabled is False and e.model.use_local_oct_branch is False
    checkpoint = e.checkpoint
    parameters = []
    for name, p in e.model.named_parameters():
        parameters.append(dict(name=name, shape=list(p.shape), count=p.numel(), trainable=not name.startswith('vision.')))
    save('parameters.json', dict(parameters=parameters,
        trainable_count=sum(p['count'] for p in parameters if p['trainable']),
        frozen_count=sum(p['count'] for p in parameters if not p['trainable']),
        old_freeze_config=checkpoint['freeze_config'],
        new_freeze='All original vision.* weights frozen and vision.eval(); every other existing parameter trainable',
        vision_normalization_modules={type(m).__name__:sum(type(x)==type(m) for x in e.model.vision.modules())
            for m in e.model.vision.modules() if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm))}))
    from lerobot_training.eval_policy import DEFAULT_ACTION_LIMIT_DEG_DEG_MM
    card = dict(name=OUT.name, created=now(), hypothesis='Original v5 DiT-RF can learn local terminal actions after same-observation executable moving recovery supervision on both OCT sides.',
        init_checkpoint=str(REF), init_sha256=REF_SHA, architecture=checkpoint['architecture'], policy_type=checkpoint['policy_type'],
        checkpoint_schema=checkpoint['schema_version'], targets=targets, train_target='E40', heldout_target='E41',
        segments_per_target=list(KINDS), teacher_cycles_each=40, prefix_cycles=26,
        observations='Two true RGB frames + same-call signed measured OCT + actual encoder ABC; original float32 state finite-difference velocity and previous accepted action, as production get_action',
        setup='Existing pilot.make_start: target-C +/-90um then24 approach3um +2 reverse3um G10 ticks; truth only for setup/scoring, never model input',
        data_reuse='Original prefix/teacher interfaces reused; missing RGB requires four fresh40-tick matched teacher continuations; no ridge X/Y used as visual data',
        windows='horizon16 action labels, RGB2 and history continuous within each40-tick segment; final prefix frames supply prior observation, no segment concatenation',
        teacher='homog_replay_common.make_teacher.geometry_action_from_observation, actual previous velocity and same-call strict RGBD; labels are production accepted actions',
        teacher_caps_deg_deg_mm=[.2,.1,.03], model_caps_deg_deg_mm=list(DEFAULT_ACTION_LIMIT_DEG_DEG_MM),
        training=dict(candidates=1, optimizer_updates=128, batch_size=4, seed=TRAIN_SEED, learning_rate=3e-5,
            learning_rate_source='v5 A attention_lr current effective prescription; constant', optimizer='AdamW', weight_decay=1e-4,
            amp=True, amp_dtype='float16', amp_scaler='Original dynamic GradScaler defaults; skipped overflow attempts do not update parameters or count toward128 updates; at most144 attempts', gradient_clip_norm=1., loss='Unmodified model.rectified_flow_loss on all16 actions',
            freeze='vision.* requires_grad False, eval, buffers bit-exact; all other original parameters trainable',
            no_retries_or_extra_steps=True),
        sampling=dict(K=1, flow_steps=4, exec_steps=1, sample_seeds=list(SEEDS), flow_loss_probe_seeds=list(FLOW_SEEDS),
            evaluation_batch_size=1, inference_autocast=False, fixed_same_real_inputs=True),
        offline_gate=dict(primary='E41 moving_encoder_C MAE_um strictly decreases AND direction accuracy on nonzero C moving_encoder rows strictly increases',
            ab='E41 all-window AND moving-encoder A and B MAE each <= before, numeric zero tolerance',
            output_scale='physical first action clipped only by unchanged production caps; raw metrics also reported',
            moving='abs(actual encoder C velocity mm/s)>1e-6, sign accuracy further requires abs(accepted teacher C mm)>1e-9',
            groups='all, moving_encoder, nonzero_C, approaching, receding, crossing, hold; absent groups explicit count0/null'),
        closed_loop=dict(only_if_offline_gate=True, target='E41', starts=list(KINDS), models=['original_v5','finetuned_v5'],
            cycles_each=240, max_parallel_workers=4, same_seed=434341, control_period_s=.05, driver='g10',
            all_actions='Only unchanged original neural model.sample through PolicyEvaluator.get_action; prefix history seeded from actual RGB/encoder/accepted commands; no teacher instantiated',
            metrics=['all240 post-action true distances including final','last200 maximum','longest<=20um','endpoint','all actions neural']),
        short_gate=dict(model='original_v5', target='E41', start='below_recede', cycles=2,
            before_full_pipeline=True, checks=['true prefix history','normal production sample agrees with standalone input/sample','production limits andG10 actual first command','finite actual state']),
        budgets_seconds=dict(total=900, short_gate=90, collection=300, training=180, offline=240, closed_loop=360),
        publication=False, hyperparameter_search=False, dev20=False, standard_reset_long_range=False,
        desktop=DESKTOP, runtime='him1 host, deployment .venv, GPU0 RTX4090, NVIDIA EGL')
    save('asset_audit.json', dict(time=now(), status='passed', host=os.uname().nodename, checkpoint_sha256=sha(REF),
        ridge_array_shapes=ridge, ridge_is_complete_visual_data=False, trajectory_audit=old_rows,
        old_data_supervision_note='Homogeneous original metadata labels zero-velocity intent; this pilot recollects executable current-velocity accepted teacher actions',
        source_interfaces_reviewed=[str(p) for p in files if 'tinyfit' in str(p) or 'direction_pair' in str(p) or p == OLD_CODE],
        gpu=torch.cuda.get_device_name(0), direct_224_preprocessing=e.direct_224_preprocessing,
        original_normalization={k:v for k,v in checkpoint['config'].items() if k.endswith('_mean') or k.endswith('_std')}))
    save('experiment_card.json', card)
    files += [OUT/'experiment_card.json', OUT/'parameters.json']
    save('input_fingerprints.json', dict(files={str(p):sha(p) for p in files}, runtime=frozen_sources()))
    probe = subprocess.run(['xdpyinfo','-display',':1'], env={**os.environ, **DESKTOP}, capture_output=True, text=True, timeout=8)
    assert probe.returncode == 0, probe.stderr
    save('host_preflight.json',dict(time=now(), gpu=torch.cuda.get_device_name(0), executable=sys.executable,
        display_probe_returncode=probe.returncode, display_probe=probe.stdout[:600], desktop=DESKTOP))
    state('prepared', status='prepared', training_started=False, duplicate_directory_check='absent before creation')
    record('preparation_passed', dict(checkpoint_sha256=REF_SHA, train='E40', heldout='E41', rgb='missing; recollect4x40',
        trainable_count=sum(p['count'] for p in parameters if p['trainable']), card=str(OUT/'experiment_card.json')))


def identity_check():
    manifest = read(OUT/'input_fingerprints.json')
    for path, value in manifest['files'].items():
        assert sha(path) == value, f'Frozen input changed: {path}'
    assert frozen_sources() == manifest['runtime'], 'Production/scene identity changed'


def teacher_module():
    # Reuse the exact setup and teacher evaluation interfaces from the prior pilot.
    sys.path.insert(0, str(OLD_CODE.parent))
    spec = importlib.util.spec_from_file_location('prior_terminal_pilot', OLD_CODE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def oct_from_observation(obs):
    import numpy as np
    result = np.zeros(4, np.float32)
    navigation = obs.get('oct_navigation_depth')
    delta = None if navigation is None else navigation.get('delta_xyz_m')
    if delta is not None and len(delta) == 3 and np.isfinite(delta).all():
        result[:3] = np.asarray(delta)*1000
        result[3] = 1.
    return result


def snapshot(scene, sensor):
    import numpy as np
    from lerobot_training.eval_policy import TASK_TO_DEG_DEG_MM
    sensor.observation_time_s = scene.motor_execution.time_s
    obs = sensor.observe()
    return dict(rgb=np.asarray(obs['rgb'], dtype=np.uint8).copy(), oct=oct_from_observation(obs),
        state=np.asarray(scene.current_task())*TASK_TO_DEG_DEG_MM,
        velocity=np.asarray(scene.motor_execution.actual_velocity)*TASK_TO_DEG_DEG_MM,
        time=float(scene.motor_execution.time_s))


def start_with_history(index, kind):
    """Exact old motor prefix plus measured last two RGB/encoder frames."""
    import numpy as np
    from rcm.scene_v9 import RCMSceneV9
    from lerobot_training.eval_policy import StrictRGBDSensor, visual_servo_settings
    from scripts.research.homog_replay_common import QUALITY
    prior = teacher_module()
    scene = RCMSceneV9()
    sensor = None
    try:
        task = prior.task(index)
        world = np.asarray(scene.robot_to_world(np.asarray(task['target_robot_mm'])))
        scene.set_target_world(world)
        q = scene.target_task(validate=True).copy()
        above = kind.startswith('above')
        sign = -1. if above else 1.
        q[2] += .000090 if above else -.000090
        scene.set_task(q)
        scene.enable_motor_execution()
        scene.reset_motor_time(0.)
        settings = visual_servo_settings(scene)
        settings.update(perception_backend='strict_rgbd', rgbd_render_quality=QUALITY.copy())
        sensor = StrictRGBDSensor(scene, settings)
        commands = [np.array([0.,0.,sign*.003]) for _ in range(24)]+[np.array([0.,0.,-sign*.003]) for _ in range(2)]
        history = []
        for j, action in enumerate(commands):
            if j >= 24:
                row = snapshot(scene, sensor)
                row['previous_action'] = commands[j-1].copy()
                row['accepted_action'] = action.copy()
                history.append(row)
            scene.apply_prepared_motor(scene.motor_execution.preview(scene.current_task()+action*prior.UNIT, .05))
        assert len(history) == 2 and len(scene.motor_execution.records) == 26
        return scene, sensor, world, history, commands[-1].copy()
    except BaseException:
        if sensor is not None:
            sensor.close()
        scene.close()
        raise


def seed_history(e, history, previous):
    import numpy as np
    import torch
    # Install only past real observations at reset; production get_action appends current.
    row = history[-1]
    e.obs_buffer = [dict(image=e.preprocess_image(row['rgb']),
        state=torch.as_tensor(np.asarray(row['state'],np.float32)),
        oct=torch.as_tensor(np.asarray(row['oct'],np.float32)))]
    e.previous_action = np.asarray(previous,np.float32).copy()
    e.regression_previous_action = e.previous_action.copy()


def arguments(e, previous_row, current_row, previous):
    import numpy as np
    import torch
    images = torch.stack([e.preprocess_image(r['rgb']) for r in (previous_row,current_row)]).unsqueeze(0).cuda()
    states = torch.as_tensor(np.asarray([previous_row['state'],current_row['state']],np.float32)).unsqueeze(0).cuda()
    octs = torch.as_tensor(np.asarray([previous_row['oct'],current_row['oct']],np.float32)).unsqueeze(0).cuda()
    velocity = (states[:,-1]-states[:,-2])/.05
    previous = torch.as_tensor(np.asarray(previous,np.float32)).unsqueeze(0).cuda()
    return (images, states, octs, velocity, previous)


def score(post):
    import numpy as np
    post = np.asarray(post,float)
    run = longest = 0
    for inside in post<=20:
        run = run+1 if inside else 0
        longest = max(run,longest)
    return dict(cycles=len(post), final_true_error_um=float(post[-1]),
        last200_max_true_error_um=float(post[-200:].max()) if len(post)>=200 else None,
        longest_le20_cycles=longest, longest_le20_s=longest*.05, terminal_le20_cycles=run,
        all_last200_le20=bool(len(post)>=200 and post[-200:].max()<=20))


def neural_rollout(path, kind, cycles, folder, startup=False):
    import numpy as np
    import torch
    from lerobot_training.eval_policy import DEFAULT_ACTION_LIMIT_DEG_DEG_MM
    folder = OUT/folder
    folder.mkdir(parents=True, exist_ok=False)
    scene = sensor = None
    try:
        e = evaluator(path)
        scene, sensor, world, history, previous = start_with_history(41,kind)
        save(folder/'process_receipt.json',dict(time=now(), pid=os.getpid(), ppid=os.getppid(), model=str(path), model_sha256=sha(path),
            target='E41', kind=kind, cycles=cycles, desktop={k:os.environ.get(k) for k in DESKTOP}, argv=sys.argv))
        save(folder/'prefix_history.json', [{k:v for k,v in r.items() if k!='rgb'} for r in history])
        np.savez_compressed(folder/'prefix_rgb.npz', rgb=np.stack([r['rgb'] for r in history]))
        original_reset = e.reset_buffer
        def reset(seed=None):
            original_reset(seed=seed)
            seed_history(e,history,previous)
        e.reset_buffer = reset
        trace = (folder/'post_action_trace.jsonl').open('x', buffering=1)
        post = []
        input_rows = []
        latest = {}
        original_observe = sensor.observe
        def observe():
            obs = original_observe()
            latest['obs'] = obs
            return obs
        sensor.observe = observe
        original_action = e.get_action
        parity = {}
        def action(rgb, state_abc, oct_feature):
            current = dict(rgb=np.asarray(rgb),state=np.asarray(state_abc),oct=np.asarray(oct_feature))
            if not input_rows:
                inputs = arguments(e,history[-1],current,previous)
                generator_state = e.action_generator.get_state()
                g = torch.Generator(device='cuda')
                g.set_state(generator_state)
                direct = e.model.sample(*inputs, steps=4, generator=g)[0,0].cpu().numpy()
            predicted = original_action(rgb,state_abc,oct_feature)
            if not input_rows:
                np.testing.assert_array_equal(predicted,direct)
                parity.update(status='passed', production_and_direct_action=predicted.tolist(), physical_units='deg,deg,mm',
                    first_action_max_abs_difference=0., velocity_input=inputs[3].cpu().tolist(),
                    actual_encoder_velocity=(scene.motor_execution.actual_velocity*np.array([180/np.pi,180/np.pi,1000])).tolist(),
                    previous_accepted=inputs[4].cpu().tolist(), n_obs_steps=2, K=1, flow4=True, exec1=True)
                save(folder/'first_input_parity.json',parity)
                torch.save({k:v.cpu() for k,v in zip(INPUT_KEYS,inputs)},folder/'first_real_input.pt')
            input_rows.append(dict(state=np.asarray(state_abc).tolist(), oct=np.asarray(oct_feature).tolist(),
                time_s=scene.motor_execution.time_s, actual_velocity_si=scene.motor_execution.actual_velocity.tolist(),
                previous_accepted=e.previous_action.tolist(), rgb_sha256=hashlib.sha256(np.ascontiguousarray(rgb)).hexdigest(),
                raw_model_action=np.asarray(predicted).tolist()))
            return predicted
        e.get_action = action
        def observer(event):
            # Independent MuJoCo needle site is read only after the model action executed.
            distance = float(np.linalg.norm(scene.site_mm('needle_tip')-world)*1000)
            post.append(distance)
            row = dict(time=now(),post_action_cycle=len(post),post_true_error_um=distance,
                input=input_rows[-1], event=event, encoder_after=list(scene.motor_execution.backend.robot))
            trace.write(json.dumps(row, default=default)+'\n')
            if len(post)==1:
                save(folder/'actual_first_action.json',dict(status='actual_first_action_completed',**row))
        e.step_observer = observer
        e.model.eval()
        ep = e.evaluate_episode(scene,world,max_steps=cycles,error_threshold_mm=.00035,control_period_s=.05,
            action_limit_deg_deg_mm=DEFAULT_ACTION_LIMIT_DEG_DEG_MM, max_consecutive_safety_limited=None,
            seed=434341,sensor=sensor,stop_on_success=False,progress_every=40)
        trace.close()
        ep['motor_records'] = ep['motor_records'][26:]
        ep.update(model_sha256=sha(path),model_path=str(path),start_kind=kind,target_id='E41',prefix='identical26 real G10 ticks',
            teacher_instances=0, neural_action_calls=len(input_rows), all_ABC_actions_from_original_neural_model=True,
            no_control_handoff=True, prior_observations='real prefix RGB/encoder/OCT, no reset gap')
        save(folder/'evaluation_results.json',ep)
        result = dict(status='completed', time=now(), model_sha256=sha(path),model_path=str(path),target='E41',kind=kind,
            early_stop_reason=ep['early_stop_reason'], neural_action_calls=len(input_rows),teacher_instances=0,
            all_ABC_actions_from_original_neural_model=True, input_parity=parity, **score(post))
        assert len(post)==cycles and ep['early_stop_reason'] is None
        assert np.isclose(post[-1],ep['final_error_mm']*1000,atol=1e-10,rtol=0)
        assert all(r['driver']['variant']=='g10' and r['duration_s']==.05 for r in ep['motor_records'])
        save(folder/'summary.json',result)
        record('short_gate_passed' if startup else 'closed_loop_completed',result)
        if startup:
            ack(dict(status='actual_first_action_completed',time=now(),pid=os.getpid(),short_gate='2/2 cycles passed',
                production_input_action_parity='bit_exact',receipt=str(folder/'actual_first_action.json'),
                first_action=read(folder/'actual_first_action.json'), workflow_continues=True))
        return result
    finally:
        if sensor is not None:
            sensor.close()
        if scene is not None:
            scene.close()


def collect(index,kind):
    import numpy as np
    from lerobot_training.eval_policy import PolicyEvaluator
    from scripts.research.homog_replay_common import make_teacher
    prior = teacher_module()
    folder = OUT/'data'/f'E{index:02d}_{kind}'
    folder.mkdir(parents=True,exist_ok=False)
    scene = sensor = teacher = None
    try:
        scene,sensor,world,history,previous = start_with_history(index,kind)
        # Same teacher as the previous pilot; only data collection may instantiate it.
        teacher = make_teacher(scene)
        sensor.close()
        sensor = teacher.sensor
        rows = []
        latest = {}
        original_observe = sensor.observe
        def observe():
            obs = original_observe()
            latest['obs'] = obs
            return obs
        sensor.observe = observe
        class TeacherEvaluator(PolicyEvaluator):
            def __init__(self):
                self.use_oct=True
                self.measure_inference=False
                self.controller_variant='feedback'
                self.fine_evaluator=None
                self.last_terminal_solution=None
                self.terminal_locked_target_world_mm=None
            def reset_buffer(self, seed=None):
                self.previous=previous.copy()
                self.control_mode_counts={'teacher_diagnostic':0}
            def record_applied_action(self, action):
                self.previous=np.asarray(action).copy()
            def invalidate_action_chunk(self):
                pass
            def get_action(self,rgb,state_abc,oct_feature):
                assert oct_feature[3]>.5
                velocity=scene.motor_execution.actual_velocity.copy()
                q=np.asarray(state_abc)*prior.UNIT
                result=teacher.geometry_action_from_observation(q,latest['obs'],
                    previous_velocity=velocity,command_q=q)
                assert result['valid'], 'Teacher invalid: technical/data failure, not neural negative'
                row=dict(rgb=np.asarray(rgb,np.uint8).copy(),state=np.asarray(state_abc).copy(),oct=np.asarray(oct_feature).copy(),
                    actual_velocity_si=velocity,previous_action=self.previous.copy(),time=scene.motor_execution.time_s,
                    raw_teacher=np.asarray(result['action_deg_deg_mm']).copy())
                rows.append(row)
                self.control_mode_counts['teacher_diagnostic']+=1
                return row['raw_teacher']
        e = TeacherEvaluator()
        post = []
        def observer(event):
            row=rows[-1]
            row['accepted_action']=np.asarray(event['accepted_action_deg_deg_mm']).copy()
            row['next_state_si']=scene.current_task().copy()
            row['encoder_after']=np.asarray(scene.motor_execution.backend.robot).copy()
            row['post_true_error_um']=float(np.linalg.norm(scene.site_mm('needle_tip')-world)*1000)
            post.append(row['post_true_error_um'])
        e.step_observer=observer
        ep=e.evaluate_episode(scene,world,max_steps=40,error_threshold_mm=.00035,control_period_s=.05,
            action_limit_deg_deg_mm=(.2,.1,.03),max_consecutive_safety_limited=None,
            seed=434300+index,sensor=sensor,stop_on_success=False,progress_every=40)
        ep['motor_records']=ep['motor_records'][26:]
        save(folder/'teacher_trajectory.json',ep)
        assert len(rows)==40 and len(ep['motor_records'])==40 and ep['early_stop_reason'] is None
        times=np.asarray([r['time'] for r in rows])
        np.testing.assert_allclose(np.diff(times),.05,atol=1e-8,rtol=0)
        for j,(r,m) in enumerate(zip(rows,ep['motor_records'])):
            assert m['driver']['variant']=='g10'
            np.testing.assert_allclose(r['state']*prior.UNIT,m['actual_before_task_rad_rad_m'],atol=1e-11,rtol=0)
            np.testing.assert_allclose(r['next_state_si'],m['actual_after_task_rad_rad_m'],atol=1e-11,rtol=0)
            np.testing.assert_allclose(r['previous_action'],rows[j-1]['accepted_action'] if j else previous,atol=1e-12,rtol=0)
            if j:
                np.testing.assert_allclose(r['actual_velocity_si'],ep['motor_records'][j-1]['actual_velocity_rad_rad_m_per_s'],atol=1e-11,rtol=0)
            assert abs(m['start_time_s']-r['time'])<1e-8
        np.savez_compressed(folder/'observations.npz',rgb=np.stack([r['rgb'] for r in history]+[r['rgb'] for r in rows]),
            state=np.stack([r['state'] for r in history]+[r['state'] for r in rows]),
            oct=np.stack([r['oct'] for r in history]+[r['oct'] for r in rows]),
            previous_action=np.stack([r['previous_action'] for r in history]+[r['previous_action'] for r in rows]),
            timestamp_s=np.asarray([r['time'] for r in history]+[r['time'] for r in rows]),
            accepted_teacher_action=np.stack([r['accepted_action'] for r in rows]),
            actual_encoder_velocity_si=np.stack([r['actual_velocity_si'] for r in rows]),
            actual_next_state_si=np.stack([r['next_state_si'] for r in rows]),
            encoder_after=np.stack([r['encoder_after'] for r in rows]),post_true_error_um=np.asarray(post))
        save(folder/'alignment.json',[{k:v for k,v in r.items() if k!='rgb'} for r in rows])
        # Preserve old trajectory identity and compare replayed state/action prefix where available.
        old_path=OLD/f'trajectories/train_E{index:02d}_{kind}.json'
        parity=None
        if old_path.exists():
            old=read(old_path)
            parity=dict(source=str(old_path),state_max_abs_diff=float(np.max(np.abs(np.asarray(ep['trajectory']['states'])-old['trajectory']['states']))),
                accepted_action_max_abs_diff=float(np.max(np.abs(np.asarray(ep['trajectory']['actions'])-old['trajectory']['actions']))))
            np.testing.assert_allclose(ep['trajectory']['states'],old['trajectory']['states'],atol=1e-10,rtol=0)
            np.testing.assert_allclose(ep['trajectory']['actions'],old['trajectory']['actions'],atol=1e-10,rtol=0)
        result=dict(status='completed',time=now(),target=f'E{index}',kind=kind,role='train' if index==40 else 'heldout',
            observations=42,teacher_actions=40,windows_horizon16=25,source_sha256=sha(folder/'observations.npz'),
            final_true_error_um=float(post[-1]),last10_max_true_error_um=float(max(post[-10:])),
            teacher_terminal_gate=bool(max(post[-10:])<=20), prior_replay_parity=parity,
            sample_alignment='same-call RGB/OCT+ABC/actual encoder velocity/previous accepted at t; accepted label at t; actual state/error at t+.05')
        save(folder/'summary.json',result)
        record('teacher_segment_completed',result)
        return result
    finally:
        if sensor is not None:
            sensor.close()
        if scene is not None:
            scene.close()


def build_windows(e,index):
    import numpy as np
    import torch
    samples=[]
    metas=[]
    for kind in KINDS:
        path=OUT/'data'/f'E{index:02d}_{kind}'/'observations.npz'
        with np.load(path,allow_pickle=False) as archive:
            a={k:archive[k].copy() for k in archive.files}
        for start in range(25):
            j=start+2
            previous=dict(rgb=a['rgb'][j-1],state=a['state'][j-1],oct=a['oct'][j-1])
            current=dict(rgb=a['rgb'][j],state=a['state'][j],oct=a['oct'][j])
            inputs=arguments(e,previous,current,a['previous_action'][j])
            b={k:v[0].cpu() for k,v in zip(INPUT_KEYS,inputs)}
            b['action']=torch.as_tensor(a['accepted_teacher_action'][start:start+16],dtype=torch.float32)
            assert b['action'].shape==(16,3) and np.all(a['oct'][j-1:j+1,3]>.5)
            assert np.allclose(np.diff(a['timestamp_s'][j-1:j+16]),.05,atol=1e-8,rtol=0)
            # These are independent scoring groups only, never model inputs/loss weights.
            tr=read(path.parent/'teacher_trajectory.json')['trajectory']
            pre=float(tr['errors'][start]*1000)
            post=float(a['post_true_error_um'][start])
            oct_now=a['oct'][j,:3]
            oct_next=a['oct'][j+1,:3] if j+1<len(a['oct']) else oct_now
            crossed=bool(oct_now[2]*oct_next[2]<0)
            c=float(a['accepted_teacher_action'][start,2])
            velocity=float(a['actual_encoder_velocity_si'][start,2]*1000)
            group='hold' if abs(c)<=1e-9 and abs(velocity)<=1e-6 else 'crossing' if crossed else 'approaching' if post<pre else 'receding'
            metas.append(dict(target=f'E{index}',kind=kind,start=start,source=str(path),source_sha256=sha(path),
                history_frames=[j-1,j],action_frames_inclusive=[start,start+15],group=group,
                moving_encoder=abs(velocity)>1e-6,nonzero_C=abs(c)>1e-9,teacher_C_mm=c,
                actual_C_velocity_mm_s=velocity,pre_true_error_um=pre,post_true_error_um=post))
            samples.append(b)
    return {k:torch.stack([b[k] for b in samples]) for k in samples[0]},metas


def loss(m,b):
    return m.rectified_flow_loss(*(b[k] for k in INPUT_KEYS),b['action'])


def batch_select(batch,indices):
    return {k:v[indices].cuda() for k,v in batch.items()}


def configure_training(m):
    for name,p in m.named_parameters():
        p.requires_grad_(not name.startswith('vision.'))
    m.train()
    m.vision.eval()
    assert not any(p.requires_grad for p in m.vision.parameters())
    assert not any(x.training for x in m.vision.modules())


def step0_parity(e,batch):
    import torch
    other=evaluator()
    other.model.eval()
    configure_training(e.model)
    b=batch_select(batch,slice(0,1))
    for k in SEEDS:
        p=e.model.sample(*(b[n] for n in INPUT_KEYS),steps=4,generator=torch.Generator(device='cuda').manual_seed(k))
        q=other.model.sample(*(b[n] for n in INPUT_KEYS),steps=4,generator=torch.Generator(device='cuda').manual_seed(k))
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    for amp in (False,True):
        with torch.no_grad(),torch.autocast('cuda',enabled=amp,dtype=torch.float16):
            torch.manual_seed(8001)
            a=loss(e.model,b)
            torch.manual_seed(8001)
            c=loss(other.model,b)
        torch.testing.assert_close(a,c,rtol=0,atol=0)
    for name,buffer in other.model.named_buffers():
        torch.testing.assert_close(dict(e.model.named_buffers())[name],buffer,rtol=0,atol=0)
    result=dict(status='passed',time=now(),step=0,physical_actions_bit_exact_4_seeds=True,
        loss_fp32_and_original_amp_bit_exact=True,all_normalization_buffers_bit_exact=True,
        train_mode_vs_original_eval='identical with frozen vision.eval(); no dropout/BN drift',production_entry='PolicyEvaluator original checkpoint')
    save('step0_parity.json',result)
    record('step0_parity_passed',result)
    e.model.eval()
    del other
    torch.cuda.empty_cache()


def offline(e,batch,metas,label,split):
    import numpy as np
    import torch
    from lerobot_training.eval_policy import DEFAULT_ACTION_LIMIT_DEG_DEG_MM
    started=time.monotonic()
    e.model.eval()
    predictions=[]
    flow=[]
    with torch.no_grad():
        for j in range(len(metas)):
            b=batch_select(batch,slice(j,j+1))
            actions=[]
            losses=[]
            for k in SEEDS:
                p=e.model.sample(*(b[n] for n in INPUT_KEYS),steps=4,generator=torch.Generator(device='cuda').manual_seed(k))
                actions.append(p[0,0].cpu().numpy())
            for k in FLOW_SEEDS:
                torch.manual_seed(k)
                losses.append(float(loss(e.model,b)))
            predictions.append(actions)
            flow.append(losses)
    pred=np.asarray(predictions) # window,seed,ABC; each draw is exactly K1
    target=batch['action'][:,0].numpy()
    clipped=np.clip(pred,-np.asarray(DEFAULT_ACTION_LIMIT_DEG_DEG_MM),np.asarray(DEFAULT_ACTION_LIMIT_DEG_DEG_MM))
    masks=dict(all=np.ones(len(metas),bool), moving_encoder=np.asarray([r['moving_encoder'] for r in metas]),
        nonzero_C=np.asarray([r['nonzero_C'] for r in metas]))
    masks.update({g:np.asarray([r['group']==g for r in metas]) for g in ('approaching','receding','crossing','hold')})
    masks.update({kind:np.asarray([r['kind']==kind for r in metas]) for kind in KINDS})
    masks['teacher_C_positive']=target[:,2]>1e-9
    masks['teacher_C_negative']=target[:,2]<-1e-9
    groups={}
    for name,mask in masks.items():
        signs=mask & masks['nonzero_C']
        groups[name]=dict(windows=int(mask.sum()),nonzero_C_windows=int(signs.sum()),
            raw_first_action_mae_deg_deg_um=(np.abs(pred[mask]-target[mask,None]).mean((0,1))*[1,1,1000]).tolist() if mask.any() else None,
            production_clipped_first_action_mae_deg_deg_um=(np.abs(clipped[mask]-target[mask,None]).mean((0,1))*[1,1,1000]).tolist() if mask.any() else None,
            C_direction_accuracy=float((np.sign(clipped[signs,:,2])==np.sign(target[signs,None,2])).mean()) if signs.any() else None,
            C_direction_correct=int((np.sign(clipped[signs,:,2])==np.sign(target[signs,None,2])).sum()),
            C_direction_draws=int(signs.sum()*len(SEEDS)),
            per_seed_production_mae_deg_deg_um=(np.abs(clipped[mask]-target[mask,None]).mean(0)*[1,1,1000]).tolist() if mask.any() else None,
            per_seed_C_direction_accuracy=(np.sign(clipped[signs,:,2])==np.sign(target[signs,None,2])).mean(0).tolist() if signs.any() else None)
    folder=OUT/'offline'/label
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/f'{split}_predictions.npz',raw=pred,production_clipped=clipped,teacher=target,
        flow_loss=np.asarray(flow),seeds=np.asarray(SEEDS),flow_seeds=np.asarray(FLOW_SEEDS))
    result=dict(status='completed',time=now(),model=label,split=split,windows=len(metas),sample_seeds=list(SEEDS),
        model_K=1,flow_steps=4,exec_steps=1,groups=groups,flow_loss_4_fixed_probes=float(np.mean(flow)),
        flow_loss_is_secondary_not_gate=True,elapsed_s=time.monotonic()-started)
    save(folder/f'{split}_metrics.json',result)
    record('offline_completed',dict(model=label,split=split,windows=len(metas),metrics=str(folder/f'{split}_metrics.json'),
        moving=groups['moving_encoder'],all=groups['all'],flow_loss=result['flow_loss_4_fixed_probes']))
    return result


def train_once(e,batch,metas):
    import torch
    configure_training(e.model)
    m=e.model
    frozen={k:v.detach().cpu().clone() for k,v in m.state_dict().items() if k.startswith('vision.')}
    buffers={k:v.detach().cpu().clone() for k,v in m.named_buffers()}
    active=[p for p in m.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(active,lr=3e-5,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=True)
    rng=torch.Generator().manual_seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    torch.cuda.manual_seed_all(TRAIN_SEED)
    train_log=[]
    started=time.monotonic()
    # Preserve original AMP dynamic scaling; only successful optimizer updates count.
    updates=0
    for attempt in range(144):
        if updates==128:
            break
        assert time.monotonic()-started<180, 'Training budget exceeded; no extra steps'
        indices=torch.randint(len(metas),(4,),generator=rng)
        b=batch_select(batch,indices)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            value=loss(m,b)
        assert torch.isfinite(value), 'Nonfinite loss: technical training failure'
        scaler.scale(value).backward()
        scaler.unscale_(optimizer)
        gradient=torch.nn.utils.clip_grad_norm_(active,1.)
        scale_before=scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        updated=scaler.get_scale()>=scale_before
        updates+=int(updated)
        row=dict(attempt=attempt+1,step=updates,loss=float(value.detach()),gradient_norm=float(gradient) if torch.isfinite(gradient) else None,optimizer_updated=updated,
            grad_scale=scaler.get_scale(),sample_indices=indices.tolist(),elapsed_s=time.monotonic()-started)
        train_log.append(row)
        with (OUT/'training.jsonl').open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n')
        assert not updated or torch.isfinite(gradient), 'Nonfinite gradient accepted by AMP: technical failure'
        if attempt==0 or (updated and updates%16==0):
            print('TRAIN',json.dumps(row),flush=True)
    assert updates==128, 'AMP did not complete128 updates within144 attempts; technical failure'
    for k,v in m.state_dict().items():
        if k in frozen:
            torch.testing.assert_close(v.cpu(),frozen[k],rtol=0,atol=0)
    for k,v in m.named_buffers():
        torch.testing.assert_close(v.cpu(),buffers[k],rtol=0,atol=0)
    parameters=read(OUT/'parameters.json')
    payload={k:v for k,v in e.checkpoint.items() if k not in ('optimizer_state_dict','scheduler_state_dict','scaler_state_dict','rng_state','model_state_dict')}
    payload.update(model_state_dict=m.state_dict(), optimizer_state_dict=optimizer.state_dict(),scaler_state_dict=scaler.state_dict(),scheduler_state_dict=None,
        train_loss=train_log[-1]['loss'],val_loss=None,epoch=0,
        training_module='original_neural_terminal_pilot_v1_unmodified_flow_loss',
        dataset_sha256=sha(OUT/'real_input_windows.pt'),correction_dataset_sha256=None,
        sampling_contract=dict(mode='uniform_with_replacement',batch_size=4,seed=TRAIN_SEED,train_target='E40',heldout_target='E41'),
        config=copy.deepcopy(e.checkpoint['config']),
        freeze_config=dict(vision_frozen=True,nonvision_all_trainable=True,trainable_count=parameters['trainable_count'],
            trainable_parameters=[p['name'] for p in parameters['parameters'] if p['trainable']],
            frozen_parameters=[p['name'] for p in parameters['parameters'] if not p['trainable']]),
        progress=dict(global_step=128,optimizer_updates=128),
        pilot=dict(name=OUT.name,init_sha256=REF_SHA,seed=TRAIN_SEED,train_target='E40',heldout_target='E41',
            data_manifest_sha256=sha(OUT/'window_manifest.json'),vision_bit_exact=True,normalization_bit_exact=True))
    # Preserve preprocessing/data identity keys; update training metadata without changing input transforms.
    payload['config'].update(require_temporal_bias_only=False,freeze_config=payload['freeze_config'],
        lr=3e-5,backbone_lr=3e-5,attention_lr=3e-5,lr_schedule='constant',seed=TRAIN_SEED,
        init_from=str(REF),output_dir=str(OUT),pilot_dataset=str(OUT/'real_input_windows.pt'),
        pilot_learning_rate=3e-5,pilot_optimizer_updates=128,pilot_train_target='E40')
    torch.save(payload,OUT/'after.pth')
    save('training_result.json',dict(status='completed',time=now(),optimizer_updates=128,attempted_batches=len(train_log),amp_skipped_attempts=len(train_log)-128,
        init_sha256=REF_SHA,after_sha256=sha(OUT/'after.pth'),vision_all_parameters_buffers_bit_exact=True,
        normalization_all_buffers_bit_exact=True,trainable_count=parameters['trainable_count'],
        trainable_parameter_manifest=str(OUT/'parameters.json'),elapsed_s=time.monotonic()-started,
        loss_first=train_log[0]['loss'],loss_last=train_log[-1]['loss']))
    record('training_completed',read(OUT/'training_result.json'))
    m.eval()


def gate(before,after):
    b=before['groups']
    a=after['groups']
    checks=[]
    def add(name,passed,left,right,op):
        checks.append(dict(name=name,passed=bool(passed),before=left,after=right,required=op))
    assert b['moving_encoder']['windows']>0 and b['moving_encoder']['nonzero_C_windows']>0
    x=b['moving_encoder']['production_clipped_first_action_mae_deg_deg_um'][2]
    y=a['moving_encoder']['production_clipped_first_action_mae_deg_deg_um'][2]
    add('heldout_moving_C_MAE_um',y<x,x,y,'after < before')
    x=b['moving_encoder']['C_direction_accuracy']
    y=a['moving_encoder']['C_direction_accuracy']
    add('heldout_moving_nonzero_C_direction_accuracy',y>x,x,y,'after > before')
    for group in ('all','moving_encoder'):
        for i,axis in enumerate(('A','B')):
            x=b[group]['production_clipped_first_action_mae_deg_deg_um'][i]
            y=a[group]['production_clipped_first_action_mae_deg_deg_um'][i]
            add(f'heldout_{group}_{axis}_MAE_deg',y<=x,x,y,'after <= before; zero tolerance')
    result=dict(status='passed' if all(c['passed'] for c in checks) else 'failed',time=now(),checks=checks,
        scientific_scope='One128-update neural candidate on E40, whole-target E41 local windows; flow loss excluded from promotion')
    save('offline_gate.json',result)
    record('offline_gate_'+result['status'],result)
    return result


def closed_loop():
    children=[]
    threads=[]
    def tee(pipe,path):
        with path.open('x',buffering=1) as f:
            for line in pipe:
                f.write(line)
                print(line,end='',flush=True)
    try:
        for label,path in (('original_v5',REF),('finetuned_v5',OUT/'after.pth')):
            for kind in KINDS:
                name=f'{label}_{kind}'
                cmd=[sys.executable,'-E','-B','-u',str(Path(__file__)),'worker','--model',str(path),'--kind',kind,'--folder',f'closed_loop/{name}']
                with (OUT/'commands.jsonl').open('a') as f:
                    f.write(json.dumps(dict(time=now(),command=cmd,cwd=str(ROOT)))+'\n')
                p=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                children.append((name,p))
                t=threading.Thread(target=tee,args=(p.stdout,OUT/f'{name}.terminal.log'))
                t.start()
                threads.append(t)
        save('closed_loop_processes.json',dict(time=now(),workers=[dict(name=n,pid=p.pid) for n,p in children]))
        deadline=time.monotonic()+360
        codes=[p.wait(timeout=max(.1,deadline-time.monotonic())) for _,p in children]
        for t in threads:
            t.join()
        assert all(c==0 for c in codes), f'Worker technical/incomplete failure: {codes}'
        results=[read(OUT/'closed_loop'/n/'summary.json') for n,_ in children]
        for kind in KINDS:
            paths=[OUT/'closed_loop'/f'{label}_{kind}' for label in ('original_v5','finetuned_v5')]
            assert sha(paths[0]/'prefix_history.json')==sha(paths[1]/'prefix_history.json')
            assert sha(paths[0]/'prefix_rgb.npz')==sha(paths[1]/'prefix_rgb.npz')
        save('closed_loop_summary.json',dict(status='completed',time=now(),results=results,
            paired_actual_prefix_histories_bit_exact=True, all_actions_neural=True, teacher_instances=0,
            local20um_passes=sum(r['all_last200_le20'] for r in results),standard_reset_or_dev20_claim=False))
        record('closed_loop_all_completed',read(OUT/'closed_loop_summary.json'))
        return results
    finally:
        for _,p in children:
            if p.poll() is None:
                p.terminate()
        for _,p in children:
            if p.poll() is None:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()


def run():
    import torch
    started=time.monotonic()
    with (OUT/'.execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/'actual_start_receipt.json').exists(), 'Refuse duplicate experiment execution'
        save('actual_start_receipt.json',dict(time=now(),pid=os.getpid(),ppid=os.getppid(),argv=sys.argv,host=os.uname().nodename,
            desktop={k:os.environ.get(k) for k in DESKTOP}))
        stage='identity'
        try:
            identity_check()
            stage='short_gate'
            state(stage,status='running')
            neural_rollout(REF,'below_recede',2,'short_gate',startup=True)
            stage='collection'
            state(stage,status='running')
            summaries=[]
            collection_started=time.monotonic()
            for i in (40,41):
                for kind in KINDS:
                    summaries.append(collect(i,kind))
                    assert time.monotonic()-collection_started<300, 'Collection time cap'
            save('collection_summary.json',dict(status='completed',time=now(),segments=summaries))
            if not all(s['teacher_terminal_gate'] for s in summaries):
                result=dict(status='teacher_gate_failed',stage='collection',time=now(),training_updates=0,
                    closed_loop_started=False,reason='A required40-tick executable teacher segment does not hold final10 ticks <=20um; neural feasibility untested')
                save('summary.json',result)
                record('teacher_gate_failed',result)
                state(stage,**{k:v for k,v in result.items() if k!='stage'})
                return 0
            stage='windows_and_step0'
            state(stage,status='running')
            e=evaluator()
            train,train_meta=build_windows(e,40)
            heldout,heldout_meta=build_windows(e,41)
            save('window_manifest.json',dict(train=train_meta,heldout=heldout_meta,
                train_target='E40',heldout_target='E41',whole_target_isolation=True,overlap_between_splits=False))
            torch.save(dict(train=train,heldout=heldout),OUT/'real_input_windows.pt')
            # Exact original payload before any update, including native checkpoint schema.
            import shutil
            shutil.copyfile(REF,OUT/'before.pth')
            assert sha(OUT/'before.pth')==REF_SHA
            step0_parity(e,train)
            stage='offline_before'
            state(stage,status='running')
            before_train=offline(e,train,train_meta,'before','train_E40')
            before_val=offline(e,heldout,heldout_meta,'before','heldout_E41')
            stage='training'
            state(stage,status='running',requested_updates=128)
            train_once(e,train,train_meta)
            del e
            torch.cuda.empty_cache()
            # Production checkpoint loader must load the after checkpoint directly.
            e=evaluator(OUT/'after.pth')
            stage='offline_after'
            state(stage,status='running')
            after_train=offline(e,train,train_meta,'after','train_E40')
            after_val=offline(e,heldout,heldout_meta,'after','heldout_E41')
            promotion=gate(before_val,after_val)
            del e,train,heldout
            torch.cuda.empty_cache()
            if promotion['status']!='passed':
                result=dict(status='completed_offline_gate_failed',time=now(),optimizer_updates=128,
                    closed_loop_started=False,offline_gate=promotion,elapsed_s=time.monotonic()-started,
                    scientific_conclusion='This single candidate did not pass action-improvement gate; no claim architecture cannot learn, no extra updates/candidates')
            else:
                stage='closed_loop'
                state(stage,status='running')
                results=closed_loop()
                result=dict(status='completed',time=now(),optimizer_updates=128,closed_loop_started=True,closed_loop_results=results,
                    offline_gate=promotion,elapsed_s=time.monotonic()-started,local_only=True,standard_reset_or_dev20_claim=False)
            identity_check()
            save('summary.json',result)
            state('finished',**result)
            (OUT/'report.md').write_text('# Original neural terminal pilot\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            record('workflow_finished',dict(status=result['status'],summary=str(OUT/'summary.json'),elapsed_s=result['elapsed_s']))
            print('FINAL_RESULT_PATH',OUT/'summary.json',flush=True)
            return 0
        except BaseException as exc:
            result=dict(status='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'technical_failure',time=now(),stage=stage,
                error=repr(exc),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-started,
                scientific_negative=False, no_unrequested_retry=True)
            save('failure.json',result)
            save('workflow_status.json',result)
            record('workflow_technical_failure',result)
            if not (OUT/'startup_first_action_receipt.json').exists():
                ack({**result,'status':'startup_technical_failure'})
            traceback.print_exc()
            return 1


def launch():
    with (OUT/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/'gnome_launch_request.json').exists(), 'Refuse second launch'
        identity_check()
        os.mkfifo(OUT/'startup_ack.fifo',0o600)
        fd=os.open(OUT/'startup_ack.fifo',os.O_RDWR|os.O_NONBLOCK)
        cmd=['gnome-terminal','--window','--title=Original v5 neural terminal pilot E40 train E41 heldout 128 updates',
            '--working-directory='+str(ROOT),'--','bash',str(OUT/'launch.sh')]
        save('gnome_launch_request.json',dict(time=now(),command=cmd,desktop=DESKTOP))
        p=subprocess.run(cmd,env={**os.environ,**DESKTOP},capture_output=True,text=True,timeout=20)
        save('gnome_launch_receipt.json',dict(time=now(),command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
        if p.returncode:
            os.close(fd)
            record('gnome_launch_failed',dict(returncode=p.returncode,stderr=p.stderr,background_fallback=False))
            return 1
        # Bounded startup acknowledgement only; no subsequent monitoring/polling.
        sel=selectors.DefaultSelector()
        sel.register(fd,selectors.EVENT_READ)
        events=sel.select(90)
        received=os.read(fd,65536).decode() if events else ''
        sel.close()
        os.close(fd)
        result=dict(time=now(),kind='one bounded startup acknowledgement; not run monitoring',
            receipt=json.loads(received.splitlines()[0]) if received else None,timeout=not bool(received))
        save('launch_startup_acknowledgement.json',result)
        record('gnome_startup_acknowledged',result)
        print(json.dumps(result,ensure_ascii=False,default=default),flush=True)
        return 0 if result['receipt'] and result['receipt']['status']=='actual_first_action_completed' else 1


def main():
    import torch
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','launch','run','worker'])
    p.add_argument('--model')
    p.add_argument('--kind',choices=KINDS)
    p.add_argument('--folder')
    a=p.parse_args()
    if a.mode=='prepare':
        prepare()
        return 0
    if a.mode=='launch':
        return launch()
    if a.mode=='run':
        return run()
    identity_check()
    neural_rollout(Path(a.model),a.kind,240,a.folder)
    return 0


if __name__=='__main__':
    sys.exit(main())
