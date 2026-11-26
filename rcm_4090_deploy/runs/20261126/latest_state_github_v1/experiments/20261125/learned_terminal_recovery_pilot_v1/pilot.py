"""One bounded local experiment; no tuning, external publication, dev20 or final test."""
import argparse
import ast
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path('/home/him1/mujoco_bc/rcm_4090_deploy')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from student_policy import FEATURE_NAMES, LearnedPolicy, measured_features

SOURCE = ROOT / 'data/homog_clean_replay_20261123_mixed_1600_600_12_hold_v2'
SHARDS = SOURCE / 'shards/tasks'
PAIR = ROOT / 'runs/20261125/output_head_sanity_v1/recovery_v2/paired'
UNIT = np.array([np.pi / 180, np.pi / 180, .001])
DT = .05
ALPHA = .01
CAPS = (.2, .1, .03)
TRAIN_IDS = (40, 41, 42)
HELDOUT_ID = 43
AXIAL = ('below_approach', 'below_recede', 'above_approach', 'above_recede')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(name, data):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()
        if isinstance(x, np.generic) else str(x)))
    temp.replace(path)


def record(stage, result):
    stamp = datetime.now().astimezone().isoformat()
    line = (f'\n- {stamp}: learned_terminal_recovery_pilot_v1 {stage}; '
            f'command/config={OUT}/experiment_card.json; result={result}; '
            f'output={OUT}. Local training-split pre-screen only; MuJoCo simulation only.\n')
    with (ROOT / 'TEST_LOG.md').open('ab') as stream:
        stream.write(line.encode())
    print(f'{stamp} {stage}: {result}', flush=True)


def task(index):
    return read(SHARDS / f'collection_E{index:02d}/E{index:02d}.json')['task']


def source_files():
    files = [Path(__file__), OUT/'student_policy.py', SOURCE/'metadata.json',
             SOURCE/'shards/contract.json', SOURCE/'dataset_receipt.json',
             ROOT/'configs/target_split_manifest.json',
             ROOT/'configs/v12_reachable_development_targets_0p35um.json',
             ROOT/'runs/20261125/train_direction_pair_v1/card.json']
    for i in (*TRAIN_IDS, HELDOUT_ID):
        base = SHARDS/f'collection_E{i:02d}'
        files += [base/f'E{i:02d}.json']
        if i in TRAIN_IDS:
            files += [base/'policy_trajectory.json', base/'frames.jsonl', base/'receipt.json']
    for direction in ('approach', 'recede'):
        files += [PAIR/f'{direction}_teacher_raw.json', PAIR/f'{direction}_teacher.json']
    return files


def prepare():
    if (OUT/'experiment_card.json').exists():
        raise FileExistsError('Refuse to replace an existing experiment card')
    metadata = read(SOURCE/'metadata.json')
    registered = read(SOURCE/'shards/contract.json')['groups']['collection']['tasks']
    split = read(ROOT/'configs/target_split_manifest.json')
    forbidden = np.asarray([r['point_mm'] for key in ('development_targets', 'final_test_targets')
                            for r in split[key]])
    reachable = read(ROOT/'configs/v12_reachable_development_targets_0p35um.json')['targets']
    dev = np.asarray([r.get('reachable_target_robot_mm', r['point_mm']) for r in reachable])
    targets = []
    for i in (*TRAIN_IDS, HELDOUT_ID):
        t = task(i)
        assert t['id'] == f'E{i:02d}' and t['episode_index'] == i
        np.testing.assert_array_equal(t['target_robot_mm'], metadata['episodes'][i]['target_robot_mm'])
        np.testing.assert_array_equal(t['target_robot_mm'], registered[i]['target_robot_mm'])
        point = np.asarray(t['target_robot_mm'])
        gap = float(np.linalg.norm(forbidden-point, axis=1).min()*1000)
        final_gap = float(np.linalg.norm(np.asarray([r['point_mm'] for r in split['final_test_targets']])-point, axis=1).min()*1000)
        dev_gap = float(np.linalg.norm(dev-point, axis=1).min()*1000)
        assert min(gap, dev_gap) > 1, (i, gap, dev_gap)
        targets.append(dict(t, role='fit' if i in TRAIN_IDS else 'isolated_training_split_prescreen',
            registry='homog_clean_replay_20261123_mixed_1600_600_12_hold_v2/shards/contract.json#collection',
            nearest_legacy_dev_or_final_um=gap, nearest_reachable_dev_um=dev_gap,
            nearest_final_um=final_gap))
    points = np.asarray([t['target_robot_mm'] for t in targets])
    gaps = np.linalg.norm(points[:, None]-points[None, :], axis=-1)*1000
    np.fill_diagonal(gaps, np.inf)
    assert gaps.min() > 1
    reuse = []
    for i in TRAIN_IDS:
        ep = read(SHARDS/f'collection_E{i:02d}/policy_trajectory.json')
        reuse += [dict(kind='lateral_recovery', target=f'E{i}', start=0, end=40,
                        path=str(SHARDS/f'collection_E{i}/policy_trajectory.json')),
                  dict(kind='hold', target=f'E{i}', start=400, end=600, selected_spaced_frames=16,
                        path=str(SHARDS/f'collection_E{i}/policy_trajectory.json'))]
        assert len(ep['trajectory']['actions']) == 600 and ep['final_error_mm']*1000 < 20
    for direction in ('approach', 'recede'):
        ep = read(PAIR/f'{direction}_teacher_raw.json')
        start = int(read(PAIR/f'{direction}_teacher.json')['step'])
        assert len(ep['trajectory']['actions']) == start+20
        reuse.append(dict(kind='below_'+direction, target='E40', start=start, end=start+20,
                          path=str(PAIR/f'{direction}_teacher_raw.json')))
    reuse_checks=[]
    for item in reuse:
        ep=read(item['path'])
        _,_,_,verified=segment(ep,item['start'],item['end'],item['target'],item['kind'],
                              item['path'],item['kind']=='hold')
        assert verified['retained_for_fit']
        reuse_checks.append(verified)
    audit = dict(status='completed', exact_whole_experiment_reusable=False,
        evaluation_reusable=False, reason='No existing twelve-feature learned ridge with E43 isolated three-start 240-cycle matched evaluation. Existing paired teacher continuations are E40 only and 20 cycles.',
        training_reuse=reuse, new_short_clips=10, total_training_segments=18,
        isolation='E43 target metadata may construct/evaluate starts; no E43 trajectory, action, OCT, statistics or normalization enters fit.',
        legacy_registry_note='The 2026-11-23 homogeneous collection is a separately registered training set; its E40-E43 are not claimed as entries of the older target_split_manifest training list.',
        target_pair_min_separation_um=float(gaps.min()), selected_segment_alignment_audits=reuse_checks)
    save('reuse_audit.json', audit)
    card = dict(name='learned_terminal_recovery_pilot_v1', hypothesis='Measured signed OCT, ABC, velocity and previous accepted command permit a fully learned local braking/recovery/hold policy.',
        output_dir=str(OUT), targets=targets, training_ids=['E40','E41','E42'], isolated_target='E43',
        initialization='none; no neural checkpoint loaded', model='standardized weighted linear ridge, all 39 output coefficients/intercepts learned',
        alpha=ALPHA, alpha_definition='min sum_i normalized_weight_i ||y_i - intercept - standardized_x_i W||^2 + 0.01 ||W||^2',
        features=FEATURE_NAMES, units='incremental A degrees, B degrees, C millimeters per 50 ms control tick',
        fixed_action_caps_deg_deg_mm=list(CAPS), action_caps_source='Existing homogeneous-replay collection and matched-pair evaluation caps, identical for student and teacher',
        data_budget=dict(reused_segments=8, new_segments=10, new_segment_cycles=40,
            total_segments=18, hold_subsample_per_target=16, stationary_weight_per_segment_max=.25),
        balance='equal total target weight; equal segment weight within target; dynamic/static split 0.75/0.25 when both present, equally within each class',
        setup='G10 executed prefix from goal C +/-90um, 22 approach ticks or 24 approach + 2 reverse ticks, +/-3um requested per tick; target coordinates used only for offline setup, audit and independent scoring',
        validation=dict(target='E43', starts=['below_recede','above_recede','lateral'],
            cycles=240, control_period_s=DT, student_alone=True, teacher_same_starts=True,
            lateral_start='Registered E43 T3 ABC-perturbed initial task, snapped by actual G10 initialization',
            stop_on_success=False, safety_streak_stop=None, legacy_strict_threshold_um=.35,
            gate='All 3 student last-200 post-action maximum errors <=20um AND same-start teacher last-200 maxima <=20um; all exactly240 ticks'),
        constraints=['No manual OCT-action signs, Jacobian, geometry or braking coefficients in student',
            'Teacher not constructed on student rollouts; no privileged coordinates or truth in student inputs',
            'No hyperparameter search, retries, dev20/final test, long-range evaluation, external publication or modification of R2 tasks'],
        budgets_s=dict(total=900, collection=300, fit=10, validation=600), workers=1,
        scene='RCMSceneV9, strict RGBD same-call OCT, production PolicyEvaluator.evaluate_episode feedback + MotorBackend G10',
        training_teacher='Raw same-observation geometry_action_from_observation with actual previous velocity, request capped by existing production caps; executed accepted actions are labels',
        reused_teacher='Original accepted executable bounded_teacher_intent_t3_settlement_v1 plus matched raw-teacher E40 branches; pre-acceleration labels are excluded',
        timestamps=dict(prepared=datetime.now().astimezone().isoformat()))
    save('experiment_card.json', card)
    from scripts.tools.experiment_workflow import scene_and_code
    save('input_fingerprints.json', dict(files={str(p):digest(p) for p in source_files()},
                                        runtime_identity=scene_and_code()))
    save('workflow_status.json', dict(status='prepared', stage='awaiting_single_gnome_launch',
         completed_segments=0, completed_local_rollouts=0, independent_standard_acceptance=False))
    record('reuse/preparation completed', dict(training=card['training_ids'], isolated='E43',
        reusable_segments=8, supplement_segments=10, validation_rollouts=6,
        exact_existing_evaluation=False))


def check_identity():
    identities = read(OUT/'input_fingerprints.json')
    for path, value in identities['files'].items():
        assert digest(path) == value, f'Changed input: {path}'
    from scripts.tools.experiment_workflow import scene_and_code
    assert scene_and_code() == identities['runtime_identity'], 'Scene/production-code changed'


def make_start(index, kind):
    from rcm.scene_v9 import RCMSceneV9
    scene = RCMSceneV9()
    t = task(index)
    world = np.asarray(scene.robot_to_world(np.asarray(t['target_robot_mm'])))
    scene.set_target_world(world)
    if kind == 'lateral':
        initial = np.asarray(t['initial_state_si'])
        prefix = []
    else:
        initial = scene.target_task(validate=True).copy()
        above = kind.startswith('above')
        sign = -1. if above else 1.
        initial[2] += .000090 if above else -.000090
        if kind.endswith('approach'):
            prefix = [np.array([0.,0.,sign*.003]) for _ in range(22)]
        else:
            prefix = ([np.array([0.,0.,sign*.003]) for _ in range(24)]
                      + [np.array([0.,0.,-sign*.003]) for _ in range(2)])
    scene.set_task(initial)
    scene.enable_motor_execution()
    scene.reset_motor_time(0.)
    for action in prefix:
        prepared = scene.motor_execution.preview(scene.current_task()+action*UNIT, DT)
        scene.apply_prepared_motor(prepared)
    prefix_records = scene.motor_execution.records.copy()
    previous = prefix[-1].copy() if prefix else np.zeros(3)
    return scene, world, previous, prefix_records


def boundary_check():
    source = (OUT/'student_policy.py').read_text()
    tree = ast.parse(source)
    forbidden = {'scene', 'teacher', 'target_world_mm', 'target_robot_mm', 'truth', 'jacobian'}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not (names & forbidden)
    model = OUT/'model.json'
    if model.exists():
        learned = LearnedPolicy(model)
        x = np.arange(12, dtype=float)*.001
        np.testing.assert_array_equal(learned.predict(x), learned.predict(x))
    result = dict(status='passed', explicit_measured_inputs=FEATURE_NAMES,
        pure_student_module=str(OUT/'student_policy.py'), student_sha256=digest(OUT/'student_policy.py'),
        ast_forbidden_symbols_present=sorted(names & forbidden),
        runtime_student_teacher_instances=0,
        runtime_control='get_action builds only signed OCT + evaluator supplied ABC + motor actual velocity + prior accepted action, then LearnedPolicy.predict',
        limit='Sensor perceives the scene/target marker as production does. Scoring/setup retain truth outside student; this is software boundary evidence, not an information-theoretic guarantee.')
    save('control_boundary_audit.json', result)
    return result


def rollout(index, kind, controller, cycles, label):
    from lerobot_training.eval_policy import PolicyEvaluator, StrictRGBDSensor, visual_servo_settings
    from scripts.research.homog_replay_common import QUALITY, make_teacher
    from scripts.research.clean_oct_workers import egl_identity

    scene, world, initial_previous, prefix = make_start(index, kind)
    teacher = sensor = None
    started = time.monotonic()
    control_inputs, teacher_labels, initial_observation = [], [], {}
    try:
        if controller == 'teacher':
            teacher = make_teacher(scene)
            sensor = teacher.sensor
        else:
            settings = visual_servo_settings(scene)
            settings.update(perception_backend='strict_rgbd', rgbd_render_quality=QUALITY.copy())
            sensor = StrictRGBDSensor(scene, settings)
        latest = {}
        original_observe = sensor.observe

        def observe_once():
            obs = original_observe()
            latest['obs'] = obs
            if not initial_observation:
                initial_observation['rgb_sha256'] = hashlib.sha256(np.ascontiguousarray(obs['rgb'])).hexdigest()
            return obs

        sensor.observe = observe_once

        class LocalEvaluator(PolicyEvaluator):
            # Reuse the production execution method without loading an unrelated neural policy.
            def __init__(self):
                self.use_oct = True
                self.measure_inference = False
                self.controller_variant = 'feedback'
                self.fine_evaluator = None
                self.learned = LearnedPolicy(OUT/'model.json') if controller == 'student' else None

            def reset_buffer(self, seed=None):
                self.previous = initial_previous.copy()
                self.control_mode_counts = {'learned':0, 'teacher_diagnostic':0}
                self.last_terminal_solution = None
                self.terminal_locked_target_world_mm = None

            def record_applied_action(self, action):
                self.previous = np.asarray(action, dtype=float).copy()

            def invalidate_action_chunk(self):
                pass

            def get_action(self, rgb, state, oct_feature):
                if not bool(oct_feature[3] > .5):
                    raise ValueError('Invalid measured OCT; no fallback controller allowed')
                velocity = scene.motor_execution.actual_velocity.copy()/UNIT
                x = measured_features(oct_feature[:3], state, velocity, self.previous)
                control_inputs.append(x.copy())
                if controller == 'student':
                    self.control_mode_counts['learned'] += 1
                    return self.learned.predict(x)
                q = np.asarray(state)*UNIT
                result = teacher.geometry_action_from_observation(q, latest['obs'],
                    previous_velocity=velocity*UNIT, command_q=q)
                if not result['valid']:
                    raise ValueError('Executable teacher produced invalid action')
                self.control_mode_counts['teacher_diagnostic'] += 1
                action = np.asarray(result['action_deg_deg_mm'], dtype=float)
                teacher_labels.append(action.copy())
                return action

        evaluator = LocalEvaluator()
        before = dict(state_deg_deg_mm=scene.current_task()/UNIT,
            actual_velocity_deg_deg_mm_s=scene.motor_execution.actual_velocity/UNIT,
            previous_accepted_deg_deg_mm=initial_previous,
            actual_encoder_ABCE=np.asarray(scene.motor_execution.backend.robot).copy(),
            truth_error_um=float(np.linalg.norm(scene.site_mm('needle_tip')-world)*1000),
            simulation_time_s=scene.motor_execution.time_s)
        ep = evaluator.evaluate_episode(scene, world, sensor=sensor, seed=434300+index,
            max_steps=cycles, stop_on_success=False, control_period_s=DT,
            action_limit_deg_deg_mm=CAPS,
            error_threshold_mm=.00035, max_consecutive_safety_limited=None,
            progress_every=40)
        ep['motor_records'] = ep['motor_records'][len(prefix):]
        ep.update(target_id=f'E{index}', start_kind=kind, controller=controller,
            diagnostic_teacher_only=controller=='teacher', prefix_motor_records=prefix,
            initial_boundary=before, initial_observation=initial_observation,
            measured_policy_inputs=control_inputs, teacher_action_same_tick=teacher_labels,
            requested_horizon=cycles, elapsed_s=time.monotonic()-started,
            student_teacher_instances=int(teacher is not None) if controller=='student' else None,
            egl=egl_identity())
        save(f'trajectories/{label}.json', ep)
        assert ep['early_stop_reason'] is None and len(ep['trajectory']['actions']) == cycles
        assert len(ep['motor_records']) == cycles
        # Replay the same pure prediction from the exact recorded twelve features.
        if controller == 'student':
            predicted = np.vstack([evaluator.learned.predict(x) for x in control_inputs])
            np.testing.assert_allclose(predicted, ep['trajectory']['raw_actions'], atol=1e-14, rtol=0)
        return ep
    finally:
        if sensor is not None:
            sensor.close()
        scene.close()


def segment(ep, start, end, target, name, source, hold=False):
    tr = ep['trajectory']
    assert end <= len(tr['actions'])
    states = np.asarray(tr['states'], float)
    actions = np.asarray(tr['actions'], float)
    velocities = np.asarray(tr['velocities'], float)
    octs = np.asarray(tr['oct_navigation_mm'], float)
    times = np.asarray(tr['simulation_time_s'], float)
    motor = ep['motor_records']
    assert len(motor) == len(actions)
    np.testing.assert_allclose(np.diff(times), DT, atol=1e-8, rtol=0)
    valid = np.asarray(tr['oct_navigation_valid'], bool)
    assert valid[start:end].all()
    features, labels, rows = [], [], []
    selected = np.linspace(start, end-1, 16, dtype=int) if hold else np.arange(start, end)
    for j in selected:
        m = motor[j]
        assert m['driver']['variant'] == 'g10'
        assert m['driver']['kinematics_sha256'] == digest(ROOT/'mujoco_model/vendor/simulation_driver/ros_msg/src/robot_control/kinematics.py')
        assert abs(m['start_time_s']-times[j]) < 1e-8
        np.testing.assert_allclose(m['actual_before_task_rad_rad_m'], states[j]*UNIT, atol=1e-11, rtol=0)
        np.testing.assert_allclose(m['requested_task_rad_rad_m'], (states[j]+actions[j])*UNIT, atol=1e-11, rtol=0)
        if j+1 < len(actions):
            np.testing.assert_allclose(m['actual_after_task_rad_rad_m'], states[j+1]*UNIT, atol=1e-11, rtol=0)
        else:
            np.testing.assert_allclose(m['actual_after_task_rad_rad_m'], np.asarray(ep['final_state_A_deg_B_deg_C_mm'])*UNIT, atol=1e-11, rtol=0)
        if j:
            np.testing.assert_allclose(velocities[j]*UNIT, motor[j-1]['actual_velocity_rad_rad_m_per_s'], atol=1e-11, rtol=0)
            previous = actions[j-1]
        else:
            previous = np.asarray(ep.get('initial_boundary', {}).get('previous_accepted_deg_deg_mm', [0.,0.,0.]))
        x = measured_features(octs[j], states[j], velocities[j], previous)
        post_error = float(1000*(tr['errors'][j+1] if j+1 < len(actions) else ep['final_error_mm']))
        row = dict(target_id=target, segment=name, source_path=str(source), source_frame=int(j),
            timestamp_s=float(times[j]), measured_input=x, accepted_teacher_action_deg_deg_mm=actions[j],
            pre_true_error_um=float(tr['errors'][j]*1000), post_true_error_um=post_error,
            actual_after_task_si=m['actual_after_task_rad_rad_m'],
            actual_encoder_after=m['actual_after_robot_ABCE'], hold=hold)
        features.append(x); labels.append(actions[j]); rows.append(row)
    post = np.asarray([r['post_true_error_um'] for r in rows])
    summary = dict(target_id=target, segment=name, frames=len(selected), source=str(source),
        start_frame=start, end_frame=end, initial_um=float(tr['errors'][start]*1000),
        final_um=float(post[-1]), post_max_um=float(post.max()),
        successful_terminal=bool(post[-1] <= 20), hold=hold,
        time_action_state_alignment='all selected rows verified against actual motor request and after-state',
        retained_for_fit=bool(post[-1] <= 20 and (not hold or post.max() <= 20)))
    return np.asarray(features), np.asarray(labels), rows, summary


def collect_training(state):
    started = time.monotonic()
    segments = []
    for i in TRAIN_IDS:
        source = SHARDS/f'collection_E{i}/policy_trajectory.json'
        ep = read(source)
        segments.append(segment(ep,0,40,f'E{i}','lateral_recovery',source))
        segments.append(segment(ep,400,600,f'E{i}','hold',source,True))
        for kind in AXIAL:
            if i == 40 and kind.startswith('below'):
                direction = kind.split('_')[1]
                source = PAIR/f'{direction}_teacher_raw.json'
                ep = read(source)
                start = int(read(PAIR/f'{direction}_teacher.json')['step'])
                # Motor-only setup parity: same real state, velocity and previous accepted action.
                scene, _, previous, prefix = make_start(i, kind)
                try:
                    np.testing.assert_allclose(scene.current_task()/UNIT, ep['trajectory']['states'][start], atol=1e-8,rtol=0)
                    np.testing.assert_allclose(scene.motor_execution.actual_velocity/UNIT, ep['trajectory']['velocities'][start], atol=1e-8,rtol=0)
                    np.testing.assert_allclose(previous, ep['trajectory']['actions'][start-1], atol=1e-12,rtol=0)
                finally:
                    scene.close()
                segments.append(segment(ep,start,start+20,f'E{i}',kind,source))
            else:
                name = f'train_E{i}_{kind}'
                ep = rollout(i,kind,'teacher',40,name)
                source = OUT/f'trajectories/{name}.json'
                segments.append(segment(ep,0,40,f'E{i}',kind,source))
            state.update(stage='training_teacher_collection', completed_segments=len(segments),
                         last_segment=segments[-1][-1], elapsed_s=time.monotonic()-state['monotonic_start'])
            save('workflow_status.json',state)
            record('training segment audited', segments[-1][-1])
            if time.monotonic()-started > 300:
                raise TimeoutError('300s collection budget exhausted; no training or validation')
    assert len(segments) == 18
    save('segment_audit.json', [s[-1] for s in segments])
    accepted = [s for s in segments if s[-1]['retained_for_fit']]
    assert {s[-1]['target_id'] for s in accepted} == {'E40','E41','E42'}
    assert any(s[-1]['hold'] for s in accepted)
    xs, ys, ws, all_rows = [], [], [], []
    for target in ('E40','E41','E42'):
        group = [s for s in accepted if s[-1]['target_id']==target]
        for x,y,rows,summary in group:
            dynamic = np.any(np.abs(y)>1e-9,axis=1)
            weights = np.empty(len(y))
            if dynamic.any() and (~dynamic).any():
                weights[dynamic] = .75/dynamic.sum()
                weights[~dynamic] = .25/(~dynamic).sum()
            else:
                weights[:] = 1/len(y)
            weights /= 3*len(group)
            xs.append(x);ys.append(y);ws.append(weights);all_rows.extend(rows)
    X,Y,W = np.vstack(xs),np.vstack(ys),np.concatenate(ws)
    assert abs(W.sum()-1)<1e-12
    assert all(r['target_id']!='E43' for r in all_rows)
    counts = Counter()
    for x,y in zip(X,Y):
        counts['signed_OCT_Z_positive' if x[2]>0 else 'signed_OCT_Z_negative' if x[2]<0 else 'signed_OCT_Z_zero'] += 1
        counts['C_velocity_positive' if x[8]>1e-6 else 'C_velocity_negative' if x[8]<-1e-6 else 'C_velocity_stationary'] += 1
        counts['nonzero_accepted_action' if np.any(abs(y)>1e-9) else 'zero_accepted_action'] += 1
        counts['lateral_OCT_over_5um' if np.linalg.norm(x[:2])*1000>5 else 'lateral_OCT_le_5um'] += 1
        counts['samples'] += 1
    for row in all_rows:
        counts['teacher_post_improves' if row['post_true_error_um'] < row['pre_true_error_um']-.001 else 'teacher_post_equal_or_worse'] += 1
    assert all(counts[k]>0 for k in ('signed_OCT_Z_positive','signed_OCT_Z_negative','C_velocity_positive','C_velocity_negative'))
    # Low-cost independent truth/units check on fixed selected frames; audit only.
    from rcm.scene_v9 import RCMSceneV9
    scene = RCMSceneV9()
    max_score_difference = 0.
    try:
        for j in np.linspace(0,len(all_rows)-1,24,dtype=int):
            row = all_rows[j]
            t = task(int(row['target_id'][1:]))
            world = np.asarray(scene.robot_to_world(t['target_robot_mm']))
            scene.set_task(row['actual_after_task_si'])
            actual = np.linalg.norm(scene.site_mm('needle_tip')-world)*1000
            max_score_difference=max(max_score_difference,abs(actual-row['post_true_error_um']))
        assert max_score_difference < .05, max_score_difference
    finally:
        scene.close()
    audit = dict(status='passed', categories=dict(counts), total_rows=len(X), successful_segments=len(accepted),
        excluded_failed_segments=[s[-1] for s in segments if not s[-1]['retained_for_fit']],
        weights_per_target={t:float(sum(w for w,r in zip(W,all_rows) if r['target_id']==t)) for t in ('E40','E41','E42')},
        weighted_stationary_fraction=float(W[np.all(abs(Y)<=1e-9,axis=1)].sum()),
        independent_truth_spotcheck_frames=24, max_truth_score_difference_um=max_score_difference,
        labels='Exactly executed/accepted teacher command; no pre-acceleration or zero-velocity relabels',
        sample_alignment='OCT/state/velocity/previous action at t; teacher accepted action at t; actual after-state and score at t+0.05s',
        cpu_reason='Twelve-variable analytic ridge and data audit are small CPU work; actual rendering uses RTX4090 NVIDIA EGL')
    save('data_audit.json',audit)
    save('training_samples.json',all_rows)
    with (OUT/'training_arrays.npz').open('wb') as f:
        np.savez_compressed(f,X=X,Y=Y,W=W)
    record('training data gate passed',audit)
    return X,Y,W


def fit_once(X,Y,W):
    started=time.monotonic()
    mean=W@X
    scale=np.sqrt(W@((X-mean)**2))
    scale=np.where(scale>1e-12,scale,1.)
    Z=(X-mean)/scale
    intercept=W@Y
    coefficients=np.linalg.solve((Z.T*W)@Z + ALPHA*np.eye(12), (Z.T*W)@(Y-intercept))
    pred=Z@coefficients+intercept
    mse=np.sum(W[:,None]*(pred-Y)**2,axis=0)
    model=dict(model_type='weighted_standardized_linear_ridge',alpha=ALPHA,
        feature_names=FEATURE_NAMES,input_mean=mean,input_scale=scale,
        standardized_coefficients_12x3=coefficients,intercept_deg_deg_mm=intercept,
        raw_input_coefficients_12x3=coefficients/scale[:,None],
        raw_input_intercept_deg_deg_mm=intercept-(mean/scale)@coefficients,
        output_names=['dA_deg','dB_deg','dC_mm'],fit_target_ids=['E40','E41','E42'],
        isolated_target_ids=['E43'],training_arrays_sha256=digest(OUT/'training_arrays.npz'),
        weighted_training_rmse_deg_deg_mm=np.sqrt(mse), training_elapsed_s=time.monotonic()-started,
        number_fits=1,validation_used_for_selection=False,fit_hyperparameter_search=False,
        setup_truth_or_teacher_inputs=False)
    assert np.isfinite(coefficients).all() and np.isfinite(intercept).all()
    save('model.json',model)
    save('training_result.json',dict(status='completed',model_sha256=digest(OUT/'model.json'),
        input_count=12,learned_scalar_parameters=39,training_rows=len(X),
        elapsed_s=time.monotonic()-started,weighted_training_rmse_deg_deg_mm=np.sqrt(mse)))
    boundary_check()
    record('single fixed ridge fit completed',read(OUT/'training_result.json'))


def metrics(ep):
    tr=ep['trajectory']
    errors=np.r_[np.asarray(tr['errors'],float),ep['final_error_mm']]*1000
    post=errors[1:]
    best=run=0
    for e in post:
        run=run+1 if e<=20 else 0
        best=max(best,run)
    enter=np.flatnonzero(errors<=20)
    exits=np.flatnonzero((errors[:-1]<=20)&(errors[1:]>20))+1
    exit_rows=[]
    for cycle in exits:
        exit_rows.append(dict(cycle=int(cycle),time_since_release_s=float(cycle*DT),
            error_um=float(errors[cycle]),
            raw_action_deg_deg_mm=tr['raw_actions'][cycle-1],
            accepted_action_deg_deg_mm=tr['actions'][cycle-1],
            actual_after_encoder_ABCE=ep['motor_records'][cycle-1]['actual_after_robot_ABCE']))
    z=np.asarray(tr['oct_navigation_mm'])[:,2]
    crossing=(np.flatnonzero(z[:-1]*z[1:]<0)+1).tolist()
    return dict(target_id=ep['target_id'],start=ep['start_kind'],controller=ep['controller'],
        diagnostic_teacher_only=ep['diagnostic_teacher_only'],cycles=len(post),
        initial_true_error_um=float(errors[0]),longest_contiguous_le20_cycles=int(best),
        last200_max_true_error_um=float(post[-200:].max()) if len(post)>=200 else None,
        final_true_3d_error_um=float(post[-1]),min_true_3d_error_um=float(errors.min()),
        first_enter_cycle=int(enter[0]) if len(enter) else None,
        first_exit=exit_rows[0] if exit_rows else None,all_exits=exit_rows,
        signed_oct_z_crossing_preaction_indices=crossing,
        final_raw_action_deg_deg_mm=tr['raw_actions'][-1],
        final_accepted_action_deg_deg_mm=tr['actions'][-1],
        final_velocity_deg_deg_mm_s=ep['final_velocity_deg_deg_mm_per_s'],
        encoder_settled_at_exit=ep['terminal_success']['encoder_settled_at_exit'],
        elapsed_s=ep['elapsed_s'],early_stop_reason=ep['early_stop_reason'],
        last200_le20=bool(len(post)==240 and post[-200:].max()<=20),
        strict_0p35um_success=bool(ep['success']),
        trace_note='Trajectory JSON contains every pre-action OCT/state/velocity/input, raw and accepted ABC action, G10 request/quantization/actual-after state; final post-action state included.')


def validate(state):
    rows=[]
    for kind in ('below_recede','above_recede','lateral'):
        student=rollout(HELDOUT_ID,kind,'student',240,f'isolated_E43_{kind}_student')
        sm=metrics(student);rows.append(sm)
        save('local_metrics.json',rows)
        record('isolated local rollout completed',sm)
        state.update(stage='local_validation',completed_local_rollouts=len(rows),elapsed_s=time.monotonic()-state['monotonic_start'])
        save('workflow_status.json',state)
        diagnostic=rollout(HELDOUT_ID,kind,'teacher',240,f'isolated_E43_{kind}_teacher')
        # Pairing is exact in the executable initial state and motor history, not just distance.
        for key in ('state_deg_deg_mm','actual_velocity_deg_deg_mm_s','previous_accepted_deg_deg_mm','actual_encoder_ABCE'):
            np.testing.assert_array_equal(student['initial_boundary'][key],diagnostic['initial_boundary'][key])
        assert student['prefix_motor_records']==diagnostic['prefix_motor_records']
        assert student['initial_observation']==diagnostic['initial_observation']
        np.testing.assert_array_equal(student['trajectory']['oct_navigation_mm'][0],diagnostic['trajectory']['oct_navigation_mm'][0])
        tm=metrics(diagnostic);tm['matched_executable_start_verified']=True
        rows.append(tm);save('local_metrics.json',rows)
        record('matched diagnostic teacher completed',tm)
        state.update(completed_local_rollouts=len(rows),elapsed_s=time.monotonic()-state['monotonic_start'])
        save('workflow_status.json',state)
    return rows


def report(rows,state):
    students=[r for r in rows if r['controller']=='student']
    teachers=[r for r in rows if r['controller']=='teacher']
    candidate=bool(len(students)==len(teachers)==3 and all(r['last200_le20'] for r in rows))
    result=dict(status='completed',local_feasibility_candidate=candidate,
        student_local_last200_passes=sum(r['last200_le20'] for r in students),student_local_cases=len(students),
        teacher_diagnostic_last200_passes=sum(r['last200_le20'] for r in teachers),teacher_cases=len(teachers),
        standard_full_trajectory_independent_acceptance=False,dev20_run=False,final_test_run=False,
        no_further_expansion=True,model_sha256=digest(OUT/'model.json'),
        actual_total_elapsed_s=time.monotonic()-state['monotonic_start'],
        isolated_target='E43',training_targets=['E40','E41','E42'],results=rows,
        teacher_failures=[r for r in teachers if not r['last200_le20']],
        interpretation='Only local isolated training-target feasibility. A failed lateral teacher invalidates comprehensive recovery claims; no case is dropped. No claim about ordinary initial positions, dev20, final test, clinical or hardware precision.')
    save('summary.json',result)
    lines=['# 学习终末恢复最小实验 v1','',
        f"流程完成；局部候选：{candidate}。学生末200全≤20µm：{result['student_local_last200_passes']}/3；教师诊断：{result['teacher_diagnostic_last200_passes']}/3。",
        '','训练：既有训练登记 E40/E41/E42；整目标隔离 E43。此处 ID 归属 homogeneous 20261123 collection registry，未冒用旧 frozen split ID。',
        f"模型 SHA256：`{result['model_sha256']}`。12输入、39个学习系数/截距，标准化 ridge α=0.01，仅拟合一次。动作单位 dA[deg], dB[deg], dC[mm]/50ms。",
        '', '| 起点 | 控制器 | 最长连续≤20周期 | 末200最大µm | 末步3D µm | 首次退出周期/秒 |',
        '|---|---|---:|---:|---:|---|']
    for r in rows:
        ex=r['first_exit'];where=f"{ex['cycle']}/{ex['time_since_release_s']:.3f}" if ex else '无（未进入时见指标）'
        lines.append(f"|{r['start']}|{r['controller']}|{r['longest_contiguous_le20_cycles']}|{r['last200_max_true_error_um']:.6f}|{r['final_true_3d_error_um']:.6f}|{where}|")
    lines += ['', '所有原始失败与侧向挑战均保留。教师仅作诊断正对照，不计学生成功。未运行dev20、final_test或常规初始位置独立全程。',
        f"实际耗时 {result['actual_total_elapsed_s']:.2f}s；单进程顺序执行，未进行超参搜索或后续扩展。",
        '', '查看：summary.json、local_metrics.json、model.json、data_audit.json、segment_audit.json、reuse_audit.json、control_boundary_audit.json、host_preflight.json、workflow_status.json、trajectories/。',
        'model.json 给出全部原单位/标准化系数与输入；local_metrics.json 给出首次退出动作，轨迹给出逐周期全部ABC动作。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    record('workflow completed',dict(local_feasibility_candidate=candidate,
        student_passes=result['student_local_last200_passes'],teacher_passes=result['teacher_diagnostic_last200_passes'],
        elapsed_s=result['actual_total_elapsed_s'],summary=str(OUT/'summary.json'),no_standard_acceptance=True))


def run():
    previous=read(OUT/'workflow_status.json')
    if previous['status']!='prepared':
        raise ValueError('Only a prepared, never-started experiment may run')
    start=time.monotonic()
    state=dict(status='running',stage='host_preflight',pid=os.getpid(),ppid=os.getppid(),
        start_time=datetime.now().astimezone().isoformat(),monotonic_start=start,
        completed_segments=0,completed_local_rollouts=0,independent_standard_acceptance=False)
    save('workflow_status.json',state)
    def interrupted(signum,_frame):
        raise KeyboardInterrupt(f'signal {signum}')
    for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(s,interrupted)
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('900s total walltime cap')))
    signal.alarm(900)
    try:
        check_identity()
        from scripts.research.homog_replay_common import gpu_check
        from rcm.scene_v9 import RCMSceneV9
        gpu=gpu_check()
        processes=subprocess.run(['ps','-eo','pid,ppid,etime,pcpu,comm,args'],capture_output=True,text=True,check=True).stdout
        selected=[line[:1000] for line in processes.splitlines() if any(k in line for k in ('r2_hold220','recovery_v2','fast5','learned_terminal_recovery'))]
        probe=RCMSceneV9();probe.close()
        host=dict(status='passed',gpu=gpu,python=sys.executable,
            environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','MUJOCO_GL','MUJOCO_EGL_DEVICE_ID','RCM_DRIVER_VARIANT')},
            relevant_process_snapshot=selected,protected_tasks_touched=False,
            pid=os.getpid(),ppid=os.getppid(),proc_start_time=datetime.now().astimezone().isoformat())
        save('host_preflight.json',host)
        boundary_check();record('host preflight passed',host)
        X,Y,W=collect_training(state)
        state['stage']='single_fixed_ridge_fit';save('workflow_status.json',state)
        fit_once(X,Y,W)
        state['stage']='isolated_local_validation';save('workflow_status.json',state)
        rows=validate(state)
        check_identity()
        report(rows,state)
        state.update(status='completed',stage='finished',elapsed_s=time.monotonic()-start,
            summary=str(OUT/'summary.json'),local_feasibility_candidate=read(OUT/'summary.json')['local_feasibility_candidate'])
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
            error=repr(exc),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-start,
            dependent_work_stopped=True,no_retry=True)
        save('failure.json',state)
        record('workflow '+state['status'],dict(stage=state['stage'],error=repr(exc),
            completed_segments=state['completed_segments'],completed_local_rollouts=state['completed_local_rollouts'],
            elapsed_s=state['elapsed_s']))
        raise
    finally:
        signal.alarm(0)
        state.pop('monotonic_start',None)
        state['end_time']=datetime.now().astimezone().isoformat()
        save('workflow_status.json',state)
        print('RESULT DIRECTORY:',OUT,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','run','terminal_failure'])
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--exit-code',type=int)
    args=parser.parse_args()
    assert args.output_dir.resolve()==OUT
    if args.command=='prepare':
        prepare()
    elif args.command=='run':
        run()
    else:
        current=read(OUT/'workflow_status.json')
        if current['status'] not in ('completed','failed','interrupted'):
            current.update(status='interrupted_or_launch_failed',terminal_exit_code=args.exit_code)
            save('workflow_status.json',current)
            record('terminal failed/interrupted',current)
