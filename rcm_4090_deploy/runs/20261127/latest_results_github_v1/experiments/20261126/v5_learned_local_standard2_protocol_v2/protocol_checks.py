"""Read-only archival checks outside policy control; bit-exact protocol gates."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
ARCHIVE=ROOT/'runs/20261125/terminal_round_v1/v5_A4000'
V5_CAPS=(0.2897999882698059,0.14434358477592468,0.04271206632256508)
LOCAL_CAPS=(.2,.1,.03)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()


def active_workers():
    found=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name)==os.getpid(): continue
        try:
            argv=[v.decode(errors='replace') for v in (process/'cmdline').read_bytes().split(b'\0') if v]
            if not argv or 'python' not in Path(argv[0]).name: continue
            if 'worker' in argv or any(Path(v).name in ('eval_policy.py','homog_replay_evaluate.py') for v in argv):
                found.append(dict(pid=int(process.name),argv=argv))
        except OSError: pass
    return found


def check_sources():
    from lerobot_training.eval_policy import DEFAULT_ACTION_LIMIT_DEG_DEG_MM
    assert tuple(DEFAULT_ACTION_LIMIT_DEG_DEG_MM)==V5_CAPS
    audit=read(OUT/'v1_deviation_audit.json')
    for value in audit['targets'].values():
        contract=value['archive_contract']
        assert tuple(contract['action_limit_deg_deg_mm'])==V5_CAPS
        assert contract['flow_integration_steps']==4 and contract['num_action_samples']==1
        assert contract['action_aggregation']=='single_sample'
    workers=active_workers()
    assert len(workers)+2<=12, f'Worker concurrency would exceed12: {workers}'
    return dict(active_workers=workers,new_workers=2,total_upper_bound=len(workers)+2,
        production_factory='scripts.research.homog_replay_evaluate:EvaluationWorker (inherits a_axis_flip_common.EvaluationWorker.__init__)',
        production_episode='PolicyEvaluator.evaluate -> evaluate_episode with DEFAULT_ACTION_LIMIT_DEG_DEG_MM',
        samples=1,execute_steps=1,flow_integration_steps=4,aggregation='single_sample',v5_caps=V5_CAPS,local_caps=LOCAL_CAPS)


def compare(folder,index,v5_count=None):
    folder=Path(folder)
    ep=read(folder/'evaluation_results.json')
    row=read(ARCHIVE/f'E{index:02d}.json')
    assert digest(row['result_path'])==row['result_sha256']
    archive=read(row['result_path'])
    old=archive['episodes'][0]
    assert old['seed']==42+index and ep['seed']==42+index
    count=ep['route_counts']['v5'] if v5_count is None else v5_count
    checks={}
    def equal(name,a,b):
        a=np.asarray(a)
        b=np.asarray(b)
        passed=bool(a.shape==b.shape and np.array_equal(a,b))
        difference=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.shape==b.shape and a.size else 0.
        mismatch=None
        if a.shape==b.shape and not passed:
            locations=np.argwhere(a!=b)
            mismatch=locations[0].tolist() if len(locations) else None
        checks[name]=dict(exact_equal=passed,max_absolute_difference=difference,first_mismatch_index=mismatch)
    equal('target_world_mm',ep['target_world_mm'],old['target_world_mm'])
    for key in ('raw_actions','actions','states','errors','velocities','oct_navigation_mm','oct_navigation_valid','simulation_time_s'):
        end=count if key in ('raw_actions','actions') else min(len(ep['trajectory'][key]),count+1)
        equal(key,ep['trajectory'][key][:end],old['trajectory'][key][:end])
    if count < len(ep['trajectory']['states']):
        state_at_boundary=ep['trajectory']['states'][count]
    else:
        state_at_boundary=ep['final_state_A_deg_B_deg_C_mm']
    old_boundary=(old['trajectory']['states'][count] if count<len(old['trajectory']['states']) else old['final_state_A_deg_B_deg_C_mm'])
    equal('state_immediately_before_route_or_end',state_at_boundary,old_boundary)
    motor_fields=('requested_task_rad_rad_m','commanded_task_rad_rad_m','requested_robot_ABCE','handles_ABCE',
        'quantized_target_robot_ABCE','actual_before_robot_ABCE','actual_after_robot_ABCE',
        'actual_before_task_rad_rad_m','actual_after_task_rad_rad_m','actual_velocity_rad_rad_m_per_s',
        'start_time_s','end_time_s','duration_s','integration_steps')
    for field in motor_fields:
        equal('motor_'+field,[r[field] for r in ep['motor_records'][:count]],[r[field] for r in old['motor_records'][:count]])
    with (folder/'cycles.jsonl').open() as f:
        cycles=[json.loads(line) for line in f]
    equal('model_raw_action', [r['model_raw_action_deg_deg_mm'] for r in cycles[:count]],old['trajectory']['raw_actions'][:count])
    expected_clipped=np.clip(np.asarray(old['trajectory']['raw_actions'][:count]),-np.asarray(V5_CAPS),V5_CAPS)
    equal('clipped_action',[r['clipped_action_deg_deg_mm'] for r in cycles[:count]],expected_clipped)
    equal('accepted_action',[r['accepted_action_deg_deg_mm'] for r in cycles[:count]],old['trajectory']['actions'][:count])
    expected_post=np.r_[old['trajectory']['errors'][1:],old['final_error_mm']][:count]*1000
    equal('post_action_independent_truth_error_um',[r['post_true_error_um'] for r in cycles[:count]],expected_post)
    units=np.array([180/np.pi,180/np.pi,1000.])
    expected_actual=[(np.asarray(r['actual_after_task_rad_rad_m'])-r['actual_before_task_rad_rad_m'])*units for r in old['motor_records'][:count]]
    equal('actual_motion_delta_deg_deg_mm',[r['actual_motion_delta_deg_deg_mm'] for r in cycles[:count]],expected_actual)
    return dict(status='passed' if all(v['exact_equal'] for v in checks.values()) else 'protocol_failed',
        compared_v5_actions=count,comparison='np.array_equal; no atol/rtol acceptance',checks=checks,
        archive_used_for_control=False,source=row['result_path'],source_sha256=row['result_sha256'],
        original_contract=archive['evaluation_contract']),old


def validate_prefix():
    result,_=compare(OUT/'prefix_gate/E08',8,v5_count=24)
    result.update(seed=50,cycles=24,source_hashes={str(OUT/name):digest(OUT/name) for name in ('standard2.py','protocol_checks.py','launch.sh')},
                  v1_first_accepted_difference_cycle=7,v1_first_raw_difference_cycle=8)
    return result
