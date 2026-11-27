"""Read-only v2 trajectory/teacher audit with actual pinned G10 conversions."""
import csv
import json
from collections import Counter
import numpy as np
import campaign as c

OLD=c.ROOT/'runs/20261127/dit_relative_v2_resume'
def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def run():
    from rcm.scene_v9 import RCMSceneV9
    from motor_backend import MotorBackend
    from driver_loader import driver_identity
    folder=c.OUT/'audit';folder.mkdir(exist_ok=True)
    unit=np.array([np.pi/180,np.pi/180,.001])
    scene=RCMSceneV9()
    standards={}; evidence=[]; thresholds=[]
    try:
        for idx in (0,2,6,8,10):
            target=f'E{idx:02d}';data=rows(OLD/'candidate1/standard'/target/'cycles.jsonl')
            terminal=data[-200:]
            for r in terminal:
                m=r['event']['motor_record'];a=np.array(r['event']['accepted_action_deg_deg_mm'])
                evidence.append(dict(target=target,cycle=r['cycle'],error_um=r['post']['true_error_um'],
                    OCT_mm=r['pre']['signed_OCT_mm'],accepted=a.tolist(),
                    requested=m['requested_robot_ABCE'],quantized=m['quantized_target_robot_ABCE'],
                    before=m['actual_before_robot_ABCE'],after=m['actual_after_robot_ABCE']))
            metrics=c.read(OLD/'candidate1/standard'/target/'summary.json')['metrics']
            q=np.array(terminal[-1]['pre']['state_deg_deg_mm'])*unit
            m=terminal[-1]['event']['motor_record'];before=m['actual_before_robot_ABCE']
            anchor=data[0]['event']['motor_record']['actual_before_robot_ABCE']
            backend=MotorBackend(initial_robot=tuple(anchor))
            def encode(delta):
                desired=tuple(round(float(v)) for v in scene.encoders(q+delta*unit))+(0,)
                command=backend.command_for_robot_target(desired)
                handles=tuple(getattr(command,'handleEncoder'+a) for a in 'ABCE')
                return backend.kin.validate_manual_jog(tuple(anchor),handles)
            bound=[]
            for axis in range(3):
                pair=[]
                for sign in (-1,1):
                    lo=0.;hi=[.03,.02,.004][axis]
                    for _ in range(48):
                        mid=(lo+hi)/2;delta=np.zeros(3);delta[axis]=sign*mid
                        if encode(delta)[axis]==before[axis]:lo=mid
                        else:hi=mid
                    pair.append(sign*hi)
                bound.append(pair)
            stationary=sum(r['event']['motor_record']['actual_before_robot_ABCE']==r['event']['motor_record']['actual_after_robot_ABCE'] for r in terminal)
            quantized_hold=sum(r['event']['motor_record']['quantized_target_robot_ABCE']==r['event']['motor_record']['actual_before_robot_ABCE'] for r in terminal)
            accepted=np.array([r['event']['accepted_action_deg_deg_mm'] for r in terminal])
            xyz=np.mean([r['post']['true_error_xyz_um'] for r in terminal],axis=0)
            octmean=np.mean([r['pre']['signed_OCT_mm'][:3] for r in terminal],axis=0)*1000
            standards[target]=dict(metrics=metrics,last200_stationary=stationary,last200_quantized_target_equals_current=quantized_hold,
                nonzero_action_cycles=int(np.sum(np.any(accepted!=0,axis=1))),mean_accepted_deg_deg_mm=accepted.mean(0).tolist(),
                mean_error_xyz_um=xyz.tolist(),mean_OCT_xyz_um=octmean.tolist(),
                signed_threshold_from_terminal_pose_deg_deg_mm=bound,anchor_ABCE=anchor)
            thresholds.append(dict(target=target,current_ABCE=before,thresholds_deg_deg_mm=bound))
    finally:scene.close()
    bins=Counter();samples=[];max_lateral={};summary_count=0
    for path in sorted((OLD/'data').glob('*/summary.json')):
        s=c.read(path)
        if s.get('status')!='completed' or s.get('branch',{}).get('target') not in ('E40','E42'):continue
        summary_count+=1;spec=s['branch'];obs=rows(path.parent/'observations.jsonl');actions=rows(path.parent/'actions.jsonl')
        for action in actions:
            if action['phase']!='teacher':continue
            pre=obs[action['pre_observation']];m=action['motor_record'];oct4=np.array(pre['signed_OCT_mm']);lateral=float(np.linalg.norm(oct4[:2])*1000)
            key=spec['target'];max_lateral[key]=max(max_lateral.get(key,0),lateral)
            norm=float(np.linalg.norm(oct4[:3])*1000);hold=m['actual_before_robot_ABCE']==m['actual_after_robot_ABCE'];qh=m['quantized_target_robot_ABCE']==m['actual_before_robot_ABCE']
            band='lateral20to100' if 20<=lateral<=100 else 'norm20to100' if 20<=norm<=100 else 'other'
            bins[(key,band,'stationary' if hold else 'moving','quantized_hold' if qh else 'cross_threshold')]+=1
            if band!='other' and len(samples)<50:samples.append(dict(branch=spec['id'],observation=pre,action=action))
    result=dict(status='completed_read_only',source=str(OLD),driver=driver_identity(),standard_last200=standards,
        teacher_complete_branches=summary_count,teacher_max_OCT_lateral_um=max_lateral,
        teacher_bins=[dict(target=k[0],band=k[1],motion=k[2],encoding=k[3],count=v) for k,v in sorted(bins.items())],
        no_new_motor_ticks=True,threshold_method='binary search exact scene.encoders -> original G10 cumulative handle conversion -> validate_manual_jog; session anchor from cycle1',
        limitation='Teacher OCT lateral uses sensor x/y; independent world XYZ is reported separately. Historical driver fingerprint is verified per cycle.')
    c.save(folder/'quantization_audit.json',result);c.save(folder/'last200_perstep_evidence.json',evidence)
    c.save(folder/'teacher_examples.json',samples)
    for target,stats in standards.items():
        print(target,json.dumps(stats,ensure_ascii=False),flush=True)
    c.record('v2_read_only_quantization_and_label_audit',result)
    return result

if __name__=='__main__':run()
