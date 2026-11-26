"""Construct independent v2; never mutate the v1 processes or evidence."""
from pathlib import Path
import ast

OUT=Path(__file__).resolve().parent
V1=OUT.parent/'v5_learned_local_standard2_v1'
s=(V1/'standard2.py').read_text()
def replace(old,new):
    global s
    assert s.count(old)==1,(s.count(old),old[:100])
    s=s.replace(old,new)
replace('v5_learned_local_standard2_v1','v5_learned_local_standard2_protocol_v2') if s.count('v5_learned_local_standard2_v1')==1 else None
# The experiment name occurs in both the log and card.
s=s.replace('v5_learned_local_standard2_v1','v5_learned_local_standard2_protocol_v2')
replace("sys.path.insert(0, str(LOCAL))","sys.path.insert(0, str(LOCAL))\nfrom protocol_checks import V5_CAPS, LOCAL_CAPS, check_sources, compare, validate_prefix, active_workers")
replace("    import torch\n    from scripts.research.homog_replay_common import check_identity","    import torch\n    source_check=check_sources()\n    from scripts.research.homog_replay_common import check_identity")
replace("    files=[Path(__file__),CHECKPOINT", "    files=[Path(__file__),OUT/'protocol_checks.py',OUT/'launch.sh',CHECKPOINT")
replace("              action_caps_deg_deg_mm=[.2,.1,.03],", "              action_caps_deg_deg_mm=V5_CAPS,local_branch_caps_deg_deg_mm=LOCAL_CAPS,")
replace("    save(OUT/'experiment_card.json',card)","    card['production_source_check']=source_check\n    card['formal_acceptance_requires']='Bit-exact all pre-route raw/clipped/accepted/actual actions, state, OCT, errors and timing match archive; any mismatch is protocol failure'\n    card['v1_classification']='Restricted-action control, not standard-v5 acceptance; v1 unchanged'\n    save(OUT/'experiment_card.json',card)")
replace("def worker(index):\n    folder=OUT/f'E{index:02d}'\n    folder.mkdir(exist_ok=False)","def worker(index, steps=1600, prefix=False):\n    folder=(OUT/'prefix_gate'/f'E{index:02d}') if prefix else OUT/f'E{index:02d}'\n    folder.mkdir(parents=True,exist_ok=False)")
replace("                    action=self.learned.predict(x)\n                    mode='learned_local'", "                    model_raw=self.learned.predict(x)\n                    action=np.clip(model_raw,-np.asarray(LOCAL_CAPS),LOCAL_CAPS)\n                    mode='learned_local'")
replace("                    action=super().get_action(rgb,state,oct_feature)\n                    mode='v5'", "                    action=super().get_action(rgb,state,oct_feature)\n                    model_raw=action.copy()\n                    mode='v5'")
replace("                self.latest=dict(mode=mode,measured_input12=x.copy(),oct_valid=valid,oct_norm_mm=norm,", "                self.latest=dict(mode=mode,model_raw_action_deg_deg_mm=model_raw.copy(),branch_output_action_deg_deg_mm=action.copy(),\n                                 active_branch_caps_deg_deg_mm=LOCAL_CAPS if mode=='learned_local' else V5_CAPS,\n                                 measured_input12=x.copy(),oct_valid=valid,oct_norm_mm=norm,")
replace("        assert evaluator.fine_evaluator is None", "        assert evaluator.fine_evaluator is None\n        assert evaluator.flow_integration_steps==4 and evaluator.num_action_samples==1 and evaluator.n_action_execute_steps==1\n        save(folder/'inference_protocol.json',dict(policy_type=evaluator.policy_type,flow_integration_steps=evaluator.flow_integration_steps,\n            samples=evaluator.num_action_samples,execute_steps=evaluator.n_action_execute_steps,device=str(evaluator.device),\n            production_caps=V5_CAPS,local_branch_caps=LOCAL_CAPS,seed=42+index))")
replace("            row=dict(action_cycle=event['step']+1,**evaluator.latest,**event,", "            motor=event['motor_record']\n            actual_motion=(np.asarray(motor['actual_after_task_rad_rad_m'])-motor['actual_before_task_rad_rad_m'])*TASK_TO_DEG_DEG_MM\n            row=dict(action_cycle=event['step']+1,**evaluator.latest,**event,\n                     actual_motion_delta_deg_deg_mm=actual_motion,\n                     actual_encoder_delta_ABCE=np.asarray(motor['actual_after_robot_ABCE'])-motor['actual_before_robot_ABCE'],")
replace("                acknowledge(receipt)\n                record('actual first action',receipt)","                if not prefix: acknowledge(receipt)\n                record('prefix first action' if prefix else 'actual first action',receipt)")
replace("sensor=sensor,seed=42+index,max_steps=1600,", "sensor=sensor,seed=42+index,max_steps=steps,")
replace("error_threshold_mm=.00035,control_period_s=.05,action_limit_deg_deg_mm=(.2,.1,.03),", "error_threshold_mm=.00035,control_period_s=.05,action_limit_deg_deg_mm=V5_CAPS,")
replace("        assert len(rows)==len(ep['trajectory']['actions'])==len(ep['motor_records'])==1600", "        assert len(rows)==len(ep['trajectory']['actions'])==len(ep['motor_records'])==steps")
replace("np.arange(1,1601)*.05", "np.arange(1,steps+1)*.05")
replace("summary=dict(status='completed',target=f'E{index:02d}',seed=42+index,actions=1600,", "summary=dict(status='prefix_completed' if prefix else 'trajectory_completed_pending_protocol_gate',target=f'E{index:02d}',seed=42+index,actions=steps,")
replace("        record('target completed',summary)","        record('short prefix completed pending exact gate' if prefix else 'trajectory completed pending exact protocol gate',summary)")
replace("        acknowledge(failure)\n        record('target failed or interrupted',failure)","        if not prefix: acknowledge(failure)\n        record('prefix failed' if prefix else 'target failed or interrupted',failure)")
start=s.index('    row=read(ARCHIVE/',s.index('def finish_comparison('))
end=s.index('    summary[\'baseline\']',start)
s=s[:start]+'''    comparison,old=compare(folder,index)
    post=np.r_[old['trajectory']['errors'][1:],old['final_error_mm']]*1000
    summary['pre_switch_baseline_comparison']=comparison
    protocol_pass=comparison['status']=='passed'
    summary['status']='completed' if protocol_pass else 'protocol_failed'
    summary['formal_primary_200um_pass']=bool(protocol_pass and summary['primary_200um']['passed'])
    summary['formal_secondary_20um_pass']=bool(protocol_pass and summary['long_term_20um']['passed'])
    summary['numeric_radius_metrics_alone_are_not_formal_acceptance']=True
'''+s[end:]
replace("source_contract=archive.get('evaluation_contract'),source=row['result_path']", "source_contract=comparison['original_contract'],source=comparison['source']")
replace("        assert not (OUT/'supervisor_start_receipt.json').exists(), 'Refuse duplicate execution'", "        assert not (OUT/'supervisor_start_receipt.json').exists(), 'Refuse duplicate execution'\n        gate=read(OUT/'prefix_gate/validation.json')\n        assert gate['status']=='passed'\n        for path,value in gate['source_hashes'].items(): assert sha(path)==value, f'Changed after prefix gate: {path}'\n        assert len(active_workers())+2<=12")
replace("primary_200um_passes=sum(r['primary_200um']['passed'] for r in results),", "primary_200um_passes=sum(r['formal_primary_200um_pass'] for r in results),")
replace("long_term_20um_passes=sum(r['long_term_20um']['passed'] for r in results),", "long_term_20um_passes=sum(r['formal_secondary_20um_pass'] for r in results),")
replace("        preflight()\n        os.mkfifo", "        gate=validate_prefix()\n        assert gate['status']=='passed', 'Exact prefix gate failed; no long rollout launched'\n        prior_gate=read(OUT/'prefix_gate/validation.json')\n        assert gate==prior_gate, 'Prefix gate evidence changed'\n        preflight()\n        record('24-step E08 bit-exact prefix gate passed',dict(checks=len(gate['checks']),seed=50,cycles=24,comparison=gate['comparison']))\n        os.mkfifo")
replace("--title=v5 learned local standard2 E06-48 E08-50", "--title=v5 standard2 protocol v2 EXACT GATE PASSED E06-48 E08-50")
replace("choices=['launch','supervise','worker']", "choices=['launch','supervise','worker','prefix','gate']")
replace("    if args.mode=='worker': sys.exit(worker(args.index))", "    if args.mode=='prefix':\n        check_sources()\n        sys.exit(worker(8,steps=24,prefix=True))\n    if args.mode=='gate':\n        gate=validate_prefix()\n        save(OUT/'prefix_gate/validation.json',gate)\n        record('short prefix exact gate',gate)\n        sys.exit(0 if gate['status']=='passed' else 1)\n    if args.mode=='worker': sys.exit(worker(args.index))")
ast.parse(s)
(OUT/'standard2.py').write_text(s)
shell=(V1/'launch.sh').read_text().replace('v5_learned_local_standard2_v1','v5_learned_local_standard2_protocol_v2')
(OUT/'launch.sh').write_text(shell)
print('Created independent protocol-v2 runner; v1 is untouched')
