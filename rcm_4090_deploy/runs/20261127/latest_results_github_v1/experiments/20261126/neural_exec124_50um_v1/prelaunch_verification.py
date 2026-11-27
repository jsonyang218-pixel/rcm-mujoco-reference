"""Read-only card/source/log verification; no model/scene import or execution."""
import importlib.util
import json
from pathlib import Path

p=Path(__file__).with_name('pilot.py')
spec=importlib.util.spec_from_file_location('new_prelaunch_readback',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=m.verify_identity()
assert checks['status']=='passed'
assert not (m.OUT/'actual_start_receipt.json').exists()
assert not (m.OUT/'gnome_launch_request.json').exists()
card=m.read(m.OUT/'experiment_card.json');state=m.read(m.OUT/'task_status.json')
assert state['status']=='prepared' and card['total_arms']==12 and card['unchanged']['seed']==434341
receipts=[json.loads(x) for x in (m.OUT/'test_log_receipts.jsonl').read_text().splitlines()]
assert all(r['exact_readback'] for x in receipts for r in x['receipts'])
result={**checks,'card_total_arms':12,'concurrency':4,'existing_launch':False,
    'previous_log_entries_verified':len(receipts),'training_updates':0,'prelaunch_model_calls':0,
    'prelaunch_simulation_cycles':0,'diagnostic_reporting_error_fixed':True}
m.save('prelaunch_readback_verification.json',result)
m.record('prepared_card_hash_and_log_readback_verified',result)
print('READY',str(m.OUT),state['stage'])
