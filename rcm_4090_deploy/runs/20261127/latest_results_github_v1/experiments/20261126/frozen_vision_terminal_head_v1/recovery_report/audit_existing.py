import ast
import math
import shutil
from pathlib import Path
from recovery_utils import ROOT, TASK, OUT, sha, read, save, record

snapshot = OUT / 'original_snapshot'
snapshot.mkdir()
original = {}
for p in sorted(TASK.iterdir()):
    if p.is_file():
        original[p.name] = dict(bytes=p.stat().st_size, sha256=sha(p))
        if p.suffix in ('.py', '.json', '.jsonl', '.md', '.sh', '.log'):
            shutil.copy2(p, snapshot / p.name)
save('original_task_manifest.json', original)
record('original_snapshot', dict(files=len(original), source_sha256=original['pilot.py']['sha256'],
       originals_preserved=True, snapshot=str(snapshot)))

hashes = read(TASK / 'input_hashes.json')
identities = []
for path, expected in hashes.items():
    p = Path(path)
    actual = sha(p) if p.is_file() else None
    identities.append(dict(path=path, expected_sha256=expected, actual_sha256=actual,
                           matches=actual == expected))
save('input_identity_before_fix.json', dict(files=identities, mismatches=[x for x in identities if not x['matches']]))
record('input_identity_before_fix', dict(checked=len(identities), mismatches=[x['path'] for x in identities if not x['matches']],
       checkpoint_sha256=read(TASK/'preflight.json')['checkpoint_sha256']))

card = read(TASK / 'experiment_card.json')
manifest = read(ROOT / card['manifest'])
selection = read(TASK / 'tinyfit_indices.json')
src = (TASK / 'pilot.py').read_text()
tree = ast.parse(src)
select = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'select_tiny')
scope = {}
exec(compile(ast.Module(body=[select], type_ignores=[]), '<isolated selection; no model>', 'exec'), scope)
chosen = scope['select_tiny'](manifest['train'])
assert chosen == selection['indices']
updates = [__import__('json').loads(x) for x in (TASK/'tinyfit_updates.jsonl').read_text().splitlines()]
assert [x['update'] for x in updates] == list(range(1, 513))
assert all(x['batch_indices'] == chosen for x in updates)
assert all(math.isfinite(x['loss']) and math.isfinite(x['gradient_norm_before_clip']) for x in updates)
windows = []
for i, saved in zip(chosen, selection['windows']):
    m = manifest['train'][i]
    assert all(m[k] == v for k, v in saved.items())
    windows.append(dict(index=i, **m))
metrics = [__import__('json').loads(x) for x in (TASK/'tinyfit_train_metrics.jsonl').read_text().splitlines()]
assert [x['update'] for x in metrics] == [128, 256, 384, 512]
scales = read(TASK/'scales.json')
assert scales['action_std'] == card['scales']['frozen_action_std_E40_deg_deg_mm']
assert scales['action_mean'] == card['scales']['frozen_action_mean_E40_deg_deg_mm']
curve = []
for m in metrics:
    mse = [x*x for x in m['normalized_RMSE']]
    recomputed = sum(mse)/3
    assert math.isclose(recomputed, m['loss'], rel_tol=2e-6)
    c_um = m['physical_MAE_deg_deg_mm'][2]*1000
    assert math.isclose(c_um, m['C_MAE_um'], rel_tol=2e-6)
    curve.append(dict(**m, normalized_MSE_per_axis=mse,
                      physical_RMSE_deg_deg_mm=[a*b for a,b in zip(m['normalized_RMSE'], scales['action_std'])],
                      recomputed_loss=recomputed, recomputed_C_MAE_um_from_mm=c_um,
                      axis_gate=[x<=0.1 for x in m['normalized_RMSE']],
                      c_mae_gate=c_um<=0.2,
                      gate_pass=all(x<=0.1 for x in m['normalized_RMSE']) and c_um<=0.2))
assert not curve[-1]['gate_pass']
save('tinyfit_evidence.json', dict(update_count=len(updates), sequence_contiguous=True, batch_all_match_selection=True,
     selection=chosen, selected_windows=windows, selected_window_count=len(chosen),
     first_update=updates[0], final_update=updates[-1], curve=curve,
     scope='10 E40 fixed full-batch windows only; not 657-window train metrics or E41 offline',
     loss_timing='updates.jsonl loss is pre-optimizer; metrics is post-optimizer full selected batch',
     recomputation_limit='Saved aggregate metrics only; no predictions/residuals or final head available for independent per-example recomputation'))
record('tinyfit_gate_recomputed', dict(updates=len(updates), windows=len(chosen), last=curve[-1]))

parity = read(TASK/'parity.json')
failure = read(TASK/'failure.json')
preflight = read(TASK/'preflight.json')
parameters = read(TASK/'parameters.json')
parameter_count = sum(x['count'] for x in parameters['parameters'] if x['trainable'])
assert parameter_count == preflight['head_parameters'] == card['architecture']['new_head_parameter_count'] == 112323
assert parity['status']=='passed' and len(parity['checks'])==12
expected_missing = ['tinyfit_result.json', 'tinyfit_checkpoint.pth', 'tinyfit_failure_gradients_weights.pt',
                    'tinyfit_interrupted_gradients_weights.pt', 'formal_checkpoint.pth', 'candidate.pth',
                    'formal_updates.jsonl', 'offline_gate.json']
evidence = dict(preflight=preflight, shapes=read(TASK/'shapes.json'), scales=scales,
    parameter_count_from_saved_inventory=parameter_count,
    parity_status=parity['status'], parity_checks=len(parity['checks']),
    max_feature_abs=max(x['feature_max_abs'] for x in parity['checks']),
    max_normalized_action_abs=max(x['normalized_action_max_abs'] for x in parity['checks']),
    max_production_action_abs=max(x['production_get_action_max_abs'] for x in parity['checks']),
    artifacts_exist={n:(TASK/n).exists() for n in ['initial_head.pth','spatial_softmax_cache.pt']+expected_missing},
    saved_tensor_files=[str(p.relative_to(TASK)) for p in TASK.rglob('*') if p.suffix in ('.pt','.pth')],
    traceback=failure['traceback'], traceback_observed_lines=[300,313], same_latent_reporting_bug_line=304,
    initial_source='deepcopy(net.head.state_dict()) on CUDA, saved before training',
    tensor_recovery='Final trained head, final gradients, optimizer state and weight deltas were not saved; cannot recover from initial weights, scalar metrics or cached features without forbidden replay.',
    other_training_issues='No additional failure is evidenced by the existing traceback. Reporting errors prevented later save paths and formal/offline paths from executing; this audit does not certify unexecuted training code.')
save('artifact_and_traceback_evidence.json', evidence)
record('artifacts_parity_and_traceback', dict(parameter_count=parameter_count, state_dim=20, visual_dim=1536,
       initial_exists=True, final_checkpoint_exists=(TASK/'tinyfit_checkpoint.pth').exists(),
       gradients_recoverable=False, parity=parity['status'], observed_error_lines=[300,313],
       latent_same_reporting_error_line=304))

terminal = (TASK/'terminal.log').read_text()
lines = terminal.splitlines()
save('terminal_log_evidence.json', dict(train_metrics_lines=[x for x in lines if x.startswith('TRAIN_METRICS')],
     tail=lines[-14:], end_marker_present='Run exit=1' in terminal))
receipt=read(TASK/'actual_start_receipt.json')
proc=Path('/proc')/str(receipt['pid'])/'cmdline'
cmd=proc.read_bytes().replace(b'\0',b' ').decode(errors='replace') if proc.exists() else None
save('process_evidence.json', dict(original_pid=receipt['pid'], current_cmdline=cmd, original_run_ended=cmd is None or str(TASK/'pilot.py') not in cmd))
record('terminal_and_process', dict(original_pid=receipt['pid'], current_cmdline=cmd, terminal_exit1='Run exit=1' in terminal))
