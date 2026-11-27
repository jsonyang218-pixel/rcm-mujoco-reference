"""Only synthetic tensors and extracted reporting AST; never import pilot/head."""
import ast
import difflib
import py_compile
from types import SimpleNamespace
import torch
from recovery_utils import TASK, OUT, sha, read, save, record

before = (OUT/'original_snapshot/pilot.py').read_text()
after = (TASK/'pilot.py').read_text()
assert before.count('p.detach().cpu()-initial[n]') == 3
assert after == before.replace('p.detach().cpu()-initial[n]', 'p.detach().cpu()-initial[n].detach().cpu()')
diff = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                 fromfile='original/pilot.py', tofile='reporting_fixed/pilot.py'))
(OUT/'pilot_reporting_only.diff').write_text(diff)
save('source_change.json', dict(path=str(TASK/'pilot.py'), before_sha256=sha(OUT/'original_snapshot/pilot.py'),
    after_sha256=sha(TASK/'pilot.py'), replaced_expression_count=3, changed_original_lines=[300,304,313],
    scope='Only CPU device alignment in weight-difference reporting and two diagnostic save paths',
    learning_data_gates_inference_changed=False))
record('minimal_reporting_fix', dict(before_sha256=sha(OUT/'original_snapshot/pilot.py'), after_sha256=sha(TASK/'pilot.py'),
    changed_lines=[300,304,313], exact_three_expression_only=True))

for p in [TASK/'pilot.py', OUT/'audit_existing.py', OUT/'recovery_utils.py', OUT/'test_reporting_tensors.py']:
    py_compile.compile(str(p), cfile=str(OUT/(p.name+'.syntax.pyc')), doraise=True)
record('syntax_check', dict(status='passed', pilot_executed=False, pyc_destination=str(OUT)))

assert torch.cuda.is_available(), 'CUDA needed to reproduce the actual mismatch; do not claim GPU pass on CPU fallback'
torch.set_num_threads(1)
def reporting_dicts(source):
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_stage')
    return [n for n in ast.walk(fn) if isinstance(n, ast.DictComp) and 'initial[n]' in ast.unparse(n)]

old_dicts, new_dicts = reporting_dicts(before), reporting_dicts(after)
assert len(old_dicts)==len(new_dicts)==3
rows=[]
test_dir = OUT/'synthetic_tensor_test'
test_dir.mkdir()
for device in ['cpu', 'cuda:0']:
    p = torch.tensor([1.0, -2.0, 3.0], device=device)
    initial = {'synthetic_weight':torch.tensor([0.5,-1.0,4.0], device=device)}
    net = SimpleNamespace(head=SimpleNamespace(named_parameters=lambda:[('synthetic_weight',p)]))
    scope=dict(net=net,initial=initial,float=float,torch=torch)
    old_errors=[]
    for node in old_dicts:
        try:
            eval(compile(ast.Expression(node),'<original reporting expression>','eval'),scope)
            old_errors.append(None)
        except RuntimeError as e:
            old_errors.append(str(e))
    if device.startswith('cuda'):
        assert all(e is not None and 'device' in e for e in old_errors)
    else:
        assert old_errors==[None]*3
    results=[eval(compile(ast.Expression(node),'<fixed reporting expression>','eval'),scope) for node in new_dicts]
    assert results[0]['synthetic_weight']==1.0
    for result in results[1:]:
        torch.testing.assert_close(result['synthetic_weight'],torch.tensor([0.5,-1.0,-1.0]))
    rows.append(dict(device=device, original_error=old_errors, fixed_max_abs=1.0, fixed_deltas=[0.5,-1.0,-1.0]))
record('synthetic_cpu_cuda_differences', dict(status='passed', gpu=torch.cuda.get_device_name(0), cases=rows,
       network_instances=0, optimizer_updates=0, real_data_forwards=0))

# Execute exact fixed diagnostic-save statements in isolation, not the training function.
fn=next(n for n in ast.parse(after).body if isinstance(n,ast.FunctionDef) and n.name=='train_stage')
training_try=next(n for n in fn.body if isinstance(n,ast.Try))
handler=training_try.handlers[0]
normal_failure=next(n for n in training_try.body if isinstance(n,ast.If) and ast.unparse(n.test)=='tiny and (not passed)')
scope.update(dict(OUT=test_dir,last_grads={'synthetic_weight':torch.tensor([0.25,0.5,0.75])},
                  PROGRESS={'tinyfit_updates':512,'formal_updates':0}, name='tinyfit', tiny=True, passed=False))
exec(compile(ast.Module(body=[normal_failure],type_ignores=[]),'<isolated failure save>','exec'),scope)
sentinel=RuntimeError('synthetic original error must remain visible')
wrapped=ast.Try(body=[ast.Raise(exc=ast.Name(id='sentinel',ctx=ast.Load()),cause=None)], handlers=[handler], orelse=[], finalbody=[])
scope['sentinel']=sentinel
try:
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapped],type_ignores=[])), '<isolated except handler>','exec'),scope)
except RuntimeError as e:
    assert e is sentinel
else:
    raise AssertionError('original exception not re-raised')
for name in ['tinyfit_failure_gradients_weights.pt','tinyfit_interrupted_gradients_weights.pt']:
    payload=torch.load(test_dir/name,map_location='cpu',weights_only=True)
    torch.testing.assert_close(payload['weight_deltas']['synthetic_weight'],torch.tensor([0.5,-1.0,-1.0]))
    torch.testing.assert_close(payload['gradients']['synthetic_weight'],scope['last_grads']['synthetic_weight'])
    if 'last_update' in payload: assert payload['last_update']==512
save('synthetic_tensor_test_results.json', dict(status='passed', cases=rows, exact_except_save_writable=True,
    exact_failure_save_writable=True, original_exception_preserved=True,
    synthetic_artifacts_only=str(test_dir), network_instances=0, optimizer_instances=0,
    optimizer_updates=0, autograd_backward_calls=0, model_forwards=0, actual_pilot_runs=0))
record('synthetic_failure_and_except_save_readback', dict(status='passed', original_exception_preserved=True,
       saved_synthetic_files=2, real_trained_tensors_recovered=False))

# Saved initial/cache tensors are inspected on CPU only, with no model construction.
initial_saved=torch.load(TASK/'initial_head.pth',map_location='cpu',weights_only=True)
assert sum(t.numel() for t in initial_saved.values())==112323
cache=torch.load(TASK/'spatial_softmax_cache.pt',map_location='cpu',weights_only=True)
cache_shapes={k:list(v.shape) for k,v in cache.items()}
assert cache_shapes=={'train':[657,1536],'development_E41':[657,1536]}
save('saved_tensor_inventory.json',dict(initial_head_tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in initial_saved.items()},
      initial_head_numel=112323, cache_shapes=cache_shapes,
      note='CPU deserialization only; initial != final; no trained head or gradient artifact exists'))
record('saved_initial_and_cache_readonly',dict(initial_head_numel=112323,cache_shapes=cache_shapes,model_forwards=0))
