"""One authorized frozen-vision deterministic terminal-head trial. No production edits."""
from __future__ import annotations
import argparse, ast, copy, csv, fcntl, hashlib, importlib.util, json, os
from pathlib import Path
import selectors, shutil, signal, subprocess, sys, time, traceback
from contextlib import contextmanager
from datetime import datetime

ROOT=Path('/home/him1/mujoco_bc/rcm_4090_deploy')
OUT=Path(__file__).resolve().parent
PRIOR=ROOT/'runs/20261126/neural_paired_terminal_minifit_v1'
BASE=ROOT/'runs/20261126/neural_exec124_50um_v1'
CARD=OUT/'experiment_card.json'
REF=Path(json.loads(CARD.read_text())['init_checkpoint'])
REF_SHA='6cc389df3786f1f34c43cf65d6dc17fe910ab342fae8d84c8e37cea6a7db6c11'
SEED=20261126
KEYS=('observation.image','observation.state','observation.oct','observation.velocity','observation.previous_action')
SPLITS=('train','development_E41')
KINDS=('below_approaching','below_receding','above_approaching','above_receding')
DESKTOP=dict(DISPLAY=':1',DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus',XDG_RUNTIME_DIR='/run/user/1000')
RUNTIME=dict(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',RCM_DRIVER_VARIANT='g10',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',TORCH_DISABLE_NATIVE_JIT='1',MPLBACKEND='Agg')
sys.path.insert(0,str(ROOT))
DEADLINE=None
PROGRESS=dict(implementation='complete',launch='not_started',parity='not_started',tinyfit='not_started',formal_training='not_started',offline='not_started',closed_loop='not_started',tinyfit_updates=0,formal_updates=0)


def now(): return datetime.now().astimezone().isoformat()
def default(x):
    if hasattr(x,'tolist'):return x.tolist()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(name,obj,update=False):
    p=Path(name);p=p if p.is_absolute() else OUT/p
    assert p.is_relative_to(OUT),p
    p.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=default)+'\n'
    if update:
        temp=p.with_suffix(p.suffix+'.tmp')
        with temp.open('w') as f:f.write(data);f.flush();os.fsync(f.fileno())
        temp.replace(p)
    else:
        with p.open('x') as f:f.write(data);f.flush();os.fsync(f.fileno())
    assert p.read_text()==data

def append(p,row):
    with Path(p).open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,default=default)+'\n');f.flush();os.fsync(f.fileno())
def record(stage,result):
    row=dict(time=now(),experiment=OUT.name,stage=stage,result=result,command=sys.argv,card=str(CARD),checkpoint_sha256=REF_SHA,output=str(OUT))
    line=('\n- '+json.dumps(row,ensure_ascii=False,allow_nan=False,default=default)+'\n').encode();receipts=[]
    for p in (OUT/'TEST_LOG.md',ROOT/'TEST_LOG.md'):
        with p.open('a+b') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0,2);offset=f.tell();f.write(line);f.flush();os.fsync(f.fileno());f.seek(offset);assert f.read(len(line))==line
        receipts.append(dict(path=str(p),offset=offset,bytes=len(line),exact_readback=True))
    append(OUT/'test_log_receipts.jsonl',dict(time=row['time'],stage=stage,receipts=receipts))
    print(json.dumps(row,ensure_ascii=False,default=default),flush=True)
def status(stage,state='running',**extra):
    row=dict(time=now(),status=state,stage=stage,pid=os.getpid(),**PROGRESS);row.update(extra)
    save('task_status.json',row,True);append(OUT/'status_history.jsonl',row)
def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def alarm(signum,frame):raise TimeoutError('Registered stage or total900s budget exceeded; no retry')
@contextmanager
def budget(stage,seconds):
    start=time.monotonic();allowed=min(seconds,DEADLINE-start)
    if allowed<=0:raise TimeoutError('Total budget exhausted')
    signal.setitimer(signal.ITIMER_REAL,allowed)
    status(stage,allowed_s=allowed)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        append(OUT/'budget_usage.jsonl',dict(time=now(),stage=stage,allowed_s=allowed,used_s=time.monotonic()-start))
def write_csv(name,rows):
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())


def prepare():
    started=time.monotonic()
    assert not (OUT/'actual_start_receipt.json').exists() and not (OUT/'gnome_launch_request.json').exists()
    assert read(OUT/'preparation_service_failure.json')['actual_experiment_started'] is False
    status('necessary_static_and_asset_preflight','preparing')
    for name in ('pilot.py','head.py','reuse_arm.py'):ast.parse((OUT/name).read_text())
    shell=subprocess.run(['bash','-n',str(OUT/'launch.sh')],capture_output=True,text=True,timeout=5);assert shell.returncode==0,shell.stderr
    record('syntax_checks_passed',dict(files=['pilot.py','head.py','reuse_arm.py','launch.sh'],model_calls=0,simulation_cycles=0))
    import torch
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
    assert sys.executable==str(ROOT/'.venv/bin/python')
    assert sha(REF)==REF_SHA
    manifest=read(PRIOR/'window_manifest.json')
    assert all(len(manifest[s])==657 for s in SPLITS)
    assert {m['target'] for m in manifest['train']}=={'E40'} and {m['target'] for m in manifest['development_E41']}=={'E41'}
    labels=torch.load(PRIOR/'real_input_windows.pt',map_location='cpu',weights_only=False)['train']['action'][:,0].float()
    scale_floor=torch.tensor([1e-4,1e-4,1e-5]);scale_mean=labels.mean(0);scale_raw_std=labels.std(0,unbiased=False)
    card=read(CARD);card['scales']['frozen_action_mean_E40_deg_deg_mm']=scale_mean.tolist();card['scales']['frozen_action_raw_std_E40_deg_deg_mm']=scale_raw_std.tolist();card['scales']['frozen_action_std_E40_deg_deg_mm']=torch.maximum(scale_raw_std,scale_floor).tolist();card['scales']['frozen_at']=now();card['scales']['E41_metrics_seen_when_frozen']=False
    save(CARD,card,True)
    record('E40_action_scaling_frozen_in_card_before_E41_evaluation',card['scales'])
    del labels
    tiny=select_tiny(manifest['train']);assert len(tiny)==10 and len({manifest['train'][i]['episode'] for i in tiny[:6]})==6
    save('tinyfit_indices.json',dict(indices=tiny,windows=[{k:manifest['train'][i][k] for k in ('window_index','episode','teacher_step','input_class','side')} for i in tiny],selection=read(CARD)['tinyfit']['selection']))
    # This check exercises only the new head and index utilities on CPU; no visual inference or scientific update.
    import head
    module=head.HeadOnly();assert sum(p.numel() for p in module.parameters())==112323
    sampler=import_file('prior_sampler_preflight',PRIOR/'pilot.py');pools=sampler.pools_for(manifest['train'])
    g1=torch.Generator().manual_seed(SEED);g2=torch.Generator().manual_seed(SEED)
    assert [sampler.draw_batch(pools,g1,64) for _ in range(3)]==[sampler.draw_batch(pools,g2,64) for _ in range(3)]
    for kind in KINDS:
        for name in ('initial_state.json','first_real_production_input.pt','summary.json','cycles.jsonl'):
            assert (BASE/'arms'/kind/'exec1'/name).is_file(),(kind,name)
    display=subprocess.run(['xdpyinfo','-display',':1'],env={**os.environ,**DESKTOP},capture_output=True,text=True,timeout=8);assert display.returncode==0,display.stderr
    assert shutil.which('gnome-terminal') and Path('/run/user/1000/bus').exists()
    assert shutil.disk_usage(OUT).free>8*1024**3
    files={REF,PRIOR/'pilot.py',PRIOR/'real_input_windows.pt',PRIOR/'window_manifest.json',PRIOR/'experiment_card.json',BASE/'pilot.py',BASE/'experiment_card.json',ROOT.parent/'AGENTS.md',ROOT/'docs/EXPERIMENT_SOP.md',CARD}
    files|={OUT/n for n in ('pilot.py','head.py','reuse_arm.py','launch.sh')}
    files|={PRIOR/f'offline_before_{s}_per_window_seed.csv' for s in SPLITS}
    for kind in KINDS:files|={p for p in (BASE/'arms'/kind/'exec1').iterdir() if p.is_file()}
    for folder in ('rcm','lerobot_training','mujoco_model','restricted_perception'):
        files|={p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.xml','.json')}
    hashes={str(p):sha(p) for p in sorted(files)};save('input_hashes.json',hashes)
    result=dict(status='passed',time=now(),executable=sys.executable,gpu=torch.cuda.get_device_name(0),checkpoint_sha256=REF_SHA,hashed_files=len(hashes),windows={s:len(manifest[s]) for s in SPLITS},head_parameters=112323,state_dim=20,visual_dim=1536,duplicate_execution=False,model_forward_calls=0,simulation_cycles=0,old_failure_preserved=True)
    save('preflight.json',result);save('preparation_budget.json',dict(elapsed_compute_s=time.monotonic()-started,reading_and_writing_excluded=True))
    status('prepared','prepared');record('necessary_preflight_passed',result)


def select_tiny(meta):
    picks=[i for i,m in enumerate(meta) if m['designated_anchor'] and m['input_class']=='receding']
    assert len(picks)==6
    for side in ('below','above'):
        for cls in ('approaching','settled'):
            candidates=[i for i,m in enumerate(meta) if m['side']==side and m['input_class']==cls]
            picks.append(min(candidates,key=lambda i:(meta[i]['episode'],meta[i]['teacher_step'],i)))
    assert len(set(picks))==len(picks)<=12
    return picks


def setup_torch():
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
    return torch


def normalize_scales(model,train):
    import torch
    names=('state_mean','state_std','velocity_mean','velocity_std','previous_action_mean','previous_action_std')
    sc={n:getattr(model,n).detach().cpu().clone().reshape(3) for n in names}
    y=train['action'][:,0].float();floor=torch.tensor([1e-4,1e-4,1e-5])
    sc['action_mean']=y.mean(0);sc['action_std_unfloored']=y.std(0,unbiased=False);sc['action_std']=torch.maximum(sc['action_std_unfloored'],floor)
    frozen=read(CARD)['scales'];assert sc['action_mean'].tolist()==frozen['frozen_action_mean_E40_deg_deg_mm'] and sc['action_std'].tolist()==frozen['frozen_action_std_E40_deg_deg_mm']
    for n in names:
        assert torch.isfinite(sc[n]).all()
        if n.endswith('std'):assert (sc[n]>0).all()
    save('scales.json',dict(**sc,action_std_floor=floor,action_units=['deg','deg','mm'],action_statistics_source='E40 first accepted action only; no E41 statistics',action_std_definition='population std correction0',floor_reason=read(CARD)['scales']['floor_reason'],OCT_xyz_scale_mm=.1,valid_preserved=True,state_dimension=20,state_layout=['ABC_tminus1(3)','ABC_t(3)','OCT_tminus1_xyz+valid(4)','OCT_t_xyz+valid(4)','finite_difference_velocity_current(3)','previous_acceptedABC(3)']))
    record('E40_action_and_original_state_scales_frozen',dict(scales=str(OUT/'scales.json'),action_mean=sc['action_mean'],action_std=sc['action_std'],E41_statistics_used=False))
    return sc


def prepare_features():
    torch=setup_torch();import numpy as np
    from PIL import Image
    from lerobot_training.eval_policy import PolicyEvaluator
    from head import TerminalHead
    sampler=import_file('unchanged_pair_utilities',PRIOR/'pilot.py')
    e=PolicyEvaluator(str(REF),device='cuda',num_action_samples=1,n_action_execute_steps=1,flow_integration_steps=4,measure_inference=True)
    assert e.fine_evaluator is None and e.n_obs_steps==2 and e.model.dit_dim==384
    data=torch.load(PRIOR/'real_input_windows.pt',map_location='cpu',weights_only=False);manifest=read(PRIOR/'window_manifest.json')
    assert all(len(data[s]['action'])==657 for s in SPLITS)
    scales=normalize_scales(e.model,data['train']) # freeze before any E41 predictions/features
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    net=TerminalHead(e.model.vision,scales).cuda().eval()
    initial=copy.deepcopy(net.head.state_dict());torch.save(initial,OUT/'initial_head.pth')
    vision_before={k:v.detach().cpu().clone() for k,v in net.vision.state_dict().items()}
    params=[dict(name=n,shape=list(p.shape),count=p.numel(),trainable=p.requires_grad) for n,p in net.named_parameters()]
    save('parameters.json',dict(parameters=params,trainable=sum(r['count'] for r in params if r['trainable']),frozen=sum(r['count'] for r in params if not r['trainable']),vision_eval=not net.vision.training,GroupNorm=sum(isinstance(m,torch.nn.GroupNorm) for m in net.vision.modules()),BatchNorm=sum(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in net.vision.modules())))
    assert sum(p.numel() for p in net.head.parameters())==112323
    assert sum(p.numel() for p in net.vision.parameters())==11373504
    # Six real strata per split, same preprocessing from saved RGB through original production entry.
    selected=[]
    for split in SPLITS:
        for side in ('below','above'):
            for cls in ('approaching','receding','settled'):
                i=next(i for i,m in enumerate(manifest[split]) if m['side']==side and m['input_class']==cls)
                selected.append((split,i))
    shape_capture=[]
    hook=net.vision.project.register_forward_hook(lambda mod,inp,out:shape_capture.append(list(out.shape)))
    baseline_audit={}
    for split in SPLITS:baseline_audit[split]=audit_baseline(data[split],manifest[split],split)
    save('baseline_alignment.json',dict(status='passed',splits=baseline_audit,seeds=[92001,92002,92003,92004],caps=read(CARD)['offline']['caps']))
    record('saved_baseline_window_label_caps_verified',read(OUT/'baseline_alignment.json'))
    # Full cache in FP32/GPU, batch8, bounded CPU image transfer; duplicate RGB frames are not new data.
    cache={}
    with torch.no_grad():
        for split in SPLITS:
            blocks=[]
            for start in range(0,657,8):
                images=data[split][KEYS[0]][start:start+8].cuda();blocks.append(net.visual_features(images).cpu())
            cache[split]=torch.cat(blocks).cuda()
            status('feature_cache',split=split,completed_windows=657)
    parity=[]
    with torch.no_grad():
        for split,i in selected:
            m=manifest[split][i];rgb=[];observations=[]
            for path in m['image_paths']:
                with Image.open(PRIOR/path) as im:rgb.append(np.asarray(im.convert('RGB')).copy())
            ep=PRIOR/'data'/m['episode']/'observations.jsonl'
            wanted=set(m['history_observation_indices'])
            with ep.open() as f:
                for j,line in enumerate(f):
                    if j in wanted:observations.append(json.loads(line))
            assert len(observations)==2
            past=dict(rgb=rgb[0],state=observations[0]['state_deg_deg_mm'],oct=observations[0]['signed_OCT_mm'])
            current=dict(rgb=rgb[1],state=observations[1]['state_deg_deg_mm'],oct=observations[1]['signed_OCT_mm'])
            args=sampler.original().arguments(e,past,current,observations[1]['previous_accepted_deg_deg_mm'])
            diffs=[]
            for key,arg in zip(KEYS,args):
                torch.testing.assert_close(arg.cpu(),data[split][key][i:i+1],atol=0,rtol=0);diffs.append(0.)
            v=net.visual_features(args[0]);cv=cache[split][i:i+1]
            torch.testing.assert_close(v,cv,atol=1e-6,rtol=1e-5)
            online=net(*args);cached=net.from_features(cv,*args[1:])
            torch.testing.assert_close(online,cached,atol=1e-6,rtol=1e-5)
            # Production get_action constructs actual history/velocity/accepted feedback, but executes no motor action.
            oldmodel=e.model;e.model=net;e.reset_buffer(seed=SEED)
            sampler.original().seed_history(e,[past],observations[1]['previous_accepted_deg_deg_mm'])
            actual=e.get_action(current['rgb'],np.asarray(current['state']),np.asarray(current['oct'],np.float32))
            e.model=oldmodel
            torch.testing.assert_close(torch.from_numpy(actual).cuda(),online[0],atol=1e-6,rtol=1e-5)
            parity.append(dict(split=split,window=i,side=m['side'],input_class=m['input_class'],preprocess_max_abs=diffs,feature_max_abs=float((v-cv).abs().max()),normalized_action_max_abs=float(((online-cached)/net.action_std).abs().max()),physical_action_max_abs_deg_deg_mm=(online-cached).abs().max(0).values.cpu(),production_get_action_max_abs=float(np.max(np.abs(actual-online[0].cpu().numpy())))))
    hook.remove();assert shape_capture and all(x[1:]==[384,7,7] for x in shape_capture)
    save('shapes.json',dict(original_vision_forward=list(net.last_token_shape),original_project_feature_maps=sorted({tuple(s) for s in shape_capture}),head_visual_features=[None,1536],state=[None,20],action=[None,3],FP32=True))
    result=dict(status='passed',atol=1e-6,rtol=1e-5,checks=parity,spatial_softmax_temperature=1.,coordinates='[-1,1]xy',cache_dtype='float32',cached_windows={s:657 for s in SPLITS},original_vision_tokens_layout='B,2,49,384; transpose/reshape exactly inverse original flatten',groupnorm_and_preprocessing_unchanged=True)
    save('parity.json',result);record('new_path_FP32_cache_online_parity_passed',result)
    torch.save({s:cache[s].cpu() for s in SPLITS},OUT/'spatial_softmax_cache.pt')
    return net,initial,vision_before,data,cache,manifest,sampler,baseline_audit,scales


def audit_baseline(batch,meta,split):
    import numpy as np
    path=PRIOR/f'offline_before_{split}_per_window_seed.csv';rows=list(csv.DictReader(path.open()))
    assert len(rows)==657*4
    seen=set();caps=np.array(read(CARD)['offline']['caps']);max_label=0.
    for row in rows:
        i=int(row['window']);seed=int(row['seed']);assert seed in (92001,92002,92003,92004) and (i,seed) not in seen;seen.add((i,seed))
        assert all(str(meta[i][k])==row[k] for k in ('episode','teacher_step','input_class','side'))
        label=np.array([float(row[f'teacher_{a}_{u}']) for a,u in zip('ABC',('deg','deg','mm'))]);actual=batch['action'][i,0].numpy()
        np.testing.assert_array_equal(label,actual.astype(float));np.testing.assert_array_equal(label,np.asarray(meta[i]['accepted_label'],np.float32).astype(float));max_label=max(max_label,float(np.max(np.abs(label-actual))))
        raw=np.array([float(row[f'raw_{a}_{u}']) for a,u in zip('ABC',('deg','deg','mm'))]);prod=np.array([float(row[f'production_{a}_{u}']) for a,u in zip('ABC',('deg','deg','mm'))])
        np.testing.assert_array_equal(prod,np.clip(raw,-caps,caps))
    return dict(path=str(path),sha256=sha(path),rows=len(rows),windows=len(meta),seeds=[92001,92002,92003,92004],window_label_exact=True,max_label_difference=max_label,caps_exact=True)


def training_metrics(net,features,batch,indices=None):
    import torch
    args=[batch[k].cuda() for k in KEYS[1:]];label=batch['action'][:,0].cuda()
    if indices is not None:features=features[indices];args=[a[indices] for a in args];label=label[indices]
    with torch.no_grad():
        pred=net.from_features(features,*args);delta=pred-label;rmse=((delta/net.action_std)**2).mean(0).sqrt();mae=delta.abs().mean(0)
    return dict(loss=float(((delta/net.action_std)**2).mean()),normalized_RMSE=rmse.cpu(),physical_MAE_deg_deg_mm=mae.cpu(),C_MAE_um=float(mae[2]*1000))


def train_stage(net,initial,data,features,meta,sampler,tiny=False):
    import torch
    net.head.load_state_dict(initial);net.head.train();net.vision.eval()
    optimizer=torch.optim.AdamW(net.head.parameters(),lr=1e-3,weight_decay=1e-4)
    rng=torch.Generator().manual_seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    args=[data[k].cuda() for k in KEYS[1:]];label=data['action'][:,0].cuda()
    pools=sampler.pools_for(meta);selected=read(OUT/'tinyfit_indices.json')['indices']
    name='tinyfit' if tiny else 'formal';max_updates=512 if tiny else 1024
    if not tiny:save('sampling_pools.json',dict(pools=pools,class_probability=1/3,side_probability=.5,replacement=True,seed=SEED,draws_are_not_new_data=True))
    PROGRESS['tinyfit' if tiny else 'formal_training']='running';last=None;last_grads={}
    try:
        for step in range(1,max_updates+1):
            ids=selected if tiny else sampler.draw_batch(pools,rng,size=64)
            optimizer.zero_grad(set_to_none=True)
            pred=net.from_features(features[ids],*[a[ids] for a in args]);loss=(((pred-label[ids])/net.action_std)**2).mean()
            assert torch.isfinite(loss),'Nonfinite single-step normalized MSE'
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.head.parameters(),1.)
            assert torch.isfinite(grad),'Nonfinite head gradients'
            last_grads={n:p.grad.detach().cpu().clone() for n,p in net.head.named_parameters()}
            optimizer.step();assert all(torch.isfinite(p).all() for p in net.head.parameters())
            PROGRESS['tinyfit_updates' if tiny else 'formal_updates']=step
            row=dict(time=now(),update=step,loss=float(loss.detach()),gradient_norm_before_clip=float(grad),batch_indices=ids)
            append(OUT/f'{name}_updates.jsonl',row)
            report=step%128==0 if tiny else step in (128,512,1024)
            if report:
                last=training_metrics(net,features,data,selected if tiny else None)
                detail=dict(update=step,**last);append(OUT/f'{name}_train_metrics.jsonl',detail)
                save(f'{name}_progress.json',dict(status='running',update=step,train_metrics=last),True)
                status('tinyfit' if tiny else 'formal_training',train_metrics=last)
                print('TRAIN_METRICS',name,json.dumps(detail,default=default),flush=True)
                if tiny and bool((last['normalized_RMSE']<=.1).all()) and last['C_MAE_um']<=.2:break
        passed=not tiny or bool((last['normalized_RMSE']<=.1).all()) and last['C_MAE_um']<=.2
        changes={n:float((p.detach().cpu()-initial[n].detach().cpu()).abs().max()) for n,p in net.head.named_parameters()}
        result=dict(status='passed' if passed else 'failed',updates=step,train_metrics=last,weight_max_abs_changes=changes,initial_head_sha256=sha(OUT/'initial_head.pth'),AMP=False,optimizer='AdamW',lr=.001,weight_decay=.0001,clip=1.)
        save(f'{name}_result.json',result)
        torch.save(dict(head=net.head.state_dict(),optimizer=optimizer.state_dict(),update=step),OUT/f'{name}_checkpoint.pth')
        if tiny and not passed:torch.save(dict(gradients=last_grads,weight_deltas={n:p.detach().cpu()-initial[n].detach().cpu() for n,p in net.head.named_parameters()}),OUT/'tinyfit_failure_gradients_weights.pt')
        if not tiny:
            replay=torch.Generator().manual_seed(SEED)
            with (OUT/'formal_updates.jsonl').open() as f:
                for line in f:assert sampler.draw_batch(pools,replay,64)==json.loads(line)['batch_indices']
            assert all(int(x['step'])==1024 for x in optimizer.state.values())
        PROGRESS['tinyfit' if tiny else 'formal_training']=result['status'];record(name+'_'+result['status'],result)
        return passed,result
    except BaseException:
        torch.save(dict(gradients=last_grads,weight_deltas={n:p.detach().cpu()-initial[n].detach().cpu() for n,p in net.head.named_parameters()},last_update=PROGRESS['tinyfit_updates' if tiny else 'formal_updates']),OUT/f'{name}_interrupted_gradients_weights.pt')
        raise
    finally:net.head.eval();net.vision.eval()


def group_metrics(pred,label,meta):
    import numpy as np
    masks=dict(all=np.ones(len(meta),bool));masks.update({c:np.array([m['input_class']==c for m in meta]) for c in ('approaching','receding','settled')});masks.update({s:np.array([m['side']==s for m in meta]) for s in ('below','above')})
    masks['teacher_C_nonzero']=np.abs(label[:,2])>1e-9;masks['teacher_C_zero']=~masks['teacher_C_nonzero'];masks['strict_hold']=np.array([m['strict_hold'] for m in meta]);masks['settled_nonzero_command']=masks['settled']&masks['teacher_C_nonzero']
    groups={}
    for name,mask in masks.items():
        nonzero=mask&masks['teacher_C_nonzero'];mae=(np.abs(pred[mask]-label[mask,None]).mean((0,1))*[1,1,1000]).tolist() if mask.any() else None
        direction=float((np.sign(pred[nonzero,:,2])==np.sign(label[nonzero,None,2])).mean()) if nonzero.any() else None
        groups[name]=dict(windows=int(mask.sum()),physical_first_action_MAE_deg_deg_um=mae,C_direction_accuracy=direction,independent_episodes=len({meta[i]['episode'] for i in np.flatnonzero(mask)}),per_seed_physical_MAE_deg_deg_um=(np.abs(pred[mask]-label[mask,None]).mean(0)*[1,1,1000]).tolist() if mask.any() else None)
    weighted=((np.array(groups['approaching']['physical_first_action_MAE_deg_deg_um'])+groups['receding']['physical_first_action_MAE_deg_deg_um'])/2).tolist()
    return dict(groups=groups,equal_motion_weighted_MAE_deg_deg_um=weighted)


def offline(net,data,cache,manifest,sampler):
    import numpy as np,torch
    PROGRESS['offline']='running';caps=np.array(read(CARD)['offline']['caps']);results={}
    for split in SPLITS:
        batch=data[split];meta=manifest[split];args=[batch[k].cuda() for k in KEYS[1:]]
        with torch.no_grad():raw=net.from_features(cache[split],*args).cpu().numpy().astype(float)
        assert np.isfinite(raw).all();labels=batch['action'][:,0].numpy().astype(float);pred=np.clip(raw,-caps,caps)
        rows=[]
        for i,m in enumerate(meta):
            r={k:m[k] for k in ('episode','teacher_step','input_class','side','strict_hold','settled_nonzero_command')};r.update(window=i,model='new_deterministic_head',prediction_count=1)
            for j,(a,u) in enumerate(zip('ABC',('deg','deg','mm'))):r.update({f'raw_{a}_{u}':raw[i,j],f'production_{a}_{u}':pred[i,j],f'teacher_{a}_{u}':labels[i,j]})
            rows.append(r)
        write_csv(f'offline_new_{split}_per_window.csv',rows)
        baseline=np.empty((657,4,3),float)
        for r in csv.DictReader((PRIOR/f'offline_before_{split}_per_window_seed.csv').open()):baseline[int(r['window']),int(r['seed'])-92001]=[float(r[f'production_{a}_{u}']) for a,u in zip('ABC',('deg','deg','mm'))]
        before=group_metrics(baseline,labels,meta);after=group_metrics(pred[:,None,:],labels,meta)
        save(f'offline_{split}_metrics.json',dict(before=before,after=after,baseline_seeds=[92001,92002,92003,92004],candidate_predictions_per_window=1,comparison='baseline metrics average over4 saved seeds; no candidate replicate or claim of independent seeds'))
        results[split]=dict(before=before,after=after)
        record('offline_all_windows_completed',dict(split=split,windows=657,before=before['equal_motion_weighted_MAE_deg_deg_um'],after=after['equal_motion_weighted_MAE_deg_deg_um']))
    # Reuse the exact six prior registered zero-tolerance comparisons; permit closed loop ONLY here.
    gate=sampler.promotion(results['development_E41']['before'],results['development_E41']['after'])
    gate['closed_loop_allowed']=gate['status']=='passed';gate['baseline_saved_predictions_only']=True;gate['candidate_predictions_per_window']=1
    save('offline_gate.json',gate,True);PROGRESS['offline']=gate['status'];record('six_gate_decision',gate)
    return gate


def closed_loops():
    PROGRESS['closed_loop']='running';children=[];logs=[];started=time.monotonic();allowed=min(480.,DEADLINE-started);deadline=started+allowed
    try:
        for kind in KINDS:
            f=(OUT/f'worker_{kind}.log').open('x');logs.append(f)
            cmd=[str(ROOT/'.venv/bin/python'),'-E','-B','-u',str(OUT/'pilot.py'),'worker','--kind',kind]
            p=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,**RUNTIME},stdout=f,stderr=subprocess.STDOUT,start_new_session=True);children.append((kind,p))
            append(OUT/'worker_launches.jsonl',dict(time=now(),kind=kind,pid=p.pid,command=cmd))
        # Internal experiment orchestration only; the assistant does not poll after launching GNOME.
        import concurrent.futures
        pool=concurrent.futures.ThreadPoolExecutor(max_workers=4)
        try:
            futures={pool.submit(p.wait):kind for kind,p in children}
            for future in concurrent.futures.as_completed(futures,timeout=max(.01,deadline-time.monotonic())):
                rc=future.result();assert rc==0,f'Closed-loop worker {futures[future]} failed returncode{rc}; stop peers'
        finally:pool.shutdown(wait=False,cancel_futures=True)
        results=[read(OUT/'arms'/kind/'exec1'/'summary.json') for kind in KINDS]
        PROGRESS['closed_loop']='completed';record('four_closed_loops_completed',dict(arms=4,cycles=960,strict20=sum(r['metrics']['strict20um_pass'] for r in results)))
        return results
    finally:
        for _,p in children:
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for _,p in children:
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for f in logs:f.close()


def worker(kind):
    torch=setup_torch()
    from head import TerminalHead
    from lerobot_training.eval_policy import PolicyEvaluator
    helper=import_file('local_reused_closed_loop',OUT/'reuse_arm.py')
    e=PolicyEvaluator(str(REF),device='cuda',num_action_samples=1,n_action_execute_steps=1,flow_integration_steps=4,measure_inference=True,controller_variant='feedback')
    cp=torch.load(OUT/'candidate.pth',map_location='cuda',weights_only=False)
    net=TerminalHead(e.model.vision,cp['scales']).cuda();net.head.load_state_dict(cp['head']);net.eval();e.model=net
    assert read(OUT/'offline_gate.json')['closed_loop_allowed'] is True
    prior=import_file('local_prefix_specs',PRIOR/'pilot.py')
    spec=next(s for s in prior.plans() if s['target']=='E41' and s['level_um']==50 and s['side']+'_'+s['requested_incoming']==kind)
    try:
        result=helper.arm(e,kind,1,spec)
        baseline=read(BASE/'arms'/kind/'exec1'/'summary.json')
        comparison=dict(kind=kind,baseline_path=str(BASE/'arms'/kind/'exec1'),baseline_sha256=sha(BASE/'arms'/kind/'exec1'/'summary.json'),before=baseline['metrics'],after=result['metrics'],delta_last200_max_um=result['metrics']['last200_max_post_error_um']-baseline['metrics']['last200_max_post_error_um'],delta_endpoint_um=result['metrics']['endpoint_um']-baseline['metrics']['endpoint_um'])
        save(Path('arms')/kind/'exec1'/'baseline_comparison.json',comparison);record('closed_loop_saved_exec1_comparison',comparison)
        return 0
    except BaseException as err:
        record('closed_loop_worker_failed',dict(kind=kind,error=repr(err)));return 1


def finish(state,started,**extra):
    result=dict(time=now(),workflow_status=state,progress=PROGRESS,runtime_s=time.monotonic()-started,actual_budget_total_s=900,scientific_scope='E40 training / E41 selected development; software MuJoCo only; no clinical or hardware claim',single_factor_causality=False,**extra)
    save('summary.json',result,True)
    report=['# Frozen vision terminal head v1',f'\nWorkflow: {state}',f'\nProgress: {json.dumps(PROGRESS,ensure_ascii=False)}','\nE41 is selected development, not an independent test. Targets/scales/model size change together; no single-factor architecture claim.','\nSoftware/MuJoCo only; no hardware or clinical claim.']
    for filename in ('tinyfit_result.json','formal_result.json','offline_gate.json'):
        if (OUT/filename).exists():report+=['\n## '+filename,'```json',json.dumps(read(OUT/filename),ensure_ascii=False,indent=2),'```']
    if extra.get('closed_loop_results'):
        report+=['\n## Closed loop','|start|last200 max um|endpoint um|longest/terminal <=20 cycles|strict20|','|---|---:|---:|---|---|']
        for a in extra['closed_loop_results']:
            m=a['metrics'];report.append(f"|{a['kind']}|{m['last200_max_post_error_um']}|{m['endpoint_um']}|{m['longest_le20_cycles']}/{m['terminal_le20_cycles']}|{m['strict20um_pass']}|")
    (OUT/'report.md').write_text('\n'.join(report)+'\n')
    status('finished',state);record('workflow_finished',dict(status=state,progress=PROGRESS,summary=str(OUT/'summary.json'),report=str(OUT/'report.md')))


def run():
    global DEADLINE
    started=time.monotonic()
    with (OUT/'.execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/'actual_start_receipt.json').exists(),'Never restart scientific trial'
        receipt=dict(time=now(),status='actual_process_started',pid=os.getpid(),ppid=os.getppid(),argv=sys.argv,GNOME_TERMINAL_SCREEN=os.environ.get('GNOME_TERMINAL_SCREEN'),stage='identity_then_cache_parity',not_scientific_success=True)
        save('actual_start_receipt.json',receipt);PROGRESS['launch']='actual_started'
        try:
            fd=os.open(OUT/'startup_ack.fifo',os.O_WRONLY|os.O_NONBLOCK);os.write(fd,(json.dumps(receipt)+'\n').encode());os.close(fd)
        except OSError:pass
        prior=read(OUT/'preparation_budget.json')['elapsed_compute_s']
        DEADLINE=started+900-prior
        signal.signal(signal.SIGALRM,alarm)
        for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,lambda s,f:(_ for _ in ()).throw(InterruptedError(f'signal{s}')))
        try:
            assert receipt['GNOME_TERMINAL_SCREEN'],'Must be actual GNOME terminal'
            for k,v in {**DESKTOP,**RUNTIME}.items():assert os.environ.get(k)==v,(k,os.environ.get(k))
            record('actual_execution_started',receipt)
            PROGRESS['parity']='running'
            with budget('identity_cache_and_new_path_parity',240):
                changes=[p for p,h in read(OUT/'input_hashes.json').items() if not Path(p).exists() or sha(p)!=h]
                save('runtime_input_hash_verification.json',dict(status='passed' if not changes else 'failed',differences=changes));assert not changes,changes
                record('runtime_input_hashes_verified',dict(changed=changes))
                net,initial,vision_before,data,cache,manifest,sampler,baseline_audit,scales=prepare_features();PROGRESS['parity']='passed'
            # Redirect only imported utility result writers to the NEW task directory.
            sampler.OUT=OUT;sampler.save=save;sampler.record=record
            with budget('tinyfit_formal_training_and_offline',180):
                passed,tiny=train_stage(net,initial,data['train'],cache['train'],manifest['train'],sampler,tiny=True)
                if not passed:
                    PROGRESS.update(formal_training='skipped_tinyfit_gate',offline='skipped_tinyfit_gate',closed_loop='skipped_tinyfit_gate')
                    finish('tinyfit_gate_failed',started);return 0
                passed,formal=train_stage(net,initial,data['train'],cache['train'],manifest['train'],sampler,tiny=False)
                import torch
                for k,v in net.vision.state_dict().items():torch.testing.assert_close(v.cpu(),vision_before[k],atol=0,rtol=0)
                save('frozen_vision_verification.json',dict(status='passed',tensors=len(vision_before),all_bit_exact=True,training_mode=False));record('frozen_vision_bit_exact_after_training',dict(tensors=len(vision_before),all_bit_exact=True))
                torch.save(dict(head=net.head.state_dict(),scales=scales,step=1024,init_checkpoint=str(REF),init_sha256=REF_SHA,initial_head_sha256=sha(OUT/'initial_head.pth')),OUT/'candidate.pth')
                save('checkpoint_hashes.json',{n:sha(OUT/n) for n in ('initial_head.pth','candidate.pth','tinyfit_checkpoint.pth','formal_checkpoint.pth')})
                gate=offline(net,data,cache,manifest,sampler)
            if gate['status']!='passed':
                PROGRESS['closed_loop']='skipped_offline_gate';finish('offline_gate_failed',started);return 0
            del data,cache,net,initial,vision_before
            torch.cuda.empty_cache()
            with budget('conditional_four_local_closed_loops',480):results=closed_loops()
            finish('completed',started,closed_loop_results=results,strict20_pass_arms=sum(x['metrics']['strict20um_pass'] for x in results),closed_loop_arms=4)
            return 0
        except BaseException as err:
            signal.setitimer(signal.ITIMER_REAL,0)
            for key,value in list(PROGRESS.items()):
                if value=='running':PROGRESS[key]='failed'
            failure=dict(time=now(),error=repr(err),traceback=traceback.format_exc(),progress=PROGRESS,no_retry=True)
            save('failure.json',failure,True);record('runtime_stopped',failure)
            if PROGRESS['closed_loop']=='not_started':PROGRESS['closed_loop']='skipped_failure'
            finish('budget_exceeded' if isinstance(err,TimeoutError) else 'technical_failure',started,error=repr(err));return 1


def launch():
    with (OUT/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/'gnome_launch_request.json').exists() and not (OUT/'actual_start_receipt.json').exists(),'Refuse duplicate launch'
        assert read(OUT/'preflight.json')['status']=='passed'
        os.mkfifo(OUT/'startup_ack.fifo',0o600);fd=os.open(OUT/'startup_ack.fifo',os.O_RDWR|os.O_NONBLOCK)
        cmd=['gnome-terminal','--window','--title=Frozen vision deterministic terminal head v1','--working-directory='+str(ROOT),'--','bash',str(OUT/'launch.sh')]
        save('gnome_launch_request.json',dict(time=now(),command=cmd,desktop=DESKTOP,single_launch=True))
        try:
            done=subprocess.run(cmd,env={**os.environ,**DESKTOP},capture_output=True,text=True,timeout=15)
            receipt=dict(time=now(),returncode=done.returncode,command=cmd,stdout=done.stdout,stderr=done.stderr)
            save('gnome_launch_receipt.json',receipt);assert done.returncode==0,done.stderr
            selector=selectors.DefaultSelector();selector.register(fd,selectors.EVENT_READ);ready=selector.select(20)
            message=os.read(fd,65536).decode() if ready else '';selector.close()
            ack=dict(time=now(),one_startup_ack_only=True,no_phase_polling=True,actual_start=json.loads(message.splitlines()[0]) if message else None,timed_out=not bool(message),launch_receipt=receipt)
            save('launch_startup_acknowledgement.json',ack);record('gnome_startup_acknowledgement',ack)
            print('ACTUAL_START_RECEIPT',json.dumps(ack,ensure_ascii=False),flush=True)
            return 0 if message else 1
        except BaseException as err:
            failure=dict(time=now(),error=repr(err),traceback=traceback.format_exc(),no_retry=True,no_background_fallback=True)
            save('launch_failure.json',failure);record('gnome_launch_failed',failure)
            return 1
        finally:os.close(fd)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','launch','run','worker']);p.add_argument('--kind',choices=KINDS);a=p.parse_args()
    if a.mode=='prepare':
        try:prepare();return 0
        except BaseException as err:
            failure=dict(time=now(),error=repr(err),traceback=traceback.format_exc(),no_scientific_run_started=True)
            save(f'preparation_failure_{time.time_ns()}.json',failure);status('preparation_failed','preparation_failed',error=repr(err));record('preparation_failed',failure);raise
    if a.mode=='launch':return launch()
    if a.mode=='worker':return worker(a.kind)
    return run()
if __name__=='__main__':sys.exit(main())
