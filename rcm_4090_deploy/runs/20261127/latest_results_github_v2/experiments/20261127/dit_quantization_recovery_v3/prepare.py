"""Generate the explicit authorized experiment card; does not launch work."""
import hashlib
import json
import campaign as c

old=c.OUT.parent/'dit_relative_v2_resume'
base=c.read(old/'experiment_card.json')
card={
    'name':c.OUT.name,'created_at':c.now(),'output_scope':str(c.OUT),
    'hypothesis':'Nonzero v2 actions remain below actual G10 thresholds; expand measured-geometry real G10 teacher lateral recovery coverage and warmstart the original near DiT.',
    'init_checkpoint':str(old/'candidate1/checkpoint.pth'),'init_sha256':c.sha(old/'candidate1/checkpoint.pth'),
    'far_checkpoint':str(c.REF),'targets':base['targets'],'near_task':base['near_task'],
    'baseline':dict(source=str(old),near_pass='4/4',standard_pass='1/5',
        E00_E06_E08_E10_um=[46.567830434903776,31.04965603182882,38.592537304154675,35.978458963537875]),
    'data':dict(training=['E40','E42'],excluded=['E41','E00','E02','E06','E08','E10'],
        new_branches=92,factorial='2targets x5 lateral20/35/50/75/100um x8 AB directions +2targets x2 Csigns x3 axial20/50/100um',
        collector='real strict RGB2/OCT + teacher-only measured-geometry inverse and nearest executable G10 candidate; no true XYZ in action decisions',
        trajectory='2 actual prefix ticks +64 teacher ticks; one observation per motor timestamp; no in-flight reset',
        labels='actual safety-accepted ABC; genuine contiguous future16; no synthetic label/interpolation/padding',
        reuse='v2 real E40/E42 teacher windows only where sensor lateral<=5um, to retain axial/hold behavior',
        balance='stationary executable recovery / recovery / new hold / v2 axial retention'),
    'architecture':dict(base['architecture'],
        inputs='unchanged two OCT frames + measured velocity from actual encoder frames + previous accepted ABC; same RGB/absoluteABC mask',
        output_scales='frozen v2 train-only mean/std; no external gain, geometric correction or command accumulator during policy evaluation'),
    'training':dict(updates=2400,batch=12,lr=3e-5,optimizer='AdamW',weight_decay=1e-4,gradient_clip=1.,
        loss='first-action normalized MSE +.02 genuine16-action MSE +.002 original RF loss',
        precision='v2 FP32 parity with TF32 off; shared trainer AMP defaults unchanged',
        validation='E40/E42-only fixed offline recovery/hold diagnostics; finite loss/gradient, all6blocks changed; no E41 fitting/scales',
        checkpoints='every600 and finally, optimizer plus CPU/CUDA/sampler RNG'),
    'near':base['near'],'standard':base['standard'],'protocol':base['protocol'],
    'near_regression_branch':dict(trigger='any of four E41 strict20um gates fails',
        diagnosis='save last200 independent XYZ/signedOCT/acceptedABC/encoder hold counts against v2',
        repair='one1200update fit atlr1e-5 initialized from candidate1; same E40/E42-only data with increased axial/hold retention; no E41 relabeling',
        next='repeat original four E41 starts once; only4/4 strict pass authorizes standard five',
        stop='second near strict failure blocks standard; retain all results'),
    'budget_s':2700,'concurrency_max':12,
    'stage_max_seconds':dict(collection=760,build=160,train=450,near=360,repair_train=240,repair_near=360,standard=970,shared_hard=2700,cleanup_reserve=30),
    'budget_policy':'All stage caps subordinate to ONE shared2700s deadline; workers terminate at2670 with30s cleanup reserve',
    'launch':'once in new GNOME; FIFO actual PID acknowledgement only; no postlaunch wait/poll',
    'reporting':'root/local TEST_LOG locked/fsync/readback each outcome; real RGB/OCT, raw/clipped/accepted actions, requested/quantized/actual encoder values and independent pre/postXYZ',
    'implementation_scope':'all modified implementation under new run directory; root TEST_LOG append only',
    'simulation_only':True,
}
c.save('experiment_card.json',card)
failure=c.OUT/'preflight_console.log'
if failure.exists() and not (c.OUT/'preflight.json').exists():
    archived=c.OUT/'preflight_attempt1_missing_card.log'
    if not archived.exists():failure.rename(archived)
    c.record('prelaunch_engineering_failure_repaired',dict(status='repaired_before_launch',
        reason='Card generation command had a Python syntax error; preflight correctly rejected missing card',
        evidence=str(archived),committed_motor_ticks=0,optimizer_updates=0,recovery='Generate card using task-local prepare.py before repeating interface preflight'))
