"""Bounded prelaunch validation; never trains or launches closed-loop evaluation."""
import ast
from pathlib import Path
import os
import sys
import time
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT))
import campaign as c
for key,value in c.RUNTIME.items():os.environ[key]=value
import data_pipeline as d

def main():
    started=time.monotonic()
    try:
        for name in ('campaign.py','model.py','collection_helper.py','data_pipeline.py','preflight_validate.py'):
            ast.parse((OUT/name).read_text(),filename=name)
        assert c.sha(c.REF)==c.REF_SHA
        assert c.read(d.OLD/'summary.json')['workflow_status']=='batch_complete_near_negative'
        targets={s['target'] for s in d.plans()};assert targets=={'E40','E42'}
        for target in targets:
            for side in ('below','above'):
                for motion in ('approaching','receding','hold'):
                    specs=[s for s in d.plans() if s['target']==target and s['side']==side and s['requested_incoming']==motion]
                    assert {tuple(s['AB_sign']) for s in specs}=={(-1,-1),(-1,1),(1,-1),(1,1)}
        import shutil
        assert shutil.which('gnome-terminal')
        assert Path(c.DESKTOP['XDG_RUNTIME_DIR']).exists()
        assets=[c.REF,d.OLD/'candidate1/checkpoint.pth',d.OLD/'original_visual_cache.pt',c.PRIOR/'window_manifest.json',
            c.ROOT/'configs/v12_reachable_development_targets_0p35um.json',c.PRIOR/'pilot.py',c.ORIGINAL/'pilot.py']
        for path in assets:assert path.is_file(),path
        torch=c.torch_setup();e=c.evaluator();m=e.model
        from model import install
        # Reuse real full-token windows for the input counterfactual; no new dataset or model fitting.
        old=torch.load(d.OLD/'original_visual_cache.pt',map_location='cpu',weights_only=False)
        mini={s:{k:v[:2].clone() for k,v in b.items()} for s,b in old.items()}
        del old
        for b in mini.values():b[c.KEYS[0]]=torch.zeros(2,2,1)
        install(m)
        checks=d.verify_inputs(m,mini,OUT/'preflight_input_ablation.json')
        # Verify actual six-block differentiability and a finite training loss without an optimizer update.
        m.train();args=tuple(mini['train'][k][:1].cuda() for k in c.KEYS)
        from model import integrate
        output=integrate(m,args)
        loss=((output[:,0]-mini['train']['action'][:1,0].cuda())/m.action_std).square().mean()
        loss.backward()
        probes={str(i):max(float(p.grad.norm()) for n,p in m.named_parameters()
                    if n.startswith(f'blocks.{i}.self_attention.') and p.grad is not None) for i in range(6)}
        assert torch.isfinite(loss) and all(v>0 for v in probes.values())
        c.save('preflight_gradient_path.json',dict(status='passed',optimizer_updates=0,loss=float(loss.detach()),six_block_gradient_norms=probes))
        del e,m,mini;torch.cuda.empty_cache()
        # Genuine sensor/teacher setup check at both registered training targets.
        import numpy as np
        from rcm.scene_v9 import RCMSceneV9
        from scripts.research.homog_replay_common import make_teacher
        from lerobot_training.eval_policy import TASK_TO_DEG_DEG_MM
        rows=[]
        for target in ('E40','E42'):
            scene=teacher=None
            try:
                scene=RCMSceneV9()
                task=next(x['task'] for x in c.read(OUT/'experiment_card.json')['targets'] if x['task']['id']==target)
                world=np.asarray(scene.robot_to_world(np.asarray(task['target_robot_mm'])))
                np.testing.assert_allclose(world,task['target_world_mm'],atol=1e-9,rtol=0)
                scene.set_target_world(world);q=scene.target_task(validate=True).copy();q[2]+=.00005;scene.set_task(q)
                scene.enable_motor_execution();scene.reset_motor_time(0.)
                teacher=make_teacher(scene);teacher.sensor.observation_time_s=0.
                obs=teacher.sensor.observe()
                result=teacher.geometry_action_from_observation(scene.current_task(),obs,
                    previous_velocity=scene.motor_execution.actual_velocity.copy(),command_q=scene.current_task())
                assert result['valid'] and obs.get('oct_navigation_depth',{}).get('delta_xyz_m') is not None
                rows.append(dict(target=target,teacher_valid=True,teacher_ABC=result['action_deg_deg_mm'],
                    OCT=obs['oct_navigation_depth'],RGB_shape=list(np.asarray(obs['rgb']).shape),motor_driver=scene.motor_execution.last_record,
                    action_executed=False,truth_used_only_target_initialization=True))
            finally:
                if teacher is not None:teacher.sensor.close()
                if scene is not None:scene.close()
        c.save('preflight_teacher_sensor.json',dict(status='passed',targets=rows))
        c.save('preflight.json',dict(status='passed',time=c.now(),previous_nonrun='preflight_attempt1_not_run.json',
            device='physicalGPU0 RTX4090',target_split=dict(train=['E40','E42'],development_only=['E41'],standard_excluded=True),
            input_ablation_passed=checks['image_and_absolute_ABC_invariant'],real_teacher_sensor_validated=True,
            optimizer_updates=0,collection_branches=len(d.plans()),output_scope=str(OUT),
            launch='one GNOME launch; FIFO startup ack only; no status polling',validation_s=time.monotonic()-started))
        c.record('corrected_static_and_interface_preflight',c.read(OUT/'preflight.json'))
    except BaseException as err:
        import traceback
        path=OUT/f'preflight_failure_{time.time_ns()}.json'
        c.save(path,dict(status='failed',time=c.now(),error=repr(err),traceback=traceback.format_exc(),validation_s=time.monotonic()-started))
        c.record('corrected_preflight_failure',c.read(path));raise
    finally:
        ledger=OUT/'preparation_budget.json'
        old=c.read(ledger) if ledger.exists() else dict(validation_s=0.)
        old['validation_s']+=time.monotonic()-started
        c.save(ledger,old)

if __name__=='__main__':main()
