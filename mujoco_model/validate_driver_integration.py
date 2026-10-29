"""Exercise the unchanged uploaded controller against an isolated motor plant.

No ROS node, socket, real robot topic or environment network settings are used.
The 84 s closed loop advances a virtual clock; it is not measured hardware data.
"""
from pathlib import Path
import argparse
import json
import math
import traceback
import numpy as np

from driver_loader import load_driver, get_message_types
from motor_backend import MotorBackend

HERE=Path(__file__).resolve().parent
KIN=load_driver()
CONTROL,ENCODER=get_message_types()
HOME=(0,1466018,893518,0)
ORIGIN=(147,-13,-30901,-20)

def state_from_message(message):
    return {'robot':tuple(getattr(message,'robotEncoder'+a) for a in 'ABCE'),
            'handle':tuple(getattr(message,'handleEncoder'+a) for a in 'ABCE'),
            'position':tuple(getattr(message,'pos'+a) for a in 'XYZ'),
            'forward':bool(message.stepForward),'backward':bool(message.stepBackward)}

def command(handles):
    return KIN.make_control_message(handles,'hold')

def reject_without_mutation(backend,message,now):
    before=(backend.robot,backend.handles,backend.target,backend.last_command_s,len(backend.history))
    try:
        backend.consume(message,now)
    except (ValueError,TypeError,OverflowError) as exc:
        assert before==(backend.robot,backend.handles,backend.target,backend.last_command_s,len(backend.history)), 'Rejected packet mutated plant'
        return str(exc)
    raise AssertionError('Invalid command was accepted')

def protocol_checks():
    plant=MotorBackend(initial_handles=ORIGIN,tau_s=0,timeout_s=.5)
    target=tuple(a+b for a,b in zip(ORIGIN,(9,-7,13,0)))
    packet=command(target)
    receipt=plant.consume(packet,0)
    expected=KIN.predicted_robot_encoders(HOME,(9,-7,13,0))
    assert plant.target==expected
    plant.step(.02,.02)
    robot=plant.robot
    duplicate=plant.consume(packet,.03)
    assert duplicate['duplicate'] and plant.target==expected
    plant.step(.01,.04)
    assert plant.robot==robot,'Absolute retransmit produced another movement'
    assert state_from_message(plant.feedback())['handle']==target
    split=MotorBackend(initial_handles=ORIGIN,tau_s=0)
    for index,delta in enumerate([(1,-1,1,0),(4,-3,6,0),(9,-7,13,0)]):
        split.consume(command(tuple(a+b for a,b in zip(ORIGIN,delta))),index*.02)
    assert split.target==expected,'Packet splitting accumulated encoder roundoff'
    reasons={}
    invalid=command(target);invalid.handleEncoderE+=1
    reasons['unmapped_E']=reject_without_mutation(plant,invalid,.05)
    invalid=command(target);invalid.stepForward=True;invalid.stepBackward=True
    reasons['both_direction_flags']=reject_without_mutation(plant,invalid,.05)
    invalid=command(target);invalid.stepForward=True
    reasons['unspecified_direction_rate']=reject_without_mutation(plant,invalid,.05)
    invalid=command(target);invalid.handleEncoderA=2**31
    reasons['int32_overflow']=reject_without_mutation(plant,invalid,.05)
    invalid=command(target);invalid.handleEncoderA=10**6
    reasons['robot_limit']=reject_without_mutation(plant,invalid,.05)
    watchdog=MotorBackend(tau_s=.5,timeout_s=.1)
    watchdog.consume(command((0,0,100,0)),0)
    watchdog.step(.01,.01);current=watchdog.robot
    watchdog.step(.2,.21)
    assert watchdog.fault=='feedback_command_timeout'
    assert watchdog.target==watchdog.robot==current
    watchdog.step(.01,.22);assert watchdog.robot==current
    from robot_control.response_experiments import ResponseExperiment
    from robot_control.response_signals import ResponseSignal
    stale=ResponseExperiment(ResponseSignal(kind='step',baseline_s=2,hold_s=2,recovery_s=2),
                            state_from_message(MotorBackend().feedback()),(0,0,0,0),
                            100.,1700000000.,mode='c_axis',tip_offset_mm=4,offset_sign=-1)
    stale.observe(state_from_message(MotorBackend().feedback()),1700000000.,100.,1)
    stale.tick(100.)
    stale.tick(101.1)
    assert stale.status=='stopped' and stale.reason=='feedback_timeout'
    return {'status':'PASS','absolute_packets_idempotent':True,'nonzero_session_origin':ORIGIN,
            'same_target_after_packet_splitting':True,'original_predicted_target':expected,
            'atomic_rejections':reasons,'watchdog_status':watchdog.status,
            'watchdog_held_robot':watchdog.robot,
            'unchanged_controller_stale_feedback_reason':stale.reason,'no_ROS_or_network':True}

def original_closed_loop():
    from robot_control.base_position import BasePositionExperiment
    from robot_control.response_signals import ResponseSignal
    from model_sync import DriverModel
    initial=(HOME[0],HOME[1],HOME[2]+round(.5*KIN.ENCODER_COUNTS_PER_MM),HOME[3])
    backend=MotorBackend(initial_robot=initial,initial_handles=ORIGIN,needle_profile='new',tau_s=.04)
    scene=DriverModel(needle_profile='new')
    max_fk_error=0.
    def model_feedback():
        nonlocal max_fk_error
        diagnostic=scene.sync(backend.robot)
        error=float(diagnostic['max_fk_error_m'])
        assert math.isfinite(error) and error<1e-9,'MuJoCo/driver FK mismatch'
        max_fk_error=max(max_fk_error,error)
        return scene.feedback_from_model(backend.feedback())
    signal=ResponseSignal(kind='sine',amplitude_um=10,frequency_hz=.05,cycles=4,
                          baseline_s=2,recovery_s=2)
    engine=BasePositionExperiment(signal,state_from_message(model_feedback()),backend.handles,
                                  100.,1700000000.,tip_offset_mm=4,offset_sign=-1)
    elapsed=0
    for index in range(1,round((signal.duration_s+.1)/.01)+1):
        elapsed=index*.01;now=100+elapsed
        backend.step(.01,now)
        engine.observe(state_from_message(model_feedback()),1700000000+elapsed,now,index)
        decision=engine.tick(now)
        if decision and decision['action']=='send':
            if not engine.validate_dispatch(decision,now):break
            backend.consume(command(decision['handles']),now)
            engine.published(decision,now,1700000000+elapsed)
        if engine.status!='running':break
    snap=engine.snapshot()
    rows=snap['feedback']
    assert rows,'Original controller processed zero feedback'
    assert len(snap['commands'])>0,'Original controller emitted no commands'
    assert engine.status=='completed' and engine.reason=='schedule_finished', (engine.status,engine.reason)
    phases={phase:sum(r['phase']==phase for r in rows) for phase in sorted(set(r['phase'] for r in rows))}
    tail=rows[-25:]
    final=max(r['error_norm_um'] for r in tail)
    commands=snap['commands']
    per_axis={a:max(abs(c['handles'][j]-ORIGIN[j]) for c in commands) for j,a in enumerate('ABCE')}
    assert all(per_axis[a]>0 for a in 'ABC'),'Not all three task axes exercised'
    return {'status':'PASS','source_controller':type(engine).__name__,
            'source_controller_outcome':engine.status,'source_controller_reason':engine.reason,
            'signal':snap['signal'],'simulated_duration_s':elapsed,'wall_clock_duration_claimed':False,
            'plant':'chosen first-order lag tau=.04s and speed limits; not measured motor dynamics',
            'feedback_path':'MuJoCo driver_junction site -> inverse world transform -> Encoder float32 -> unchanged controller',
            'max_closed_loop_mujoco_fk_error_m':max_fk_error,
            'workpoint_junction_radius_mm':math.sqrt(sum(v*v for v in KIN.model_tip_from_robot_encoders(initial,0))),
            'initial_robot':initial,'initial_handles':ORIGIN,'needle_profile':'new',
            'feedback_count':len(rows),'published_command_count':len(commands),'phase_counts':phases,
            'maximum_error_um':max(r['error_norm_um'] for r in rows),
            'final_250ms_max_error_um':final,
            'maximum_absolute_session_handle_changes':per_axis,
            'plant_watchdog_holds':sum(h['event']=='watchdog_hold' for h in backend.history),
            'watchdog_note':'Controller only publishes corrections; silent deadband can trigger local hold. No synthetic keepalives injected.',
            'samples':rows[::50]+[rows[-1]],'commands':commands}, rows

def model_checks(rows):
    from model_sync import DriverModel
    reports=[]
    robots=[HOME]
    for da in (-300,0,300):
        for db in (-400,0,400):
            for dc in (-100,0,100):
                robots.append(KIN.predicted_robot_encoders(HOME,(da,db,dc,0)))
    robots.extend(tuple(r['robot']) for r in rows[::50])
    for profile in ('new','legacy'):
        scene=DriverModel(needle_profile=profile)
        diagnostics=[]
        for robot in robots:
            report=scene.sync(robot)
            if not isinstance(report,dict):raise TypeError('DriverModel.sync must return diagnostics')
            error=float(report['max_fk_error_m'])
            assert math.isfinite(error) and error<1e-9,'MuJoCo/driver FK mismatch'
            diagnostics.append(report)
        reports.append({'needle_profile':profile,'samples':len(robots),
                        'max_fk_error_m':max(float(r['max_fk_error_m']) for r in diagnostics),
                        'diagnostics':diagnostics})
    return {'status':'PASS','scope':'Actual MuJoCo site positions vs unchanged driver FK; scan-to-world alignment remains a chosen transform',
            'profile_reports':reports}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=HERE/'outputs')
    args=parser.parse_args();args.output_dir.mkdir(exist_ok=True,parents=True)
    result={'scope':'ROS-free original controller integration; virtual time, not real robot or ROS transport validation'}
    try:
        result['protocol']=protocol_checks()
        result['original_controller'],rows=original_closed_loop()
        result['mujoco']=model_checks(rows)
        result['status']='PASS'
    except Exception as exc:
        result.update(status='FAIL',error=str(exc),traceback=traceback.format_exc())
        (args.output_dir/'driver_validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,default=str)+'\n')
        raise
    (args.output_dir/'driver_validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,default=str)+'\n')
    print(json.dumps({'status':result['status'],'protocol':result['protocol']['status'],
        'controller_status':result['original_controller']['source_controller_outcome'],
        'controller_reason':result['original_controller']['source_controller_reason'],
        'virtual_seconds':result['original_controller']['simulated_duration_s'],
        'feedback':result['original_controller']['feedback_count'],
        'commands':result['original_controller']['published_command_count'],
        'max_error_um':result['original_controller']['maximum_error_um'],
        'model_profiles':len(result['mujoco']['profile_reports'])},indent=2))

if __name__=='__main__':main()
