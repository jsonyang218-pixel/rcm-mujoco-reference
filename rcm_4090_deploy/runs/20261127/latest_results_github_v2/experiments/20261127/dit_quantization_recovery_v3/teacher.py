"""TRAINING ONLY: measured-geometry teacher, nearest executable G10 command.

No inference code imports this module. No command accumulator or scalar action
amplifier: choose the reachable pose with smallest measured geometric residual.
The original production evaluator clips, previews and commits every command.
"""
import itertools
import numpy as np

def model_tip(scene,q):
    return np.asarray(scene.robot_to_world(scene.nominal.forward(scene.state_from_task(q),enforce_limits=False)))

class QuantizedTeacher:
    def __init__(self, original, scene):
        self.original=original;self.scene=scene;self.sensor=original.sensor
        self.last_diagnostic=None

    def geometry_action_from_observation(self,q,observation,previous_velocity=None,command_q=None):
        check=self.original.geometry_action_from_observation(q,observation,previous_velocity,q)
        if not check['valid']:return check
        geometry=observation['geometry']
        tip=np.asarray(geometry['tip_world_mm'],float)
        goal=self.original._visual_goal(geometry['target_world_mm'],tip,observation)
        error=np.asarray(goal)-tip
        jac=self.original._model_jacobian(q)
        correction=np.linalg.lstsq(jac,error,rcond=None)[0]
        unit=np.array([np.pi/180,np.pi/180,.001]);caps=np.array([.2,.1,.03])
        ideal=np.clip(correction/unit,-caps,caps)
        # Exact protocol quanta; these generate teacher candidates only.
        backend=self.scene.motor_execution.backend;kin=backend.kin
        quanta=np.array([2*kin.CONTROL_GEAR*3/4000,kin.CONTROL_GEAR*3/4000,
                         3*kin.CONTROL_GEAR*10000/4000/kin.ENCODER_COUNTS_PER_MM])
        base=model_tip(self.scene,q)
        candidates=[np.zeros(3)]+[ideal+np.asarray(offset)*quanta for offset in itertools.product((-1,0,1),repeat=3)]
        seen=set();best=None
        for request in candidates:
            if np.any(np.abs(request)>caps+1e-12):continue
            try:
                desired=self.scene.validate_task(q+request*unit)
                encoded=tuple(round(float(v)) for v in self.scene.encoders(desired))+(0,)
                command=backend.command_for_robot_target(encoded)
                total=tuple(getattr(command,'handleEncoder'+a)-h for a,h in zip('ABCE',backend._anchor_handles))
                target=tuple(kin.validate_manual_jog(backend._anchor_robot,total))
                if target in seen:continue
                seen.add(target)
                qt=self.scene.task_from_encoders(target[:3],reference_task=q)
                predicted=model_tip(self.scene,qt)-base
                residual=error-predicted
                cost=float(residual@residual)+1e-12*float(np.sum((request/caps)**2))
                if best is None or cost<best[0]:best=(cost,request.copy(),encoded,target,predicted,residual)
            except ValueError:continue
        if best is None:raise RuntimeError('No valid G10 teacher candidate')
        self.last_diagnostic=dict(measured_error_world_mm=error.tolist(),ideal_inverse_action_deg_deg_mm=ideal.tolist(),
            chosen_action_deg_deg_mm=best[1].tolist(),requested_ABCE=best[2],quantized_target_ABCE=best[3],
            before_ABCE=list(backend.robot),predicted_displacement_mm=best[4].tolist(),
            predicted_residual_um=float(np.linalg.norm(best[5])*1000),candidate_count=len(seen),
            teacher_only=True,truth_used=False,accumulator=False)
        return dict(check,action_deg_deg_mm=best[1].tolist())
