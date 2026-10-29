"""Video-informed silver cover display law, separate from task-tip kinematics."""
from pathlib import Path
import json,math
import numpy as np

def settings(root,kin):
    cfg=json.loads((Path(root)/'silver_motion_config.json').read_text())
    axis=np.array(cfg['world_translation_axis'],dtype=float)
    if not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-8:raise ValueError('Invalid silver guide axis')
    axis/=np.linalg.norm(axis)
    gain=float(cfg['gain'])*float(cfg['direction_sign'])
    if not math.isfinite(gain) or gain==0:raise ValueError('Invalid visual travel gain')
    zero=int(cfg['encoder_reference'])
    lo,hi=kin.ROBOT_ENCODER_LIMITS['B']
    bounds=sorted([(v-zero)/kin.ENCODER_COUNTS_PER_MM*.001*gain for v in (lo,hi)])
    return cfg,axis,gain,zero,bounds

def native_polynomial(kin,gain,zero):
    alpha0=kin.b_alpha_from_robot_encoder(zero)
    limits=[math.radians(kin.b_alpha_from_robot_encoder(v)-alpha0) for v in kin.ROBOT_ENCODER_LIMITS['B']]
    x=np.linspace(*limits,1201)
    y=np.array([(kin.b_distance_from_alpha(alpha0+math.degrees(t))-kin.b_distance_from_alpha(alpha0))*.001*gain for t in x])
    basis=np.column_stack([x**i for i in range(1,5)])
    coefficients=np.r_[0,np.linalg.lstsq(basis,y,rcond=None)[0]]
    maximum_error=float(np.max(np.abs(basis@coefficients[1:]-y)))
    return coefficients,maximum_error
