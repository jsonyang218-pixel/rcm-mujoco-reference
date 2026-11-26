"""Pure supervised terminal policy. Only the explicit twelve measured inputs enter."""
import json
from pathlib import Path

import numpy as np


FEATURE_NAMES = [
    'signed_oct_x_mm', 'signed_oct_y_mm', 'signed_oct_z_mm',
    'A_deg', 'B_deg', 'C_mm',
    'actual_velocity_A_deg_s', 'actual_velocity_B_deg_s', 'actual_velocity_C_mm_s',
    'previous_accepted_dA_deg', 'previous_accepted_dB_deg', 'previous_accepted_dC_mm',
]


def measured_features(signed_oct_mm, state_deg_deg_mm,
                      actual_velocity_deg_deg_mm_s, previous_accepted_deg_deg_mm):
    values = np.concatenate([np.asarray(x, dtype=np.float64) for x in (
        signed_oct_mm, state_deg_deg_mm, actual_velocity_deg_deg_mm_s,
        previous_accepted_deg_deg_mm)])
    if values.shape != (12,) or not np.isfinite(values).all():
        raise ValueError('Policy requires exactly twelve finite measured features')
    return values


class LearnedPolicy:
    def __init__(self, model_path):
        model = json.loads(Path(model_path).read_text())
        if model['feature_names'] != FEATURE_NAMES:
            raise ValueError('Feature schema mismatch')
        self.mean = np.asarray(model['input_mean'], dtype=np.float64)
        self.scale = np.asarray(model['input_scale'], dtype=np.float64)
        self.coefficients = np.asarray(model['standardized_coefficients_12x3'], dtype=np.float64)
        self.intercept = np.asarray(model['intercept_deg_deg_mm'], dtype=np.float64)
        if self.coefficients.shape != (12, 3) or np.any(self.scale <= 0):
            raise ValueError('Invalid learned model')

    def predict(self, measured_input):
        x = np.asarray(measured_input, dtype=np.float64)
        if x.shape != (12,) or not np.isfinite(x).all():
            raise ValueError('Invalid measured input')
        return ((x - self.mean) / self.scale) @ self.coefficients + self.intercept
