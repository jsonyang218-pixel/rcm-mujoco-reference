from __future__ import annotations

import unittest

from restricted_perception.stage4_analysis import (
    classify_failure,
    summarize_values,
)


def _episode(
    *,
    minimum_um,
    final_um,
    success=False,
    reason=None,
):
    return {
        "minimum_error_mm": minimum_um / 1000.0,
        "final_error_mm": final_um / 1000.0,
        "success_by_threshold_um": {"20": success},
        "early_stop_reason": reason,
        "steps": 1,
        "fallback_state_frames": {},
    }


class Stage4AnalysisTests(unittest.TestCase):
    def test_failure_classification_separates_hold_and_initialization(self):
        self.assertEqual(
            classify_failure(_episode(minimum_um=5.0, final_um=60.0)),
            "REACHED_20UM_BUT_DID_NOT_HOLD",
        )
        self.assertEqual(
            classify_failure(_episode(minimum_um=39.0, final_um=100.0)),
            "NEAR_30_TO_50UM",
        )
        self.assertEqual(
            classify_failure(
                _episode(
                    minimum_um=30000.0,
                    final_um=30000.0,
                    reason="EXCEPTION:ValueError:B 轴当前机器人编码器 1 不在已确认安全范围",
                )
            ),
            "UNSAFE_INITIALIZATION",
        )
        self.assertEqual(
            classify_failure(_episode(minimum_um=1.0, final_um=9.0, success=True)),
            "SUCCESS_20UM",
        )

    def test_value_summary_uses_requested_population_statistics(self):
        summary = summarize_values([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 4.0)
        self.assertAlmostEqual(summary["mean"], 2.5)


if __name__ == "__main__":
    unittest.main()
