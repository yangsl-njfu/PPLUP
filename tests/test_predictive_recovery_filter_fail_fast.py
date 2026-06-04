import unittest
from types import SimpleNamespace

import numpy as np

from ppl.utils.predictive_recovery_filter import (
    FrenetProjection,
    PredictiveRecoveryConfig,
    PredictiveRecoveryFilter,
    TrajectoryCandidate,
    TrajectoryRollout,
    TrajectoryScore,
)


class PredictiveRecoveryFilterFailFastTest(unittest.TestCase):
    def test_risk_metrics_attribute_error_raises_without_emergency_brake(self):
        recovery_filter = PredictiveRecoveryFilter(PredictiveRecoveryConfig())
        vehicle = SimpleNamespace(
            position=np.array([0.0, 0.0], dtype=float),
            heading_theta=0.0,
            speed=5.0,
            length=4.8,
            width=2.0,
            contact_results=None,
        )
        env = SimpleNamespace(vehicle=vehicle, user_data={"cost": [0.0]})
        frame = SimpleNamespace(
            project_point=lambda position, heading, validate_heading=True: FrenetProjection(
                True,
                s=0.0,
                l=0.0,
            )
        )

        recovery_filter._max_steer_rad = lambda *args, **kwargs: 1.0
        recovery_filter._ultra_light_gate = lambda *args, **kwargs: ""
        recovery_filter._ultra_short_horizon_emergency_check = (
            lambda *args, **kwargs: {"emergency_detected": False}
        )
        recovery_filter._get_route_frame = lambda *args, **kwargs: frame
        recovery_filter._parse_scene = lambda *args, **kwargs: ([], {})
        recovery_filter._generate_candidate_specs = lambda *args, **kwargs: []
        recovery_filter._evaluate_raw_action = lambda *args, **kwargs: {
            "predicted_collision": False,
            "predicted_out_of_road": False,
            "cost_risk": 0.0,
            "min_vehicle_margin": float("inf"),
            "min_static_margin": float("inf"),
            "min_boundary_margin": float("inf"),
            "deadlock_risk": 0.0,
            "total_score": 0.0,
            "failure_reason": "",
        }

        def raise_attribute_error(*args, **kwargs):
            raise AttributeError("synthetic risk metric failure")

        recovery_filter._risk_metrics = raise_attribute_error

        with self.assertLogs("ppl.utils.predictive_recovery_filter", level="ERROR") as captured:
            with self.assertRaisesRegex(AttributeError, "synthetic risk metric failure"):
                recovery_filter.filter(env, None, np.array([0.25, 0.4], dtype=np.float32))

        self.assertIn("PredictiveRecoveryFilter failed", "\n".join(captured.output))
        self.assertEqual(recovery_filter._intervention_count_this_ep, 0)
        self.assertEqual(recovery_filter._cooldown_remaining, 0)
        self.assertFalse(recovery_filter._last_recovery_active)
        self.assertIsNone(recovery_filter._last_safe_action)

    def test_hard_recovery_prefers_bypass_over_stop(self):
        recovery_filter = PredictiveRecoveryFilter(PredictiveRecoveryConfig())

        def make_rollout(candidate_type, speed_target, total_score):
            candidate = TrajectoryCandidate(
                candidate_id=0,
                candidate_type=candidate_type,
                lateral_target=1.5 if candidate_type.startswith("left_") else 0.0,
                speed_target=speed_target,
                raw_action=np.array([0.0, 0.0], dtype=np.float32),
            )
            rollout = TrajectoryRollout(
                candidate=candidate,
                times=np.array([0.2], dtype=float),
                positions=np.zeros((1, 2), dtype=float),
                headings=np.zeros(1, dtype=float),
                speeds=np.array([speed_target], dtype=float),
                accelerations=np.zeros(1, dtype=float),
                steer_actions=np.zeros(1, dtype=float),
                throttle_actions=np.zeros(1, dtype=float),
                frenet_s=np.zeros(1, dtype=float),
                frenet_l=np.zeros(1, dtype=float),
                hard_safe=True,
                min_obstacle_margin=2.0,
            )
            return TrajectoryScore(total_score=total_score), rollout

        stop = make_rollout("keep_stop", 0.0, 1.0)
        bypass = make_rollout("left_pass", 4.0, 10.0)

        _, selected = recovery_filter._select_hard_recovery_rollout([stop, bypass])

        self.assertEqual(selected.candidate.candidate_type, "left_pass")

    def test_raw_predicted_collision_is_hard_unsafe(self):
        recovery_filter = PredictiveRecoveryFilter(PredictiveRecoveryConfig())
        raw_eval = {
            "predicted_collision": True,
            "predicted_out_of_road": False,
            "failure_reason": "cut_in_danger",
        }
        metrics = {
            "ego_speed": 12.0,
            "raw_boundary_margin": 3.0,
            "ttc_vehicle_min": float("inf"),
            "ttc_static_min": float("inf"),
        }

        unsafe, reason = recovery_filter._raw_action_hard_unsafe(raw_eval, metrics, ego_speed=12.0)

        self.assertTrue(unsafe)
        self.assertEqual(reason, "raw_collision_risk:cut_in_danger")


if __name__ == "__main__":
    unittest.main()
