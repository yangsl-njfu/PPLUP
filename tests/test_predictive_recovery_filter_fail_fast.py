import unittest
from types import SimpleNamespace

import numpy as np

from ppl.utils.predictive_recovery_filter import (
    FrenetProjection,
    PredictiveRecoveryConfig,
    PredictiveRecoveryFilter,
    RoadBoundaryInfo,
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

    def test_longitudinal_vehicle_risk_checks_lane_change_before_stop(self):
        recovery_filter = PredictiveRecoveryFilter(PredictiveRecoveryConfig())

        def make_rollout(candidate_type, lateral_target, speed_target, total_score):
            candidate = TrajectoryCandidate(
                candidate_id=0,
                candidate_type=candidate_type,
                lateral_target=lateral_target,
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
                frenet_s=np.array([1.0], dtype=float),
                frenet_l=np.array([0.0], dtype=float),
                hard_safe=True,
                min_boundary_margin=2.0,
                min_obstacle_margin=2.0,
            )
            return TrajectoryScore(total_score=total_score), rollout

        stop = make_rollout("keep_stop", 0.0, 0.0, 1.0)
        follow = make_rollout("keep_pass", 0.0, 5.0, 0.5)
        lane_change = make_rollout("right_pass", -2.8, 4.0, 100.0)
        scene_info = {
            "front_blocking_object_type": "vehicle",
            "front_blocking_object_distance": 18.0,
            "blocker_left_gap": 0.0,
            "blocker_right_gap": 2.0,
        }

        _, selected = recovery_filter._select_hard_recovery_rollout(
            [stop, follow, lane_change],
            hard_risk_reason="vehicle_ttc",
            scene_info=scene_info,
        )

        self.assertEqual(selected.candidate.candidate_type, "right_pass")

    def test_hard_bypass_action_uses_lookahead_steer(self):
        config = PredictiveRecoveryConfig()
        config.hard_bypass_steer_lookahead_time = 0.75
        recovery_filter = PredictiveRecoveryFilter(config)
        candidate = TrajectoryCandidate(
            candidate_id=0,
            candidate_type="right_pass",
            lateral_target=-3.0,
            speed_target=8.0,
            raw_action=np.array([0.0, 0.0], dtype=np.float32),
        )
        rollout = TrajectoryRollout(
            candidate=candidate,
            times=np.array([0.25, 0.50, 0.75, 1.00], dtype=float),
            positions=np.zeros((4, 2), dtype=float),
            headings=np.zeros(4, dtype=float),
            speeds=np.full(4, 8.0, dtype=float),
            accelerations=np.zeros(4, dtype=float),
            steer_actions=np.array([0.03, -0.20, -0.45, -0.35], dtype=float),
            throttle_actions=np.array([0.10, 0.20, 0.20, 0.10], dtype=float),
            frenet_s=np.arange(4, dtype=float),
            frenet_l=np.array([0.0, -0.4, -1.2, -2.0], dtype=float),
            hard_safe=True,
            min_boundary_margin=2.0,
            min_obstacle_margin=2.0,
        )

        nominal_action = recovery_filter._candidate_action_from_rollout(rollout)
        bypass_action = recovery_filter._candidate_action_from_rollout(
            rollout,
            use_bypass_lookahead=True,
        )

        self.assertAlmostEqual(float(nominal_action[0]), 0.03)
        self.assertAlmostEqual(float(bypass_action[0]), -0.45)
        self.assertAlmostEqual(float(bypass_action[1]), 0.10)

    def test_static_hard_risk_moving_bypass_does_not_enter_safety_hold(self):
        recovery_filter = PredictiveRecoveryFilter(PredictiveRecoveryConfig())

        def make_rollout(candidate_type, speed_target):
            candidate = TrajectoryCandidate(
                candidate_id=0,
                candidate_type=candidate_type,
                lateral_target=1.5 if candidate_type.startswith("left_") else 0.0,
                speed_target=speed_target,
                raw_action=np.array([0.0, 0.0], dtype=np.float32),
            )
            return TrajectoryRollout(
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

        reason = "static_object_distance;raw_collision_risk:static_collision"
        bypass = make_rollout("left_pass", 4.0)
        stop = make_rollout("keep_stop", 0.0)

        self.assertFalse(
            recovery_filter._hard_intervention_should_enter_safety_hold(reason, False, bypass)
        )
        self.assertTrue(
            recovery_filter._hard_intervention_should_enter_safety_hold(reason, False, stop)
        )

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

    def test_low_boundary_margin_is_hard_unsafe(self):
        recovery_filter = PredictiveRecoveryFilter(PredictiveRecoveryConfig())
        raw_eval = {
            "predicted_collision": False,
            "predicted_out_of_road": False,
            "min_boundary_margin": 0.25,
        }
        metrics = {
            "raw_boundary_margin": 0.25,
            "ttc_vehicle_min": float("inf"),
            "ttc_static_min": float("inf"),
        }

        unsafe, reason = recovery_filter._raw_action_hard_unsafe(raw_eval, metrics, ego_speed=8.0)

        self.assertTrue(unsafe)
        self.assertEqual(reason, "raw_boundary_margin_low")

    def test_hard_recovery_latched_side_is_preferred(self):
        recovery_filter = PredictiveRecoveryFilter(PredictiveRecoveryConfig())
        recovery_filter._hard_bypass_side = "right"
        recovery_filter._hard_bypass_remaining = 3

        def make_rollout(candidate_type, total_score):
            candidate = TrajectoryCandidate(
                candidate_id=0,
                candidate_type=candidate_type,
                lateral_target=-1.5 if candidate_type.startswith("right_") else 1.5,
                speed_target=4.0,
                raw_action=np.array([0.0, 0.0], dtype=np.float32),
            )
            rollout = TrajectoryRollout(
                candidate=candidate,
                times=np.array([0.2], dtype=float),
                positions=np.zeros((1, 2), dtype=float),
                headings=np.zeros(1, dtype=float),
                speeds=np.array([4.0], dtype=float),
                accelerations=np.zeros(1, dtype=float),
                steer_actions=np.zeros(1, dtype=float),
                throttle_actions=np.zeros(1, dtype=float),
                frenet_s=np.zeros(1, dtype=float),
                frenet_l=np.zeros(1, dtype=float),
                hard_safe=True,
                min_boundary_margin=2.0,
                min_obstacle_margin=2.0,
            )
            return TrajectoryScore(total_score=total_score), rollout

        left = make_rollout("left_pass", 1.0)
        right = make_rollout("right_pass", 10.0)

        _, selected = recovery_filter._select_hard_recovery_rollout([left, right])

        self.assertEqual(selected.candidate.candidate_type, "right_pass")

    def test_dynamic_blocker_overtake_selects_passing_rollout(self):
        recovery_filter = PredictiveRecoveryFilter(PredictiveRecoveryConfig())

        def make_rollout(
            candidate_type,
            lateral_target,
            speed_target,
            total_score,
            terminal_s,
            min_vehicle_margin=1.5,
        ):
            candidate = TrajectoryCandidate(
                candidate_id=0,
                candidate_type=candidate_type,
                lateral_target=lateral_target,
                speed_target=speed_target,
                raw_action=np.array([0.0, 0.0], dtype=np.float32),
            )
            rollout = TrajectoryRollout(
                candidate=candidate,
                times=np.array([0.2, 0.4], dtype=float),
                positions=np.zeros((2, 2), dtype=float),
                headings=np.zeros(2, dtype=float),
                speeds=np.full(2, speed_target, dtype=float),
                accelerations=np.zeros(2, dtype=float),
                steer_actions=np.zeros(2, dtype=float),
                throttle_actions=np.zeros(2, dtype=float),
                frenet_s=np.array([1.0, terminal_s], dtype=float),
                frenet_l=np.array([0.0, lateral_target], dtype=float),
                hard_safe=True,
                min_boundary_margin=2.0,
                min_vehicle_margin=min_vehicle_margin,
                min_obstacle_margin=2.0,
            )
            return TrajectoryScore(
                total_score=total_score,
                terminal_passed_blocker=terminal_s > 26.0,
                terminal_recoverable=terminal_s > 20.0,
            ), rollout

        scene_info = {
            "front_blocking_object_type": "vehicle",
            "front_blocking_is_dynamic_vehicle": True,
            "front_blocking_object_distance": 18.0,
            "front_blocking_object_l": 0.0,
            "blocker_left_gap": 2.5,
            "blocker_right_gap": 0.2,
        }
        metrics = {
            "raw_boundary_margin": 3.0,
            "ttc_vehicle_min": float("inf"),
        }
        raw_eval = {
            "predicted_collision": False,
            "predicted_out_of_road": False,
        }

        self.assertTrue(
            recovery_filter._dynamic_blocker_overtake_required(
                scene_info,
                metrics,
                raw_eval,
                ego_speed=12.0,
            )
        )

        follow = make_rollout("keep_pass", 0.0, 5.0, 0.1, 22.0)
        shallow_pass = make_rollout("left_pass", 1.5, 5.0, 0.2, 24.0)
        unsafe_overtake = make_rollout("left_pass", 3.0, 5.0, -10.0, 32.0, min_vehicle_margin=-1.0)
        overtake = make_rollout("left_pass", 2.8, 5.0, 5.0, 30.0)

        self.assertFalse(
            recovery_filter._dynamic_blocker_overtake_candidate_acceptable(unsafe_overtake[1])
        )

        _, selected = recovery_filter._select_dynamic_blocker_overtake_rollout(
            [follow, shallow_pass, unsafe_overtake, overtake],
            scene_info,
        )

        self.assertEqual(selected.candidate.candidate_type, "left_pass")
        self.assertEqual(selected.candidate.lateral_target, 2.8)

    def test_road_line_contact_text_is_not_hard_contact(self):
        recovery_filter = PredictiveRecoveryFilter(PredictiveRecoveryConfig())

        self.assertTrue(
            recovery_filter._contact_text_only_benign_road_marking("{ROAD_LINE_SOLID_SINGLE_WHITE}")
        )
        self.assertFalse(
            recovery_filter._contact_text_only_benign_road_marking("{ROAD_LINE_SOLID_SINGLE_WHITE, sidewalk}")
        )

    def test_front_vehicle_becomes_dynamic_blocker_with_overtake_candidates(self):
        config = PredictiveRecoveryConfig()
        recovery_filter = PredictiveRecoveryFilter(config)
        ego = SimpleNamespace(
            position=np.array([0.0, 0.0], dtype=float),
            heading_theta=0.0,
            speed=12.0,
            length=4.8,
            width=2.0,
            navigation=True,
        )
        front_vehicle = SimpleNamespace(
            id="front_vehicle",
            position=np.array([35.0, 0.0], dtype=float),
            heading_theta=0.0,
            speed=10.0,
            length=4.8,
            width=2.0,
            navigation=True,
        )
        env = SimpleNamespace(vehicle=ego, objects=[front_vehicle])

        def project_point(position, heading, validate_heading=True):
            del heading, validate_heading
            return FrenetProjection(True, s=float(position[0]), l=float(position[1]))

        def boundary_at(s, l, ego_width, safety_margin):
            del s
            l_min = -5.25
            l_max = 5.25
            left_margin = l_max - float(l) - ego_width * 0.5 - safety_margin
            right_margin = float(l) - l_min - ego_width * 0.5 - safety_margin
            return RoadBoundaryInfo(
                s=0.0,
                l_min=l_min,
                l_max=l_max,
                left_margin=left_margin,
                right_margin=right_margin,
                boundary_margin=min(left_margin, right_margin),
            )

        frame = SimpleNamespace(project_point=project_point, boundary_at=boundary_at)
        ego_projection = FrenetProjection(True, s=0.0, l=0.0)

        objects, scene_info = recovery_filter._parse_scene(
            env,
            frame,
            ego_projection,
            ego_speed=12.0,
            ego_width=2.0,
        )
        candidates = recovery_filter._generate_candidate_specs(
            frame,
            ego_projection,
            ego_pos=np.array([0.0, 0.0], dtype=float),
            ego_heading=0.0,
            ego_speed=12.0,
            ego_width=2.0,
            raw_action=np.array([0.0, 0.8], dtype=np.float32),
            scene_info=scene_info,
            max_steer_rad=1.0,
        )

        self.assertEqual(len(objects), 1)
        self.assertEqual(scene_info["front_blocking_object_type"], "vehicle")
        self.assertEqual(scene_info["front_blocking_object_reason"], "dynamic_vehicle")
        self.assertTrue(scene_info["front_blocking_is_dynamic_vehicle"])
        self.assertTrue(
            any(
                abs(candidate.lateral_target - ego_projection.l) >= config.overtake_min_lateral_shift
                and candidate.speed_target > 1.0
                and (
                    candidate.candidate_type.startswith("left_")
                    or candidate.candidate_type.startswith("right_")
                )
                for candidate in candidates
            )
        )


if __name__ == "__main__":
    unittest.main()
