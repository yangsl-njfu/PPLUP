"""Unit tests for the lightweight static RSS filter.

Run:
    python test_static_rss_filter.py
"""

import unittest

from ppl.utils.rss_cbf_filter import RSSCBFConfig, RSSCBFFilter
from ppl.utils.static_rss_filter import StaticRSSConfig, StaticRSSFilter


def make_state(with_obstacle=True, left_available=False, right_available=False):
    lanes = {
        "current": {
            "lane_id": "center",
            "width": 3.5,
        }
    }
    if left_available:
        lanes["left"] = {
            "lane_id": "left",
            "width": 3.5,
            "available": True,
            "drivable": True,
        }
    if right_available:
        lanes["right"] = {
            "lane_id": "right",
            "width": 3.5,
            "available": True,
            "drivable": True,
        }

    state = {
        "ego": {
            "x": 0.0,
            "y": 0.0,
            "heading": 0.0,
            "speed": 8.0,
            "lane_id": "center",
            "lane_width": 3.5,
        },
        "static_obstacles": [],
        "vehicles": [],
        "lanes": lanes,
    }
    if with_obstacle:
        state["static_obstacles"].append(
            {
                "x": 30.0,
                "y": 0.0,
                "length": 4.5,
                "width": 2.0,
                "heading": 0.0,
                "lane_id": "center",
            }
        )
    return state


class DummyMetaDriveObject:
    LENGTH = 4.5
    WIDTH = 2.0

    def __init__(self, x, y, speed=None, object_type=None, name=""):
        self.position = (x, y)
        self.heading_theta = 0.0
        self.name = name
        if speed is not None:
            self.speed = speed
        if object_type is not None:
            self.object_type = object_type


class DummyMetaDriveEngine:
    def __init__(self, vehicles):
        self.vehicles = vehicles


class DummyMetaDriveEnv:
    def __init__(self, ego, objects):
        self.vehicle = ego
        self.engine = DummyMetaDriveEngine([ego] + objects)


class StaticRSSFilterTest(unittest.TestCase):
    def setUp(self):
        self.filter = StaticRSSFilter(StaticRSSConfig())

    def test_static_rss_default_clearance_guard_disabled(self):
        self.assertFalse(self.filter.config.enable_predictive_clearance_guard)
        self.assertTrue(self.filter.config.preserve_steer_on_stop)
        self.assertFalse(self.filter.config.fallback_to_brake)
        self.assertTrue(self.filter.config.enable_recovery_mode)
        self.assertEqual(self.filter.config.intervention_margin_threshold, 0.0)

    def test_no_obstacle_returns_nominal_action(self):
        u_nom = [1.0, 0.1]
        u_safe, info = self.filter.filter_action(make_state(with_obstacle=False), u_nom)

        self.assertEqual(u_safe, u_nom)
        self.assertEqual(info["mode"], "normal")
        self.assertFalse(info["obstacle_detected"])

    def test_metadrive_action_adapter_uses_steer_throttle_brake_order(self):
        control = {"acc": -5.0, "steer": 0.25}
        env_action = self.filter.to_env_action(control, "metadrive")
        components = self.filter.env_action_components(env_action, "metadrive")
        brake_state = self.filter.action_brake_state(control, "metadrive")

        self.assertEqual(env_action, [0.25, -1.0])
        self.assertEqual(components["steer"], 0.25)
        self.assertEqual(components["throttle_brake"], -1.0)
        self.assertEqual(self.filter.from_policy_action(env_action, "metadrive"), control)
        self.assertTrue(brake_state["selected_acc_maps_to_brake"])
        self.assertEqual(
            brake_state["action_order_detected"],
            "env:[steer, throttle_brake]; internal/control:[acc, steer]",
        )

    def test_prediction_model_negative_acc_reduces_speed(self):
        state = make_state(with_obstacle=False)
        state["ego"]["speed"] = 6.0

        next_state = self.filter._simulate_next_state(state, {"acc": -1.0, "steer": 0.0})
        self.assertLess(next_state["ego"]["speed"], state["ego"]["speed"])
        self.assertEqual(next_state["ego"]["_prediction_action_order"], "internal/control:[acc, steer]")

    def test_obstacle_ahead_no_bypass_stops(self):
        u_safe, info = self.filter.filter_action(
            make_state(with_obstacle=True, left_available=False, right_available=False),
            [1.0, 0.0],
        )

        self.assertEqual(info["mode"], "stop")
        self.assertLessEqual(u_safe[0], 0.0)
        self.assertFalse(info["left_feasible"])
        self.assertFalse(info["right_feasible"])

    def test_static_stop_unsafe_uses_recovery_mode(self):
        state = make_state(with_obstacle=False, left_available=False, right_available=False)
        state["static_obstacles"] = [
            {
                "x": 10.0,
                "y": 0.0,
                "length": 4.5,
                "width": 2.0,
                "heading": 0.0,
                "lane_id": "center",
            }
        ]

        u_safe, info = self.filter.filter_action(state, [1.0, 0.0])
        margins = info["selected"]["margins"]

        self.assertEqual(info["mode"], "stop")
        self.assertLess(info["rss_margin"], 0.0)
        self.assertLessEqual(u_safe[0], 0.0)
        self.assertTrue(margins["recovery_mode_used"])

    def test_obstacle_ahead_left_bypass_feasible(self):
        u_safe, info = self.filter.filter_action(
            make_state(with_obstacle=True, left_available=True, right_available=False),
            [1.0, 0.0],
        )

        self.assertIn(info["mode"], {"stop", "left_bypass"})
        self.assertTrue(info["left_feasible"])
        self.assertFalse(info["right_feasible"])
        if info["mode"] == "left_bypass":
            self.assertGreater(u_safe[1], 0.0)

    def test_margin_gate_still_allows_preemptive_bypass(self):
        gated_filter = StaticRSSFilter(
            StaticRSSConfig(
                enable_bypass=True,
                enforce_intervention_margin=True,
                intervention_margin_threshold=0.0,
            )
        )

        u_safe, info = gated_filter.filter_action(
            make_state(with_obstacle=True, left_available=True, right_available=False),
            [1.0, 0.0],
        )

        self.assertEqual(info["mode"], "left_bypass")
        self.assertGreater(u_safe[1], 0.0)
        self.assertTrue(info["rss_margin_gate_active"])
        self.assertTrue(info["preemptive_bypass_allowed"])

    def test_margin_gate_without_bypass_returns_normal_until_rss_violation(self):
        gated_filter = StaticRSSFilter(
            StaticRSSConfig(
                enable_bypass=False,
                enforce_intervention_margin=True,
                intervention_margin_threshold=0.0,
            )
        )
        u_nom = [1.0, 0.0]

        u_safe, info = gated_filter.filter_action(
            make_state(with_obstacle=True, left_available=True, right_available=False),
            u_nom,
        )

        self.assertEqual(u_safe, u_nom)
        self.assertEqual(info["mode"], "normal")
        self.assertEqual(info["reason"], "rss_margin_positive_no_intervention")

    def test_dynamic_front_vehicle_unsafe_stops(self):
        dynamic_filter = StaticRSSFilter(StaticRSSConfig(intervention_margin_threshold=1.0))
        state = make_state(with_obstacle=False, left_available=True, right_available=False)
        state["vehicles"] = [
            {
                "x": 20.0,
                "y": 0.0,
                "heading": 0.0,
                "speed": 1.0,
                "length": 4.5,
                "width": 2.0,
                "lane_id": "center",
            }
        ]

        u_safe, info = dynamic_filter.filter_action(state, [1.0, 0.0])

        self.assertEqual(info["mode"], "dynamic_stop")
        self.assertLessEqual(u_safe[0], 0.0)
        self.assertTrue(info["dynamic_vehicle_detected"])

    def test_dynamic_front_vehicle_unsafe_uses_recovery_mode(self):
        state = make_state(with_obstacle=False, left_available=True, right_available=False)
        state["vehicles"] = [
            {
                "x": 8.0,
                "y": 0.0,
                "heading": 0.0,
                "speed": 1.0,
                "length": 4.5,
                "width": 2.0,
                "lane_id": "center",
            }
        ]

        u_safe, info = self.filter.filter_action(state, [1.0, 0.0])
        margins = info["selected"]["margins"]

        self.assertEqual(info["mode"], "dynamic_stop")
        self.assertLess(info["rss_margin"], 0.0)
        self.assertLessEqual(u_safe[0], 0.0)
        self.assertTrue(margins["recovery_mode_used"])
        self.assertGreaterEqual(margins["margin_improvement"], self.filter.config.recovery_margin_improvement)

    def test_dynamic_front_vehicle_unsafe_without_recovery_falls_back_to_nominal(self):
        no_recovery_filter = StaticRSSFilter(StaticRSSConfig(enable_recovery_mode=False))
        state = make_state(with_obstacle=False, left_available=True, right_available=False)
        state["vehicles"] = [
            {
                "x": 8.0,
                "y": 0.0,
                "heading": 0.0,
                "speed": 1.0,
                "length": 4.5,
                "width": 2.0,
                "lane_id": "center",
            }
        ]

        u_safe, info = no_recovery_filter.filter_action(state, [1.0, 0.0])

        self.assertEqual(info["mode"], "fallback_no_safe_candidate")
        self.assertEqual(info["reason"], "no_safe_candidate_keep_nominal")
        self.assertEqual(u_safe, [1.0, 0.0])

    def test_side_clearance_risk_triggers_filter(self):
        clearance_filter = StaticRSSFilter(StaticRSSConfig(enable_predictive_clearance_guard=True))
        state = make_state(with_obstacle=False, left_available=True, right_available=False)
        state["static_obstacles"] = [
            {
                "x": 0.0,
                "y": 2.5,
                "heading": 0.0,
                "speed": 0.0,
                "length": 4.5,
                "width": 2.0,
                "lane_id": "shoulder_object",
            }
        ]

        u_safe, info = clearance_filter.filter_action(state, [1.0, 0.0])

        self.assertIn(info["mode"], {"clearance_stop", "fallback_no_safe_candidate"})
        self.assertEqual(info["mode"], "clearance_stop")
        self.assertLessEqual(u_safe[0], 0.0)
        self.assertTrue(info["selected"]["margins"]["recovery_mode_used"])

    def test_no_safe_candidate_can_still_fallback_to_brake_when_enabled(self):
        brake_filter = StaticRSSFilter(
            StaticRSSConfig(
                enable_predictive_clearance_guard=True,
                fallback_to_brake=True,
            )
        )
        state = make_state(with_obstacle=False, left_available=True, right_available=False)
        state["static_obstacles"] = [
            {
                "x": 0.0,
                "y": 2.5,
                "heading": 0.0,
                "speed": 0.0,
                "length": 4.5,
                "width": 2.0,
                "lane_id": "shoulder_object",
            }
        ]

        u_safe, info = brake_filter.filter_action(state, [1.0, 0.0])

        self.assertIn(info["mode"], {"clearance_stop", "fallback_stop"})
        if info["mode"] == "fallback_stop":
            self.assertLessEqual(u_safe[0], 0.0)

    def test_lidar_fallback_adds_front_obstacle(self):
        state = make_state(with_obstacle=False, left_available=True, right_available=False)
        obs = [0.0] * 20 + [1.0] * 240
        obs[-240] = 0.1

        augmented = self.filter.augment_state_from_observation(state, obs)

        self.assertEqual(len(augmented["static_obstacles"]), 1)
        self.assertTrue(augmented["adapter_debug"]["observation_lidar_fallback_used"])

    def test_metadrive_adapter_ignores_unknown_object_types(self):
        ego = DummyMetaDriveObject(0.0, 0.0, speed=8.0, object_type="vehicle", name="ego")
        stopped_vehicle = DummyMetaDriveObject(10.0, 0.0, speed=0.0, object_type="vehicle", name="stopped")
        moving_vehicle = DummyMetaDriveObject(20.0, 0.0, speed=4.0, object_type="vehicle", name="moving")
        traffic_object = DummyMetaDriveObject(30.0, 0.0, object_type="traffic_object", name="traffic_object")
        unknown_node = DummyMetaDriveObject(40.0, 0.0, object_type="road_node", name="road_node")
        env = DummyMetaDriveEnv(
            ego,
            [stopped_vehicle, moving_vehicle, traffic_object, unknown_node],
        )

        state = self.filter.parse_state_from_metadrive(env)

        self.assertEqual(len(state["static_obstacles"]), 2)
        self.assertEqual(len(state["vehicles"]), 1)
        self.assertEqual(state["adapter_debug"]["ignored_count"], 1)
        self.assertEqual(state["adapter_debug"]["ignored_types"], {"road_node": 1})
        self.assertEqual(
            sorted(obj["object_id"] for obj in state["static_obstacles"]),
            ["stopped", "traffic_object"],
        )
        self.assertEqual(state["vehicles"][0]["object_id"], "moving")

    def test_rss_cbf_filter_uses_formal_modes_and_preserves_steer(self):
        rss_cbf_filter = RSSCBFFilter(RSSCBFConfig(enable_2d_rss_cbf=False))
        state = make_state(with_obstacle=False, left_available=True, right_available=False)
        state["vehicles"] = [
            {
                "x": 8.0,
                "y": 0.0,
                "heading": 0.0,
                "speed": 1.0,
                "length": 4.5,
                "width": 2.0,
                "lane_id": "center",
            }
        ]

        u_safe, info = rss_cbf_filter.filter_action(state, [1.0, 0.3])

        self.assertIn(info["mode"], {"rss_cbf_recovery", "fallback_no_safe_candidate"})
        self.assertEqual(u_safe[1], 0.3)
        self.assertTrue(info["dynamic_vehicle_detected"])
        for key in [
            "rss_margin",
            "action_delta",
            "acc_nominal",
            "acc_safe",
            "steer_nominal",
            "steer_safe",
            "acc_delta",
            "steer_delta",
            "reason",
        ]:
            self.assertIn(key, info)

    def test_2d_rss_cbf_same_lane_front_vehicle_is_longitudinally_unsafe(self):
        rss_cbf_filter = RSSCBFFilter(RSSCBFConfig(enable_2d_rss_cbf=True))
        state = make_state(with_obstacle=False)
        front_vehicle = {
            "x": 8.0,
            "y": 0.0,
            "heading": 0.0,
            "speed": 1.0,
            "length": 4.5,
            "width": 2.0,
            "lane_id": "center",
            "object_type": "vehicle",
            "object_id": "same_lane_front",
        }

        margin = rss_cbf_filter.compute_2d_rss_cbf_margin(state, front_vehicle, "dynamic")

        self.assertEqual(margin["relation"], "front")
        self.assertLess(margin["h_2d"], 0.0)
        self.assertEqual(margin["lat_clearance"], 0.0)

    def test_2d_rss_cbf_adjacent_parallel_vehicle_is_laterally_safe(self):
        rss_cbf_filter = RSSCBFFilter(
            RSSCBFConfig(enable_2d_rss_cbf=True, rss_2d_lateral_margin=0.5)
        )
        state = make_state(with_obstacle=False)
        adjacent_vehicle = {
            "x": 0.0,
            "y": 3.7,
            "heading": 0.0,
            "speed": 8.0,
            "length": 4.5,
            "width": 2.0,
            "lane_id": "left",
            "object_type": "vehicle",
            "object_id": "adjacent_parallel",
        }

        margin = rss_cbf_filter.compute_2d_rss_cbf_margin(state, adjacent_vehicle, "dynamic")

        self.assertEqual(margin["relation"], "side")
        self.assertGreater(margin["lat_clearance"], margin["d_l_safe"])
        self.assertGreaterEqual(margin["h_2d"], 0.0)

    def test_2d_rss_cbf_diagonal_vehicle_uses_both_axes(self):
        rss_cbf_filter = RSSCBFFilter(
            RSSCBFConfig(enable_2d_rss_cbf=True, rss_2d_lateral_margin=0.5, rss_2d_power=4.0)
        )
        state = make_state(with_obstacle=False)
        diagonal_vehicle = {
            "x": 14.0,
            "y": 2.25,
            "heading": 0.0,
            "speed": 3.0,
            "length": 4.5,
            "width": 2.0,
            "lane_id": "left",
            "object_type": "vehicle",
            "object_id": "diagonal",
        }

        margin = rss_cbf_filter.compute_2d_rss_cbf_margin(state, diagonal_vehicle, "dynamic")
        expected = (
            (margin["long_clearance"] / max(margin["d_s_safe"], rss_cbf_filter.config.rss_2d_eps)) ** 4
            + (margin["lat_clearance"] / max(margin["d_l_safe"], rss_cbf_filter.config.rss_2d_eps)) ** 4
            - 1.0
        )

        self.assertGreater(margin["long_clearance"], 0.0)
        self.assertGreater(margin["lat_clearance"], 0.0)
        self.assertAlmostEqual(margin["h_2d"], expected)

    def test_2d_rss_cbf_road_boundary_rejects_otherwise_safe_nominal(self):
        rss_cbf_filter = RSSCBFFilter(RSSCBFConfig(enable_2d_rss_cbf=True))
        state = make_state(with_obstacle=False)
        state["ego"]["dist_to_left_side"] = 0.2
        state["ego"]["dist_to_right_side"] = 5.0

        u_safe, info = rss_cbf_filter.filter_action(state, [1.0, 0.0])

        self.assertFalse(info["road_boundary_safe"])
        self.assertEqual(info["safety_function_mode"], "unified_safety_object_2d_cbf")
        self.assertNotEqual(info["mode"], "normal")
        self.assertLess(info["nominal_H"], 0.0)
        self.assertEqual(info["worst_object_type"], "road_boundary")
        self.assertLess(u_safe[1], 0.0)
        self.assertGreaterEqual(u_safe[0], rss_cbf_filter.config.rss_2d_min_speed_preserve_acc)
        self.assertGreater(info["selected_final_boundary_margin"], info["road_boundary_margin_current"])

    def test_2d_rss_cbf_near_boundary_searches_steer_before_braking(self):
        rss_cbf_filter = RSSCBFFilter(
            RSSCBFConfig(enable_2d_rss_cbf=True, rss_2d_boundary_margin_threshold=0.5)
        )
        state = make_state(with_obstacle=False)
        state["ego"]["dist_to_left_side"] = 0.6
        state["ego"]["dist_to_right_side"] = 5.0

        u_safe, info = rss_cbf_filter.filter_action(state, [1.0, 0.0])

        self.assertLess(info["nominal_H"], 0.0)
        self.assertEqual(info["reason"], "selected_least_unsafe")
        self.assertEqual(info["worst_object_type"], "road_boundary")
        self.assertLess(u_safe[1], 0.0)
        self.assertGreaterEqual(u_safe[0], rss_cbf_filter.config.rss_2d_min_speed_preserve_acc)
        self.assertGreater(info["selected_final_boundary_margin"], info["road_boundary_margin_current"])

    def test_2d_rss_cbf_nominal_lazy_safe_skips_candidate_search(self):
        rss_cbf_filter = RSSCBFFilter(RSSCBFConfig(enable_2d_rss_cbf=True))
        state = make_state(with_obstacle=False)
        state["ego"]["dist_to_left_side"] = 5.0
        state["ego"]["dist_to_right_side"] = 5.0

        u_safe, info = rss_cbf_filter.filter_action(state, [0.2, 0.05])

        self.assertEqual(u_safe, [0.2, 0.05])
        self.assertEqual(info["mode"], "normal")
        self.assertEqual(info["reason"], "nominal_lazy_safe")
        self.assertEqual(info["candidate_count"], 0)
        self.assertEqual(info["safety_object_count_total"], 0)
        self.assertEqual(info["safety_object_count_local"], 0)
        self.assertEqual(info["horizon_steps"], rss_cbf_filter.config.prediction_horizon_steps)
        self.assertIn("total_filter_time", info)

    def test_2d_rss_cbf_spatial_filter_skips_far_objects(self):
        rss_cbf_filter = RSSCBFFilter(
            RSSCBFConfig(
                enable_2d_rss_cbf=True,
                max_check_distance=20.0,
                broad_phase_radius=20.0,
            )
        )
        state = make_state(with_obstacle=False)
        state["ego"]["dist_to_left_side"] = 5.0
        state["ego"]["dist_to_right_side"] = 5.0
        state["static_obstacles"].append(
            {
                "x": 100.0,
                "y": 0.0,
                "length": 4.5,
                "width": 2.0,
                "heading": 0.0,
                "object_id": "far_static",
            }
        )

        _, info = rss_cbf_filter.filter_action(state, [0.2, 0.0])

        self.assertEqual(info["safety_object_count_total"], 1)
        self.assertEqual(info["safety_object_count_local"], 0)
        self.assertEqual(info["candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
