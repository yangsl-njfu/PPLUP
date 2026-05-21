"""Unit tests for the lightweight static RSS filter.

Run:
    python test_static_rss_filter.py
"""

import unittest

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


class StaticRSSFilterTest(unittest.TestCase):
    def setUp(self):
        self.filter = StaticRSSFilter(StaticRSSConfig())

    def test_no_obstacle_returns_nominal_action(self):
        u_nom = [1.0, 0.1]
        u_safe, info = self.filter.filter_action(make_state(with_obstacle=False), u_nom)

        self.assertEqual(u_safe, u_nom)
        self.assertEqual(info["mode"], "normal")
        self.assertFalse(info["obstacle_detected"])

    def test_obstacle_ahead_no_bypass_stops(self):
        u_safe, info = self.filter.filter_action(
            make_state(with_obstacle=True, left_available=False, right_available=False),
            [1.0, 0.0],
        )

        self.assertEqual(info["mode"], "stop")
        self.assertLessEqual(u_safe[0], 0.0)
        self.assertFalse(info["left_feasible"])
        self.assertFalse(info["right_feasible"])

    def test_obstacle_ahead_left_bypass_feasible(self):
        u_safe, info = self.filter.filter_action(
            make_state(with_obstacle=True, left_available=True, right_available=False),
            [1.0, 0.0],
        )

        self.assertEqual(info["mode"], "left_bypass")
        self.assertGreater(u_safe[1], 0.0)
        self.assertTrue(info["left_feasible"])
        self.assertFalse(info["right_feasible"])

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

    def test_side_clearance_risk_triggers_filter(self):
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

        u_safe, info = self.filter.filter_action(state, [1.0, 0.0])

        self.assertIn(info["mode"], {"clearance_stop", "fallback_stop"})
        self.assertLessEqual(u_safe[0], 0.0)

    def test_lidar_fallback_adds_front_obstacle(self):
        state = make_state(with_obstacle=False, left_available=True, right_available=False)
        obs = [0.0] * 20 + [1.0] * 240
        obs[-240] = 0.1

        augmented = self.filter.augment_state_from_observation(state, obs)

        self.assertEqual(len(augmented["static_obstacles"]), 1)
        self.assertTrue(augmented["adapter_debug"]["observation_lidar_fallback_used"])


if __name__ == "__main__":
    unittest.main()
