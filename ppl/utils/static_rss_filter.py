"""Lightweight progress-aware RSS filter for static obstacle stagnation.

This module implements a small, dependency-light prototype of
"Progress-Aware RSS Action Projection for Static Obstacle Stagnation".
It is intentionally independent from MetaDrive internals: the core filter
expects a generic dict state and returns a safe internal action [acc, steer].

The rollout model is a lightweight point-mass / heading-rate approximation.
It is meant for fast runtime filtering experiments and can be replaced by a
simulator-specific dynamics adapter later.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


State = Dict[str, Any]
Action = List[float]


@dataclass
class StaticRSSConfig:
    """Configuration for the static-obstacle RSS action projection filter."""

    rho: float = 0.5
    a_max: float = 2.0
    b_min: float = 4.0
    vehicle_length: float = 4.5
    vehicle_width: float = 2.0
    obstacle_margin: float = 0.8
    lane_margin: float = 0.3
    min_bypass_width: float = 2.8
    max_steer: float = 1.0
    min_acc: float = -5.0
    max_acc: float = 2.0
    target_bypass_speed: float = 4.0
    stagnation_speed_threshold: float = 0.5
    ray_steps: int = 40
    horizon_steps: int = 10
    dt: float = 0.1

    # Lightweight rollout and selection parameters.
    v_max: float = 30.0
    steer_gain: float = 0.5
    safe_steer_center: float = 0.4
    bypass_acc_gain: float = 0.5
    default_lane_width: float = 3.5
    small_tolerance: float = 1e-6
    progress_preference_margin: float = 0.2
    min_bypass_lateral_progress: float = 0.05
    min_bypass_steer: float = 0.05
    adjacent_vehicle_margin: float = 1.0
    metadrive_static_speed_threshold: float = 0.2
    metadrive_object_scan_radius: float = 80.0
    metadrive_assume_adjacent_lanes: bool = False
    enable_observation_lidar_fallback: bool = True
    metadrive_lidar_num_lasers: int = 240
    metadrive_lidar_max_distance: float = 50.0
    metadrive_lidar_front_sector_ratio: float = 0.08
    metadrive_lidar_distance_trigger_ratio: float = 0.35
    metadrive_lidar_proximity_trigger_ratio: float = 0.65
    enable_dynamic_front_vehicle_rss: bool = True
    enable_predictive_clearance_guard: bool = False
    enable_bypass: bool = True
    allow_bypass_before_rss_violation: bool = True
    fallback_to_brake: bool = False
    enable_recovery_mode: bool = True
    recovery_margin_improvement: float = 0.05
    recovery_allow_equal_margin: bool = True
    enforce_intervention_margin: bool = False
    intervention_margin_threshold: float = 0.0
    metadrive_steer_sign: float = 1.0
    preserve_steer_on_stop: bool = True


@dataclass
class StaticRSSCandidate:
    """Debuggable candidate action for one RSS-certified action set."""

    action: Action
    mode: str
    safe: bool
    progress_score: float
    intervention_cost: float
    margins: Dict[str, float] = field(default_factory=dict)
    projection_debug: Dict[str, Any] = field(default_factory=dict)
    feasible_debug: Dict[str, Any] = field(default_factory=dict)


class StaticRSSFilter:
    """Progress-aware RSS action projection for static obstacle stagnation.

    The filter treats the original policy action as an internal action
    ``[acc, steer]``. For simulators such as MetaDrive that use
    ``[steer, throttle_brake]``, use ``to_internal_action`` and
    ``from_internal_action`` at the integration boundary.
    """

    def __init__(self, config: Optional[StaticRSSConfig] = None):
        self.config = config or StaticRSSConfig()
        self._last_frenet_reference_lane: Any = None

    def filter_action(self, state: State, u_nom: Sequence[float]) -> Tuple[Action, Dict[str, Any]]:
        """Filter a nominal policy action and return ``(u_safe, info)``.

        Args:
            state: Generic driving state dict. See the project demo for the
                expected minimal schema.
            u_nom: Nominal internal action ``[acc, steer]``.

        Returns:
            A safe internal action and debug info containing selected mode,
            obstacle detection results, feasibility checks, margins, and
            candidate scores.
        """
        u_original = self._clip_action(u_nom)
        obstacle_info = self.detect_static_obstacle_ahead(state)
        dynamic_info = self.detect_dynamic_vehicle_ahead(state)
        clearance_info = None
        if self.config.enable_predictive_clearance_guard:
            clearance_info = self.detect_predictive_clearance_risk(state, u_original)

        if obstacle_info is None and dynamic_info is None and clearance_info is None:
            return u_original, self._with_action_debug({
                "mode": "normal",
                "obstacle_detected": False,
                "dynamic_vehicle_detected": False,
                "left_feasible": False,
                "right_feasible": False,
                "state_debug": self._state_debug(state),
                "candidates": [],
            }, u_original, u_original)

        if self.config.enable_dynamic_front_vehicle_rss and dynamic_info is not None:
            vehicle, d_front = dynamic_info
            d_dynamic = self.compute_dynamic_front_distance(
                self._ego_speed(state),
                float(vehicle.get("speed", 0.0)),
            )
            dynamic_margin = d_front - d_dynamic
            static_margin = math.inf
            if obstacle_info is not None:
                static_obstacle, static_d_obs = obstacle_info
                static_margin = static_d_obs - self.compute_brake_distance(self._ego_speed(state))
            if dynamic_margin <= self.config.intervention_margin_threshold and dynamic_margin <= static_margin:
                return self._filter_dynamic_front_vehicle(
                    state, vehicle, d_front, d_dynamic, dynamic_margin, u_original
                )

        if clearance_info is not None and obstacle_info is None:
            return self._filter_predictive_clearance_risk(state, clearance_info, u_original)

        if obstacle_info is None:
            return u_original, self._with_action_debug({
                "mode": "normal",
                "obstacle_detected": False,
                "dynamic_vehicle_detected": True,
                "state_debug": self._state_debug(state),
                "candidates": [],
            }, u_original, u_original)

        obstacle, d_obs = obstacle_info
        ego_speed = self._ego_speed(state)
        d_brake = self.compute_brake_distance(ego_speed)
        current_rss_margin = d_obs - d_brake
        rss_margin_gate_active = (
            self.config.enforce_intervention_margin
            and current_rss_margin > self.config.intervention_margin_threshold
        )
        preemptive_bypass_allowed = (
            self.config.enable_bypass
            and self.config.allow_bypass_before_rss_violation
        )

        if rss_margin_gate_active and not preemptive_bypass_allowed:
            if clearance_info is not None:
                return self._filter_predictive_clearance_risk(state, clearance_info, u_original)
            return u_original, self._with_action_debug({
                "mode": "normal",
                "obstacle_detected": True,
                "d_obs": d_obs,
                "d_brake": d_brake,
                "rss_margin": current_rss_margin,
                "rss_margin_gate_active": True,
                "preemptive_bypass_allowed": False,
                "left_feasible": False,
                "right_feasible": False,
                "state_debug": self._state_debug(state),
                "blocking_object": self._object_debug(obstacle, state),
                "candidates": [],
                "reason": "rss_margin_positive_no_intervention",
            }, u_original, u_original)

        candidates: List[StaticRSSCandidate] = []

        if not rss_margin_gate_active:
            if self.config.preserve_steer_on_stop:
                u_stop_nom = list(u_original)
                stop_center = self._clip_action([self.config.min_acc, u_original[1]])
            else:
                u_stop_nom = self.generate_stop_nominal_action(state)
                stop_center = None
            u_stop_proj, stop_project_debug = self.ray_project_action(
                state, obstacle, u_stop_nom, "stop", center_override=stop_center
            )
            stop_safe, stop_margins = self.is_action_safe_for_mode(state, obstacle, u_stop_proj, "stop")
            candidates.append(
                self._make_candidate(
                    action=u_stop_proj,
                    mode="stop",
                    safe=stop_safe,
                    margins=stop_margins,
                    projection_debug=stop_project_debug,
                    feasible_debug={"feasible": True},
                    u_original=u_original,
                    state=state,
                )
            )

        left_feasible, left_debug = False, {"feasible": False, "reason": "bypass_disabled"}
        if self.config.enable_bypass:
            left_feasible, left_debug = self.check_bypass_feasibility(state, obstacle, "left")
        if left_feasible:
            u_left_nom = self.generate_bypass_nominal_action(state, "left")
            u_left_proj, left_project_debug = self.ray_project_action(
                state, obstacle, u_left_nom, "left_bypass"
            )
            left_safe, left_margins = self.is_action_safe_for_mode(
                state, obstacle, u_left_proj, "left_bypass"
            )
            candidates.append(
                self._make_candidate(
                    action=u_left_proj,
                    mode="left_bypass",
                    safe=left_safe,
                    margins=left_margins,
                    projection_debug=left_project_debug,
                    feasible_debug=left_debug,
                    u_original=u_original,
                    state=state,
                )
            )

        right_feasible, right_debug = False, {"feasible": False, "reason": "bypass_disabled"}
        if self.config.enable_bypass:
            right_feasible, right_debug = self.check_bypass_feasibility(state, obstacle, "right")
        if right_feasible:
            u_right_nom = self.generate_bypass_nominal_action(state, "right")
            u_right_proj, right_project_debug = self.ray_project_action(
                state, obstacle, u_right_nom, "right_bypass"
            )
            right_safe, right_margins = self.is_action_safe_for_mode(
                state, obstacle, u_right_proj, "right_bypass"
            )
            candidates.append(
                self._make_candidate(
                    action=u_right_proj,
                    mode="right_bypass",
                    safe=right_safe,
                    margins=right_margins,
                    projection_debug=right_project_debug,
                    feasible_debug=right_debug,
                    u_original=u_original,
                    state=state,
                )
            )

        if rss_margin_gate_active:
            safe_bypass_candidates = [
                candidate
                for candidate in candidates
                if candidate.safe and candidate.mode in {"left_bypass", "right_bypass"}
            ]
            if not safe_bypass_candidates:
                if clearance_info is not None:
                    return self._filter_predictive_clearance_risk(state, clearance_info, u_original)
                candidate_debug = [asdict(candidate) for candidate in candidates]
                return u_original, self._with_action_debug({
                    "mode": "normal",
                    "obstacle_detected": True,
                    "dynamic_vehicle_detected": dynamic_info is not None,
                    "d_obs": d_obs,
                    "d_brake": d_brake,
                    "rss_margin": current_rss_margin,
                    "rss_margin_gate_active": True,
                    "preemptive_bypass_allowed": True,
                    "left_feasible": left_feasible,
                    "right_feasible": right_feasible,
                    "state_debug": self._state_debug(state),
                    "blocking_object": self._object_debug(obstacle, state),
                    "candidates": candidate_debug,
                    "reason": "rss_margin_positive_no_safe_bypass",
                }, u_original, u_original)

        u_safe, selected_info = self.select_action(candidates, u_original)
        candidate_debug = [asdict(candidate) for candidate in candidates]

        return u_safe, self._with_action_debug({
            "mode": selected_info["mode"],
            "obstacle_detected": True,
            "dynamic_vehicle_detected": dynamic_info is not None,
            "d_obs": d_obs,
            "d_brake": d_brake,
            "rss_margin": current_rss_margin,
            "rss_margin_gate_active": rss_margin_gate_active,
            "preemptive_bypass_allowed": preemptive_bypass_allowed,
            "left_feasible": left_feasible,
            "right_feasible": right_feasible,
            "state_debug": self._state_debug(state),
            "blocking_object": self._object_debug(obstacle, state),
            "selected_score": selected_info.get("selected_score"),
            "projection_debug": selected_info.get("projection_debug", {}),
            "selected": selected_info,
            "candidates": candidate_debug,
        }, u_original, u_safe)

    def detect_static_obstacle_ahead(self, state: State) -> Optional[Tuple[Dict[str, Any], float]]:
        """Return the closest static obstacle blocking the current lane/path.

        The lightweight detector projects obstacle centers into ego coordinates:
        longitudinal distance must be positive and lateral overlap must intersect
        the current lane corridor.
        """
        ego = self._ego(state)
        lane_width = self._current_lane_width(state)
        closest: Optional[Tuple[Dict[str, Any], float, float]] = None

        for obstacle in state.get("static_obstacles", []):
            if self._lane_ids_differ(ego, obstacle):
                continue

            longitudinal, lateral = self._relative_position(ego, obstacle)
            obstacle_width = self._object_width(obstacle, self.config.vehicle_width)
            if longitudinal <= 0.0:
                continue
            lateral_limit = lane_width / 2.0 + obstacle_width / 2.0
            if abs(lateral) > lateral_limit:
                continue

            d_obs = self._distance_to_obstacle_front(state, obstacle)
            if closest is None or d_obs < closest[1]:
                closest = (obstacle, d_obs, longitudinal)

        if closest is None:
            return None
        return closest[0], closest[1]

    def detect_dynamic_vehicle_ahead(self, state: State) -> Optional[Tuple[Dict[str, Any], float]]:
        """Return the closest moving front vehicle blocking the current lane/path."""
        ego = self._ego(state)
        lane_width = self._current_lane_width(state)
        closest: Optional[Tuple[Dict[str, Any], float, float]] = None

        for vehicle in state.get("vehicles", []):
            if self._lane_ids_differ(ego, vehicle):
                continue

            longitudinal, lateral = self._relative_position(ego, vehicle)
            vehicle_width = self._object_width(vehicle, self.config.vehicle_width)
            if longitudinal <= 0.0:
                continue
            lateral_limit = lane_width / 2.0 + vehicle_width / 2.0
            if abs(lateral) > lateral_limit:
                continue

            d_front = self._distance_to_obstacle_front(state, vehicle)
            if closest is None or d_front < closest[1]:
                closest = (vehicle, d_front, longitudinal)

        if closest is None:
            return None
        return closest[0], closest[1]

    def detect_predictive_clearance_risk(
        self, state: State, action: Sequence[float]
    ) -> Optional[Dict[str, Any]]:
        """Predict whether nominal action will violate clearance to any object.

        This guard catches risks outside the narrow "front object" RSS detector,
        such as side contacts, curved-road contacts, or already-sticky contact
        states. The clearance margin already includes ``obstacle_margin``.
        """
        objects = list(state.get("static_obstacles", [])) + list(state.get("vehicles", []))
        if not objects:
            return None

        min_margin = math.inf
        riskiest_object = None
        riskiest_step = 0
        rollout_states = [state] + self._rollout_states(state, action, self.config.horizon_steps)
        for step, rollout_state in enumerate(rollout_states):
            for obj in objects:
                margin = self._obstacle_clearance_margin(rollout_state, obj)
                if margin < min_margin:
                    min_margin = margin
                    riskiest_object = obj
                    riskiest_step = step

        if riskiest_object is None or min_margin >= -self.config.small_tolerance:
            return None

        return {
            "object": riskiest_object,
            "clearance_margin": min_margin,
            "risk_step": riskiest_step,
        }

    def compute_brake_distance(self, speed: float) -> float:
        """Compute RSS static-obstacle braking distance with a vehicle margin."""
        v = max(0.0, float(speed))
        cfg = self.config
        margin = cfg.vehicle_length / 2.0 + cfg.obstacle_margin
        response_speed = v + cfg.a_max * cfg.rho
        return (
            v * cfg.rho
            + 0.5 * cfg.a_max * cfg.rho ** 2
            + response_speed ** 2 / (2.0 * cfg.b_min)
            + margin
        )

    def compute_rss_distance(
        self,
        speed: float,
        front_speed: float = 0.0,
        margin: Optional[float] = None,
    ) -> float:
        """Compute the forward RSS safety distance used by runtime assurance."""
        v = max(0.0, float(speed))
        cfg = self.config
        if margin is None:
            margin = cfg.vehicle_length / 2.0 + cfg.obstacle_margin
        response_speed = v + cfg.a_max * cfg.rho
        distance = (
            v * cfg.rho
            + 0.5 * cfg.a_max * cfg.rho ** 2
            + response_speed ** 2 / (2.0 * cfg.b_min)
            + float(margin)
        )
        return max(0.0, distance)

    def compute_dynamic_front_distance(self, ego_speed: float, front_speed: float) -> float:
        """Compute RSS front-distance for a moving lead vehicle."""
        return self._rss_front_distance(max(0.0, ego_speed), max(0.0, front_speed))

    def check_stop_safety(
        self, state: State, obstacle: Dict[str, Any], action: Sequence[float]
    ) -> Tuple[bool, float]:
        """Check stop-mode safety and return ``(safe, min_h_stop)``."""
        safe, margins = self._check_stop_horizon(state, obstacle, action)
        return safe, margins["h_stop_min"]

    def check_bypass_feasibility(
        self, state: State, obstacle: Dict[str, Any], direction: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if the left or right side is RSS-feasible for bypass.

        This is a passability and adjacent-vehicle check. The detailed action
        safety is handled by ``is_action_safe_for_mode``.
        """
        self._validate_direction(direction)
        lanes = state.get("lanes", {}) or {}
        target_lane = lanes.get(direction)
        debug: Dict[str, Any] = {
            "direction": direction,
            "feasible": False,
            "reason": "",
        }

        if not target_lane:
            debug["reason"] = "missing_target_lane"
            return False, debug
        if target_lane.get("available", True) is False or target_lane.get("drivable", True) is False:
            debug["reason"] = "target_lane_unavailable"
            return False, debug

        available_width = self._lane_available_width(state, target_lane)
        required_width = max(
            self.config.min_bypass_width,
            self.config.vehicle_width + 2.0 * self.config.lane_margin,
        )
        width_margin = available_width - required_width
        front_margin, rear_margin, vehicle_debug = self._adjacent_vehicle_rss_margins(state, direction)

        debug.update(
            {
                "available_width": available_width,
                "required_width": required_width,
                "width_margin": width_margin,
                "front_rss_margin": front_margin,
                "rear_rss_margin": rear_margin,
                "vehicles": vehicle_debug,
            }
        )

        feasible = (
            width_margin >= -self.config.small_tolerance
            and front_margin >= -self.config.small_tolerance
            and rear_margin >= -self.config.small_tolerance
        )
        debug["feasible"] = feasible
        if not feasible and not debug["reason"]:
            debug["reason"] = "width_or_adjacent_rss_violation"
        return feasible, debug

    def generate_stop_nominal_action(self, state: State) -> Action:
        """Generate the nominal RSS stop action in internal ``[acc, steer]`` form."""
        return self._clip_action([self.config.min_acc, 0.0])

    def generate_stop_nominal_action_from_original(self, u_original: Sequence[float]) -> Action:
        """Generate stop nominal action, optionally preserving original steering."""
        if self.config.preserve_steer_on_stop:
            return self._clip_action([self.config.min_acc, u_original[1]])
        return self._clip_action([self.config.min_acc, 0.0])

    def generate_bypass_nominal_action(self, state: State, direction: str) -> Action:
        """Generate a simple progress-preserving bypass nominal action."""
        self._validate_direction(direction)
        sign = self._direction_sign(direction)
        speed = self._ego_speed(state)
        acc = self.config.bypass_acc_gain * (self.config.target_bypass_speed - speed)
        steer = sign * self.config.safe_steer_center
        return self._clip_action([acc, steer])

    def is_action_safe_for_mode(
        self, state: State, obstacle: Dict[str, Any], action: Sequence[float], mode: str
    ) -> Tuple[bool, Dict[str, float]]:
        """Check if an action is safe for a named action-set mode.

        Modes:
            ``stop``: RSS stop horizon condition.
            ``left_bypass`` / ``right_bypass``: obstacle clearance, lane bounds,
            adjacent-lane RSS, and progress checks under a lightweight rollout.
        """
        action = self._clip_action(action)
        if mode == "stop":
            return self._check_stop_horizon(state, obstacle, action)
        if mode == "dynamic_stop":
            return self._check_dynamic_front_horizon(state, obstacle, action)
        if mode == "clearance_stop":
            return self._check_clearance_horizon(state, obstacle, action)

        if mode not in {"left_bypass", "right_bypass"}:
            raise ValueError("Unsupported mode: {}".format(mode))

        direction = "left" if mode == "left_bypass" else "right"
        feasible, feasible_debug = self.check_bypass_feasibility(state, obstacle, direction)
        if not feasible:
            return False, {
                "feasibility": -1.0,
                "width_margin": float(feasible_debug.get("width_margin", -math.inf)),
                "front_rss_margin": float(feasible_debug.get("front_rss_margin", -math.inf)),
                "rear_rss_margin": float(feasible_debug.get("rear_rss_margin", -math.inf)),
            }

        rollout = self._rollout_states(state, action, self.config.horizon_steps)
        initial_ego = self._ego(state)
        min_clearance = math.inf
        min_boundary = math.inf
        min_front_rss = math.inf
        min_rear_rss = math.inf

        for rollout_state in rollout:
            min_clearance = min(min_clearance, self._obstacle_clearance_margin(rollout_state, obstacle))
            min_boundary = min(
                min_boundary,
                self._lane_boundary_margin(initial_ego, rollout_state, direction, state),
            )
            front_margin, rear_margin, _ = self._adjacent_vehicle_rss_margins(rollout_state, direction)
            min_front_rss = min(min_front_rss, front_margin)
            min_rear_rss = min(min_rear_rss, rear_margin)

        final_state = rollout[-1] if rollout else state
        forward_progress = self._longitudinal_progress(initial_ego, self._ego(final_state))
        lateral_progress = self._direction_sign(direction) * self._lateral_progress(initial_ego, self._ego(final_state))
        progress_required = self.config.stagnation_speed_threshold * self.config.dt * self.config.horizon_steps
        progress_margin = forward_progress - progress_required
        lateral_progress_margin = lateral_progress - self.config.min_bypass_lateral_progress
        steer_margin = self._direction_sign(direction) * action[1] - self.config.min_bypass_steer

        margins = {
            "obstacle_clearance_margin": min_clearance,
            "lane_boundary_margin": min_boundary,
            "front_rss_margin": min_front_rss,
            "rear_rss_margin": min_rear_rss,
            "progress_margin": progress_margin,
            "lateral_progress_margin": lateral_progress_margin,
            "steer_direction_margin": steer_margin,
        }
        safe = min(margins.values()) >= -self.config.small_tolerance
        return safe, margins

    def ray_project_action(
        self,
        state: State,
        obstacle: Dict[str, Any],
        u_nom: Sequence[float],
        mode: str,
        center_override: Optional[Action] = None,
    ) -> Tuple[Action, Dict[str, Any]]:
        """Project an action to a mode-specific safe set via ray search.

        The ray starts at a conservative safe center ``c`` and points toward
        ``u_nom``. The largest safe lambda in ``[0, 1]`` is selected.

        When ``center_override`` is provided, it replaces the default
        ``_safe_center_for_mode(mode)``. This enables 1-D projection for
        stop modes when ``preserve_steer_on_stop`` is active: the center
        and nominal share the same steer, so only acc is searched.
        """
        u_nom = self._clip_action(u_nom)
        center = center_override if center_override is not None else self._safe_center_for_mode(mode)
        tests = []

        nominal_safe, nominal_margins = self.is_action_safe_for_mode(state, obstacle, u_nom, mode)
        if nominal_safe:
            return u_nom, {
                "lambda": 1.0,
                "projection_failed": False,
                "center": center,
                "tested": [{"lambda": 1.0, "safe": True, "margins": nominal_margins}],
            }

        steps = max(1, int(self.config.ray_steps))
        for step in range(steps + 1):
            lam = 1.0 - float(step) / float(steps)
            action = self._clip_action(
                [
                    center[0] + lam * (u_nom[0] - center[0]),
                    center[1] + lam * (u_nom[1] - center[1]),
                ]
            )
            safe, margins = self.is_action_safe_for_mode(state, obstacle, action, mode)
            tests.append({"lambda": lam, "safe": safe, "margins": margins})
            if safe:
                return action, {
                    "lambda": lam,
                    "projection_failed": False,
                    "center": center,
                    "tested": tests,
                }

        center_safe, center_margins = self.is_action_safe_for_mode(state, obstacle, center, mode)
        tests.append({"lambda": 0.0, "safe": center_safe, "margins": center_margins})
        return center, {
            "lambda": 0.0,
            "projection_failed": True,
            "center": center,
            "tested": tests,
        }

    def select_action(
        self, candidates: Sequence[StaticRSSCandidate], u_original: Sequence[float]
    ) -> Tuple[Action, Dict[str, Any]]:
        """Select the final action by safety, progress, then intervention cost."""
        safe_candidates = [candidate for candidate in candidates if candidate.safe]
        if self.config.preserve_steer_on_stop:
            fallback = self._clip_action([self.config.min_acc, u_original[1]])
        else:
            fallback = self._clip_action([self.config.min_acc, 0.0])

        if not safe_candidates:
            if self.config.fallback_to_brake:
                return fallback, {
                    "mode": "fallback_stop",
                    "action": fallback,
                    "selected_score": None,
                    "projection_debug": {"reason": "no_safe_candidate"},
                }
            nominal = self._clip_action(u_original)
            return nominal, {
                "mode": "fallback_no_safe_candidate",
                "action": nominal,
                "selected_score": None,
                "reason": "no_safe_candidate_keep_nominal",
                "projection_debug": {"reason": "no_safe_candidate_keep_nominal"},
            }

        stop_candidates = [candidate for candidate in safe_candidates if candidate.mode == "stop"]
        stop_candidate = min(stop_candidates, key=lambda item: item.intervention_cost) if stop_candidates else None
        bypass_candidates = [candidate for candidate in safe_candidates if candidate.mode != "stop"]

        stop_progress = stop_candidate.progress_score if stop_candidate else -math.inf
        eligible_bypass = [
            candidate
            for candidate in bypass_candidates
            if candidate.progress_score >= stop_progress + self.config.progress_preference_margin
        ]

        if eligible_bypass:
            chosen = max(
                eligible_bypass,
                key=lambda item: (item.progress_score, -item.intervention_cost),
            )
        elif stop_candidate is not None:
            chosen = stop_candidate
        elif bypass_candidates:
            chosen = max(
                bypass_candidates,
                key=lambda item: (item.progress_score, -item.intervention_cost),
            )
        else:
            chosen = min(safe_candidates, key=lambda item: item.intervention_cost)

        selected_score = chosen.progress_score - 0.01 * chosen.intervention_cost
        selected = asdict(chosen)
        selected.update({"selected_score": selected_score})
        return chosen.action, selected

    def to_internal_action(self, action: Sequence[float], action_format: str = "acc_steer") -> Action:
        """Convert an environment action to internal ``[acc, steer]`` format.

        Supported formats:
            ``acc_steer``: already internal.
            ``steer_throttle`` or ``metadrive``: ``[steer, throttle_brake]``.

        The throttle/brake mapping is intentionally simple and should be
        calibrated against the target simulator before formal experiments.
        """
        if action_format == "acc_steer":
            return self._clip_action(action)
        if action_format in {"steer_throttle", "metadrive"}:
            steer = float(action[0]) * self.config.metadrive_steer_sign
            throttle_brake = max(-1.0, min(1.0, float(action[1])))
            if throttle_brake >= 0.0:
                acc = throttle_brake * self.config.max_acc
            else:
                acc = throttle_brake * abs(self.config.min_acc)
            return self._clip_action([acc, steer])
        raise ValueError("Unsupported action_format: {}".format(action_format))

    def from_internal_action(self, action: Sequence[float], action_format: str = "acc_steer") -> Action:
        """Convert internal ``[acc, steer]`` action to an environment format."""
        acc, steer = self._clip_action(action)
        if action_format == "acc_steer":
            return [acc, steer]
        if action_format in {"steer_throttle", "metadrive"}:
            if acc >= 0.0:
                throttle_brake = acc / max(self.config.max_acc, self.config.small_tolerance)
            else:
                throttle_brake = acc / max(abs(self.config.min_acc), self.config.small_tolerance)
            return [
                max(-self.config.max_steer, min(self.config.max_steer, steer * self.config.metadrive_steer_sign)),
                max(-1.0, min(1.0, throttle_brake)),
            ]
        raise ValueError("Unsupported action_format: {}".format(action_format))

    def parse_state_from_metadrive(self, env: Any) -> State:
        """Best-effort adapter from a MetaDrive env/wrapper to generic state.

        The adapter intentionally uses attribute introspection instead of a hard
        dependency on one MetaDrive version. Stopped traffic objects are treated
        as static obstacles, while moving objects are kept as adjacent vehicles
        for RSS bypass checks.
        """
        raw_env = self._unwrap_env(env)
        vehicle = self._resolve_attr(raw_env, "vehicle", None)
        if vehicle is None:
            vehicle = self._resolve_attr(raw_env, "agent", None)
        if vehicle is None:
            raise ValueError("Cannot parse MetaDrive state: env has no vehicle/agent attribute.")

        reference_lane = self._metadrive_current_lane(vehicle)
        if bool(getattr(self.config, "enable_frenet_coordinates", False)):
            self._last_frenet_reference_lane = reference_lane

        ego = self._metadrive_vehicle_to_dict(vehicle, default_speed=0.0)
        ego["_metadrive_source"] = vehicle
        self._attach_frenet_to_entity(ego, vehicle, reference_lane, role="ego")
        ego["lane_width"] = self._metadrive_current_lane_width(vehicle)
        ego["lane_id"] = self._metadrive_lane_id(vehicle)
        ego.update(self._metadrive_lane_boundary_info(vehicle))

        lanes = self._metadrive_lanes(vehicle)
        route_corridors = self._metadrive_route_corridors(vehicle)
        objects = self._collect_metadrive_objects(raw_env)
        static_obstacles = []
        vehicles = []
        ignored_count = 0
        ignored_types: Dict[str, int] = {}
        static_object_types = {"traffic_object", "obstacle", "cone", "barrier", "static_obstacle"}

        for obj in objects:
            if obj is vehicle:
                continue
            parsed = self._metadrive_object_to_dict(obj, default_speed=0.0)
            parsed["_metadrive_source"] = obj
            self._attach_frenet_to_entity(parsed, obj, reference_lane, role="object")
            if self._distance_xy(ego, parsed) > self.config.metadrive_object_scan_radius:
                continue

            _, lateral = self._relative_position(ego, parsed)
            current_width = ego.get("lane_width", self.config.default_lane_width)
            if lateral > current_width / 2.0:
                parsed["relative_lane"] = -1
            elif lateral < -current_width / 2.0:
                parsed["relative_lane"] = 1
            else:
                parsed["relative_lane"] = 0

            object_type = str(parsed.get("object_type", "unknown")).lower()
            if object_type == "vehicle":
                if parsed.get("speed", 0.0) <= self.config.metadrive_static_speed_threshold:
                    static_obstacles.append(parsed)
                else:
                    vehicles.append(parsed)
            elif object_type in static_object_types:
                static_obstacles.append(parsed)
            else:
                ignored_count += 1
                ignored_types[object_type] = ignored_types.get(object_type, 0) + 1

        return {
            "ego": ego,
            "static_obstacles": static_obstacles,
            "vehicles": vehicles,
            "lanes": lanes,
            "route_corridors": route_corridors,
            "_frenet_reference_lane": reference_lane if bool(getattr(self.config, "enable_frenet_coordinates", False)) else None,
            "adapter_debug": {
                "parse_ok": True,
                "raw_object_count": len(objects),
                "static_count": len(static_obstacles),
                "vehicle_count": len(vehicles),
                "ignored_count": ignored_count,
                "ignored_types": ignored_types,
            },
        }

    def augment_state_from_observation(self, state: State, observation: Any) -> State:
        """Add a lidar-derived front obstacle when MetaDrive object parsing is empty.

        MetaDrive versions differ in where traffic/static objects live. If the
        adapter cannot find any object, this fallback uses the current
        observation's lidar-like tail segment to avoid a silent no-op filter.
        """
        if not self.config.enable_observation_lidar_fallback:
            return state
        if len(state.get("static_obstacles", [])) + len(state.get("vehicles", [])) > 0:
            return state

        detection = self._front_lidar_detection_from_observation(observation)
        if detection is None:
            return state

        augmented = self._copy_state_preserving_frenet_reference(state)
        ego = self._ego(augmented)
        heading = float(ego.get("heading", 0.0))
        distance = detection["distance"]
        obstacle = {
            "x": float(ego.get("x", 0.0)) + distance * math.cos(heading),
            "y": float(ego.get("y", 0.0)) + distance * math.sin(heading),
            "heading": heading,
            "speed": 0.0,
            "length": self.config.vehicle_length,
            "width": self.config.vehicle_width,
            "lane_id": ego.get("lane_id"),
            "object_type": "lidar_fallback",
            "class_name": "ObservationLidarFallback",
            "object_id": "front_lidar",
        }
        self._attach_frenet_to_entity(
            obstacle,
            obstacle,
            augmented.get("_frenet_reference_lane", self._last_frenet_reference_lane),
            role="object",
        )
        augmented.setdefault("static_obstacles", []).append(obstacle)
        adapter_debug = augmented.setdefault("adapter_debug", {})
        adapter_debug.update(
            {
                "observation_lidar_fallback_used": True,
                "observation_lidar_distance": distance,
                "observation_lidar_source": detection["source"],
                "observation_lidar_value": detection["value"],
            }
        )
        return augmented

    def inject_forced_static_obstacle(
        self,
        state: State,
        distance: float = 30.0,
        lateral: float = 0.0,
        length: float = 4.5,
        width: float = 2.0,
    ) -> State:
        """Inject a debug static obstacle at a fixed distance ahead of ego.

        This is a first-iteration verification tool: the obstacle exists only
        in the RSS filter state, not in the physical simulator. It allows
        testing stop / bypass mode switching without a real MetaDrive obstacle.
        """
        augmented = self._copy_state_preserving_frenet_reference(state)
        ego = self._ego(augmented)
        heading = float(ego.get("heading", 0.0))
        ego_x = float(ego.get("x", 0.0))
        ego_y = float(ego.get("y", 0.0))
        obs_x = ego_x + distance * math.cos(heading) - lateral * math.sin(heading)
        obs_y = ego_y + distance * math.sin(heading) + lateral * math.cos(heading)
        obstacle = {
            "x": obs_x,
            "y": obs_y,
            "heading": heading,
            "speed": 0.0,
            "length": length,
            "width": width,
            "lane_id": ego.get("lane_id"),
            "object_type": "forced_debug_obstacle",
            "class_name": "ForcedDebugObstacle",
            "object_id": "forced_static_obstacle",
        }
        self._attach_frenet_to_entity(
            obstacle,
            obstacle,
            augmented.get("_frenet_reference_lane", self._last_frenet_reference_lane),
            role="object",
        )
        augmented.setdefault("static_obstacles", []).append(obstacle)
        adapter_debug = augmented.setdefault("adapter_debug", {})
        adapter_debug["forced_obstacle_injected"] = True
        adapter_debug["forced_obstacle_distance"] = distance
        adapter_debug["forced_obstacle_lateral"] = lateral
        return augmented

    def _make_candidate(
        self,
        action: Action,
        mode: str,
        safe: bool,
        margins: Dict[str, float],
        projection_debug: Dict[str, Any],
        feasible_debug: Dict[str, Any],
        u_original: Action,
        state: State,
    ) -> StaticRSSCandidate:
        progress_score = self._score_progress_after_rollout(state, action)
        intervention_cost = self._action_distance_sq(action, u_original)
        return StaticRSSCandidate(
            action=action,
            mode=mode,
            safe=safe,
            progress_score=progress_score,
            intervention_cost=intervention_cost,
            margins=margins,
            projection_debug=projection_debug,
            feasible_debug=feasible_debug,
        )

    def _with_action_debug(
        self, info: Dict[str, Any], u_original: Sequence[float], u_safe: Sequence[float]
    ) -> Dict[str, Any]:
        info["acc_nominal"] = float(u_original[0])
        info["acc_safe"] = float(u_safe[0])
        info["acc_delta"] = float(u_safe[0] - u_original[0])
        info["steer_nominal"] = float(u_original[1])
        info["steer_safe"] = float(u_safe[1])
        info["steer_delta"] = float(u_safe[1] - u_original[1])
        info["action_delta"] = float(math.sqrt(info["acc_delta"] ** 2 + info["steer_delta"] ** 2))
        info.setdefault("reason", "")
        return info

    def _check_stop_horizon(
        self, state: State, obstacle: Dict[str, Any], action: Sequence[float]
    ) -> Tuple[bool, Dict[str, float]]:
        return self._check_margin_horizon(
            state=state,
            action=action,
            margin_fn=lambda rollout_state: self._stop_margin(rollout_state, obstacle),
            min_margin_key="h_stop_min",
        )

    def _check_dynamic_front_horizon(
        self, state: State, vehicle: Dict[str, Any], action: Sequence[float]
    ) -> Tuple[bool, Dict[str, float]]:
        return self._check_margin_horizon(
            state=state,
            action=action,
            margin_fn=lambda rollout_state: self._dynamic_front_margin(rollout_state, vehicle),
            min_margin_key="dynamic_front_margin_min",
        )

    def _check_clearance_horizon(
        self, state: State, obj: Dict[str, Any], action: Sequence[float]
    ) -> Tuple[bool, Dict[str, float]]:
        return self._check_margin_horizon(
            state=state,
            action=action,
            margin_fn=lambda rollout_state: self._obstacle_clearance_margin(rollout_state, obj),
            min_margin_key="clearance_margin_min",
        )

    def _check_margin_horizon(
        self,
        state: State,
        action: Sequence[float],
        margin_fn: Any,
        min_margin_key: str,
    ) -> Tuple[bool, Dict[str, float]]:
        current_margin = margin_fn(state)
        final_margin = current_margin
        min_margin = current_margin
        rollout_state = state

        for _ in range(self.config.horizon_steps):
            rollout_state = self._simulate_next_state(rollout_state, action)
            final_margin = margin_fn(rollout_state)
            min_margin = min(min_margin, final_margin)

        margin_improvement = final_margin - current_margin
        recovery_mode_used = False
        safe = min_margin >= -self.config.small_tolerance

        if (
            not safe
            and self.config.enable_recovery_mode
            and current_margin < -self.config.small_tolerance
            and self._clip_action(action)[0] <= self.config.small_tolerance
        ):
            improves_enough = margin_improvement >= self.config.recovery_margin_improvement
            holds_margin = self.config.recovery_allow_equal_margin and final_margin >= current_margin
            recovery_mode_used = improves_enough or holds_margin
            safe = recovery_mode_used

        margins = {
            min_margin_key: min_margin,
            "current_margin": current_margin,
            "final_margin": final_margin,
            "min_margin": min_margin,
            "margin_improvement": margin_improvement,
            "recovery_mode_used": recovery_mode_used,
        }
        return safe, margins

    def road_boundary_horizon_metrics(
        self,
        state: State,
        action: Sequence[float],
        steps: Optional[int] = None,
        margin: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate lane/road boundary safety separately from object safety."""
        horizon = self.config.horizon_steps if steps is None else int(steps)
        rollout_state = state
        left_margins: List[float] = []
        right_margins: List[float] = []
        lateral_positions: List[float] = []
        lower_limit = -math.inf
        upper_limit = math.inf
        min_margin = math.inf
        final_margin = math.inf
        source = ""

        for step in range(max(0, horizon) + 1):
            if step > 0:
                rollout_state = self._simulate_next_state(rollout_state, action)
            metrics = self._basic_road_boundary_margins_for_state(state, rollout_state, margin)
            left_margin = float(metrics["left_margin"])
            right_margin = float(metrics["right_margin"])
            boundary_margin = float(metrics["margin"])
            source = str(metrics.get("source", source))
            left_margins.append(left_margin)
            right_margins.append(right_margin)
            lateral_positions.append(float(metrics["lateral"]))
            lower_limit = float(metrics["lower"])
            upper_limit = float(metrics["upper"])
            min_margin = min(min_margin, boundary_margin)
            final_margin = boundary_margin

        ego = self._ego(state)
        hard_violation = bool(
            not bool(ego.get("on_lane", True))
            or bool(ego.get("out_of_route", False))
            or bool(ego.get("crash_sidewalk", False))
        )
        left_min = min(left_margins) if left_margins else math.inf
        right_min = min(right_margins) if right_margins else math.inf
        current_margin = min(left_margins[0], right_margins[0]) if left_margins and right_margins else math.inf
        current_lateral = lateral_positions[0] if lateral_positions else 0.0
        final_lateral = lateral_positions[-1] if lateral_positions else current_lateral
        center_lateral = 0.5 * (lower_limit + upper_limit)
        current_center_error = abs(current_lateral - center_lateral)
        final_center_error = abs(final_lateral - center_lateral)
        road_safe = (
            min_margin >= -self.config.small_tolerance
            and not hard_violation
        )
        return {
            "road_boundary_safe": bool(road_safe),
            "left_boundary_safe": bool(left_min >= -self.config.small_tolerance),
            "right_boundary_safe": bool(right_min >= -self.config.small_tolerance),
            "left_boundary_margin": float(left_min),
            "right_boundary_margin": float(right_min),
            "min_boundary_margin": float(min_margin),
            "road_boundary_margin_current": float(current_margin),
            "road_boundary_margin_min_pred": float(min_margin),
            "road_boundary_margin_final_pred": float(final_margin),
            "road_boundary_margin_improvement": float(final_margin - current_margin),
            "road_boundary_margin_source": source,
            "road_boundary_horizon_steps": int(max(0, horizon)),
            "road_boundary_hard_violation": bool(hard_violation),
            "road_boundary_left_margin_min_pred": float(left_min),
            "road_boundary_right_margin_min_pred": float(right_min),
            "road_boundary_left_margin_current": float(left_margins[0]) if left_margins else math.inf,
            "road_boundary_right_margin_current": float(right_margins[0]) if right_margins else math.inf,
            "road_boundary_left_margin_final_pred": float(left_margins[-1]) if left_margins else math.inf,
            "road_boundary_right_margin_final_pred": float(right_margins[-1]) if right_margins else math.inf,
            "predicted_lateral_position": float(final_lateral),
            "predicted_lane_offset": float(final_lateral),
            "road_boundary_current_lateral_position": float(current_lateral),
            "road_boundary_center_lateral": float(center_lateral),
            "road_boundary_centering_improvement": float(current_center_error - final_center_error),
            "ego_dist_to_left_side": self._safe_float(ego.get("dist_to_left_side", math.nan), math.nan),
            "ego_dist_to_right_side": self._safe_float(ego.get("dist_to_right_side", math.nan), math.nan),
            "ego_on_lane": bool(ego.get("on_lane", True)),
            "ego_out_of_route": bool(ego.get("out_of_route", False)),
            "ego_crash_sidewalk": bool(ego.get("crash_sidewalk", False)),
        }

    def _basic_road_boundary_margins_for_state(
        self,
        reference_state: State,
        rollout_state: State,
        margin: Optional[float] = None,
    ) -> Dict[str, Any]:
        initial_ego = self._ego(reference_state)
        _, lateral = self._relative_position(initial_ego, self._ego(rollout_state))
        lower, upper, source = self._basic_road_boundary_limits_from_state(reference_state, margin)
        left_margin = upper - lateral
        right_margin = lateral - lower
        return {
            "left_margin": float(left_margin),
            "right_margin": float(right_margin),
            "margin": float(min(left_margin, right_margin)),
            "lateral": float(lateral),
            "lower": float(lower),
            "upper": float(upper),
            "source": source,
        }

    def _basic_road_boundary_limits_from_state(
        self,
        state: State,
        margin: Optional[float] = None,
    ) -> Tuple[float, float, str]:
        margin_value = self.config.lane_margin if margin is None else float(margin)
        ego = self._ego(state)
        left_distance = self._safe_float(ego.get("dist_to_left_side", math.nan), math.nan)
        right_distance = self._safe_float(ego.get("dist_to_right_side", math.nan), math.nan)
        if math.isfinite(left_distance) and math.isfinite(right_distance) and left_distance + right_distance > 0.0:
            return (
                -right_distance + margin_value,
                left_distance - margin_value,
                "metadrive_side_distance",
            )

        current_width = self._current_lane_width(state)
        lanes = state.get("lanes", {}) or {}
        left_lane = lanes.get("left")
        right_lane = lanes.get("right")
        left_available = bool((left_lane or {}).get("available", False)) and not bool(
            ego.get("left_lane_line_prohibited", False)
        )
        right_available = bool((right_lane or {}).get("available", False)) and not bool(
            ego.get("right_lane_line_prohibited", False)
        )
        left_width = self._lane_available_width(state, left_lane) if left_available else 0.0
        right_width = self._lane_available_width(state, right_lane) if right_available else 0.0
        upper = current_width / 2.0 + left_width - margin_value
        lower = -current_width / 2.0 - right_width + margin_value
        return lower, upper, "lane_width_estimate"

    def _stop_margin(self, state: State, obstacle: Dict[str, Any]) -> float:
        d_obs = self._distance_to_obstacle_front(state, obstacle)
        return d_obs - self.compute_brake_distance(self._ego_speed(state))

    def _dynamic_front_margin(self, state: State, vehicle: Dict[str, Any]) -> float:
        d_front = self._distance_to_obstacle_front(state, vehicle)
        return d_front - self.compute_dynamic_front_distance(
            self._ego_speed(state),
            float(vehicle.get("speed", 0.0)),
        )

    def _filter_dynamic_front_vehicle(
        self,
        state: State,
        vehicle: Dict[str, Any],
        d_front: float,
        d_dynamic: float,
        dynamic_margin: float,
        u_original: Action,
    ) -> Tuple[Action, Dict[str, Any]]:
        if self.config.preserve_steer_on_stop:
            u_stop_nom = list(u_original)
            stop_center = self._clip_action([self.config.min_acc, u_original[1]])
        else:
            u_stop_nom = self.generate_stop_nominal_action(state)
            stop_center = None
        u_stop_proj, project_debug = self.ray_project_action(
            state, vehicle, u_stop_nom, "dynamic_stop", center_override=stop_center
        )
        stop_safe, margins = self.is_action_safe_for_mode(state, vehicle, u_stop_proj, "dynamic_stop")
        candidate = self._make_candidate(
            action=u_stop_proj,
            mode="dynamic_stop",
            safe=stop_safe,
            margins=margins,
            projection_debug=project_debug,
            feasible_debug={"feasible": True, "reason": "front_vehicle_rss"},
            u_original=u_original,
            state=state,
        )
        u_safe, selected_info = self.select_action([candidate], u_original)
        return u_safe, self._with_action_debug({
            "mode": selected_info["mode"],
            "obstacle_detected": False,
            "dynamic_vehicle_detected": True,
            "d_front": d_front,
            "d_dynamic": d_dynamic,
            "rss_margin": dynamic_margin,
            "reason": selected_info.get("reason", ""),
            "left_feasible": False,
            "right_feasible": False,
            "state_debug": self._state_debug(state),
            "blocking_object": self._object_debug(vehicle, state),
            "selected_score": selected_info.get("selected_score"),
            "projection_debug": selected_info.get("projection_debug", {}),
            "selected": selected_info,
            "candidates": [asdict(candidate)],
        }, u_original, u_safe)

    def _filter_predictive_clearance_risk(
        self, state: State, clearance_info: Dict[str, Any], u_original: Action
    ) -> Tuple[Action, Dict[str, Any]]:
        obj = clearance_info["object"]
        if self.config.preserve_steer_on_stop:
            u_stop_nom = list(u_original)
            stop_center = self._clip_action([self.config.min_acc, u_original[1]])
        else:
            u_stop_nom = self.generate_stop_nominal_action(state)
            stop_center = None
        u_stop_proj, project_debug = self.ray_project_action(
            state, obj, u_stop_nom, "clearance_stop", center_override=stop_center
        )
        stop_safe, margins = self.is_action_safe_for_mode(state, obj, u_stop_proj, "clearance_stop")
        candidate = self._make_candidate(
            action=u_stop_proj,
            mode="clearance_stop",
            safe=stop_safe,
            margins=margins,
            projection_debug=project_debug,
            feasible_debug={
                "feasible": True,
                "reason": "predictive_clearance_violation",
                "risk_step": clearance_info.get("risk_step", 0),
            },
            u_original=u_original,
            state=state,
        )
        u_safe, selected_info = self.select_action([candidate], u_original)
        return u_safe, self._with_action_debug({
            "mode": selected_info["mode"],
            "obstacle_detected": obj in state.get("static_obstacles", []),
            "dynamic_vehicle_detected": obj in state.get("vehicles", []),
            "clearance_margin": clearance_info.get("clearance_margin"),
            "risk_step": clearance_info.get("risk_step"),
            "rss_margin": clearance_info.get("clearance_margin"),
            "reason": selected_info.get("reason", ""),
            "left_feasible": False,
            "right_feasible": False,
            "state_debug": self._state_debug(state),
            "blocking_object": self._object_debug(obj, state),
            "selected_score": selected_info.get("selected_score"),
            "projection_debug": selected_info.get("projection_debug", {}),
            "selected": selected_info,
            "candidates": [asdict(candidate)],
        }, u_original, u_safe)

    def _score_progress_after_rollout(self, state: State, action: Sequence[float]) -> float:
        rollout = self._rollout_states(state, action, self.config.horizon_steps)
        if not rollout:
            return 0.0
        return self._longitudinal_progress(self._ego(state), self._ego(rollout[-1]))

    def _state_debug(self, state: State) -> Dict[str, Any]:
        nearest = None
        nearest_distance = math.inf
        ego = self._ego(state)
        for obj in list(state.get("static_obstacles", [])) + list(state.get("vehicles", [])):
            distance = self._distance_xy(ego, obj)
            if distance < nearest_distance:
                nearest = obj
                nearest_distance = distance
        return {
            "num_static_obstacles": len(state.get("static_obstacles", [])),
            "num_dynamic_vehicles": len(state.get("vehicles", [])),
            "nearest_object": self._object_debug(nearest, state) if nearest is not None else {},
            "coordinate_mode": ego.get("coordinate_mode", "ego_local_fallback"),
            "frenet_valid": bool(ego.get("frenet_valid", False)),
            "frenet_fallback_reason": ego.get("frenet_fallback_reason", ""),
            "s_ego": ego.get("frenet_s", math.nan),
            "l_ego": ego.get("frenet_l", math.nan),
            "heading_ref_ego": ego.get("frenet_heading_ref", math.nan),
            "v_ego_s": ego.get("frenet_v_s", math.nan),
            "adapter_debug": state.get("adapter_debug", {}),
        }

    def _front_lidar_detection_from_observation(self, observation: Any) -> Optional[Dict[str, Any]]:
        try:
            arr = np.asarray(observation, dtype=np.float32).reshape(-1)
        except Exception:
            return None

        n_lasers = min(int(self.config.metadrive_lidar_num_lasers), arr.size)
        if n_lasers <= 8:
            return None

        lidar = arr[-n_lasers:]
        lidar = lidar[np.isfinite(lidar)]
        if lidar.size <= 8:
            return None
        lidar_unit = lidar[(lidar >= 0.0) & (lidar <= 1.0)]
        if lidar_unit.size <= 8:
            return None
        global_median = float(np.median(lidar_unit))
        use_distance_encoding = global_median > 0.2
        use_proximity_encoding = not use_distance_encoding

        window = max(2, int(n_lasers * self.config.metadrive_lidar_front_sector_ratio))
        candidates = []

        def add_sector(center: int, name: str) -> None:
            indices = [(center + offset) % n_lasers for offset in range(-window, window + 1)]
            sector = arr[-n_lasers:][indices]
            sector = sector[np.isfinite(sector)]
            sector = sector[(sector >= 0.0) & (sector <= 1.0)]
            if sector.size < max(2, window // 2):
                return
            positive = sector[sector > self.config.small_tolerance]
            if use_distance_encoding and positive.size:
                min_distance_ratio = float(np.min(positive))
                if min_distance_ratio <= self.config.metadrive_lidar_distance_trigger_ratio:
                    candidates.append(
                        {
                            "source": name + "_distance",
                            "value": min_distance_ratio,
                            "distance": max(
                                self.config.vehicle_length,
                                min_distance_ratio * self.config.metadrive_lidar_max_distance,
                            ),
                        }
                    )
            max_proximity = float(np.max(sector))
            if use_proximity_encoding and max_proximity >= self.config.metadrive_lidar_proximity_trigger_ratio:
                candidates.append(
                    {
                        "source": name + "_proximity",
                        "value": max_proximity,
                        "distance": max(
                            self.config.vehicle_length,
                            (1.0 - max_proximity) * self.config.metadrive_lidar_max_distance,
                        ),
                    }
                )

        add_sector(0, "front_zero_index")
        add_sector(n_lasers // 2, "front_mid_index")

        if not candidates:
            return None
        return min(candidates, key=lambda item: item["distance"])

    def _object_debug(self, obj: Optional[Dict[str, Any]], state: State) -> Dict[str, Any]:
        if not obj:
            return {}
        longitudinal, lateral = self._relative_position(self._ego(state), obj)
        return {
            "object_type": obj.get("object_type", ""),
            "class_name": obj.get("class_name", ""),
            "object_id": obj.get("object_id", ""),
            "x": obj.get("x", float("nan")),
            "y": obj.get("y", float("nan")),
            "speed": obj.get("speed", float("nan")),
            "relative_lane": obj.get("relative_lane", ""),
            "longitudinal": longitudinal,
            "lateral": lateral,
            "coordinate_mode": obj.get("coordinate_mode", "ego_local_fallback"),
            "frenet_valid": bool(obj.get("frenet_valid", False)),
            "frenet_fallback_reason": obj.get("frenet_fallback_reason", ""),
            "frenet_s": obj.get("frenet_s", float("nan")),
            "frenet_l": obj.get("frenet_l", float("nan")),
            "frenet_v_s": obj.get("frenet_v_s", float("nan")),
            "distance": self._distance_xy(self._ego(state), obj),
            "lane_id": obj.get("lane_id", ""),
        }

    def _safe_center_for_mode(self, mode: str) -> Action:
        if mode in {"stop", "dynamic_stop", "clearance_stop"}:
            return self._clip_action([self.config.min_acc, 0.0])
        if mode == "left_bypass":
            return self._clip_action([0.0, self.config.safe_steer_center])
        if mode == "right_bypass":
            return self._clip_action([0.0, -self.config.safe_steer_center])
        raise ValueError("Unsupported mode: {}".format(mode))

    def _rollout_states(self, state: State, action: Sequence[float], steps: int) -> List[State]:
        rollout = []
        rollout_state = state
        for _ in range(max(0, int(steps))):
            rollout_state = self._simulate_next_state(rollout_state, action)
            rollout.append(rollout_state)
        return rollout

    def _simulate_next_state(self, state: State, action: Sequence[float]) -> State:
        """Lightweight rollout model; replace with real dynamics if available."""
        acc, steer = self._clip_action(action)
        cfg = self.config
        reference_lane = state.get("_frenet_reference_lane", self._last_frenet_reference_lane)
        next_state = self._copy_state_preserving_frenet_reference(state)
        ego = copy.deepcopy(self._ego(state))

        x = float(ego.get("x", 0.0))
        y = float(ego.get("y", 0.0))
        heading = float(ego.get("heading", 0.0))
        speed = max(0.0, float(ego.get("speed", 0.0)))

        ego["x"] = x + speed * math.cos(heading) * cfg.dt
        ego["y"] = y + speed * math.sin(heading) * cfg.dt
        ego["speed"] = max(0.0, min(cfg.v_max, speed + acc * cfg.dt))
        ego["heading"] = heading + steer * cfg.steer_gain * cfg.dt
        self._attach_frenet_to_entity(ego, ego, reference_lane, role="ego")
        next_state["ego"] = ego
        return next_state

    def _copy_state_preserving_frenet_reference(self, state: State) -> State:
        reference_lane = state.get("_frenet_reference_lane", None)
        copy_state = self._strip_runtime_object_refs(state)
        if reference_lane is None:
            return copy.deepcopy(copy_state)
        copy_state["_frenet_reference_lane"] = None
        copied = copy.deepcopy(copy_state)
        copied["_frenet_reference_lane"] = reference_lane
        return copied

    def _strip_runtime_object_refs(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: self._strip_runtime_object_refs(item)
                for key, item in value.items()
                if key != "_metadrive_source"
            }
        if isinstance(value, list):
            return [self._strip_runtime_object_refs(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._strip_runtime_object_refs(item) for item in value)
        return value

    def _copy_entity_without_runtime_refs(self, entity: Any) -> Any:
        return copy.deepcopy(self._strip_runtime_object_refs(entity))

    def _copy_entity_preserving_runtime_refs(self, entity: Any) -> Any:
        if not isinstance(entity, dict):
            return copy.deepcopy(entity)
        source = entity.get("_metadrive_source", None)
        copied = self._copy_entity_without_runtime_refs(entity)
        if source is not None and isinstance(copied, dict):
            copied["_metadrive_source"] = source
        return copied

    def _obstacle_clearance_margin(self, state: State, obstacle: Dict[str, Any]) -> float:
        ego = self._ego(state)
        longitudinal, lateral = self._relative_position(ego, obstacle)
        obstacle_length = self._object_length(obstacle, self.config.vehicle_length)
        obstacle_width = self._object_width(obstacle, self.config.vehicle_width)
        longitudinal_gap = abs(longitudinal) - (
            self.config.vehicle_length / 2.0 + obstacle_length / 2.0 + self.config.obstacle_margin
        )
        lateral_gap = abs(lateral) - (
            self.config.vehicle_width / 2.0 + obstacle_width / 2.0 + self.config.obstacle_margin
        )
        return max(longitudinal_gap, lateral_gap)

    def _lane_boundary_margin(
        self, initial_ego: Dict[str, Any], rollout_state: State, direction: str, reference_state: State
    ) -> float:
        _, lateral = self._relative_position(initial_ego, self._ego(rollout_state))
        current_width = self._current_lane_width(reference_state)
        target_lane = (reference_state.get("lanes", {}) or {}).get(direction, {})
        target_width = self._lane_available_width(reference_state, target_lane)

        if direction == "left":
            lower = -current_width / 2.0 + self.config.lane_margin
            upper = current_width / 2.0 + target_width - self.config.lane_margin
        else:
            lower = -current_width / 2.0 - target_width + self.config.lane_margin
            upper = current_width / 2.0 - self.config.lane_margin
        return min(lateral - lower, upper - lateral)

    def _adjacent_vehicle_rss_margins(
        self, state: State, direction: str
    ) -> Tuple[float, float, List[Dict[str, Any]]]:
        ego = self._ego(state)
        ego_speed = max(0.0, float(ego.get("speed", 0.0)))
        front_margin = math.inf
        rear_margin = math.inf
        debug = []

        for vehicle in state.get("vehicles", []):
            if not self._vehicle_in_target_lane(state, vehicle, direction):
                continue

            longitudinal, _ = self._relative_position(ego, vehicle)
            vehicle_length = self._object_length(vehicle, self.config.vehicle_length)
            bumper_distance = abs(longitudinal) - (self.config.vehicle_length + vehicle_length) / 2.0
            vehicle_speed = max(0.0, float(vehicle.get("speed", 0.0)))

            if longitudinal >= 0.0:
                required = self._rss_front_distance(ego_speed, vehicle_speed)
                margin = bumper_distance - required
                front_margin = min(front_margin, margin)
                relation = "front"
            else:
                required = self._rss_rear_distance(ego_speed, vehicle_speed)
                margin = bumper_distance - required
                rear_margin = min(rear_margin, margin)
                relation = "rear"

            debug.append(
                {
                    "relation": relation,
                    "longitudinal": longitudinal,
                    "bumper_distance": bumper_distance,
                    "required_rss_distance": required,
                    "margin": margin,
                }
            )

        return front_margin, rear_margin, debug

    def _rss_front_distance(self, ego_speed: float, front_speed: float) -> float:
        cfg = self.config
        response_speed = ego_speed + cfg.a_max * cfg.rho
        distance = (
            ego_speed * cfg.rho
            + 0.5 * cfg.a_max * cfg.rho ** 2
            + response_speed ** 2 / (2.0 * cfg.b_min)
            - front_speed ** 2 / (2.0 * cfg.b_min)
            + cfg.adjacent_vehicle_margin
        )
        return max(0.0, distance)

    def _rss_rear_distance(self, ego_speed: float, rear_speed: float) -> float:
        cfg = self.config
        response_speed = rear_speed + cfg.a_max * cfg.rho
        distance = (
            rear_speed * cfg.rho
            + 0.5 * cfg.a_max * cfg.rho ** 2
            + response_speed ** 2 / (2.0 * cfg.b_min)
            - ego_speed ** 2 / (2.0 * cfg.b_min)
            + cfg.adjacent_vehicle_margin
        )
        return max(0.0, distance)

    def _vehicle_in_target_lane(self, state: State, vehicle: Dict[str, Any], direction: str) -> bool:
        relative_lane = vehicle.get("relative_lane")
        if relative_lane is not None:
            return int(relative_lane) == (-1 if direction == "left" else 1)

        lanes = state.get("lanes", {}) or {}
        target_lane = lanes.get(direction, {}) or {}
        target_lane_id = target_lane.get("lane_id", target_lane.get("id"))
        vehicle_lane_id = vehicle.get("lane_id")
        if target_lane_id is not None and vehicle_lane_id is not None:
            return target_lane_id == vehicle_lane_id

        _, lateral = self._relative_position(self._ego(state), vehicle)
        current_width = self._current_lane_width(state)
        target_width = self._lane_available_width(state, target_lane)
        if direction == "left":
            return current_width / 2.0 <= lateral <= current_width / 2.0 + target_width
        return -current_width / 2.0 - target_width <= lateral <= -current_width / 2.0

    def _distance_to_obstacle_front(self, state: State, obstacle: Dict[str, Any]) -> float:
        longitudinal, _ = self._relative_position(self._ego(state), obstacle)
        obstacle_length = self._object_length(obstacle, self.config.vehicle_length)
        return longitudinal - obstacle_length / 2.0 - self.config.vehicle_length / 2.0

    def get_vehicle_frenet(self, vehicle: Any, reference_lane: Any) -> Dict[str, Any]:
        if reference_lane is None:
            return {
                "s": math.nan,
                "l": math.nan,
                "heading_ref": math.nan,
                "v_s": math.nan,
                "v_l": math.nan,
                "valid": False,
                "fallback_reason": "missing_reference_lane",
            }

        position = self._frenet_position(vehicle)
        if position is None:
            return {
                "s": math.nan,
                "l": math.nan,
                "heading_ref": math.nan,
                "v_s": math.nan,
                "v_l": math.nan,
                "valid": False,
                "fallback_reason": "missing_position",
            }

        try:
            s_value, l_value = reference_lane.local_coordinates(position)
            s_value = float(s_value)
            l_value = float(l_value)
        except Exception as exc:
            return {
                "s": math.nan,
                "l": math.nan,
                "heading_ref": math.nan,
                "v_s": math.nan,
                "v_l": math.nan,
                "valid": False,
                "fallback_reason": "local_coordinates_failed:{}".format(type(exc).__name__),
            }

        heading_ref = math.nan
        try:
            heading_ref = float(reference_lane.heading_at(s_value))
        except Exception as exc:
            return {
                "s": float(s_value),
                "l": float(l_value),
                "heading_ref": math.nan,
                "v_s": math.nan,
                "v_l": math.nan,
                "valid": False,
                "fallback_reason": "heading_at_failed:{}".format(type(exc).__name__),
            }

        heading = self._entity_heading(vehicle, heading_ref)
        speed = self._entity_speed(vehicle, 0.0)
        heading_error = heading - heading_ref
        return {
            "s": float(s_value),
            "l": float(l_value),
            "heading_ref": float(heading_ref),
            "v_s": float(speed * math.cos(heading_error)),
            "v_l": float(speed * math.sin(heading_error)),
            "valid": True,
            "fallback_reason": "",
        }

    def _attach_frenet_to_entity(
        self,
        entity_dict: Dict[str, Any],
        source: Any,
        reference_lane: Any,
        role: str = "object",
    ) -> None:
        if not bool(getattr(self.config, "enable_frenet_coordinates", False)):
            entity_dict.update(
                {
                    "coordinate_mode": "ego_local_fallback",
                    "frenet_valid": False,
                    "frenet_fallback_reason": "disabled",
                }
            )
            return

        frenet = self.get_vehicle_frenet(source, reference_lane)
        valid = bool(frenet.get("valid", False))
        entity_dict.update(
            {
                "coordinate_mode": "frenet" if valid else "ego_local_fallback",
                "frenet_valid": valid,
                "frenet_fallback_reason": str(frenet.get("fallback_reason", "")),
                "frenet_s": float(frenet.get("s", math.nan)),
                "frenet_l": float(frenet.get("l", math.nan)),
                "frenet_heading_ref": float(frenet.get("heading_ref", math.nan)),
                "frenet_v_s": float(frenet.get("v_s", math.nan)),
                "frenet_v_l": float(frenet.get("v_l", math.nan)),
                "frenet_role": role,
            }
        )
        if role == "ego" and valid:
            entity_dict["lateral_speed"] = float(frenet.get("v_l", 0.0))

    def _frenet_position(self, entity: Any) -> Optional[Any]:
        if isinstance(entity, dict):
            x = self._safe_float(entity.get("x", math.nan), math.nan)
            y = self._safe_float(entity.get("y", math.nan), math.nan)
            if math.isfinite(x) and math.isfinite(y):
                return np.asarray([x, y], dtype=float)
            position = entity.get("position", None)
            if position is not None:
                try:
                    arr = np.asarray(position, dtype=float)
                    if arr.size >= 2 and math.isfinite(float(arr[0])) and math.isfinite(float(arr[1])):
                        return arr[:2]
                except Exception:
                    return None
            return None
        position = self._resolve_attr(entity, "position", None)
        if position is None:
            position = self._resolve_attr(entity, "get_position", None)
        return position

    def _entity_heading(self, entity: Any, default: float = 0.0) -> float:
        if isinstance(entity, dict):
            return self._safe_float(entity.get("heading", entity.get("heading_theta", default)), default)
        heading = self._resolve_attr(
            entity,
            "heading_theta",
            self._resolve_attr(entity, "heading", default),
        )
        return self._safe_float(heading, default)

    def _entity_speed(self, entity: Any, default: float = 0.0) -> float:
        if isinstance(entity, dict):
            speed = self._safe_float(entity.get("speed", math.nan), math.nan)
            if math.isfinite(speed):
                return max(0.0, speed)
            velocity = entity.get("velocity", None)
            if velocity is not None:
                try:
                    return float(np.linalg.norm(np.asarray(velocity, dtype=float)))
                except Exception:
                    pass
            return max(0.0, float(default))
        speed = self._resolve_attr(entity, "speed", None)
        if speed is not None:
            return max(0.0, self._safe_float(speed, default))
        velocity = self._resolve_attr(entity, "velocity", None)
        if velocity is None:
            return max(0.0, float(default))
        try:
            return float(np.linalg.norm(np.asarray(velocity, dtype=float)))
        except Exception:
            return max(0.0, float(default))

    def _ego_local_relative_position(self, ego: Dict[str, Any], obj: Dict[str, Any]) -> Tuple[float, float]:
        ego_position = self._frenet_position(ego)
        obj_position = self._frenet_position(obj)
        if ego_position is not None and obj_position is not None:
            dx = float(obj_position[0]) - float(ego_position[0])
            dy = float(obj_position[1]) - float(ego_position[1])
        else:
            dx = float(obj.get("x", 0.0)) - float(ego.get("x", 0.0))
            dy = float(obj.get("y", 0.0)) - float(ego.get("y", 0.0))
        heading = float(ego.get("heading", 0.0))
        longitudinal = dx * math.cos(heading) + dy * math.sin(heading)
        lateral = -dx * math.sin(heading) + dy * math.cos(heading)
        return longitudinal, lateral

    def _relative_position_metrics(self, ego: Dict[str, Any], obj: Dict[str, Any]) -> Dict[str, Any]:
        old_delta_s, old_delta_l = self._ego_local_relative_position(ego, obj)
        enabled = bool(getattr(self.config, "enable_frenet_coordinates", False))
        fallback_allowed = bool(getattr(self.config, "frenet_fallback_to_ego_local", True))
        ego_valid = bool(ego.get("frenet_valid", False))
        obj_valid = bool(obj.get("frenet_valid", False))
        if enabled and ego_valid and obj_valid:
            s_ego = self._safe_float(ego.get("frenet_s", math.nan), math.nan)
            l_ego = self._safe_float(ego.get("frenet_l", math.nan), math.nan)
            s_obj = self._safe_float(obj.get("frenet_s", math.nan), math.nan)
            l_obj = self._safe_float(obj.get("frenet_l", math.nan), math.nan)
            if math.isfinite(s_ego) and math.isfinite(l_ego) and math.isfinite(s_obj) and math.isfinite(l_obj):
                return {
                    "delta_s": float(s_obj - s_ego),
                    "delta_l": float(l_obj - l_ego),
                    "coordinate_mode": "frenet",
                    "frenet_valid": True,
                    "frenet_fallback_reason": "",
                    "s_ego": float(s_ego),
                    "l_ego": float(l_ego),
                    "heading_ref_ego": self._safe_float(ego.get("frenet_heading_ref", math.nan), math.nan),
                    "v_ego_s": self._safe_float(ego.get("frenet_v_s", math.nan), math.nan),
                    "v_ego_l": self._safe_float(ego.get("frenet_v_l", math.nan), math.nan),
                    "s_obj": float(s_obj),
                    "l_obj": float(l_obj),
                    "heading_ref_obj": self._safe_float(obj.get("frenet_heading_ref", math.nan), math.nan),
                    "v_obj_s": self._safe_float(obj.get("frenet_v_s", math.nan), math.nan),
                    "v_obj_l": self._safe_float(obj.get("frenet_v_l", math.nan), math.nan),
                    "old_delta_s": float(old_delta_s),
                    "old_delta_l": float(old_delta_l),
                }

        if enabled and not fallback_allowed:
            reason = "frenet_invalid"
        elif not enabled:
            reason = "disabled"
        else:
            reason = str(
                ego.get("frenet_fallback_reason", "")
                or obj.get("frenet_fallback_reason", "")
                or "frenet_invalid"
            )
        return {
            "delta_s": float(old_delta_s),
            "delta_l": float(old_delta_l),
            "coordinate_mode": "ego_local_fallback",
            "frenet_valid": False,
            "frenet_fallback_reason": reason,
            "s_ego": self._safe_float(ego.get("frenet_s", math.nan), math.nan),
            "l_ego": self._safe_float(ego.get("frenet_l", math.nan), math.nan),
            "heading_ref_ego": self._safe_float(ego.get("frenet_heading_ref", math.nan), math.nan),
            "v_ego_s": self._safe_float(ego.get("frenet_v_s", math.nan), math.nan),
            "v_ego_l": self._safe_float(ego.get("frenet_v_l", math.nan), math.nan),
            "s_obj": self._safe_float(obj.get("frenet_s", math.nan), math.nan),
            "l_obj": self._safe_float(obj.get("frenet_l", math.nan), math.nan),
            "heading_ref_obj": self._safe_float(obj.get("frenet_heading_ref", math.nan), math.nan),
            "v_obj_s": self._safe_float(obj.get("frenet_v_s", math.nan), math.nan),
            "v_obj_l": self._safe_float(obj.get("frenet_v_l", math.nan), math.nan),
            "old_delta_s": float(old_delta_s),
            "old_delta_l": float(old_delta_l),
        }

    def _relative_position(self, ego: Dict[str, Any], obj: Dict[str, Any]) -> Tuple[float, float]:
        metrics = self._relative_position_metrics(ego, obj)
        return float(metrics["delta_s"]), float(metrics["delta_l"])

    def _longitudinal_progress(self, initial_ego: Dict[str, Any], final_ego: Dict[str, Any]) -> float:
        dx = float(final_ego.get("x", 0.0)) - float(initial_ego.get("x", 0.0))
        dy = float(final_ego.get("y", 0.0)) - float(initial_ego.get("y", 0.0))
        heading = float(initial_ego.get("heading", 0.0))
        return dx * math.cos(heading) + dy * math.sin(heading)

    def _lateral_progress(self, initial_ego: Dict[str, Any], final_ego: Dict[str, Any]) -> float:
        dx = float(final_ego.get("x", 0.0)) - float(initial_ego.get("x", 0.0))
        dy = float(final_ego.get("y", 0.0)) - float(initial_ego.get("y", 0.0))
        heading = float(initial_ego.get("heading", 0.0))
        return -dx * math.sin(heading) + dy * math.cos(heading)

    def _lane_ids_differ(self, ego: Dict[str, Any], obstacle: Dict[str, Any]) -> bool:
        ego_lane = ego.get("lane_id")
        obstacle_lane = obstacle.get("lane_id")
        return ego_lane is not None and obstacle_lane is not None and ego_lane != obstacle_lane

    def _current_lane_width(self, state: State) -> float:
        ego = self._ego(state)
        current_lane = (state.get("lanes", {}) or {}).get("current", {}) or {}
        return float(
            ego.get(
                "lane_width",
                current_lane.get("available_width", current_lane.get("width", self.config.default_lane_width)),
            )
        )

    def _lane_available_width(self, state: State, lane: Optional[Dict[str, Any]]) -> float:
        if not lane:
            return self.config.default_lane_width
        return float(lane.get("available_width", lane.get("width", self.config.default_lane_width)))

    def _ego(self, state: State) -> Dict[str, Any]:
        return state.get("ego", {})

    def _ego_speed(self, state: State) -> float:
        return max(0.0, float(self._ego(state).get("speed", 0.0)))

    def _object_length(self, obj: Dict[str, Any], default: float) -> float:
        return float(obj.get("length", default))

    def _object_width(self, obj: Dict[str, Any], default: float) -> float:
        return float(obj.get("width", default))

    def _direction_sign(self, direction: str) -> float:
        self._validate_direction(direction)
        return 1.0 if direction == "left" else -1.0

    def _validate_direction(self, direction: str) -> None:
        if direction not in {"left", "right"}:
            raise ValueError("direction must be 'left' or 'right', got {}".format(direction))

    def _clip_action(self, action: Sequence[float]) -> Action:
        if len(action) != 2:
            raise ValueError("Action must be [acc, steer], got {}".format(action))
        acc = max(self.config.min_acc, min(self.config.max_acc, float(action[0])))
        steer = max(-self.config.max_steer, min(self.config.max_steer, float(action[1])))
        return [acc, steer]

    def _action_distance_sq(self, action: Sequence[float], reference: Sequence[float]) -> float:
        action = self._clip_action(action)
        reference = self._clip_action(reference)
        return (action[0] - reference[0]) ** 2 + (action[1] - reference[1]) ** 2

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _unwrap_env(self, env: Any) -> Any:
        raw_env = env
        seen = set()
        while hasattr(raw_env, "env") and id(raw_env) not in seen:
            seen.add(id(raw_env))
            raw_env = raw_env.env
        return raw_env

    def _resolve_attr(self, obj: Any, name: str, default: Any = None) -> Any:
        """Read a MetaDrive attribute that may be a property or zero-arg method."""
        value = getattr(obj, name, default)
        if callable(value):
            try:
                return value()
            except TypeError:
                return value
            except Exception:
                return default
        return value

    def _metadrive_position_xy(self, obj: Any) -> Tuple[float, float]:
        """Return ``(x, y)`` for MetaDrive/Panda3D objects across API variants."""
        position = self._resolve_attr(obj, "position", None)

        if position is None:
            position = self._resolve_attr(obj, "get_position", None)

        if position is None:
            return 0.0, 0.0

        if hasattr(position, "x") and hasattr(position, "y"):
            try:
                return float(position.x), float(position.y)
            except Exception:
                pass

        try:
            return float(position[0]), float(position[1])
        except Exception:
            pass

        try:
            return float(position.getX()), float(position.getY())
        except Exception:
            pass

        return 0.0, 0.0

    def _metadrive_vehicle_to_dict(self, vehicle: Any, default_speed: float = 0.0) -> Dict[str, Any]:
        x, y = self._metadrive_position_xy(vehicle)
        heading = self._resolve_attr(
            vehicle,
            "heading_theta",
            self._resolve_attr(vehicle, "heading", 0.0),
        )
        speed = self._resolve_attr(vehicle, "speed", default_speed)
        length = self._resolve_attr(vehicle, "LENGTH", self.config.vehicle_length)
        width = self._resolve_attr(vehicle, "WIDTH", self.config.vehicle_width)
        def _bool_attr(name: str, default: bool = False) -> bool:
            value = self._resolve_attr(vehicle, name, default)
            return bool(default if value is None else value)

        return {
            "x": float(x),
            "y": float(y),
            "heading": self._safe_float(heading, 0.0),
            "speed": max(0.0, self._safe_float(speed, default_speed)),
            "length": self._safe_float(length, self.config.vehicle_length),
            "width": self._safe_float(width, self.config.vehicle_width),
            "lane_id": self._metadrive_lane_id(vehicle),
            "dist_to_left_side": self._safe_float(self._resolve_attr(vehicle, "dist_to_left_side", math.nan), math.nan),
            "dist_to_right_side": self._safe_float(self._resolve_attr(vehicle, "dist_to_right_side", math.nan), math.nan),
            "on_lane": _bool_attr("on_lane", True),
            "out_of_route": _bool_attr("out_of_route", False),
            "crash_sidewalk": _bool_attr("crash_sidewalk", False),
            "on_yellow_continuous_line": _bool_attr("on_yellow_continuous_line", False),
            "on_white_continuous_line": _bool_attr("on_white_continuous_line", False),
            "on_broken_line": _bool_attr("on_broken_line", False),
        }

    def _metadrive_object_to_dict(self, obj: Any, default_speed: float = 0.0) -> Dict[str, Any]:
        parsed = self._metadrive_vehicle_to_dict(obj, default_speed=default_speed)
        class_name = type(obj).__name__
        has_vehicle_dynamics = hasattr(obj, "speed") or hasattr(obj, "before_step") or hasattr(obj, "set_velocity")
        object_type = self._metadrive_object_type(obj, class_name, has_vehicle_dynamics)
        parsed.update(
            {
                "object_type": object_type,
                "class_name": class_name,
                "object_id": self._resolve_attr(obj, "id", self._resolve_attr(obj, "name", "")),
            }
        )
        if object_type != "vehicle":
            parsed["speed"] = 0.0
        return parsed

    def _metadrive_object_type(self, obj: Any, class_name: str, has_vehicle_dynamics: bool) -> str:
        raw_type = self._resolve_attr(obj, "object_type", None)
        if raw_type is None:
            raw_type = self._resolve_attr(obj, "type", None)
        if raw_type is None:
            raw_type = self._resolve_attr(obj, "TYPE", None)

        normalized = self._normalize_object_type(raw_type)
        known_types = {"vehicle", "traffic_object", "obstacle", "cone", "barrier", "static_obstacle"}
        if normalized in known_types:
            return normalized
        if normalized:
            return normalized
        if has_vehicle_dynamics:
            return "vehicle"

        lowered_class = str(class_name).lower()
        if "vehicle" in lowered_class or "trafficparticipant" in lowered_class:
            return "vehicle"
        if "cone" in lowered_class:
            return "cone"
        if "barrier" in lowered_class:
            return "barrier"
        if "trafficobject" in lowered_class or "traffic_object" in lowered_class:
            return "traffic_object"
        if "staticobstacle" in lowered_class or "static_obstacle" in lowered_class:
            return "static_obstacle"
        if "obstacle" in lowered_class:
            return "obstacle"
        return normalized or "unknown"

    def _normalize_object_type(self, value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            value = getattr(value, "name", value)
        text = str(value).strip().lower()
        text = text.replace("-", "_").replace(" ", "_")
        if "." in text:
            text = text.rsplit(".", 1)[-1]
        aliases = {
            "trafficobject": "traffic_object",
            "traffic_cone": "cone",
            "trafficcone": "cone",
            "staticobject": "static_obstacle",
            "static_object": "static_obstacle",
            "vehicleobject": "vehicle",
        }
        return aliases.get(text, text)

    def _metadrive_lane_id(self, vehicle: Any) -> Optional[Any]:
        navigation = self._resolve_attr(vehicle, "navigation", None)
        if navigation is not None:
            current_road = self._resolve_attr(navigation, "current_road", None)
            if current_road is not None:
                return (
                    self._resolve_attr(current_road, "start_node", None),
                    self._resolve_attr(current_road, "end_node", None),
                )
            current_lanes = self._resolve_attr(navigation, "current_ref_lanes", None)
            if current_lanes:
                try:
                    return self._resolve_attr(current_lanes[0], "index", None)
                except Exception:
                    pass
        lane = self._resolve_attr(vehicle, "lane", None)
        return self._resolve_attr(lane, "index", None) if lane is not None else None

    def _metadrive_current_lane_width(self, vehicle: Any) -> float:
        lane = self._metadrive_current_lane(vehicle)
        if lane is not None:
            width = self._resolve_attr(lane, "width", None)
            if width is not None:
                try:
                    return float(width)
                except (TypeError, ValueError):
                    pass
        return self.config.default_lane_width

    def _metadrive_current_lane(self, vehicle: Any) -> Optional[Any]:
        navigation = self._resolve_attr(vehicle, "navigation", None)
        if navigation is not None:
            current_lanes = self._resolve_attr(navigation, "current_ref_lanes", None)
            if current_lanes:
                try:
                    return current_lanes[0]
                except Exception:
                    pass
            current_lane = self._resolve_attr(navigation, "current_lane", None)
            if current_lane is not None:
                return current_lane
        return self._resolve_attr(vehicle, "lane", None)

    def _metadrive_lane_boundary_info(self, vehicle: Any) -> Dict[str, Any]:
        lane = self._metadrive_current_lane(vehicle)

        def _enum_name(value: Any) -> str:
            if value is None:
                return ""
            name = getattr(value, "name", None)
            if name is not None:
                return str(name)
            return str(value)

        def _item(values: Any, index: int) -> Any:
            if values is None:
                return None
            try:
                return values[index]
            except Exception:
                return None

        def _prohibited(line_type_name: str) -> bool:
            upper = line_type_name.upper()
            return "SOLID" in upper or "CONTINUOUS" in upper or "SIDE" in upper or "GUARDRAIL" in upper

        line_types = self._resolve_attr(lane, "line_types", None) if lane is not None else None
        line_colors = self._resolve_attr(lane, "line_colors", None) if lane is not None else None
        left_type = _enum_name(_item(line_types, 0))
        right_type = _enum_name(_item(line_types, 1))
        left_color = _enum_name(_item(line_colors, 0))
        right_color = _enum_name(_item(line_colors, 1))
        return {
            "left_lane_line_type": left_type,
            "right_lane_line_type": right_type,
            "left_lane_line_color": left_color,
            "right_lane_line_color": right_color,
            "left_lane_line_prohibited": _prohibited(left_type),
            "right_lane_line_prohibited": _prohibited(right_type),
        }

    def _metadrive_lanes(self, vehicle: Any) -> Dict[str, Dict[str, Any]]:
        current_lane = self._metadrive_current_lane(vehicle)
        current_width = self._metadrive_current_lane_width(vehicle)
        lanes = {
            "current": {
                "lane_id": self._resolve_attr(current_lane, "index", None) if current_lane is not None else None,
                "width": current_width,
                "available": True,
                "drivable": True,
            }
        }

        left_lane = self._metadrive_side_lane(vehicle, +1)
        right_lane = self._metadrive_side_lane(vehicle, -1)

        if left_lane is not None or self.config.metadrive_assume_adjacent_lanes:
            lanes["left"] = {
                "lane_id": self._resolve_attr(left_lane, "index", "left") if left_lane is not None else "left",
                "width": self._safe_float(
                    self._resolve_attr(left_lane, "width", current_width) if left_lane is not None else current_width,
                    current_width,
                ),
                "available": True,
                "drivable": True,
            }
        if right_lane is not None or self.config.metadrive_assume_adjacent_lanes:
            lanes["right"] = {
                "lane_id": self._resolve_attr(right_lane, "index", "right") if right_lane is not None else "right",
                "width": self._safe_float(
                    self._resolve_attr(right_lane, "width", current_width) if right_lane is not None else current_width,
                    current_width,
                ),
                "available": True,
                "drivable": True,
            }
        return lanes

    def _metadrive_route_corridors(self, vehicle: Any) -> List[Dict[str, Any]]:
        navigation = self._resolve_attr(vehicle, "navigation", None)
        if navigation is None:
            return []

        corridors: List[Dict[str, Any]] = []

        def add_group(source: str, lanes: Any) -> None:
            if not lanes:
                return
            try:
                lanes_list = list(lanes)
            except Exception:
                return
            if not lanes_list:
                return

            ref_lane = lanes_list[0]
            lane_width = self._safe_float(self._resolve_attr(ref_lane, "width", self.config.default_lane_width), self.config.default_lane_width)
            length = self._safe_float(self._resolve_attr(ref_lane, "length", 0.0), 0.0)
            if length <= self.config.small_tolerance:
                return

            reference_vehicle_longitudinal = math.nan
            reference_vehicle_lateral = math.nan
            vehicle_position = self._resolve_attr(vehicle, "position", None)
            if vehicle_position is not None:
                try:
                    lon, lat = ref_lane.local_coordinates(vehicle_position)
                    reference_vehicle_longitudinal = float(lon)
                    reference_vehicle_lateral = float(lat)
                except Exception:
                    pass

            total_width = 0.0
            for lane in lanes_list:
                total_width += self._safe_float(self._resolve_attr(lane, "width", lane_width), lane_width)
            total_width = max(lane_width, total_width)

            sample_count = max(2, min(80, int(math.ceil(length / 2.0)) + 1))
            points: List[List[float]] = []
            for s in np.linspace(0.0, length, sample_count):
                try:
                    point = ref_lane.position(float(s), 0.0)
                    points.append([float(point[0]), float(point[1])])
                except Exception:
                    return
            corridors.append(
                {
                    "source": source,
                    "points": points,
                    "reference_lane_width": float(lane_width),
                    "total_width": float(total_width),
                    "length": float(length),
                    "lane_count": len(lanes_list),
                    "reference_vehicle_longitudinal": reference_vehicle_longitudinal,
                    "reference_vehicle_lateral": reference_vehicle_lateral,
                }
            )

        add_group("current_ref_lanes", self._resolve_attr(navigation, "current_ref_lanes", None))
        add_group("next_ref_lanes", self._resolve_attr(navigation, "next_ref_lanes", None))
        return corridors

    def _metadrive_side_lane(self, vehicle: Any, lane_offset: int) -> Optional[Any]:
        current_lane = self._metadrive_current_lane(vehicle)
        lane_index = self._resolve_attr(current_lane, "index", None) if current_lane is not None else None
        if not isinstance(lane_index, tuple) or len(lane_index) < 3:
            return None

        side_index = list(lane_index)
        if not isinstance(side_index[-1], int):
            return None
        side_index[-1] += lane_offset
        side_index = tuple(side_index)

        navigation = self._resolve_attr(vehicle, "navigation", None)
        road_network = None
        if navigation is not None:
            nav_map = self._resolve_attr(navigation, "map", None)
            road_network = self._resolve_attr(nav_map, "road_network", None) if nav_map is not None else None
        if road_network is None:
            return None

        try:
            return road_network.get_lane(side_index)
        except Exception:
            return None

    def _collect_metadrive_objects(self, raw_env: Any) -> List[Any]:
        engine = self._resolve_attr(raw_env, "engine", None)
        if engine is None:
            return []

        objects = []
        seen = set()

        def add_item(item: Any) -> None:
            if item is None or id(item) in seen:
                return
            try:
                self._metadrive_position_xy(item)
            except Exception:
                return
            seen.add(id(item))
            objects.append(item)

        def add_container(container: Any) -> None:
            if container is None:
                return
            if isinstance(container, dict):
                iterable = container.values()
            elif isinstance(container, (list, tuple, set)):
                iterable = container
            else:
                iterable = None
            if iterable is not None:
                for item in iterable:
                    add_item(item)

        object_attrs = (
            "vehicles",
            "traffic_vehicles",
            "_vehicles",
            "objects",
            "_objects",
            "spawned_objects",
            "_spawned_objects",
            "static_objects",
            "_static_objects",
            "traffic_objects",
            "_traffic_objects",
            "agents",
            "_agents",
            "nodes",
            "_nodes",
        )

        for attr in (
            *object_attrs,
        ):
            add_container(self._resolve_attr(engine, attr, None))

        for manager_name in (
            "traffic_manager",
            "object_manager",
            "agent_manager",
            "map_manager",
            "spawn_manager",
            "vehicle_manager",
            "static_object_manager",
        ):
            manager = self._resolve_attr(engine, manager_name, None)
            if manager is None:
                continue
            for attr in object_attrs:
                add_container(self._resolve_attr(manager, attr, None))

        managers = self._resolve_attr(engine, "managers", None)
        if isinstance(managers, dict):
            for manager in managers.values():
                for attr in object_attrs:
                    add_container(getattr(manager, attr, None))
                try:
                    manager_values = vars(manager).values()
                except TypeError:
                    manager_values = []
                for value in manager_values:
                    add_container(value)

        try:
            engine_values = vars(engine).values()
        except TypeError:
            engine_values = []
        for value in engine_values:
            add_container(value)

        return objects

    def _distance_xy(self, first: Dict[str, Any], second: Dict[str, Any]) -> float:
        return math.sqrt(
            (float(first.get("x", 0.0)) - float(second.get("x", 0.0))) ** 2
            + (float(first.get("y", 0.0)) - float(second.get("y", 0.0))) ** 2
        )
