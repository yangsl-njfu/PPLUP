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
    enable_predictive_clearance_guard: bool = True
    enable_bypass: bool = True
    enforce_intervention_margin: bool = False
    intervention_margin_threshold: float = 0.0
    metadrive_steer_sign: float = 1.0


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
            return u_original, {
                "mode": "normal",
                "obstacle_detected": False,
                "dynamic_vehicle_detected": False,
                "left_feasible": False,
                "right_feasible": False,
                "state_debug": self._state_debug(state),
                "candidates": [],
            }

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
            return u_original, {
                "mode": "normal",
                "obstacle_detected": False,
                "dynamic_vehicle_detected": True,
                "state_debug": self._state_debug(state),
                "candidates": [],
            }

        obstacle, d_obs = obstacle_info
        ego_speed = self._ego_speed(state)
        d_brake = self.compute_brake_distance(ego_speed)
        current_rss_margin = d_obs - d_brake

        if (
            self.config.enforce_intervention_margin
            and current_rss_margin > self.config.intervention_margin_threshold
        ):
            if clearance_info is not None:
                return self._filter_predictive_clearance_risk(state, clearance_info, u_original)
            return u_original, {
                "mode": "normal",
                "obstacle_detected": True,
                "d_obs": d_obs,
                "d_brake": d_brake,
                "rss_margin": current_rss_margin,
                "left_feasible": False,
                "right_feasible": False,
                "state_debug": self._state_debug(state),
                "blocking_object": self._object_debug(obstacle, state),
                "candidates": [],
                "reason": "rss_margin_positive_no_intervention",
            }

        candidates: List[StaticRSSCandidate] = []

        u_stop_nom = self.generate_stop_nominal_action(state)
        u_stop_proj, stop_project_debug = self.ray_project_action(state, obstacle, u_stop_nom, "stop")
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

        u_safe, selected_info = self.select_action(candidates, u_original)
        candidate_debug = [asdict(candidate) for candidate in candidates]

        return u_safe, {
            "mode": selected_info["mode"],
            "obstacle_detected": True,
            "dynamic_vehicle_detected": dynamic_info is not None,
            "d_obs": d_obs,
            "d_brake": d_brake,
            "rss_margin": current_rss_margin,
            "left_feasible": left_feasible,
            "right_feasible": right_feasible,
            "state_debug": self._state_debug(state),
            "blocking_object": self._object_debug(obstacle, state),
            "selected_score": selected_info.get("selected_score"),
            "projection_debug": selected_info.get("projection_debug", {}),
            "selected": selected_info,
            "candidates": candidate_debug,
        }

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
        self, state: State, obstacle: Dict[str, Any], u_nom: Sequence[float], mode: str
    ) -> Tuple[Action, Dict[str, Any]]:
        """Project an action to a mode-specific safe set via ray search.

        The ray starts at a conservative safe center ``c`` and points toward
        ``u_nom``. The largest safe lambda in ``[0, 1]`` is selected.
        """
        u_nom = self._clip_action(u_nom)
        center = self._safe_center_for_mode(mode)
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
        fallback = self._clip_action([self.config.min_acc, 0.0])

        if not safe_candidates:
            return fallback, {
                "mode": "fallback_stop",
                "action": fallback,
                "selected_score": None,
                "projection_debug": {"reason": "no_safe_candidate"},
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
        vehicle = getattr(raw_env, "vehicle", None)
        if vehicle is None and hasattr(raw_env, "agent"):
            vehicle = getattr(raw_env, "agent", None)
        if vehicle is None:
            raise ValueError("Cannot parse MetaDrive state: env has no vehicle/agent attribute.")

        ego = self._metadrive_vehicle_to_dict(vehicle, default_speed=0.0)
        ego["lane_width"] = self._metadrive_current_lane_width(vehicle)
        ego["lane_id"] = self._metadrive_lane_id(vehicle)

        lanes = self._metadrive_lanes(vehicle)
        objects = self._collect_metadrive_objects(raw_env)
        static_obstacles = []
        vehicles = []

        for obj in objects:
            if obj is vehicle:
                continue
            parsed = self._metadrive_object_to_dict(obj, default_speed=0.0)
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

            if parsed.get("object_type") != "vehicle" or parsed.get("speed", 0.0) <= self.config.metadrive_static_speed_threshold:
                static_obstacles.append(parsed)
            else:
                vehicles.append(parsed)

        return {
            "ego": ego,
            "static_obstacles": static_obstacles,
            "vehicles": vehicles,
            "lanes": lanes,
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

        augmented = copy.deepcopy(state)
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

    def _check_stop_horizon(
        self, state: State, obstacle: Dict[str, Any], action: Sequence[float]
    ) -> Tuple[bool, Dict[str, float]]:
        min_margin = self._stop_margin(state, obstacle)
        rollout_state = state
        for _ in range(self.config.horizon_steps):
            rollout_state = self._simulate_next_state(rollout_state, action)
            min_margin = min(min_margin, self._stop_margin(rollout_state, obstacle))
        return min_margin >= -self.config.small_tolerance, {"h_stop_min": min_margin}

    def _check_dynamic_front_horizon(
        self, state: State, vehicle: Dict[str, Any], action: Sequence[float]
    ) -> Tuple[bool, Dict[str, float]]:
        min_margin = self._dynamic_front_margin(state, vehicle)
        rollout_state = state
        for _ in range(self.config.horizon_steps):
            rollout_state = self._simulate_next_state(rollout_state, action)
            min_margin = min(min_margin, self._dynamic_front_margin(rollout_state, vehicle))
        return min_margin >= -self.config.small_tolerance, {"dynamic_front_margin_min": min_margin}

    def _check_clearance_horizon(
        self, state: State, obj: Dict[str, Any], action: Sequence[float]
    ) -> Tuple[bool, Dict[str, float]]:
        min_margin = self._obstacle_clearance_margin(state, obj)
        rollout_state = state
        for _ in range(self.config.horizon_steps):
            rollout_state = self._simulate_next_state(rollout_state, action)
            min_margin = min(min_margin, self._obstacle_clearance_margin(rollout_state, obj))
        return min_margin >= -self.config.small_tolerance, {"clearance_margin_min": min_margin}

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
        u_stop_nom = self.generate_stop_nominal_action(state)
        u_stop_proj, project_debug = self.ray_project_action(state, vehicle, u_stop_nom, "dynamic_stop")
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
        return u_safe, {
            "mode": selected_info["mode"],
            "obstacle_detected": False,
            "dynamic_vehicle_detected": True,
            "d_front": d_front,
            "d_dynamic": d_dynamic,
            "rss_margin": dynamic_margin,
            "left_feasible": False,
            "right_feasible": False,
            "state_debug": self._state_debug(state),
            "blocking_object": self._object_debug(vehicle, state),
            "selected_score": selected_info.get("selected_score"),
            "projection_debug": selected_info.get("projection_debug", {}),
            "selected": selected_info,
            "candidates": [asdict(candidate)],
        }

    def _filter_predictive_clearance_risk(
        self, state: State, clearance_info: Dict[str, Any], u_original: Action
    ) -> Tuple[Action, Dict[str, Any]]:
        obj = clearance_info["object"]
        u_stop_nom = self.generate_stop_nominal_action(state)
        u_stop_proj, project_debug = self.ray_project_action(state, obj, u_stop_nom, "clearance_stop")
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
        return u_safe, {
            "mode": selected_info["mode"],
            "obstacle_detected": obj in state.get("static_obstacles", []),
            "dynamic_vehicle_detected": obj in state.get("vehicles", []),
            "clearance_margin": clearance_info.get("clearance_margin"),
            "risk_step": clearance_info.get("risk_step"),
            "rss_margin": clearance_info.get("clearance_margin"),
            "left_feasible": False,
            "right_feasible": False,
            "state_debug": self._state_debug(state),
            "blocking_object": self._object_debug(obj, state),
            "selected_score": selected_info.get("selected_score"),
            "projection_debug": selected_info.get("projection_debug", {}),
            "selected": selected_info,
            "candidates": [asdict(candidate)],
        }

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
            "speed": obj.get("speed", float("nan")),
            "relative_lane": obj.get("relative_lane", ""),
            "longitudinal": longitudinal,
            "lateral": lateral,
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
        next_state = copy.deepcopy(state)
        ego = copy.deepcopy(self._ego(state))

        x = float(ego.get("x", 0.0))
        y = float(ego.get("y", 0.0))
        heading = float(ego.get("heading", 0.0))
        speed = max(0.0, float(ego.get("speed", 0.0)))

        ego["x"] = x + speed * math.cos(heading) * cfg.dt
        ego["y"] = y + speed * math.sin(heading) * cfg.dt
        ego["speed"] = max(0.0, min(cfg.v_max, speed + acc * cfg.dt))
        ego["heading"] = heading + steer * cfg.steer_gain * cfg.dt
        next_state["ego"] = ego
        return next_state

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

    def _relative_position(self, ego: Dict[str, Any], obj: Dict[str, Any]) -> Tuple[float, float]:
        dx = float(obj.get("x", 0.0)) - float(ego.get("x", 0.0))
        dy = float(obj.get("y", 0.0)) - float(ego.get("y", 0.0))
        heading = float(ego.get("heading", 0.0))
        longitudinal = dx * math.cos(heading) + dy * math.sin(heading)
        lateral = -dx * math.sin(heading) + dy * math.cos(heading)
        return longitudinal, lateral

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

    def _unwrap_env(self, env: Any) -> Any:
        raw_env = env
        seen = set()
        while hasattr(raw_env, "env") and id(raw_env) not in seen:
            seen.add(id(raw_env))
            raw_env = raw_env.env
        return raw_env

    def _metadrive_vehicle_to_dict(self, vehicle: Any, default_speed: float = 0.0) -> Dict[str, Any]:
        position = getattr(vehicle, "position", (0.0, 0.0))
        heading = getattr(vehicle, "heading_theta", getattr(vehicle, "heading", 0.0))
        speed = getattr(vehicle, "speed", default_speed)
        length = getattr(vehicle, "LENGTH", self.config.vehicle_length)
        width = getattr(vehicle, "WIDTH", self.config.vehicle_width)
        return {
            "x": float(position[0]),
            "y": float(position[1]),
            "heading": float(heading),
            "speed": max(0.0, float(speed)),
            "length": float(length),
            "width": float(width),
            "lane_id": self._metadrive_lane_id(vehicle),
        }

    def _metadrive_object_to_dict(self, obj: Any, default_speed: float = 0.0) -> Dict[str, Any]:
        parsed = self._metadrive_vehicle_to_dict(obj, default_speed=default_speed)
        class_name = type(obj).__name__
        has_vehicle_dynamics = hasattr(obj, "speed") or hasattr(obj, "before_step") or hasattr(obj, "set_velocity")
        parsed.update(
            {
                "object_type": "vehicle" if has_vehicle_dynamics else "object",
                "class_name": class_name,
                "object_id": getattr(obj, "id", getattr(obj, "name", "")),
            }
        )
        if not has_vehicle_dynamics:
            parsed["speed"] = 0.0
        return parsed

    def _metadrive_lane_id(self, vehicle: Any) -> Optional[Any]:
        navigation = getattr(vehicle, "navigation", None)
        if navigation is not None:
            current_road = getattr(navigation, "current_road", None)
            if current_road is not None:
                return (
                    getattr(current_road, "start_node", None),
                    getattr(current_road, "end_node", None),
                )
            current_lanes = getattr(navigation, "current_ref_lanes", None)
            if current_lanes:
                return getattr(current_lanes[0], "index", None)
        lane = getattr(vehicle, "lane", None)
        return getattr(lane, "index", None)

    def _metadrive_current_lane_width(self, vehicle: Any) -> float:
        lane = self._metadrive_current_lane(vehicle)
        if lane is not None:
            width = getattr(lane, "width", None)
            if width is not None:
                try:
                    return float(width)
                except (TypeError, ValueError):
                    pass
        return self.config.default_lane_width

    def _metadrive_current_lane(self, vehicle: Any) -> Optional[Any]:
        navigation = getattr(vehicle, "navigation", None)
        if navigation is not None:
            current_lanes = getattr(navigation, "current_ref_lanes", None)
            if current_lanes:
                return current_lanes[0]
        return getattr(vehicle, "lane", None)

    def _metadrive_lanes(self, vehicle: Any) -> Dict[str, Dict[str, Any]]:
        current_lane = self._metadrive_current_lane(vehicle)
        current_width = self._metadrive_current_lane_width(vehicle)
        lanes = {
            "current": {
                "lane_id": getattr(current_lane, "index", None),
                "width": current_width,
                "available": True,
                "drivable": True,
            }
        }

        left_lane = self._metadrive_side_lane(vehicle, +1)
        right_lane = self._metadrive_side_lane(vehicle, -1)

        if left_lane is not None or self.config.metadrive_assume_adjacent_lanes:
            lanes["left"] = {
                "lane_id": getattr(left_lane, "index", "left"),
                "width": float(getattr(left_lane, "width", current_width)),
                "available": True,
                "drivable": True,
            }
        if right_lane is not None or self.config.metadrive_assume_adjacent_lanes:
            lanes["right"] = {
                "lane_id": getattr(right_lane, "index", "right"),
                "width": float(getattr(right_lane, "width", current_width)),
                "available": True,
                "drivable": True,
            }
        return lanes

    def _metadrive_side_lane(self, vehicle: Any, lane_offset: int) -> Optional[Any]:
        current_lane = self._metadrive_current_lane(vehicle)
        lane_index = getattr(current_lane, "index", None)
        if not isinstance(lane_index, tuple) or len(lane_index) < 3:
            return None

        side_index = list(lane_index)
        if not isinstance(side_index[-1], int):
            return None
        side_index[-1] += lane_offset
        side_index = tuple(side_index)

        navigation = getattr(vehicle, "navigation", None)
        road_network = None
        if navigation is not None:
            nav_map = getattr(navigation, "map", None)
            road_network = getattr(nav_map, "road_network", None)
        if road_network is None:
            return None

        try:
            return road_network.get_lane(side_index)
        except Exception:
            return None

    def _collect_metadrive_objects(self, raw_env: Any) -> List[Any]:
        engine = getattr(raw_env, "engine", None)
        if engine is None:
            return []

        objects = []
        seen = set()

        def add_item(item: Any) -> None:
            if item is None or id(item) in seen:
                return
            if not hasattr(item, "position"):
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
            add_container(getattr(engine, attr, None))

        for manager_name in (
            "traffic_manager",
            "object_manager",
            "agent_manager",
            "map_manager",
            "spawn_manager",
            "vehicle_manager",
            "static_object_manager",
        ):
            manager = getattr(engine, manager_name, None)
            if manager is None:
                continue
            for attr in object_attrs:
                add_container(getattr(manager, attr, None))

        managers = getattr(engine, "managers", None)
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
