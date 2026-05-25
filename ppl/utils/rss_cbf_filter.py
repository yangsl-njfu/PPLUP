"""RSS-CBF runtime assurance filter for PPL/TD3 policy evaluation.

The public runtime path is intentionally small:

    u_nom -> RSSCBF Runtime Assurance -> u_safe

The filter keeps the existing MetaDrive adapters and action conversion helpers
from ``StaticRSSFilter``, but exposes only the RSS-CBF forward-distance safety
contract for evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ppl.utils.static_rss_filter import Action, State, StaticRSSConfig, StaticRSSFilter


@dataclass
class RSSCBFConfig(StaticRSSConfig):
    """Stable defaults for RSS-CBF runtime assurance evaluation."""

    enable_predictive_clearance_guard: bool = False
    preserve_steer_on_stop: bool = True
    fallback_to_brake: bool = False
    enable_recovery_mode: bool = True
    intervention_margin_threshold: float = 0.0

    enable_lateral_rss: bool = True
    lateral_rss_min_clearance: float = 0.3
    lateral_rss_response_time: float = 0.1
    lateral_rss_max_lateral_speed: float = 3.0
    lateral_rss_min_lateral_decel: float = 2.0
    certified_lateral_escape_margin_buffer: float = 1.0

    # RSS-CBF evaluation does not expose bypass or clearance-guard experiments.
    enable_bypass: bool = False
    allow_bypass_before_rss_violation: bool = False
    enforce_intervention_margin: bool = False


class RSSCBFFilter(StaticRSSFilter):
    """Forward RSS-CBF runtime assurance filter.

    The filter primarily changes longitudinal acceleration and preserves the
    policy steering command by default. It returns only the formal evaluation
    modes: ``normal``, ``rss_cbf_intervention``, ``rss_cbf_recovery``, and
    ``fallback_no_safe_candidate``.
    """

    FORMAL_MODES = {
        "normal",
        "rss_cbf_intervention",
        "rss_cbf_recovery",
        "fallback_no_safe_candidate",
    }

    def __init__(self, config: Optional[RSSCBFConfig] = None):
        super().__init__(config or RSSCBFConfig())

    def filter_action(self, state: State, u_nom: Sequence[float]) -> Tuple[Action, Dict[str, Any]]:
        """Return the minimally adjusted safe internal action and diagnostics."""
        u_original = self._clip_action(u_nom)
        front = self._select_front_rss_object(state)

        if front is None:
            return u_original, self._make_rss_cbf_info(
                state=state,
                obj=None,
                object_kind="none",
                mode="normal",
                reason="no_front_rss_object",
                u_original=u_original,
                u_safe=u_original,
                d_front=math.inf,
                d_rss=0.0,
                rss_margin=math.inf,
                dynamic_vehicle_detected=False,
                obstacle_detected=False,
                nominal_margins={},
                projection_debug={},
            )

        object_kind, obj, d_front, d_rss, rss_margin, dynamic_detected, static_detected = front
        nominal_margins = self._rss_cbf_horizon_margins(state, obj, object_kind, u_original)

        if self._rss_cbf_nominal_is_safe(rss_margin, nominal_margins):
            return u_original, self._make_rss_cbf_info(
                state=state,
                obj=obj,
                object_kind=object_kind,
                mode="normal",
                reason="nominal_keeps_rss_margin",
                u_original=u_original,
                u_safe=u_original,
                d_front=d_front,
                d_rss=d_rss,
                rss_margin=rss_margin,
                dynamic_vehicle_detected=dynamic_detected,
                obstacle_detected=static_detected,
                nominal_margins=nominal_margins,
                projection_debug={},
            )

        u_safe, projection_debug = self._project_acceleration_for_rss_cbf(
            state, obj, object_kind, u_original, rss_margin
        )
        if u_safe is None:
            if self.config.fallback_to_brake:
                u_safe = self._clip_action([self.config.min_acc, u_original[1]])
                mode = "rss_cbf_intervention"
                reason = "no_safe_candidate_brake_fallback"
            else:
                u_safe = u_original
                mode = "fallback_no_safe_candidate"
                reason = "no_safe_candidate_keep_nominal"
        else:
            mode = "rss_cbf_recovery" if rss_margin < -self.config.small_tolerance else "rss_cbf_intervention"
            reason = "rss_margin_negative_recovery" if mode == "rss_cbf_recovery" else "nominal_worsens_rss_margin"

        return u_safe, self._make_rss_cbf_info(
            state=state,
            obj=obj,
            object_kind=object_kind,
            mode=mode,
            reason=reason,
            u_original=u_original,
            u_safe=u_safe,
            d_front=d_front,
            d_rss=d_rss,
            rss_margin=rss_margin,
            dynamic_vehicle_detected=dynamic_detected,
            obstacle_detected=static_detected,
            nominal_margins=nominal_margins,
            projection_debug=projection_debug,
        )

    def _select_front_rss_object(
        self, state: State
    ) -> Optional[Tuple[str, Dict[str, Any], float, float, float, bool, bool]]:
        dynamic_info = self.detect_dynamic_vehicle_ahead(state)
        static_info = self.detect_static_obstacle_ahead(state)
        candidates = []

        if dynamic_info is not None:
            vehicle, d_front = dynamic_info
            d_rss = self._rss_cbf_distance(state, vehicle, "dynamic")
            candidates.append(("dynamic", vehicle, d_front, d_rss, d_front - d_rss))

        if static_info is not None:
            obstacle, d_front = static_info
            d_rss = self._rss_cbf_distance(state, obstacle, "static")
            candidates.append(("static", obstacle, d_front, d_rss, d_front - d_rss))

        if not candidates:
            return None

        object_kind, obj, d_front, d_rss, rss_margin = min(candidates, key=lambda item: item[4])
        return (
            object_kind,
            obj,
            d_front,
            d_rss,
            rss_margin,
            dynamic_info is not None,
            static_info is not None,
        )

    def _rss_cbf_distance(self, state: State, obj: Dict[str, Any], object_kind: str) -> float:
        margin = (
            self.config.adjacent_vehicle_margin
            if object_kind == "dynamic"
            else self.config.vehicle_length / 2.0 + self.config.obstacle_margin
        )
        return self.compute_rss_distance(
            self._ego_speed(state),
            front_speed=float(obj.get("speed", 0.0)),
            margin=margin,
        )

    def _rss_cbf_margin(self, state: State, obj: Dict[str, Any], object_kind: str) -> float:
        return self._distance_to_obstacle_front(state, obj) - self._rss_cbf_distance(state, obj, object_kind)

    def _rss_cbf_horizon_margins(
        self,
        state: State,
        obj: Dict[str, Any],
        object_kind: str,
        action: Sequence[float],
    ) -> Dict[str, float]:
        _, margins = self._check_margin_horizon(
            state=state,
            action=action,
            margin_fn=lambda rollout_state: self._rss_cbf_margin(rollout_state, obj, object_kind),
            min_margin_key="rss_margin_min",
        )
        return margins

    def _rss_cbf_nominal_is_safe(self, current_margin: float, margins: Dict[str, float]) -> bool:
        if current_margin < -self.config.small_tolerance:
            return False
        if not margins:
            return True
        final_margin = float(margins.get("final_margin", current_margin))
        min_margin = float(margins.get("min_margin", current_margin))
        return (
            min_margin >= -self.config.small_tolerance
            and final_margin + self.config.small_tolerance >= current_margin
        )

    def _rss_cbf_candidate_is_safe(
        self,
        current_margin: float,
        margins: Dict[str, float],
        action: Sequence[float],
    ) -> bool:
        final_margin = float(margins.get("final_margin", current_margin))
        min_margin = float(margins.get("min_margin", current_margin))

        if current_margin >= -self.config.small_tolerance:
            return (
                min_margin >= -self.config.small_tolerance
                and final_margin + self.config.small_tolerance >= current_margin
            )

        if not self.config.enable_recovery_mode:
            return min_margin >= -self.config.small_tolerance

        if self._clip_action(action)[0] > self.config.small_tolerance:
            return False

        margin_improvement = float(margins.get("margin_improvement", final_margin - current_margin))
        improves_enough = margin_improvement + self.config.small_tolerance >= self.config.recovery_margin_improvement
        holds_margin = self.config.recovery_allow_equal_margin and final_margin + self.config.small_tolerance >= current_margin
        return improves_enough or holds_margin

    def _project_acceleration_for_rss_cbf(
        self,
        state: State,
        obj: Dict[str, Any],
        object_kind: str,
        u_original: Action,
        current_margin: float,
    ) -> Tuple[Optional[Action], Dict[str, Any]]:
        ray_steps = max(1, int(self.config.ray_steps))
        start_acc = float(u_original[0])
        candidate_accs = np.linspace(start_acc, self.config.min_acc, ray_steps + 1)
        candidates = []

        for acc in candidate_accs:
            action = self._clip_action([float(acc), u_original[1]])
            margins = self._rss_cbf_horizon_margins(state, obj, object_kind, action)
            safe = self._rss_cbf_candidate_is_safe(current_margin, margins, action)
            candidate = {
                "mode": "rss_cbf_recovery" if current_margin < -self.config.small_tolerance else "rss_cbf_intervention",
                "action": action,
                "safe": safe,
                "margins": margins,
                "intervention_cost": self._action_distance_sq(action, u_original),
                "progress_score": self._score_progress_after_rollout(state, action),
            }
            candidates.append(candidate)
            if safe:
                return action, {
                    "projection_failed": False,
                    "candidate_count": len(candidates),
                    "selected": candidate,
                    "margins": margins,
                    "candidates": candidates,
                }

        return None, {
            "projection_failed": True,
            "candidate_count": len(candidates),
            "candidates": candidates,
        }

    def compute_lateral_rss_safe_distance(self, lateral_speed: float = 0.0) -> float:
        cfg = self.config
        rho = max(0.0, float(cfg.lateral_rss_response_time))
        v_lat = max(0.0, abs(float(lateral_speed)))
        v_response = v_lat + cfg.lateral_rss_max_lateral_speed * rho
        braking = v_response ** 2 / (2.0 * max(0.1, cfg.lateral_rss_min_lateral_decel))
        return max(0.0, v_lat * rho + braking + cfg.lateral_rss_min_clearance)

    def compute_lateral_rss_margin(
        self,
        state: State,
        obj: Dict[str, Any],
    ) -> float:
        ego = self._ego(state)
        _, lateral = self._relative_position(ego, obj)
        lateral_distance = abs(lateral)
        half_ego = self.config.vehicle_width / 2.0
        half_obj = self._object_width(obj, self.config.vehicle_width) / 2.0
        lateral_gap = lateral_distance - half_ego - half_obj
        lateral_speed = float(ego.get("lateral_speed", 0.0))
        safe_distance = self.compute_lateral_rss_safe_distance(lateral_speed)
        return lateral_gap - safe_distance

    def check_path_overlap(
        self,
        state: State,
        obj: Dict[str, Any],
    ) -> Tuple[bool, float, float]:
        ego = self._ego(state)
        _, lateral = self._relative_position(ego, obj)
        half_ego = self.config.vehicle_width / 2.0 + self.config.obstacle_margin
        half_obj = self._object_width(obj, self.config.vehicle_width) / 2.0
        overlap = abs(lateral) < (half_ego + half_obj)
        lateral_distance = abs(lateral)
        lateral_margin = lateral_distance - half_ego - half_obj
        return bool(overlap), float(lateral_distance), float(lateral_margin)

    def check_lateral_deconflicted(
        self,
        state: State,
        obj: Dict[str, Any],
        lateral_rss_margin: float,
    ) -> bool:
        if lateral_rss_margin >= 0.0:
            return True
        _, lateral = self._relative_position(self._ego(state), obj)
        half_ego = self.config.vehicle_width / 2.0 + self.config.obstacle_margin
        half_obj = self._object_width(obj, self.config.vehicle_width) / 2.0
        return abs(lateral) >= (half_ego + half_obj)

    def _make_rss_cbf_info(
        self,
        state: State,
        obj: Optional[Dict[str, Any]],
        object_kind: str,
        mode: str,
        reason: str,
        u_original: Action,
        u_safe: Action,
        d_front: float,
        d_rss: float,
        rss_margin: float,
        dynamic_vehicle_detected: bool,
        obstacle_detected: bool,
        nominal_margins: Dict[str, float],
        projection_debug: Dict[str, Any],
    ) -> Dict[str, Any]:
        if mode not in self.FORMAL_MODES:
            mode = "rss_cbf_intervention"

        selected = projection_debug.get("selected", {}) if isinstance(projection_debug, dict) else {}
        candidates = projection_debug.get("candidates", []) if isinstance(projection_debug, dict) else []
        info = {
            "mode": mode,
            "reason": reason,
            "rss_margin": rss_margin,
            "rss_distance": d_rss,
            "d_front": d_front,
            "d_dynamic": d_rss if object_kind == "dynamic" else math.nan,
            "d_obs": d_front if object_kind == "static" else math.nan,
            "d_brake": d_rss,
            "object_kind": object_kind,
            "dynamic_vehicle_detected": bool(dynamic_vehicle_detected),
            "obstacle_detected": bool(obstacle_detected),
            "left_feasible": False,
            "right_feasible": False,
            "state_debug": self._state_debug(state),
            "blocking_object": self._object_debug(obj, state) if obj is not None else {},
            "nominal_margins": nominal_margins,
            "projection_debug": projection_debug,
            "selected": selected,
            "candidates": candidates,
            "selected_score": selected.get("progress_score") if isinstance(selected, dict) else math.nan,
        }
        if obj is not None and self.config.enable_lateral_rss:
            lat_margin = self.compute_lateral_rss_margin(state, obj)
            path_overlap, lat_dist, lat_mrg = self.check_path_overlap(state, obj)
            deconflicted = self.check_lateral_deconflicted(state, obj, lat_margin)
            info.update({
                "lateral_rss_margin": float(lat_margin),
                "front_object_path_overlap": bool(path_overlap),
                "front_object_lateral_distance": float(lat_dist),
                "front_object_lateral_margin": float(lat_mrg),
                "front_object_deconflicted": bool(deconflicted),
                "longitudinal_constraint_relaxed_by_lateral_escape": bool(deconflicted and not path_overlap),
            })
        else:
            info.update({
                "lateral_rss_margin": math.nan,
                "front_object_path_overlap": False,
                "front_object_lateral_distance": math.nan,
                "front_object_lateral_margin": math.nan,
                "front_object_deconflicted": False,
                "longitudinal_constraint_relaxed_by_lateral_escape": False,
            })
        return self._with_action_debug(info, u_original, u_safe)
