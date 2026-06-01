"""RSS-CBF runtime assurance filter for PPL/TD3 policy evaluation.

The public runtime path is intentionally small:

    u_nom -> RSSCBF Runtime Assurance -> u_safe

The filter keeps the existing MetaDrive adapters and action conversion helpers
from ``StaticRSSFilter``, but exposes only the RSS-CBF forward-distance safety
contract for evaluation.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    certified_lateral_creep_critical_margin: float = -1.5
    certified_lateral_creep_max_acc: float = 1.0

    enable_2d_rss_cbf: bool = True
    rss_2d_power: float = 4.0
    rss_2d_lateral_margin: float = 0.5
    rss_2d_eps: float = 1e-3
    rss_2d_use_superellipse: bool = True

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
        if bool(getattr(self.config, "enable_2d_rss_cbf", False)):
            return self._filter_action_2d_rss_cbf(state, u_original)

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

    def _filter_action_2d_rss_cbf(self, state: State, u_original: Action) -> Tuple[Action, Dict[str, Any]]:
        objects = self._rss_2d_relevant_objects(state)
        nominal_eval = self._evaluate_2d_rss_cbf_candidate(state, u_original, objects)
        nominal_eval.update(self.road_boundary_horizon_metrics(state, u_original))
        nominal_eval["safe"] = bool(
            nominal_eval.get("objects_safe", True)
            and nominal_eval.get("road_boundary_safe", True)
        )

        if nominal_eval["safe"]:
            return u_original, self._make_rss_2d_cbf_info(
                state=state,
                objects=objects,
                mode="normal",
                reason="nominal_2d_cbf_safe",
                u_original=u_original,
                u_safe=u_original,
                nominal_eval=nominal_eval,
                selected_eval=nominal_eval,
                projection_debug={},
            )

        u_safe, projection_debug = self._project_action_for_2d_rss_cbf(
            state, objects, u_original
        )
        selected = projection_debug.get("selected", {}) if isinstance(projection_debug, dict) else {}
        selected_eval = selected.get("evaluation", nominal_eval) if isinstance(selected, dict) else nominal_eval
        selected_safe = bool(selected.get("safe", False)) if isinstance(selected, dict) else False

        if not bool(projection_debug.get("road_boundary_any_safe", True)):
            mode = "fallback_no_safe_candidate"
            reason = "no_road_boundary_safe_candidate_brake_fallback"
        elif selected_safe:
            mode = (
                "rss_2d_cbf_recovery"
                if float(nominal_eval.get("min_h_2d", math.inf)) < -self.config.small_tolerance
                else "rss_2d_cbf_intervention"
            )
            reason = "rss_2d_cbf_candidate_safe"
        else:
            mode = "rss_2d_cbf_recovery"
            reason = str(
                projection_debug.get(
                    "reason",
                    "no_fully_safe_candidate_selected_max_h_2d",
                )
            )

        return u_safe, self._make_rss_2d_cbf_info(
            state=state,
            objects=objects,
            mode=mode,
            reason=reason,
            u_original=u_original,
            u_safe=u_safe,
            nominal_eval=nominal_eval,
            selected_eval=selected_eval,
            projection_debug=projection_debug,
        )

    def _rss_2d_relevant_objects(self, state: State) -> List[Tuple[str, Dict[str, Any]]]:
        objects: List[Tuple[str, Dict[str, Any]]] = []
        for obstacle in state.get("static_obstacles", []):
            objects.append(("static", obstacle))
        for vehicle in state.get("vehicles", []):
            objects.append(("dynamic", vehicle))
        return objects

    def compute_2d_rss_cbf_margin(
        self,
        state: State,
        obj: Dict[str, Any],
        object_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute one RSS-informed 2D CBF margin in ego-local coordinates."""
        ego = self._ego(state)
        delta_s, delta_l = self._relative_position(ego, obj)
        ego_length = self._object_length(ego, self.config.vehicle_length)
        ego_width = self._object_width(ego, self.config.vehicle_width)
        obj_length = self._object_length(obj, self.config.vehicle_length)
        obj_width = self._object_width(obj, self.config.vehicle_width)
        long_clearance = max(0.0, abs(delta_s) - 0.5 * (ego_length + obj_length))
        lat_clearance = max(0.0, abs(delta_l) - 0.5 * (ego_width + obj_width))
        relation = self._rss_2d_object_relation(delta_s, ego_length, obj_length)
        kind = object_kind or self._rss_2d_object_kind(obj)
        d_s_safe = self._rss_2d_longitudinal_safe_distance(state, obj, kind, relation)
        d_l_safe = max(0.0, float(getattr(self.config, "rss_2d_lateral_margin", 0.5)))
        eps = max(float(getattr(self.config, "rss_2d_eps", 1e-3)), self.config.small_tolerance)
        power = (
            float(getattr(self.config, "rss_2d_power", 4.0))
            if bool(getattr(self.config, "rss_2d_use_superellipse", True))
            else 2.0
        )
        power = max(1.0, power)
        h_2d = (
            (long_clearance / max(d_s_safe, eps)) ** power
            + (lat_clearance / max(d_l_safe, eps)) ** power
            - 1.0
        )
        object_type = obj.get("object_type", obj.get("class_name", kind))
        return {
            "h_2d": float(h_2d),
            "delta_s": float(delta_s),
            "delta_l": float(delta_l),
            "long_clearance": float(long_clearance),
            "lat_clearance": float(lat_clearance),
            "d_s_safe": float(d_s_safe),
            "d_l_safe": float(d_l_safe),
            "object_id": obj.get("object_id", obj.get("id", "")),
            "object_type": object_type,
            "object_kind": kind,
            "relation": relation,
            "target_longitudinal_speed": float(self._rss_2d_object_longitudinal_speed(state, obj, kind)),
            "ego_length": float(ego_length),
            "ego_width": float(ego_width),
            "object_length": float(obj_length),
            "object_width": float(obj_width),
            "power": float(power),
            "eps": float(eps),
        }

    def _rss_2d_object_kind(self, obj: Dict[str, Any]) -> str:
        object_type = str(obj.get("object_type", "")).lower()
        if object_type == "vehicle":
            return "dynamic"
        return "static"

    def _rss_2d_object_relation(self, delta_s: float, ego_length: float, obj_length: float) -> str:
        longitudinal_overlap = 0.5 * (ego_length + obj_length)
        if delta_s > longitudinal_overlap:
            return "front"
        if delta_s < -longitudinal_overlap:
            return "rear"
        return "side"

    def _rss_2d_object_longitudinal_speed(
        self,
        state: State,
        obj: Dict[str, Any],
        object_kind: str,
    ) -> float:
        if object_kind != "dynamic":
            return 0.0
        speed = max(0.0, float(obj.get("speed", 0.0)))
        ego_heading = float(self._ego(state).get("heading", 0.0))
        obj_heading = float(obj.get("heading", ego_heading))
        return speed * math.cos(obj_heading - ego_heading)

    def _rss_2d_longitudinal_safe_distance(
        self,
        state: State,
        obj: Dict[str, Any],
        object_kind: str,
        relation: str,
    ) -> float:
        ego_speed = self._ego_speed(state)
        if object_kind == "dynamic":
            target_speed = max(0.0, self._rss_2d_object_longitudinal_speed(state, obj, object_kind))
            if relation == "rear":
                return self._rss_rear_distance(ego_speed, target_speed)
            if relation == "side":
                return max(
                    self._rss_front_distance(ego_speed, target_speed),
                    self._rss_rear_distance(ego_speed, target_speed),
                )
            return self.compute_dynamic_front_distance(ego_speed, target_speed)
        return self.compute_rss_distance(
            ego_speed,
            front_speed=0.0,
            margin=self.config.vehicle_length / 2.0 + self.config.obstacle_margin,
        )

    def _evaluate_2d_rss_cbf_candidate(
        self,
        state: State,
        action: Sequence[float],
        objects: Sequence[Tuple[str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        action = self._clip_action(action)
        horizon = max(0, int(self.config.horizon_steps))
        rollout_state = state
        rollout_objects = [(kind, copy.deepcopy(obj)) for kind, obj in objects]
        min_h = math.inf
        current_h = math.inf
        final_h = math.inf
        worst: Dict[str, Any] = {}
        per_object: List[Dict[str, Any]] = []

        for step in range(horizon + 1):
            step_min = math.inf
            for object_kind, obj in rollout_objects:
                margin = self.compute_2d_rss_cbf_margin(rollout_state, obj, object_kind)
                margin["step"] = int(step)
                margin["object_debug"] = self._object_debug(obj, rollout_state)
                per_object.append(margin)
                h_value = float(margin["h_2d"])
                if h_value < step_min:
                    step_min = h_value
                if h_value < min_h:
                    min_h = h_value
                    worst = margin

            if step == 0:
                current_h = step_min
            final_h = step_min
            if step >= horizon:
                break

            next_state = self._simulate_next_state(rollout_state, action)
            rollout_objects = [
                (object_kind, self._advance_object_for_rss_cbf(rollout_state, obj, object_kind))
                for object_kind, obj in rollout_objects
            ]
            rollout_state = next_state

        objects_safe = min_h >= -self.config.small_tolerance
        return {
            "objects_safe": bool(objects_safe),
            "min_h_2d": float(min_h),
            "current_h_2d": float(current_h),
            "final_h_2d": float(final_h),
            "worst": worst,
            "object_margins": per_object,
            "num_objects": int(len(objects)),
        }

    def _project_action_for_2d_rss_cbf(
        self,
        state: State,
        objects: Sequence[Tuple[str, Dict[str, Any]]],
        u_original: Action,
    ) -> Tuple[Action, Dict[str, Any]]:
        ray_steps = max(1, int(self.config.ray_steps))
        candidate_accs = np.linspace(float(u_original[0]), self.config.min_acc, ray_steps + 1)
        candidates: List[Dict[str, Any]] = []
        seen = set()

        for acc in candidate_accs:
            action = self._clip_action([float(acc), u_original[1]])
            key = (round(action[0], 8), round(action[1], 8))
            if key in seen:
                continue
            seen.add(key)
            evaluation = self._evaluate_2d_rss_cbf_candidate(state, action, objects)
            boundary = self.road_boundary_horizon_metrics(state, action)
            evaluation.update(boundary)
            safe = bool(evaluation["objects_safe"] and evaluation["road_boundary_safe"])
            candidate = {
                "mode": "rss_2d_cbf_candidate",
                "action": action,
                "safe": safe,
                "road_boundary_safe": bool(evaluation["road_boundary_safe"]),
                "min_h_2d": float(evaluation["min_h_2d"]),
                "final_h_2d": float(evaluation["final_h_2d"]),
                "evaluation": evaluation,
                "margins": {
                    "min_h_2d": float(evaluation["min_h_2d"]),
                    "current_h_2d": float(evaluation["current_h_2d"]),
                    "final_h_2d": float(evaluation["final_h_2d"]),
                    "road_boundary_margin_min_pred": float(evaluation["road_boundary_margin_min_pred"]),
                },
                "intervention_cost": self._action_distance_sq(action, u_original),
                "progress_score": self._score_progress_after_rollout(state, action),
                "reject_reason": "",
            }
            if not safe:
                candidate["reject_reason"] = self._rss_2d_candidate_reject_reason(evaluation)
            candidates.append(candidate)

        selected = self._select_2d_rss_cbf_candidate(candidates, u_original)
        if selected is None:
            fallback = self._clip_action([self.config.min_acc, u_original[1]])
            evaluation = self._evaluate_2d_rss_cbf_candidate(state, fallback, objects)
            evaluation.update(self.road_boundary_horizon_metrics(state, fallback))
            selected = {
                "mode": "rss_2d_cbf_brake_fallback",
                "action": fallback,
                "safe": False,
                "road_boundary_safe": bool(evaluation.get("road_boundary_safe", False)),
                "min_h_2d": float(evaluation.get("min_h_2d", -math.inf)),
                "final_h_2d": float(evaluation.get("final_h_2d", -math.inf)),
                "evaluation": evaluation,
                "margins": {
                    "min_h_2d": float(evaluation.get("min_h_2d", -math.inf)),
                    "current_h_2d": float(evaluation.get("current_h_2d", -math.inf)),
                    "final_h_2d": float(evaluation.get("final_h_2d", -math.inf)),
                    "road_boundary_margin_min_pred": float(evaluation.get("road_boundary_margin_min_pred", -math.inf)),
                },
                "intervention_cost": self._action_distance_sq(fallback, u_original),
                "progress_score": self._score_progress_after_rollout(state, fallback),
                "reject_reason": "road_boundary_violation",
            }
            candidates.append(selected)

        safe_candidates = [candidate for candidate in candidates if candidate.get("safe", False)]
        road_safe_candidates = [
            candidate for candidate in candidates if candidate.get("road_boundary_safe", False)
        ]
        reject_reasons = self._rss_2d_candidate_reject_reasons(candidates)
        return selected["action"], {
            "projection_failed": not bool(selected.get("safe", False)),
            "candidate_count": len(candidates),
            "candidate_reject_reasons": reject_reasons,
            "road_boundary_any_safe": bool(road_safe_candidates),
            "safe_candidate_count": len(safe_candidates),
            "road_safe_candidate_count": len(road_safe_candidates),
            "selected": selected,
            "margins": selected.get("margins", {}),
            "candidates": candidates,
            "reason": (
                ""
                if selected.get("safe", False)
                else selected.get("reject_reason", "no_fully_safe_candidate_selected_max_h_2d")
            ),
        }

    def _select_2d_rss_cbf_candidate(
        self,
        candidates: Sequence[Dict[str, Any]],
        u_original: Action,
    ) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None
        road_safe = [candidate for candidate in candidates if candidate.get("road_boundary_safe", False)]
        if not road_safe:
            return None
        fully_safe = [candidate for candidate in road_safe if candidate.get("safe", False)]
        if fully_safe:
            return max(
                fully_safe,
                key=lambda item: (
                    float(item.get("min_h_2d", -math.inf)),
                    -float(item.get("intervention_cost", math.inf)),
                ),
            )
        return max(
            road_safe,
            key=lambda item: (
                float(item.get("min_h_2d", -math.inf)),
                float(item.get("final_h_2d", -math.inf)),
                -float(item.get("action", [math.inf, 0.0])[0]),
                -float(item.get("intervention_cost", math.inf)),
            ),
        )

    def _rss_2d_candidate_reject_reason(self, evaluation: Dict[str, Any]) -> str:
        if not bool(evaluation.get("road_boundary_safe", True)):
            return "road_boundary_violation"
        if float(evaluation.get("min_h_2d", math.inf)) < -self.config.small_tolerance:
            return "rss_2d_cbf_violation"
        return ""

    def _rss_2d_candidate_reject_reasons(self, candidates: Sequence[Dict[str, Any]]) -> str:
        reasons = [
            str(candidate.get("reject_reason", ""))
            for candidate in candidates
            if candidate.get("reject_reason", "")
        ]
        return ";".join(dict.fromkeys(reasons))

    def _make_rss_2d_cbf_info(
        self,
        state: State,
        objects: Sequence[Tuple[str, Dict[str, Any]]],
        mode: str,
        reason: str,
        u_original: Action,
        u_safe: Action,
        nominal_eval: Dict[str, Any],
        selected_eval: Dict[str, Any],
        projection_debug: Dict[str, Any],
    ) -> Dict[str, Any]:
        worst = selected_eval.get("worst", {}) if isinstance(selected_eval, dict) else {}
        worst_debug = worst.get("object_debug", {}) if isinstance(worst, dict) else {}
        object_kind = str(worst.get("object_kind", "none")) if worst else "none"
        d_s_safe = float(worst.get("d_s_safe", math.nan)) if worst else math.nan
        long_clearance = float(worst.get("long_clearance", math.nan)) if worst else math.nan
        selected = projection_debug.get("selected", {}) if isinstance(projection_debug, dict) else {}
        candidates = projection_debug.get("candidates", []) if isinstance(projection_debug, dict) else []
        road_safe = bool(selected_eval.get("road_boundary_safe", True))
        info = {
            "mode": mode,
            "reason": reason,
            "safety_function_mode": "rss_2d_cbf",
            "rss_cbf_variant": "rss_2d_cbf",
            "cbf_mode": mode,
            "min_h_2d": float(selected_eval.get("min_h_2d", math.inf)),
            "current_h_2d": float(selected_eval.get("current_h_2d", math.inf)),
            "final_h_2d": float(selected_eval.get("final_h_2d", math.inf)),
            "nominal_min_h_2d": float(nominal_eval.get("min_h_2d", math.inf)),
            "nominal_final_h_2d": float(nominal_eval.get("final_h_2d", math.inf)),
            "worst_object_id": worst.get("object_id", "") if worst else "",
            "worst_object_type": worst.get("object_type", "") if worst else "",
            "worst_object_kind": object_kind,
            "worst_object_relation": worst.get("relation", "") if worst else "",
            "worst_delta_s": float(worst.get("delta_s", math.nan)) if worst else math.nan,
            "worst_delta_l": float(worst.get("delta_l", math.nan)) if worst else math.nan,
            "worst_long_clearance": long_clearance,
            "worst_lat_clearance": float(worst.get("lat_clearance", math.nan)) if worst else math.nan,
            "worst_d_s_safe": d_s_safe,
            "worst_d_l_safe": float(worst.get("d_l_safe", math.nan)) if worst else math.nan,
            "road_boundary_safe": road_safe,
            "left_boundary_safe": bool(selected_eval.get("left_boundary_safe", True)),
            "right_boundary_safe": bool(selected_eval.get("right_boundary_safe", True)),
            "road_boundary_margin_current": float(selected_eval.get("road_boundary_margin_current", math.nan)),
            "road_boundary_margin_min_pred": float(selected_eval.get("road_boundary_margin_min_pred", math.nan)),
            "road_boundary_margin_final_pred": float(selected_eval.get("road_boundary_margin_final_pred", math.nan)),
            "road_boundary_margin_improvement": float(selected_eval.get("road_boundary_margin_improvement", math.nan)),
            "road_boundary_margin_source": selected_eval.get("road_boundary_margin_source", ""),
            "selected_action": list(u_safe),
            "nominal_action": list(u_original),
            "filter_intervened": bool(self._action_distance_sq(u_safe, u_original) > self.config.small_tolerance),
            "candidate_reject_reasons": projection_debug.get("candidate_reject_reasons", ""),
            "candidate_count": int(projection_debug.get("candidate_count", 0)),
            "safe_candidate_count": int(projection_debug.get("safe_candidate_count", 0)),
            "road_safe_candidate_count": int(projection_debug.get("road_safe_candidate_count", 0)),
            "rss_margin": float(selected_eval.get("min_h_2d", math.inf)),
            "rss_distance": d_s_safe,
            "d_front": long_clearance,
            "d_dynamic": d_s_safe if object_kind == "dynamic" else math.nan,
            "d_obs": long_clearance if object_kind == "static" else math.nan,
            "d_brake": d_s_safe,
            "object_kind": object_kind,
            "dynamic_vehicle_detected": any(kind == "dynamic" for kind, _ in objects),
            "obstacle_detected": any(kind == "static" for kind, _ in objects),
            "left_feasible": bool(selected_eval.get("left_boundary_safe", False)),
            "right_feasible": bool(selected_eval.get("right_boundary_safe", False)),
            "state_debug": self._state_debug(state),
            "blocking_object": worst_debug,
            "nominal_margins": {
                "min_h_2d": float(nominal_eval.get("min_h_2d", math.inf)),
                "final_h_2d": float(nominal_eval.get("final_h_2d", math.inf)),
                "road_boundary_margin_min_pred": float(nominal_eval.get("road_boundary_margin_min_pred", math.nan)),
            },
            "projection_debug": projection_debug,
            "selected": selected,
            "candidates": candidates,
            "selected_score": selected.get("progress_score") if isinstance(selected, dict) else math.nan,
            "lateral_rss_margin": (
                float(worst.get("lat_clearance", math.nan)) - float(worst.get("d_l_safe", math.nan))
                if worst else math.nan
            ),
            "lateral_rss_safe_distance": float(worst.get("d_l_safe", math.nan)) if worst else math.nan,
            "longitudinal_distance": float(worst.get("delta_s", math.nan)) if worst else math.nan,
            "front_object_path_overlap": bool(
                worst and float(worst.get("lat_clearance", 0.0)) <= self.config.small_tolerance
            ),
            "front_object_path_overlap_reducing": False,
            "front_object_lateral_distance": abs(float(worst.get("delta_l", math.nan))) if worst else math.nan,
            "front_object_lateral_margin": float(worst.get("lat_clearance", math.nan)) if worst else math.nan,
            "front_object_deconflicted": bool(
                worst and float(worst.get("lat_clearance", -math.inf)) >= float(worst.get("d_l_safe", math.inf))
            ),
            "front_object_terminal_lateral_distance": math.nan,
            "front_object_terminal_lateral_margin": math.nan,
            "longitudinal_constraint_relaxed_by_lateral_escape": False,
        }
        for key in [
            "road_boundary_left_margin_min_pred",
            "road_boundary_right_margin_min_pred",
            "ego_dist_to_left_side",
            "ego_dist_to_right_side",
            "ego_on_lane",
            "ego_out_of_route",
            "ego_crash_sidewalk",
        ]:
            if key in selected_eval:
                info[key] = selected_eval[key]
        return self._with_action_debug(info, u_original, u_safe)

    def _select_front_rss_object(
        self, state: State
    ) -> Optional[Tuple[str, Dict[str, Any], float, float, float, bool, bool]]:
        dynamic_info = self.detect_dynamic_vehicle_ahead(state)
        static_info = self.detect_static_obstacle_ahead(state)
        candidates = []

        if dynamic_info is not None:
            vehicle, d_front = dynamic_info
            d_rss = self._rss_cbf_distance(state, vehicle, "dynamic")
            candidates.append(("dynamic", vehicle, d_front, d_rss, self._rss_cbf_margin(state, vehicle, "dynamic")))

        if static_info is not None:
            obstacle, d_front = static_info
            d_rss = self._rss_cbf_distance(state, obstacle, "static")
            candidates.append(("static", obstacle, d_front, d_rss, self._rss_cbf_margin(state, obstacle, "static")))

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
        if self._front_object_lateral_constraint_relaxed(state, obj):
            return math.inf
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
        if self.config.enable_lateral_rss:
            lateral_horizon = self._rss_cbf_lateral_horizon_metrics(state, obj, object_kind, action)
            initial = lateral_horizon.get("initial", {})
            final = lateral_horizon.get("final", {})
            margins.update({
                "lateral_rss_margin_current": float(initial.get("lateral_rss_margin", math.nan)),
                "lateral_rss_margin_final": float(final.get("lateral_rss_margin", math.nan)),
                "path_overlap_initial": bool(initial.get("path_overlap", False)),
                "path_overlap_final": bool(final.get("path_overlap", False)),
                "path_overlap_reducing": bool(lateral_horizon.get("path_overlap_reducing", False)),
                "terminal_lateral_separation_safe": bool(lateral_horizon.get("terminal_lateral_separation_safe", False)),
                "longitudinal_constraint_relaxed_by_lateral_escape": bool(
                    lateral_horizon.get("path_overlap_reducing", False)
                    and lateral_horizon.get("terminal_lateral_separation_safe", False)
                ),
            })
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
        lateral_escape_relaxes_longitudinal = bool(
            margins.get("longitudinal_constraint_relaxed_by_lateral_escape", False)
        )
        if lateral_escape_relaxes_longitudinal:
            max_lateral_escape_acc = max(0.0, min(self.config.max_acc, self.config.a_max * 0.5))
            margin_floor = -abs(float(self.config.certified_lateral_escape_margin_buffer))
            if (
                self._clip_action(action)[0] <= max_lateral_escape_acc + self.config.small_tolerance
                and min_margin >= margin_floor
            ):
                return True

        if current_margin >= -self.config.small_tolerance:
            return (
                min_margin >= -self.config.small_tolerance
                and final_margin + self.config.small_tolerance >= current_margin
            )

        if not self.config.enable_recovery_mode:
            return min_margin >= -self.config.small_tolerance

        clipped_acc = self._clip_action(action)[0]
        if clipped_acc > self.config.small_tolerance:
            is_lateral_creep_pass_through = bool(
                margins.get("terminal_lateral_separation_safe", False)
                and margins.get("path_overlap_reducing", False)
                and min_margin >= float(getattr(self.config, "certified_lateral_creep_critical_margin", -1.5))
                and clipped_acc <= float(getattr(self.config, "certified_lateral_creep_max_acc", 1.0)) + self.config.small_tolerance
                and abs(float(action[1])) > 0.05
            )
            if is_lateral_creep_pass_through:
                return True
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
        return float(self.compute_lateral_rss_metrics(state, obj)["lateral_rss_margin"])

    def compute_lateral_rss_metrics(
        self,
        state: State,
        obj: Dict[str, Any],
    ) -> Dict[str, float]:
        ego = self._ego(state)
        longitudinal, lateral = self._relative_position(ego, obj)
        lateral_distance = abs(lateral)
        half_ego_rss = self.config.vehicle_width / 2.0
        half_ego_path = self.config.vehicle_width / 2.0 + self.config.obstacle_margin
        half_obj = self._object_width(obj, self.config.vehicle_width) / 2.0
        lateral_gap = lateral_distance - half_ego_rss - half_obj
        path_overlap_amount = max(0.0, half_ego_path + half_obj - lateral_distance)
        lateral_speed = float(ego.get("lateral_speed", 0.0))
        safe_distance = self.compute_lateral_rss_safe_distance(lateral_speed)
        lateral_rss_margin = lateral_gap - safe_distance
        path_overlap = path_overlap_amount > self.config.small_tolerance
        lateral_deconflicted = lateral_rss_margin > self.config.small_tolerance and not path_overlap
        return {
            "longitudinal_distance": float(longitudinal),
            "lateral_signed": float(lateral),
            "lateral_distance": float(lateral_distance),
            "lateral_gap": float(lateral_gap),
            "lateral_rss_safe_distance": float(safe_distance),
            "lateral_rss_margin": float(lateral_rss_margin),
            "path_overlap": bool(path_overlap),
            "path_overlap_amount": float(path_overlap_amount),
            "lateral_deconflicted": bool(lateral_deconflicted),
        }

    def check_path_overlap(
        self,
        state: State,
        obj: Dict[str, Any],
    ) -> Tuple[bool, float, float]:
        metrics = self.compute_lateral_rss_metrics(state, obj)
        half_ego = self.config.vehicle_width / 2.0 + self.config.obstacle_margin
        half_obj = self._object_width(obj, self.config.vehicle_width) / 2.0
        lateral_margin = float(metrics["lateral_distance"]) - half_ego - half_obj
        return bool(metrics["path_overlap"]), float(metrics["lateral_distance"]), float(lateral_margin)

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

    def _front_object_lateral_constraint_relaxed(self, state: State, obj: Dict[str, Any]) -> bool:
        if not self.config.enable_lateral_rss:
            return False
        metrics = self.compute_lateral_rss_metrics(state, obj)
        if float(metrics["longitudinal_distance"]) <= 0.0:
            return False
        return bool(
            float(metrics["lateral_rss_margin"]) > self.config.small_tolerance
            and not bool(metrics["path_overlap"])
        )

    def _advance_object_for_rss_cbf(self, state: State, obj: Dict[str, Any], object_kind: str) -> Dict[str, Any]:
        next_obj = copy.deepcopy(obj)
        if object_kind == "dynamic":
            heading = float(next_obj.get("heading", self._ego(state).get("heading", 0.0)))
            speed = max(0.0, float(next_obj.get("speed", 0.0)))
            next_obj["x"] = float(next_obj.get("x", 0.0)) + speed * math.cos(heading) * self.config.dt
            next_obj["y"] = float(next_obj.get("y", 0.0)) + speed * math.sin(heading) * self.config.dt
        return next_obj

    def _rss_cbf_lateral_horizon_metrics(
        self,
        state: State,
        obj: Dict[str, Any],
        object_kind: str,
        action: Sequence[float],
    ) -> Dict[str, Any]:
        initial = self.compute_lateral_rss_metrics(state, obj)
        rollout_state = state
        rollout_obj = copy.deepcopy(obj)
        final = initial
        min_lateral_rss_margin = float(initial["lateral_rss_margin"])
        min_path_overlap_amount = float(initial["path_overlap_amount"])

        for _ in range(max(0, int(self.config.horizon_steps))):
            rollout_state = self._simulate_next_state(rollout_state, action)
            rollout_obj = self._advance_object_for_rss_cbf(rollout_state, rollout_obj, object_kind)
            final = self.compute_lateral_rss_metrics(rollout_state, rollout_obj)
            min_lateral_rss_margin = min(min_lateral_rss_margin, float(final["lateral_rss_margin"]))
            min_path_overlap_amount = min(min_path_overlap_amount, float(final["path_overlap_amount"]))

        path_overlap_reducing = (
            float(final["path_overlap_amount"]) + self.config.small_tolerance < float(initial["path_overlap_amount"])
            or float(final["lateral_distance"]) > float(initial["lateral_distance"]) + self.config.small_tolerance
            or (bool(initial["path_overlap"]) and not bool(final["path_overlap"]))
        )
        terminal_lateral_safe = (
            float(final["lateral_rss_margin"]) > self.config.small_tolerance
        )
        return {
            "initial": initial,
            "final": final,
            "min_lateral_rss_margin": float(min_lateral_rss_margin),
            "min_path_overlap_amount": float(min_path_overlap_amount),
            "path_overlap_reducing": bool(path_overlap_reducing),
            "terminal_lateral_separation_safe": bool(terminal_lateral_safe),
            "lateral_deconflicted": bool(final.get("lateral_deconflicted", False)),
        }

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
            "safety_function_mode": "original_rss_cbf",
            "rss_cbf_variant": "original_rss_cbf",
            "cbf_mode": mode,
            "min_h_2d": math.nan,
            "current_h_2d": math.nan,
            "final_h_2d": math.nan,
            "nominal_min_h_2d": math.nan,
            "nominal_final_h_2d": math.nan,
            "worst_object_id": "",
            "worst_object_type": "",
            "worst_object_kind": object_kind,
            "worst_object_relation": "",
            "worst_delta_s": math.nan,
            "worst_delta_l": math.nan,
            "worst_long_clearance": math.nan,
            "worst_lat_clearance": math.nan,
            "worst_d_s_safe": math.nan,
            "worst_d_l_safe": math.nan,
            "selected_action": list(u_safe),
            "nominal_action": list(u_original),
            "filter_intervened": bool(self._action_distance_sq(u_safe, u_original) > self.config.small_tolerance),
            "candidate_reject_reasons": "",
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
        boundary_metrics = self.road_boundary_horizon_metrics(state, u_safe)
        info.update(boundary_metrics)
        if obj is not None and self.config.enable_lateral_rss:
            lat_metrics = self.compute_lateral_rss_metrics(state, obj)
            path_overlap, lat_dist, lat_mrg = self.check_path_overlap(state, obj)
            lat_margin = float(lat_metrics["lateral_rss_margin"])
            horizon_metrics = self._rss_cbf_lateral_horizon_metrics(state, obj, object_kind, u_original)
            terminal_metrics = horizon_metrics.get("final", {})
            deconflicted = self.check_lateral_deconflicted(state, obj, lat_margin)
            path_overlap_reducing = bool(horizon_metrics.get("path_overlap_reducing", False))
            relaxed_by_lateral_escape = bool(
                self._front_object_lateral_constraint_relaxed(state, obj)
                or (
                    lat_margin > self.config.small_tolerance
                    and path_overlap_reducing
                    and bool(horizon_metrics.get("terminal_lateral_separation_safe", False))
                )
            )
            info.update({
                "lateral_rss_margin": float(lat_margin),
                "lateral_rss_safe_distance": float(lat_metrics["lateral_rss_safe_distance"]),
                "longitudinal_distance": float(lat_metrics["longitudinal_distance"]),
                "front_object_path_overlap": bool(path_overlap),
                "front_object_path_overlap_reducing": bool(path_overlap_reducing),
                "front_object_lateral_distance": float(lat_dist),
                "front_object_lateral_margin": float(lat_mrg),
                "front_object_deconflicted": bool(deconflicted),
                "front_object_terminal_lateral_distance": float(terminal_metrics.get("lateral_distance", math.nan)),
                "front_object_terminal_lateral_margin": float(terminal_metrics.get("lateral_rss_margin", math.nan)),
                "longitudinal_constraint_relaxed_by_lateral_escape": relaxed_by_lateral_escape,
            })
        else:
            info.update({
                "lateral_rss_margin": math.nan,
                "lateral_rss_safe_distance": math.nan,
                "longitudinal_distance": math.nan,
                "front_object_path_overlap": False,
                "front_object_path_overlap_reducing": False,
                "front_object_lateral_distance": math.nan,
                "front_object_lateral_margin": math.nan,
                "front_object_deconflicted": False,
                "front_object_terminal_lateral_distance": math.nan,
                "front_object_terminal_lateral_margin": math.nan,
                "longitudinal_constraint_relaxed_by_lateral_escape": False,
            })
        return self._with_action_debug(info, u_original, u_safe)
