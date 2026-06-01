"""RSS-CBF runtime assurance filter for PPL/TD3 policy evaluation.

The public runtime path is intentionally small:

    u_nom -> RSSCBF Runtime Assurance -> u_safe

The filter keeps the existing MetaDrive adapters and action conversion helpers
from ``StaticRSSFilter``, but exposes only the RSS-CBF forward-distance safety
contract for evaluation.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ppl.utils.static_rss_filter import Action, State, StaticRSSConfig, StaticRSSFilter


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SafetyObject:
    """Unified safety-object wrapper used by the 2D RSS-informed CBF path."""

    object_id: str
    object_type: str
    geometry_type: str
    object_kind: str
    payload: Optional[Dict[str, Any]] = None
    is_dynamic: bool = False
    center: Tuple[float, float] = (0.0, 0.0)
    radius: float = 0.0


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
    rss_2d_safety_margin: float = 0.0
    rss_2d_use_superellipse: bool = True
    rss_2d_sample_steer: bool = True
    rss_2d_steer_samples: Tuple[float, ...] = (-0.5, -0.25, 0.0, 0.25, 0.5)
    rss_2d_acc_samples: int = 7
    rss_2d_prefer_centering: bool = True
    rss_2d_boundary_margin_threshold: float = 0.5
    rss_2d_boundary_penalty_weight: float = 2.0
    rss_2d_nominal_action_weight: float = 0.5
    rss_2d_speed_preserve_weight: float = 0.3
    rss_2d_min_speed_preserve_acc: float = -0.2
    enable_unified_filter_profiling: bool = True
    debug_log_interval: int = 50
    prediction_horizon_steps: int = 2
    lazy_safety_margin: float = 0.2
    candidate_search_mode: str = "lazy_coarse_to_fine"
    coarse_acc_samples: Tuple[float, ...] = (0.0, -0.2, -0.5)
    coarse_steer_samples: Tuple[float, ...] = (-0.5, -0.25, 0.0, 0.25, 0.5)
    fine_acc_samples: int = 7
    fine_steer_samples: int = 5
    enable_static_safety_object_cache: bool = True
    enable_spatial_filtering: bool = True
    enable_vectorized_broad_phase: bool = True
    max_check_distance: float = 40.0
    broad_phase_radius: float = 40.0
    max_local_safety_objects: int = 64
    max_static_obstacles: int = 16
    max_dynamic_vehicles: int = 16
    dynamic_check_distance: float = 50.0
    max_boundary_segments: int = 16
    boundary_sample_interval: float = 2.0

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
        self._rss_2d_static_cache_signature: Optional[Tuple[Any, ...]] = None
        self._rss_2d_static_cache: List[SafetyObject] = []
        self._rss_2d_static_grid: Dict[Tuple[int, int], List[SafetyObject]] = {}
        self._rss_2d_static_grid_cell_size: float = 1.0
        self._rss_2d_filter_step: int = 0
        self._rss_2d_last_profile: Dict[str, Any] = {}

    def reset(self) -> None:
        """Clear per-episode RSS-2D caches without changing configuration."""
        self._rss_2d_static_cache_signature = None
        self._rss_2d_static_cache = []
        self._rss_2d_static_grid = {}
        self._rss_2d_static_grid_cell_size = 1.0
        self._rss_2d_filter_step = 0
        self._rss_2d_last_profile = {}

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
        self._rss_2d_filter_step += 1
        profile = self._new_rss_2d_profile()
        total_start = time.perf_counter()

        start = time.perf_counter()
        all_objects = self._rss_2d_build_safety_objects(state)
        profile["build_safety_objects_time"] = time.perf_counter() - start
        profile["safety_object_count_total"] = int(len(all_objects))

        start = time.perf_counter()
        objects = self._rss_2d_query_local_objects(state, all_objects)
        profile["query_local_objects_time"] = time.perf_counter() - start
        profile["safety_object_count_local"] = int(len(objects))
        profile["horizon_steps"] = int(self._rss_2d_prediction_horizon_steps())

        start = time.perf_counter()
        nominal_eval = self._evaluate_2d_rss_cbf_candidate(
            state,
            u_original,
            objects,
            collect_debug=True,
        )
        profile["nominal_eval_time"] = time.perf_counter() - start
        profile["geometry_distance_time"] += float(nominal_eval.get("geometry_distance_time", 0.0))
        safety_margin = float(getattr(self.config, "rss_2d_safety_margin", 0.0))
        nominal_eval["safe"] = bool(float(nominal_eval.get("H", -math.inf)) >= safety_margin)
        lazy_margin = float(getattr(self.config, "lazy_safety_margin", safety_margin))

        if float(nominal_eval.get("H", -math.inf)) >= lazy_margin:
            profile["total_filter_time"] = time.perf_counter() - total_start
            profile["candidate_count"] = 0
            self._rss_2d_last_profile = dict(profile)
            self._maybe_log_rss_2d_profile(profile, mode="normal")
            return u_original, self._make_rss_2d_cbf_info(
                state=state,
                objects=objects,
                mode="normal",
                reason="nominal_lazy_safe",
                u_original=u_original,
                u_safe=u_original,
                nominal_eval=nominal_eval,
                selected_eval=nominal_eval,
                projection_debug={"profile": profile},
            )

        u_safe, projection_debug = self._project_action_for_2d_rss_cbf(
            state,
            objects,
            u_original,
            nominal_eval=nominal_eval,
            profile=profile,
        )
        selected = projection_debug.get("selected", {}) if isinstance(projection_debug, dict) else {}
        selected_eval = selected.get("evaluation", nominal_eval) if isinstance(selected, dict) else nominal_eval
        selected_safe = bool(float(selected.get("H", selected_eval.get("H", -math.inf))) >= -self.config.small_tolerance)

        if selected_safe:
            mode = "rss_2d_cbf_intervention"
            reason = "selected_safe_candidate"
        else:
            mode = "rss_2d_cbf_recovery"
            reason = "selected_least_unsafe"

        profile["total_filter_time"] = time.perf_counter() - total_start
        projection_debug["profile"] = profile
        self._rss_2d_last_profile = dict(profile)
        self._maybe_log_rss_2d_profile(profile, mode=mode)

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

    def _rss_2d_relevant_objects(self, state: State) -> List[SafetyObject]:
        all_objects = self._rss_2d_build_safety_objects(state)
        return self._rss_2d_query_local_objects(state, all_objects)

    def _rss_2d_build_safety_objects(self, state: State) -> List[SafetyObject]:
        static_obstacles = list(state.get("static_obstacles", []) or [])
        static_objects: List[SafetyObject]
        signature = self._rss_2d_static_obstacle_signature(static_obstacles)
        cache_enabled = bool(getattr(self.config, "enable_static_safety_object_cache", True))
        if (
            cache_enabled
            and self._rss_2d_static_cache_signature == signature
            and self._rss_2d_static_cache
        ):
            static_objects = list(self._rss_2d_static_cache)
        else:
            static_objects = [
                self._make_static_safety_object(obstacle, index)
                for index, obstacle in enumerate(static_obstacles)
            ]
            if cache_enabled:
                self._rss_2d_static_cache_signature = signature
                self._rss_2d_static_cache = list(static_objects)
            self._rebuild_rss_2d_static_grid(static_objects)

        objects: List[SafetyObject] = []
        objects.extend(static_objects)
        for index, vehicle in enumerate(state.get("vehicles", []) or []):
            objects.append(self._make_dynamic_safety_object(vehicle, index))
        return objects

    def _rss_2d_static_obstacle_signature(self, obstacles: Sequence[Dict[str, Any]]) -> Tuple[Any, ...]:
        return tuple(
            (
                obstacle.get("object_id", obstacle.get("id", index)),
                self._safe_float(obstacle.get("x", 0.0), 0.0),
                self._safe_float(obstacle.get("y", 0.0), 0.0),
                self._safe_float(obstacle.get("heading", 0.0), 0.0),
                self._safe_float(obstacle.get("length", self.config.vehicle_length), self.config.vehicle_length),
                self._safe_float(obstacle.get("width", self.config.vehicle_width), self.config.vehicle_width),
                str(obstacle.get("object_type", "static_obstacle") or "static_obstacle"),
            )
            for index, obstacle in enumerate(obstacles)
        )

    def _make_static_safety_object(self, obstacle: Dict[str, Any], index: int) -> SafetyObject:
        object_type = str(obstacle.get("object_type", "static_obstacle") or "static_obstacle")
        center, radius = self._safety_object_center_radius(obstacle)
        return SafetyObject(
            object_id=str(obstacle.get("object_id", obstacle.get("id", "static_{}".format(index)))),
            object_type=object_type,
            geometry_type="bbox",
            object_kind="static",
            payload=obstacle,
            is_dynamic=False,
            center=center,
            radius=radius,
        )

    def _make_dynamic_safety_object(self, vehicle: Dict[str, Any], index: int) -> SafetyObject:
        center, radius = self._safety_object_center_radius(vehicle)
        return SafetyObject(
            object_id=str(vehicle.get("object_id", vehicle.get("id", "vehicle_{}".format(index)))),
            object_type="dynamic_vehicle",
            geometry_type="bbox",
            object_kind="dynamic",
            payload=vehicle,
            is_dynamic=True,
            center=center,
            radius=radius,
        )

    def _safety_object_center_radius(self, obj: Dict[str, Any]) -> Tuple[Tuple[float, float], float]:
        center = (
            self._safe_float(obj.get("x", 0.0), 0.0),
            self._safe_float(obj.get("y", 0.0), 0.0),
        )
        length = self._object_length(obj, self.config.vehicle_length)
        width = self._object_width(obj, self.config.vehicle_width)
        return center, 0.5 * math.hypot(length, width)

    def _rebuild_rss_2d_static_grid(self, static_objects: Sequence[SafetyObject]) -> None:
        cell_size = max(
            5.0,
            min(
                float(getattr(self.config, "max_check_distance", 40.0)),
                float(getattr(self.config, "broad_phase_radius", 40.0)),
            ),
        )
        self._rss_2d_static_grid_cell_size = cell_size
        grid: Dict[Tuple[int, int], List[SafetyObject]] = {}
        for safety_object in static_objects:
            key = self._rss_2d_grid_key(safety_object.center, cell_size)
            grid.setdefault(key, []).append(safety_object)
        self._rss_2d_static_grid = grid

    def _rss_2d_grid_key(self, center: Tuple[float, float], cell_size: float) -> Tuple[int, int]:
        safe_cell = max(cell_size, self.config.small_tolerance)
        return (
            int(math.floor(float(center[0]) / safe_cell)),
            int(math.floor(float(center[1]) / safe_cell)),
        )

    def _query_rss_2d_static_grid(self, state: State, radius: float) -> List[SafetyObject]:
        if not self._rss_2d_static_grid:
            return list(self._rss_2d_static_cache)
        ego = self._ego(state)
        center = (
            self._safe_float(ego.get("x", 0.0), 0.0),
            self._safe_float(ego.get("y", 0.0), 0.0),
        )
        cell_size = max(self._rss_2d_static_grid_cell_size, self.config.small_tolerance)
        center_key = self._rss_2d_grid_key(center, cell_size)
        cell_radius = max(1, int(math.ceil(max(0.0, radius) / cell_size)) + 1)
        candidates: List[SafetyObject] = []
        seen = set()
        for ix in range(center_key[0] - cell_radius, center_key[0] + cell_radius + 1):
            for iy in range(center_key[1] - cell_radius, center_key[1] + cell_radius + 1):
                for safety_object in self._rss_2d_static_grid.get((ix, iy), []):
                    key = safety_object.object_id
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(safety_object)
        return candidates

    def _rss_2d_query_local_objects(
        self,
        state: State,
        objects: Sequence[SafetyObject],
    ) -> List[SafetyObject]:
        if not bool(getattr(self.config, "enable_spatial_filtering", True)):
            return list(objects)

        ego = self._ego(state)
        local_radius = max(0.0, float(getattr(self.config, "max_check_distance", 40.0)))
        broad_radius = max(local_radius, float(getattr(self.config, "broad_phase_radius", local_radius)))
        dynamic_radius = max(broad_radius, float(getattr(self.config, "dynamic_check_distance", broad_radius)))
        static_rows: List[Tuple[float, float, SafetyObject]] = []
        dynamic_rows: List[Tuple[float, float, SafetyObject]] = []
        if bool(getattr(self.config, "enable_static_safety_object_cache", True)):
            static_candidates = self._query_rss_2d_static_grid(state, broad_radius)
            dynamic_candidates = [obj for obj in objects if obj.is_dynamic]
            candidate_objects = static_candidates + dynamic_candidates
        else:
            candidate_objects = list(objects)

        if bool(getattr(self.config, "enable_vectorized_broad_phase", True)):
            static_rows, dynamic_rows = self._rss_2d_broad_phase_rows_vectorized(
                ego,
                candidate_objects,
                broad_radius,
                dynamic_radius,
            )
        else:
            for safety_object in candidate_objects:
                payload = safety_object.payload
                if payload is None:
                    continue
                delta_s, delta_l = self._relative_position(ego, payload)
                center_distance = math.hypot(delta_s, delta_l)
                radius = max(0.0, float(safety_object.radius))
                check_radius = dynamic_radius if safety_object.is_dynamic else broad_radius
                if center_distance - radius > check_radius:
                    continue
                if abs(delta_s) - radius > check_radius and abs(delta_l) - radius > check_radius:
                    continue
                risk_key = max(0.0, center_distance - radius)
                row = (risk_key, abs(delta_s), safety_object)
                if safety_object.is_dynamic:
                    dynamic_rows.append(row)
                else:
                    static_rows.append(row)

        static_rows.sort(key=lambda item: (item[0], item[1]))
        dynamic_rows.sort(key=lambda item: (item[0], item[1]))
        max_static = max(0, int(getattr(self.config, "max_static_obstacles", 16)))
        max_dynamic = max(0, int(getattr(self.config, "max_dynamic_vehicles", 16)))
        selected_rows = static_rows[:max_static] + dynamic_rows[:max_dynamic]
        selected_rows.sort(key=lambda item: (item[0], item[1]))
        max_total = max(0, int(getattr(self.config, "max_local_safety_objects", 64)))
        return [item[2] for item in selected_rows[:max_total]]

    def _rss_2d_broad_phase_rows_vectorized(
        self,
        ego: Dict[str, Any],
        candidate_objects: Sequence[SafetyObject],
        broad_radius: float,
        dynamic_radius: float,
    ) -> Tuple[List[Tuple[float, float, SafetyObject]], List[Tuple[float, float, SafetyObject]]]:
        active_objects = [item for item in candidate_objects if item.payload is not None]
        if not active_objects:
            return [], []

        centers = np.asarray([item.center for item in active_objects], dtype=float)
        radii = np.asarray([max(0.0, float(item.radius)) for item in active_objects], dtype=float)
        dynamic_mask = np.asarray([bool(item.is_dynamic) for item in active_objects], dtype=bool)
        ego_x = self._safe_float(ego.get("x", 0.0), 0.0)
        ego_y = self._safe_float(ego.get("y", 0.0), 0.0)
        heading = self._safe_float(ego.get("heading", 0.0), 0.0)
        dx = centers[:, 0] - ego_x
        dy = centers[:, 1] - ego_y
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        delta_s = dx * cos_h + dy * sin_h
        delta_l = -dx * sin_h + dy * cos_h
        center_distance = np.hypot(delta_s, delta_l)
        check_radius = np.where(dynamic_mask, float(dynamic_radius), float(broad_radius))
        too_far_radial = center_distance - radii > check_radius
        too_far_box = (np.abs(delta_s) - radii > check_radius) & (np.abs(delta_l) - radii > check_radius)
        keep_indices = np.flatnonzero(~too_far_radial & ~too_far_box)

        static_rows: List[Tuple[float, float, SafetyObject]] = []
        dynamic_rows: List[Tuple[float, float, SafetyObject]] = []
        for index in keep_indices:
            safety_object = active_objects[int(index)]
            risk_key = max(0.0, float(center_distance[index] - radii[index]))
            row = (risk_key, abs(float(delta_s[index])), safety_object)
            if safety_object.is_dynamic:
                dynamic_rows.append(row)
            else:
                static_rows.append(row)
        return static_rows, dynamic_rows

    def _new_rss_2d_profile(self) -> Dict[str, Any]:
        return {
            "build_safety_objects_time": 0.0,
            "query_local_objects_time": 0.0,
            "nominal_eval_time": 0.0,
            "candidate_generation_time": 0.0,
            "candidate_eval_time": 0.0,
            "geometry_distance_time": 0.0,
            "total_filter_time": 0.0,
            "safety_object_count_total": 0,
            "safety_object_count_local": 0,
            "candidate_count": 0,
            "horizon_steps": int(self._rss_2d_prediction_horizon_steps()),
        }

    def _maybe_log_rss_2d_profile(self, profile: Dict[str, Any], mode: str) -> None:
        if not bool(getattr(self.config, "enable_unified_filter_profiling", True)):
            return
        interval = int(getattr(self.config, "debug_log_interval", 50))
        if interval <= 0 or self._rss_2d_filter_step % interval != 0:
            return
        LOGGER.debug(
            "rss_2d_cbf_profile step=%s mode=%s total=%.6f build=%.6f query=%.6f "
            "nominal=%.6f candidate_eval=%.6f geometry=%.6f objects=%s/%s candidates=%s horizon=%s",
            self._rss_2d_filter_step,
            mode,
            float(profile.get("total_filter_time", 0.0)),
            float(profile.get("build_safety_objects_time", 0.0)),
            float(profile.get("query_local_objects_time", 0.0)),
            float(profile.get("nominal_eval_time", 0.0)),
            float(profile.get("candidate_eval_time", 0.0)),
            float(profile.get("geometry_distance_time", 0.0)),
            int(profile.get("safety_object_count_local", 0)),
            int(profile.get("safety_object_count_total", 0)),
            int(profile.get("candidate_count", 0)),
            int(profile.get("horizon_steps", 0)),
        )

    def _rss_2d_prediction_horizon_steps(self) -> int:
        configured = getattr(self.config, "prediction_horizon_steps", None)
        if configured is None:
            configured = getattr(self.config, "rss_2d_prediction_horizon_steps", self.config.horizon_steps)
        return max(0, int(configured))

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
        objects: Sequence[SafetyObject],
        collect_debug: bool = False,
    ) -> Dict[str, Any]:
        action = self._clip_action(action)
        horizon = self._rss_2d_prediction_horizon_steps()
        rollout_state = state
        rollout_objects = [self._copy_safety_object(obj) for obj in objects]
        H = math.inf
        current_H = math.inf
        final_H = math.inf
        worst: Dict[str, Any] = {}
        constraints: List[Dict[str, Any]] = []
        constraint_object_ids = set()
        left_margins: List[float] = []
        right_margins: List[float] = []
        lateral_positions: List[float] = []
        boundary_source = ""
        geometry_distance_time = 0.0

        for step in range(horizon + 1):
            step_constraints: List[Dict[str, Any]] = []
            for safety_object in rollout_objects:
                if safety_object.payload is None:
                    continue
                geometry_start = time.perf_counter()
                margin = self.compute_2d_rss_cbf_margin(
                    rollout_state,
                    safety_object.payload,
                    safety_object.object_kind,
                )
                geometry_distance_time += time.perf_counter() - geometry_start
                constraint = self._rss_2d_object_constraint(
                    safety_object=safety_object,
                    margin=margin,
                    rollout_state=rollout_state,
                    step=step,
                    collect_debug=collect_debug,
                )
                step_constraints.append(constraint)

            geometry_start = time.perf_counter()
            boundary_metrics = self._basic_road_boundary_margins_for_state(state, rollout_state)
            geometry_distance_time += time.perf_counter() - geometry_start
            boundary_source = str(boundary_metrics.get("source", boundary_source))
            left_margin = float(boundary_metrics["left_margin"])
            right_margin = float(boundary_metrics["right_margin"])
            left_margins.append(left_margin)
            right_margins.append(right_margin)
            lateral_positions.append(float(boundary_metrics.get("lateral", 0.0)))
            step_constraints.extend(
                self._rss_2d_boundary_constraints(boundary_metrics, step)
            )

            hard_constraint = self._rss_2d_state_flag_constraint(state, step)
            if hard_constraint is not None:
                step_constraints.append(hard_constraint)

            if collect_debug:
                constraints.extend(step_constraints)
            step_H = min((float(item["h"]) for item in step_constraints), default=math.inf)
            if step == 0:
                current_H = step_H
            final_H = step_H
            for constraint in step_constraints:
                constraint_object_ids.add(str(constraint.get("object_id", "")))
                h_value = float(constraint["h"])
                if h_value < H:
                    H = h_value
                    worst = constraint

            if step >= horizon:
                break

            next_state = self._simulate_next_state(rollout_state, action)
            rollout_objects = [
                self._advance_safety_object_for_rss_cbf(rollout_state, safety_object)
                for safety_object in rollout_objects
            ]
            rollout_state = next_state

        left_min = min(left_margins) if left_margins else math.inf
        right_min = min(right_margins) if right_margins else math.inf
        current_boundary_margin = min(left_margins[0], right_margins[0]) if left_margins and right_margins else math.inf
        final_boundary_margin = min(left_margins[-1], right_margins[-1]) if left_margins and right_margins else math.inf
        current_lateral = lateral_positions[0] if lateral_positions else 0.0
        final_lateral = lateral_positions[-1] if lateral_positions else 0.0
        lower_limit, upper_limit, _ = self._basic_road_boundary_limits_from_state(state)
        center_lateral = 0.5 * (lower_limit + upper_limit)
        centering_improvement = abs(current_lateral - center_lateral) - abs(final_lateral - center_lateral)
        boundary_safe = min(left_min, right_min) >= float(getattr(self.config, "rss_2d_boundary_margin_threshold", 0.5))
        return {
            "safe": bool(H >= -self.config.small_tolerance),
            "objects_safe": bool(H >= -self.config.small_tolerance),
            "H": float(H),
            "current_H": float(current_H),
            "final_H": float(final_H),
            "min_h_2d": float(H),
            "current_h_2d": float(current_H),
            "final_h_2d": float(final_H),
            "worst_h": float(H),
            "worst": worst,
            "constraints": constraints,
            "object_margins": constraints,
            "safety_object_count": len(constraint_object_ids),
            "num_objects": int(len(objects)),
            "horizon_steps": int(horizon),
            "geometry_distance_time": float(geometry_distance_time),
            "road_boundary_safe": bool(boundary_safe),
            "left_boundary_safe": bool(left_min >= float(getattr(self.config, "rss_2d_boundary_margin_threshold", 0.5))),
            "right_boundary_safe": bool(right_min >= float(getattr(self.config, "rss_2d_boundary_margin_threshold", 0.5))),
            "left_boundary_margin": float(left_min),
            "right_boundary_margin": float(right_min),
            "min_boundary_margin": float(min(left_min, right_min)),
            "road_boundary_margin_current": float(current_boundary_margin),
            "road_boundary_margin_min_pred": float(min(left_min, right_min)),
            "road_boundary_margin_final_pred": float(final_boundary_margin),
            "road_boundary_margin_improvement": float(final_boundary_margin - current_boundary_margin),
            "road_boundary_margin_source": boundary_source,
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
            "road_boundary_centering_improvement": float(centering_improvement),
        }

    def _copy_safety_object(self, safety_object: SafetyObject) -> SafetyObject:
        return SafetyObject(
            object_id=safety_object.object_id,
            object_type=safety_object.object_type,
            geometry_type=safety_object.geometry_type,
            object_kind=safety_object.object_kind,
            payload=copy.deepcopy(safety_object.payload),
            is_dynamic=safety_object.is_dynamic,
            center=safety_object.center,
            radius=safety_object.radius,
        )

    def _advance_safety_object_for_rss_cbf(
        self,
        state: State,
        safety_object: SafetyObject,
    ) -> SafetyObject:
        if safety_object.payload is None:
            return safety_object
        next_payload = self._advance_object_for_rss_cbf(
            state,
            safety_object.payload,
            safety_object.object_kind,
        )
        center, radius = self._safety_object_center_radius(next_payload)
        return SafetyObject(
            object_id=safety_object.object_id,
            object_type=safety_object.object_type,
            geometry_type=safety_object.geometry_type,
            object_kind=safety_object.object_kind,
            payload=next_payload,
            is_dynamic=safety_object.is_dynamic,
            center=center,
            radius=radius,
        )

    def _rss_2d_object_constraint(
        self,
        safety_object: SafetyObject,
        margin: Dict[str, Any],
        rollout_state: State,
        step: int,
        collect_debug: bool = False,
    ) -> Dict[str, Any]:
        h_value = float(margin["h_2d"])
        constraint = dict(margin)
        constraint.update(
            {
                "h": h_value,
                "constraint_type": "safety_object",
                "object_id": safety_object.object_id,
                "object_type": safety_object.object_type,
                "object_kind": safety_object.object_kind,
                "geometry_type": safety_object.geometry_type,
                "is_dynamic": bool(safety_object.is_dynamic),
                "step": int(step),
                "object_debug": (
                    self._object_debug(safety_object.payload, rollout_state)
                    if collect_debug and safety_object.payload is not None
                    else {}
                ),
            }
        )
        return constraint

    def _rss_2d_boundary_constraints(
        self,
        boundary_metrics: Dict[str, Any],
        step: int,
    ) -> List[Dict[str, Any]]:
        scale = max(float(getattr(self.config, "rss_2d_boundary_margin_threshold", 0.5)), self.config.small_tolerance)
        constraints = []
        for side, raw_margin in (
            ("left", float(boundary_metrics["left_margin"])),
            ("right", float(boundary_metrics["right_margin"])),
        ):
            h_value = (raw_margin - scale) / scale
            signed_lateral = raw_margin if side == "left" else -raw_margin
            constraints.append(
                {
                    "h": float(h_value),
                    "h_2d": float(h_value),
                    "constraint_type": "safety_object",
                    "object_id": "road_boundary_{}".format(side),
                    "object_type": "road_boundary",
                    "object_kind": "static",
                    "geometry_type": "boundary_segment",
                    "is_dynamic": False,
                    "relation": side,
                    "step": int(step),
                    "delta_s": math.inf,
                    "delta_l": float(signed_lateral),
                    "long_clearance": math.inf,
                    "lat_clearance": float(raw_margin),
                    "d_s_safe": math.inf,
                    "d_l_safe": float(scale),
                    "boundary_margin": float(raw_margin),
                    "object_debug": {},
                }
            )
        return constraints

    def _rss_2d_state_flag_constraint(self, state: State, step: int) -> Optional[Dict[str, Any]]:
        ego = self._ego(state)
        hard_violation = bool(
            not bool(ego.get("on_lane", True))
            or bool(ego.get("out_of_route", False))
            or bool(ego.get("crash_sidewalk", False))
        )
        if not hard_violation:
            return None
        return {
            "h": -1.0,
            "h_2d": -1.0,
            "constraint_type": "safety_object",
            "object_id": "road_departure_state",
            "object_type": "road_departure",
            "object_kind": "static",
            "geometry_type": "state_flag",
            "is_dynamic": False,
            "relation": "road_departure",
            "step": int(step),
            "delta_s": math.nan,
            "delta_l": math.nan,
            "long_clearance": math.nan,
            "lat_clearance": math.nan,
            "d_s_safe": math.nan,
            "d_l_safe": math.nan,
            "object_debug": {},
        }

    def _project_action_for_2d_rss_cbf(
        self,
        state: State,
        objects: Sequence[SafetyObject],
        u_original: Action,
        nominal_eval: Optional[Dict[str, Any]] = None,
        profile: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Action, Dict[str, Any]]:
        profile = profile if profile is not None else self._new_rss_2d_profile()
        candidates: List[Dict[str, Any]] = []
        seen = set()
        acc_candidates: List[float] = []
        steer_candidates: List[float] = []
        search_mode = str(getattr(self.config, "candidate_search_mode", "lazy_coarse_to_fine"))

        if nominal_eval is not None:
            nominal_action = self._clip_action(u_original)
            nominal_key = (round(nominal_action[0], 8), round(nominal_action[1], 8))
            seen.add(nominal_key)
            candidates.append(
                self._make_2d_rss_cbf_candidate(
                    state,
                    objects,
                    u_original,
                    nominal_action,
                    evaluation=nominal_eval,
                    search_stage="nominal",
                )
            )

        stages = ["fine"] if search_mode == "full_grid" else ["coarse", "fine"]
        selected: Optional[Dict[str, Any]] = None
        stopped_after_stage = ""

        for stage in stages:
            generation_start = time.perf_counter()
            actions, stage_accs, stage_steers = self._rss_2d_candidate_actions(u_original, stage)
            profile["candidate_generation_time"] += time.perf_counter() - generation_start
            acc_candidates.extend(stage_accs)
            steer_candidates.extend(stage_steers)

            stage_candidates: List[Dict[str, Any]] = []
            for action in actions:
                key = (round(action[0], 8), round(action[1], 8))
                if key in seen:
                    continue
                seen.add(key)
                eval_start = time.perf_counter()
                candidate = self._make_2d_rss_cbf_candidate(
                    state,
                    objects,
                    u_original,
                    action,
                    search_stage=stage,
                )
                profile["candidate_eval_time"] += time.perf_counter() - eval_start
                profile["geometry_distance_time"] += float(
                    candidate.get("evaluation", {}).get("geometry_distance_time", 0.0)
                )
                stage_candidates.append(candidate)
                candidates.append(candidate)

            selected = self._select_2d_rss_cbf_candidate(candidates, u_original)
            if (
                search_mode == "lazy_coarse_to_fine"
                and stage == "coarse"
                and any(float(candidate.get("H", -math.inf)) >= -self.config.small_tolerance for candidate in stage_candidates)
            ):
                stopped_after_stage = stage
                break

        if selected is None:
            selected = self._select_2d_rss_cbf_candidate(candidates, u_original)
        if selected is None and candidates:
            selected = candidates[0]
        if selected is None:
            selected = self._make_2d_rss_cbf_candidate(
                state,
                objects,
                u_original,
                u_original,
                evaluation=nominal_eval,
                search_stage="empty",
            )

        safe_candidates = [
            candidate
            for candidate in candidates
            if float(candidate.get("H", -math.inf)) >= -self.config.small_tolerance
        ]
        road_safe_candidates = [
            candidate
            for candidate in candidates
            if bool(candidate.get("road_boundary_safe", False))
        ]
        reject_reasons = self._rss_2d_candidate_reject_reasons(candidates)
        compact_candidates = [self._compact_2d_rss_cbf_candidate(candidate) for candidate in candidates]
        profile["candidate_count"] = int(len(candidates))
        acc_candidates = self._unique_float_list(acc_candidates)
        steer_candidates = self._unique_float_list(steer_candidates)
        return selected["action"], {
            "projection_failed": not bool(float(selected.get("H", -math.inf)) >= -self.config.small_tolerance),
            "candidate_count": len(candidates),
            "candidate_reject_reasons": reject_reasons,
            "safe_candidate_count": len(safe_candidates),
            "road_safe_candidate_count": len(road_safe_candidates),
            "acc_candidates": [float(acc) for acc in acc_candidates],
            "steer_candidates": [float(steer) for steer in steer_candidates],
            "steer_sign_convention": "internal_positive_left",
            "candidate_search_mode": search_mode,
            "candidate_search_stopped_after_stage": stopped_after_stage,
            "selected": selected,
            "margins": selected.get("margins", {}),
            "candidates": compact_candidates,
            "reason": (
                "selected_safe_candidate"
                if float(selected.get("H", -math.inf)) >= -self.config.small_tolerance
                else "selected_least_unsafe"
            ),
        }

    def _rss_2d_candidate_actions(
        self,
        u_original: Action,
        stage: str,
    ) -> Tuple[List[Action], List[float], List[float]]:
        if stage == "coarse":
            accs = self._rss_2d_coarse_acc_candidates(u_original)
            steers = self._rss_2d_coarse_steer_candidates(u_original)
        else:
            accs = self._rss_2d_acc_candidates(u_original)
            steers = self._rss_2d_steer_candidates(u_original)

        actions: List[Action] = []
        seen = set()
        for acc in accs:
            for steer in steers:
                action = self._clip_action([float(acc), float(steer)])
                key = (round(action[0], 8), round(action[1], 8))
                if key in seen:
                    continue
                seen.add(key)
                actions.append(action)
        return actions, accs, steers

    def _rss_2d_coarse_acc_candidates(self, u_original: Action) -> List[float]:
        values: List[float] = [float(u_original[0])]
        values.extend(float(value) for value in getattr(self.config, "coarse_acc_samples", ()))
        values.append(float(getattr(self.config, "rss_2d_min_speed_preserve_acc", -0.2)))
        return self._unique_clipped_values(values, index=0)

    def _rss_2d_coarse_steer_candidates(self, u_original: Action) -> List[float]:
        if not bool(getattr(self.config, "rss_2d_sample_steer", True)):
            return [float(u_original[1])]
        values: List[float] = [float(u_original[1]), 0.0]
        values.extend(float(value) for value in getattr(self.config, "coarse_steer_samples", ()))
        return self._unique_clipped_values(values, index=1)

    def _rss_2d_acc_candidates(self, u_original: Action) -> List[float]:
        num_samples = int(getattr(self.config, "fine_acc_samples", 0))
        if num_samples <= 0:
            num_samples = int(getattr(self.config, "rss_2d_acc_samples", 7))
        num_samples = max(2, num_samples)
        values = list(np.linspace(float(u_original[0]), self.config.min_acc, num_samples))
        values.extend([float(u_original[0]), 0.0, float(getattr(self.config, "rss_2d_min_speed_preserve_acc", -0.2))])
        return self._unique_clipped_values(values, index=0)

    def _rss_2d_steer_candidates(self, u_original: Action) -> List[float]:
        if not bool(getattr(self.config, "rss_2d_sample_steer", True)):
            return [float(u_original[1])]

        values: List[float] = [float(u_original[1]), 0.0]
        values.extend(float(value) for value in getattr(self.config, "rss_2d_steer_samples", ()))
        fine_count = int(getattr(self.config, "fine_steer_samples", 0))
        if fine_count > 0:
            values.extend(np.linspace(-self.config.max_steer, self.config.max_steer, max(2, fine_count)))
        return self._unique_clipped_values(values, index=1)

    def _unique_clipped_values(self, values: Sequence[float], index: int) -> List[float]:
        unique: List[float] = []
        seen = set()
        for value in values:
            action = [0.0, 0.0]
            action[index] = float(value)
            clipped = self._clip_action(action)[index]
            key = round(float(clipped), 8)
            if key in seen:
                continue
            seen.add(key)
            unique.append(float(clipped))
        return unique

    def _unique_float_list(self, values: Sequence[float]) -> List[float]:
        unique: List[float] = []
        seen = set()
        for value in values:
            key = round(float(value), 8)
            if key in seen:
                continue
            seen.add(key)
            unique.append(float(value))
        return unique

    def _make_2d_rss_cbf_candidate(
        self,
        state: State,
        objects: Sequence[SafetyObject],
        u_original: Action,
        action: Action,
        evaluation: Optional[Dict[str, Any]] = None,
        search_stage: str = "fine",
    ) -> Dict[str, Any]:
        if evaluation is None:
            evaluation = self._evaluate_2d_rss_cbf_candidate(state, action, objects)
        H = float(evaluation.get("H", -math.inf))
        final_H = float(evaluation.get("final_H", H))
        current_H = float(evaluation.get("current_H", H))
        H_improvement = final_H - current_H
        safe = H >= -self.config.small_tolerance
        speed_preserve_score = min(0.0, float(action[0]) - float(getattr(self.config, "rss_2d_min_speed_preserve_acc", -0.2)))
        intervention_cost = self._action_distance_sq(action, u_original)
        acc_penalty = max(0.0, -float(action[0]))
        steer_penalty = abs(float(action[1] - u_original[1]))
        unsafe_score = (
            H
            + 0.5 * final_H
            + 0.5 * H_improvement
            + float(getattr(self.config, "rss_2d_boundary_penalty_weight", 2.0))
            * float(evaluation.get("road_boundary_margin_improvement", 0.0))
            - float(getattr(self.config, "rss_2d_nominal_action_weight", 0.5)) * intervention_cost
            + float(getattr(self.config, "rss_2d_speed_preserve_weight", 0.3)) * speed_preserve_score
        )
        candidate = {
            "mode": "rss_2d_cbf_candidate",
            "search_stage": search_stage,
            "action": action,
            "safe": safe,
            "H": H,
            "final_H": final_H,
            "current_H": current_H,
            "H_improvement": H_improvement,
            "objects_safe": bool(evaluation["safe"]),
            "road_boundary_safe": bool(evaluation.get("road_boundary_safe", False)),
            "min_h_2d": H,
            "final_h_2d": final_H,
            "min_boundary_margin": float(evaluation.get("min_boundary_margin", math.nan)),
            "left_boundary_margin": float(evaluation.get("left_boundary_margin", math.nan)),
            "right_boundary_margin": float(evaluation.get("right_boundary_margin", math.nan)),
            "final_boundary_margin": float(evaluation.get("road_boundary_margin_final_pred", math.nan)),
            "boundary_margin_improvement": float(evaluation.get("road_boundary_margin_improvement", math.nan)),
            "speed_preserve_score": float(speed_preserve_score),
            "selection_score": float(unsafe_score),
            "acc_penalty": float(acc_penalty),
            "steer_penalty": float(steer_penalty),
            "evaluation": evaluation,
            "margins": {
                "H": H,
                "final_H": final_H,
                "H_improvement": H_improvement,
                "min_h_2d": H,
                "current_h_2d": current_H,
                "final_h_2d": final_H,
            },
            "intervention_cost": intervention_cost,
            "progress_score": self._score_progress_after_rollout(state, action),
            "reject_reason": "",
        }
        if not safe:
            candidate["reject_reason"] = self._rss_2d_candidate_reject_reason(evaluation)
        return candidate

    def _compact_2d_rss_cbf_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "mode": candidate.get("mode", ""),
            "search_stage": candidate.get("search_stage", ""),
            "action": list(candidate.get("action", [])),
            "safe": bool(candidate.get("safe", False)),
            "H": float(candidate.get("H", math.nan)),
            "final_H": float(candidate.get("final_H", math.nan)),
            "H_improvement": float(candidate.get("H_improvement", math.nan)),
            "min_boundary_margin": float(candidate.get("min_boundary_margin", math.nan)),
            "final_boundary_margin": float(candidate.get("final_boundary_margin", math.nan)),
            "boundary_margin_improvement": float(candidate.get("boundary_margin_improvement", math.nan)),
            "intervention_cost": float(candidate.get("intervention_cost", math.nan)),
            "selection_score": float(candidate.get("selection_score", math.nan)),
            "reject_reason": candidate.get("reject_reason", ""),
        }

    def _select_2d_rss_cbf_candidate(
        self,
        candidates: Sequence[Dict[str, Any]],
        u_original: Action,
    ) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None
        safe_candidates = [
            candidate
            for candidate in candidates
            if float(candidate.get("H", -math.inf)) >= -self.config.small_tolerance
        ]
        if safe_candidates:
            return max(
                safe_candidates,
                key=lambda item: (
                    -float(item.get("intervention_cost", math.inf)),
                    float(item.get("H", -math.inf)),
                    float(item.get("min_boundary_margin", -math.inf)),
                    float(item.get("boundary_margin_improvement", -math.inf)),
                    -float(item.get("acc_penalty", math.inf)),
                    -float(item.get("steer_penalty", math.inf)),
                ),
            )
        return max(candidates, key=self._rss_2d_candidate_sort_key)

    def _rss_2d_candidate_sort_key(self, item: Dict[str, Any]) -> Tuple[float, ...]:
        return (
            float(item.get("H", -math.inf)),
            float(item.get("final_H", -math.inf)),
            float(item.get("H_improvement", -math.inf)),
            float(item.get("min_boundary_margin", -math.inf)),
            float(item.get("final_boundary_margin", -math.inf)),
            float(item.get("boundary_margin_improvement", -math.inf)),
            float(item.get("selection_score", -math.inf)),
            -float(item.get("intervention_cost", math.inf)),
            -float(item.get("acc_penalty", math.inf)),
            -float(item.get("steer_penalty", math.inf)),
        )

    def _rss_2d_candidate_reject_reason(self, evaluation: Dict[str, Any]) -> str:
        if float(evaluation.get("H", math.inf)) < -self.config.small_tolerance:
            worst = evaluation.get("worst", {}) if isinstance(evaluation, dict) else {}
            object_type = str(worst.get("object_type", "unknown"))
            return "unified_H_violation:{}".format(object_type)
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
        objects: Sequence[SafetyObject],
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
        profile = projection_debug.get("profile", {}) if isinstance(projection_debug, dict) else {}
        road_safe = bool(selected_eval.get("road_boundary_safe", True))
        info = {
            "mode": mode,
            "reason": reason,
            "safety_function_mode": "unified_safety_object_2d_cbf",
            "rss_cbf_variant": "unified_safety_object_2d_cbf",
            "cbf_mode": mode,
            "min_h_2d": float(selected_eval.get("min_h_2d", math.inf)),
            "current_h_2d": float(selected_eval.get("current_h_2d", math.inf)),
            "final_h_2d": float(selected_eval.get("final_h_2d", math.inf)),
            "nominal_H": float(nominal_eval.get("H", nominal_eval.get("min_h_2d", math.inf))),
            "selected_H": float(selected_eval.get("H", selected_eval.get("min_h_2d", math.inf))),
            "worst_h": float(worst.get("h", selected_eval.get("H", math.inf))) if worst else math.inf,
            "safety_object_count": int(selected_eval.get("safety_object_count", 0)),
            "fallback_used": False,
            "selected_steering_changed": bool(abs(float(u_safe[1]) - float(u_original[1])) > self.config.small_tolerance),
            "excessive_braking": bool(
                float(u_safe[0]) < float(getattr(self.config, "rss_2d_min_speed_preserve_acc", -0.2))
                - self.config.small_tolerance
            ),
            "nominal_min_h_2d": float(nominal_eval.get("min_h_2d", math.inf)),
            "nominal_final_h_2d": float(nominal_eval.get("final_h_2d", math.inf)),
            "nominal_hard_safe": bool(float(nominal_eval.get("H", -math.inf)) >= -self.config.small_tolerance),
            "nominal_boundary_margin_above_threshold": bool(nominal_eval.get("road_boundary_safe", False)),
            "rss_2d_safety_margin": float(getattr(self.config, "rss_2d_safety_margin", 0.0)),
            "rss_2d_boundary_margin_threshold": float(
                getattr(self.config, "rss_2d_boundary_margin_threshold", math.nan)
            ),
            "worst_object_geometry_type": worst.get("geometry_type", "") if worst else "",
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
            "left_boundary_margin": float(selected_eval.get("left_boundary_margin", math.nan)),
            "right_boundary_margin": float(selected_eval.get("right_boundary_margin", math.nan)),
            "min_boundary_margin": float(selected_eval.get("min_boundary_margin", math.nan)),
            "road_boundary_margin_current": float(selected_eval.get("road_boundary_margin_current", math.nan)),
            "road_boundary_margin_min_pred": float(selected_eval.get("road_boundary_margin_min_pred", math.nan)),
            "road_boundary_margin_final_pred": float(selected_eval.get("road_boundary_margin_final_pred", math.nan)),
            "road_boundary_margin_improvement": float(selected_eval.get("road_boundary_margin_improvement", math.nan)),
            "road_boundary_margin_source": selected_eval.get("road_boundary_margin_source", ""),
            "predicted_lateral_position": float(selected_eval.get("predicted_lateral_position", math.nan)),
            "predicted_lane_offset": float(selected_eval.get("predicted_lane_offset", math.nan)),
            "road_boundary_centering_improvement": float(selected_eval.get("road_boundary_centering_improvement", math.nan)),
            "selected_action": list(u_safe),
            "nominal_action": list(u_original),
            "filter_intervened": bool(self._action_distance_sq(u_safe, u_original) > self.config.small_tolerance),
            "candidate_reject_reasons": projection_debug.get("candidate_reject_reasons", ""),
            "candidate_count": int(projection_debug.get("candidate_count", 0)),
            "rss_2d_candidate_search_mode": projection_debug.get("candidate_search_mode", ""),
            "rss_2d_candidate_search_stopped_after_stage": projection_debug.get(
                "candidate_search_stopped_after_stage", ""
            ),
            "safe_candidate_count": int(projection_debug.get("safe_candidate_count", 0)),
            "road_safe_candidate_count": int(projection_debug.get("road_safe_candidate_count", 0)),
            "rss_2d_acc_candidate_count": len(projection_debug.get("acc_candidates", [])),
            "rss_2d_steer_candidate_count": len(projection_debug.get("steer_candidates", [])),
            "build_safety_objects_time": float(profile.get("build_safety_objects_time", 0.0)),
            "query_local_objects_time": float(profile.get("query_local_objects_time", 0.0)),
            "nominal_eval_time": float(profile.get("nominal_eval_time", 0.0)),
            "candidate_generation_time": float(profile.get("candidate_generation_time", 0.0)),
            "candidate_eval_time": float(profile.get("candidate_eval_time", 0.0)),
            "geometry_distance_time": float(profile.get("geometry_distance_time", 0.0)),
            "total_filter_time": float(profile.get("total_filter_time", 0.0)),
            "safety_object_count_total": int(profile.get("safety_object_count_total", len(objects))),
            "safety_object_count_local": int(profile.get("safety_object_count_local", len(objects))),
            "horizon_steps": int(profile.get("horizon_steps", selected_eval.get("horizon_steps", 0))),
            "selected_min_boundary_margin": float(selected.get("min_boundary_margin", math.nan)) if isinstance(selected, dict) else math.nan,
            "selected_final_boundary_margin": float(selected.get("final_boundary_margin", math.nan)) if isinstance(selected, dict) else math.nan,
            "selected_boundary_margin_improvement": float(selected.get("boundary_margin_improvement", math.nan)) if isinstance(selected, dict) else math.nan,
            "selected_centering_score": float(selected.get("centering_score", math.nan)) if isinstance(selected, dict) else math.nan,
            "selected_speed_preserve_score": float(selected.get("speed_preserve_score", math.nan)) if isinstance(selected, dict) else math.nan,
            "selected_selection_score": float(selected.get("selection_score", math.nan)) if isinstance(selected, dict) else math.nan,
            "selected_acc_penalty": float(selected.get("acc_penalty", math.nan)) if isinstance(selected, dict) else math.nan,
            "selected_steer_penalty": float(selected.get("steer_penalty", math.nan)) if isinstance(selected, dict) else math.nan,
            "rss_margin": float(selected_eval.get("min_h_2d", math.inf)),
            "rss_distance": d_s_safe,
            "d_front": long_clearance,
            "d_dynamic": d_s_safe if object_kind == "dynamic" else math.nan,
            "d_obs": long_clearance if object_kind == "static" else math.nan,
            "d_brake": d_s_safe,
            "object_kind": object_kind,
            "dynamic_vehicle_detected": any(obj.object_kind == "dynamic" for obj in objects),
            "obstacle_detected": any(obj.object_kind == "static" for obj in objects),
            "left_feasible": bool(selected_eval.get("left_boundary_safe", False)),
            "right_feasible": bool(selected_eval.get("right_boundary_safe", False)),
            "state_debug": self._state_debug(state),
            "blocking_object": worst_debug,
            "nominal_margins": {
                "H": float(nominal_eval.get("H", math.inf)),
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
