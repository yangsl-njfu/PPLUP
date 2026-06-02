"""RSS-CBF runtime assurance filter for PPL/TD3 policy evaluation.

The public runtime path is intentionally small:

    u_nom -> RSSCBF Runtime Assurance -> u_safe

The filter keeps the existing MetaDrive adapters and action conversion helpers
from ``StaticRSSFilter``, but exposes only the RSS-CBF forward-distance safety
contract for evaluation.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
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
    enable_frenet_coordinates: bool = True
    frenet_fallback_to_ego_local: bool = True
    frenet_debug_log: bool = True
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
    adaptive_recovery_base_delta_h_weight: float = 8.0
    adaptive_recovery_risk_delta_h_weight: float = 4.0
    adaptive_recovery_h_weight: float = 4.0
    adaptive_recovery_base_center_weight: float = 3.0
    adaptive_recovery_risk_center_weight: float = 2.0
    adaptive_recovery_center_distance_weight: float = 0.25
    adaptive_recovery_base_speed_weight: float = 0.5
    adaptive_recovery_risk_speed_weight: float = 1.5
    adaptive_recovery_current_speed_weight: float = 0.1
    adaptive_recovery_action_weight: float = 0.05
    adaptive_recovery_smooth_weight: float = 0.25
    adaptive_recovery_delta_h_tie_tolerance: float = 0.02
    rss_2d_recovery_accel_weight: float = 2.0
    rss_2d_recovery_action_weight: float = 0.1
    unsafe_recovery_force_brake: bool = True
    unsafe_recovery_speed_threshold: float = 3.0
    unsafe_recovery_deep_violation_threshold: float = -0.5
    unsafe_recovery_preferred_acc_max: float = -0.2
    unsafe_recovery_brake_delta_h_tolerance: float = 0.05
    boundary_recovery_horizon_steps: int = 3
    rss_2d_recovery_zero_acc_fast_weight: float = 10.0
    rss_2d_recovery_speed_weight: float = 0.2
    enable_unified_filter_profiling: bool = True
    debug_log_interval: int = 50
    rss_debug_log_level: str = "event"
    rss_debug_log_to_console: bool = False
    rss_debug_log_to_file: bool = True
    rss_debug_log_file: str = "logs/rss_cbf_debug.jsonl"
    rss_debug_log_interval: int = 100
    rss_debug_summary_interval: int = 100
    rss_debug_log_only_intervention: bool = True
    rss_debug_log_only_anomaly: bool = True
    rss_debug_log_near_margin: float = 0.2
    rss_debug_log_large_drop_threshold: float = 0.1
    rss_debug_profile_interval: int = 100
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
    ROAD_SAFETY_OBJECT_KINDS = {"boundary", "no_drive_area", "road_edge"}
    ROAD_BOUNDARY_OBJECT_KINDS = {"boundary", "road_edge"}

    def __init__(self, config: Optional[RSSCBFConfig] = None):
        super().__init__(config or RSSCBFConfig())
        self._rss_2d_static_cache_signature: Optional[Tuple[Any, ...]] = None
        self._rss_2d_static_cache: List[SafetyObject] = []
        self._rss_2d_static_grid: Dict[Tuple[int, int], List[SafetyObject]] = {}
        self._rss_2d_static_grid_cell_size: float = 1.0
        self._rss_2d_filter_step: int = 0
        self._rss_2d_last_profile: Dict[str, Any] = {}
        self._rss_2d_prev_selected_action: Optional[Action] = None
        self._rss_debug_recent_interventions: int = 0
        self._rss_debug_file_warning_emitted: bool = False

    def reset(self) -> None:
        """Clear per-episode RSS-2D caches without changing configuration."""
        self._rss_2d_static_cache_signature = None
        self._rss_2d_static_cache = []
        self._rss_2d_static_grid = {}
        self._rss_2d_static_grid_cell_size = 1.0
        self._rss_2d_filter_step = 0
        self._rss_2d_last_profile = {}
        self._rss_2d_prev_selected_action = None
        self._rss_debug_recent_interventions = 0
        self._rss_debug_file_warning_emitted = False

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
        nominal_eval["safe"] = bool(float(nominal_eval.get("H", -math.inf)) >= 0.0)
        lazy_margin = float(getattr(self.config, "lazy_safety_margin", safety_margin))

        if float(nominal_eval.get("H", -math.inf)) >= lazy_margin:
            profile["total_filter_time"] = time.perf_counter() - total_start
            profile["candidate_count"] = 0
            self._rss_2d_last_profile = dict(profile)
            self._maybe_log_rss_2d_profile(profile, mode="normal")
            info = self._make_rss_2d_cbf_info(
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
            self._rss_2d_prev_selected_action = self._clip_action(u_original)
            return u_original, info

        u_safe, projection_debug = self._project_action_for_2d_rss_cbf(
            state,
            objects,
            u_original,
            nominal_eval=nominal_eval,
            profile=profile,
        )
        selected = projection_debug.get("selected", {}) if isinstance(projection_debug, dict) else {}
        selected_eval = selected.get("evaluation", nominal_eval) if isinstance(selected, dict) else nominal_eval
        selected_safe = bool(float(selected.get("H", selected_eval.get("H", -math.inf))) >= 0.0)

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

        info = self._make_rss_2d_cbf_info(
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
        self._rss_2d_prev_selected_action = self._clip_action(u_safe)
        return u_safe, info

    def _rss_2d_relevant_objects(self, state: State) -> List[SafetyObject]:
        all_objects = self._rss_2d_build_safety_objects(state)
        return self._rss_2d_query_local_objects(state, all_objects)

    def _rss_2d_build_safety_objects(self, state: State) -> List[SafetyObject]:
        static_obstacles = list(state.get("static_obstacles", []) or [])
        static_objects: List[SafetyObject]
        signature = self._rss_2d_static_obstacle_signature(static_obstacles)
        cache_enabled = bool(getattr(self.config, "enable_static_safety_object_cache", True)) and not bool(
            getattr(self.config, "enable_frenet_coordinates", False)
        )
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
        objects.extend(self._make_road_safety_objects(state))
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

    def _make_road_safety_objects(self, state: State) -> List[SafetyObject]:
        return [
            self._make_boundary_safety_object(state, "left"),
            self._make_boundary_safety_object(state, "right"),
            self._make_no_drive_area_safety_object(state),
        ]

    def _make_boundary_safety_object(self, state: State, side: str) -> SafetyObject:
        ego = self._ego(state)
        object_kind = self._road_boundary_object_kind_for_side(state, side)
        line_type = str(ego.get("{}_lane_line_type".format(side), "") or "")
        line_color = str(ego.get("{}_lane_line_color".format(side), "") or "")
        object_type = self._road_boundary_object_type(object_kind, line_type, line_color)
        payload = {
            "object_id": "road_boundary_{}".format(side),
            "object_type": object_type,
            "object_kind": object_kind,
            "geometry_type": "signed_boundary",
            "side": side,
            "boundary_side": side,
            "line_type": line_type,
            "line_color": line_color,
            "line_prohibited": bool(ego.get("{}_lane_line_prohibited".format(side), False)),
            "x": self._safe_float(ego.get("x", 0.0), 0.0),
            "y": self._safe_float(ego.get("y", 0.0), 0.0),
            "heading": self._safe_float(ego.get("heading", 0.0), 0.0),
            "speed": 0.0,
            "length": 0.0,
            "width": 0.0,
        }
        return SafetyObject(
            object_id=str(payload["object_id"]),
            object_type=object_type,
            geometry_type="signed_boundary",
            object_kind=object_kind,
            payload=payload,
            is_dynamic=False,
            center=(float(payload["x"]), float(payload["y"])),
            radius=0.0,
        )

    def _make_no_drive_area_safety_object(self, state: State) -> SafetyObject:
        ego = self._ego(state)
        payload = {
            "object_id": "no_drive_area_state",
            "object_type": "no_drive_area",
            "object_kind": "no_drive_area",
            "geometry_type": "state_flag",
            "x": self._safe_float(ego.get("x", 0.0), 0.0),
            "y": self._safe_float(ego.get("y", 0.0), 0.0),
            "heading": self._safe_float(ego.get("heading", 0.0), 0.0),
            "speed": 0.0,
            "length": 0.0,
            "width": 0.0,
        }
        return SafetyObject(
            object_id=str(payload["object_id"]),
            object_type="no_drive_area",
            geometry_type="state_flag",
            object_kind="no_drive_area",
            payload=payload,
            is_dynamic=False,
            center=(float(payload["x"]), float(payload["y"])),
            radius=0.0,
        )

    def _road_boundary_object_kind_for_side(self, state: State, side: str) -> str:
        ego = self._ego(state)
        lanes = state.get("lanes", {}) or {}
        adjacent_lane = lanes.get(side) or {}
        line_type = str(ego.get("{}_lane_line_type".format(side), "") or "").upper()
        line_color = str(ego.get("{}_lane_line_color".format(side), "") or "").upper()
        adjacent_available = bool(adjacent_lane.get("available", False)) and bool(adjacent_lane.get("drivable", True))
        edge_marker = "SIDE" in line_type or "GUARDRAIL" in line_type or "CURB" in line_type
        if "YELLOW" in line_color:
            return "boundary"
        if edge_marker or not adjacent_available:
            return "road_edge"
        if bool(ego.get("{}_lane_line_prohibited".format(side), False)):
            return "boundary"
        return "road_edge"

    def _road_boundary_object_type(self, object_kind: str, line_type: str, line_color: str) -> str:
        if object_kind == "road_edge":
            return "road_edge"
        if "YELLOW" in str(line_color).upper():
            return "yellow_line"
        if line_type:
            return "lane_boundary"
        return "road_boundary"

    def _is_rss_2d_road_safety_object(self, safety_object: SafetyObject) -> bool:
        return str(safety_object.object_kind) in self.ROAD_SAFETY_OBJECT_KINDS

    def _is_rss_2d_boundary_safety_object(self, safety_object: SafetyObject) -> bool:
        return str(safety_object.object_kind) in self.ROAD_BOUNDARY_OBJECT_KINDS

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

        road_objects = [obj for obj in objects if self._is_rss_2d_road_safety_object(obj)]
        spatial_objects = [obj for obj in objects if not self._is_rss_2d_road_safety_object(obj)]
        ego = self._ego(state)
        local_radius = max(0.0, float(getattr(self.config, "max_check_distance", 40.0)))
        broad_radius = max(local_radius, float(getattr(self.config, "broad_phase_radius", local_radius)))
        dynamic_radius = max(broad_radius, float(getattr(self.config, "dynamic_check_distance", broad_radius)))
        static_rows: List[Tuple[float, float, SafetyObject]] = []
        dynamic_rows: List[Tuple[float, float, SafetyObject]] = []
        static_cache_enabled = bool(getattr(self.config, "enable_static_safety_object_cache", True)) and not bool(
            getattr(self.config, "enable_frenet_coordinates", False)
        )
        if static_cache_enabled:
            static_candidates = self._query_rss_2d_static_grid(state, broad_radius)
            dynamic_candidates = [obj for obj in spatial_objects if obj.is_dynamic]
            candidate_objects = static_candidates + dynamic_candidates
        else:
            candidate_objects = list(spatial_objects)

        if bool(getattr(self.config, "enable_vectorized_broad_phase", True)):
            static_rows, dynamic_rows = self._rss_2d_broad_phase_rows_vectorized(
                state,
                candidate_objects,
                broad_radius,
                dynamic_radius,
            )
        else:
            for safety_object in candidate_objects:
                payload = safety_object.payload
                if payload is None:
                    continue
                relative = self._rss_2d_relative_position_metrics(state, payload)
                delta_s = float(relative["delta_s"])
                delta_l = float(relative["delta_l"])
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
        spatial_budget = max(0, max_total - len(road_objects)) if max_total > 0 else 0
        selected_objects = [item[2] for item in selected_rows[:spatial_budget]]
        selected_objects.extend(road_objects)
        return selected_objects

    def _rss_2d_broad_phase_rows_vectorized(
        self,
        state: State,
        candidate_objects: Sequence[SafetyObject],
        broad_radius: float,
        dynamic_radius: float,
    ) -> Tuple[List[Tuple[float, float, SafetyObject]], List[Tuple[float, float, SafetyObject]]]:
        active_objects = [item for item in candidate_objects if item.payload is not None]
        if not active_objects:
            return [], []

        static_rows: List[Tuple[float, float, SafetyObject]] = []
        dynamic_rows: List[Tuple[float, float, SafetyObject]] = []
        for safety_object in active_objects:
            payload = safety_object.payload
            if payload is None:
                continue
            relative = self._rss_2d_relative_position_metrics(state, payload)
            delta_s = float(relative["delta_s"])
            delta_l = float(relative["delta_l"])
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
        log_level = self._rss_debug_log_level()
        if log_level == "off":
            return
        interval = int(getattr(self.config, "rss_debug_profile_interval", 100))
        if interval <= 0:
            interval = int(getattr(self.config, "debug_log_interval", 50))
        if interval <= 0 or self._rss_2d_filter_step % interval != 0:
            return
        if not (self._rss_debug_console_enabled("profile") or self._rss_debug_file_enabled("profile")):
            return
        self._emit_rss_debug_log(
            {
                "event": "profile",
                "step": int(self._rss_2d_filter_step),
                "mode": mode,
                "filter_time_ms": 1000.0 * float(profile.get("total_filter_time", 0.0)),
                "build_objects_time_ms": 1000.0 * float(profile.get("build_safety_objects_time", 0.0)),
                "candidate_eval_time_ms": 1000.0 * float(profile.get("candidate_eval_time", 0.0)),
                "num_safety_objects": int(profile.get("safety_object_count_local", 0)),
                "num_candidates": int(profile.get("candidate_count", 0)),
            },
            level="summary",
        )

    def _rss_2d_prediction_horizon_steps(self) -> int:
        configured = getattr(self.config, "prediction_horizon_steps", None)
        if configured is None:
            configured = getattr(self.config, "rss_2d_prediction_horizon_steps", self.config.horizon_steps)
        boundary_recovery_horizon = int(getattr(self.config, "boundary_recovery_horizon_steps", 0))
        return max(0, int(configured), boundary_recovery_horizon)

    def _rss_2d_reference_lane(self, state: State) -> Any:
        lane = state.get("_frenet_reference_lane", None)
        if lane is not None:
            return lane
        ego_source = self._runtime_entity_source(self._ego(state))
        if ego_source is not None:
            lane = self._metadrive_current_lane(ego_source)
            if lane is not None:
                self._last_frenet_reference_lane = lane
                return lane
        return self._last_frenet_reference_lane

    def _runtime_entity_source(self, entity: Any) -> Any:
        if isinstance(entity, dict):
            return entity.get("_metadrive_source", None)
        return entity

    def _rss_project_point_to_lane(self, point: Any, lane: Any) -> Dict[str, Any]:
        if lane is None:
            return {
                "valid": False,
                "s": math.nan,
                "l": math.nan,
                "heading_ref": math.nan,
                "reason": "missing_reference_lane",
            }
        if point is None:
            return {
                "valid": False,
                "s": math.nan,
                "l": math.nan,
                "heading_ref": math.nan,
                "reason": "missing_position",
            }
        try:
            s_value, l_value = lane.local_coordinates(point)
            s_value = float(s_value)
            l_value = float(l_value)
        except Exception as exc:
            return {
                "valid": False,
                "s": math.nan,
                "l": math.nan,
                "heading_ref": math.nan,
                "reason": "local_coordinates_failed:{}".format(type(exc).__name__),
            }
        heading_ref, heading_reason = self._lane_heading_ref(lane, s_value)
        if not math.isfinite(heading_ref):
            return {
                "valid": False,
                "s": s_value,
                "l": l_value,
                "heading_ref": math.nan,
                "reason": heading_reason,
            }
        return {
            "valid": True,
            "s": s_value,
            "l": l_value,
            "heading_ref": heading_ref,
            "reason": "",
        }

    def _rss_project_entity_to_frenet(self, state: State, entity: Dict[str, Any], role: str) -> Dict[str, Any]:
        if not bool(getattr(self.config, "enable_frenet_coordinates", False)):
            return {
                "valid": False,
                "mode": "ego_local_fallback",
                "s": math.nan,
                "l": math.nan,
                "heading_ref": math.nan,
                "v_s": math.nan,
                "v_l": math.nan,
                "ref_lane_valid": False,
                "reason": "disabled",
            }

        lane = self._rss_2d_reference_lane(state)
        source = self._runtime_entity_source(entity)
        computed_reason = ""
        if source is not None and lane is not None:
            projection = self._rss_project_point_to_lane(self._frenet_position(source), lane)
            if bool(projection.get("valid", False)):
                heading_ref = float(projection["heading_ref"])
                heading = self._entity_heading(source, self._entity_heading(entity, heading_ref))
                speed = self._entity_speed(source, self._entity_speed(entity, 0.0))
                heading_error = heading - heading_ref
                return {
                    "valid": True,
                    "mode": "computed_frenet",
                    "s": float(projection["s"]),
                    "l": float(projection["l"]),
                    "heading_ref": heading_ref,
                    "v_s": float(speed * math.cos(heading_error)),
                    "v_l": float(speed * math.sin(heading_error)),
                    "ref_lane_valid": True,
                    "reason": "",
                }
            computed_reason = str(projection.get("reason", "projection_failed"))

        payload_s = self._safe_float(entity.get("frenet_s", math.nan), math.nan)
        payload_l = self._safe_float(entity.get("frenet_l", math.nan), math.nan)
        payload_valid = bool(entity.get("frenet_valid", False)) and math.isfinite(payload_s) and math.isfinite(payload_l)
        if payload_valid:
            return {
                "valid": True,
                "mode": str(entity.get("frenet_source_mode", "payload_frenet")),
                "s": payload_s,
                "l": payload_l,
                "heading_ref": self._safe_float(entity.get("frenet_heading_ref", math.nan), math.nan),
                "v_s": self._safe_float(entity.get("frenet_v_s", math.nan), math.nan),
                "v_l": self._safe_float(entity.get("frenet_v_l", math.nan), math.nan),
                "ref_lane_valid": lane is not None,
                "reason": "",
            }

        if computed_reason:
            reason = computed_reason
        elif lane is None:
            reason = "missing_reference_lane"
        elif source is None:
            reason = "missing_runtime_source"
        else:
            reason = str(entity.get("frenet_fallback_reason", "") or "frenet_invalid")
        return {
            "valid": False,
            "mode": "ego_local_fallback",
            "s": self._safe_float(entity.get("frenet_s", math.nan), math.nan),
            "l": self._safe_float(entity.get("frenet_l", math.nan), math.nan),
            "heading_ref": self._safe_float(entity.get("frenet_heading_ref", math.nan), math.nan),
            "v_s": self._safe_float(entity.get("frenet_v_s", math.nan), math.nan),
            "v_l": self._safe_float(entity.get("frenet_v_l", math.nan), math.nan),
            "ref_lane_valid": lane is not None,
            "reason": reason,
        }

    def _rss_2d_relative_position_metrics(self, state: State, obj: Dict[str, Any]) -> Dict[str, Any]:
        ego = self._ego(state)
        old_delta_s, old_delta_l = self._ego_local_relative_position(ego, obj)
        fallback_allowed = bool(getattr(self.config, "frenet_fallback_to_ego_local", True))
        ego_frenet = self._rss_project_entity_to_frenet(state, ego, role="ego")
        obj_frenet = self._rss_project_entity_to_frenet(state, obj, role="object")
        if bool(ego_frenet.get("valid", False)) and bool(obj_frenet.get("valid", False)):
            s_ego = float(ego_frenet["s"])
            l_ego = float(ego_frenet["l"])
            s_obj = float(obj_frenet["s"])
            l_obj = float(obj_frenet["l"])
            coordinate_mode = (
                "computed_frenet"
                if ego_frenet.get("mode") == "computed_frenet" or obj_frenet.get("mode") == "computed_frenet"
                else "payload_frenet"
            )
            return {
                "delta_s": float(s_obj - s_ego),
                "delta_l": float(l_obj - l_ego),
                "coordinate_mode": coordinate_mode,
                "frenet_valid": True,
                "frenet_fallback_reason": "",
                "ego_ref_lane_valid": bool(ego_frenet.get("ref_lane_valid", False)),
                "ego_frenet_valid": True,
                "object_frenet_valid": True,
                "s_ego": s_ego,
                "l_ego": l_ego,
                "heading_ref_ego": self._safe_float(ego_frenet.get("heading_ref", math.nan), math.nan),
                "v_ego_s": self._safe_float(ego_frenet.get("v_s", math.nan), math.nan),
                "v_ego_l": self._safe_float(ego_frenet.get("v_l", math.nan), math.nan),
                "s_obj": s_obj,
                "l_obj": l_obj,
                "heading_ref_obj": self._safe_float(obj_frenet.get("heading_ref", math.nan), math.nan),
                "v_obj_s": self._safe_float(obj_frenet.get("v_s", math.nan), math.nan),
                "v_obj_l": self._safe_float(obj_frenet.get("v_l", math.nan), math.nan),
                "old_delta_s": float(old_delta_s),
                "old_delta_l": float(old_delta_l),
                "old_ego_local_delta_s": float(old_delta_s),
                "old_ego_local_delta_l": float(old_delta_l),
                "ego_s": s_ego,
                "ego_l": l_ego,
                "ego_v_s": self._safe_float(ego_frenet.get("v_s", math.nan), math.nan),
                "ego_heading_ref": self._safe_float(ego_frenet.get("heading_ref", math.nan), math.nan),
                "object_s": s_obj,
                "object_l": l_obj,
                "object_v_s": self._safe_float(obj_frenet.get("v_s", math.nan), math.nan),
            }

        if not fallback_allowed and bool(getattr(self.config, "enable_frenet_coordinates", False)):
            reason = "frenet_invalid"
        else:
            reason = str(ego_frenet.get("reason", "") or obj_frenet.get("reason", "") or "frenet_invalid")
        return {
            "delta_s": float(old_delta_s),
            "delta_l": float(old_delta_l),
            "coordinate_mode": "ego_local_fallback",
            "frenet_valid": False,
            "frenet_fallback_reason": reason,
            "ego_ref_lane_valid": bool(ego_frenet.get("ref_lane_valid", False)),
            "ego_frenet_valid": bool(ego_frenet.get("valid", False)),
            "object_frenet_valid": bool(obj_frenet.get("valid", False)),
            "s_ego": self._safe_float(ego_frenet.get("s", math.nan), math.nan),
            "l_ego": self._safe_float(ego_frenet.get("l", math.nan), math.nan),
            "heading_ref_ego": self._safe_float(ego_frenet.get("heading_ref", math.nan), math.nan),
            "v_ego_s": self._safe_float(ego_frenet.get("v_s", math.nan), math.nan),
            "v_ego_l": self._safe_float(ego_frenet.get("v_l", math.nan), math.nan),
            "s_obj": self._safe_float(obj_frenet.get("s", math.nan), math.nan),
            "l_obj": self._safe_float(obj_frenet.get("l", math.nan), math.nan),
            "heading_ref_obj": self._safe_float(obj_frenet.get("heading_ref", math.nan), math.nan),
            "v_obj_s": self._safe_float(obj_frenet.get("v_s", math.nan), math.nan),
            "v_obj_l": self._safe_float(obj_frenet.get("v_l", math.nan), math.nan),
            "old_delta_s": float(old_delta_s),
            "old_delta_l": float(old_delta_l),
            "old_ego_local_delta_s": float(old_delta_s),
            "old_ego_local_delta_l": float(old_delta_l),
            "ego_s": self._safe_float(ego_frenet.get("s", math.nan), math.nan),
            "ego_l": self._safe_float(ego_frenet.get("l", math.nan), math.nan),
            "ego_v_s": self._safe_float(ego_frenet.get("v_s", math.nan), math.nan),
            "ego_heading_ref": self._safe_float(ego_frenet.get("heading_ref", math.nan), math.nan),
            "object_s": self._safe_float(obj_frenet.get("s", math.nan), math.nan),
            "object_l": self._safe_float(obj_frenet.get("l", math.nan), math.nan),
            "object_v_s": self._safe_float(obj_frenet.get("v_s", math.nan), math.nan),
        }

    def compute_2d_rss_cbf_margin(
        self,
        state: State,
        obj: Dict[str, Any],
        object_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute one RSS-informed 2D CBF margin in the configured coordinate frame."""
        ego = self._ego(state)
        relative = self._rss_2d_relative_position_metrics(state, obj)
        delta_s = float(relative["delta_s"])
        delta_l = float(relative["delta_l"])
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
            "coordinate_mode": str(relative.get("coordinate_mode", "ego_local_fallback")),
            "frenet_valid": bool(relative.get("frenet_valid", False)),
            "frenet_fallback_reason": str(relative.get("frenet_fallback_reason", "")),
            "ego_ref_lane_valid": bool(relative.get("ego_ref_lane_valid", False)),
            "ego_frenet_valid": bool(relative.get("ego_frenet_valid", False)),
            "object_frenet_valid": bool(relative.get("object_frenet_valid", False)),
            "s_ego": float(relative.get("s_ego", math.nan)),
            "l_ego": float(relative.get("l_ego", math.nan)),
            "heading_ref_ego": float(relative.get("heading_ref_ego", math.nan)),
            "v_ego_s": float(relative.get("v_ego_s", math.nan)),
            "v_ego_l": float(relative.get("v_ego_l", math.nan)),
            "s_obj": float(relative.get("s_obj", math.nan)),
            "l_obj": float(relative.get("l_obj", math.nan)),
            "heading_ref_obj": float(relative.get("heading_ref_obj", math.nan)),
            "v_obj_s": float(relative.get("v_obj_s", math.nan)),
            "v_obj_l": float(relative.get("v_obj_l", math.nan)),
            "old_delta_s": float(relative.get("old_delta_s", math.nan)),
            "old_delta_l": float(relative.get("old_delta_l", math.nan)),
            "old_ego_local_delta_s": float(relative.get("old_ego_local_delta_s", math.nan)),
            "old_ego_local_delta_l": float(relative.get("old_ego_local_delta_l", math.nan)),
        }

    def compute_signed_boundary_cbf_margin(
        self,
        reference_state: State,
        rollout_state: State,
        safety_object: SafetyObject,
    ) -> Dict[str, Any]:
        """Compute a signed road-boundary/no-drive CBF margin.

        Positive values mean the ego is inside the drivable side of the
        boundary, zero is the boundary itself, and negative values mean the
        ego has crossed into a no-drive region.
        """
        payload = safety_object.payload or {}
        object_kind = str(safety_object.object_kind)
        if object_kind == "no_drive_area":
            return self._compute_no_drive_area_cbf_margin(reference_state, rollout_state, safety_object)

        side = str(payload.get("side", payload.get("boundary_side", "left")) or "left")
        if side not in {"left", "right"}:
            side = "left"
        boundary_metrics = self._signed_frenet_boundary_metrics(reference_state, rollout_state)
        h_left = float(boundary_metrics.get("h_left", math.nan))
        h_right = float(boundary_metrics.get("h_right", math.nan))
        raw_margin = h_right if side == "left" else h_left
        if not math.isfinite(raw_margin):
            fallback = self._basic_road_boundary_margins_for_state(reference_state, rollout_state, margin=0.0)
            raw_margin = float(fallback["left_margin"] if side == "left" else fallback["right_margin"])
            boundary_metrics.update(
                {
                    "h_left": float(fallback["right_margin"]),
                    "h_right": float(fallback["left_margin"]),
                    "h_boundary": float(min(fallback["left_margin"], fallback["right_margin"])),
                    "boundary_margin": 0.0,
                    "lane_l_min": float(fallback.get("lower", math.nan)),
                    "lane_l_max": float(fallback.get("upper", math.nan)),
                    "source": "ego_local_boundary_fallback",
                    "frenet_fallback_reason": str(boundary_metrics.get("frenet_fallback_reason", "frenet_invalid")),
                }
            )
            h_left = float(boundary_metrics.get("h_left", math.nan))
            h_right = float(boundary_metrics.get("h_right", math.nan))
        h_boundary = float(raw_margin)
        lateral = float(boundary_metrics.get("ego_l", boundary_metrics.get("lateral", math.nan)))
        signed_lateral = h_boundary if side == "left" else -h_boundary
        return {
            "h_2d": float(h_boundary),
            "h_boundary": float(h_boundary),
            "signed_boundary_margin": float(raw_margin),
            "boundary_margin": float(raw_margin),
            "left_boundary_margin": float(h_right),
            "right_boundary_margin": float(h_left),
            "delta_s": math.inf,
            "delta_l": float(signed_lateral),
            "long_clearance": math.inf,
            "lat_clearance": float(raw_margin),
            "d_s_safe": math.inf,
            "d_l_safe": float(boundary_metrics.get("boundary_margin", self.config.lane_margin)),
            "object_id": safety_object.object_id,
            "object_type": safety_object.object_type,
            "object_kind": object_kind,
            "relation": side,
            "target_longitudinal_speed": 0.0,
            "ego_length": float(self.config.vehicle_length),
            "ego_width": float(self.config.vehicle_width),
            "object_length": 0.0,
            "object_width": 0.0,
            "power": 1.0,
            "eps": float(self.config.small_tolerance),
            "road_boundary_margin_source": str(boundary_metrics.get("source", "")),
            "road_boundary_lateral_position": float(lateral),
            "road_boundary_lower": float(boundary_metrics.get("lane_l_min", math.nan)),
            "road_boundary_upper": float(boundary_metrics.get("lane_l_max", math.nan)),
            "coordinate_mode": str(boundary_metrics.get("coordinate_mode", "ego_local_fallback")),
            "frenet_valid": bool(boundary_metrics.get("ego_frenet_valid", False)),
            "ego_ref_lane_valid": bool(boundary_metrics.get("ego_ref_lane_valid", False)),
            "ego_frenet_valid": bool(boundary_metrics.get("ego_frenet_valid", False)),
            "object_frenet_valid": bool(boundary_metrics.get("ego_frenet_valid", False)),
            "frenet_fallback_reason": str(boundary_metrics.get("frenet_fallback_reason", "")),
            "s_ego": float(boundary_metrics.get("ego_s", math.nan)),
            "l_ego": float(boundary_metrics.get("ego_l", math.nan)),
            "ego_s": float(boundary_metrics.get("ego_s", math.nan)),
            "ego_l": float(boundary_metrics.get("ego_l", math.nan)),
            "v_ego_s": float(boundary_metrics.get("ego_v_s", math.nan)),
            "ego_v_s": float(boundary_metrics.get("ego_v_s", math.nan)),
            "heading_ref_ego": float(boundary_metrics.get("ego_heading_ref", math.nan)),
            "ego_heading_ref": float(boundary_metrics.get("ego_heading_ref", math.nan)),
            "s_obj": float(boundary_metrics.get("ego_s", math.nan)),
            "l_obj": float(boundary_metrics.get("ego_l", math.nan)),
            "object_s": float(boundary_metrics.get("ego_s", math.nan)),
            "object_l": float(boundary_metrics.get("ego_l", math.nan)),
            "object_v_s": 0.0,
            "old_delta_s": 0.0,
            "old_delta_l": float(lateral),
            "old_ego_local_delta_s": 0.0,
            "old_ego_local_delta_l": float(lateral),
            "lane_l_min": float(boundary_metrics.get("lane_l_min", math.nan)),
            "lane_l_max": float(boundary_metrics.get("lane_l_max", math.nan)),
            "boundary_margin_config": float(boundary_metrics.get("boundary_margin", math.nan)),
            "h_left": float(boundary_metrics.get("h_left", math.nan)),
            "h_right": float(boundary_metrics.get("h_right", math.nan)),
            "line_type": str(payload.get("line_type", "")),
            "line_color": str(payload.get("line_color", "")),
            "line_prohibited": bool(payload.get("line_prohibited", False)),
            "ego_on_lane": bool(boundary_metrics.get("on_lane", True)),
            "ego_out_of_road": bool(boundary_metrics.get("out_of_road", False)),
            "ego_out_of_route": bool(boundary_metrics.get("out_of_route", False)),
            "ego_crash_sidewalk": bool(boundary_metrics.get("crash_sidewalk", False)),
        }

    def _compute_no_drive_area_cbf_margin(
        self,
        reference_state: State,
        rollout_state: State,
        safety_object: SafetyObject,
    ) -> Dict[str, Any]:
        ego = self._ego(rollout_state)
        boundary_metrics = self._signed_frenet_boundary_metrics(reference_state, rollout_state)
        hard_violation = bool(
            not bool(ego.get("on_lane", True))
            or bool(ego.get("out_of_road", False))
            or bool(ego.get("out_of_route", False))
            or bool(ego.get("crash_sidewalk", False))
            or bool(ego.get("on_yellow_continuous_line", False))
            or bool(ego.get("on_white_continuous_line", False))
        )
        h_value = float(boundary_metrics.get("h_boundary", math.nan))
        if not math.isfinite(h_value):
            h_value = -1.0 if hard_violation else math.inf
        return {
            "h_2d": float(h_value),
            "h_boundary": float(h_value) if math.isfinite(h_value) else math.nan,
            "delta_s": math.nan,
            "delta_l": float(boundary_metrics.get("ego_l", math.nan)),
            "long_clearance": math.nan,
            "lat_clearance": float(h_value) if math.isfinite(h_value) else math.nan,
            "d_s_safe": math.nan,
            "d_l_safe": math.nan,
            "object_id": safety_object.object_id,
            "object_type": safety_object.object_type,
            "object_kind": safety_object.object_kind,
            "relation": "no_drive_area",
            "target_longitudinal_speed": 0.0,
            "ego_length": float(self.config.vehicle_length),
            "ego_width": float(self.config.vehicle_width),
            "object_length": 0.0,
            "object_width": 0.0,
            "power": 1.0,
            "eps": float(self.config.small_tolerance),
            "coordinate_mode": str(boundary_metrics.get("coordinate_mode", "ego_local_fallback")),
            "frenet_valid": bool(boundary_metrics.get("ego_frenet_valid", False)),
            "ego_ref_lane_valid": bool(boundary_metrics.get("ego_ref_lane_valid", False)),
            "ego_frenet_valid": bool(boundary_metrics.get("ego_frenet_valid", False)),
            "object_frenet_valid": bool(boundary_metrics.get("ego_frenet_valid", False)),
            "frenet_fallback_reason": str(boundary_metrics.get("frenet_fallback_reason", "")),
            "s_ego": float(boundary_metrics.get("ego_s", math.nan)),
            "l_ego": float(boundary_metrics.get("ego_l", math.nan)),
            "ego_s": float(boundary_metrics.get("ego_s", math.nan)),
            "ego_l": float(boundary_metrics.get("ego_l", math.nan)),
            "v_ego_s": float(boundary_metrics.get("ego_v_s", math.nan)),
            "ego_v_s": float(boundary_metrics.get("ego_v_s", math.nan)),
            "heading_ref_ego": float(boundary_metrics.get("ego_heading_ref", math.nan)),
            "ego_heading_ref": float(boundary_metrics.get("ego_heading_ref", math.nan)),
            "s_obj": float(boundary_metrics.get("ego_s", math.nan)),
            "l_obj": float(boundary_metrics.get("ego_l", math.nan)),
            "object_s": float(boundary_metrics.get("ego_s", math.nan)),
            "object_l": float(boundary_metrics.get("ego_l", math.nan)),
            "object_v_s": 0.0,
            "old_delta_s": 0.0,
            "old_delta_l": float(boundary_metrics.get("ego_l", math.nan)),
            "old_ego_local_delta_s": 0.0,
            "old_ego_local_delta_l": float(boundary_metrics.get("ego_l", math.nan)),
            "lane_l_min": float(boundary_metrics.get("lane_l_min", math.nan)),
            "lane_l_max": float(boundary_metrics.get("lane_l_max", math.nan)),
            "boundary_margin_config": float(boundary_metrics.get("boundary_margin", math.nan)),
            "h_left": float(boundary_metrics.get("h_left", math.nan)),
            "h_right": float(boundary_metrics.get("h_right", math.nan)),
            "ego_on_lane": bool(ego.get("on_lane", True)),
            "ego_out_of_road": bool(ego.get("out_of_road", False)),
            "ego_out_of_route": bool(ego.get("out_of_route", False)),
            "ego_crash_sidewalk": bool(ego.get("crash_sidewalk", False)),
            "ego_on_yellow_continuous_line": bool(ego.get("on_yellow_continuous_line", False)),
            "ego_on_white_continuous_line": bool(ego.get("on_white_continuous_line", False)),
            "hard_violation": bool(hard_violation),
        }

    def _signed_frenet_boundary_metrics(self, reference_state: State, rollout_state: State) -> Dict[str, Any]:
        reference_ego = self._ego(reference_state)
        ego = self._ego(rollout_state)
        ego_frenet = self._rss_project_entity_to_frenet(reference_state, ego, role="ego")
        reference_frenet = self._rss_project_entity_to_frenet(reference_state, reference_ego, role="ego")
        ego_l = self._safe_float(ego_frenet.get("l", math.nan), math.nan)
        reference_l = self._safe_float(reference_frenet.get("l", math.nan), math.nan)
        ego_width = self._object_width(ego, self.config.vehicle_width)
        boundary_margin = max(0.0, float(getattr(self.config, "lane_margin", 0.0)))

        if math.isfinite(ego_l):
            current_width = self._current_lane_width(reference_state)
            lanes = reference_state.get("lanes", {}) or {}
            left_lane = lanes.get("left")
            right_lane = lanes.get("right")
            left_available = bool((left_lane or {}).get("available", False)) and not bool(
                reference_ego.get("left_lane_line_prohibited", False)
            )
            right_available = bool((right_lane or {}).get("available", False)) and not bool(
                reference_ego.get("right_lane_line_prohibited", False)
            )
            left_width = self._lane_available_width(reference_state, left_lane) if left_available else 0.0
            right_width = self._lane_available_width(reference_state, right_lane) if right_available else 0.0
            lane_l_min_raw = -current_width / 2.0 - right_width
            lane_l_max_raw = current_width / 2.0 + left_width
            source = "frenet_lane_width_estimate"
        else:
            reason = str(ego_frenet.get("reason", "") or reference_frenet.get("reason", "") or "missing_frenet_l")
            return {
                "coordinate_mode": "ego_local_fallback",
                "ego_ref_lane_valid": bool(ego_frenet.get("ref_lane_valid", False)),
                "ego_frenet_valid": False,
                "frenet_fallback_reason": reason,
                "ego_s": self._safe_float(ego_frenet.get("s", math.nan), math.nan),
                "ego_l": ego_l,
                "ego_v_s": self._safe_float(ego_frenet.get("v_s", math.nan), math.nan),
                "ego_heading_ref": self._safe_float(ego_frenet.get("heading_ref", math.nan), math.nan),
                "lane_l_min": math.nan,
                "lane_l_max": math.nan,
                "boundary_margin": boundary_margin,
                "h_left": math.nan,
                "h_right": math.nan,
                "h_boundary": math.nan,
                "source": "frenet_unavailable",
                "on_lane": bool(ego.get("on_lane", True)),
                "out_of_road": bool(ego.get("out_of_road", False)),
                "out_of_route": bool(ego.get("out_of_route", False)),
                "crash_sidewalk": bool(ego.get("crash_sidewalk", False)),
            }

        lane_l_min = float(lane_l_min_raw + ego_width / 2.0 + boundary_margin)
        lane_l_max = float(lane_l_max_raw - ego_width / 2.0 - boundary_margin)
        h_left = float(ego_l - lane_l_min) if math.isfinite(ego_l) else math.nan
        h_right = float(lane_l_max - ego_l) if math.isfinite(ego_l) else math.nan
        h_boundary = min(h_left, h_right) if math.isfinite(h_left) and math.isfinite(h_right) else math.nan
        coordinate_mode = str(ego_frenet.get("mode", "ego_local_fallback"))
        if coordinate_mode == "computed_frenet":
            frenet_reason = ""
        else:
            frenet_reason = str(ego_frenet.get("reason", "") or reference_frenet.get("reason", ""))
        return {
            "coordinate_mode": coordinate_mode,
            "ego_ref_lane_valid": bool(ego_frenet.get("ref_lane_valid", False)),
            "ego_frenet_valid": bool(ego_frenet.get("valid", False)),
            "frenet_fallback_reason": frenet_reason,
            "ego_s": self._safe_float(ego_frenet.get("s", math.nan), math.nan),
            "ego_l": ego_l,
            "ego_v_s": self._safe_float(ego_frenet.get("v_s", math.nan), math.nan),
            "ego_heading_ref": self._safe_float(ego_frenet.get("heading_ref", math.nan), math.nan),
            "lane_l_min": lane_l_min,
            "lane_l_max": lane_l_max,
            "boundary_margin": boundary_margin,
            "h_left": h_left,
            "h_right": h_right,
            "h_boundary": h_boundary,
            "source": source,
            "on_lane": bool(ego.get("on_lane", True)),
            "out_of_road": bool(ego.get("out_of_road", False)),
            "out_of_route": bool(ego.get("out_of_route", False)),
            "crash_sidewalk": bool(ego.get("crash_sidewalk", False)),
        }

    def _compute_2d_rss_cbf_safety_object_margin(
        self,
        reference_state: State,
        rollout_state: State,
        safety_object: SafetyObject,
    ) -> Dict[str, Any]:
        if self._is_rss_2d_road_safety_object(safety_object):
            return self.compute_signed_boundary_cbf_margin(reference_state, rollout_state, safety_object)
        return self.compute_2d_rss_cbf_margin(
            rollout_state,
            safety_object.payload or {},
            safety_object.object_kind,
        )

    def _rss_2d_object_kind(self, obj: Dict[str, Any]) -> str:
        object_type = str(obj.get("object_type", "")).lower()
        if object_type in {"road_boundary", "lane_boundary", "yellow_line", "boundary"}:
            return "boundary"
        if object_type in {"road_edge", "sidewalk"}:
            return "road_edge"
        if object_type in {"no_drive_area", "road_departure"}:
            return "no_drive_area"
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
        if bool(getattr(self.config, "enable_frenet_coordinates", False)):
            frenet = self._rss_project_entity_to_frenet(state, obj, role="object")
            v_s = self._safe_float(frenet.get("v_s", math.nan), math.nan)
            if math.isfinite(v_s):
                return v_s
        speed = max(0.0, float(obj.get("speed", 0.0)))
        ego_heading = float(self._ego(state).get("heading", 0.0))
        obj_heading = float(obj.get("heading", ego_heading))
        return speed * math.cos(obj_heading - ego_heading)

    def _rss_ego_longitudinal_speed(self, state: State) -> float:
        ego = self._ego(state)
        if bool(getattr(self.config, "enable_frenet_coordinates", False)):
            frenet = self._rss_project_entity_to_frenet(state, ego, role="ego")
            v_s = self._safe_float(frenet.get("v_s", math.nan), math.nan)
            if math.isfinite(v_s):
                return max(0.0, v_s)
        return self._ego_speed(state)

    def _rss_2d_longitudinal_safe_distance(
        self,
        state: State,
        obj: Dict[str, Any],
        object_kind: str,
        relation: str,
    ) -> float:
        ego_speed = self._rss_ego_longitudinal_speed(state)
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
        current_worst: Dict[str, Any] = {}
        constraints: List[Dict[str, Any]] = []
        constraint_object_ids = set()
        left_margins: List[float] = []
        right_margins: List[float] = []
        lateral_positions: List[float] = []
        boundary_h_values: List[float] = []
        boundary_h_current = math.inf
        boundary_h_final = math.inf
        boundary_source = ""
        step_H_values: List[float] = []
        boundary_step_metrics: List[Dict[str, float]] = []
        geometry_distance_time = 0.0

        for step in range(horizon + 1):
            step_constraints: List[Dict[str, Any]] = []
            for safety_object in rollout_objects:
                if safety_object.payload is None:
                    continue
                geometry_start = time.perf_counter()
                margin = self._compute_2d_rss_cbf_safety_object_margin(
                    state,
                    rollout_state,
                    safety_object,
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

            step_boundary_h_values: List[float] = []
            step_left_margin = math.nan
            step_right_margin = math.nan
            step_lateral = math.nan
            step_lane_l_min = math.nan
            step_lane_l_max = math.nan
            step_h_left = math.nan
            step_h_right = math.nan
            step_h_boundary = math.nan
            for constraint in step_constraints:
                constraint_kind = str(constraint.get("object_kind", ""))
                if constraint_kind not in self.ROAD_SAFETY_OBJECT_KINDS:
                    continue
                h_value = float(constraint.get("h", constraint.get("h_2d", math.inf)))
                step_boundary_h_values.append(h_value)
                side = str(constraint.get("relation", ""))
                if side == "left":
                    step_left_margin = float(constraint.get("boundary_margin", math.nan))
                elif side == "right":
                    step_right_margin = float(constraint.get("boundary_margin", math.nan))
                if math.isnan(step_lateral):
                    step_lateral = float(constraint.get("road_boundary_lateral_position", math.nan))
                if math.isnan(step_lane_l_min):
                    step_lane_l_min = float(constraint.get("lane_l_min", math.nan))
                if math.isnan(step_lane_l_max):
                    step_lane_l_max = float(constraint.get("lane_l_max", math.nan))
                if math.isnan(step_h_left):
                    step_h_left = float(constraint.get("h_left", math.nan))
                if math.isnan(step_h_right):
                    step_h_right = float(constraint.get("h_right", math.nan))
                if math.isnan(step_h_boundary):
                    step_h_boundary = float(constraint.get("h_boundary", math.nan))
                boundary_source = str(constraint.get("road_boundary_margin_source", boundary_source))

            if step_boundary_h_values:
                step_boundary_h = min(step_boundary_h_values)
                boundary_h_values.append(step_boundary_h)
                if step == 0:
                    boundary_h_current = step_boundary_h
                boundary_h_final = step_boundary_h
            if math.isfinite(step_left_margin):
                left_margins.append(step_left_margin)
            if math.isfinite(step_right_margin):
                right_margins.append(step_right_margin)
            if math.isfinite(step_lateral):
                lateral_positions.append(step_lateral)

            if collect_debug:
                constraints.extend(step_constraints)
            step_H = min((float(item["h"]) for item in step_constraints), default=math.inf)
            step_worst = min(step_constraints, key=lambda item: float(item["h"]), default={})
            step_H_values.append(float(step_H))
            if math.isfinite(step_lateral) or math.isfinite(step_lane_l_min) or math.isfinite(step_lane_l_max):
                lane_center = (
                    0.5 * (step_lane_l_min + step_lane_l_max)
                    if math.isfinite(step_lane_l_min) and math.isfinite(step_lane_l_max)
                    else 0.0
                )
                boundary_step_metrics.append(
                    {
                        "step": float(step),
                        "ego_l": float(step_lateral),
                        "lane_l_min": float(step_lane_l_min),
                        "lane_l_max": float(step_lane_l_max),
                        "lane_center_l": float(lane_center),
                        "l_error": float(step_lateral - lane_center) if math.isfinite(step_lateral) else math.nan,
                        "h_left": float(step_h_left),
                        "h_right": float(step_h_right),
                        "h_boundary": float(step_h_boundary),
                    }
                )
            if step == 0:
                current_H = step_H
                current_worst = step_worst
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
        first_boundary_metric = boundary_step_metrics[0] if boundary_step_metrics else {}
        final_boundary_metric = boundary_step_metrics[-1] if boundary_step_metrics else {}
        center_lateral = float(first_boundary_metric.get("lane_center_l", 0.0))
        centering_improvement = abs(current_lateral - center_lateral) - abs(final_lateral - center_lateral)
        boundary_h_min = min(boundary_h_values) if boundary_h_values else math.inf
        boundary_safe = boundary_h_min >= 0.0
        H_next = final_H
        delta_H = float(H_next - current_H) if math.isfinite(H_next) and math.isfinite(current_H) else math.nan
        predicted_boundary_metric = final_boundary_metric
        current_ego_l = self._safe_float(first_boundary_metric.get("ego_l", math.nan), math.nan)
        predicted_ego_l = self._safe_float(predicted_boundary_metric.get("ego_l", math.nan), math.nan)
        lane_center_l = self._safe_float(first_boundary_metric.get("lane_center_l", 0.0), 0.0)
        current_l_error = current_ego_l - lane_center_l if math.isfinite(current_ego_l) else math.nan
        predicted_l_error = predicted_ego_l - lane_center_l if math.isfinite(predicted_ego_l) else math.nan
        boundary_recovery_score = (
            abs(current_l_error) - abs(predicted_l_error)
            if math.isfinite(current_l_error) and math.isfinite(predicted_l_error)
            else math.nan
        )
        ego = self._ego(state)
        final_ego = self._ego(rollout_state)
        current_speed = self._safe_float(ego.get("speed", math.nan), math.nan)
        predicted_speed = self._safe_float(final_ego.get("speed", math.nan), math.nan)
        return {
            "safe": bool(H >= 0.0),
            "objects_safe": bool(H >= 0.0),
            "H": float(H),
            "current_H": float(current_H),
            "H_current": float(current_H),
            "H_next": float(H_next),
            "delta_H": float(delta_H),
            "final_H": float(final_H),
            "min_h_2d": float(H),
            "current_h_2d": float(current_H),
            "final_h_2d": float(final_H),
            "worst_h": float(H),
            "worst": worst,
            "current_worst": current_worst,
            "current_worst_object_type": str(current_worst.get("object_type", "")) if current_worst else "",
            "current_worst_object_kind": str(current_worst.get("object_kind", "")) if current_worst else "",
            "current_speed": float(current_speed),
            "predicted_speed": float(predicted_speed),
            "constraints": constraints,
            "object_margins": constraints,
            "safety_object_count": len(constraint_object_ids),
            "num_objects": int(len(objects)),
            "horizon_steps": int(horizon),
            "geometry_distance_time": float(geometry_distance_time),
            "road_boundary_safe": bool(boundary_safe),
            "left_boundary_safe": bool(left_min >= 0.0),
            "right_boundary_safe": bool(right_min >= 0.0),
            "left_boundary_margin": float(left_min),
            "right_boundary_margin": float(right_min),
            "min_boundary_margin": float(min(left_min, right_min)),
            "boundary_h_current": float(boundary_h_current),
            "boundary_h_min_pred": float(boundary_h_min),
            "boundary_h_final_pred": float(boundary_h_final),
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
            "current_ego_l": float(current_ego_l),
            "predicted_ego_l": float(predicted_ego_l),
            "lane_center_l": float(lane_center_l),
            "current_l_error": float(current_l_error),
            "predicted_l_error": float(predicted_l_error),
            "boundary_recovery_score": float(boundary_recovery_score),
            "ego_dist_to_left_side": self._safe_float(ego.get("dist_to_left_side", math.nan), math.nan),
            "ego_dist_to_right_side": self._safe_float(ego.get("dist_to_right_side", math.nan), math.nan),
            "ego_on_lane": bool(ego.get("on_lane", True)),
            "ego_out_of_route": bool(ego.get("out_of_route", False)),
            "ego_crash_sidewalk": bool(ego.get("crash_sidewalk", False)),
            "ego_on_yellow_continuous_line": bool(ego.get("on_yellow_continuous_line", False)),
            "ego_on_white_continuous_line": bool(ego.get("on_white_continuous_line", False)),
        }

    def _copy_safety_object(self, safety_object: SafetyObject) -> SafetyObject:
        return SafetyObject(
            object_id=safety_object.object_id,
            object_type=safety_object.object_type,
            geometry_type=safety_object.geometry_type,
            object_kind=safety_object.object_kind,
            payload=self._copy_entity_preserving_runtime_refs(safety_object.payload),
            is_dynamic=safety_object.is_dynamic,
            center=safety_object.center,
            radius=safety_object.radius,
        )

    def _advance_safety_object_for_rss_cbf(
        self,
        state: State,
        safety_object: SafetyObject,
    ) -> SafetyObject:
        if self._is_rss_2d_road_safety_object(safety_object):
            return safety_object
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
            h_value = raw_margin / scale
            signed_lateral = raw_margin if side == "left" else -raw_margin
            constraints.append(
                {
                    "h": float(h_value),
                    "h_2d": float(h_value),
                    "h_boundary": float(h_value),
                    "constraint_type": "safety_object",
                    "object_id": "road_boundary_{}".format(side),
                    "object_type": "road_boundary",
                    "object_kind": "boundary",
                    "geometry_type": "signed_boundary",
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
                    "signed_boundary_margin": float(raw_margin),
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
            "object_type": "no_drive_area",
            "object_kind": "no_drive_area",
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
        unsafe_boundary_recovery = self._rss_2d_unsafe_boundary_recovery_context(nominal_eval or {})

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
            actions, stage_accs, stage_steers = self._rss_2d_candidate_actions(
                u_original,
                stage,
                unsafe_boundary_recovery=unsafe_boundary_recovery,
            )
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
                and not unsafe_boundary_recovery
                and stage == "coarse"
                and any(float(candidate.get("H", -math.inf)) >= 0.0 for candidate in stage_candidates)
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
            if float(candidate.get("H", -math.inf)) >= 0.0
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
        unsafe_candidates = [
            candidate
            for candidate in candidates
            if float(candidate.get("current_H", candidate.get("H_current", math.inf))) < 0.0
        ]
        valid_recovery_candidate_count = sum(
            1 for candidate in unsafe_candidates if self._rss_2d_valid_recovery_candidate(candidate)
        )
        rejected_because_negative_delta_H_count = sum(
            1
            for candidate in unsafe_candidates
            if self._safe_float(candidate.get("delta_H", math.nan), math.nan) <= self.config.small_tolerance
        )
        least_unsafe_candidate_count = max(0, len(unsafe_candidates) - valid_recovery_candidate_count)
        selected_reason = str(selected.get("_selected_reason", ""))
        if not selected_reason:
            if float(selected.get("current_H", selected.get("H_current", math.inf))) < 0.0:
                selected_reason = (
                    "selected_adaptive_recovery"
                    if self._rss_2d_valid_recovery_candidate(selected)
                    else "least_unsafe_adaptive_recovery"
                )
            else:
                selected_reason = (
                    "selected_safe_candidate"
                    if float(selected.get("H", -math.inf)) >= 0.0
                    else "selected_least_unsafe"
                )
        selected["selected_reason"] = selected_reason
        return selected["action"], {
            "projection_failed": not bool(float(selected.get("H", -math.inf)) >= 0.0),
            "candidate_count": len(candidates),
            "candidate_reject_reasons": reject_reasons,
            "safe_candidate_count": len(safe_candidates),
            "road_safe_candidate_count": len(road_safe_candidates),
            "valid_recovery_candidate_count": int(valid_recovery_candidate_count),
            "least_unsafe_candidate_count": int(least_unsafe_candidate_count),
            "rejected_because_negative_delta_H_count": int(rejected_because_negative_delta_H_count),
            "selected_candidate_delta_H": float(selected.get("delta_H", math.nan)),
            "selected_candidate_boundary_recovery_score": float(selected.get("boundary_recovery_score", math.nan)),
            "acc_candidates": [float(acc) for acc in acc_candidates],
            "steer_candidates": [float(steer) for steer in steer_candidates],
            "steer_sign_convention": "internal_positive_left",
            "candidate_search_mode": search_mode,
            "candidate_search_stopped_after_stage": stopped_after_stage,
            "selected": selected,
            "margins": selected.get("margins", {}),
            "candidates": compact_candidates,
            "reason": selected_reason,
        }

    def _rss_2d_candidate_actions(
        self,
        u_original: Action,
        stage: str,
        unsafe_boundary_recovery: bool = False,
    ) -> Tuple[List[Action], List[float], List[float]]:
        if stage == "coarse":
            accs = self._rss_2d_coarse_acc_candidates(u_original, unsafe_boundary_recovery=unsafe_boundary_recovery)
            steers = self._rss_2d_coarse_steer_candidates(
                u_original,
                unsafe_boundary_recovery=unsafe_boundary_recovery,
            )
        else:
            accs = self._rss_2d_acc_candidates(u_original, unsafe_boundary_recovery=unsafe_boundary_recovery)
            steers = self._rss_2d_steer_candidates(u_original, unsafe_boundary_recovery=unsafe_boundary_recovery)

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

    def _rss_2d_coarse_acc_candidates(
        self,
        u_original: Action,
        unsafe_boundary_recovery: bool = False,
    ) -> List[float]:
        if unsafe_boundary_recovery:
            values = self._rss_2d_adaptive_recovery_acc_candidates(u_original)
        else:
            values = [float(u_original[0])]
            values.extend(float(value) for value in getattr(self.config, "coarse_acc_samples", ()))
            values.append(float(getattr(self.config, "rss_2d_min_speed_preserve_acc", -0.2)))
        return self._unique_clipped_values(values, index=0)

    def _rss_2d_coarse_steer_candidates(
        self,
        u_original: Action,
        unsafe_boundary_recovery: bool = False,
    ) -> List[float]:
        values: List[float] = [float(u_original[1]), 0.0, -0.25, 0.25, -0.5, 0.5]
        values.extend(float(value) for value in getattr(self.config, "coarse_steer_samples", ()))
        if unsafe_boundary_recovery:
            values.extend(self._rss_2d_adaptive_recovery_steer_candidates(u_original))
        return self._unique_clipped_values(values, index=1)

    def _rss_2d_acc_candidates(
        self,
        u_original: Action,
        unsafe_boundary_recovery: bool = False,
    ) -> List[float]:
        if unsafe_boundary_recovery:
            return self._unique_clipped_values(
                self._rss_2d_adaptive_recovery_acc_candidates(u_original),
                index=0,
            )
        num_samples = int(getattr(self.config, "fine_acc_samples", 0))
        if num_samples <= 0:
            num_samples = int(getattr(self.config, "rss_2d_acc_samples", 7))
        num_samples = max(2, num_samples)
        values = list(np.linspace(float(u_original[0]), self.config.min_acc, num_samples))
        values.extend([float(u_original[0]), 0.0, float(getattr(self.config, "rss_2d_min_speed_preserve_acc", -0.2))])
        return self._unique_clipped_values(values, index=0)

    def _rss_2d_steer_candidates(
        self,
        u_original: Action,
        unsafe_boundary_recovery: bool = False,
    ) -> List[float]:
        values: List[float] = [float(u_original[1]), 0.0, -0.25, 0.25, -0.5, 0.5]
        values.extend(float(value) for value in getattr(self.config, "rss_2d_steer_samples", ()))
        fine_count = int(getattr(self.config, "fine_steer_samples", 0))
        if fine_count > 0:
            values.extend(np.linspace(-self.config.max_steer, self.config.max_steer, max(2, fine_count)))
        if unsafe_boundary_recovery:
            values.extend(self._rss_2d_adaptive_recovery_steer_candidates(u_original))
        return self._unique_clipped_values(values, index=1)

    def _rss_2d_adaptive_recovery_acc_candidates(self, u_original: Action) -> List[float]:
        nominal_acc = float(u_original[0])
        values: List[float] = [-1.0, -0.7, -0.5, -0.2, 0.0, 0.2]
        values.extend([nominal_acc - 0.2, nominal_acc, nominal_acc + 0.2])
        if self._rss_2d_prev_selected_action is not None:
            values.append(float(self._rss_2d_prev_selected_action[0]))
        return values

    def _rss_2d_adaptive_recovery_steer_candidates(self, u_original: Action) -> List[float]:
        nominal_steer = float(u_original[1])
        values: List[float] = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
        values.extend([nominal_steer - 0.25, nominal_steer, nominal_steer + 0.25])
        if self._rss_2d_prev_selected_action is not None:
            values.append(float(self._rss_2d_prev_selected_action[1]))
        return values

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

    def _rss_2d_unsafe_boundary_recovery_context(self, evaluation: Dict[str, Any]) -> bool:
        current_H = self._safe_float(evaluation.get("current_H", evaluation.get("H_current", math.inf)), math.inf)
        if current_H >= 0.0:
            return False
        object_type = str(evaluation.get("current_worst_object_type", ""))
        object_kind = str(evaluation.get("current_worst_object_kind", ""))
        worst = evaluation.get("current_worst", {}) if isinstance(evaluation, dict) else {}
        if isinstance(worst, dict):
            object_type = object_type or str(worst.get("object_type", ""))
            object_kind = object_kind or str(worst.get("object_kind", ""))
        return self._rss_2d_is_boundary_recovery_object(object_type, object_kind)

    def _rss_2d_is_boundary_recovery_object(self, object_type: str, object_kind: str) -> bool:
        object_type = str(object_type)
        object_kind = str(object_kind)
        return (
            object_kind in self.ROAD_SAFETY_OBJECT_KINDS
            or object_type
            in {
                "road_edge",
                "no_drive_area",
                "road_departure",
                "yellow_line",
                "lane_boundary",
                "road_boundary",
            }
        )

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
        H_next = float(evaluation.get("H_next", final_H))
        delta_H = float(evaluation.get("delta_H", H_next - current_H))
        H_improvement = final_H - current_H
        safe = H >= 0.0
        speed_preserve_score = min(0.0, float(action[0]) - float(getattr(self.config, "rss_2d_min_speed_preserve_acc", -0.2)))
        intervention_cost = self._action_distance_sq(action, u_original)
        acc_penalty = max(0.0, -float(action[0]))
        excessive_brake_penalty = max(
            0.0,
            float(getattr(self.config, "rss_2d_min_speed_preserve_acc", -0.2)) - float(action[0]),
        )
        steer_penalty = abs(float(action[1] - u_original[1]))
        boundary_recovery_score = self._safe_float(evaluation.get("boundary_recovery_score", math.nan), math.nan)
        if not math.isfinite(boundary_recovery_score):
            boundary_recovery_score = 0.0
        current_speed = self._safe_float(evaluation.get("current_speed", self._ego_speed(state)), self._ego_speed(state))
        predicted_speed = self._safe_float(evaluation.get("predicted_speed", current_speed), current_speed)
        speed_reduction = current_speed - predicted_speed
        current_l_error = self._safe_float(evaluation.get("current_l_error", math.nan), math.nan)
        center_distance = abs(current_l_error) if math.isfinite(current_l_error) else 0.0
        adaptive_recovery_mode = bool(current_H < 0.0)
        risk = max(0.0, -current_H) if math.isfinite(current_H) else 0.0
        w_delta_H = (
            float(getattr(self.config, "adaptive_recovery_base_delta_h_weight", 8.0))
            + float(getattr(self.config, "adaptive_recovery_risk_delta_h_weight", 4.0)) * risk
        )
        w_H = float(getattr(self.config, "adaptive_recovery_h_weight", 4.0))
        w_center = (
            float(getattr(self.config, "adaptive_recovery_base_center_weight", 3.0))
            + float(getattr(self.config, "adaptive_recovery_risk_center_weight", 2.0)) * risk
            + float(getattr(self.config, "adaptive_recovery_center_distance_weight", 0.25)) * center_distance
        )
        w_speed_reduction = (
            float(getattr(self.config, "adaptive_recovery_base_speed_weight", 0.5))
            + float(getattr(self.config, "adaptive_recovery_risk_speed_weight", 1.5)) * risk
            + float(getattr(self.config, "adaptive_recovery_current_speed_weight", 0.1)) * current_speed
        )
        w_action = float(getattr(self.config, "adaptive_recovery_action_weight", 0.05))
        w_smooth = float(getattr(self.config, "adaptive_recovery_smooth_weight", 0.25))
        previous_action = self._rss_2d_prev_selected_action
        smoothness_cost = (
            self._action_distance_sq(action, previous_action)
            if previous_action is not None
            else 0.0
        )
        unsafe_score = (
            w_delta_H * delta_H
            + w_H * H_next
            + w_center * boundary_recovery_score
            + w_speed_reduction * speed_reduction
            - w_action * intervention_cost
            - w_smooth * smoothness_cost
        )
        candidate = {
            "mode": "rss_2d_cbf_candidate",
            "search_stage": search_stage,
            "action": action,
            "safe": safe,
            "H": H,
            "final_H": final_H,
            "current_H": current_H,
            "H_current": current_H,
            "H_next": H_next,
            "delta_H": delta_H,
            "H_improvement": H_improvement,
            "objects_safe": bool(evaluation["safe"]),
            "road_boundary_safe": bool(evaluation.get("road_boundary_safe", False)),
            "min_h_2d": H,
            "final_h_2d": final_H,
            "min_boundary_margin": float(evaluation.get("min_boundary_margin", math.nan)),
            "boundary_h_current": float(evaluation.get("boundary_h_current", math.nan)),
            "boundary_h_min_pred": float(evaluation.get("boundary_h_min_pred", math.nan)),
            "boundary_h_final_pred": float(evaluation.get("boundary_h_final_pred", math.nan)),
            "left_boundary_margin": float(evaluation.get("left_boundary_margin", math.nan)),
            "right_boundary_margin": float(evaluation.get("right_boundary_margin", math.nan)),
            "final_boundary_margin": float(evaluation.get("road_boundary_margin_final_pred", math.nan)),
            "boundary_margin_improvement": float(evaluation.get("road_boundary_margin_improvement", math.nan)),
            "current_ego_l": float(evaluation.get("current_ego_l", math.nan)),
            "predicted_ego_l": float(evaluation.get("predicted_ego_l", math.nan)),
            "lane_center_l": float(evaluation.get("lane_center_l", math.nan)),
            "current_l_error": float(evaluation.get("current_l_error", math.nan)),
            "predicted_l_error": float(evaluation.get("predicted_l_error", math.nan)),
            "boundary_recovery_score": float(boundary_recovery_score),
            "current_speed": float(current_speed),
            "predicted_speed": float(predicted_speed),
            "speed_reduction": float(speed_reduction),
            "adaptive_recovery_mode": bool(adaptive_recovery_mode),
            "risk": float(risk),
            "w_delta_H": float(w_delta_H),
            "w_H": float(w_H),
            "w_center": float(w_center),
            "w_speed_reduction": float(w_speed_reduction),
            "w_action": float(w_action),
            "w_smooth": float(w_smooth),
            "speed_preserve_score": float(speed_preserve_score),
            "selection_score": float(unsafe_score),
            "action_distance": float(intervention_cost),
            "smoothness_cost": float(smoothness_cost),
            "acc_penalty": float(acc_penalty),
            "excessive_brake_penalty": float(excessive_brake_penalty),
            "steer_penalty": float(steer_penalty),
            "evaluation": evaluation,
            "margins": {
                "H": H,
                "final_H": final_H,
                "H_current": current_H,
                "H_next": H_next,
                "delta_H": delta_H,
                "H_improvement": H_improvement,
                "min_h_2d": H,
                "current_h_2d": current_H,
                "final_h_2d": final_H,
                "boundary_h_current": float(evaluation.get("boundary_h_current", math.nan)),
                "boundary_h_min_pred": float(evaluation.get("boundary_h_min_pred", math.nan)),
                "boundary_h_final_pred": float(evaluation.get("boundary_h_final_pred", math.nan)),
                "boundary_recovery_score": float(boundary_recovery_score),
                "current_speed": float(current_speed),
                "predicted_speed": float(predicted_speed),
                "speed_reduction": float(speed_reduction),
                "adaptive_recovery_mode": bool(adaptive_recovery_mode),
                "risk": float(risk),
                "w_delta_H": float(w_delta_H),
                "w_center": float(w_center),
                "w_speed_reduction": float(w_speed_reduction),
                "action_distance": float(intervention_cost),
                "smoothness_cost": float(smoothness_cost),
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
            "H_current": float(candidate.get("H_current", candidate.get("current_H", math.nan))),
            "H_next": float(candidate.get("H_next", math.nan)),
            "delta_H": float(candidate.get("delta_H", math.nan)),
            "final_H": float(candidate.get("final_H", math.nan)),
            "H_improvement": float(candidate.get("H_improvement", math.nan)),
            "current_ego_l": float(candidate.get("current_ego_l", math.nan)),
            "predicted_ego_l": float(candidate.get("predicted_ego_l", math.nan)),
            "lane_center_l": float(candidate.get("lane_center_l", math.nan)),
            "current_l_error": float(candidate.get("current_l_error", math.nan)),
            "predicted_l_error": float(candidate.get("predicted_l_error", math.nan)),
            "boundary_recovery_score": float(candidate.get("boundary_recovery_score", math.nan)),
            "current_speed": float(candidate.get("current_speed", math.nan)),
            "predicted_speed": float(candidate.get("predicted_speed", math.nan)),
            "speed_reduction": float(candidate.get("speed_reduction", math.nan)),
            "adaptive_recovery_mode": bool(candidate.get("adaptive_recovery_mode", False)),
            "risk": float(candidate.get("risk", math.nan)),
            "w_delta_H": float(candidate.get("w_delta_H", math.nan)),
            "w_H": float(candidate.get("w_H", math.nan)),
            "w_center": float(candidate.get("w_center", math.nan)),
            "w_speed_reduction": float(candidate.get("w_speed_reduction", math.nan)),
            "w_action": float(candidate.get("w_action", math.nan)),
            "w_smooth": float(candidate.get("w_smooth", math.nan)),
            "action_distance": float(candidate.get("action_distance", math.nan)),
            "smoothness_cost": float(candidate.get("smoothness_cost", math.nan)),
            "boundary_h_current": float(candidate.get("boundary_h_current", math.nan)),
            "boundary_h_min_pred": float(candidate.get("boundary_h_min_pred", math.nan)),
            "boundary_h_final_pred": float(candidate.get("boundary_h_final_pred", math.nan)),
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
        currently_unsafe = any(float(candidate.get("current_H", candidate.get("H_current", math.inf))) < 0.0 for candidate in candidates)
        if currently_unsafe:
            valid_recovery_candidates = [
                candidate for candidate in candidates if self._rss_2d_valid_recovery_candidate(candidate)
            ]
            if valid_recovery_candidates:
                selected = max(valid_recovery_candidates, key=self._rss_2d_recovery_candidate_sort_key)
                selected["_selected_reason"] = "selected_adaptive_recovery"
                return selected
            selected = max(candidates, key=self._rss_2d_least_unsafe_recovery_candidate_sort_key)
            selected["_selected_reason"] = "least_unsafe_adaptive_recovery"
            return selected
        safe_candidates = [
            candidate
            for candidate in candidates
            if float(candidate.get("H", -math.inf)) >= 0.0
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

    def _rss_2d_valid_recovery_candidate(self, item: Dict[str, Any]) -> bool:
        current_H = self._safe_float(item.get("current_H", item.get("H_current", math.inf)), math.inf)
        if current_H >= 0.0:
            return False
        delta_H = self._safe_float(item.get("delta_H", math.nan), math.nan)
        if not math.isfinite(delta_H) or delta_H <= self.config.small_tolerance:
            return False
        return True

    def _rss_2d_boundary_recovery_candidate(self, item: Dict[str, Any]) -> bool:
        evaluation = item.get("evaluation", {}) if isinstance(item, dict) else {}
        object_type = str(evaluation.get("current_worst_object_type", ""))
        object_kind = str(evaluation.get("current_worst_object_kind", ""))
        current_worst = evaluation.get("current_worst", {}) if isinstance(evaluation, dict) else {}
        if isinstance(current_worst, dict):
            object_type = object_type or str(current_worst.get("object_type", ""))
            object_kind = object_kind or str(current_worst.get("object_kind", ""))
        return self._rss_2d_is_boundary_recovery_object(object_type, object_kind)

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

    def _rss_2d_recovery_candidate_sort_key(self, item: Dict[str, Any]) -> Tuple[float, ...]:
        delta_H = self._safe_float(item.get("delta_H", -math.inf), -math.inf)
        return (
            self._safe_float(item.get("selection_score", -math.inf), -math.inf),
            delta_H,
            self._safe_float(item.get("H_next", -math.inf), -math.inf),
            self._safe_float(item.get("boundary_recovery_score", -math.inf), -math.inf),
            self._safe_float(item.get("H", -math.inf), -math.inf),
            self._safe_float(item.get("final_H", -math.inf), -math.inf),
            -self._safe_float(item.get("intervention_cost", math.inf), math.inf),
            -self._safe_float(item.get("excessive_brake_penalty", math.inf), math.inf),
            -self._safe_float(item.get("acc_penalty", math.inf), math.inf),
            -self._safe_float(item.get("steer_penalty", math.inf), math.inf),
        )

    def _rss_2d_least_unsafe_recovery_candidate_sort_key(self, item: Dict[str, Any]) -> Tuple[float, ...]:
        delta_H = self._safe_float(item.get("delta_H", -math.inf), -math.inf)
        tolerance = max(
            float(getattr(self.config, "adaptive_recovery_delta_h_tie_tolerance", 0.02)),
            self.config.small_tolerance,
        )
        delta_bucket = math.trunc(delta_H / tolerance) if math.isfinite(delta_H) else -math.inf
        return (
            float(delta_bucket),
            self._safe_float(item.get("speed_reduction", -math.inf), -math.inf),
            delta_H,
            self._safe_float(item.get("H_next", -math.inf), -math.inf),
            self._safe_float(item.get("boundary_recovery_score", -math.inf), -math.inf),
            self._safe_float(item.get("final_H", -math.inf), -math.inf),
            self._safe_float(item.get("selection_score", -math.inf), -math.inf),
            -self._safe_float(item.get("intervention_cost", math.inf), math.inf),
            -self._safe_float(item.get("excessive_brake_penalty", math.inf), math.inf),
            -self._safe_float(item.get("acc_penalty", math.inf), math.inf),
            -self._safe_float(item.get("steer_penalty", math.inf), math.inf),
        )

    def _rss_2d_candidate_reject_reason(self, evaluation: Dict[str, Any]) -> str:
        if float(evaluation.get("H", math.inf)) < 0.0:
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
            "prediction_action_order": "internal:[acc, steer]",
            "control_action_order": "internal/control:[acc, steer]",
            "min_h_2d": float(selected_eval.get("min_h_2d", math.inf)),
            "current_h_2d": float(selected_eval.get("current_h_2d", math.inf)),
            "final_h_2d": float(selected_eval.get("final_h_2d", math.inf)),
            "H_nominal": float(nominal_eval.get("H", nominal_eval.get("min_h_2d", math.inf))),
            "H_selected": float(selected_eval.get("H", selected_eval.get("min_h_2d", math.inf))),
            "nominal_H": float(nominal_eval.get("H", nominal_eval.get("min_h_2d", math.inf))),
            "selected_H": float(selected_eval.get("H", selected_eval.get("min_h_2d", math.inf))),
            "H_current": float(selected.get("H_current", selected_eval.get("H_current", selected_eval.get("current_H", math.nan)))) if isinstance(selected, dict) else float(selected_eval.get("H_current", selected_eval.get("current_H", math.nan))),
            "H_next": float(selected.get("H_next", selected_eval.get("H_next", math.nan))) if isinstance(selected, dict) else float(selected_eval.get("H_next", math.nan)),
            "delta_H": float(selected.get("delta_H", selected_eval.get("delta_H", math.nan))) if isinstance(selected, dict) else float(selected_eval.get("delta_H", math.nan)),
            "final_H": float(selected.get("final_H", selected_eval.get("final_H", math.nan))) if isinstance(selected, dict) else float(selected_eval.get("final_H", math.nan)),
            "current_speed": float(selected.get("current_speed", selected_eval.get("current_speed", math.nan))) if isinstance(selected, dict) else float(selected_eval.get("current_speed", math.nan)),
            "predicted_speed": float(selected.get("predicted_speed", selected_eval.get("predicted_speed", math.nan))) if isinstance(selected, dict) else float(selected_eval.get("predicted_speed", math.nan)),
            "adaptive_recovery_mode": bool(selected.get("adaptive_recovery_mode", False)) if isinstance(selected, dict) else False,
            "risk": float(selected.get("risk", math.nan)) if isinstance(selected, dict) else math.nan,
            "w_delta_H": float(selected.get("w_delta_H", math.nan)) if isinstance(selected, dict) else math.nan,
            "w_center": float(selected.get("w_center", math.nan)) if isinstance(selected, dict) else math.nan,
            "w_speed_reduction": float(selected.get("w_speed_reduction", math.nan)) if isinstance(selected, dict) else math.nan,
            "selected_score": float(selected.get("selection_score", math.nan)) if isinstance(selected, dict) else math.nan,
            "selected_delta_H": float(selected.get("delta_H", math.nan)) if isinstance(selected, dict) else math.nan,
            "selected_center_recovery": float(selected.get("boundary_recovery_score", math.nan)) if isinstance(selected, dict) else math.nan,
            "selected_speed_reduction": float(selected.get("speed_reduction", math.nan)) if isinstance(selected, dict) else math.nan,
            "selected_action_distance": float(selected.get("action_distance", math.nan)) if isinstance(selected, dict) else math.nan,
            "selected_smoothness_cost": float(selected.get("smoothness_cost", math.nan)) if isinstance(selected, dict) else math.nan,
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
            "nominal_hard_safe": bool(float(nominal_eval.get("H", -math.inf)) >= 0.0),
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
            "coordinate_mode": worst.get("coordinate_mode", "ego_local_fallback") if worst else "ego_local_fallback",
            "frenet_valid": bool(worst.get("frenet_valid", False)) if worst else False,
            "frenet_fallback_reason": worst.get("frenet_fallback_reason", "") if worst else "",
            "ego_ref_lane_valid": bool(worst.get("ego_ref_lane_valid", False)) if worst else False,
            "ego_frenet_valid": bool(worst.get("ego_frenet_valid", False)) if worst else False,
            "object_frenet_valid": bool(worst.get("object_frenet_valid", False)) if worst else False,
            "s_ego": float(worst.get("s_ego", math.nan)) if worst else math.nan,
            "l_ego": float(worst.get("l_ego", math.nan)) if worst else math.nan,
            "heading_ref_ego": float(worst.get("heading_ref_ego", math.nan)) if worst else math.nan,
            "v_ego_s": float(worst.get("v_ego_s", math.nan)) if worst else math.nan,
            "ego_s": float(worst.get("ego_s", worst.get("s_ego", math.nan))) if worst else math.nan,
            "ego_l": float(worst.get("ego_l", worst.get("l_ego", math.nan))) if worst else math.nan,
            "ego_v_s": float(worst.get("ego_v_s", worst.get("v_ego_s", math.nan))) if worst else math.nan,
            "ego_heading_ref": float(worst.get("ego_heading_ref", worst.get("heading_ref_ego", math.nan))) if worst else math.nan,
            "worst_s_obj": float(worst.get("s_obj", math.nan)) if worst else math.nan,
            "worst_l_obj": float(worst.get("l_obj", math.nan)) if worst else math.nan,
            "worst_v_obj_s": float(worst.get("v_obj_s", math.nan)) if worst else math.nan,
            "object_s": float(worst.get("object_s", worst.get("s_obj", math.nan))) if worst else math.nan,
            "object_l": float(worst.get("object_l", worst.get("l_obj", math.nan))) if worst else math.nan,
            "object_v_s": float(worst.get("object_v_s", worst.get("v_obj_s", math.nan))) if worst else math.nan,
            "old_delta_s": float(worst.get("old_delta_s", math.nan)) if worst else math.nan,
            "old_delta_l": float(worst.get("old_delta_l", math.nan)) if worst else math.nan,
            "old_ego_local_delta_s": float(worst.get("old_ego_local_delta_s", math.nan)) if worst else math.nan,
            "old_ego_local_delta_l": float(worst.get("old_ego_local_delta_l", math.nan)) if worst else math.nan,
            "worst_h_2d": float(worst.get("h_2d", math.nan)) if worst else math.nan,
            "lane_l_min": float(worst.get("lane_l_min", math.nan)) if worst else math.nan,
            "lane_l_max": float(worst.get("lane_l_max", math.nan)) if worst else math.nan,
            "boundary_margin": float(worst.get("boundary_margin_config", worst.get("boundary_margin", math.nan))) if worst else math.nan,
            "h_left": float(worst.get("h_left", math.nan)) if worst else math.nan,
            "h_right": float(worst.get("h_right", math.nan)) if worst else math.nan,
            "h_boundary": float(worst.get("h_boundary", math.nan)) if worst else math.nan,
            "on_lane": bool(worst.get("ego_on_lane", self._ego(state).get("on_lane", True))) if worst else bool(self._ego(state).get("on_lane", True)),
            "out_of_road": bool(worst.get("ego_out_of_road", self._ego(state).get("out_of_road", False))) if worst else bool(self._ego(state).get("out_of_road", False)),
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
            "boundary_h_current": float(selected_eval.get("boundary_h_current", math.nan)),
            "boundary_h_selected": float(
                selected_eval.get("boundary_h_min_pred", selected_eval.get("boundary_h_current", math.nan))
            ),
            "boundary_h_nominal": float(
                nominal_eval.get("boundary_h_min_pred", nominal_eval.get("boundary_h_current", math.nan))
            ),
            "boundary_h_min_pred": float(selected_eval.get("boundary_h_min_pred", math.nan)),
            "boundary_h_final_pred": float(selected_eval.get("boundary_h_final_pred", math.nan)),
            "road_boundary_margin_current": float(selected_eval.get("road_boundary_margin_current", math.nan)),
            "road_boundary_margin_min_pred": float(selected_eval.get("road_boundary_margin_min_pred", math.nan)),
            "road_boundary_margin_final_pred": float(selected_eval.get("road_boundary_margin_final_pred", math.nan)),
            "road_boundary_margin_improvement": float(selected_eval.get("road_boundary_margin_improvement", math.nan)),
            "current_ego_l": float(selected.get("current_ego_l", selected_eval.get("current_ego_l", math.nan))) if isinstance(selected, dict) else float(selected_eval.get("current_ego_l", math.nan)),
            "predicted_ego_l": float(selected.get("predicted_ego_l", selected_eval.get("predicted_ego_l", math.nan))) if isinstance(selected, dict) else float(selected_eval.get("predicted_ego_l", math.nan)),
            "lane_center_l": float(selected.get("lane_center_l", selected_eval.get("lane_center_l", math.nan))) if isinstance(selected, dict) else float(selected_eval.get("lane_center_l", math.nan)),
            "current_l_error": float(selected.get("current_l_error", selected_eval.get("current_l_error", math.nan))) if isinstance(selected, dict) else float(selected_eval.get("current_l_error", math.nan)),
            "predicted_l_error": float(selected.get("predicted_l_error", selected_eval.get("predicted_l_error", math.nan))) if isinstance(selected, dict) else float(selected_eval.get("predicted_l_error", math.nan)),
            "boundary_recovery_score": float(selected.get("boundary_recovery_score", selected_eval.get("boundary_recovery_score", math.nan))) if isinstance(selected, dict) else float(selected_eval.get("boundary_recovery_score", math.nan)),
            "road_boundary_margin_source": selected_eval.get("road_boundary_margin_source", ""),
            "predicted_lateral_position": float(selected_eval.get("predicted_lateral_position", math.nan)),
            "predicted_lane_offset": float(selected_eval.get("predicted_lane_offset", math.nan)),
            "road_boundary_centering_improvement": float(selected_eval.get("road_boundary_centering_improvement", math.nan)),
            "rss_filter_selected_control_action": self.control_action_from_internal(u_safe),
            "rss_filter_selected_acc": float(u_safe[0]),
            "rss_filter_selected_steer": float(u_safe[1]),
            "selected_action": list(u_safe),
            "selected_acc": float(u_safe[0]),
            "selected_steer": float(u_safe[1]),
            "nominal_action": list(u_original),
            "selected_reason": selected.get("selected_reason", projection_debug.get("reason", reason)) if isinstance(selected, dict) else projection_debug.get("reason", reason),
            "filter_intervened": bool(self._action_distance_sq(u_safe, u_original) > self.config.small_tolerance),
            "candidate_reject_reasons": projection_debug.get("candidate_reject_reasons", ""),
            "candidate_count": int(projection_debug.get("candidate_count", 0)),
            "rss_2d_candidate_search_mode": projection_debug.get("candidate_search_mode", ""),
            "rss_2d_candidate_search_stopped_after_stage": projection_debug.get(
                "candidate_search_stopped_after_stage", ""
            ),
            "safe_candidate_count": int(projection_debug.get("safe_candidate_count", 0)),
            "road_safe_candidate_count": int(projection_debug.get("road_safe_candidate_count", 0)),
            "valid_recovery_candidate_count": int(projection_debug.get("valid_recovery_candidate_count", 0)),
            "least_unsafe_candidate_count": int(projection_debug.get("least_unsafe_candidate_count", 0)),
            "rejected_because_negative_delta_H_count": int(
                projection_debug.get("rejected_because_negative_delta_H_count", 0)
            ),
            "selected_candidate_delta_H": float(projection_debug.get("selected_candidate_delta_H", math.nan)),
            "selected_candidate_boundary_recovery_score": float(
                projection_debug.get("selected_candidate_boundary_recovery_score", math.nan)
            ),
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
            "selected_boundary_h": float(selected.get("boundary_h_min_pred", math.nan)) if isinstance(selected, dict) else math.nan,
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
                "boundary_h_current": float(nominal_eval.get("boundary_h_current", math.nan)),
                "boundary_h_min_pred": float(nominal_eval.get("boundary_h_min_pred", math.nan)),
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
            "boundary_h_current",
            "boundary_h_min_pred",
            "boundary_h_final_pred",
            "ego_dist_to_left_side",
            "ego_dist_to_right_side",
            "ego_on_lane",
            "ego_out_of_route",
            "ego_crash_sidewalk",
            "ego_on_yellow_continuous_line",
            "ego_on_white_continuous_line",
        ]:
            if key in selected_eval:
                info[key] = selected_eval[key]
        info = self._with_action_debug(info, u_original, u_safe)
        self._maybe_log_frenet_runtime_verification(info)
        return info

    def _maybe_log_frenet_runtime_verification(self, info: Dict[str, Any]) -> None:
        if not bool(getattr(self.config, "frenet_debug_log", True)):
            return
        log_level = self._rss_debug_log_level()
        if log_level == "off":
            return

        filter_intervened = bool(info.get("filter_intervened", False))
        if filter_intervened:
            self._rss_debug_recent_interventions += 1

        if log_level == "verbose":
            self._log_frenet_runtime_verbose(info, filter_intervened)
            return

        if log_level == "summary":
            self._maybe_log_frenet_runtime_summary(info)
            return

        if log_level == "event" and self._rss_debug_event_should_log(info, filter_intervened):
            self._log_frenet_runtime_event(info, filter_intervened)

    def _rss_debug_log_level(self) -> str:
        level = str(getattr(self.config, "rss_debug_log_level", "event")).strip().lower()
        if level not in {"off", "summary", "event", "verbose"}:
            return "event"
        return level

    def _rss_debug_interval(self) -> int:
        interval = int(getattr(self.config, "rss_debug_log_interval", 50))
        if interval <= 0:
            interval = int(getattr(self.config, "debug_log_interval", 50))
        return max(1, interval)

    def _rss_debug_summary_interval(self) -> int:
        interval = int(getattr(self.config, "rss_debug_summary_interval", self._rss_debug_interval()))
        if interval <= 0:
            interval = self._rss_debug_interval()
        return max(1, interval)

    def _rss_debug_event_should_log(self, info: Dict[str, Any], filter_intervened: bool) -> bool:
        worst_type = str(info.get("worst_object_type", ""))
        selected_reason = str(info.get("selected_reason", ""))
        coordinate_mode = str(info.get("coordinate_mode", ""))
        h_current = self._safe_float(info.get("H_current", math.inf), math.inf)
        h_next = self._safe_float(info.get("H_next", math.inf), math.inf)
        delta_h = self._safe_float(info.get("delta_H", math.inf), math.inf)
        near_margin = float(getattr(self.config, "rss_debug_log_near_margin", 0.2))
        large_drop = float(getattr(self.config, "rss_debug_log_large_drop_threshold", 0.1))
        anomaly = bool(
            h_current < 0.0
            or h_next < near_margin
            or delta_h < -large_drop
            or "least_unsafe" in selected_reason
            or "fallback" in selected_reason
            or (worst_type in {"road_edge", "no_drive_area"} and h_current < 0.0)
            or coordinate_mode != "computed_frenet"
            or bool(info.get("out_of_road", info.get("ego_out_of_road", False)))
            or bool(info.get("crash", False))
            or bool(info.get("ego_crash_sidewalk", False))
            or bool(info.get("crash_sidewalk", False))
            or bool(info.get("road_boundary_hard_violation", False))
        )
        only_intervention = bool(getattr(self.config, "rss_debug_log_only_intervention", True))
        only_anomaly = bool(getattr(self.config, "rss_debug_log_only_anomaly", True))
        if only_intervention and only_anomaly:
            return bool(filter_intervened or anomaly)
        if only_intervention:
            return bool(filter_intervened)
        if only_anomaly:
            return bool(anomaly)
        return True

    def _rss_debug_console_enabled(self, level: str) -> bool:
        log_level = self._rss_debug_log_level()
        if log_level == "off" or not LOGGER.isEnabledFor(logging.INFO):
            return False
        if log_level == "verbose":
            return True
        if log_level == "summary":
            return level in {"summary", "profile"}
        if log_level == "event":
            return bool(getattr(self.config, "rss_debug_log_to_console", False)) and level == "event"
        return False

    def _rss_debug_file_enabled(self, level: str) -> bool:
        log_level = self._rss_debug_log_level()
        if log_level == "off" or not bool(getattr(self.config, "rss_debug_log_to_file", True)):
            return False
        if log_level == "verbose":
            return True
        if log_level == "summary":
            return level in {"summary", "profile"}
        if log_level == "event":
            return level in {"event", "profile"}
        return False

    def _emit_rss_debug_log(self, event_dict: Dict[str, Any], level: str = "event") -> None:
        if not event_dict:
            return
        should_console = self._rss_debug_console_enabled(level)
        should_file = self._rss_debug_file_enabled(level)
        if not should_console and not should_file:
            return

        if should_console:
            self._emit_rss_debug_console(event_dict, level)
        if should_file:
            self._write_rss_debug_jsonl(event_dict)

    def _emit_rss_debug_console(self, event: Dict[str, Any], level: str) -> None:
        event_name = str(event.get("event", level))
        if event_name == "summary":
            LOGGER.info(
                "RSS-CBF summary step=%s filter_ms=%.3f H=%.3f worst=%s v=%.3f "
                "l=%.3f intv_recent=%s/%s objs=%s cands=%s reason=%s",
                int(event.get("step", self._rss_2d_filter_step)),
                self._safe_float(event.get("filter_time_ms", math.nan), math.nan),
                self._safe_float(event.get("H_current", math.nan), math.nan),
                str(event.get("worst_object_type", "")),
                self._safe_float(event.get("current_speed", math.nan), math.nan),
                self._safe_float(event.get("ego_l", math.nan), math.nan),
                int(event.get("filter_intervened_count_recent", 0)),
                int(event.get("summary_interval", self._rss_debug_summary_interval())),
                int(event.get("num_safety_objects", 0)),
                int(event.get("num_candidates", 0)),
                str(event.get("selected_reason", "")),
            )
            return
        if event_name == "profile":
            LOGGER.info(
                "RSS-CBF profile step=%s filter_ms=%.3f build_ms=%.3f eval_ms=%.3f objs=%s cands=%s",
                int(event.get("step", self._rss_2d_filter_step)),
                self._safe_float(event.get("filter_time_ms", math.nan), math.nan),
                self._safe_float(event.get("build_objects_time_ms", math.nan), math.nan),
                self._safe_float(event.get("candidate_eval_time_ms", math.nan), math.nan),
                int(event.get("num_safety_objects", 0)),
                int(event.get("num_candidates", 0)),
            )
            return
        if event_name == "verbose":
            LOGGER.info("RSS-CBF verbose %s", self._json_safe(event))
            return
        LOGGER.info(
            "RSS-CBF event step=%s intv=%s coord=%s worst=%s H=%.3f Hn=%.3f "
            "dH=%.3f l=%.3f v=%.3f action=%s reason=%s",
            int(event.get("step", self._rss_2d_filter_step)),
            bool(event.get("filter_intervened", False)),
            str(event.get("coordinate_mode", "")),
            str(event.get("worst_object_type", "")),
            self._safe_float(event.get("H_current", math.nan), math.nan),
            self._safe_float(event.get("H_next", math.nan), math.nan),
            self._safe_float(event.get("delta_H", math.nan), math.nan),
            self._safe_float(event.get("ego_l", math.nan), math.nan),
            self._safe_float(event.get("current_speed", math.nan), math.nan),
            list(event.get("selected_action", [])),
            str(event.get("selected_reason", "")),
        )

    def _write_rss_debug_jsonl(self, event: Dict[str, Any]) -> None:
        path_text = str(getattr(self.config, "rss_debug_log_file", "logs/rss_cbf_debug.jsonl"))
        if not path_text:
            return
        path = Path(path_text)
        if not path.is_absolute():
            path = Path.cwd() / path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(self._json_safe(event), ensure_ascii=True, sort_keys=True) + "\n")
        except OSError as exc:
            if not self._rss_debug_file_warning_emitted and LOGGER.isEnabledFor(logging.WARNING):
                LOGGER.warning("RSS-CBF debug log file write failed: %s", exc)
                self._rss_debug_file_warning_emitted = True

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return self._json_safe(value.tolist())
        if isinstance(value, np.generic):
            return self._json_safe(value.item())
        if isinstance(value, float):
            if math.isfinite(value):
                return value
            return None
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        return str(value)

    def _rss_debug_event_payload(self, info: Dict[str, Any], filter_intervened: bool) -> Dict[str, Any]:
        return {
            "event": "event",
            "step": int(self._rss_2d_filter_step),
            "filter_intervened": bool(filter_intervened),
            "coordinate_mode": str(info.get("coordinate_mode", "")),
            "worst_object_type": str(info.get("worst_object_type", "")),
            "worst_h": self._safe_float(info.get("worst_h", math.nan), math.nan),
            "H_current": self._safe_float(info.get("H_current", math.nan), math.nan),
            "H_next": self._safe_float(info.get("H_next", math.nan), math.nan),
            "delta_H": self._safe_float(info.get("delta_H", math.nan), math.nan),
            "final_H": self._safe_float(info.get("final_H", math.nan), math.nan),
            "ego_l": self._safe_float(info.get("ego_l", info.get("l_ego", info.get("current_ego_l", math.nan))), math.nan),
            "lane_l_min": self._safe_float(info.get("lane_l_min", math.nan), math.nan),
            "lane_l_max": self._safe_float(info.get("lane_l_max", math.nan), math.nan),
            "current_speed": self._safe_float(info.get("current_speed", math.nan), math.nan),
            "predicted_speed": self._safe_float(info.get("predicted_speed", math.nan), math.nan),
            "selected_action": list(info.get("selected_action", [])),
            "nominal_action": list(info.get("nominal_action", [])),
            "selected_reason": str(info.get("selected_reason", "")),
            "ego_frenet_valid": bool(info.get("ego_frenet_valid", False)),
            "obj_frenet_valid": bool(info.get("object_frenet_valid", False)),
            "adaptive_recovery_mode": bool(info.get("adaptive_recovery_mode", False)),
            "risk": self._safe_float(info.get("risk", math.nan), math.nan),
            "valid_recovery_candidate_count": int(info.get("valid_recovery_candidate_count", 0)),
            "least_unsafe_candidate_count": int(info.get("least_unsafe_candidate_count", 0)),
            "rejected_because_negative_delta_H_count": int(info.get("rejected_because_negative_delta_H_count", 0)),
            "num_safety_objects": int(info.get("safety_object_count_local", info.get("safety_object_count", 0))),
            "num_candidates": int(info.get("candidate_count", 0)),
        }

    def _rss_debug_summary_payload(self, info: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "event": "summary",
            "step": int(self._rss_2d_filter_step),
            "summary_interval": int(self._rss_debug_summary_interval()),
            "filter_time_ms": 1000.0 * self._safe_float(info.get("total_filter_time", math.nan), math.nan),
            "filter_intervened_count_recent": int(self._rss_debug_recent_interventions),
            "H_current": self._safe_float(info.get("H_current", math.nan), math.nan),
            "worst_object_type": str(info.get("worst_object_type", "")),
            "selected_reason": str(info.get("selected_reason", "")),
            "current_speed": self._safe_float(info.get("current_speed", math.nan), math.nan),
            "ego_l": self._safe_float(info.get("ego_l", info.get("l_ego", info.get("current_ego_l", math.nan))), math.nan),
            "selected_action": list(info.get("selected_action", [])),
            "num_safety_objects": int(info.get("safety_object_count_local", info.get("safety_object_count", 0))),
            "num_candidates": int(info.get("candidate_count", 0)),
        }

    def _log_frenet_runtime_event(self, info: Dict[str, Any], filter_intervened: bool) -> None:
        if not (self._rss_debug_console_enabled("event") or self._rss_debug_file_enabled("event")):
            return
        self._emit_rss_debug_log(self._rss_debug_event_payload(info, filter_intervened), level="event")

    def _maybe_log_frenet_runtime_summary(self, info: Dict[str, Any]) -> None:
        interval = self._rss_debug_summary_interval()
        if self._rss_2d_filter_step % interval != 0:
            return
        if not (self._rss_debug_console_enabled("summary") or self._rss_debug_file_enabled("summary")):
            return
        self._emit_rss_debug_log(self._rss_debug_summary_payload(info), level="summary")
        self._rss_debug_recent_interventions = 0

    def _log_frenet_runtime_verbose(self, info: Dict[str, Any], filter_intervened: bool) -> None:
        if not (self._rss_debug_console_enabled("verbose") or self._rss_debug_file_enabled("verbose")):
            return
        verbose_payload = self._rss_debug_event_payload(info, filter_intervened)
        verbose_payload.update(
            {
                "event": "verbose",
                "ego_s": self._safe_float(info.get("ego_s", info.get("s_ego", math.nan)), math.nan),
                "object_s": self._safe_float(info.get("object_s", info.get("worst_s_obj", math.nan)), math.nan),
                "object_l": self._safe_float(info.get("object_l", info.get("worst_l_obj", math.nan)), math.nan),
                "worst_delta_s": self._safe_float(info.get("worst_delta_s", math.nan), math.nan),
                "worst_delta_l": self._safe_float(info.get("worst_delta_l", math.nan), math.nan),
                "old_ego_local_delta_s": self._safe_float(
                    info.get("old_ego_local_delta_s", info.get("old_delta_s", math.nan)), math.nan
                ),
                "old_ego_local_delta_l": self._safe_float(
                    info.get("old_ego_local_delta_l", info.get("old_delta_l", math.nan)), math.nan
                ),
                "frenet_fallback_reason": str(info.get("frenet_fallback_reason", "")),
                "boundary_margin": self._safe_float(info.get("boundary_margin", math.nan), math.nan),
                "h_left": self._safe_float(info.get("h_left", math.nan), math.nan),
                "h_right": self._safe_float(info.get("h_right", math.nan), math.nan),
                "h_boundary": self._safe_float(info.get("h_boundary", math.nan), math.nan),
                "on_lane": bool(info.get("on_lane", info.get("ego_on_lane", True))),
                "out_of_road": bool(info.get("out_of_road", info.get("ego_out_of_road", False))),
                "current_ego_l": self._safe_float(info.get("current_ego_l", math.nan), math.nan),
                "predicted_ego_l": self._safe_float(info.get("predicted_ego_l", math.nan), math.nan),
                "lane_center_l": self._safe_float(info.get("lane_center_l", math.nan), math.nan),
                "current_l_error": self._safe_float(info.get("current_l_error", math.nan), math.nan),
                "predicted_l_error": self._safe_float(info.get("predicted_l_error", math.nan), math.nan),
                "boundary_recovery_score": self._safe_float(info.get("boundary_recovery_score", math.nan), math.nan),
                "selected_acc": self._safe_float(info.get("selected_acc", math.nan), math.nan),
                "selected_steer": self._safe_float(info.get("selected_steer", math.nan), math.nan),
                "w_delta_H": self._safe_float(info.get("w_delta_H", math.nan), math.nan),
                "w_center": self._safe_float(info.get("w_center", math.nan), math.nan),
                "w_speed_reduction": self._safe_float(info.get("w_speed_reduction", math.nan), math.nan),
                "selected_score": self._safe_float(info.get("selected_score", math.nan), math.nan),
                "selected_delta_H": self._safe_float(info.get("selected_delta_H", math.nan), math.nan),
                "selected_center_recovery": self._safe_float(info.get("selected_center_recovery", math.nan), math.nan),
                "selected_speed_reduction": self._safe_float(info.get("selected_speed_reduction", math.nan), math.nan),
                "selected_action_distance": self._safe_float(info.get("selected_action_distance", math.nan), math.nan),
                "selected_smoothness_cost": self._safe_float(info.get("selected_smoothness_cost", math.nan), math.nan),
                "selected_candidate_delta_H": self._safe_float(info.get("selected_candidate_delta_H", math.nan), math.nan),
                "selected_candidate_boundary_recovery_score": self._safe_float(
                    info.get("selected_candidate_boundary_recovery_score", math.nan), math.nan
                ),
            }
        )
        self._emit_rss_debug_log(verbose_payload, level="verbose")

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
        front_speed = (
            max(0.0, self._rss_2d_object_longitudinal_speed(state, obj, "dynamic"))
            if object_kind == "dynamic"
            else 0.0
        )
        return self.compute_rss_distance(
            self._rss_ego_longitudinal_speed(state),
            front_speed=front_speed,
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
        next_obj = self._copy_entity_without_runtime_refs(obj)
        if object_kind == "dynamic":
            heading = float(next_obj.get("heading", self._ego(state).get("heading", 0.0)))
            speed = max(0.0, float(next_obj.get("speed", 0.0)))
            next_obj["x"] = float(next_obj.get("x", 0.0)) + speed * math.cos(heading) * self.config.dt
            next_obj["y"] = float(next_obj.get("y", 0.0)) + speed * math.sin(heading) * self.config.dt
            self._attach_frenet_to_entity(
                next_obj,
                next_obj,
                state.get("_frenet_reference_lane", self._last_frenet_reference_lane),
                role="object",
            )
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
        rollout_obj = self._copy_entity_without_runtime_refs(obj)
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
        relative = self._rss_2d_relative_position_metrics(state, obj) if obj is not None else {}
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
            "worst_delta_s": float(relative.get("delta_s", math.nan)) if relative else math.nan,
            "worst_delta_l": float(relative.get("delta_l", math.nan)) if relative else math.nan,
            "coordinate_mode": relative.get("coordinate_mode", "ego_local_fallback") if relative else "ego_local_fallback",
            "frenet_valid": bool(relative.get("frenet_valid", False)) if relative else False,
            "frenet_fallback_reason": relative.get("frenet_fallback_reason", "") if relative else "",
            "ego_ref_lane_valid": bool(relative.get("ego_ref_lane_valid", False)) if relative else False,
            "ego_frenet_valid": bool(relative.get("ego_frenet_valid", False)) if relative else False,
            "object_frenet_valid": bool(relative.get("object_frenet_valid", False)) if relative else False,
            "s_ego": float(relative.get("s_ego", math.nan)) if relative else math.nan,
            "l_ego": float(relative.get("l_ego", math.nan)) if relative else math.nan,
            "heading_ref_ego": float(relative.get("heading_ref_ego", math.nan)) if relative else math.nan,
            "v_ego_s": float(relative.get("v_ego_s", math.nan)) if relative else math.nan,
            "ego_s": float(relative.get("ego_s", relative.get("s_ego", math.nan))) if relative else math.nan,
            "ego_l": float(relative.get("ego_l", relative.get("l_ego", math.nan))) if relative else math.nan,
            "ego_v_s": float(relative.get("ego_v_s", relative.get("v_ego_s", math.nan))) if relative else math.nan,
            "ego_heading_ref": float(relative.get("ego_heading_ref", relative.get("heading_ref_ego", math.nan))) if relative else math.nan,
            "worst_s_obj": float(relative.get("s_obj", math.nan)) if relative else math.nan,
            "worst_l_obj": float(relative.get("l_obj", math.nan)) if relative else math.nan,
            "worst_v_obj_s": float(relative.get("v_obj_s", math.nan)) if relative else math.nan,
            "object_s": float(relative.get("object_s", relative.get("s_obj", math.nan))) if relative else math.nan,
            "object_l": float(relative.get("object_l", relative.get("l_obj", math.nan))) if relative else math.nan,
            "object_v_s": float(relative.get("object_v_s", relative.get("v_obj_s", math.nan))) if relative else math.nan,
            "old_delta_s": float(relative.get("old_delta_s", math.nan)) if relative else math.nan,
            "old_delta_l": float(relative.get("old_delta_l", math.nan)) if relative else math.nan,
            "old_ego_local_delta_s": float(relative.get("old_ego_local_delta_s", math.nan)) if relative else math.nan,
            "old_ego_local_delta_l": float(relative.get("old_ego_local_delta_l", math.nan)) if relative else math.nan,
            "worst_long_clearance": math.nan,
            "worst_lat_clearance": math.nan,
            "worst_d_s_safe": math.nan,
            "worst_d_l_safe": math.nan,
            "selected_action": list(u_safe),
            "rss_filter_selected_control_action": self.control_action_from_internal(u_safe),
            "rss_filter_selected_acc": float(u_safe[0]),
            "rss_filter_selected_steer": float(u_safe[1]),
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
