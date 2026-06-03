import csv
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


RECOVERY_DIAGNOSTIC_FIELDS = [
    "recovery_mode",
    "recovery_certified",
    "selected_candidate_type",
    "selected_lateral_target",
    "selected_speed_target",
    "num_candidates",
    "num_hard_safe_candidates",
    "collision_free",
    "boundary_safe",
    "control_feasible",
    "rss_longitudinal_margin",
    "rss_lateral_margin",
    "rss_risk_score",
    "rss_longitudinal_score",
    "rss_lateral_score",
    "obstacle_margin_score",
    "boundary_margin_score",
    "progress_score",
    "deadlock_penalty",
    "smoothness_score",
    "nominal_deviation_score",
    "continuity_score",
    "terminal_recovery_score",
    "total_score",
    "fallback_reason",
    "filter_time_ms",
    "frenet_valid",
    "frenet_s",
    "frenet_l",
    "front_blocking_object_type",
    "front_blocking_object_distance",
    "left_space_available",
    "right_space_available",
    "ultra_fast_gate_reason",
    "route_cache_hit",
    "route_build_time_ms",
    "scene_parse_time_ms",
    "candidate_eval_time_ms",
    "num_candidate_specs",
    "num_early_rejected",
    "early_reject_reasons",
    "selected_terminal_passed_blocker",
    "selected_terminal_recoverable",
    "blocker_left_gap",
    "blocker_right_gap",
]


@dataclass
class PredictiveRecoveryConfig:
    horizon: float = 3.0
    dt: float = 0.2
    num_lateral_targets: int = 9
    num_speed_targets: int = 5
    max_objects: int = 6
    max_candidates: int = 12
    max_rollout_steps: int = 12
    object_scan_limit: int = 12
    debug: bool = False
    log_csv_path: str = ""

    route_sample_interval: float = 4.0
    route_min_length: float = 40.0
    route_extra_length: float = 30.0
    route_cache_steps: int = 30
    route_cache_distance: float = 20.0
    route_max_future_groups: int = 1
    default_lane_width: float = 3.5
    default_road_half_width: float = 5.25
    max_projection_distance: float = 5.5
    max_heading_error: float = math.radians(115.0)

    safety_margin: float = 0.35
    obstacle_margin: float = 0.45
    hard_collision_margin: float = 0.10
    max_lateral_offset: float = 5.0
    route_lateral_ignore: float = 12.0
    far_behind_s: float = 12.0
    route_front_distance: float = 60.0

    max_speed: float = 35.0
    max_accel: float = 3.0
    max_decel: float = 6.0
    max_steer_action: float = 1.0
    max_steer_rate: float = 8.0
    default_max_steer_rad: float = math.radians(40.0)

    reaction_time: float = 0.8
    ego_max_decel: float = 6.0
    front_max_decel: float = 4.0
    min_longitudinal_safe_distance: float = 2.0
    lateral_response_time: float = 0.5
    lateral_safe_distance: float = 0.8
    lateral_rss_hard_margin: float = -1.0

    blocking_speed_threshold: float = 2.0
    low_speed_ratio: float = 0.45
    filter_time_warn_ms: float = 40.0
    ultra_brake_threshold: float = -0.15
    ultra_low_throttle_threshold: float = 0.05
    ultra_low_speed_threshold: float = 1.0
    ultra_low_progress_threshold: float = 0.25
    ultra_low_progress_steps: int = 5
    ultra_front_block_distance: float = 28.0
    ultra_front_lateral_window: float = 2.8

    w_progress: float = 1.8
    w_deadlock: float = 5.0
    w_rss_longitudinal: float = 4.0
    w_rss_lateral: float = 3.0
    w_obstacle: float = 5.0
    w_boundary: float = 2.0
    w_smoothness: float = 0.3
    w_nominal: float = 0.18
    w_continuity: float = 0.5
    w_terminal: float = 4.0


@dataclass
class FrenetProjection:
    frenet_valid: bool
    s: float = 0.0
    l: float = 0.0
    heading_error: float = 0.0
    projection_distance: float = float("inf")
    lane_id: Optional[str] = None
    lane_index: Any = None
    reason: str = ""


@dataclass
class RoadBoundaryInfo:
    s: float
    l_min: float
    l_max: float
    left_margin: float
    right_margin: float
    boundary_margin: float
    valid: bool = True
    reason: str = ""


@dataclass
class SceneObject:
    object_id: str
    object_type: str
    position_xy: np.ndarray
    heading: float
    speed: float
    length: float
    width: float
    is_static: bool
    is_vehicle: bool
    is_blocking: bool
    frenet_valid: bool
    s: float
    l: float
    predicted_occupancy: List[Tuple[float, float, float, float, float]] = field(default_factory=list)


@dataclass
class TrajectoryCandidate:
    candidate_id: int
    candidate_type: str
    lateral_target: float
    speed_target: float
    raw_action: np.ndarray


@dataclass
class TrajectoryRollout:
    candidate: TrajectoryCandidate
    times: np.ndarray
    positions: np.ndarray
    headings: np.ndarray
    speeds: np.ndarray
    accelerations: np.ndarray
    steer_actions: np.ndarray
    throttle_actions: np.ndarray
    frenet_s: np.ndarray
    frenet_l: np.ndarray
    hard_safe: bool = False
    failure_reason: str = ""
    min_boundary_margin: float = float("inf")
    min_obstacle_margin: float = float("inf")


@dataclass
class TrajectoryScore:
    progress_score: float = 0.0
    deadlock_penalty: float = 0.0
    rss_longitudinal_score: float = 0.0
    rss_lateral_score: float = 0.0
    obstacle_margin_score: float = 0.0
    boundary_margin_score: float = 0.0
    smoothness_score: float = 0.0
    nominal_deviation_score: float = 0.0
    continuity_score: float = 0.0
    terminal_recovery_score: float = 0.0
    total_score: float = float("inf")
    rss_longitudinal_margin: float = float("inf")
    rss_lateral_margin: float = float("inf")
    rss_risk_score: float = 0.0
    collision_free: bool = True
    boundary_safe: bool = True
    control_feasible: bool = True
    terminal_passed_blocker: bool = False
    terminal_recoverable: bool = False


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if callable(value):
            value = value()
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            value = value.reshape(-1)[0]
        if value is None:
            return default
        value = float(value)
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


def _as_xy(value: Any) -> Optional[np.ndarray]:
    try:
        if callable(value):
            value = value()
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size < 2:
            return None
        return arr[:2].astype(float)
    except Exception:
        return None


def _angle_wrap(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _angle_diff(a: float, b: float) -> float:
    return _angle_wrap(float(a) - float(b))


def _heading_from_vector(vec: np.ndarray, fallback: float = 0.0) -> float:
    if np.linalg.norm(vec[:2]) < 1e-8:
        return fallback
    return math.atan2(float(vec[1]), float(vec[0]))


def _unit_from_heading(heading: float) -> np.ndarray:
    return np.array([math.cos(heading), math.sin(heading)], dtype=float)


def _left_normal_from_heading(heading: float) -> np.ndarray:
    return np.array([-math.sin(heading), math.cos(heading)], dtype=float)


def _unwrap_env(env: Any) -> Any:
    current = env
    seen = set()
    for _ in range(12):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        if hasattr(current, "vehicle") or hasattr(current, "engine"):
            return current
        nxt = getattr(current, "env", None)
        if nxt is None or nxt is current:
            break
        current = nxt
    return current


def _find_attr_in_wrappers(env: Any, attr: str, default: Any = None) -> Any:
    current = env
    seen = set()
    for _ in range(12):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        if hasattr(current, attr):
            try:
                return getattr(current, attr)
            except Exception:
                return default
        nxt = getattr(current, "env", None)
        if nxt is None or nxt is current:
            break
        current = nxt
    return default


def _get_ego_vehicle(env: Any) -> Any:
    vehicle = _find_attr_in_wrappers(env, "vehicle", None)
    if vehicle is not None:
        return vehicle
    current_track_vehicle = _find_attr_in_wrappers(env, "current_track_vehicle", None)
    if current_track_vehicle is not None:
        return current_track_vehicle
    agent = _find_attr_in_wrappers(env, "agent", None)
    return agent


def _get_position(obj: Any) -> Optional[np.ndarray]:
    for attr in ("position", "last_position", "pos"):
        if hasattr(obj, attr):
            xy = _as_xy(getattr(obj, attr))
            if xy is not None:
                return xy
    if hasattr(obj, "origin"):
        try:
            return _as_xy(obj.origin.getPos())
        except Exception:
            pass
    return None


def _get_heading(obj: Any, default: float = 0.0) -> float:
    for attr in ("heading_theta", "heading", "last_heading_theta"):
        if hasattr(obj, attr):
            return _safe_float(getattr(obj, attr), default)
    if hasattr(obj, "heading_dir"):
        vec = _as_xy(getattr(obj, "heading_dir"))
        if vec is not None:
            return _heading_from_vector(vec, default)
    if hasattr(obj, "last_heading_dir"):
        vec = _as_xy(getattr(obj, "last_heading_dir"))
        if vec is not None:
            return _heading_from_vector(vec, default)
    return default


def _get_speed(obj: Any, default: float = 0.0) -> float:
    for attr in ("speed", "last_speed"):
        if hasattr(obj, attr):
            return max(0.0, _safe_float(getattr(obj, attr), default))
    for attr in ("velocity", "last_velocity"):
        if hasattr(obj, attr):
            vel = _as_xy(getattr(obj, attr))
            if vel is not None:
                return float(np.linalg.norm(vel))
    return default


def _get_size(obj: Any, default_length: float = 4.8, default_width: float = 2.0) -> Tuple[float, float]:
    length = None
    width = None
    for attr in ("LENGTH", "length"):
        if hasattr(obj, attr):
            length = _safe_float(getattr(obj, attr), default_length)
            break
    for attr in ("WIDTH", "width"):
        if hasattr(obj, attr):
            width = _safe_float(getattr(obj, attr), default_width)
            break
    if hasattr(obj, "size"):
        try:
            size = np.asarray(getattr(obj, "size"), dtype=float).reshape(-1)
            if size.size >= 2:
                length = float(size[0]) if length is None else length
                width = float(size[1]) if width is None else width
        except Exception:
            pass
    return max(float(length or default_length), 0.1), max(float(width or default_width), 0.1)


def _lane_id(lane: Any) -> str:
    for attr in ("index", "lane_index", "id"):
        if hasattr(lane, attr):
            try:
                return str(getattr(lane, attr))
            except Exception:
                pass
    return str(id(lane))


def _lane_length(lane: Any, default: float = 80.0) -> float:
    for attr in ("length", "Length"):
        if hasattr(lane, attr):
            value = _safe_float(getattr(lane, attr), default)
            if value > 1e-6:
                return value
    return default


def _lane_width(lane: Any, longitudinal: float, default: float) -> float:
    for name in ("width_at", "get_width"):
        if hasattr(lane, name):
            try:
                width = _safe_float(getattr(lane, name)(longitudinal), default)
                if width > 0:
                    return width
            except Exception:
                pass
    for attr in ("width", "WIDTH"):
        if hasattr(lane, attr):
            width = _safe_float(getattr(lane, attr), default)
            if width > 0:
                return width
    return default


def _lane_position(lane: Any, longitudinal: float, lateral: float = 0.0) -> Optional[np.ndarray]:
    for name in ("position", "get_position"):
        if hasattr(lane, name):
            try:
                pos = getattr(lane, name)(float(longitudinal), float(lateral))
                xy = _as_xy(pos)
                if xy is not None:
                    return xy
            except Exception:
                pass
    return None


def _lane_heading(lane: Any, longitudinal: float, fallback: float = 0.0) -> float:
    for name in ("heading_theta_at", "heading_at"):
        if hasattr(lane, name):
            try:
                return _safe_float(getattr(lane, name)(float(longitudinal)), fallback)
            except Exception:
                pass
    eps = 0.5
    p0 = _lane_position(lane, max(0.0, longitudinal - eps), 0.0)
    p1 = _lane_position(lane, longitudinal + eps, 0.0)
    if p0 is not None and p1 is not None:
        return _heading_from_vector(p1 - p0, fallback)
    return fallback


def _lane_local_coordinates(lane: Any, position_xy: np.ndarray) -> Optional[Tuple[float, float]]:
    for name in ("local_coordinates", "local_coordinate", "to_local_coordinates"):
        if hasattr(lane, name):
            try:
                ret = getattr(lane, name)(position_xy)
                arr = np.asarray(ret, dtype=float).reshape(-1)
                if arr.size >= 2:
                    return float(arr[0]), float(arr[1])
            except Exception:
                pass
    return None


def _as_lane_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if v is not None]
    try:
        return [v for v in list(value) if v is not None]
    except Exception:
        return [value]


def _dedupe_lanes(lanes: Iterable[Any]) -> List[Any]:
    ret = []
    seen = set()
    for lane in lanes:
        if lane is None:
            continue
        key = _lane_id(lane)
        if key in seen:
            continue
        seen.add(key)
        ret.append(lane)
    return ret


class RouteFrenetFrame:
    def __init__(self, config: PredictiveRecoveryConfig):
        self.config = config
        self.center_points = np.zeros((0, 2), dtype=float)
        self.s_samples = np.zeros((0,), dtype=float)
        self.headings = np.zeros((0,), dtype=float)
        self.boundary_l_min = np.zeros((0,), dtype=float)
        self.boundary_l_max = np.zeros((0,), dtype=float)
        self.segment_lane_ids: List[str] = []
        self.segment_lane_indices: List[Any] = []
        self.valid = False
        self.has_real_route = False
        self.fallback_reason = "not_built"

    @classmethod
    def build(cls, env: Any, config: PredictiveRecoveryConfig) -> "RouteFrenetFrame":
        frame = cls(config)
        root = _unwrap_env(env)
        vehicle = _get_ego_vehicle(env)
        ego_pos = _get_position(vehicle)
        ego_heading = _get_heading(vehicle, 0.0)
        ego_speed = _get_speed(vehicle, 0.0)
        if ego_pos is None:
            frame._build_heading_fallback(np.zeros(2, dtype=float), ego_heading, config.route_min_length)
            frame.fallback_reason = "missing_ego_position"
            return frame

        route_length = max(config.route_min_length, ego_speed * config.horizon + config.route_extra_length)
        nav = getattr(vehicle, "navigation", None)
        current_group = _dedupe_lanes(_as_lane_list(getattr(nav, "current_ref_lanes", None)))
        next_group = _dedupe_lanes(_as_lane_list(getattr(nav, "next_ref_lanes", None)))

        direct_lane = getattr(vehicle, "lane", None)
        if direct_lane is not None:
            current_group = _dedupe_lanes([direct_lane] + current_group)

        if not current_group:
            current_group = frame._lanes_from_navigation_road(nav, "current_road")
        if not next_group:
            next_group = frame._lanes_from_navigation_road(nav, "next_road")

        selected_current = frame._choose_closest_lane(current_group, ego_pos)
        if selected_current is None:
            frame._build_heading_fallback(ego_pos, ego_heading, route_length)
            frame.fallback_reason = "missing_route_lanes"
            return frame

        route_groups = [current_group or [selected_current]]
        if next_group:
            route_groups.append(next_group)
        route_groups.extend(frame._future_lane_groups_from_navigation(nav, limit=config.route_max_future_groups))

        lane_sequence = [selected_current]
        lane_groups = [current_group or [selected_current]]
        previous_lane = selected_current
        for group in route_groups[1:]:
            group = _dedupe_lanes(group)
            if not group:
                continue
            successor = frame._choose_successor_lane(previous_lane, group)
            if successor is None:
                continue
            if _lane_id(successor) == _lane_id(previous_lane):
                continue
            lane_sequence.append(successor)
            lane_groups.append(group)
            previous_lane = successor

        points: List[np.ndarray] = []
        lane_refs: List[Any] = []
        group_refs: List[List[Any]] = []
        total_added = 0.0
        first = True
        for lane, group in zip(lane_sequence, lane_groups):
            if lane is None or total_added >= route_length:
                continue
            lane_len = _lane_length(lane, route_length)
            local = _lane_local_coordinates(lane, ego_pos) if first else None
            start_s = 0.0
            if local is not None:
                start_s = max(0.0, local[0] - 8.0)
            sample_s = start_s
            while sample_s <= lane_len + 1e-6 and total_added < route_length:
                pos = _lane_position(lane, sample_s, 0.0)
                if pos is not None:
                    if not points or np.linalg.norm(pos - points[-1]) > 0.25:
                        points.append(pos)
                        lane_refs.append(lane)
                        group_refs.append(group or [lane])
                        if len(points) > 1:
                            total_added += float(np.linalg.norm(points[-1] - points[-2]))
                sample_s += config.route_sample_interval
            first = False

        if len(points) < 2:
            frame._build_heading_fallback(ego_pos, ego_heading, route_length)
            frame.fallback_reason = "insufficient_route_points"
            return frame

        frame._set_centerline(points, lane_refs, group_refs)
        frame.has_real_route = True
        frame.fallback_reason = ""
        frame._widen_boundary_from_vehicle_sides(vehicle)
        return frame

    def _build_heading_fallback(self, ego_pos: np.ndarray, heading: float, route_length: float) -> None:
        direction = _unit_from_heading(heading)
        distances = np.arange(-5.0, route_length + self.config.route_sample_interval, self.config.route_sample_interval)
        points = [ego_pos + direction * d for d in distances]
        self._set_centerline(points, [None] * len(points), [[None]] * len(points))
        self.has_real_route = False
        self.fallback_reason = "heading_fallback"

    def _lanes_from_navigation_road(self, nav: Any, road_attr: str) -> List[Any]:
        if nav is None:
            return []
        road = getattr(nav, road_attr, None)
        return self._lanes_from_road(nav, road)

    def _lanes_from_road(self, nav: Any, road: Any) -> List[Any]:
        road_network = getattr(getattr(nav, "map", None), "road_network", None)
        if road is None or road_network is None:
            return []
        lanes = []
        for name in ("get_lanes", "get_lanes_on_road", "get_all_lanes"):
            if hasattr(road_network, name):
                try:
                    lanes = _as_lane_list(getattr(road_network, name)(road))
                    if lanes:
                        return lanes
                except Exception:
                    pass
        return lanes

    def _future_lane_groups_from_navigation(self, nav: Any, limit: int = 6) -> List[List[Any]]:
        if nav is None:
            return []
        checkpoints = getattr(nav, "checkpoints", None)
        if checkpoints is None:
            return []
        try:
            checkpoint_list = list(checkpoints)
        except Exception:
            return []
        start_index = int(_safe_float(getattr(nav, "_target_checkpoints_index", 0), 0.0))
        groups = []
        seen = set()
        for road in checkpoint_list[max(0, start_index):max(0, start_index) + limit]:
            lanes = self._lanes_from_road(nav, road)
            if not lanes and isinstance(road, (tuple, list)) and len(road) >= 2:
                try:
                    from metadrive.component.road_network import Road
                    lanes = self._lanes_from_road(nav, Road(road[0], road[1]))
                except Exception:
                    lanes = []
            ids = tuple(sorted(_lane_id(lane) for lane in lanes))
            if not ids or ids in seen:
                continue
            seen.add(ids)
            groups.append(lanes)
        return groups

    def _choose_closest_lane(self, lanes: Sequence[Any], ego_pos: np.ndarray) -> Any:
        best_lane = None
        best_dist = float("inf")
        for lane in lanes:
            local = _lane_local_coordinates(lane, ego_pos)
            if local is not None:
                dist = abs(local[1])
            else:
                length = _lane_length(lane, self.config.route_min_length)
                samples = np.linspace(0.0, length, num=8)
                dists = []
                for s in samples:
                    pos = _lane_position(lane, s, 0.0)
                    if pos is not None:
                        dists.append(np.linalg.norm(pos - ego_pos))
                dist = min(dists) if dists else float("inf")
            if dist < best_dist:
                best_dist = dist
                best_lane = lane
        return best_lane

    def _choose_successor_lane(self, current_lane: Any, next_lanes: Sequence[Any]) -> Any:
        if not next_lanes:
            return None
        current_end = _lane_position(current_lane, _lane_length(current_lane, self.config.route_min_length), 0.0)
        if current_end is None:
            return next_lanes[0]
        best_lane = None
        best_dist = float("inf")
        current_index = getattr(current_lane, "index", None)
        for lane in next_lanes:
            lane_start = _lane_position(lane, 0.0, 0.0)
            dist = np.linalg.norm(lane_start - current_end) if lane_start is not None else float("inf")
            lane_index = getattr(lane, "index", None)
            if current_index is not None and lane_index is not None and str(current_index).split("_")[-1] == str(lane_index).split("_")[-1]:
                dist *= 0.8
            if dist < best_dist:
                best_dist = dist
                best_lane = lane
        return best_lane

    def _set_centerline(self, points: Sequence[np.ndarray], lane_refs: Sequence[Any], group_refs: Sequence[Sequence[Any]]) -> None:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if pts.shape[0] < 2:
            self.valid = False
            return
        diffs = np.diff(pts, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        keep = np.concatenate([[True], seg_lengths > 1e-6])
        pts = pts[keep]
        lane_refs = [lane for lane, keep_item in zip(lane_refs, keep) if keep_item]
        group_refs = [group for group, keep_item in zip(group_refs, keep) if keep_item]
        if pts.shape[0] < 2:
            self.valid = False
            return

        diffs = np.diff(pts, axis=0)
        seg_lengths = np.maximum(np.linalg.norm(diffs, axis=1), 1e-6)
        self.center_points = pts
        self.s_samples = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        seg_headings = np.array([_heading_from_vector(v) for v in diffs], dtype=float)
        self.headings = np.concatenate([seg_headings, [seg_headings[-1]]])
        self.segment_lane_ids = [_lane_id(lane) if lane is not None else "fallback" for lane in lane_refs[:-1]]
        self.segment_lane_indices = [getattr(lane, "index", None) if lane is not None else None for lane in lane_refs[:-1]]
        self._build_boundary_samples(lane_refs, group_refs)
        self.valid = True

    def _build_boundary_samples(self, lane_refs: Sequence[Any], group_refs: Sequence[Sequence[Any]]) -> None:
        mins = []
        maxs = []
        for idx, point in enumerate(self.center_points):
            heading = self.headings[min(idx, len(self.headings) - 1)] if len(self.headings) else 0.0
            normal = _left_normal_from_heading(heading)
            group = group_refs[min(idx, len(group_refs) - 1)] if group_refs else []
            l_min = float("inf")
            l_max = -float("inf")
            for lane in group:
                if lane is None:
                    continue
                local = _lane_local_coordinates(lane, point)
                lane_s = local[0] if local is not None else 0.0
                center = _lane_position(lane, lane_s, 0.0)
                if center is None:
                    continue
                offset = float(np.dot(center - point, normal))
                width = _lane_width(lane, lane_s, self.config.default_lane_width)
                l_min = min(l_min, offset - width * 0.5)
                l_max = max(l_max, offset + width * 0.5)
            if not np.isfinite(l_min) or not np.isfinite(l_max) or l_max <= l_min:
                l_min = -self.config.default_road_half_width
                l_max = self.config.default_road_half_width
            mins.append(l_min)
            maxs.append(l_max)
        self.boundary_l_min = np.asarray(mins, dtype=float)
        self.boundary_l_max = np.asarray(maxs, dtype=float)

    def _widen_boundary_from_vehicle_sides(self, vehicle: Any) -> None:
        if vehicle is None or self.boundary_l_min.size == 0:
            return
        pos = _get_position(vehicle)
        if pos is None:
            return
        ego_proj = self.project_point(pos, _get_heading(vehicle, 0.0), validate_heading=False)
        left = _safe_float(getattr(vehicle, "dist_to_left_side", None), -1.0)
        right = _safe_float(getattr(vehicle, "dist_to_right_side", None), -1.0)
        if 0.1 < left < 25.0:
            real_l_max = ego_proj.l + left
            self.boundary_l_max = np.minimum(self.boundary_l_max, real_l_max)
        if 0.1 < right < 25.0:
            real_l_min = ego_proj.l - right
            self.boundary_l_min = np.maximum(self.boundary_l_min, real_l_min)
        invalid = self.boundary_l_min >= self.boundary_l_max
        if np.any(invalid):
            center = ego_proj.l
            half_width = max(0.5, min(left if left > 0 else self.config.default_road_half_width, right if right > 0 else self.config.default_road_half_width))
            self.boundary_l_min[invalid] = center - half_width
            self.boundary_l_max[invalid] = center + half_width

    def project_point(self, point_xy: Any, heading: Optional[float] = None, validate_heading: bool = True) -> FrenetProjection:
        point = _as_xy(point_xy)
        if point is None or not self.valid or self.center_points.shape[0] < 2:
            return FrenetProjection(False, reason="invalid_route_or_point")

        best_dist = float("inf")
        best_s = 0.0
        best_l = 0.0
        best_heading = 0.0
        best_idx = 0
        for idx in range(self.center_points.shape[0] - 1):
            p0 = self.center_points[idx]
            p1 = self.center_points[idx + 1]
            seg = p1 - p0
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-8:
                continue
            tangent = seg / seg_len
            rel = point - p0
            t = _clip(float(np.dot(rel, tangent) / seg_len), 0.0, 1.0)
            closest = p0 + tangent * (t * seg_len)
            delta = point - closest
            dist = float(np.linalg.norm(delta))
            if dist < best_dist:
                normal = np.array([-tangent[1], tangent[0]], dtype=float)
                best_dist = dist
                best_s = float(self.s_samples[idx] + t * seg_len)
                best_l = float(np.dot(delta, normal))
                best_heading = math.atan2(float(tangent[1]), float(tangent[0]))
                best_idx = idx

        heading_error = 0.0
        if heading is not None:
            heading_error = _angle_diff(float(heading), best_heading)
        reason = ""
        frenet_valid = True
        if best_dist > self.config.max_projection_distance:
            frenet_valid = False
            reason = "projection_distance"
        if validate_heading and heading is not None and abs(heading_error) > self.config.max_heading_error:
            frenet_valid = False
            reason = "heading_error" if not reason else reason + "|heading_error"
        if not self.has_real_route:
            reason = "heading_fallback" if not reason else reason + "|heading_fallback"
        lane_id = self.segment_lane_ids[min(best_idx, len(self.segment_lane_ids) - 1)] if self.segment_lane_ids else None
        lane_index = self.segment_lane_indices[min(best_idx, len(self.segment_lane_indices) - 1)] if self.segment_lane_indices else None
        return FrenetProjection(
            frenet_valid=frenet_valid,
            s=best_s,
            l=best_l,
            heading_error=heading_error,
            projection_distance=best_dist,
            lane_id=lane_id,
            lane_index=lane_index,
            reason=reason,
        )

    def project_points(self, points_xy: Sequence[Any]) -> List[FrenetProjection]:
        return [self.project_point(point) for point in points_xy]

    def frenet_to_world(self, s: float, l: float) -> Tuple[np.ndarray, float]:
        if not self.valid or self.center_points.shape[0] < 2:
            return np.zeros(2, dtype=float), 0.0
        s_value = float(s)
        if s_value <= self.s_samples[0]:
            idx = 0
            ds = s_value - self.s_samples[0]
        elif s_value >= self.s_samples[-1]:
            idx = len(self.s_samples) - 2
            ds = s_value - self.s_samples[idx]
        else:
            idx = int(np.searchsorted(self.s_samples, s_value, side="right") - 1)
            idx = max(0, min(idx, len(self.s_samples) - 2))
            ds = s_value - self.s_samples[idx]
        p0 = self.center_points[idx]
        p1 = self.center_points[idx + 1]
        seg = p1 - p0
        seg_len = max(float(np.linalg.norm(seg)), 1e-6)
        tangent = seg / seg_len
        heading = math.atan2(float(tangent[1]), float(tangent[0]))
        base = p0 + tangent * ds
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        return base + normal * float(l), heading

    def boundary_at(self, s: float, l: float, ego_width: float, safety_margin: float) -> RoadBoundaryInfo:
        if not self.valid or self.boundary_l_min.size == 0:
            l_min = -self.config.default_road_half_width
            l_max = self.config.default_road_half_width
            reason = "fallback_boundary"
        else:
            s_clamped = _clip(float(s), float(self.s_samples[0]), float(self.s_samples[-1]))
            l_min = float(np.interp(s_clamped, self.s_samples, self.boundary_l_min))
            l_max = float(np.interp(s_clamped, self.s_samples, self.boundary_l_max))
            reason = ""
        left_margin = l_max - float(l) - ego_width * 0.5 - safety_margin
        right_margin = float(l) - l_min - ego_width * 0.5 - safety_margin
        boundary_margin = min(left_margin, right_margin)
        return RoadBoundaryInfo(
            s=float(s),
            l_min=l_min,
            l_max=l_max,
            left_margin=left_margin,
            right_margin=right_margin,
            boundary_margin=boundary_margin,
            valid=left_margin >= 0.0 and right_margin >= 0.0,
            reason=reason if left_margin >= 0.0 and right_margin >= 0.0 else "out_of_boundary",
        )

    def heading_at(self, s: float) -> float:
        if not self.valid or self.headings.size == 0:
            return 0.0
        s_clamped = _clip(float(s), float(self.s_samples[0]), float(self.s_samples[-1]))
        idx = int(np.searchsorted(self.s_samples, s_clamped, side="right") - 1)
        idx = max(0, min(idx, len(self.headings) - 1))
        return float(self.headings[idx])


class PredictiveRecoveryFilter:
    def __init__(self, config: Optional[PredictiveRecoveryConfig] = None):
        self.config = config or PredictiveRecoveryConfig()
        self.last_selected_lateral: Optional[float] = None
        self.last_selected_speed: Optional[float] = None
        self._csv_header_written = False
        self._static_object_cache: Dict[str, Tuple[float, float]] = {}
        self._route_cache_frame: Optional[RouteFrenetFrame] = None
        self._route_cache_key: Optional[Tuple[Any, ...]] = None
        self._route_cache_position: Optional[np.ndarray] = None
        self._route_cache_age = 0
        self._last_ego_pos: Optional[np.ndarray] = None
        self._low_progress_count = 0
        self._last_recovery_active = False
        self._last_route_cache_hit = False
        self._last_route_build_time_ms = 0.0

    def filter(self, env: Any, obs: Any, raw_action: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        del obs
        start = time.time()
        raw_action_np = self._coerce_action(raw_action)
        info = self._empty_info()
        try:
            vehicle = _get_ego_vehicle(env)
            ego_pos = _get_position(vehicle)
            ego_heading = _get_heading(vehicle, 0.0)
            ego_speed = _get_speed(vehicle, 0.0)
            ego_length, ego_width = _get_size(vehicle)
            max_steer_rad = self._max_steer_rad(vehicle)

            gate_reason = self._ultra_fast_gate(env, vehicle, ego_pos, ego_heading, ego_speed, raw_action_np)
            info["ultra_fast_gate_reason"] = gate_reason
            if gate_reason == "low_risk_passthrough":
                safe_action = raw_action_np.astype(np.float32)
                info.update({
                    "recovery_mode": "ultra_fast_passthrough",
                    "recovery_certified": True,
                    "selected_candidate_type": "raw_action",
                    "collision_free": True,
                    "boundary_safe": True,
                    "control_feasible": True,
                    "fallback_reason": "",
                    "route_cache_hit": False,
                })
                self._last_recovery_active = False
                elapsed_ms = (time.time() - start) * 1000.0
                info["filter_time_ms"] = elapsed_ms
                return safe_action, info

            frame = self._get_route_frame(env, vehicle, ego_pos)
            route_build_time_ms = self._last_route_build_time_ms
            ego_projection = frame.project_point(ego_pos, ego_heading) if ego_pos is not None else FrenetProjection(False, reason="missing_ego_position")
            scene_start = time.time()
            objects, scene_info = self._parse_scene(env, frame, ego_projection, ego_speed, ego_width)
            scene_parse_time_ms = (time.time() - scene_start) * 1000.0

            candidate_specs = self._generate_candidate_specs(
                frame,
                ego_projection,
                ego_pos,
                ego_heading,
                ego_speed,
                ego_width,
                raw_action_np,
                scene_info,
                max_steer_rad,
            )
            safe_rollouts: List[TrajectoryRollout] = []
            scored: List[Tuple[TrajectoryScore, TrajectoryRollout]] = []
            early_rejected = 0
            early_reject_reasons: Dict[str, int] = {}

            candidate_start = time.time()
            for candidate in candidate_specs:
                rollout = self._rollout_and_check_candidate(
                    candidate,
                    frame,
                    objects,
                    ego_projection,
                    ego_pos,
                    ego_heading,
                    ego_speed,
                    ego_length,
                    ego_width,
                    max_steer_rad,
                )
                if rollout.hard_safe:
                    score = self._score_rollout(rollout, frame, objects, ego_projection, ego_length, ego_width, raw_action_np, scene_info)
                    scored.append((score, rollout))
                    safe_rollouts.append(rollout)
                else:
                    early_rejected += 1
                    reason = rollout.failure_reason or "unknown"
                    early_reject_reasons[reason] = early_reject_reasons.get(reason, 0) + 1
            candidate_eval_time_ms = (time.time() - candidate_start) * 1000.0

            info.update({
                "num_candidates": len(candidate_specs),
                "num_candidate_specs": len(candidate_specs),
                "num_hard_safe_candidates": len(safe_rollouts),
                "num_early_rejected": early_rejected,
                "early_reject_reasons": self._format_reject_reasons(early_reject_reasons),
                "frenet_valid": bool(ego_projection.frenet_valid),
                "frenet_s": ego_projection.s,
                "frenet_l": ego_projection.l,
                "fallback_reason": ego_projection.reason or frame.fallback_reason,
                "route_cache_hit": bool(self._last_route_cache_hit),
                "route_build_time_ms": route_build_time_ms,
                "scene_parse_time_ms": scene_parse_time_ms,
                "candidate_eval_time_ms": candidate_eval_time_ms,
            })
            info.update(scene_info)

            if scored:
                selected_score, selected_rollout = min(scored, key=lambda item: item[0].total_score)
                safe_action = np.array(
                    [selected_rollout.steer_actions[0], selected_rollout.throttle_actions[0]],
                    dtype=np.float32,
                )
                self.last_selected_lateral = selected_rollout.candidate.lateral_target
                self.last_selected_speed = selected_rollout.candidate.speed_target
                info.update(self._score_to_info(selected_score))
                info.update({
                    "recovery_mode": "predictive_recovery",
                    "recovery_certified": True,
                    "selected_candidate_type": selected_rollout.candidate.candidate_type,
                    "selected_lateral_target": selected_rollout.candidate.lateral_target,
                    "selected_speed_target": selected_rollout.candidate.speed_target,
                    "collision_free": True,
                    "boundary_safe": True,
                    "control_feasible": True,
                    "fallback_reason": info.get("fallback_reason", ""),
                    "selected_terminal_passed_blocker": selected_score.terminal_passed_blocker,
                    "selected_terminal_recoverable": selected_score.terminal_recoverable,
                })
                self._last_recovery_active = bool(np.isfinite(_safe_float(scene_info.get("front_blocking_object_distance"), float("inf"))))
            else:
                safe_action = self._minimum_risk_stop(vehicle)
                info.update({
                    "recovery_mode": "minimum_risk_stop",
                    "recovery_certified": False,
                    "selected_candidate_type": "minimum_risk_stop",
                    "selected_lateral_target": 0.0,
                    "selected_speed_target": 0.0,
                    "collision_free": False,
                    "boundary_safe": False,
                    "control_feasible": True,
                    "fallback_reason": "no_hard_safe_candidate",
                })
                self._last_recovery_active = True

        except Exception as exc:
            safe_action = self._minimum_risk_stop(_get_ego_vehicle(env))
            info.update({
                "recovery_mode": "exception_fallback",
                "recovery_certified": False,
                "selected_candidate_type": "exception_fallback",
                "fallback_reason": "exception:{}".format(type(exc).__name__),
                "control_feasible": True,
            })
            self._last_recovery_active = True
            if self.config.debug:
                info["debug_exception"] = str(exc)

        elapsed_ms = (time.time() - start) * 1000.0
        info["filter_time_ms"] = elapsed_ms
        if elapsed_ms > self.config.filter_time_warn_ms:
            info["filter_time_warning"] = True
        if self._should_write_detailed_log(info):
            self._write_csv(info)
        return np.asarray(safe_action, dtype=np.float32), info

    def _empty_info(self) -> Dict[str, Any]:
        info = {field_name: 0 for field_name in RECOVERY_DIAGNOSTIC_FIELDS}
        info.update({
            "recovery_mode": "inactive",
            "recovery_certified": False,
            "selected_candidate_type": "",
            "fallback_reason": "",
            "front_blocking_object_type": "",
            "front_blocking_object_distance": float("inf"),
            "rss_longitudinal_margin": float("inf"),
            "rss_lateral_margin": float("inf"),
            "left_space_available": 0.0,
            "right_space_available": 0.0,
            "ultra_fast_gate_reason": "",
            "early_reject_reasons": "",
            "route_cache_hit": False,
            "selected_terminal_passed_blocker": False,
            "selected_terminal_recoverable": False,
            "blocker_left_gap": 0.0,
            "blocker_right_gap": 0.0,
        })
        return info

    def _coerce_action(self, action: Any) -> np.ndarray:
        try:
            arr = np.asarray(action, dtype=float).reshape(-1)
        except Exception:
            arr = np.zeros(2, dtype=float)
        if arr.size == 0:
            arr = np.zeros(2, dtype=float)
        if arr.size == 1:
            arr = np.array([arr[0], 0.0], dtype=float)
        return np.clip(arr[:2], -1.0, 1.0)

    def _max_steer_rad(self, vehicle: Any) -> float:
        config = getattr(vehicle, "config", {}) if vehicle is not None else {}
        if isinstance(config, dict) and "max_steering" in config:
            return math.radians(_safe_float(config.get("max_steering"), math.degrees(self.config.default_max_steer_rad)))
        return self.config.default_max_steer_rad

    def _get_route_frame(self, env: Any, vehicle: Any, ego_pos: Optional[np.ndarray]) -> RouteFrenetFrame:
        key = self._route_cache_key_for(vehicle)
        if (
            self._route_cache_frame is not None
            and self._route_cache_frame.valid
            and self._route_cache_key == key
            and self._route_cache_age < max(1, self.config.route_cache_steps)
            and ego_pos is not None
            and self._route_cache_position is not None
            and np.linalg.norm(ego_pos - self._route_cache_position) <= self.config.route_cache_distance
        ):
            self._route_cache_age += 1
            self._last_route_cache_hit = True
            self._last_route_build_time_ms = 0.0
            return self._route_cache_frame

        start = time.time()
        frame = RouteFrenetFrame.build(env, self.config)
        self._route_cache_frame = frame
        self._route_cache_key = key
        self._route_cache_position = None if ego_pos is None else ego_pos.copy()
        self._route_cache_age = 0
        self._last_route_cache_hit = False
        self._last_route_build_time_ms = (time.time() - start) * 1000.0
        return frame

    def _route_cache_key_for(self, vehicle: Any) -> Tuple[Any, ...]:
        nav = getattr(vehicle, "navigation", None)
        if nav is None:
            return ("no_navigation",)
        current = tuple(_lane_id(lane) for lane in _as_lane_list(getattr(nav, "current_ref_lanes", None)))
        nxt = tuple(_lane_id(lane) for lane in _as_lane_list(getattr(nav, "next_ref_lanes", None)))
        current_road = getattr(nav, "current_road", None)
        next_road = getattr(nav, "next_road", None)
        current_road_key = (
            getattr(current_road, "start_node", None),
            getattr(current_road, "end_node", None),
        )
        next_road_key = (
            getattr(next_road, "start_node", None),
            getattr(next_road, "end_node", None),
        )
        lane_index = getattr(vehicle, "lane_index", None)
        if lane_index is None:
            lane = getattr(vehicle, "lane", None)
            lane_index = getattr(lane, "index", None)
        return (str(lane_index), current, nxt, current_road_key, next_road_key)

    def _action_to_accel(self, throttle_brake: float) -> float:
        action = _clip(throttle_brake, -1.0, 1.0)
        return action * self.config.max_accel if action >= 0.0 else action * self.config.max_decel

    def _accel_to_action(self, accel: float) -> float:
        if accel >= 0.0:
            return _clip(accel / max(self.config.max_accel, 1e-6), 0.0, 1.0)
        return _clip(accel / max(self.config.max_decel, 1e-6), -1.0, 0.0)

    def _ultra_fast_gate(
        self,
        env: Any,
        vehicle: Any,
        ego_pos: Optional[np.ndarray],
        ego_heading: float,
        ego_speed: float,
        raw_action: np.ndarray,
    ) -> str:
        progress = 0.0
        if ego_pos is not None and self._last_ego_pos is not None:
            progress = float(np.linalg.norm(ego_pos - self._last_ego_pos))
        if ego_pos is not None:
            self._last_ego_pos = ego_pos.copy()

        if ego_speed < self.config.ultra_low_speed_threshold and progress < self.config.ultra_low_progress_threshold:
            self._low_progress_count += 1
        else:
            self._low_progress_count = 0

        if raw_action[1] <= self.config.ultra_brake_threshold:
            return "raw_action_brake"
        if raw_action[1] <= self.config.ultra_low_throttle_threshold and ego_speed < 3.0:
            return "raw_action_low_throttle"
        if self._low_progress_count >= self.config.ultra_low_progress_steps:
            return "low_speed_low_progress"
        if self._last_recovery_active:
            return "previous_recovery"
        if self._vehicle_risk_flag(vehicle):
            return "vehicle_risk_flag"
        if self._coarse_front_blocker(env, vehicle, ego_pos, ego_heading):
            return "coarse_front_blocker"
        return "low_risk_passthrough"

    def _vehicle_risk_flag(self, vehicle: Any) -> bool:
        if vehicle is None:
            return False
        for attr in ("crash_vehicle", "crash_object", "crash_sidewalk", "crash_building", "out_of_route"):
            try:
                if bool(getattr(vehicle, attr, False)):
                    return True
            except Exception:
                pass
        try:
            if hasattr(vehicle, "on_lane") and not bool(getattr(vehicle, "on_lane")):
                return True
        except Exception:
            pass
        return False

    def _coarse_front_blocker(self, env: Any, vehicle: Any, ego_pos: Optional[np.ndarray], ego_heading: float) -> bool:
        if ego_pos is None:
            return False
        forward = _unit_from_heading(ego_heading)
        left = _left_normal_from_heading(ego_heading)
        scanned = 0
        for obj in self._iter_environment_objects(env):
            if obj is None or obj is vehicle:
                continue
            scanned += 1
            if scanned > max(4, int(self.config.object_scan_limit)):
                break
            pos = _get_position(obj)
            if pos is None:
                continue
            delta = pos - ego_pos
            long = float(np.dot(delta, forward))
            if long <= 0.0 or long > self.config.ultra_front_block_distance:
                continue
            lat = abs(float(np.dot(delta, left)))
            _, obj_width = _get_size(obj)
            if lat <= self.config.ultra_front_lateral_window + obj_width * 0.5:
                speed = _get_speed(obj, 0.0)
                if speed <= max(self.config.blocking_speed_threshold, _get_speed(vehicle, 0.0) * self.config.low_speed_ratio):
                    return True
        return False

    def _format_reject_reasons(self, reasons: Dict[str, int]) -> str:
        return ";".join("{}:{}".format(k, reasons[k]) for k in sorted(reasons))

    def _should_write_detailed_log(self, info: Dict[str, Any]) -> bool:
        if not self.config.log_csv_path:
            return False
        mode = info.get("recovery_mode", "")
        return bool(
            mode in ("predictive_recovery", "minimum_risk_stop", "exception_fallback")
            or info.get("filter_time_warning", False)
        )

    def _low_risk_fast_path(
        self,
        frame: RouteFrenetFrame,
        ego_projection: FrenetProjection,
        ego_width: float,
        raw_action: np.ndarray,
        objects: Sequence[SceneObject],
        scene_info: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if self.config.debug:
            return None
        blocking_distance = _safe_float(scene_info.get("front_blocking_object_distance"), float("inf"))
        if np.isfinite(blocking_distance):
            return None
        for obj in objects:
            if not obj.frenet_valid:
                return None
            rel_s = obj.s - ego_projection.s
            rel_l = obj.l - ego_projection.l
            if -8.0 <= rel_s <= 30.0 and abs(rel_l) <= 5.0:
                return None
        if not ego_projection.frenet_valid:
            return None
        boundary = frame.boundary_at(ego_projection.s, ego_projection.l, ego_width, self.config.safety_margin)
        if not boundary.valid or boundary.boundary_margin < 0.75:
            return None
        if abs(float(raw_action[0])) > 0.85 or abs(float(raw_action[1])) > 0.95:
            return None
        return {
            "recovery_mode": "low_risk_pass_through",
            "recovery_certified": True,
            "selected_candidate_type": "raw_action_low_risk",
            "selected_lateral_target": ego_projection.l,
            "selected_speed_target": 0.0,
            "num_candidates": 0,
            "num_hard_safe_candidates": 0,
            "collision_free": True,
            "boundary_safe": True,
            "control_feasible": True,
            "boundary_margin_score": 1.0 / max(boundary.boundary_margin, 0.2),
            "total_score": 0.0,
            "fallback_reason": "",
        }

    def _parse_scene(
        self,
        env: Any,
        frame: RouteFrenetFrame,
        ego_projection: FrenetProjection,
        ego_speed: float,
        ego_width: float,
    ) -> Tuple[List[SceneObject], Dict[str, Any]]:
        vehicle = _get_ego_vehicle(env)
        ego_pos = _get_position(vehicle)
        ego_id = str(id(vehicle))
        candidates = []
        seen = set()
        scanned = 0
        scan_limit = max(int(self.config.object_scan_limit), int(self.config.max_objects) * 6)
        for obj in self._iter_environment_objects(env):
            scanned += 1
            if scanned > scan_limit:
                break
            if obj is None or obj is vehicle or str(id(obj)) == ego_id:
                continue
            key = str(id(obj))
            if key in seen:
                continue
            seen.add(key)
            pos = _get_position(obj)
            if pos is None or ego_pos is None:
                continue
            heading = _get_heading(obj, 0.0)
            speed = _get_speed(obj, 0.0)
            length, width = _get_size(obj)
            obj_id = self._object_id(obj)
            class_name = obj.__class__.__name__.lower()
            is_vehicle = "vehicle" in class_name or hasattr(obj, "throttle_brake") or hasattr(obj, "speed")
            is_static = speed < 0.5 and not is_vehicle
            object_type = "vehicle" if is_vehicle else "static_obstacle"
            projection = frame.project_point(pos, heading, validate_heading=False)
            rel_s = projection.s - ego_projection.s
            rel_l = projection.l - ego_projection.l
            if projection.frenet_valid:
                if rel_s < -self.config.far_behind_s and abs(rel_l) > self.config.route_lateral_ignore * 0.5:
                    continue
                if rel_s > self.config.route_front_distance:
                    continue
                if abs(rel_l) > self.config.route_lateral_ignore:
                    continue
            else:
                if np.linalg.norm(pos - ego_pos) > self.config.route_front_distance:
                    continue

            low_speed = speed <= max(self.config.blocking_speed_threshold, ego_speed * self.config.low_speed_ratio)
            lateral_overlap = abs(rel_l) <= (ego_width + width) * 0.5 + self.config.safety_margin
            is_blocking = bool(projection.frenet_valid and rel_s > 0.0 and lateral_overlap and low_speed)
            candidates.append(SceneObject(
                object_id=obj_id,
                object_type=object_type,
                position_xy=pos,
                heading=heading,
                speed=speed,
                length=length,
                width=width,
                is_static=is_static or speed < 0.5,
                is_vehicle=is_vehicle,
                is_blocking=is_blocking,
                frenet_valid=projection.frenet_valid,
                s=projection.s,
                l=projection.l,
            ))

        candidates.sort(key=lambda obj: abs(obj.s - ego_projection.s) if obj.frenet_valid else float(np.linalg.norm(obj.position_xy - ego_pos)))
        objects = candidates[:max(0, int(self.config.max_objects))]
        blocking = [obj for obj in objects if obj.is_blocking]
        front = min(blocking, key=lambda obj: obj.s - ego_projection.s) if blocking else None
        left_space, right_space = self._estimate_side_space(frame, ego_projection, ego_width, objects)
        blocker_left_gap = left_space
        blocker_right_gap = right_space
        if front is not None:
            blocker_boundary = frame.boundary_at(front.s, front.l, ego_width, self.config.safety_margin)
            blocker_left_gap = (
                blocker_boundary.l_max
                - front.l
                - front.width * 0.5
                - ego_width * 0.5
                - self.config.safety_margin
            )
            blocker_right_gap = (
                front.l
                - front.width * 0.5
                - blocker_boundary.l_min
                - ego_width * 0.5
                - self.config.safety_margin
            )
        info = {
            "front_blocking_object_type": front.object_type if front is not None else "",
            "front_blocking_object_distance": (front.s - ego_projection.s) if front is not None else float("inf"),
            "front_blocking_object_s": front.s if front is not None else float("inf"),
            "front_blocking_object_l": front.l if front is not None else float("inf"),
            "front_blocking_object_width": front.width if front is not None else 0.0,
            "left_space_available": left_space,
            "right_space_available": right_space,
            "blocker_left_gap": blocker_left_gap,
            "blocker_right_gap": blocker_right_gap,
        }
        return objects, info

    def _iter_environment_objects(self, env: Any) -> Iterable[Any]:
        root = _unwrap_env(env)
        engine = getattr(root, "engine", None)
        sources = []
        if engine is not None:
            traffic_manager = getattr(engine, "traffic_manager", None)
            if traffic_manager is not None:
                for attr in ("vehicles", "traffic_vehicles", "_traffic_vehicles"):
                    if hasattr(traffic_manager, attr):
                        sources.append(getattr(traffic_manager, attr))

            agent_manager = getattr(engine, "agent_manager", None)
            if agent_manager is not None and hasattr(agent_manager, "active_agents"):
                sources.append(getattr(agent_manager, "active_agents"))

            if self.config.debug and hasattr(engine, "get_objects"):
                try:
                    sources.append(engine.get_objects())
                except Exception:
                    pass

        for holder in (root,):
            if holder is None:
                continue
            for attr in ("vehicles", "agents", "objects"):
                if hasattr(holder, attr):
                    sources.append(getattr(holder, attr))

        vehicle = _get_ego_vehicle(env)
        lidar = getattr(vehicle, "lidar", None)
        if lidar is not None and hasattr(lidar, "get_surrounding_vehicles"):
            detected = getattr(lidar, "detected_objects", None)
            if detected is not None:
                try:
                    sources.append(lidar.get_surrounding_vehicles(detected))
                except Exception:
                    pass
        for source in sources:
            for obj in self._iter_source(source):
                yield obj

    def _iter_source(self, source: Any) -> Iterable[Any]:
        if source is None:
            return
        try:
            if callable(source):
                source = source()
        except Exception:
            return
        if isinstance(source, dict):
            values = source.values()
        elif isinstance(source, (list, tuple, set)):
            values = source
        else:
            try:
                values = list(source)
            except Exception:
                values = [source]
        for value in values:
            if isinstance(value, dict):
                for sub_value in value.values():
                    yield sub_value
            else:
                yield value

    def _object_id(self, obj: Any) -> str:
        for attr in ("id", "name", "object_id"):
            if hasattr(obj, attr):
                try:
                    return str(getattr(obj, attr))
                except Exception:
                    pass
        return str(id(obj))

    def _estimate_side_space(
        self,
        frame: RouteFrenetFrame,
        ego_projection: FrenetProjection,
        ego_width: float,
        objects: Sequence[SceneObject],
    ) -> Tuple[float, float]:
        boundary = frame.boundary_at(ego_projection.s, ego_projection.l, ego_width, self.config.safety_margin)
        left = boundary.left_margin
        right = boundary.right_margin
        for obj in objects:
            if not obj.frenet_valid:
                continue
            rel_s = obj.s - ego_projection.s
            if rel_s < -5.0 or rel_s > 35.0:
                continue
            rel_l = obj.l - ego_projection.l
            lateral_clearance = abs(rel_l) - obj.width * 0.5 - ego_width * 0.5 - self.config.safety_margin
            if rel_l > 0.0:
                left = min(left, lateral_clearance)
            elif rel_l < 0.0:
                right = min(right, lateral_clearance)
        return float(left), float(right)

    def _generate_candidate_specs(
        self,
        frame: RouteFrenetFrame,
        ego_projection: FrenetProjection,
        ego_pos: Optional[np.ndarray],
        ego_heading: float,
        ego_speed: float,
        ego_width: float,
        raw_action: np.ndarray,
        scene_info: Dict[str, Any],
        max_steer_rad: float,
    ) -> List[TrajectoryCandidate]:
        del max_steer_rad
        if ego_pos is None:
            ego_pos = np.zeros(2, dtype=float)
        current_s = ego_projection.s
        current_l = ego_projection.l if ego_projection.frenet_valid else 0.0
        terminal_s = current_s + max(ego_speed, 5.0) * self.config.horizon
        boundary = frame.boundary_at(terminal_s, current_l, ego_width, self.config.safety_margin)
        l_low = boundary.l_min + ego_width * 0.5 + self.config.safety_margin
        l_high = boundary.l_max - ego_width * 0.5 - self.config.safety_margin
        if l_low >= l_high:
            l_low = -self.config.default_lane_width * 0.5
            l_high = self.config.default_lane_width * 0.5

        target_values = [current_l, 0.0]
        n_lat = max(1, int(self.config.num_lateral_targets))
        offsets = np.linspace(-self.config.max_lateral_offset, self.config.max_lateral_offset, num=n_lat)
        target_values.extend([current_l + float(offset) for offset in offsets])

        blocking_distance = _safe_float(scene_info.get("front_blocking_object_distance"), float("inf"))
        if np.isfinite(blocking_distance):
            blocking_l = _safe_float(scene_info.get("front_blocking_object_l"), current_l)
            blocking_width = _safe_float(scene_info.get("front_blocking_object_width"), ego_width)
            target_values.append(blocking_l + blocking_width * 0.5 + ego_width * 0.5 + 0.8)
            target_values.append(blocking_l - blocking_width * 0.5 - ego_width * 0.5 - 0.8)

        lateral_targets = []
        for value in target_values:
            clipped = _clip(value, l_low, l_high)
            if all(abs(clipped - existing) > 0.25 for existing in lateral_targets):
                lateral_targets.append(clipped)
        lateral_targets = sorted(lateral_targets, key=lambda x: abs(x - current_l))[:max(1, n_lat + 2)]

        raw_target_speed = _clip(ego_speed + self._action_to_accel(raw_action[1]) * 1.5, 0.0, self.config.max_speed)
        speed_values = [
            raw_target_speed,
            min(ego_speed, max(2.0, ego_speed * 0.6)),
            3.0,
            min(8.0, self.config.max_speed),
            0.0,
        ]
        if not np.isfinite(blocking_distance):
            speed_values = [raw_target_speed, ego_speed, min(10.0, self.config.max_speed), 3.0, 0.0]
        speed_targets = []
        for value in speed_values:
            value = _clip(value, 0.0, self.config.max_speed)
            if all(abs(value - existing) > 0.4 for existing in speed_targets):
                speed_targets.append(value)
        speed_targets = speed_targets[:max(1, int(self.config.num_speed_targets))]

        combos = [(lat, speed) for lat in lateral_targets for speed in speed_targets]
        combos.sort(key=lambda item: self._candidate_priority(item[0], item[1], current_l, raw_target_speed, has_blocking=np.isfinite(blocking_distance)))
        max_candidates = max(1, int(self.config.max_candidates))
        selected_combos = combos[:max_candidates]
        keep_stop = min(combos, key=lambda item: abs(item[0] - current_l) + abs(item[1])) if combos else None
        if keep_stop is not None and keep_stop not in selected_combos:
            selected_combos[-1] = keep_stop

        specs = []
        for candidate_id, (lateral_target, speed_target) in enumerate(selected_combos):
            candidate_type = self._candidate_type(current_l, lateral_target, ego_speed, speed_target)
            candidate = TrajectoryCandidate(candidate_id, candidate_type, lateral_target, speed_target, raw_action.copy())
            specs.append(candidate)
        return specs

    def _candidate_priority(
        self,
        lateral_target: float,
        speed_target: float,
        current_l: float,
        raw_target_speed: float,
        has_blocking: bool,
    ) -> float:
        lateral_shift = abs(lateral_target - current_l)
        speed_shift = abs(speed_target - raw_target_speed)
        priority = lateral_shift + 0.08 * speed_shift
        if has_blocking and lateral_shift > 0.5 and speed_target > 1.0:
            priority -= 1.0
        if speed_target < 0.5:
            priority -= 0.2
        return priority

    def _candidate_type(self, current_l: float, lateral_target: float, ego_speed: float, speed_target: float) -> str:
        if speed_target < 0.5:
            base = "stop"
        elif speed_target < ego_speed - 1.0:
            base = "slow_follow"
        else:
            base = "pass"
        delta_l = lateral_target - current_l
        if delta_l > 0.5:
            return "left_" + base
        if delta_l < -0.5:
            return "right_" + base
        return "keep_" + base

    def _rollout_and_check_candidate(
        self,
        candidate: TrajectoryCandidate,
        frame: RouteFrenetFrame,
        objects: Sequence[SceneObject],
        ego_projection: FrenetProjection,
        ego_pos: Optional[np.ndarray],
        ego_heading: float,
        ego_speed: float,
        ego_length: float,
        ego_width: float,
        max_steer_rad: float,
    ) -> TrajectoryRollout:
        if ego_pos is None:
            ego_pos = np.zeros(2, dtype=float)
        current_s = ego_projection.s
        current_l = ego_projection.l if ego_projection.frenet_valid else 0.0
        requested_dt = max(self.config.dt, 1e-3)
        requested_steps = max(2, int(round(self.config.horizon / requested_dt)))
        steps = min(requested_steps, max(2, int(self.config.max_rollout_steps)))
        dt = max(self.config.horizon / float(steps), requested_dt)

        times: List[float] = []
        positions: List[np.ndarray] = []
        headings: List[float] = []
        speeds: List[float] = []
        accelerations: List[float] = []
        steer_actions: List[float] = []
        throttle_actions: List[float] = []
        frenet_s_values: List[float] = []
        frenet_l_values: List[float] = []

        prev_speed = ego_speed
        prev_heading = ego_heading
        prev_position = ego_pos
        prev_steer = 0.0
        s_value = current_s
        min_boundary_margin = float("inf")
        min_obstacle_margin = float("inf")
        hard_safe = True
        failure_reason = ""

        for idx in range(steps):
            t = (idx + 1) * dt
            u = _clip(t / max(self.config.horizon, dt), 0.0, 1.0)
            smooth = 3.0 * u ** 2 - 2.0 * u ** 3
            l_value = current_l + (candidate.lateral_target - current_l) * smooth

            delta_v = candidate.speed_target - prev_speed
            dv = _clip(delta_v, -self.config.max_decel * dt, self.config.max_accel * dt)
            speed = _clip(prev_speed + dv, 0.0, self.config.max_speed)
            accel = (speed - prev_speed) / dt
            s_value += 0.5 * (prev_speed + speed) * dt
            position, _ = frame.frenet_to_world(s_value, l_value)
            heading = _heading_from_vector(position - prev_position, prev_heading)
            heading_error = _angle_diff(heading, prev_heading)
            steer_action = _clip(heading_error / max(max_steer_rad, 1e-3), -1.0, 1.0)
            throttle_action = self._accel_to_action(accel)

            times.append(t)
            positions.append(position)
            headings.append(heading)
            speeds.append(speed)
            accelerations.append(accel)
            steer_actions.append(steer_action)
            throttle_actions.append(throttle_action)
            frenet_s_values.append(s_value)
            frenet_l_values.append(l_value)

            if speed < -1e-4:
                hard_safe = False
                failure_reason = "negative_speed"
                break
            if abs(steer_action) > self.config.max_steer_action + 1e-6 or abs(throttle_action) > 1.0 + 1e-6:
                hard_safe = False
                failure_reason = "control_limit"
                break
            steer_rate = abs(steer_action - prev_steer) / dt
            if steer_rate > self.config.max_steer_rate:
                hard_safe = False
                failure_reason = "steer_rate_limit"
                break

            boundary = frame.boundary_at(s_value, l_value, ego_width, self.config.safety_margin)
            min_boundary_margin = min(min_boundary_margin, boundary.boundary_margin)
            if not boundary.valid:
                hard_safe = False
                failure_reason = "boundary"
                break

            for obj in objects:
                obj_pos = self._predict_object_position(obj, t)
                gap = self._center_gap(position, ego_length, ego_width, obj_pos, obj.length, obj.width)
                min_obstacle_margin = min(min_obstacle_margin, gap)
                if gap < self.config.obstacle_margin and _rectangles_intersect(
                    position,
                    heading,
                    ego_length + 2.0 * self.config.hard_collision_margin,
                    ego_width + 2.0 * self.config.hard_collision_margin,
                    obj_pos,
                    obj.heading,
                    obj.length + 2.0 * self.config.hard_collision_margin,
                    obj.width + 2.0 * self.config.hard_collision_margin,
                ):
                    hard_safe = False
                    failure_reason = "collision"
                    break
                if obj.is_vehicle and obj.frenet_valid:
                    rel_s = obj.s - s_value
                    lateral_gap = abs(obj.l - l_value) - ego_width * 0.5 - obj.width * 0.5
                    if -4.0 < rel_s < 3.0 and lateral_gap < -0.1:
                        hard_safe = False
                        failure_reason = "cut_in_danger"
                        break
            if not hard_safe:
                break

            prev_speed = speed
            prev_heading = heading
            prev_position = position
            prev_steer = steer_action

        rollout = TrajectoryRollout(
            candidate=candidate,
            times=np.asarray(times, dtype=float),
            positions=np.asarray(positions, dtype=float).reshape(-1, 2) if positions else np.zeros((0, 2), dtype=float),
            headings=np.asarray(headings, dtype=float),
            speeds=np.asarray(speeds, dtype=float),
            accelerations=np.asarray(accelerations, dtype=float),
            steer_actions=np.asarray(steer_actions, dtype=float),
            throttle_actions=np.asarray(throttle_actions, dtype=float),
            frenet_s=np.asarray(frenet_s_values, dtype=float),
            frenet_l=np.asarray(frenet_l_values, dtype=float),
            hard_safe=hard_safe and len(times) == steps,
            failure_reason=failure_reason,
            min_boundary_margin=min_boundary_margin,
            min_obstacle_margin=min_obstacle_margin,
        )
        return rollout

    def _rollout_candidate(
        self,
        candidate: TrajectoryCandidate,
        frame: RouteFrenetFrame,
        current_s: float,
        current_l: float,
        ego_pos: np.ndarray,
        ego_heading: float,
        ego_speed: float,
        max_steer_rad: float,
    ) -> TrajectoryRollout:
        requested_dt = max(self.config.dt, 1e-3)
        requested_steps = max(2, int(round(self.config.horizon / requested_dt)))
        steps = min(requested_steps, max(2, int(self.config.max_rollout_steps)))
        dt = max(self.config.horizon / float(steps), requested_dt)
        times = np.arange(1, steps + 1, dtype=float) * dt
        u = np.clip(times / max(self.config.horizon, dt), 0.0, 1.0)
        smooth = 3.0 * u ** 2 - 2.0 * u ** 3
        frenet_l = current_l + (candidate.lateral_target - current_l) * smooth

        speeds = np.zeros(steps, dtype=float)
        accelerations = np.zeros(steps, dtype=float)
        frenet_s = np.zeros(steps, dtype=float)
        prev_speed = ego_speed
        s_value = current_s
        for idx in range(steps):
            delta_v = candidate.speed_target - prev_speed
            dv = _clip(delta_v, -self.config.max_decel * dt, self.config.max_accel * dt)
            new_speed = _clip(prev_speed + dv, 0.0, self.config.max_speed)
            accel = (new_speed - prev_speed) / dt
            s_value += 0.5 * (prev_speed + new_speed) * dt
            speeds[idx] = new_speed
            accelerations[idx] = accel
            frenet_s[idx] = s_value
            prev_speed = new_speed

        positions = np.zeros((steps, 2), dtype=float)
        road_headings = np.zeros(steps, dtype=float)
        for idx, (s_value, l_value) in enumerate(zip(frenet_s, frenet_l)):
            pos, heading = frame.frenet_to_world(float(s_value), float(l_value))
            positions[idx] = pos
            road_headings[idx] = heading

        headings = np.zeros(steps, dtype=float)
        previous = ego_pos
        previous_heading = ego_heading
        for idx in range(steps):
            delta = positions[idx] - previous
            headings[idx] = _heading_from_vector(delta, previous_heading)
            previous = positions[idx]
            previous_heading = headings[idx]

        steer_actions = np.zeros(steps, dtype=float)
        throttle_actions = np.zeros(steps, dtype=float)
        previous_heading = ego_heading
        for idx in range(steps):
            heading_error = _angle_diff(headings[idx], previous_heading)
            steer_actions[idx] = _clip(heading_error / max(max_steer_rad, 1e-3), -1.0, 1.0)
            throttle_actions[idx] = self._accel_to_action(accelerations[idx])
            previous_heading = headings[idx]
        return TrajectoryRollout(
            candidate=candidate,
            times=times,
            positions=positions,
            headings=headings,
            speeds=speeds,
            accelerations=accelerations,
            steer_actions=steer_actions,
            throttle_actions=throttle_actions,
            frenet_s=frenet_s,
            frenet_l=frenet_l,
        )

    def _check_hard_safety(
        self,
        rollout: TrajectoryRollout,
        frame: RouteFrenetFrame,
        objects: Sequence[SceneObject],
        ego_length: float,
        ego_width: float,
    ) -> TrajectoryRollout:
        collision_free = True
        boundary_safe = True
        control_feasible = True
        min_boundary_margin = float("inf")
        min_obstacle_margin = float("inf")
        prev_steer = 0.0
        prev_time = 0.0
        for idx in range(len(rollout.times)):
            if rollout.speeds[idx] < -1e-4:
                control_feasible = False
                rollout.failure_reason = "negative_speed"
                break
            if abs(rollout.steer_actions[idx]) > self.config.max_steer_action + 1e-6 or abs(rollout.throttle_actions[idx]) > 1.0 + 1e-6:
                control_feasible = False
                rollout.failure_reason = "control_limit"
                break
            step_dt = max(float(rollout.times[idx] - prev_time), 1e-3)
            steer_rate = abs(rollout.steer_actions[idx] - prev_steer) / step_dt
            if steer_rate > self.config.max_steer_rate:
                control_feasible = False
                rollout.failure_reason = "steer_rate_limit"
                break
            prev_steer = rollout.steer_actions[idx]
            prev_time = float(rollout.times[idx])

            boundary = frame.boundary_at(rollout.frenet_s[idx], rollout.frenet_l[idx], ego_width, self.config.safety_margin)
            min_boundary_margin = min(min_boundary_margin, boundary.boundary_margin)
            if not boundary.valid:
                boundary_safe = False
                rollout.failure_reason = "boundary"
                break

            for obj in objects:
                obj_pos = self._predict_object_position(obj, rollout.times[idx])
                gap = self._center_gap(rollout.positions[idx], ego_length, ego_width, obj_pos, obj.length, obj.width)
                min_obstacle_margin = min(min_obstacle_margin, gap)
                if gap < self.config.obstacle_margin and _rectangles_intersect(
                    rollout.positions[idx],
                    rollout.headings[idx],
                    ego_length + 2.0 * self.config.hard_collision_margin,
                    ego_width + 2.0 * self.config.hard_collision_margin,
                    obj_pos,
                    obj.heading,
                    obj.length + 2.0 * self.config.hard_collision_margin,
                    obj.width + 2.0 * self.config.hard_collision_margin,
                ):
                    collision_free = False
                    rollout.failure_reason = "collision"
                    break
            if not collision_free:
                break

        rollout.hard_safe = collision_free and boundary_safe and control_feasible
        rollout.min_boundary_margin = min_boundary_margin
        rollout.min_obstacle_margin = min_obstacle_margin
        return rollout

    def _score_rollout(
        self,
        rollout: TrajectoryRollout,
        frame: RouteFrenetFrame,
        objects: Sequence[SceneObject],
        ego_projection: FrenetProjection,
        ego_length: float,
        ego_width: float,
        raw_action: np.ndarray,
        scene_info: Dict[str, Any],
    ) -> TrajectoryScore:
        progress = float(rollout.frenet_s[-1] - ego_projection.s)
        progress_score = -progress
        avg_speed = float(np.mean(rollout.speeds)) if rollout.speeds.size else 0.0
        blocking_distance = _safe_float(scene_info.get("front_blocking_object_distance"), float("inf"))
        has_blocking = np.isfinite(blocking_distance)
        blocker_l = _safe_float(scene_info.get("front_blocking_object_l"), ego_projection.l)
        blocker_width = _safe_float(scene_info.get("front_blocking_object_width"), ego_width)
        blocker_left_gap = _safe_float(scene_info.get("blocker_left_gap"), 0.0)
        blocker_right_gap = _safe_float(scene_info.get("blocker_right_gap"), 0.0)
        side_channel_available = max(blocker_left_gap, blocker_right_gap) > 0.35
        deadlock_penalty = 0.0
        if has_blocking:
            if avg_speed < 1.5:
                deadlock_penalty += 10.0 if side_channel_available else 4.0
            target_progress = min(blocking_distance + 6.0, max(8.0, self.config.horizon * 5.0))
            if progress < target_progress:
                deadlock_penalty += max(0.0, target_progress - progress) * (0.9 if side_channel_available else 0.35)

        rss_long_score, rss_long_margin = self._longitudinal_rss_score(rollout, frame, objects, ego_length, ego_width)
        rss_lat_score, rss_lat_margin = self._lateral_rss_score(rollout, objects, ego_length, ego_width)
        obstacle_margin_score = 1.0 / max(rollout.min_obstacle_margin, 0.2) if np.isfinite(rollout.min_obstacle_margin) else 0.0
        boundary_margin_score = 1.0 / max(rollout.min_boundary_margin, 0.2) if np.isfinite(rollout.min_boundary_margin) else 0.0
        if rollout.frenet_l.size > 1:
            lateral_velocity = np.diff(rollout.frenet_l) / np.maximum(np.diff(rollout.times), 1e-3)
            boundary_margin_score += max(0.0, float(np.max(np.abs(lateral_velocity))) - 1.5) * 0.1
        smoothness_score = float(np.mean(np.abs(np.diff(rollout.steer_actions)))) if rollout.steer_actions.size > 1 else 0.0
        smoothness_score += 0.2 * (float(np.mean(np.abs(np.diff(rollout.accelerations)))) if rollout.accelerations.size > 1 else 0.0)
        nominal_deviation_score = float(np.linalg.norm(np.array([rollout.steer_actions[0], rollout.throttle_actions[0]]) - raw_action))
        continuity_score = 0.0
        if self.last_selected_lateral is not None:
            continuity_score += abs(rollout.candidate.lateral_target - self.last_selected_lateral)
        if self.last_selected_speed is not None:
            continuity_score += 0.1 * abs(rollout.candidate.speed_target - self.last_selected_speed)

        terminal_recovery_score = 0.0
        terminal_passed_blocker = False
        terminal_recoverable = False
        if has_blocking:
            terminal_s = rollout.frenet_s[-1]
            terminal_l = rollout.frenet_l[-1]
            front_s = ego_projection.s + blocking_distance
            lateral_clearance_at_terminal = abs(terminal_l - blocker_l) - blocker_width * 0.5 - ego_width * 0.5
            terminal_passed_blocker = terminal_s > front_s + ego_length * 0.5
            terminal_recoverable = terminal_passed_blocker or (
                lateral_clearance_at_terminal > self.config.safety_margin + 0.25
                and terminal_s > front_s - ego_length
            )
            if terminal_recoverable:
                terminal_recovery_score -= 12.0
                progress_score -= 0.5 * max(0.0, progress - blocking_distance)
            elif rollout.speeds[-1] < 1.0 and terminal_s < front_s:
                terminal_recovery_score += 14.0 if side_channel_available else 5.0
                deadlock_penalty += 8.0 if side_channel_available else 2.0
            elif terminal_s < front_s and avg_speed < 2.0:
                terminal_recovery_score += 8.0 if side_channel_available else 3.0

        rss_risk_score = rss_long_score + rss_lat_score
        total = (
            self.config.w_progress * progress_score
            + self.config.w_deadlock * deadlock_penalty
            + self.config.w_rss_longitudinal * rss_long_score
            + self.config.w_rss_lateral * rss_lat_score
            + self.config.w_obstacle * obstacle_margin_score
            + self.config.w_boundary * boundary_margin_score
            + self.config.w_smoothness * smoothness_score
            + self.config.w_nominal * nominal_deviation_score
            + self.config.w_continuity * continuity_score
            + self.config.w_terminal * terminal_recovery_score
        )
        return TrajectoryScore(
            progress_score=progress_score,
            deadlock_penalty=deadlock_penalty,
            rss_longitudinal_score=rss_long_score,
            rss_lateral_score=rss_lat_score,
            obstacle_margin_score=obstacle_margin_score,
            boundary_margin_score=boundary_margin_score,
            smoothness_score=smoothness_score,
            nominal_deviation_score=nominal_deviation_score,
            continuity_score=continuity_score,
            terminal_recovery_score=terminal_recovery_score,
            total_score=total,
            rss_longitudinal_margin=rss_long_margin,
            rss_lateral_margin=rss_lat_margin,
            rss_risk_score=rss_risk_score,
            collision_free=True,
            boundary_safe=True,
            control_feasible=True,
            terminal_passed_blocker=terminal_passed_blocker,
            terminal_recoverable=terminal_recoverable,
        )

    def _longitudinal_rss_score(
        self,
        rollout: TrajectoryRollout,
        frame: RouteFrenetFrame,
        objects: Sequence[SceneObject],
        ego_length: float,
        ego_width: float,
    ) -> Tuple[float, float]:
        min_margin = float("inf")
        risk = 0.0
        for idx, t in enumerate(rollout.times):
            ego_s = rollout.frenet_s[idx]
            ego_l = rollout.frenet_l[idx]
            ego_v = rollout.speeds[idx]
            for obj in objects:
                obj_s, obj_l = self._object_frenet_at(obj, frame, t)
                lateral_overlap = abs(obj_l - ego_l) <= (ego_width + obj.width) * 0.5 + 0.8
                if not lateral_overlap:
                    continue
                long_gap = obj_s - ego_s - ego_length * 0.5 - obj.length * 0.5
                if long_gap < -2.0:
                    continue
                required = (
                    self.config.min_longitudinal_safe_distance
                    + ego_v * self.config.reaction_time
                    + max(0.0, ego_v ** 2 / (2.0 * self.config.ego_max_decel) - obj.speed ** 2 / (2.0 * self.config.front_max_decel))
                )
                margin = long_gap - required
                min_margin = min(min_margin, margin)
                if margin < 0.0:
                    risk += min(20.0, -margin) / 10.0
        return risk / max(1, len(rollout.times)), min_margin

    def _lateral_rss_score(
        self,
        rollout: TrajectoryRollout,
        objects: Sequence[SceneObject],
        ego_length: float,
        ego_width: float,
    ) -> Tuple[float, float]:
        min_margin = float("inf")
        risk = 0.0
        time_delta = np.diff(np.concatenate([[0.0], rollout.times]))
        l_dot = np.diff(np.concatenate([[rollout.frenet_l[0]], rollout.frenet_l])) / np.maximum(time_delta, 1e-3)
        for idx in range(len(rollout.times)):
            ego_s = rollout.frenet_s[idx]
            ego_l = rollout.frenet_l[idx]
            for obj in objects:
                if not obj.frenet_valid:
                    continue
                longitudinal_near = abs(obj.s - ego_s) <= ego_length + obj.length + 6.0
                if not longitudinal_near:
                    continue
                lateral_gap = abs(obj.l - ego_l) - ego_width * 0.5 - obj.width * 0.5
                required = self.config.lateral_safe_distance + abs(l_dot[idx]) * self.config.lateral_response_time
                margin = lateral_gap - required
                min_margin = min(min_margin, margin)
                if margin < 0.0:
                    risk += min(10.0, -margin) / 5.0
        return risk / max(1, len(rollout.times)), min_margin

    def _object_frenet_at(self, obj: SceneObject, frame: RouteFrenetFrame, t: float) -> Tuple[float, float]:
        if obj.frenet_valid:
            route_heading = frame.heading_at(obj.s)
            forward_speed = obj.speed * math.cos(_angle_diff(obj.heading, route_heading))
            return obj.s + forward_speed * float(t), obj.l
        pos = self._predict_object_position(obj, t)
        proj = frame.project_point(pos, obj.heading, validate_heading=False)
        return proj.s, proj.l

    def _score_to_info(self, score: TrajectoryScore) -> Dict[str, Any]:
        return {
            "rss_longitudinal_margin": score.rss_longitudinal_margin,
            "rss_lateral_margin": score.rss_lateral_margin,
            "rss_risk_score": score.rss_risk_score,
            "rss_longitudinal_score": score.rss_longitudinal_score,
            "rss_lateral_score": score.rss_lateral_score,
            "obstacle_margin_score": score.obstacle_margin_score,
            "boundary_margin_score": score.boundary_margin_score,
            "progress_score": score.progress_score,
            "deadlock_penalty": score.deadlock_penalty,
            "smoothness_score": score.smoothness_score,
            "nominal_deviation_score": score.nominal_deviation_score,
            "continuity_score": score.continuity_score,
            "terminal_recovery_score": score.terminal_recovery_score,
            "total_score": score.total_score,
            "selected_terminal_passed_blocker": score.terminal_passed_blocker,
            "selected_terminal_recoverable": score.terminal_recoverable,
        }

    def _predict_object_position(self, obj: SceneObject, t: float) -> np.ndarray:
        direction = _unit_from_heading(obj.heading)
        return obj.position_xy + direction * obj.speed * float(t)

    def _center_gap(
        self,
        pos_a: np.ndarray,
        length_a: float,
        width_a: float,
        pos_b: np.ndarray,
        length_b: float,
        width_b: float,
    ) -> float:
        center_dist = float(np.linalg.norm(pos_a - pos_b))
        radius_a = 0.5 * math.sqrt(length_a ** 2 + width_a ** 2)
        radius_b = 0.5 * math.sqrt(length_b ** 2 + width_b ** 2)
        return center_dist - radius_a - radius_b

    def _minimum_risk_stop(self, vehicle: Any) -> np.ndarray:
        current_steer = _safe_float(getattr(vehicle, "steering", 0.0), 0.0)
        speed = _get_speed(vehicle, 0.0)
        brake = -0.8 if speed > 1.0 else -0.3
        return np.array([_clip(current_steer * 0.4, -0.25, 0.25), brake], dtype=np.float32)

    def _write_csv(self, info: Dict[str, Any]) -> None:
        path = self.config.log_csv_path
        if not path:
            return
        try:
            folder = os.path.dirname(path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            file_exists = os.path.exists(path)
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=RECOVERY_DIAGNOSTIC_FIELDS)
                if not file_exists or not self._csv_header_written:
                    writer.writeheader()
                    self._csv_header_written = True
                writer.writerow({k: self._scalar_for_csv(info.get(k, "")) for k in RECOVERY_DIAGNOSTIC_FIELDS})
        except Exception:
            if self.config.debug:
                raise

    def _scalar_for_csv(self, value: Any) -> Any:
        if isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        return str(value)


def _rectangle_corners(center: np.ndarray, heading: float, length: float, width: float) -> np.ndarray:
    forward = _unit_from_heading(heading)
    left = _left_normal_from_heading(heading)
    half_l = length * 0.5
    half_w = width * 0.5
    return np.array([
        center + forward * half_l + left * half_w,
        center + forward * half_l - left * half_w,
        center - forward * half_l - left * half_w,
        center - forward * half_l + left * half_w,
    ], dtype=float)


def _project_polygon(axis: np.ndarray, points: np.ndarray) -> Tuple[float, float]:
    values = points.dot(axis)
    return float(np.min(values)), float(np.max(values))


def _rectangles_intersect(
    center_a: np.ndarray,
    heading_a: float,
    length_a: float,
    width_a: float,
    center_b: np.ndarray,
    heading_b: float,
    length_b: float,
    width_b: float,
) -> bool:
    corners_a = _rectangle_corners(center_a, heading_a, length_a, width_a)
    corners_b = _rectangle_corners(center_b, heading_b, length_b, width_b)
    axes = []
    for corners in (corners_a, corners_b):
        for idx in range(2):
            edge = corners[(idx + 1) % 4] - corners[idx]
            norm = np.linalg.norm(edge)
            if norm > 1e-8:
                axis = np.array([-edge[1], edge[0]], dtype=float) / norm
                axes.append(axis)
    for axis in axes:
        min_a, max_a = _project_polygon(axis, corners_a)
        min_b, max_b = _project_polygon(axis, corners_b)
        if max_a < min_b or max_b < min_a:
            return False
    return True
