import csv
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


RECOVERY_DIAGNOSTIC_FIELDS = [
    # 运行时状态
    "recovery_mode",
    "recovery_certified",
    "accepted_by_filter",
    "filter_intervened",
    "actual_action_modified",
    "model_predicted_safe",
    # 选中的候选
    "selected_candidate_type",
    "selected_lateral_target",
    "selected_speed_target",
    "num_candidates",
    "num_hard_safe_candidates",
    # 安全指标
    "collision_free",
    "boundary_safe",
    "control_feasible",
    "rss_longitudinal_margin",
    "rss_lateral_margin",
    "rss_risk_score",
    # 评分
    "obstacle_margin_score",
    "boundary_margin_score",
    "progress_score",
    "deadlock_penalty",
    "smoothness_score",
    "nominal_deviation_score",
    "continuity_score",
    "terminal_recovery_score",
    "total_score",
    # 诊断信息
    "fallback_reason",
    "filter_time_ms",
    "frenet_valid",
    "frenet_s",
    "frenet_l",
    "front_blocking_object_type",
    "front_blocking_object_distance",
    "left_space_available",
    "right_space_available",
    "ultra_light_gate_reason",
    "ultra_light_gate_passed",
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
    # Raw action 对照（核心）
    "raw_predicted_collision",
    "raw_predicted_out_of_road",
    "raw_predicted_cost_risk",
    "raw_min_vehicle_margin",
    "raw_min_static_margin",
    "raw_min_boundary_margin",
    "raw_deadlock_risk",
    "raw_total_score",
    # 接管判断（核心）
    "allow_intervention",
    "intervention_reason",
    "intervention_rejected_reason",
    "intervention_score_gain",
    "filter_would_intervene",
    # 候选对比
    "candidate_predicted_collision",
    "candidate_predicted_out_of_road",
    "candidate_predicted_cost_risk",
    "candidate_min_vehicle_margin",
    "candidate_min_static_margin",
    "candidate_min_boundary_margin",
    "cost_risk_improved",
    "collision_risk_improved",
    "boundary_risk_improved",
    "vehicle_margin_worse",
    # Action 限幅
    "action_limited_by_raw_delta",
    "safe_raw_steer_delta",
    "safe_raw_acc_delta",
    # 候选评估
    "vehicle_margin_min",
    "static_margin_min",
    "boundary_margin_min",
    "collision_hard_reject_count",
    "severe_lateral_rss_reject_count",
    "cost_risk_hard_reject_count",
    # Episode 统计
    "intervention_count_this_ep",
    "cooldown_remaining",
    "temporary_passthrough",
    # 调试选项
    "debug_shadow_record",
    "would_selected_candidate_type",
    "would_safe_steer",
    "would_safe_acc",
    # Cost streak 熔断
    "cost_streak_sum_recent_5",
    "disable_real_intervention",
    "cost_streak_guard_triggered",
    "cost_streak_guard_reason",
    # Emergency check 详情
    "emergency_detected",
    "emergency_reason",
    "nearest_vehicle_distance",
    "nearest_static_object_distance",
    "contact_state_detected",
    "front_blocking_distance",
]


@dataclass
class PredictiveRecoveryConfig:
    horizon: float = 3.0
    dt: float = 0.2
    num_lateral_targets: int = 7
    num_speed_targets: int = 4
    max_objects: int = 6
    max_candidates: int = 10
    max_rollout_steps: int = 12
    object_scan_limit: int = 12
    debug: bool = False
    log_csv_path: str = ""

    route_sample_interval: float = 3.0
    route_min_length: float = 40.0
    route_extra_length: float = 15.0
    route_cache_steps: int = 30
    route_cache_distance: float = 20.0
    route_max_future_groups: int = 1
    route_front_distance: float = 60.0
    default_lane_width: float = 3.5
    default_road_half_width: float = 5.25
    max_projection_distance: float = 5.5
    max_heading_error: float = math.radians(115.0)

    safety_margin: float = 0.60
    boundary_hard_margin: float = 0.15
    boundary_comfort_margin: float = 1.0
    obstacle_margin: float = 0.80
    hard_collision_margin: float = 0.35
    max_lateral_offset: float = 5.0
    route_lateral_ignore: float = 12.0
    far_behind_s: float = 12.0

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
    filter_time_warn_ms: float = 35.0
    ultra_brake_threshold: float = -0.15
    ultra_low_throttle_threshold: float = 0.05
    ultra_low_speed_threshold: float = 1.0
    ultra_low_progress_threshold: float = 0.25
    ultra_low_progress_steps: int = 5
    ultra_front_block_distance: float = 25.0
    ultra_front_lateral_window: float = 2.8
    max_lateral_without_blocker: float = 0.35
    max_lateral_when_frenet_unstable: float = 0.5

    # === 接管限制 ===
    intervention_score_margin: float = 2.5  # 提高阈值，更保守
    max_steer_delta_from_raw: float = 0.20
    max_acc_delta_from_raw: float = 0.30
    max_steer_delta_from_prev: float = 0.20
    max_acc_delta_from_prev: float = 0.30
    max_interventions_per_episode: int = 10
    min_steps_between_interventions: int = 10
    intervention_cooldown_steps: int = 10

    # === 车辆碰撞硬约束 ===
    hard_vehicle_lateral_margin: float = 0.80
    hard_vehicle_longitudinal_margin: float = 1.50
    hard_static_lateral_margin: float = 0.60
    hard_static_longitudinal_margin: float = 1.00
    severe_lateral_rss_margin: float = -0.50

    # === 边界约束 ===
    min_boundary_margin_for_bypass: float = 0.60  # 提高阈值
    min_vehicle_margin_for_intervention: float = 0.60  # 新增：接管最低车辆 margin
    comfort_boundary_margin: float = 1.00

    # === 调试 ===
    _debug_shadow_record: bool = False

    # === 权重 ===
    w_progress: float = 1.8
    w_deadlock: float = 5.0
    w_rss_longitudinal: float = 4.0
    w_rss_lateral: float = 3.0
    w_obstacle: float = 5.0
    w_boundary: float = 8.0
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
    min_vehicle_margin: float = float("inf")
    min_static_margin: float = float("inf")
    predicted_collision: bool = False
    predicted_out_of_road: bool = False
    predicted_cost_risk: float = 0.0


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

        # === 新增：接管次数限制 ===
        self._intervention_count_this_ep: int = 0
        self._last_intervention_step: int = 0
        self._cooldown_remaining: int = 0
        self._temporary_passthrough: bool = False
        self._cost_history: List[float] = []
        self._recent_interventions: List[int] = []  # 记录最近接管后的 cost 增量

        # === 新增：cost streak 熔断 ===
        self._cost_streak_recent: List[float] = []  # 最近 5 步的 cost 列表
        self._disable_real_intervention_for_episode: bool = False
        self._cost_streak_guard_triggered: bool = False
        self._cost_streak_guard_reason: str = ""

        # 前一时刻的 safe action（用于限幅）
        self._prev_safe_steer: Optional[float] = None
        self._prev_safe_acc: Optional[float] = None

        # 记录状态
        self._last_vehicle_margin_min: float = float("inf")
        self._last_static_margin_min: float = float("inf")
        self._last_rss_lateral_margin: float = float("inf")

    def filter(self, env: Any, obs: Any, raw_action: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        del obs
        start = time.time()
        raw_action_np = self._coerce_action(raw_action)
        info = self._empty_info()

        # 调试标记
        info["debug_shadow_record"] = self.config._debug_shadow_record
        info["intervention_count_this_ep"] = self._intervention_count_this_ep
        info["cooldown_remaining"] = self._cooldown_remaining
        info["temporary_passthrough"] = self._temporary_passthrough

        try:
            vehicle = _get_ego_vehicle(env)
            ego_pos = _get_position(vehicle)
            ego_heading = _get_heading(vehicle, 0.0)
            ego_speed = _get_speed(vehicle, 0.0)
            ego_length, ego_width = _get_size(vehicle)
            max_steer_rad = self._max_steer_rad(vehicle)

            # ===== cost streak 熔断检查 =====
            # 获取当前 step 的 cost
            current_cost = 0.0
            try:
                user_data = getattr(env, "user_data", {})
                cost_list = user_data.get("cost", [])
                if isinstance(cost_list, list) and len(cost_list) > 0:
                    current_cost = float(cost_list[-1])
                else:
                    current_cost = float(cost_list) if cost_list else 0.0
            except Exception:
                current_cost = 0.0

            # 更新 cost_streak_recent
            self._cost_streak_recent.append(current_cost)
            if len(self._cost_streak_recent) > 5:
                self._cost_streak_recent.pop(0)

            # 计算最近 5 步 cost_sum
            cost_sum_recent = sum(self._cost_streak_recent)

            # 检查熔断条件
            if self._disable_real_intervention_for_episode:
                # 已经触发熔断，只记录，不做真实接管
                pass
            elif cost_sum_recent >= 3.0:
                # 最近 5 步 cost_sum >= 3，触发熔断
                self._disable_real_intervention_for_episode = True
                self._cost_streak_guard_triggered = True
                self._cost_streak_guard_reason = "cost_streak_guard"
                if self.config.debug:
                    print(f"[CostStreakGuard] Triggered: cost_sum_recent={cost_sum_recent:.2f}, recent_costs={self._cost_streak_recent}")

            # 记录到 info
            info["cost_streak_sum_recent_5"] = cost_sum_recent
            info["disable_real_intervention"] = self._disable_real_intervention_for_episode
            info["cost_streak_guard_triggered"] = self._cost_streak_guard_triggered
            info["cost_streak_guard_reason"] = self._cost_streak_guard_reason

            # ===== 步骤0：超轻量门控（最优先，最便宜） =====
            gate_reason = self._ultra_light_gate(env, vehicle, ego_pos, ego_heading, ego_speed, raw_action_np)
            info["ultra_light_gate_reason"] = gate_reason
            info["ultra_light_gate_passed"] = True

            # 车辆风险标志 - 立即最小风险停车
            if gate_reason == "vehicle_risk_flag":
                # 检查是否已经 contact/crash
                contact_results = getattr(vehicle, "contact_results", None)
                contact_state = False
                contact_info = {}
                if contact_results is not None:
                    contact_str = str(contact_results).lower()
                    dangerous_types = ["vehicle", "traffic_object", "traffic_cone", "traffic_barrier", "crash"]
                    for dtype in dangerous_types:
                        if dtype in contact_str:
                            contact_state = True
                            contact_info["contact_state"] = True
                            contact_info["contact_type"] = dtype
                            break

                # 如果已经 contact/crash，使用 contact_state_handler
                if contact_state:
                    safe_action = self._minimum_risk_stop(vehicle, contact_state=True, contact_info=contact_info)
                    recovery_mode = "contact_state_handler"
                    selected_candidate_type = "contact_state_handler_stop"
                    fallback_reason = "contact_state_detected"
                else:
                    safe_action = self._minimum_risk_stop(vehicle, contact_state=False)
                    recovery_mode = "minimum_risk_stop"
                    selected_candidate_type = "minimum_risk_stop_vehicle_risk"
                    fallback_reason = "vehicle_risk_before_route_build"

                # 判断 actual_action_modified：检查 safe_action 与 raw_action 的差异
                action_modified = (
                    abs(safe_action[0] - raw_action_np[0]) > 1e-3
                    or abs(safe_action[1] - raw_action_np[1]) > 1e-3
                )
                info.update({
                    "recovery_mode": recovery_mode,
                    "recovery_certified": False,
                    "accepted_by_filter": True,
                    "model_predicted_safe": False,
                    "selected_candidate_type": selected_candidate_type,
                    "control_feasible": True,
                    "fallback_reason": fallback_reason,
                    "route_cache_hit": False,
                    "ultra_light_gate_passed": False,
                    "intervention_reason": "vehicle_risk_flag",
                    "filter_intervened": action_modified,
                    "actual_action_modified": action_modified,
                    "contact_state_detected": contact_state,
                })
                self._last_recovery_active = True
                self._intervention_count_this_ep += 1
                self._last_intervention_step = self._intervention_count_this_ep
                self._cooldown_remaining = self.config.intervention_cooldown_steps
                elapsed_ms = (time.time() - start) * 1000.0
                info["filter_time_ms"] = elapsed_ms
                if elapsed_ms > self.config.filter_time_warn_ms:
                    info["filter_time_warning"] = True
                return safe_action, info

            # 低风险场景 - 直接返回 raw_action，不构建道路、不解析场景、不生成候选
            # 但在此之前，先做 ultra-short-horizon emergency check（必须在 low_risk_passthrough 之前）
            if gate_reason == "low_risk_passthrough":
                emergency_check_result = self._ultra_short_horizon_emergency_check(
                    env, vehicle, ego_pos, ego_heading, ego_speed, raw_action_np
                )
                if emergency_check_result["emergency_detected"]:
                    # 紧急检查发现问题，返回 emergency_stop
                    safe_action = np.array([0.0, -0.8], dtype=np.float32)  # 紧急刹车
                    action_modified = (
                        abs(safe_action[0] - raw_action_np[0]) > 1e-3
                        or abs(safe_action[1] - raw_action_np[1]) > 1e-3
                    )
                    info.update({
                        "recovery_mode": "ultra_short_horizon_emergency",
                        "recovery_certified": False,
                        "accepted_by_filter": True,
                        "model_predicted_safe": False,
                        "selected_candidate_type": "ultra_short_horizon_emergency_stop",
                        "collision_free": True,
                        "boundary_safe": True,
                        "control_feasible": True,
                        "fallback_reason": emergency_check_result["emergency_reason"],
                        "route_cache_hit": False,
                        "ultra_light_gate_passed": True,
                        "allow_intervention": False,
                        "intervention_reason": emergency_check_result["emergency_reason"],
                        "filter_intervened": action_modified,
                        "actual_action_modified": action_modified,
                    })
                    # 记录 emergency check 详情
                    for key in ["emergency_detected", "emergency_reason", "nearest_vehicle_distance",
                                "nearest_static_object_distance", "contact_state_detected", "front_blocking_distance"]:
                        if key in emergency_check_result:
                            info[key] = emergency_check_result[key]
                    self._last_recovery_active = True
                    self._intervention_count_this_ep += 1
                    self._last_intervention_step = self._intervention_count_this_ep
                    self._cooldown_remaining = self.config.intervention_cooldown_steps
                    elapsed_ms = (time.time() - start) * 1000.0
                    info["filter_time_ms"] = elapsed_ms
                    return safe_action, info

                safe_action = raw_action_np.astype(np.float32)
                info.update({
                    "recovery_mode": "inactive_raw_action",
                    "recovery_certified": False,
                    "accepted_by_filter": False,
                    "model_predicted_safe": False,
                    "selected_candidate_type": "raw_action",
                    "collision_free": True,
                    "boundary_safe": True,
                    "control_feasible": True,
                    "fallback_reason": "low_risk_passthrough",
                    "route_cache_hit": False,
                    "ultra_light_gate_passed": True,
                    "allow_intervention": False,
                    "intervention_reason": "none",
                    "filter_intervened": False,
                    "actual_action_modified": False,
                })
                self._last_recovery_active = False
                self._reset_prev_safe()
                elapsed_ms = (time.time() - start) * 1000.0
                info["filter_time_ms"] = elapsed_ms
                return safe_action, info

            # ===== 更新 cooldown ===
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1

            # ===== 检查是否在 cooldown 或 temporary passthrough ===
            if self._cooldown_remaining > 0 or self._temporary_passthrough:
                safe_action = raw_action_np.astype(np.float32)
                info.update({
                    "recovery_mode": "inactive_raw_action",
                    "recovery_certified": False,
                    "accepted_by_filter": False,
                    "model_predicted_safe": False,
                    "selected_candidate_type": "raw_action",
                    "collision_free": True,
                    "boundary_safe": True,
                    "control_feasible": True,
                    "fallback_reason": "cooldown_or_passthrough",
                    "route_cache_hit": False,
                    "ultra_light_gate_passed": True,
                    "allow_intervention": False,
                    "intervention_reason": "cooldown" if self._cooldown_remaining > 0 else "temporary_passthrough",
                    "filter_intervened": False,
                    "actual_action_modified": False,
                })
                self._last_recovery_active = False
                self._reset_prev_safe()
                elapsed_ms = (time.time() - start) * 1000.0
                info["filter_time_ms"] = elapsed_ms
                return safe_action, info

            # ===== 步骤1：构建道路坐标（仅当需要时） =====
            frame = self._get_route_frame(env, vehicle, ego_pos)
            route_build_time_ms = self._last_route_build_time_ms
            ego_projection = frame.project_point(ego_pos, ego_heading) if ego_pos is not None else FrenetProjection(False, reason="missing_ego_position")

            # ===== 步骤2：解析场景物体 =====
            scene_start = time.time()
            objects, scene_info = self._parse_scene(env, frame, ego_projection, ego_speed, ego_width)
            scene_parse_time_ms = (time.time() - scene_start) * 1000.0
            info.update(scene_info)
            blocking_distance = _safe_float(scene_info.get("front_blocking_object_distance"), float("inf"))
            has_front_blocker = np.isfinite(blocking_distance)

            # 无前方阻塞物时 - 返回 raw_action（除非有明确风险信号）
            if not has_front_blocker:
                if gate_reason == "raw_action_brake" or gate_reason == "low_speed_low_progress":
                    # 明确的减速/卡死风险，允许候选替换
                    pass
                else:
                    safe_action = raw_action_np.astype(np.float32)
                    info.update({
                        "recovery_mode": "inactive_raw_action",
                        "recovery_certified": False,
                        "accepted_by_filter": False,
                        "model_predicted_safe": False,
                        "selected_candidate_type": "raw_action_no_front_blocker",
                        "fallback_reason": "no_front_blocking_object",
                        "route_cache_hit": bool(self._last_route_cache_hit),
                        "route_build_time_ms": route_build_time_ms,
                        "scene_parse_time_ms": scene_parse_time_ms,
                        "allow_intervention": False,
                        "intervention_reason": "no_front_blocker",
                        "filter_intervened": False,
                        "actual_action_modified": False,
                    })
                    self._last_recovery_active = False
                    self._reset_prev_safe()
                    elapsed_ms = (time.time() - start) * 1000.0
                    info["filter_time_ms"] = elapsed_ms
                    return np.asarray(safe_action, dtype=np.float32), info

            # ===== 步骤3：生成候选规范 =====
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
            info["num_candidate_specs"] = len(candidate_specs)

            # ===== 步骤4：对 raw_action 做对照预测 =====
            raw_eval = self._evaluate_raw_action(
                frame,
                ego_projection,
                ego_pos,
                ego_heading,
                ego_speed,
                ego_length,
                ego_width,
                raw_action_np,
                scene_info,
            )
            info.update({
                "raw_predicted_collision": raw_eval.get("predicted_collision", False),
                "raw_predicted_out_of_road": raw_eval.get("predicted_out_of_road", False),
                "raw_predicted_cost_risk": raw_eval.get("cost_risk", 0.0),
                "raw_min_vehicle_margin": raw_eval.get("min_vehicle_margin", float("inf")),
                "raw_min_static_margin": raw_eval.get("min_static_margin", float("inf")),
                "raw_min_boundary_margin": raw_eval.get("min_boundary_margin", float("inf")),
                "raw_deadlock_risk": raw_eval.get("deadlock_risk", 0.0),
                "raw_total_score": raw_eval.get("total_score", float("inf")),
            })

            # 如果 raw_action 预测安全，直接返回
            if raw_eval.get("predicted_collision", False) == False and raw_eval.get("predicted_out_of_road", False) == False:
                if raw_eval.get("cost_risk", 0.0) < 1.0 and raw_eval.get("deadlock_risk", 0.0) < 2.0:
                    # 在返回 raw_action_safe 之前，再次做 ultra-short-horizon emergency check
                    emergency_check = self._ultra_short_horizon_emergency_check(
                        env, vehicle, ego_pos, ego_heading, ego_speed, raw_action_np
                    )
                    if emergency_check["emergency_detected"]:
                        # 紧急检查发现问题，返回 emergency_stop
                        safe_action = np.array([0.0, -0.8], dtype=np.float32)
                        action_modified = (
                            abs(safe_action[0] - raw_action_np[0]) > 1e-3
                            or abs(safe_action[1] - raw_action_np[1]) > 1e-3
                        )
                        info.update({
                            "recovery_mode": "ultra_short_horizon_emergency",
                            "recovery_certified": False,
                            "accepted_by_filter": True,
                            "model_predicted_safe": False,
                            "selected_candidate_type": "ultra_short_horizon_emergency_stop",
                            "collision_free": True,
                            "boundary_safe": True,
                            "control_feasible": True,
                            "fallback_reason": emergency_check["emergency_reason"],
                            "route_cache_hit": bool(self._last_route_cache_hit),
                            "route_build_time_ms": route_build_time_ms,
                            "scene_parse_time_ms": scene_parse_time_ms,
                            "allow_intervention": False,
                            "intervention_reason": emergency_check["emergency_reason"],
                            "filter_intervened": action_modified,
                            "actual_action_modified": action_modified,
                        })
                        # 记录 emergency check 详情
                        for key in ["emergency_detected", "emergency_reason", "nearest_vehicle_distance",
                                    "nearest_static_object_distance", "contact_state_detected", "front_blocking_distance"]:
                            if key in emergency_check:
                                info[key] = emergency_check[key]
                        self._last_recovery_active = True
                        self._intervention_count_this_ep += 1
                        self._last_intervention_step = self._intervention_count_this_ep
                        self._cooldown_remaining = self.config.intervention_cooldown_steps
                        elapsed_ms = (time.time() - start) * 1000.0
                        info["filter_time_ms"] = elapsed_ms
                        return safe_action, info

                    safe_action = raw_action_np.astype(np.float32)
                    info.update({
                        "recovery_mode": "inactive_raw_action",
                        "recovery_certified": False,
                        "accepted_by_filter": False,
                        "model_predicted_safe": True,
                        "selected_candidate_type": "raw_action_safe",
                        "collision_free": True,
                        "boundary_safe": True,
                        "control_feasible": True,
                        "fallback_reason": "raw_action_predicted_safe",
                        "route_cache_hit": bool(self._last_route_cache_hit),
                        "route_build_time_ms": route_build_time_ms,
                        "scene_parse_time_ms": scene_parse_time_ms,
                        "allow_intervention": False,
                        "intervention_reason": "raw_action_safe",
                        "filter_intervened": False,
                        "actual_action_modified": False,
                    })
                    self._last_recovery_active = False
                    self._reset_prev_safe()
                    elapsed_ms = (time.time() - start) * 1000.0
                    info["filter_time_ms"] = elapsed_ms
                    return safe_action, info

            # ===== 步骤5：Rollout 候选并边展开边淘汰 =====
            safe_rollouts: List[Tuple[TrajectoryScore, TrajectoryRollout]] = []
            collision_hard_reject_count = 0
            severe_lateral_rss_reject_count = 0
            cost_risk_hard_reject_count = 0
            vehicle_margin_min = float("inf")
            static_margin_min = float("inf")
            boundary_margin_min = float("inf")
            early_rejected = 0
            early_reject_reasons: Dict[str, int] = {}

            candidate_start = time.time()
            for candidate in candidate_specs:
                rollout, reject_reason = self._rollout_candidate_with_hard_check(
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
                    raw_eval,
                )
                if rollout.hard_safe:
                    score = self._score_rollout(rollout, frame, objects, ego_projection, ego_length, ego_width, raw_action_np, scene_info)
                    safe_rollouts.append((score, rollout))
                    vehicle_margin_min = min(vehicle_margin_min, rollout.min_vehicle_margin)
                    static_margin_min = min(static_margin_min, rollout.min_static_margin)
                    boundary_margin_min = min(boundary_margin_min, rollout.min_boundary_margin)
                else:
                    early_rejected += 1
                    reason = reject_reason or rollout.failure_reason or "unknown"
                    early_reject_reasons[reason] = early_reject_reasons.get(reason, 0) + 1
                    if "collision" in reason or "vehicle_margin" in reason or "static_margin" in reason:
                        collision_hard_reject_count += 1
                    if "severe_lateral_rss" in reason or "lateral_rss" in reason:
                        severe_lateral_rss_reject_count += 1
                    if "cost_risk" in reason:
                        cost_risk_hard_reject_count += 1

            candidate_eval_time_ms = (time.time() - candidate_start) * 1000.0

            info.update({
                "num_candidates": len(candidate_specs),
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
                "vehicle_margin_min": vehicle_margin_min,
                "static_margin_min": static_margin_min,
                "boundary_margin_min": boundary_margin_min,
                "collision_hard_reject_count": collision_hard_reject_count,
                "severe_lateral_rss_reject_count": severe_lateral_rss_reject_count,
                "cost_risk_hard_reject_count": cost_risk_hard_reject_count,
            })

            # ===== 步骤6：判断是否接管（关键！） =====
            allow_intervention = False
            intervention_reason = "none"
            intervention_rejected_reason = ""
            best_score: Optional[TrajectoryScore] = None
            best_rollout: Optional[TrajectoryRollout] = None

            if safe_rollouts:
                best_score, best_rollout = min(safe_rollouts, key=lambda item: item[0].total_score)

                # 检查接管条件
                raw_has_clear_risk = (
                    raw_eval.get("predicted_collision", False) == True
                    or raw_eval.get("predicted_out_of_road", False) == True
                    or raw_eval.get("cost_risk", 0.0) > 2.0
                    or (gate_reason == "low_speed_low_progress" and raw_eval.get("deadlock_risk", 0.0) > 3.0)
                    or (gate_reason == "raw_action_brake" and raw_eval.get("deadlock_risk", 0.0) > 2.0)
                )

                # 检查 candidate 是否真的比 raw_action 更好
                candidate_better = (
                    best_score.hard_safe
                    and (not raw_eval.get("predicted_collision", False) or best_rollout.predicted_collision == False)
                    and (not raw_eval.get("predicted_out_of_road", False) or best_rollout.predicted_out_of_road == False)
                    and (best_rollout.predicted_cost_risk <= raw_eval.get("cost_risk", 0.0) + 0.5)
                    and (best_rollout.min_vehicle_margin >= raw_eval.get("min_vehicle_margin", float("inf")) - 0.20)
                    and (best_rollout.min_boundary_margin >= 0.60)
                    and (best_rollout.min_boundary_margin >= raw_eval.get("min_boundary_margin", float("inf")) - 0.10)
                )

                # 检查接管次数限制
                if self._intervention_count_this_ep >= self.config.max_interventions_per_episode:
                    candidate_better = False
                    intervention_rejected_reason = "max_interventions_reached"

                score_gain = raw_eval.get("total_score", float("inf")) - best_score.total_score

                if raw_has_clear_risk and candidate_better:
                    if score_gain >= self.config.intervention_score_margin:
                        # cost streak 熔断检查：如果已触发熔断，禁止真实接管
                        if self._disable_real_intervention_for_episode:
                            allow_intervention = False
                            intervention_rejected_reason = "cost_streak_guard_disabled"
                            intervention_reason = "cost_streak_guard"
                        else:
                            allow_intervention = True
                            if raw_eval.get("predicted_collision", False):
                                intervention_reason = "raw_collision_risk"
                            elif raw_eval.get("predicted_out_of_road", False):
                                intervention_reason = "raw_boundary_risk"
                            elif raw_eval.get("cost_risk", 0.0) > 2.0:
                                intervention_reason = "raw_cost_risk"
                            elif raw_eval.get("deadlock_risk", 0.0) > 3.0:
                                intervention_reason = "raw_deadlock_risk"
                            else:
                                intervention_reason = "raw_clear_risk"
                    else:
                        intervention_reason = "candidate_not_significantly_better"
                        intervention_rejected_reason = "insufficient_score_gain"
                else:
                    if not candidate_better:
                        if best_rollout.min_vehicle_margin < raw_eval.get("min_vehicle_margin", float("inf")) - 0.20:
                            intervention_rejected_reason = "vehicle_margin_worse"
                        elif best_rollout.min_boundary_margin < 0.60:
                            intervention_rejected_reason = "boundary_margin_insufficient"
                        elif best_rollout.predicted_cost_risk > raw_eval.get("cost_risk", 0.0):
                            intervention_rejected_reason = "cost_risk_increased"
                        else:
                            intervention_rejected_reason = "candidate_not_better"
                    intervention_reason = "raw_action_safe_or_candidate_not_better"

                info["allow_intervention"] = allow_intervention
                info["intervention_reason"] = intervention_reason
                info["intervention_rejected_reason"] = intervention_rejected_reason
                info["intervention_score_gain"] = score_gain

                # 记录候选对比
                info["candidate_predicted_collision"] = best_rollout.predicted_collision if best_rollout else False
                info["candidate_predicted_out_of_road"] = best_rollout.predicted_out_of_road if best_rollout else False
                info["candidate_predicted_cost_risk"] = best_rollout.predicted_cost_risk if best_rollout else 0.0
                info["candidate_min_vehicle_margin"] = best_rollout.min_vehicle_margin if best_rollout else float("inf")
                info["candidate_min_static_margin"] = best_rollout.min_static_margin if best_rollout else float("inf")
                info["candidate_min_boundary_margin"] = best_rollout.min_boundary_margin if best_rollout else float("inf")
                info["cost_risk_improved"] = (best_rollout.predicted_cost_risk < raw_eval.get("cost_risk", 0.0)) if best_rollout else False
                info["collision_risk_improved"] = ((not raw_eval.get("predicted_collision", False)) or (best_rollout and not best_rollout.predicted_collision)) if best_rollout else False
                info["boundary_risk_improved"] = ((not raw_eval.get("predicted_out_of_road", False)) or (best_rollout and not best_rollout.predicted_out_of_road)) if best_rollout else False
                info["vehicle_margin_worse"] = (best_rollout.min_vehicle_margin < raw_eval.get("min_vehicle_margin", float("inf")) - 0.20) if best_rollout else False
            else:
                info["allow_intervention"] = False
                info["intervention_reason"] = "no_safe_candidate"
                info["intervention_rejected_reason"] = "no_hard_safe_candidate"
                info["intervention_score_gain"] = 0.0

            # ===== 步骤7：调试影子模式 =====
            if self.config._debug_shadow_record:
                if allow_intervention and best_rollout is not None:
                    info["filter_would_intervene"] = True
                    info["would_selected_candidate_type"] = best_rollout.candidate.candidate_type
                    info["would_safe_steer"] = float(best_rollout.steer_actions[0])
                    info["would_safe_acc"] = float(best_rollout.throttle_actions[0])
                else:
                    info["filter_would_intervene"] = False
                # 调试模式下实际执行 raw_action
                safe_action = raw_action_np.astype(np.float32)
                info["filter_intervened"] = False
                info["actual_action_modified"] = False
                info["recovery_mode"] = "inactive_raw_action"
            elif allow_intervention and best_rollout is not None:
                # 实际接管
                self._intervention_count_this_ep += 1
                self._last_intervention_step = self._intervention_count_this_ep
                self._cooldown_remaining = self.config.intervention_cooldown_steps

                candidate_action = np.array(
                    [best_rollout.steer_actions[0], best_rollout.throttle_actions[0]],
                    dtype=np.float32,
                )
                # 限幅：safe_action 不能与 raw_action 差太多
                safe_action, action_info = self._clip_action_to_raw_delta(
                    candidate_action, raw_action_np, self._prev_safe_steer, self._prev_safe_acc
                )
                info.update(action_info)

                # 限幅后再次检查
                if not self._is_action_safe_after_clip(safe_action, raw_action_np, best_score, best_rollout, raw_eval):
                    safe_action = raw_action_np.astype(np.float32)
                    info["intervention_reason"] = "action_clipped_unsafe"
                    allow_intervention = False
                    self._intervention_count_this_ep -= 1
                    self._cooldown_remaining = 0
                else:
                    self._prev_safe_steer = safe_action[0]
                    self._prev_safe_acc = safe_action[1]

                info.update(self._score_to_info(best_score))
                info.update({
                    "recovery_mode": "predictive_recovery",
                    "recovery_certified": self._is_conservatively_certified(best_score, best_rollout, ego_projection, action_info, raw_eval),
                    "accepted_by_filter": allow_intervention,
                    "model_predicted_safe": True,
                    "selected_candidate_type": best_rollout.candidate.candidate_type,
                    "selected_lateral_target": best_rollout.candidate.lateral_target,
                    "selected_speed_target": best_rollout.candidate.speed_target,
                    "collision_free": True,
                    "boundary_safe": True,
                    "control_feasible": True,
                    "fallback_reason": info.get("fallback_reason", ""),
                    "selected_terminal_passed_blocker": best_score.terminal_passed_blocker,
                    "selected_terminal_recoverable": best_score.terminal_recoverable,
                })
                info["filter_intervened"] = allow_intervention
                # actual_action_modified 与 filter_intervened 保持一致（接管时 action 被修改）
                info["actual_action_modified"] = allow_intervention
                self._last_recovery_active = allow_intervention
            else:
                # 不接管，返回 raw_action
                safe_action = raw_action_np.astype(np.float32)
                info.update({
                    "recovery_mode": "inactive_raw_action",
                    "recovery_certified": False,
                    "accepted_by_filter": False,
                    "model_predicted_safe": raw_eval.get("predicted_collision", False) == False,
                    "selected_candidate_type": "raw_action",
                    "collision_free": raw_eval.get("predicted_collision", False) == False,
                    "boundary_safe": raw_eval.get("predicted_out_of_road", False) == False,
                    "control_feasible": True,
                    "fallback_reason": intervention_reason,
                })
                info["filter_intervened"] = False
                info["actual_action_modified"] = False
                self._last_recovery_active = False
                self._reset_prev_safe()

            info["would_selected_candidate_type"] = info.get("selected_candidate_type", "raw_action")
            info["would_safe_steer"] = float(safe_action[0])
            info["would_safe_acc"] = float(safe_action[1])

            # 记录状态
            self._last_vehicle_margin_min = vehicle_margin_min
            self._last_static_margin_min = static_margin_min

        except Exception as exc:
            vehicle = _get_ego_vehicle(env)
            # 检查 contact state
            contact_results = getattr(vehicle, "contact_results", None)
            contact_state = False
            if contact_results is not None:
                contact_str = str(contact_results).lower()
                dangerous_types = ["vehicle", "traffic_object", "traffic_cone", "traffic_barrier", "crash"]
                for dtype in dangerous_types:
                    if dtype in contact_str:
                        contact_state = True
                        break

            safe_action = self._minimum_risk_stop(vehicle, contact_state=contact_state)
            # 判断 actual_action_modified
            action_modified = (
                abs(safe_action[0] - raw_action_np[0]) > 1e-3
                or abs(safe_action[1] - raw_action_np[1]) > 1e-3
            )
            info.update({
                "recovery_mode": "exception_fallback",
                "recovery_certified": False,
                "accepted_by_filter": True,
                "model_predicted_safe": False,
                "selected_candidate_type": "exception_fallback",
                "fallback_reason": "exception:{}".format(type(exc).__name__),
                "control_feasible": True,
                "allow_intervention": False,
                "intervention_reason": "exception",
                "filter_intervened": True,
                "actual_action_modified": action_modified,
                "contact_state_detected": contact_state,
            })
            self._last_recovery_active = True
            self._intervention_count_this_ep += 1
            self._cooldown_remaining = self.config.intervention_cooldown_steps
            if self.config.debug:
                info["debug_exception"] = str(exc)

        elapsed_ms = (time.time() - start) * 1000.0
        info["filter_time_ms"] = elapsed_ms
        if elapsed_ms > self.config.filter_time_warn_ms:
            info["filter_time_warning"] = True
        return np.asarray(safe_action, dtype=np.float32), info

    def _empty_info(self) -> Dict[str, Any]:
        info = {field_name: 0 for field_name in RECOVERY_DIAGNOSTIC_FIELDS}
        info.update({
            "recovery_mode": "inactive",
            "recovery_certified": False,
            "accepted_by_filter": False,
            "filter_intervened": False,
            "actual_action_modified": False,
            "model_predicted_safe": False,
            "selected_candidate_type": "",
            "fallback_reason": "",
            "front_blocking_object_type": "",
            "front_blocking_object_distance": float("inf"),
            "rss_longitudinal_margin": float("inf"),
            "rss_lateral_margin": float("inf"),
            "rss_risk_score": 0.0,
            "left_space_available": 0.0,
            "right_space_available": 0.0,
            "ultra_light_gate_reason": "",
            "ultra_light_gate_passed": False,
            "early_reject_reasons": "",
            "route_cache_hit": False,
            "selected_terminal_passed_blocker": False,
            "selected_terminal_recoverable": False,
            "blocker_left_gap": 0.0,
            "blocker_right_gap": 0.0,
            # Raw action 对照
            "raw_predicted_collision": False,
            "raw_predicted_out_of_road": False,
            "raw_predicted_cost_risk": 0.0,
            "raw_min_vehicle_margin": float("inf"),
            "raw_min_static_margin": float("inf"),
            "raw_min_boundary_margin": float("inf"),
            "raw_deadlock_risk": 0.0,
            "raw_total_score": float("inf"),
            # 接管判断
            "allow_intervention": False,
            "intervention_reason": "none",
            "intervention_rejected_reason": "",
            "intervention_score_gain": 0.0,
            "filter_would_intervene": False,
            # 候选对比
            "candidate_predicted_collision": False,
            "candidate_predicted_out_of_road": False,
            "candidate_predicted_cost_risk": 0.0,
            "candidate_min_vehicle_margin": float("inf"),
            "candidate_min_static_margin": float("inf"),
            "candidate_min_boundary_margin": float("inf"),
            "cost_risk_improved": False,
            "collision_risk_improved": False,
            "boundary_risk_improved": False,
            "vehicle_margin_worse": False,
            # Action 限幅
            "action_limited_by_raw_delta": False,
            "safe_raw_steer_delta": 0.0,
            "safe_raw_acc_delta": 0.0,
            # 候选评估
            "vehicle_margin_min": float("inf"),
            "static_margin_min": float("inf"),
            "boundary_margin_min": float("inf"),
            "collision_hard_reject_count": 0,
            "severe_lateral_rss_reject_count": 0,
            "cost_risk_hard_reject_count": 0,
            # Episode 统计
            "intervention_count_this_ep": 0,
            "cooldown_remaining": 0,
            "temporary_passthrough": False,
            # Cost streak 熔断
            "cost_streak_sum_recent_5": 0.0,
            "disable_real_intervention": False,
            "cost_streak_guard_triggered": False,
            "cost_streak_guard_reason": "",
            # Emergency check 详情
            "emergency_detected": False,
            "emergency_reason": "",
            "nearest_vehicle_distance": float("inf"),
            "nearest_static_object_distance": float("inf"),
            "contact_state_detected": False,
            "front_blocking_distance": float("inf"),
            # 调试选项
            "debug_shadow_record": False,
            "would_selected_candidate_type": "raw_action",
            "would_safe_steer": 0.0,
            "would_safe_acc": 0.0,
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

    def _ultra_light_gate(
        self,
        env: Any,
        vehicle: Any,
        ego_pos: Optional[np.ndarray],
        ego_heading: float,
        ego_speed: float,
        raw_action: np.ndarray,
    ) -> str:
        """超轻量门控 - 只使用最便宜的信息判断是否需要接管。"""
        progress = 0.0
        if ego_pos is not None and self._last_ego_pos is not None:
            progress = float(np.linalg.norm(ego_pos - self._last_ego_pos))
        if ego_pos is not None:
            self._last_ego_pos = ego_pos.copy()

        if ego_speed < self.config.ultra_low_speed_threshold and progress < self.config.ultra_low_progress_threshold:
            self._low_progress_count += 1
        else:
            self._low_progress_count = 0

        # 检查车辆风险标志
        if self._vehicle_risk_flag(vehicle):
            return "vehicle_risk_flag"

        # 检查粗略前方阻塞物
        if self._coarse_front_blocker(env, vehicle, ego_pos, ego_heading):
            return "coarse_front_blocker"

        # 检查 raw_action 是否在急刹
        if raw_action[1] <= self.config.ultra_brake_threshold:
            return "raw_action_brake"

        # 检查 raw_action 是否低油门且低速
        if raw_action[1] <= self.config.ultra_low_throttle_threshold and ego_speed < 3.0:
            return "raw_action_low_throttle"

        # 检查是否连续低速且进展小
        if self._low_progress_count >= self.config.ultra_low_progress_steps:
            return "low_speed_low_progress"

        # 检查上一时刻是否处于恢复状态
        if self._last_recovery_active:
            return "previous_recovery"

        # 默认低风险，直接返回 raw_action
        return "low_risk_passthrough"

    def _vehicle_risk_flag(self, vehicle: Any) -> bool:
        if vehicle is None:
            return False
        for attr in ("crash_vehicle", "crash_object", "crash_sidewalk", "crash_building", "out_of_route", "out_of_road"):
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
        """
        增强的 coarse_front_blocker：检测 vehicle 和 traffic object / cone / barrier / static object。
        """
        if ego_pos is None:
            return False
        forward = _unit_from_heading(ego_heading)
        left = _left_normal_from_heading(ego_heading)
        ego_speed = _get_speed(vehicle, 0.0)
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
            obj_length, obj_width = _get_size(obj)

            class_name = obj.__class__.__name__.lower()
            is_vehicle_obj = "vehicle" in class_name or hasattr(obj, "throttle_brake") or hasattr(obj, "speed")
            is_traffic_object = any(t in class_name for t in ["cone", "barrier", "traffic", "sign", "light"])
            speed = _get_speed(obj, 0.0)

            # 静态物体或交通物体
            if not is_vehicle_obj and (is_traffic_object or speed < 0.5):
                # 静态物体：即使不在正前方车道，也在考虑范围内
                # 如果在前方近距离（< 15m）且横向重叠较大
                if long < 15.0 and lat < ego_width + obj_width:
                    return True
            else:
                # 车辆：检查低速阻塞
                if lat <= self.config.ultra_front_lateral_window + obj_width * 0.5:
                    if speed <= max(self.config.blocking_speed_threshold, ego_speed * self.config.low_speed_ratio):
                        return True
        return False

    def _ultra_short_horizon_emergency_check(
        self,
        env: Any,
        vehicle: Any,
        ego_pos: Optional[np.ndarray],
        ego_heading: float,
        ego_speed: float,
        raw_action: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Ultra-short-horizon emergency check：在 low_risk_passthrough 之前执行。
        检查未来 1 到 3 步 raw_action 是否会导致碰撞。

        检查：
        - 最近车辆距离
        - 最近 traffic object / cone / barrier 距离
        - ego footprint 与静态物体膨胀框
        - ego footprint 与 vehicle 膨胀框
        - contact_results 当前是否已经包含 VEHICLE / TRAFFIC_OBJECT / TRAFFIC_CONE / TRAFFIC_BARRIER
        - front_blocking_object_distance 是否过小
        - 速度较高且物体在前方近距离时，不允许 raw_action_safe
        """
        result = {
            "emergency_detected": False,
            "emergency_reason": "",
            "nearest_vehicle_distance": float("inf"),
            "nearest_static_object_distance": float("inf"),
            "contact_state_detected": False,
            "front_blocking_distance": float("inf"),
        }

        if ego_pos is None:
            return result

        # 检查当前 contact_results
        contact_results = getattr(vehicle, "contact_results", None)
        if contact_results is not None:
            contact_str = str(contact_results).lower()
            dangerous_types = ["vehicle", "traffic_object", "traffic_cone", "traffic_barrier", "crash"]
            for dtype in dangerous_types:
                if dtype in contact_str:
                    result["contact_state_detected"] = True
                    result["emergency_detected"] = True
                    result["emergency_reason"] = f"contact_{dtype}_detected"
                    return result

        # 检查 crash 标志
        for attr in ["crash", "crash_vehicle", "crash_object", "crash_sidewalk", "crash_building"]:
            try:
                if bool(getattr(vehicle, attr, False)):
                    result["emergency_detected"] = True
                    result["emergency_reason"] = f"crash_flag_{attr}"
                    return result
            except Exception:
                pass

        forward = _unit_from_heading(ego_heading)
        left_normal = _left_normal_from_heading(ego_heading)
        ego_length, ego_width = _get_size(vehicle)

        # 扫描所有物体
        nearest_vehicle_dist = float("inf")
        nearest_static_dist = float("inf")
        front_blocking_dist = float("inf")

        scanned = 0
        for obj in self._iter_environment_objects(env):
            if obj is None or obj is vehicle:
                continue
            scanned += 1
            if scanned > 15:  # 比 coarse_front_blocker 更全面
                break

            pos = _get_position(obj)
            if pos is None:
                continue

            delta = pos - ego_pos
            long = float(np.dot(delta, forward))  # 纵向距离
            lat = abs(float(np.dot(delta, left_normal)))  # 横向距离

            obj_length, obj_width = _get_size(obj)
            class_name = obj.__class__.__name__.lower()
            is_vehicle_obj = "vehicle" in class_name or hasattr(obj, "throttle_brake") or hasattr(obj, "speed")
            speed = _get_speed(obj, 0.0)

            # 膨胀后的 ego footprint
            inflated_ego_length = ego_length + 1.0
            inflated_ego_width = ego_width + 1.0

            # 检查物体类型
            is_traffic_object = any(t in class_name for t in ["cone", "barrier", "traffic", "object", "cone", "sign"])

            # 计算距离
            obj_dist = float(np.linalg.norm(delta))

            if is_vehicle_obj:
                nearest_vehicle_dist = min(nearest_vehicle_dist, obj_dist)
            elif is_traffic_object or speed < 0.5:
                nearest_static_dist = min(nearest_static_dist, obj_dist)

            # 前方近距离阻塞物检查
            if 0 < long < 20.0 and lat < (ego_width + obj_width) * 0.8:
                if long < front_blocking_dist:
                    front_blocking_dist = long

            # ===== 核心检查：检查未来 1-3 步 raw_action 是否会碰撞 =====

            # 预测未来位置（1-3 步）
            for step_ahead in [1, 2, 3]:
                dt = step_ahead * self.config.dt  # 假设 dt=0.2，则 1 步=0.2s, 2 步=0.4s, 3 步=0.6s

                # raw_action 预测
                raw_accel = self._action_to_accel(raw_action[1])
                future_speed = max(0.0, ego_speed + raw_accel * dt)

                if future_speed < 0.1 and ego_speed > 0.5:
                    # 急刹情况
                    future_ego_pos = ego_pos + forward * ego_speed * dt * 0.3  # 减速行驶
                else:
                    future_ego_pos = ego_pos + forward * future_speed * dt

                # 计算未来 ego footprint 中心
                future_center = future_ego_pos

                # 物体未来位置
                obj_direction = _unit_from_heading(_get_heading(obj, 0.0))
                future_obj_pos = pos + obj_direction * speed * dt

                # 检查两个矩形是否相交（膨胀后）
                inflated_obj_length = obj_length + 0.5
                inflated_obj_width = obj_width + 0.5

                if _rectangles_intersect(
                    future_center, ego_heading, inflated_ego_length, inflated_ego_width,
                    future_obj_pos, _get_heading(obj, 0.0), inflated_obj_length, inflated_obj_width
                ):
                    result["emergency_detected"] = True
                    if is_vehicle_obj:
                        result["emergency_reason"] = f"vehicle_collision_step_{step_ahead}"
                    else:
                        result["emergency_reason"] = f"static_collision_step_{step_ahead}"
                    return result

                # 额外检查：如果物体非常近且 ego 速度较高，不允许 raw_action
                if long > 0 and long < 8.0 and lat < ego_width * 0.5 + obj_width * 0.5 + 0.5:
                    if ego_speed > 3.0:
                        # 高速接近前方近距离物体
                        result["emergency_detected"] = True
                        result["emergency_reason"] = f"high_speed_approach_{is_vehicle_obj}"
                        return result

        result["nearest_vehicle_distance"] = nearest_vehicle_dist
        result["nearest_static_object_distance"] = nearest_static_dist
        result["front_blocking_distance"] = front_blocking_dist

        # 综合判断：如果前方有近距离物体且 ego 速度较高
        if np.isfinite(front_blocking_dist) and front_blocking_dist < 10.0 and ego_speed > 5.0:
            if nearest_vehicle_dist < 15.0 or nearest_static_dist < 10.0:
                result["emergency_detected"] = True
                result["emergency_reason"] = "high_speed_near_blocker"
                return result

        return result

    def _format_reject_reasons(self, reasons: Dict[str, int]) -> str:
        return ";".join("{}:{}".format(k, reasons[k]) for k in sorted(reasons))

    def _should_write_detailed_log(self, info: Dict[str, Any]) -> bool:
        if not self.config.log_csv_path:
            return False
        mode = info.get("recovery_mode", "")
        return bool(
            mode in ("predictive_recovery", "minimum_risk_stop", "exception_fallback")
            or info.get("filter_time_warning", False)
            or info.get("ultra_light_gate_reason", "") == "vehicle_risk_flag"
        )

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
        scan_limit = max(1, int(self.config.object_scan_limit))
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

        has_blocker = np.isfinite(_safe_float(scene_info.get("front_blocking_object_distance"), float("inf")))
        target_values = [current_l]
        if has_blocker and ego_projection.frenet_valid:
            target_values.append(0.0)
        n_lat = max(1, int(self.config.num_lateral_targets))

        if has_blocker:
            lateral_limit = self.config.max_lateral_offset if ego_projection.frenet_valid else self.config.max_lateral_when_frenet_unstable
            offsets = np.linspace(-lateral_limit, lateral_limit, num=n_lat)
            target_values.extend([current_l + float(offset) for offset in offsets])
        else:
            # 无阻塞物时只允许轻微修正
            target_values.extend([
                current_l - self.config.max_lateral_without_blocker,
                current_l + self.config.max_lateral_without_blocker,
            ])

        blocking_distance = _safe_float(scene_info.get("front_blocking_object_distance"), float("inf"))
        if np.isfinite(blocking_distance) and ego_projection.frenet_valid:
            blocking_l = _safe_float(scene_info.get("front_blocking_object_l"), current_l)
            blocking_width = _safe_float(scene_info.get("front_blocking_object_width"), ego_width)
            target_values.append(blocking_l + blocking_width * 0.5 + ego_width * 0.5 + 0.8)
            target_values.append(blocking_l - blocking_width * 0.5 - ego_width * 0.5 - 0.8)

        lateral_targets = []
        for value in target_values:
            clipped = _clip(value, l_low, l_high)
            if (
                not ego_projection.frenet_valid
                and abs(clipped - current_l) > self.config.max_lateral_when_frenet_unstable
            ):
                continue
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

    def _evaluate_raw_action(
        self,
        frame: RouteFrenetFrame,
        ego_projection: FrenetProjection,
        ego_pos: Optional[np.ndarray],
        ego_heading: float,
        ego_speed: float,
        ego_length: float,
        ego_width: float,
        raw_action: np.ndarray,
        scene_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """对 raw_action 做短时对照预测。"""
        result = {
            "predicted_collision": False,
            "predicted_out_of_road": False,
            "cost_risk": 0.0,
            "min_vehicle_margin": float("inf"),
            "min_static_margin": float("inf"),
            "min_boundary_margin": float("inf"),
            "deadlock_risk": 0.0,
            "total_score": 0.0,
        }
        if ego_pos is None:
            result["predicted_collision"] = True
            return result

        # 构建 raw_action 对应的候选
        raw_accel = self._action_to_accel(raw_action[1])
        raw_target_speed = max(0.0, ego_speed + raw_accel * self.config.horizon * 0.5)
        raw_lateral_target = ego_projection.l if ego_projection.frenet_valid else 0.0
        raw_candidate = TrajectoryCandidate(
            candidate_id=-1,
            candidate_type="raw_action",
            lateral_target=raw_lateral_target,
            speed_target=raw_target_speed,
            raw_action=raw_action.copy(),
        )

        # Rollout raw_action（只检查控制和边界，不检查障碍物）
        rollout, reject_reason = self._rollout_candidate_with_hard_check(
            raw_candidate,
            frame,
            [],  # 空障碍物列表
            ego_projection,
            ego_pos,
            ego_heading,
            ego_speed,
            ego_length,
            ego_width,
            self.config.default_max_steer_rad,
            {},  # 空 raw_eval
        )

        result["min_boundary_margin"] = rollout.min_boundary_margin
        result["cost_risk"] = rollout.predicted_cost_risk

        if not rollout.hard_safe:
            result["predicted_collision"] = True
            if "boundary" in reject_reason:
                result["predicted_out_of_road"] = True

        # 检查边界
        if ego_projection.frenet_valid:
            boundary = frame.boundary_at(ego_projection.s, ego_projection.l, ego_width, self.config.safety_margin)
            if not boundary.valid:
                result["predicted_out_of_road"] = True
                result["cost_risk"] = max(result["cost_risk"], 3.0)
            elif boundary.boundary_margin < self.config.boundary_hard_margin:
                result["cost_risk"] = max(result["cost_risk"], 2.0)

        # 检查死锁风险
        blocking_distance = _safe_float(scene_info.get("front_blocking_object_distance"), float("inf"))
        if np.isfinite(blocking_distance):
            if ego_speed < 1.5:
                result["deadlock_risk"] = 3.0
            if raw_target_speed < 2.0 and ego_speed < 2.0:
                result["deadlock_risk"] = max(result["deadlock_risk"], 4.0)

        # 简单评分
        result["total_score"] = (
            -self.config.w_progress * (raw_target_speed * self.config.horizon)
            + self.config.w_boundary * result["cost_risk"]
            + self.config.w_deadlock * result["deadlock_risk"]
        )

        return result

    def _rollout_candidate_with_hard_check(
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
        raw_eval: Optional[Dict[str, Any]] = None,
    ) -> Tuple[TrajectoryRollout, str]:
        """边展开边淘汰的候选 rollout，返回 (rollout, reject_reason)。"""
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
        min_vehicle_margin = float("inf")
        min_static_margin = float("inf")
        hard_safe = True
        failure_reason = ""
        predicted_collision = False
        predicted_out_of_road = False
        predicted_cost_risk = 0.0

        # 无阻塞物时，禁止大幅侧向候选
        has_blocking = any(obj.is_blocking for obj in objects)
        lateral_shift = abs(candidate.lateral_target - current_l)
        candidate_type = candidate.candidate_type
        if not has_blocking and objects:  # 有障碍物但无阻塞
            if "left_" in candidate_type or "right_" in candidate_type:
                if lateral_shift > self.config.max_lateral_without_blocker:
                    return TrajectoryRollout(
                        candidate=candidate,
                        times=np.zeros(0, dtype=float),
                        positions=np.zeros((0, 2), dtype=float),
                        headings=np.zeros(0, dtype=float),
                        speeds=np.zeros(0, dtype=float),
                        accelerations=np.zeros(0, dtype=float),
                        steer_actions=np.zeros(0, dtype=float),
                        throttle_actions=np.zeros(0, dtype=float),
                        frenet_s=np.zeros(0, dtype=float),
                        frenet_l=np.zeros(0, dtype=float),
                        hard_safe=False,
                        failure_reason="no_blocker_lateral_shift",
                        min_boundary_margin=float("inf"),
                        min_obstacle_margin=float("inf"),
                        min_vehicle_margin=float("inf"),
                        min_static_margin=float("inf"),
                        predicted_collision=False,
                        predicted_out_of_road=False,
                        predicted_cost_risk=0.0,
                    ), "no_blocker_lateral_shift"

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

            # 控制限制检查
            if speed < -1e-4:
                hard_safe = False
                failure_reason = "negative_speed"
                break
            if abs(steer_action) > self.config.max_steer_action + 1e-6 or abs(throttle_action) > 1.0 + 1e-6:
                hard_safe = False
                failure_reason = "control_limit"
                break

            # 边界检查
            boundary = frame.boundary_at(s_value, l_value, ego_width, self.config.safety_margin)
            min_boundary_margin = min(min_boundary_margin, boundary.boundary_margin)
            if not boundary.valid:
                hard_safe = False
                failure_reason = "boundary_invalid"
                predicted_out_of_road = True
                break
            if boundary.boundary_margin < self.config.boundary_hard_margin:
                hard_safe = False
                failure_reason = "boundary_margin"
                predicted_out_of_road = True
                break
            # 边界余量不足时禁止绕行
            if boundary.boundary_margin < self.config.min_boundary_margin_for_bypass:
                if "left_" in candidate_type or "right_" in candidate_type:
                    hard_safe = False
                    failure_reason = "boundary_margin_for_bypass"
                    break

            # cost 风险累积
            if boundary.boundary_margin < self.config.comfort_boundary_margin:
                predicted_cost_risk += (self.config.comfort_boundary_margin - boundary.boundary_margin) * 0.5

            # 障碍物检查
            for obj in objects:
                obj_pos = self._predict_object_position(obj, t)
                gap = self._center_gap(position, ego_length, ego_width, obj_pos, obj.length, obj.width)
                min_obstacle_margin = min(min_obstacle_margin, gap)

                inflated_ego_length = ego_length + 2.0 * self.config.hard_collision_margin
                inflated_ego_width = ego_width + 2.0 * self.config.hard_collision_margin

                if obj.is_vehicle:
                    inflated_obj_length = obj.length + 2.0 * self.config.hard_collision_margin
                    inflated_obj_width = obj.width + 2.0 * self.config.hard_collision_margin

                    if obj.frenet_valid:
                        rel_s = obj.s - s_value
                        rel_l = obj.l - l_value
                        lateral_gap = abs(rel_l) - inflated_ego_width * 0.5 - inflated_obj_width * 0.5

                        # 严重横向 RSS 检查
                        if obj.speed < 1.0:
                            if lateral_gap < self.config.severe_lateral_rss_margin:
                                hard_safe = False
                                failure_reason = "severe_lateral_rss"
                                predicted_collision = True
                                break

                        # 切入风险
                        if -4.0 < rel_s < 3.0 and lateral_gap < -0.1:
                            hard_safe = False
                            failure_reason = "cut_in_danger"
                            predicted_collision = True
                            break

                        # 车辆侧向硬距离
                        if abs(rel_s) < self.config.hard_vehicle_longitudinal_margin:
                            if lateral_gap < self.config.hard_vehicle_lateral_margin:
                                hard_safe = False
                                failure_reason = "vehicle_lateral_margin_violation"
                                predicted_collision = True
                                break

                        # 并排行驶时横向距离不足
                        if -1.0 < rel_s < 1.0 and lateral_gap < self.config.hard_vehicle_lateral_margin:
                            hard_safe = False
                            failure_reason = "parallel_lateral_violation"
                            predicted_collision = True
                            break

                    # 碰撞检查
                    if _rectangles_intersect(
                        position,
                        heading,
                        inflated_ego_length,
                        inflated_ego_width,
                        obj_pos,
                        obj.heading,
                        inflated_obj_length,
                        inflated_obj_width,
                    ):
                        hard_safe = False
                        failure_reason = "collision"
                        predicted_collision = True
                        break

                    min_vehicle_margin = min(min_vehicle_margin, gap)
                    # cost 风险
                    if gap < self.config.obstacle_margin:
                        predicted_cost_risk += (self.config.obstacle_margin - gap) * 0.3
                else:
                    inflated_obj_length = obj.length + 2.0 * self.config.hard_collision_margin
                    inflated_obj_width = obj.width + 2.0 * self.config.hard_collision_margin

                    if obj.frenet_valid:
                        rel_s = obj.s - s_value
                        rel_l = obj.l - l_value
                        if abs(rel_s) < self.config.hard_static_longitudinal_margin:
                            lateral_gap = abs(rel_l) - inflated_ego_width * 0.5 - inflated_obj_width * 0.5
                            if lateral_gap < self.config.hard_static_lateral_margin:
                                hard_safe = False
                                failure_reason = "static_margin_violation"
                                predicted_collision = True
                                break

                    if _rectangles_intersect(
                        position,
                        heading,
                        inflated_ego_length,
                        inflated_ego_width,
                        obj_pos,
                        obj.heading,
                        inflated_obj_length,
                        inflated_obj_width,
                    ):
                        hard_safe = False
                        failure_reason = "static_collision"
                        predicted_collision = True
                        break

                    min_static_margin = min(min_static_margin, gap)

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
            min_vehicle_margin=min_vehicle_margin,
            min_static_margin=min_static_margin,
            predicted_collision=predicted_collision,
            predicted_out_of_road=predicted_out_of_road,
            predicted_cost_risk=predicted_cost_risk,
        )
        return rollout, failure_reason

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
        progress = float(rollout.frenet_s[-1] - ego_projection.s) if len(rollout.frenet_s) > 0 else 0.0
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
        if np.isfinite(rollout.min_boundary_margin) and rollout.min_boundary_margin < self.config.boundary_comfort_margin:
            boundary_margin_score += (self.config.boundary_comfort_margin - rollout.min_boundary_margin) * 8.0
        if rollout.frenet_l.size > 1:
            lateral_velocity = np.diff(rollout.frenet_l) / np.maximum(np.diff(rollout.times), 1e-3)
            boundary_margin_score += max(0.0, float(np.max(np.abs(lateral_velocity))) - 1.5) * 0.1
        smoothness_score = float(np.mean(np.abs(np.diff(rollout.steer_actions)))) if rollout.steer_actions.size > 1 else 0.0
        smoothness_score += 0.2 * (float(np.mean(np.abs(np.diff(rollout.accelerations))))) if rollout.accelerations.size > 1 else 0.0
        nominal_deviation_score = float(np.linalg.norm(np.array([rollout.steer_actions[0], rollout.throttle_actions[0]]) - raw_action))
        continuity_score = 0.0

        terminal_recovery_score = 0.0
        terminal_passed_blocker = False
        terminal_recoverable = False
        if has_blocking:
            terminal_s = rollout.frenet_s[-1]
            terminal_l = rollout.frenet_l[-1]
            front_s = ego_projection.s + blocking_distance
            lateral_clearance_at_terminal = abs(terminal_l - blocker_l) - blocker_width * 0.5 - ego_width * 0.5
            terminal_passed_blocker = terminal_s > front_s + ego_length * 0.5
            terminal_recoverable = side_channel_available and (terminal_passed_blocker or (
                lateral_clearance_at_terminal > self.config.safety_margin + 0.25
                and terminal_s > front_s - ego_length
            ))
            if terminal_recoverable:
                terminal_recovery_score -= 12.0
                progress_score -= 0.5 * max(0.0, progress - blocking_distance)
            elif terminal_s < front_s or lateral_clearance_at_terminal <= self.config.safety_margin:
                terminal_recovery_score += 16.0 if side_channel_available else 8.0
                if rollout.speeds[-1] < 1.0 or avg_speed < 2.0:
                    deadlock_penalty += 8.0 if side_channel_available else 2.0

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

    def _is_conservatively_certified(
        self,
        score: TrajectoryScore,
        rollout: TrajectoryRollout,
        ego_projection: FrenetProjection,
        action_info: Optional[Dict[str, Any]] = None,
        raw_eval: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not ego_projection.frenet_valid or not rollout.hard_safe:
            return False
        if not (score.collision_free and score.boundary_safe and score.control_feasible):
            return False
        if rollout.min_boundary_margin < self.config.min_boundary_margin_for_bypass:
            return False
        if np.isfinite(rollout.min_vehicle_margin) and rollout.min_vehicle_margin < self.config.min_vehicle_margin_for_intervention:
            return False
        if action_info and action_info.get("action_limited_by_raw_delta", False):
            return False
        if raw_eval:
            if rollout.min_vehicle_margin < raw_eval.get("min_vehicle_margin", float("inf")) - 0.20:
                return False
            if rollout.predicted_cost_risk > raw_eval.get("cost_risk", 0.0):
                return False
        return True

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

    def _clip_action_to_raw_delta(
        self,
        candidate_action: np.ndarray,
        raw_action: np.ndarray,
        prev_steer: Optional[float],
        prev_acc: Optional[float],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """限幅 candidate_action，使其与 raw_action 和上一时刻动作的差值不超过限制。"""
        delta_steer = candidate_action[0] - raw_action[0]
        delta_acc = candidate_action[1] - raw_action[1]

        clipped_steer = _clip(delta_steer, -self.config.max_steer_delta_from_raw, self.config.max_steer_delta_from_raw)
        clipped_acc = _clip(delta_acc, -self.config.max_acc_delta_from_raw, self.config.max_acc_delta_from_raw)

        if prev_steer is not None:
            steer_diff = (raw_action[0] + clipped_steer) - prev_steer
            if abs(steer_diff) > self.config.max_steer_delta_from_prev:
                clipped_steer = _clip(clipped_steer, -self.config.max_steer_delta_from_prev - (raw_action[0] - prev_steer),
                                       self.config.max_steer_delta_from_prev - (raw_action[0] - prev_steer))

        if prev_acc is not None:
            acc_diff = (raw_action[1] + clipped_acc) - prev_acc
            if abs(acc_diff) > self.config.max_acc_delta_from_prev:
                clipped_acc = _clip(clipped_acc, -self.config.max_acc_delta_from_prev - (raw_action[1] - prev_acc),
                                    self.config.max_acc_delta_from_prev - (raw_action[1] - prev_acc))

        safe_action = np.array([raw_action[0] + clipped_steer, raw_action[1] + clipped_acc], dtype=np.float32)
        action_info = {
            "action_limited_by_raw_delta": (abs(clipped_steer - delta_steer) > 1e-4 or abs(clipped_acc - delta_acc) > 1e-4),
            "safe_raw_steer_delta": clipped_steer,
            "safe_raw_acc_delta": clipped_acc,
        }
        return safe_action, action_info

    def _is_action_safe_after_clip(
        self,
        safe_action: np.ndarray,
        raw_action: np.ndarray,
        score: Optional[TrajectoryScore],
        rollout: Optional[TrajectoryRollout],
        raw_eval: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if abs(safe_action[0] - raw_action[0]) > self.config.max_steer_delta_from_raw * 0.8:
            return False
        if abs(safe_action[1] - raw_action[1]) > self.config.max_acc_delta_from_raw * 0.8:
            return False
        if rollout is not None and not rollout.hard_safe:
            return False
        if raw_eval and rollout:
            if rollout.min_vehicle_margin < raw_eval.get("min_vehicle_margin", float("inf")) - 0.20:
                return False
        return True

    def _reset_prev_safe(self) -> None:
        self._prev_safe_steer = None
        self._prev_safe_acc = None

    def _minimum_risk_stop(self, vehicle: Any, contact_state: bool = False, contact_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        最小风险停车。
        如果 contact_state=True（已经 contact/crash），则进入 contact_state_handler：
        - 不再继续普通 minimum_risk_stop_vehicle_risk 反复刹停
        - 如果可以判断远离方向，输出远离物体的小 steer + brake
        - 否则只做保守停止
        """
        current_steer = _safe_float(getattr(vehicle, "steering", 0.0), 0.0)
        speed = _get_speed(vehicle, 0.0)
        ego_heading = _get_heading(vehicle, 0.0)

        if contact_state:
            # contact_state_handler：已经 contact/crash，不能再普通刹停
            # 检查 contact_info 中是否有远离方向
            if contact_info and "contact_direction" in contact_info:
                # 如果能判断远离方向，输出小 steer + brake
                contact_dir = contact_info["contact_direction"]
                # 将方向转换为 steer
                forward = _unit_from_heading(ego_heading)
                left_normal = _left_normal_from_heading(ego_heading)
                steer_from_dir = float(np.dot(contact_dir, left_normal)) * 0.5
                brake = -0.8 if speed > 0.5 else -0.5
                return np.array([_clip(steer_from_dir, -0.3, 0.3), brake], dtype=np.float32)

            # 否则保守停止，不顶住物体
            brake = -0.8 if speed > 0.5 else -0.4
            return np.array([_clip(current_steer * 0.2, -0.2, 0.2), brake], dtype=np.float32)
        else:
            # 普通最小风险停车
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

    def reset_episode(self):
        """重置 episode 状态。调用时在 episode 开始时。"""
        self._intervention_count_this_ep = 0
        self._last_intervention_step = 0
        self._cooldown_remaining = 0
        self._temporary_passthrough = False
        self._cost_history = []
        self._recent_interventions = []
        self._prev_safe_steer = None
        self._prev_safe_acc = None
        # === cost streak 熔断重置 ===
        self._cost_streak_recent = []
        self._disable_real_intervention_for_episode = False
        self._cost_streak_guard_triggered = False
        self._cost_streak_guard_reason = ""


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