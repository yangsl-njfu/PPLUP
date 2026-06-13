from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


RoadKey = Tuple[str, str]


@dataclass(frozen=True)
class RouteProjection:
    """A point projected to a continuous route coordinate."""

    route_s: float
    local_s: float
    local_d: float
    lane: object
    lane_index: Tuple[str, str, int]
    road_key: RoadKey


class MetaDriveRouteProjector:
    """
    Convert MetaDrive per-lane local coordinates into continuous route coordinates.

    MetaDrive lanes expose local coordinates as (longitudinal, lateral), but the
    longitudinal value resets to 0 at every lane/road segment. This helper builds
    a prefix-length table from a vehicle navigation route, so callers can compare
    positions across stitched road segments with a continuous route_s.
    """

    def __init__(self, navigation, lane_id: Optional[int] = None):
        if navigation is None:
            raise ValueError("navigation is required")
        if getattr(navigation, "checkpoints", None) is None:
            raise ValueError("navigation.checkpoints is required")
        if getattr(navigation, "map", None) is None:
            raise ValueError("navigation.map is required")

        self.navigation = navigation
        self.road_network = navigation.map.road_network
        self.checkpoints = list(navigation.checkpoints)
        self.lane_id = lane_id
        self.prefix_by_road = self._build_prefix()

    @classmethod
    def from_vehicle(cls, vehicle, lane_id: Optional[int] = None):
        """Build a projector from a MetaDrive vehicle's navigation module."""
        return cls(vehicle.navigation, lane_id=lane_id)

    def _build_prefix(self) -> Dict[RoadKey, float]:
        prefix_by_road = {}
        route_s = 0.0

        for start_node, end_node in self._road_pairs():
            lanes = self.road_network.graph[start_node][end_node]
            if len(lanes) == 0:
                continue

            prefix_by_road[(start_node, end_node)] = route_s
            route_s += self._reference_lane(lanes).length

        return prefix_by_road

    def _road_pairs(self) -> Iterable[RoadKey]:
        return zip(self.checkpoints[:-1], self.checkpoints[1:])

    def _reference_lane(self, lanes):
        if self.lane_id is not None and 0 <= self.lane_id < len(lanes):
            return lanes[self.lane_id]
        return lanes[0]

    def contains_road(self, lane_index) -> bool:
        return self._road_key(lane_index) in self.prefix_by_road

    def project_vehicle(self, vehicle, use_closest_lane: bool = False) -> Optional[RouteProjection]:
        """
        Project a vehicle onto this route.

        Args:
            vehicle: A MetaDrive vehicle-like object with position/lane/lane_index.
            use_closest_lane: If True, localize by closest lane instead of trusting
                vehicle.lane_index. Useful for projecting nearby traffic vehicles
                onto the ego route.
        """
        return self.project_position(
            vehicle.position,
            lane=getattr(vehicle, "lane", None),
            lane_index=getattr(vehicle, "lane_index", None),
            use_closest_lane=use_closest_lane,
        )

    def project_position(
        self,
        position,
        lane=None,
        lane_index=None,
        use_closest_lane: bool = False,
    ) -> Optional[RouteProjection]:
        """
        Project a world position onto the route.

        Returns None when the position's lane is not part of this route.
        """
        if use_closest_lane or lane is None or lane_index is None:
            lane_index, _ = self.road_network.get_closest_lane_index(position)
            lane = self.road_network.get_lane(lane_index)

        road_key = self._road_key(lane_index)
        prefix = self.prefix_by_road.get(road_key)
        if prefix is None:
            return None

        local_s, local_d = lane.local_coordinates(position)
        return RouteProjection(
            route_s=float(prefix + local_s),
            local_s=float(local_s),
            local_d=float(local_d),
            lane=lane,
            lane_index=lane_index,
            road_key=road_key,
        )

    def longitudinal_distance(
        self,
        rear_vehicle,
        front_vehicle,
        subtract_vehicle_lengths: bool = True,
        use_closest_lane_for_front: bool = True,
    ) -> Optional[float]:
        """
        Return front_vehicle's route distance ahead of rear_vehicle.

        Positive means front_vehicle is ahead along this route. None means either
        vehicle cannot be projected to the route.
        """
        rear = self.project_vehicle(rear_vehicle, use_closest_lane=False)
        front = self.project_vehicle(front_vehicle, use_closest_lane=use_closest_lane_for_front)
        if rear is None or front is None:
            return None

        distance = front.route_s - rear.route_s
        if subtract_vehicle_lengths:
            distance -= self._vehicle_length(rear_vehicle) / 2.0
            distance -= self._vehicle_length(front_vehicle) / 2.0
        return float(distance)

    @staticmethod
    def _road_key(lane_index) -> RoadKey:
        return lane_index[0], lane_index[1]

    @staticmethod
    def _vehicle_length(vehicle) -> float:
        length = getattr(vehicle, "LENGTH", 0.0)
        if isinstance(length, np.ndarray):
            length = float(length)
        return float(length or 0.0)
