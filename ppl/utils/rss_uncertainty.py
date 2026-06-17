"""Pluggable perception uncertainty models for RSS runtime monitors.

The Gaussian model follows the state-level noise setting used by risk-based
safety-envelope work: other-agent state (x, y, v, theta) is perturbed before RSS
route/Frenet quantities are computed. It does not mutate MetaDrive world state.
"""

import copy
import math

import numpy as np


RSS_NOISE_LEVELS = {
    "small": {
        "position_x_sigma": 0.50,
        "position_y_sigma": 0.15,
        "speed_sigma": 0.50,
        "heading_sigma": 0.02,
    },
    "medium": {
        "position_x_sigma": 1.00,
        "position_y_sigma": 0.30,
        "speed_sigma": 1.00,
        "heading_sigma": 0.05,
    },
    "large": {
        "position_x_sigma": 1.87,
        "position_y_sigma": 0.54,
        "speed_sigma": 2.64,
        "heading_sigma": 0.10,
    },
}


class PerceivedVehicleView:
    """Vehicle-like read-only view with noisy perceived kinematics."""

    def __init__(self, source, position, velocity, speed, heading_theta, noise_info=None):
        self.source = source
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.speed = float(speed)
        self.heading_theta = float(heading_theta)
        self.heading = np.asarray(
            [math.cos(self.heading_theta), math.sin(self.heading_theta)],
            dtype=float,
        )
        self.noise_info = noise_info or {}

    def __getattr__(self, name):
        return getattr(self.source, name)


class RSSUncertainty:
    """Base class for RSS perception uncertainty models."""

    name = "none"

    def reset(self):
        pass

    def perceive_vehicle(self, vehicle, context):
        return vehicle, self.vehicle_info(active=False)

    def apply_front(self, front, context):
        return front, self.front_info(front, active=False)

    def apply_lateral(self, lateral, context):
        return lateral, self.lateral_info(lateral, active=False)

    @staticmethod
    def copy_front(front):
        return None if front is None else copy.copy(front)

    @staticmethod
    def copy_lateral(lateral):
        return None if lateral is None else copy.copy(lateral)

    def vehicle_info(
        self,
        active=False,
        position_x_noise=0.0,
        position_y_noise=0.0,
        speed_noise=0.0,
        heading_noise=0.0,
    ):
        return {
            "rss_uncertainty_type": self.name,
            "rss_uncertainty_active": bool(active),
            "rss_position_x_noise": float(position_x_noise),
            "rss_position_y_noise": float(position_y_noise),
            "rss_speed_noise": float(speed_noise),
            "rss_heading_noise": float(heading_noise),
        }

    def front_info(self, front, active=False):
        if front is None:
            return {
                "rss_uncertainty_type": self.name,
                "rss_uncertainty_active": False,
                "rss_front_distance_noise": 0.0,
                "rss_front_speed_noise": 0.0,
                "rss_front_distance_raw_uncertainty": None,
                "rss_front_distance_noisy": None,
                "rss_front_speed_raw": None,
                "rss_front_speed_noisy": None,
            }

        if not active:
            return {
                "rss_uncertainty_type": self.name,
                "rss_uncertainty_active": False,
                "rss_front_distance_noise": 0.0,
                "rss_front_speed_noise": 0.0,
                "rss_front_distance_raw_uncertainty": front.get("distance"),
                "rss_front_distance_noisy": None,
                "rss_front_speed_raw": None,
                "rss_front_speed_noisy": None,
            }

        raw_distance = front.get("raw_distance", front.get("distance"))
        used_distance = front.get("distance")
        noisy_distance = front.get("noisy_distance", used_distance)
        raw_speed = front.get("raw_speed")
        noisy_speed = front.get("noisy_speed", front.get("speed"))
        distance_noise = self._difference(noisy_distance, raw_distance)
        speed_noise = self._difference(noisy_speed, raw_speed)
        info = dict(front.get("uncertainty_info", {}))
        info.update(
            {
                "rss_uncertainty_type": self.name,
                "rss_uncertainty_active": bool(active),
                "rss_front_distance_noise": distance_noise,
                "rss_front_speed_noise": speed_noise,
                "rss_front_distance_raw": raw_distance,
                "rss_front_distance_raw_uncertainty": raw_distance,
                "rss_front_distance_noisy": noisy_distance,
                "rss_front_distance_used": used_distance,
                "rss_front_speed_raw": raw_speed,
                "rss_front_speed_noisy": noisy_speed,
            }
        )
        return info

    def lateral_info(self, lateral, active=False):
        if lateral is None:
            return {
                "rss_lateral_uncertainty_type": self.name,
                "rss_lateral_uncertainty_active": False,
                "rss_lateral_gap_noise": 0.0,
                "rss_lateral_gap_raw_uncertainty": None,
                "rss_lateral_gap_noisy": None,
            }

        if not active:
            return {
                "rss_lateral_uncertainty_type": self.name,
                "rss_lateral_uncertainty_active": False,
                "rss_lateral_gap_noise": 0.0,
                "rss_lateral_gap_raw_uncertainty": lateral.get("lateral_gap"),
                "rss_lateral_gap_noisy": None,
            }

        raw_gap = lateral.get("raw_lateral_gap", lateral.get("lateral_gap"))
        used_gap = lateral.get("lateral_gap")
        noisy_gap = lateral.get("noisy_lateral_gap", used_gap)
        gap_noise = self._difference(noisy_gap, raw_gap)
        info = dict(lateral.get("uncertainty_info", {}))
        info.update(
            {
                "rss_lateral_uncertainty_type": self.name,
                "rss_lateral_uncertainty_active": bool(active),
                "rss_lateral_gap_noise": gap_noise,
                "rss_lateral_gap_raw": raw_gap,
                "rss_lateral_gap_raw_uncertainty": raw_gap,
                "rss_lateral_gap_noisy": noisy_gap,
                "rss_lateral_gap_used": used_gap,
            }
        )
        return info

    @staticmethod
    def _difference(value, baseline):
        if value is None or baseline is None:
            return 0.0
        return float(value) - float(baseline)


class NoRSSUncertainty(RSSUncertainty):
    name = "none"


class GaussianRSSUncertainty(RSSUncertainty):
    """Add zero-mean Gaussian noise to perceived other-agent state."""

    name = "gaussian"

    def __init__(
        self,
        position_x_sigma=0.0,
        position_y_sigma=0.0,
        speed_sigma=0.0,
        heading_sigma=0.0,
        seed=0,
    ):
        self.position_x_sigma = float(position_x_sigma)
        self.position_y_sigma = float(position_y_sigma)
        self.speed_sigma = float(speed_sigma)
        self.heading_sigma = float(heading_sigma)
        self.seed = None if seed is None or int(seed) < 0 else int(seed)
        self.rng = np.random.RandomState(self.seed)

    def reset(self):
        pass

    def perceive_vehicle(self, vehicle, context):
        if not self._is_active():
            return vehicle, self.vehicle_info(active=False)

        raw_position = np.asarray(getattr(vehicle, "position", [0.0, 0.0]), dtype=float)[:2]
        raw_speed = max(0.0, self._vehicle_speed(vehicle))
        raw_heading = self._vehicle_heading_theta(vehicle)

        x_noise = self._normal(self.position_x_sigma)
        y_noise = self._normal(self.position_y_sigma)
        speed_noise = self._normal(self.speed_sigma)
        heading_noise = self._normal(self.heading_sigma)

        forward_axis = np.asarray([math.cos(raw_heading), math.sin(raw_heading)], dtype=float)
        lateral_axis = np.asarray([-forward_axis[1], forward_axis[0]], dtype=float)
        noisy_position = raw_position + x_noise * forward_axis + y_noise * lateral_axis

        noisy_speed = max(0.0, raw_speed + speed_noise)
        noisy_heading = raw_heading + heading_noise
        noisy_velocity = noisy_speed * np.asarray(
            [math.cos(noisy_heading), math.sin(noisy_heading)],
            dtype=float,
        )

        info = self.vehicle_info(
            active=True,
            position_x_noise=x_noise,
            position_y_noise=y_noise,
            speed_noise=noisy_speed - raw_speed,
            heading_noise=heading_noise,
        )
        return (
            PerceivedVehicleView(
                source=vehicle,
                position=noisy_position,
                velocity=noisy_velocity,
                speed=noisy_speed,
                heading_theta=noisy_heading,
                noise_info=info,
            ),
            info,
        )

    def apply_front(self, front, context):
        active = front is not None and bool(front.get("uncertainty_active", False))
        return front, self.front_info(front, active=active)

    def apply_lateral(self, lateral, context):
        active = lateral is not None and bool(lateral.get("uncertainty_active", False))
        return lateral, self.lateral_info(lateral, active=active)

    def _is_active(self):
        return (
            self.position_x_sigma > 0.0
            or self.position_y_sigma > 0.0
            or self.speed_sigma > 0.0
            or self.heading_sigma > 0.0
        )

    def _normal(self, sigma):
        if sigma <= 0.0:
            return 0.0
        return float(self.rng.normal(loc=0.0, scale=sigma))

    @staticmethod
    def _vehicle_speed(vehicle):
        return float(getattr(vehicle, "speed", 0.0) or 0.0)

    @staticmethod
    def _vehicle_heading_theta(vehicle):
        heading_theta = getattr(vehicle, "heading_theta", None)
        if heading_theta is not None:
            return float(heading_theta)

        heading = np.asarray(getattr(vehicle, "heading", [1.0, 0.0]), dtype=float)[:2]
        norm = np.linalg.norm(heading)
        if norm <= 1e-6:
            return 0.0
        heading = heading / norm
        return float(math.atan2(heading[1], heading[0]))


def build_rss_uncertainty(
    name="none",
    noise_level=None,
    front_distance_sigma=0.0,
    lateral_gap_sigma=0.0,
    position_x_sigma=None,
    position_y_sigma=None,
    speed_sigma=0.0,
    heading_sigma=0.0,
    seed=0,
):
    if name is None or name == "none":
        return NoRSSUncertainty()
    if name == GaussianRSSUncertainty.name:
        if noise_level is not None and noise_level != "custom":
            if noise_level not in RSS_NOISE_LEVELS:
                raise ValueError("Unknown RSS noise level: {}".format(noise_level))
            preset = RSS_NOISE_LEVELS[noise_level]
            if position_x_sigma is None:
                position_x_sigma = preset["position_x_sigma"]
            if position_y_sigma is None:
                position_y_sigma = preset["position_y_sigma"]
            if speed_sigma <= 0.0:
                speed_sigma = preset["speed_sigma"]
            if heading_sigma <= 0.0:
                heading_sigma = preset["heading_sigma"]
        if position_x_sigma is None:
            position_x_sigma = front_distance_sigma
        if position_y_sigma is None:
            position_y_sigma = lateral_gap_sigma
        return GaussianRSSUncertainty(
            position_x_sigma=position_x_sigma,
            position_y_sigma=position_y_sigma,
            speed_sigma=speed_sigma,
            heading_sigma=heading_sigma,
            seed=seed,
        )
    raise ValueError("Unknown RSS uncertainty model: {}".format(name))
