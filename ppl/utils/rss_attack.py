"""Pluggable attacks for RSS monitor inputs.

The attacks in this module operate on the object-level state consumed by the
RSS monitor. They do not change MetaDrive world state, vehicle dynamics, or the
policy observation.
"""

import copy


class RSSAttack:
    """Base class for RSS input attacks."""

    name = "none"

    def reset(self):
        pass

    def apply_front(self, front, context):
        return front, self.front_info(front, front, active=False, delta=0.0)

    def apply_lateral(self, lateral, context):
        return lateral, self.lateral_info(lateral, lateral, active=False, delta=0.0)

    @staticmethod
    def copy_front(front):
        return None if front is None else copy.copy(front)

    @staticmethod
    def copy_lateral(lateral):
        return None if lateral is None else copy.copy(lateral)

    def front_info(self, raw_front, used_front, active=False, delta=0.0):
        raw_distance = None if raw_front is None else raw_front.get("distance")
        used_distance = None if used_front is None else used_front.get("distance")
        return {
            "rss_attack_type": self.name,
            "rss_attack_active": bool(active),
            "rss_attack_delta": float(delta),
            "rss_front_distance_raw": raw_distance,
            "rss_front_distance_attacked": used_distance,
            "rss_front_distance_used": used_distance,
        }

    def lateral_info(self, raw_lateral, used_lateral, active=False, delta=0.0):
        raw_gap = None if raw_lateral is None else raw_lateral.get("lateral_gap")
        used_gap = None if used_lateral is None else used_lateral.get("lateral_gap")
        return {
            "rss_lateral_attack_type": self.name,
            "rss_lateral_attack_active": bool(active),
            "rss_lateral_attack_delta": float(delta),
            "rss_lateral_gap_raw": raw_gap,
            "rss_lateral_gap_attacked": used_gap,
            "rss_lateral_gap_used": used_gap,
        }


class NoRSSAttack(RSSAttack):
    name = "none"


class FrontRangeOverestimateAttack(RSSAttack):
    """Overestimate the longitudinal distance to the selected front vehicle."""

    name = "front_range_overestimate"

    def __init__(self, distance_delta):
        self.distance_delta = float(distance_delta)

    def apply_front(self, front, context):
        if front is None or self.distance_delta <= 0.0:
            return front, self.front_info(front, front, active=False, delta=0.0)

        attacked_front = self.copy_front(front)
        attacked_front["distance"] = max(0.0, float(front["distance"]) + self.distance_delta)
        return attacked_front, self.front_info(
            raw_front=front,
            used_front=attacked_front,
            active=True,
            delta=self.distance_delta,
        )


class FrontObjectRemovalAttack(RSSAttack):
    """Remove the selected front vehicle from the RSS monitor input."""

    name = "front_object_removal"

    def apply_front(self, front, context):
        if front is None:
            return front, self.front_info(front, front, active=False, delta=0.0)
        return None, self.front_info(raw_front=front, used_front=None, active=True, delta=0.0)


class LateralGapOverestimateAttack(RSSAttack):
    """Overestimate the lateral gap to the selected lateral vehicle."""

    name = "lateral_gap_overestimate"

    def __init__(self, gap_delta):
        self.gap_delta = float(gap_delta)

    def apply_lateral(self, lateral, context):
        if lateral is None or self.gap_delta <= 0.0:
            return lateral, self.lateral_info(lateral, lateral, active=False, delta=0.0)

        attacked_lateral = self.copy_lateral(lateral)
        attacked_gap = max(0.0, float(lateral["lateral_gap"]) + self.gap_delta)
        attacked_lateral["lateral_gap"] = attacked_gap
        attacked_lateral["unsafe"] = attacked_gap < float(lateral["safe_lateral_distance"])
        attacked_lateral["violation"] = float(lateral["safe_lateral_distance"]) - attacked_gap
        return attacked_lateral, self.lateral_info(
            raw_lateral=lateral,
            used_lateral=attacked_lateral,
            active=True,
            delta=self.gap_delta,
        )


def build_rss_attack(name="none", front_distance_delta=0.0, lateral_gap_delta=0.0):
    if name is None or name == "none":
        return NoRSSAttack()
    if name == FrontRangeOverestimateAttack.name:
        return FrontRangeOverestimateAttack(distance_delta=front_distance_delta)
    if name == FrontObjectRemovalAttack.name:
        return FrontObjectRemovalAttack()
    if name == LateralGapOverestimateAttack.name:
        return LateralGapOverestimateAttack(gap_delta=lateral_gap_delta)
    raise ValueError("Unknown RSS attack: {}".format(name))
