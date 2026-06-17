"""
Evaluation script for PPL (Preference-based Policy Learning) models
trained with train_ppl_metadrive_human.py.

Usage:
    python -m ppl.eval_script.metadrive.eval_ppl_metadrive \
        --path runs/PPL/PPL_xxxx/models \
        --ret_save_folder evaluate_results/ppl_eval \
        --start_ckpt 150 \
        --num_ckpt 10 \
        --skip 150

    # Single checkpoint evaluation:
    python -m ppl.eval_script.metadrive.eval_ppl_metadrive \
        --path runs/PPL/PPL_xxxx/models \
        --ckpt_index 3000 \
        --ret_save_folder evaluate_results/ppl_eval \
        --use_render
"""

import argparse
import copy
import os
import os.path as osp
import time

import numpy as np
import pandas as pd
from metadrive.policy.env_input_policy import EnvInputPolicy

from ppl.experiments.metadrive.driving_env import DrivingEnv
from ppl.ppl import PPL
from ppl.sb3.td3.policies import TD3Policy
from ppl.utils.metadrive_route_projection import MetaDriveRouteProjector
from ppl.utils.print_dict_utils import pretty_print, RecorderEnv
from ppl.utils.rss_attack import build_rss_attack
from ppl.utils.rss_uncertainty import build_rss_uncertainty
from ppl.utils.train_eval_config import baseline_eval_config

EVAL_ENV_START = baseline_eval_config["start_seed"]


class RSSObserver:
    """Longitudinal/lateral RSS observer and runtime shield for evaluation."""

    def __init__(
        self,
        response_time=1.0,
        ego_max_accel=2.0,
        ego_min_brake=4.0,
        front_max_brake=6.0,
        dynamics_mode="standard",
        lateral_threshold=2.0,
        max_front_distance=100.0,
        lateral_max_accel=0.8,
        lateral_min_brake=1.5,
        lateral_safe_margin=1.5,
        lateral_longitudinal_threshold=8.0,
        max_lateral_distance=8.0,
        log_every_step=False,
        debug_interval=0,
        shield_mode="standard",
        shield_brake=1.0,
        lateral_shield_brake=0.5,
        lateral_steering_scale=0.25,
        spring_longitudinal_buffer=2.0,
        spring_longitudinal_k=0.08,
        damper_longitudinal_k=0.015,
        spring_lateral_buffer=0.4,
        spring_lateral_k=0.05,
        damper_lateral_k=0.02,
        spring_lateral_max_steer=0.04,
        attack=None,
        uncertainty=None,
        verbose=False,
    ):
        self.response_time = response_time
        self.ego_max_accel = ego_max_accel
        self.ego_min_brake = ego_min_brake
        self.front_max_brake = front_max_brake
        self.dynamics_mode = dynamics_mode
        self.lateral_threshold = lateral_threshold
        self.max_front_distance = max_front_distance
        self.lateral_max_accel = lateral_max_accel
        self.lateral_min_brake = lateral_min_brake
        self.lateral_safe_margin = lateral_safe_margin
        self.lateral_longitudinal_threshold = lateral_longitudinal_threshold
        self.max_lateral_distance = max_lateral_distance
        self.log_every_step = log_every_step
        self.debug_interval = debug_interval
        self.shield_mode = shield_mode
        self.shield_brake = shield_brake
        self.lateral_shield_brake = lateral_shield_brake
        self.lateral_steering_scale = lateral_steering_scale
        self.spring_longitudinal_buffer = spring_longitudinal_buffer
        self.spring_longitudinal_k = spring_longitudinal_k
        self.damper_longitudinal_k = damper_longitudinal_k
        self.spring_lateral_buffer = spring_lateral_buffer
        self.spring_lateral_k = spring_lateral_k
        self.damper_lateral_k = damper_lateral_k
        self.spring_lateral_max_steer = spring_lateral_max_steer
        self.attack = attack if attack is not None else build_rss_attack()
        self.uncertainty = uncertainty if uncertainty is not None else build_rss_uncertainty()
        self.verbose = verbose
        self.was_longitudinal_unsafe = False
        self.was_lateral_unsafe = False
        self.was_shielding = False
        self.was_lateral_shielding = False
        self.attack.reset()
        self.uncertainty.reset()

    def reset(self):
        self.was_longitudinal_unsafe = False
        self.was_lateral_unsafe = False
        self.was_shielding = False
        self.was_lateral_shielding = False
        self.attack.reset()
        self.uncertainty.reset()

    def shield_action(self, env, action, env_seed, episode, step):
        action = np.asarray(action, dtype=float).copy()
        original_steering = float(action[0])
        original_throttle = float(action[1])
        front_state = self._find_longitudinal_front(env, env_seed=env_seed, episode=episode, step=step)
        empty_info = {
            "rss_shield_active": False,
            "rss_longitudinal_shield_active": False,
            "rss_lateral_shield_active": False,
        }
        if front_state is None:
            if self.verbose and (self.was_shielding or self.was_lateral_shielding):
                print(
                    "[RSS SHIELD CLEAR] seed={} episode={} step={} no_route_projection".format(
                        env_seed, episode, step
                    )
                )
            self.was_shielding = False
            self.was_lateral_shielding = False
            return action, empty_info

        longitudinal_active = False
        response = {"response": "none", "throttle_cap": None}
        raw_front = front_state["front"]
        front, attack_info = self._attack_front_candidate(
            raw_front,
            front_state,
            env_seed=env_seed,
            episode=episode,
            step=step,
        )
        front, uncertainty_info = self._uncertainty_front_candidate(front, front_state)
        attack_info["rss_front_distance_used"] = None if front is None else front.get("distance")
        front_state["front"] = front
        front_state["raw_front"] = raw_front
        ego = front_state["ego"]
        ego_proj = front_state["ego_proj"]
        long_info = {}
        if front is None:
            if self.verbose and self.was_shielding and not self.was_lateral_shielding:
                print(
                    "[RSS SHIELD CLEAR] seed={} episode={} step={} no_front_vehicle".format(
                        env_seed, episode, step
                    )
                )
            self.was_shielding = False
        else:
            front_vehicle = front["vehicle"]
            front_proj = front["proj"]
            distance = front["distance"]
            ego_max_accel, ego_min_brake, front_max_brake = self._longitudinal_dynamics(ego, front_vehicle)
            front_speed = front.get("speed", self._object_speed(front_vehicle))
            safe_distance = self.safe_longitudinal_distance(
                ego.speed,
                front_speed,
                ego_max_accel=ego_max_accel,
                ego_min_brake=ego_min_brake,
                front_max_brake=front_max_brake,
            )
            violation = safe_distance - distance
            long_info = {
                "rss_shield_distance": distance,
                "rss_shield_safe_distance": safe_distance,
                "rss_longitudinal_shield_violation": max(0.0, float(violation)),
                "rss_shield_front_vehicle": getattr(front_vehicle, "name", getattr(front_vehicle, "id", "")),
                **attack_info,
                **uncertainty_info,
            }
            response = self._longitudinal_shield_response(
                action=action,
                distance=distance,
                safe_distance=safe_distance,
                ego_speed=ego.speed,
                front_speed=front_speed,
            )
            if response["active"]:
                if response["throttle_cap"] is not None:
                    action[1] = min(float(action[1]), response["throttle_cap"])
                else:
                    action[1] = min(float(action[1]), -response["brake"])
                longitudinal_active = response["response"] == "hard"
                if self.verbose and (not self.was_shielding or self.log_every_step):
                    print(
                        "[RSS SHIELD] seed={} episode={} step={} "
                        "distance={:.2f} safe_distance={:.2f} violation={:.2f} "
                        "response={} brake={:.2f} action_throttle={:.2f}->{:.2f} "
                        "ego_acc={:.2f} ego_brake={:.2f} front_brake={:.2f} "
                        "ego_v={:.2f} front_v={:.2f} ego_s={:.2f} front_s={:.2f} front={} type={}".format(
                            env_seed,
                            episode,
                            step,
                            distance,
                            safe_distance,
                            violation,
                            response["response"],
                            response["brake"],
                            original_throttle,
                            action[1],
                            ego_max_accel,
                            ego_min_brake,
                            front_max_brake,
                            ego.speed,
                            front_speed,
                            ego_proj.route_s,
                            front_proj.route_s,
                            getattr(front_vehicle, "name", getattr(front_vehicle, "id", "unknown")),
                            getattr(
                                front_vehicle,
                                "metadrive_type",
                                getattr(front_vehicle, "class_name", type(front_vehicle).__name__),
                            ),
                        )
                    )
                long_info.update(
                    {
                        "rss_shield_brake": response["brake"],
                        "rss_shield_response": response["response"],
                        "rss_shield_closing_speed": response["closing_speed"],
                        "rss_shield_soft_penetration": response["soft_penetration"],
                    }
                )
            elif self.verbose and self.was_shielding:
                print(
                    "[RSS SHIELD CLEAR] seed={} episode={} step={} "
                    "distance={:.2f} safe_distance={:.2f}".format(
                        env_seed, episode, step, distance, safe_distance
                    )
                )

        raw_lateral = self._find_lateral_candidate(front_state)
        lateral, lateral_attack_info = self._attack_lateral_candidate(
            raw_lateral,
            front_state,
            env_seed=env_seed,
            episode=episode,
            step=step,
        )
        lateral, lateral_uncertainty_info = self._uncertainty_lateral_candidate(lateral, front_state)
        lateral_attack_info["rss_lateral_gap_used"] = None if lateral is None else lateral.get("lateral_gap")
        front_state["lateral"] = lateral
        front_state["raw_lateral"] = raw_lateral
        lateral_unsafe = lateral is not None and lateral["unsafe"]
        lateral_active = False
        lateral_info = {**lateral_attack_info, **lateral_uncertainty_info}
        if lateral is not None:
            vehicle = lateral["vehicle"]
            signed_lateral_delta = self._relative_lateral_delta(ego, vehicle)
            steering_toward_vehicle = self._is_steering_toward_lateral_vehicle(
                original_steering, signed_lateral_delta
            )
            lateral_response = self._lateral_shield_response(
                action=action,
                lateral=lateral,
                signed_lateral_delta=signed_lateral_delta,
                steering_toward_vehicle=steering_toward_vehicle,
            )
            if lateral_response["active"]:
                action[0] = lateral_response["steering"]
                if lateral_response["brake"] > 0.0:
                    action[1] = min(float(action[1]), -lateral_response["brake"])
                lateral_active = lateral_response["response"] in {"hard", "hard_brake"}
                lateral_info.update({
                    "rss_lateral_shield_vehicle": getattr(vehicle, "name", getattr(vehicle, "id", "")),
                    "rss_lateral_shield_gap": lateral["lateral_gap"],
                    "rss_lateral_shield_safe_distance": lateral["safe_lateral_distance"],
                    "rss_lateral_shield_longitudinal_gap": lateral["longitudinal_gap"],
                    "rss_lateral_shield_brake": lateral_response["brake"],
                    "rss_lateral_shield_source": lateral.get("source", ""),
                    "rss_lateral_shield_signed_delta": signed_lateral_delta,
                    "rss_lateral_shield_steering_toward": steering_toward_vehicle,
                    "rss_lateral_shield_severe": lateral_response["severe"],
                    "rss_lateral_shield_response": lateral_response["response"],
                    "rss_lateral_shield_closing_speed": lateral_response["closing_speed"],
                    "rss_lateral_shield_soft_penetration": lateral_response["soft_penetration"],
                })
            if lateral_response["active"] and self.verbose and (not self.was_lateral_shielding or self.log_every_step):
                print(
                    "[RSS LAT SHIELD] seed={} episode={} step={} "
                        "lat_gap={:.2f} safe_lat={:.2f} long_gap={:.2f} "
                    "response={} toward={} severe={} action_steer={:.2f}->{:.2f} "
                    "action_throttle={:.2f}->{:.2f} other={} source={}".format(
                        env_seed,
                        episode,
                        step,
                        lateral["lateral_gap"],
                        lateral["safe_lateral_distance"],
                        lateral["longitudinal_gap"],
                        lateral_response["response"],
                        steering_toward_vehicle,
                        lateral_response["severe"],
                        original_steering,
                        action[0],
                        original_throttle,
                        action[1],
                        getattr(vehicle, "name", getattr(vehicle, "id", "unknown")),
                        lateral.get("source", ""),
                    )
                )
        elif self.verbose and self.was_lateral_shielding:
            print(
                "[RSS LAT SHIELD CLEAR] seed={} episode={} step={}".format(
                    env_seed, episode, step
                )
            )

        spring_soft_active = response.get("response") == "spring_damper_soft" or lateral_info.get(
            "rss_lateral_shield_response"
        ) == "spring_damper"
        shield_active = longitudinal_active or lateral_active or spring_soft_active
        shield_started = shield_active and not (self.was_shielding or self.was_lateral_shielding)
        self.was_shielding = longitudinal_active
        self.was_lateral_shielding = lateral_active

        return action, {
            "rss_shield_active": shield_active,
            "rss_shield_started": shield_started,
            "rss_longitudinal_shield_active": longitudinal_active,
            "rss_lateral_shield_active": lateral_active,
            "rss_spring_soft_active": spring_soft_active,
            "rss_longitudinal_spring_soft_active": response.get("response") == "spring_damper_soft",
            "rss_lateral_spring_soft_active": lateral_info.get("rss_lateral_shield_response") == "spring_damper",
            "rss_shield_original_throttle": original_throttle,
            "rss_shield_original_steering": original_steering,
            "rss_shield_action_throttle": float(action[1]),
            "rss_shield_action_steering": float(action[0]),
            "rss_shield_ego_speed_before": float(ego.speed),
            **long_info,
            **lateral_info,
        }

    def _proper_response_brake(self):
        return float(np.clip(self.shield_brake, 0.0, 1.0))

    def _proper_response_lateral_brake(self):
        return float(np.clip(self.lateral_shield_brake, 0.0, 1.0))

    def _longitudinal_shield_response(self, action, distance, safe_distance, ego_speed, front_speed):
        violation = safe_distance - distance
        closing_speed = max(0.0, float(ego_speed) - float(front_speed))
        response = {
            "active": False,
            "brake": 0.0,
            "response": "none",
            "closing_speed": closing_speed,
            "soft_penetration": 0.0,
            "throttle_cap": None,
        }

        if self.shield_mode == "standard":
            if violation > 0.0:
                response.update(
                    {
                        "active": True,
                        "brake": self._proper_response_brake(),
                        "response": "hard",
                    }
                )
            return response

        if violation > 0.0:
            response.update(
                {
                    "active": True,
                    "brake": self._proper_response_brake(),
                    "response": "hard",
                    "soft_penetration": float(violation),
                }
            )
            return response

        if self.shield_mode != "spring_damper":
            return response

        if self.spring_longitudinal_buffer <= 0.0:
            return response

        soft_boundary = safe_distance + self.spring_longitudinal_buffer
        soft_penetration = soft_boundary - distance
        response["soft_penetration"] = max(0.0, float(soft_penetration))
        if soft_penetration <= 0.0 or closing_speed <= 0.1 or float(action[1]) <= 0.0:
            return response

        buffer = max(self.spring_longitudinal_buffer, 1e-3)
        penetration_ratio = float(np.clip(soft_penetration / buffer, 0.0, 1.0))
        throttle_reduction = self.spring_longitudinal_k * penetration_ratio + self.damper_longitudinal_k * closing_speed
        throttle_cap = max(0.0, float(action[1]) - throttle_reduction)
        if throttle_cap >= float(action[1]) - 1e-6:
            return response

        response.update(
            {
                "active": True,
                "response": "spring_damper_soft",
                "throttle_cap": throttle_cap,
            }
        )
        return response

    def _lateral_shield_response(self, action, lateral, signed_lateral_delta, steering_toward_vehicle):
        severe = lateral["lateral_gap"] < 0.30
        response = {
            "active": False,
            "steering": float(action[0]),
            "brake": 0.0,
            "response": "none",
            "severe": severe,
            "closing_speed": 0.0,
            "soft_penetration": 0.0,
        }

        if self.shield_mode == "standard":
            if not lateral["unsafe"]:
                return response
            lateral_brake = self._proper_response_lateral_brake()
            if steering_toward_vehicle:
                response.update(
                    {
                        "active": True,
                        "steering": float(np.clip(float(action[0]) * self.lateral_steering_scale, -1.0, 1.0)),
                        "brake": lateral_brake,
                        "response": "hard",
                    }
                )
            elif severe:
                response.update(
                    {
                        "active": True,
                        "brake": lateral_brake,
                        "response": "hard_brake",
                    }
                )
            return response

        if signed_lateral_delta is None or abs(signed_lateral_delta) < 1e-6:
            return response

        if lateral["unsafe"]:
            lateral_brake = self._proper_response_lateral_brake()
            if steering_toward_vehicle:
                response.update(
                    {
                        "active": True,
                        "steering": float(np.clip(float(action[0]) * self.lateral_steering_scale, -1.0, 1.0)),
                        "brake": lateral_brake,
                        "response": "hard",
                    }
                )
            elif severe:
                response.update(
                    {
                        "active": True,
                        "brake": lateral_brake,
                        "response": "hard_brake",
                    }
                )
            return response

        if self.spring_lateral_buffer <= 0.0:
            return response

        soft_boundary = lateral["safe_lateral_distance"] + self.spring_lateral_buffer
        soft_penetration = soft_boundary - lateral["lateral_gap"]
        response["soft_penetration"] = max(0.0, float(soft_penetration))
        if soft_penetration <= 0.0:
            return response

        signed_gap = float(lateral.get("other_d", 0.0) - lateral.get("ego_d", 0.0))
        if abs(signed_gap) < 1e-6:
            signed_gap = float(signed_lateral_delta)
        relative_lateral_speed = float(lateral.get("other_lateral_speed", 0.0)) - float(
            lateral.get("ego_lateral_speed", 0.0)
        )
        closing_speed = max(0.0, -np.sign(signed_gap) * relative_lateral_speed)
        response["closing_speed"] = closing_speed

        buffer = max(self.spring_lateral_buffer, 1e-3)
        desired_away = self.spring_lateral_k * (soft_penetration / buffer) + self.damper_lateral_k * closing_speed
        desired_away = float(np.clip(desired_away, 0.0, self.spring_lateral_max_steer))
        away_direction = float(np.sign(signed_lateral_delta))
        current_away = float(action[0]) * away_direction
        steer_delta = max(0.0, desired_away - current_away)

        brake = 0.0

        if steer_delta <= 1e-6 and brake <= 1e-6:
            return response

        response.update(
            {
                "active": True,
                "steering": float(np.clip(float(action[0]) + away_direction * steer_delta, -1.0, 1.0)),
                "brake": brake,
                "response": "spring_damper",
            }
        )
        return response

    def _relative_lateral_delta(self, ego, vehicle):
        ego_pos = np.asarray(getattr(ego, "position", [0.0, 0.0]), dtype=float)[:2]
        other_pos = np.asarray(getattr(vehicle, "position", [0.0, 0.0]), dtype=float)[:2]
        heading = np.asarray(getattr(ego, "heading", [1.0, 0.0]), dtype=float)[:2]
        heading_norm = np.linalg.norm(heading)
        if heading_norm <= 1e-6:
            return None
        forward_axis = heading / heading_norm
        lateral_axis = np.asarray([-forward_axis[1], forward_axis[0]], dtype=float)
        return float(np.dot(other_pos - ego_pos, lateral_axis))

    @staticmethod
    def _is_steering_toward_lateral_vehicle(steering, signed_lateral_delta):
        if signed_lateral_delta is None or abs(signed_lateral_delta) < 1e-6:
            return False
        if abs(steering) < 1e-3:
            return False
        return float(steering) * float(signed_lateral_delta) < 0.0

    def verify_shield_step(self, env, shield_info, env_seed, episode, step):
        if not shield_info.get("rss_shield_active"):
            return {}

        raw_env = self._unwrap(env)
        ego = getattr(raw_env, "vehicle", None)
        if ego is None:
            return {}

        dt = self._step_dt(raw_env)
        speed_before = float(shield_info.get("rss_shield_ego_speed_before", 0.0))
        speed_after = float(getattr(ego, "speed", 0.0) or 0.0)
        actual_throttle = getattr(ego, "throttle_brake", None)
        actual_throttle = None if actual_throttle is None else float(actual_throttle)
        decel = (speed_before - speed_after) / dt if dt > 1e-6 else 0.0

        if self.verbose and (shield_info.get("rss_shield_started") or self.log_every_step):
            print(
                "[RSS SHIELD VERIFY] seed={} episode={} step={} "
                "requested_action={:.2f} actual_throttle={} speed={:.2f}->{:.2f} "
                "decel={:.2f} dt={:.3f}".format(
                    env_seed,
                    episode,
                    step,
                    shield_info.get("rss_shield_action_throttle", 0.0),
                    "none" if actual_throttle is None else "{:.2f}".format(actual_throttle),
                    speed_before,
                    speed_after,
                    decel,
                    dt,
                )
            )
        return {
            "rss_shield_actual_throttle": actual_throttle,
            "rss_shield_ego_speed_after": speed_after,
            "rss_shield_decel": decel,
        }

    def observe(self, env, env_seed, episode, step):
        front_state = self._find_longitudinal_front(env, env_seed=env_seed, episode=episode, step=step)
        if front_state is None:
            return {"rss_unsafe": False}

        ego = front_state["ego"]
        ego_proj = front_state["ego_proj"]
        raw_front = front_state["front"]
        front, attack_info = self._attack_front_candidate(
            raw_front,
            front_state,
            env_seed=env_seed,
            episode=episode,
            step=step,
        )
        front, uncertainty_info = self._uncertainty_front_candidate(front, front_state)
        attack_info["rss_front_distance_used"] = None if front is None else front.get("distance")
        front_state["front"] = front
        front_state["raw_front"] = raw_front
        stats = front_state["stats"]
        raw_lateral = self._find_lateral_candidate(front_state)
        lateral, lateral_attack_info = self._attack_lateral_candidate(
            raw_lateral,
            front_state,
            env_seed=env_seed,
            episode=episode,
            step=step,
        )
        lateral, lateral_uncertainty_info = self._uncertainty_lateral_candidate(lateral, front_state)
        lateral_attack_info["rss_lateral_gap_used"] = None if lateral is None else lateral.get("lateral_gap")
        front_state["lateral"] = lateral
        front_state["raw_lateral"] = raw_lateral

        longitudinal_info = self._observe_longitudinal(
            front,
            ego,
            ego_proj,
            env_seed,
            episode,
            step,
            {**attack_info, **uncertainty_info},
        )
        lateral_info = self._observe_lateral(
            lateral,
            env_seed,
            episode,
            step,
            {**lateral_attack_info, **lateral_uncertainty_info},
        )
        self._debug(env_seed, episode, step, stats, front, lateral, longitudinal_info, lateral_info)
        return {
            "rss_unsafe": longitudinal_info["rss_longitudinal_unsafe"] or lateral_info["rss_lateral_unsafe"],
            **longitudinal_info,
            **lateral_info,
        }

    def _attack_front_candidate(self, front, front_state, env_seed, episode, step):
        context = {
            "front_state": front_state,
            "env_seed": env_seed,
            "episode": episode,
            "step": step,
        }
        return self.attack.apply_front(front, context)

    def _attack_lateral_candidate(self, lateral, front_state, env_seed, episode, step):
        context = {
            "front_state": front_state,
            "env_seed": env_seed,
            "episode": episode,
            "step": step,
        }
        return self.attack.apply_lateral(lateral, context)

    def _uncertainty_front_candidate(self, front, front_state):
        return self.uncertainty.apply_front(front, {"front_state": front_state})

    def _uncertainty_lateral_candidate(self, lateral, front_state):
        return self.uncertainty.apply_lateral(lateral, {"front_state": front_state})

    def _find_longitudinal_front(self, env, env_seed=None, episode=None, step=None):
        raw_env = self._unwrap(env)
        ego = getattr(raw_env, "vehicle", None)
        if ego is None or getattr(ego, "navigation", None) is None:
            return None

        try:
            projector = MetaDriveRouteProjector.from_vehicle(ego)
            ego_proj = projector.project_vehicle(ego)
        except Exception as error:
            if self.verbose and self.log_every_step:
                print(f"[RSS] skipped: {error}")
            return None

        if ego_proj is None:
            return None

        front = None
        traffic_vehicles = []
        route_vehicle_projections = []
        stats = {
            "vehicles": 0,
            "route_vehicles": 0,
            "front_candidates": 0,
            "lateral_candidates": 0,
            "lateral_lane_candidates": 0,
            "lateral_long_filtered": 0,
            "lateral_far_filtered": 0,
        }

        for vehicle in self._vehicles(raw_env):
            stats["vehicles"] += 1
            if vehicle is ego:
                continue

            perceived_vehicle, uncertainty_info = self.uncertainty.perceive_vehicle(
                vehicle,
                {
                    "ego": ego,
                    "env_seed": env_seed,
                    "episode": episode,
                    "step": step,
                },
            )
            traffic_vehicles.append(
                {
                    "vehicle": vehicle,
                    "perceived_vehicle": perceived_vehicle,
                    "uncertainty_info": uncertainty_info,
                    "uncertainty_active": uncertainty_info.get("rss_uncertainty_active", False),
                }
            )
            raw_proj = projector.project_vehicle(vehicle, use_closest_lane=True)
            proj = projector.project_vehicle(perceived_vehicle, use_closest_lane=True)
            if proj is None:
                continue

            stats["route_vehicles"] += 1
            route_vehicle_projections.append(
                {
                    "vehicle": vehicle,
                    "perceived_vehicle": perceived_vehicle,
                    "proj": proj,
                    "raw_proj": raw_proj,
                    "uncertainty_info": uncertainty_info,
                    "uncertainty_active": uncertainty_info.get("rss_uncertainty_active", False),
                }
            )
            if proj.lane_index[2] != ego_proj.lane_index[2]:
                continue

            lateral_gap = abs(proj.local_d - ego_proj.local_d)
            if lateral_gap > self.lateral_threshold:
                continue

            distance = proj.route_s - ego_proj.route_s - self._vehicle_length(ego) / 2.0 - self._vehicle_length(
                vehicle
            ) / 2.0
            if distance <= 0 or distance > self.max_front_distance:
                continue

            stats["front_candidates"] += 1
            if front is None or distance < front["distance"]:
                raw_distance = None
                if raw_proj is not None:
                    raw_distance = (
                        raw_proj.route_s
                        - ego_proj.route_s
                        - self._vehicle_length(ego) / 2.0
                        - self._vehicle_length(vehicle) / 2.0
                    )
                raw_speed = self._object_speed(vehicle)
                noisy_speed = self._object_speed(perceived_vehicle)
                front = {
                    "vehicle": perceived_vehicle,
                    "raw_vehicle": vehicle,
                    "proj": proj,
                    "raw_proj": raw_proj,
                    "distance": distance,
                    "raw_distance": None if raw_distance is None else float(raw_distance),
                    "noisy_distance": float(distance),
                    "speed": noisy_speed,
                    "raw_speed": raw_speed,
                    "noisy_speed": noisy_speed,
                    "uncertainty_info": uncertainty_info,
                    "uncertainty_active": uncertainty_info.get("rss_uncertainty_active", False),
                }

        return {
            "ego": ego,
            "ego_proj": ego_proj,
            "projector": projector,
            "front": front,
            "traffic_vehicles": traffic_vehicles,
            "route_vehicle_projections": route_vehicle_projections,
            "stats": stats,
        }

    def _find_lateral_candidate(self, front_state):
        ego = front_state["ego"]
        ego_proj = front_state["ego_proj"]
        projector = front_state["projector"]
        ego_ref = self._reference_coordinates(projector, ego.position, ego_proj.road_key)
        stats = front_state["stats"]
        lateral = None
        projected_vehicle_ids = set()

        for vehicle_entry in front_state["route_vehicle_projections"]:
            vehicle = vehicle_entry["perceived_vehicle"]
            raw_vehicle = vehicle_entry["vehicle"]
            proj = vehicle_entry["proj"]
            if vehicle is ego:
                continue
            projected_vehicle_ids.add(id(raw_vehicle))

            if ego_ref is not None:
                route_candidate = self._lateral_candidate(
                    projector=projector,
                    ego=ego,
                    ego_ref=ego_ref,
                    vehicle=vehicle,
                    proj=proj,
                    raw_vehicle=raw_vehicle,
                    raw_proj=vehicle_entry.get("raw_proj"),
                    uncertainty_info=vehicle_entry.get("uncertainty_info"),
                    uncertainty_active=vehicle_entry.get("uncertainty_active", False),
                )
                lateral = self._record_lateral_candidate(stats, lateral, route_candidate)

            relative_candidate = self._relative_lateral_candidate(
                ego,
                vehicle,
                raw_vehicle=raw_vehicle,
                uncertainty_info=vehicle_entry.get("uncertainty_info"),
                uncertainty_active=vehicle_entry.get("uncertainty_active", False),
            )
            lateral = self._record_lateral_candidate(stats, lateral, relative_candidate)

        for vehicle_entry in front_state.get("traffic_vehicles", []):
            raw_vehicle = vehicle_entry["vehicle"]
            if id(raw_vehicle) in projected_vehicle_ids:
                continue
            relative_candidate = self._relative_lateral_candidate(
                ego,
                vehicle_entry["perceived_vehicle"],
                raw_vehicle=raw_vehicle,
                uncertainty_info=vehicle_entry.get("uncertainty_info"),
                uncertainty_active=vehicle_entry.get("uncertainty_active", False),
            )
            lateral = self._record_lateral_candidate(stats, lateral, relative_candidate)

        return lateral

    def _record_lateral_candidate(self, stats, lateral, candidate):
        if candidate == "longitudinal_gap":
            stats["lateral_long_filtered"] += 1
            return lateral
        if candidate == "lateral_distance":
            stats["lateral_far_filtered"] += 1
            return lateral
        if candidate is None:
            return lateral
        if candidate.get("same_lane", False):
            return lateral
        stats["lateral_lane_candidates"] += 1
        stats["lateral_candidates"] += 1
        return self._select_lateral_candidate(lateral, candidate)

    def _observe_longitudinal(self, front, ego, ego_proj, env_seed, episode, step, attack_info=None):
        attack_info = attack_info or {}
        if front is None:
            if self.verbose and self.was_longitudinal_unsafe:
                print(
                    "[RSS CLEAR] seed={} episode={} step={} no_front_vehicle".format(
                        env_seed, episode, step
                    )
                )
            self.was_longitudinal_unsafe = False
            return {"rss_longitudinal_unsafe": False, **attack_info}

        front_vehicle = front["vehicle"]
        front_proj = front["proj"]
        front_distance = front["distance"]
        ego_max_accel, ego_min_brake, front_max_brake = self._longitudinal_dynamics(ego, front_vehicle)
        front_speed = front.get("speed", self._object_speed(front_vehicle))
        safe_distance = self.safe_longitudinal_distance(
            ego.speed,
            front_speed,
            ego_max_accel=ego_max_accel,
            ego_min_brake=ego_min_brake,
            front_max_brake=front_max_brake,
        )
        unsafe = front_distance < safe_distance

        if self.verbose and unsafe and (not self.was_longitudinal_unsafe or self.log_every_step):
            print(
                "[RSS TRIGGER] seed={} episode={} step={} "
                "distance={:.2f} safe_distance={:.2f} "
                "ego_acc={:.2f} ego_brake={:.2f} front_brake={:.2f} "
                "ego_v={:.2f} front_v={:.2f} "
                "ego_s={:.2f} front_s={:.2f} "
                "front={} type={}".format(
                    env_seed,
                    episode,
                    step,
                    front_distance,
                    safe_distance,
                    ego_max_accel,
                    ego_min_brake,
                    front_max_brake,
                    ego.speed,
                    front_speed,
                    ego_proj.route_s,
                    front_proj.route_s,
                    getattr(front_vehicle, "name", getattr(front_vehicle, "id", "unknown")),
                    getattr(front_vehicle, "metadrive_type", getattr(front_vehicle, "class_name", type(front_vehicle).__name__)),
                )
            )
        elif self.verbose and not unsafe and self.was_longitudinal_unsafe:
            print(
                "[RSS CLEAR] seed={} episode={} step={} "
                "distance={:.2f} safe_distance={:.2f}".format(
                    env_seed, episode, step, front_distance, safe_distance
                )
            )

        self.was_longitudinal_unsafe = unsafe
        return {
            "rss_longitudinal_unsafe": unsafe,
            "rss_distance": front_distance,
            "rss_safe_distance": safe_distance,
            "rss_front_vehicle": getattr(front_vehicle, "name", getattr(front_vehicle, "id", "")),
            "rss_front_type": getattr(front_vehicle, "metadrive_type", getattr(front_vehicle, "class_name", "")),
            **attack_info,
        }

    def _observe_lateral(self, lateral, env_seed, episode, step, lateral_attack_info=None):
        lateral_attack_info = lateral_attack_info or {}
        if lateral is None:
            if self.verbose and self.was_lateral_unsafe:
                print(
                    "[RSS LAT CLEAR] seed={} episode={} step={} no_nearby_vehicle".format(
                        env_seed, episode, step
                    )
                )
            self.was_lateral_unsafe = False
            return {"rss_lateral_unsafe": False, **lateral_attack_info}

        unsafe = lateral["unsafe"]
        vehicle = lateral["vehicle"]
        if self.verbose and unsafe and (not self.was_lateral_unsafe or self.log_every_step):
            print(
                "[RSS LAT TRIGGER] seed={} episode={} step={} "
                "lat_gap={:.2f} safe_lat={:.2f} long_gap={:.2f} "
                "ego_d={:.2f} other_d={:.2f} "
                "ego_lat_v={:.2f} other_lat_v={:.2f} "
                "other={}".format(
                    env_seed,
                    episode,
                    step,
                    lateral["lateral_gap"],
                    lateral["safe_lateral_distance"],
                    lateral["longitudinal_gap"],
                    lateral["ego_d"],
                    lateral["other_d"],
                    lateral["ego_lateral_speed"],
                    lateral["other_lateral_speed"],
                    getattr(vehicle, "name", getattr(vehicle, "id", "unknown")),
                )
            )
        elif self.verbose and not unsafe and self.was_lateral_unsafe:
            print(
                "[RSS LAT CLEAR] seed={} episode={} step={} "
                "lat_gap={:.2f} safe_lat={:.2f} long_gap={:.2f}".format(
                    env_seed,
                    episode,
                    step,
                    lateral["lateral_gap"],
                    lateral["safe_lateral_distance"],
                    lateral["longitudinal_gap"],
                )
            )

        self.was_lateral_unsafe = unsafe
        return {
            "rss_lateral_unsafe": unsafe,
            "rss_lateral_gap": lateral["lateral_gap"],
            "rss_lateral_safe_distance": lateral["safe_lateral_distance"],
            "rss_lateral_vehicle": getattr(vehicle, "name", getattr(vehicle, "id", "")),
            "rss_lateral_source": lateral.get("source", ""),
            **lateral_attack_info,
        }

    def _lateral_candidate(
        self,
        projector,
        ego,
        ego_ref,
        vehicle,
        proj,
        raw_vehicle=None,
        raw_proj=None,
        uncertainty_info=None,
        uncertainty_active=False,
    ):
        other_ref = self._reference_coordinates(projector, vehicle.position, proj.road_key)
        if other_ref is None:
            return None

        ego_route_s, ego_local_s, ego_d, ego_ref_lane = ego_ref
        other_route_s, other_local_s, other_d, other_ref_lane = other_ref
        longitudinal_gap = abs(other_route_s - ego_route_s) - self._vehicle_length(ego) / 2.0 - self._vehicle_length(
            vehicle
        ) / 2.0
        longitudinal_gap = max(0.0, float(longitudinal_gap))
        if longitudinal_gap > self.lateral_longitudinal_threshold:
            return "longitudinal_gap"

        signed_lateral_delta = other_d - ego_d
        center_lateral_distance = abs(signed_lateral_delta)
        if center_lateral_distance > self.max_lateral_distance:
            return "lateral_distance"

        lateral_gap = center_lateral_distance - self._vehicle_width(ego) / 2.0 - self._vehicle_width(vehicle) / 2.0
        ego_lateral_speed = self._lateral_speed(ego, ego_ref_lane, ego_local_s, ego_d)
        other_lateral_speed = self._lateral_speed(vehicle, other_ref_lane, other_local_s, other_d)
        safe_lateral_distance = self.safe_lateral_distance(
            signed_lateral_delta, ego_lateral_speed, other_lateral_speed
        )
        unsafe = lateral_gap < safe_lateral_distance
        violation = safe_lateral_distance - lateral_gap
        lane_width_like_gap = max(3.2, self._vehicle_width(ego) + self._vehicle_width(vehicle))
        same_lane = center_lateral_distance < lane_width_like_gap * 0.5
        raw_lateral_gap = None
        if raw_vehicle is not None and raw_proj is not None:
            raw_ref = self._reference_coordinates(projector, raw_vehicle.position, raw_proj.road_key)
            if raw_ref is not None:
                raw_other_route_s, _, raw_other_d, _ = raw_ref
                raw_center_lateral_distance = abs(raw_other_d - ego_d)
                raw_lateral_gap = (
                    raw_center_lateral_distance
                    - self._vehicle_width(ego) / 2.0
                    - self._vehicle_width(raw_vehicle) / 2.0
                )
                raw_longitudinal_gap = abs(raw_other_route_s - ego_route_s) - self._vehicle_length(
                    ego
                ) / 2.0 - self._vehicle_length(raw_vehicle) / 2.0
                raw_longitudinal_gap = max(0.0, float(raw_longitudinal_gap))
            else:
                raw_longitudinal_gap = None
        else:
            raw_longitudinal_gap = None

        candidate = {
            "vehicle": vehicle,
            "raw_vehicle": raw_vehicle or vehicle,
            "unsafe": unsafe,
            "violation": violation,
            "lateral_gap": float(lateral_gap),
            "raw_lateral_gap": None if raw_lateral_gap is None else float(raw_lateral_gap),
            "noisy_lateral_gap": float(lateral_gap),
            "safe_lateral_distance": safe_lateral_distance,
            "longitudinal_gap": longitudinal_gap,
            "raw_longitudinal_gap": raw_longitudinal_gap,
            "ego_d": ego_d,
            "other_d": other_d,
            "ego_lateral_speed": ego_lateral_speed,
            "other_lateral_speed": other_lateral_speed,
            "source": "route",
            "same_lane": same_lane,
            "uncertainty_info": uncertainty_info or {},
            "uncertainty_active": bool(uncertainty_active),
        }
        return candidate

    def _relative_lateral_candidate(self, ego, vehicle, raw_vehicle=None, uncertainty_info=None, uncertainty_active=False):
        ego_pos = np.asarray(getattr(ego, "position", [0.0, 0.0]), dtype=float)[:2]
        other_pos = np.asarray(getattr(vehicle, "position", [0.0, 0.0]), dtype=float)[:2]
        heading = np.asarray(getattr(ego, "heading", [1.0, 0.0]), dtype=float)[:2]
        heading_norm = np.linalg.norm(heading)
        if heading_norm <= 1e-6:
            return None

        forward_axis = heading / heading_norm
        lateral_axis = np.asarray([-forward_axis[1], forward_axis[0]], dtype=float)
        delta = other_pos - ego_pos
        signed_longitudinal_delta = float(np.dot(delta, forward_axis))
        signed_lateral_delta = float(np.dot(delta, lateral_axis))
        longitudinal_gap = (
            abs(signed_longitudinal_delta)
            - self._vehicle_length(ego) / 2.0
            - self._vehicle_length(vehicle) / 2.0
        )
        longitudinal_gap = max(0.0, float(longitudinal_gap))
        if longitudinal_gap > self.lateral_longitudinal_threshold:
            return "longitudinal_gap"

        center_lateral_distance = abs(signed_lateral_delta)
        if center_lateral_distance > self.max_lateral_distance:
            return "lateral_distance"

        lateral_gap = center_lateral_distance - self._vehicle_width(ego) / 2.0 - self._vehicle_width(vehicle) / 2.0
        ego_velocity = np.asarray(getattr(ego, "velocity", [0.0, 0.0]), dtype=float)[:2]
        other_velocity = np.asarray(getattr(vehicle, "velocity", [0.0, 0.0]), dtype=float)[:2]
        ego_lateral_speed = float(np.dot(ego_velocity, lateral_axis))
        other_lateral_speed = float(np.dot(other_velocity, lateral_axis))
        safe_lateral_distance = self.safe_lateral_distance(
            signed_lateral_delta, ego_lateral_speed, other_lateral_speed
        )
        unsafe = lateral_gap < safe_lateral_distance
        violation = safe_lateral_distance - lateral_gap
        lane_width_like_gap = max(3.2, self._vehicle_width(ego) + self._vehicle_width(vehicle))
        same_lane = center_lateral_distance < lane_width_like_gap * 0.5
        raw_lateral_gap = None
        raw_longitudinal_gap = None
        if raw_vehicle is not None:
            raw_other_pos = np.asarray(getattr(raw_vehicle, "position", [0.0, 0.0]), dtype=float)[:2]
            raw_delta = raw_other_pos - ego_pos
            raw_signed_longitudinal_delta = float(np.dot(raw_delta, forward_axis))
            raw_signed_lateral_delta = float(np.dot(raw_delta, lateral_axis))
            raw_longitudinal_gap = (
                abs(raw_signed_longitudinal_delta)
                - self._vehicle_length(ego) / 2.0
                - self._vehicle_length(raw_vehicle) / 2.0
            )
            raw_longitudinal_gap = max(0.0, float(raw_longitudinal_gap))
            raw_lateral_gap = (
                abs(raw_signed_lateral_delta)
                - self._vehicle_width(ego) / 2.0
                - self._vehicle_width(raw_vehicle) / 2.0
            )

        return {
            "vehicle": vehicle,
            "raw_vehicle": raw_vehicle or vehicle,
            "unsafe": unsafe,
            "violation": violation,
            "lateral_gap": float(lateral_gap),
            "raw_lateral_gap": None if raw_lateral_gap is None else float(raw_lateral_gap),
            "noisy_lateral_gap": float(lateral_gap),
            "safe_lateral_distance": safe_lateral_distance,
            "longitudinal_gap": longitudinal_gap,
            "raw_longitudinal_gap": raw_longitudinal_gap,
            "ego_d": 0.0,
            "other_d": signed_lateral_delta,
            "ego_lateral_speed": ego_lateral_speed,
            "other_lateral_speed": other_lateral_speed,
            "source": "relative",
            "same_lane": same_lane,
            "uncertainty_info": uncertainty_info or {},
            "uncertainty_active": bool(uncertainty_active),
        }

    @staticmethod
    def _select_lateral_candidate(current, candidate):
        if current is None:
            return candidate
        if candidate["unsafe"] and not current["unsafe"]:
            return candidate
        if candidate["unsafe"] == current["unsafe"] and candidate["violation"] > current["violation"]:
            return candidate
        return current

    def _debug(self, env_seed, episode, step, stats, front, lateral, longitudinal_info, lateral_info):
        if not self.verbose:
            return
        if self.debug_interval <= 0 or step % self.debug_interval != 0:
            return

        front_part = "front=none"
        if front is not None:
            front_part = "front_dist={:.2f} front_safe={:.2f} front_unsafe={}".format(
                front["distance"],
                longitudinal_info.get("rss_safe_distance", float("nan")),
                longitudinal_info["rss_longitudinal_unsafe"],
            )

        lateral_part = "lat=none"
        if lateral is not None:
            lateral_part = "lat_gap={:.2f} lat_safe={:.2f} lat_unsafe={}".format(
                lateral["lateral_gap"],
                lateral["safe_lateral_distance"],
                lateral_info["rss_lateral_unsafe"],
            )

        print(
            "[RSS DEBUG] seed={} episode={} step={} vehicles={} route_vehicles={} "
            "front_candidates={} lateral_lane_candidates={} lateral_candidates={} "
            "lat_long_filtered={} lat_far_filtered={} {} {}".format(
                env_seed,
                episode,
                step,
                stats["vehicles"],
                stats["route_vehicles"],
                stats["front_candidates"],
                stats["lateral_lane_candidates"],
                stats["lateral_candidates"],
                stats["lateral_long_filtered"],
                stats["lateral_far_filtered"],
                front_part,
                lateral_part,
            )
        )

    def safe_longitudinal_distance(self, ego_speed, front_speed, ego_max_accel=None, ego_min_brake=None, front_max_brake=None):
        ego_speed = max(float(ego_speed), 0.0)
        front_speed = max(float(front_speed), 0.0)
        response = self.response_time
        ego_max_accel = self.ego_max_accel if ego_max_accel is None else ego_max_accel
        ego_min_brake = self.ego_min_brake if ego_min_brake is None else ego_min_brake
        front_max_brake = self.front_max_brake if front_max_brake is None else front_max_brake
        safe = (
            ego_speed * response
            + 0.5 * ego_max_accel * response**2
            + ((ego_speed + response * ego_max_accel)**2) / (2.0 * ego_min_brake)
            - (front_speed**2) / (2.0 * front_max_brake)
        )
        return max(0.0, float(safe))

    def _longitudinal_dynamics(self, ego, front_vehicle):
        if self.dynamics_mode == "manual":
            return self.ego_max_accel, self.ego_min_brake, self.front_max_brake
        if self.dynamics_mode == "idm":
            return 1.0, 5.0, 5.0
        if self.dynamics_mode == "standard":
            return 3.0, 4.0, 10.10
        if self.dynamics_mode == "measured":
            return 2.76, 11.19, 10.10
        if self.dynamics_mode != "vehicle":
            return self.ego_max_accel, self.ego_min_brake, self.front_max_brake
        return (
            self._max_accel_from_vehicle(ego, self.ego_max_accel),
            self._max_brake_from_vehicle(ego, self.ego_min_brake),
            self._max_brake_from_vehicle(front_vehicle, self.front_max_brake),
        )

    def safe_lateral_distance(self, signed_lateral_delta, ego_lateral_speed, other_lateral_speed):
        sign = 1.0 if signed_lateral_delta >= 0.0 else -1.0
        relative_lateral_speed = float(other_lateral_speed) - float(ego_lateral_speed)
        closing_speed = max(0.0, -sign * relative_lateral_speed)
        response = self.response_time
        safe = (
            self.lateral_safe_margin
            + closing_speed * response
            + 0.5 * self.lateral_max_accel * response**2
            + ((closing_speed + self.lateral_max_accel * response) ** 2) / (2.0 * self.lateral_min_brake)
        )
        return max(0.0, float(safe))

    def _reference_coordinates(self, projector, position, road_key):
        prefix = projector.prefix_by_road.get(road_key)
        if prefix is None:
            return None
        lanes = projector.road_network.graph[road_key[0]][road_key[1]]
        if len(lanes) == 0:
            return None
        lane_id = projector.lane_id if projector.lane_id is not None and 0 <= projector.lane_id < len(lanes) else 0
        lane = lanes[lane_id]
        local_s, local_d = lane.local_coordinates(position)
        return float(prefix + local_s), float(local_s), float(local_d), lane

    def _lateral_speed(self, vehicle, lane, local_s, local_d):
        velocity = np.asarray(getattr(vehicle, "velocity", [0.0, 0.0]), dtype=float)[:2]
        lateral_axis = self._lateral_axis(lane, local_s, local_d)
        return float(np.dot(velocity, lateral_axis))

    @staticmethod
    def _lateral_axis(lane, local_s, local_d):
        eps = 0.1
        try:
            point = np.asarray(lane.position(local_s, local_d), dtype=float)[:2]
            shifted = np.asarray(lane.position(local_s, local_d + eps), dtype=float)[:2]
            axis = shifted - point
            norm = np.linalg.norm(axis)
            if norm > 1e-6:
                return axis / norm
        except Exception:
            pass

        heading = lane.heading_theta_at(local_s)
        return np.asarray([np.sin(heading), -np.cos(heading)], dtype=float)

    @staticmethod
    def _vehicle_length(vehicle):
        return float(getattr(vehicle, "LENGTH", 0.0) or 0.0)

    @staticmethod
    def _vehicle_width(vehicle):
        return float(getattr(vehicle, "WIDTH", 0.0) or 0.0)

    @staticmethod
    def _object_speed(obj):
        return float(getattr(obj, "speed", 0.0) or 0.0)

    @staticmethod
    def _max_accel_from_vehicle(vehicle, fallback):
        params = RSSObserver._vehicle_dynamics_parameters(vehicle)
        if params is None:
            return fallback
        mass = params.get("mass", 0.0)
        max_engine_force = params.get("max_engine_force", 0.0)
        if mass <= 0.0 or max_engine_force <= 0.0:
            return fallback
        return float(max_engine_force / mass * 4.0)

    @staticmethod
    def _max_brake_from_vehicle(vehicle, fallback):
        params = RSSObserver._vehicle_dynamics_parameters(vehicle)
        if params is None:
            return fallback
        mass = params.get("mass", 0.0)
        max_brake_force = params.get("max_brake_force", 0.0)
        if mass <= 0.0 or max_brake_force <= 0.0:
            return fallback
        return float(max_brake_force / mass * 4.0)

    @staticmethod
    def _vehicle_dynamics_parameters(vehicle):
        if not hasattr(vehicle, "get_dynamics_parameters"):
            return None
        try:
            return vehicle.get_dynamics_parameters()
        except Exception:
            return None

    @staticmethod
    def _unwrap(env):
        while hasattr(env, "env"):
            env = env.env
        return env

    @staticmethod
    def _step_dt(env):
        config = getattr(env, "config", {})
        physics_dt = float(config.get("physics_world_step_size", 0.02))
        decision_repeat = float(config.get("decision_repeat", 5))
        return physics_dt * decision_repeat

    @staticmethod
    def _vehicles(env):
        traffic_manager = getattr(env.engine, "traffic_manager", None)
        if traffic_manager is not None and hasattr(traffic_manager, "vehicles"):
            vehicles = traffic_manager.vehicles
            return vehicles.values() if isinstance(vehicles, dict) else vehicles
        return []


class PolicyFunction:
    """Load a PPL checkpoint and wrap it as a callable policy."""

    def __init__(self, ckpt_path, env):
        """
        Args:
            ckpt_path: Full path to the .zip checkpoint file,
                       e.g. "runs/PPL/PPL_xxxx/models/rl_model_3000_steps.zip"
            env: A gym-compatible environment used to infer obs/act spaces.
        """
        self.algo = PPL(
            policy=TD3Policy,
            env=env,
            policy_kwargs=dict(net_arch=[256, 256]),
            # Minimal required hyper-params (not used during inference):
            replay_buffer_kwargs=dict(),
            learning_starts=1,
            buffer_size=100,
        )
        self.algo.set_parameters(load_path_or_dict=ckpt_path, exact_match=False, device=self.algo.device)
        print(f"[PolicyFunction] Loaded checkpoint: {ckpt_path}")

    def __call__(self, o, deterministic=True):
        return self.algo.predict(o, deterministic=deterministic)


def make_metadrive_env(use_render=False, direct_action_policy=False):
    """Build the evaluation environment (no manual control, fixed seed range)."""
    config = copy.deepcopy(baseline_eval_config)
    config.pop("main_exp", None)
    config["use_render"] = use_render
    if direct_action_policy:
        config["agent_policy"] = EnvInputPolicy
    if use_render:
        config["disable_model_compression"] = True
    env = DrivingEnv(config=config)
    return RecorderEnv(env)


def reset_eval_env(env, seed):
    """Reset with a fixed MetaDrive scenario across old/new Gym APIs."""
    try:
        return env.reset(force_seed=seed)
    except TypeError as force_seed_error:
        try:
            return env.reset(seed=seed)
        except TypeError:
            raise force_seed_error


def make_rss_episode_stats():
    return {
        "rss_unsafe_steps": 0,
        "rss_longitudinal_unsafe_steps": 0,
        "rss_lateral_unsafe_steps": 0,
        "rss_shield_steps": 0,
        "rss_longitudinal_shield_steps": 0,
        "rss_lateral_shield_steps": 0,
        "rss_spring_soft_steps": 0,
        "rss_longitudinal_spring_soft_steps": 0,
        "rss_lateral_spring_soft_steps": 0,
        "rss_attack_steps": 0,
        "rss_front_attack_steps": 0,
        "rss_lateral_attack_steps": 0,
        "rss_uncertainty_steps": 0,
        "rss_front_uncertainty_steps": 0,
        "rss_lateral_uncertainty_steps": 0,
        "rss_attack_delta_mean": 0.0,
        "rss_lateral_attack_delta_mean": 0.0,
        "rss_front_distance_noise_mean": 0.0,
        "rss_front_speed_noise_mean": 0.0,
        "rss_lateral_gap_noise_mean": 0.0,
        "rss_front_distance_raw_mean": 0.0,
        "rss_front_distance_attacked_mean": 0.0,
        "rss_front_distance_noisy_mean": 0.0,
        "rss_front_distance_used_mean": 0.0,
        "rss_lateral_gap_raw_mean": 0.0,
        "rss_lateral_gap_attacked_mean": 0.0,
        "rss_lateral_gap_noisy_mean": 0.0,
        "rss_lateral_gap_used_mean": 0.0,
        "rss_first_unsafe_step": -1,
        "rss_first_longitudinal_unsafe_step": -1,
        "rss_first_lateral_unsafe_step": -1,
        "rss_first_shield_step": -1,
        "rss_first_lateral_shield_step": -1,
        "rss_first_spring_soft_step": -1,
        "rss_first_attack_step": -1,
        "rss_first_front_attack_step": -1,
        "rss_first_lateral_attack_step": -1,
        "rss_first_uncertainty_step": -1,
        "rss_first_front_uncertainty_step": -1,
        "rss_first_lateral_uncertainty_step": -1,
        "_rss_attack_delta_sum": 0.0,
        "_rss_attack_delta_samples": 0,
        "_rss_lateral_attack_delta_sum": 0.0,
        "_rss_lateral_attack_delta_samples": 0,
        "_rss_front_distance_noise_sum": 0.0,
        "_rss_front_distance_noise_samples": 0,
        "_rss_front_speed_noise_sum": 0.0,
        "_rss_front_speed_noise_samples": 0,
        "_rss_lateral_gap_noise_sum": 0.0,
        "_rss_lateral_gap_noise_samples": 0,
        "_rss_front_distance_raw_sum": 0.0,
        "_rss_front_distance_raw_samples": 0,
        "_rss_front_distance_attacked_sum": 0.0,
        "_rss_front_distance_attacked_samples": 0,
        "_rss_front_distance_noisy_sum": 0.0,
        "_rss_front_distance_noisy_samples": 0,
        "_rss_front_distance_used_sum": 0.0,
        "_rss_front_distance_used_samples": 0,
        "_rss_lateral_gap_raw_sum": 0.0,
        "_rss_lateral_gap_raw_samples": 0,
        "_rss_lateral_gap_attacked_sum": 0.0,
        "_rss_lateral_gap_attacked_samples": 0,
        "_rss_lateral_gap_noisy_sum": 0.0,
        "_rss_lateral_gap_noisy_samples": 0,
        "_rss_lateral_gap_used_sum": 0.0,
        "_rss_lateral_gap_used_samples": 0,
    }


def update_rss_episode_stats(stats, info, step):
    if stats is None:
        return

    def mark(flag_name, count_name, first_name):
        if not info.get(flag_name, False):
            return
        stats[count_name] += 1
        if stats[first_name] < 0:
            stats[first_name] = step

    mark("rss_unsafe", "rss_unsafe_steps", "rss_first_unsafe_step")
    mark("rss_longitudinal_unsafe", "rss_longitudinal_unsafe_steps", "rss_first_longitudinal_unsafe_step")
    mark("rss_lateral_unsafe", "rss_lateral_unsafe_steps", "rss_first_lateral_unsafe_step")
    mark("rss_shield_active", "rss_shield_steps", "rss_first_shield_step")
    if info.get("rss_longitudinal_shield_active", False):
        stats["rss_longitudinal_shield_steps"] += 1
    mark("rss_lateral_shield_active", "rss_lateral_shield_steps", "rss_first_lateral_shield_step")
    mark("rss_spring_soft_active", "rss_spring_soft_steps", "rss_first_spring_soft_step")
    if info.get("rss_longitudinal_spring_soft_active", False):
        stats["rss_longitudinal_spring_soft_steps"] += 1
    if info.get("rss_lateral_spring_soft_active", False):
        stats["rss_lateral_spring_soft_steps"] += 1
    front_attack_active = info.get("rss_attack_active", False)
    lateral_attack_active = info.get("rss_lateral_attack_active", False)
    if front_attack_active or lateral_attack_active:
        stats["rss_attack_steps"] += 1
        if stats["rss_first_attack_step"] < 0:
            stats["rss_first_attack_step"] = step
    mark("rss_attack_active", "rss_front_attack_steps", "rss_first_front_attack_step")
    mark("rss_lateral_attack_active", "rss_lateral_attack_steps", "rss_first_lateral_attack_step")
    front_uncertainty_active = info.get("rss_uncertainty_active", False)
    lateral_uncertainty_active = info.get("rss_lateral_uncertainty_active", False)
    if front_uncertainty_active or lateral_uncertainty_active:
        stats["rss_uncertainty_steps"] += 1
        if stats["rss_first_uncertainty_step"] < 0:
            stats["rss_first_uncertainty_step"] = step
    mark("rss_uncertainty_active", "rss_front_uncertainty_steps", "rss_first_front_uncertainty_step")
    mark("rss_lateral_uncertainty_active", "rss_lateral_uncertainty_steps", "rss_first_lateral_uncertainty_step")

    raw_distance = _finite_info_value(info.get("rss_front_distance_raw"))
    attacked_distance = _finite_info_value(info.get("rss_front_distance_attacked"))
    noisy_distance = _finite_info_value(info.get("rss_front_distance_noisy"))
    used_distance = _finite_info_value(info.get("rss_front_distance_used"))
    if raw_distance is not None:
        stats["_rss_front_distance_raw_samples"] += 1
        stats["_rss_front_distance_raw_sum"] += raw_distance
        stats["rss_front_distance_raw_mean"] = (
            stats["_rss_front_distance_raw_sum"] / stats["_rss_front_distance_raw_samples"]
        )
    if attacked_distance is not None:
        stats["_rss_front_distance_attacked_samples"] += 1
        stats["_rss_front_distance_attacked_sum"] += attacked_distance
        stats["rss_front_distance_attacked_mean"] = (
            stats["_rss_front_distance_attacked_sum"] / stats["_rss_front_distance_attacked_samples"]
        )
    if noisy_distance is not None:
        stats["_rss_front_distance_noisy_samples"] += 1
        stats["_rss_front_distance_noisy_sum"] += noisy_distance
        stats["rss_front_distance_noisy_mean"] = (
            stats["_rss_front_distance_noisy_sum"] / stats["_rss_front_distance_noisy_samples"]
        )
    if used_distance is not None:
        stats["_rss_front_distance_used_samples"] += 1
        stats["_rss_front_distance_used_sum"] += used_distance
        stats["rss_front_distance_used_mean"] = (
            stats["_rss_front_distance_used_sum"] / stats["_rss_front_distance_used_samples"]
        )

    if front_attack_active:
        attack_delta = _finite_info_value(info.get("rss_attack_delta"))
        if attack_delta is not None:
            stats["_rss_attack_delta_samples"] += 1
            stats["_rss_attack_delta_sum"] += attack_delta
            stats["rss_attack_delta_mean"] = stats["_rss_attack_delta_sum"] / stats["_rss_attack_delta_samples"]

    if front_uncertainty_active:
        distance_noise = _finite_info_value(info.get("rss_front_distance_noise"))
        if distance_noise is not None:
            stats["_rss_front_distance_noise_samples"] += 1
            stats["_rss_front_distance_noise_sum"] += distance_noise
            stats["rss_front_distance_noise_mean"] = (
                stats["_rss_front_distance_noise_sum"] / stats["_rss_front_distance_noise_samples"]
            )
        speed_noise = _finite_info_value(info.get("rss_front_speed_noise"))
        if speed_noise is not None:
            stats["_rss_front_speed_noise_samples"] += 1
            stats["_rss_front_speed_noise_sum"] += speed_noise
            stats["rss_front_speed_noise_mean"] = (
                stats["_rss_front_speed_noise_sum"] / stats["_rss_front_speed_noise_samples"]
            )

    raw_lateral_gap = _finite_info_value(info.get("rss_lateral_gap_raw"))
    attacked_lateral_gap = _finite_info_value(info.get("rss_lateral_gap_attacked"))
    noisy_lateral_gap = _finite_info_value(info.get("rss_lateral_gap_noisy"))
    used_lateral_gap = _finite_info_value(info.get("rss_lateral_gap_used"))
    if raw_lateral_gap is not None:
        stats["_rss_lateral_gap_raw_samples"] += 1
        stats["_rss_lateral_gap_raw_sum"] += raw_lateral_gap
        stats["rss_lateral_gap_raw_mean"] = (
            stats["_rss_lateral_gap_raw_sum"] / stats["_rss_lateral_gap_raw_samples"]
        )
    if attacked_lateral_gap is not None:
        stats["_rss_lateral_gap_attacked_samples"] += 1
        stats["_rss_lateral_gap_attacked_sum"] += attacked_lateral_gap
        stats["rss_lateral_gap_attacked_mean"] = (
            stats["_rss_lateral_gap_attacked_sum"] / stats["_rss_lateral_gap_attacked_samples"]
        )
    if noisy_lateral_gap is not None:
        stats["_rss_lateral_gap_noisy_samples"] += 1
        stats["_rss_lateral_gap_noisy_sum"] += noisy_lateral_gap
        stats["rss_lateral_gap_noisy_mean"] = (
            stats["_rss_lateral_gap_noisy_sum"] / stats["_rss_lateral_gap_noisy_samples"]
        )
    if used_lateral_gap is not None:
        stats["_rss_lateral_gap_used_samples"] += 1
        stats["_rss_lateral_gap_used_sum"] += used_lateral_gap
        stats["rss_lateral_gap_used_mean"] = (
            stats["_rss_lateral_gap_used_sum"] / stats["_rss_lateral_gap_used_samples"]
        )

    if lateral_attack_active:
        lateral_attack_delta = _finite_info_value(info.get("rss_lateral_attack_delta"))
        if lateral_attack_delta is not None:
            stats["_rss_lateral_attack_delta_samples"] += 1
            stats["_rss_lateral_attack_delta_sum"] += lateral_attack_delta
            stats["rss_lateral_attack_delta_mean"] = (
                stats["_rss_lateral_attack_delta_sum"] / stats["_rss_lateral_attack_delta_samples"]
            )

    if lateral_uncertainty_active:
        lateral_noise = _finite_info_value(info.get("rss_lateral_gap_noise"))
        if lateral_noise is not None:
            stats["_rss_lateral_gap_noise_samples"] += 1
            stats["_rss_lateral_gap_noise_sum"] += lateral_noise
            stats["rss_lateral_gap_noise_mean"] = (
                stats["_rss_lateral_gap_noise_sum"] / stats["_rss_lateral_gap_noise_samples"]
            )


def _finite_info_value(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def finalize_rss_episode_stats(stats):
    if stats is None:
        return None
    return {key: value for key, value in stats.items() if not key.startswith("_rss_")}


def make_episode_event_stats():
    return {
        "crash": False,
        "crash_vehicle": False,
        "out_of_road": False,
        "crash_object": False,
        "crash_building": False,
        "crash_sidewalk": False,
        "first_crash_step": -1,
        "first_crash_vehicle_step": -1,
        "first_out_of_road_step": -1,
        "crash_vehicle_steps": 0,
        "out_of_road_steps": 0,
    }


def update_episode_event_stats(stats, info, step):
    def mark(flag_name, first_name=None, count_name=None):
        if not info.get(flag_name, False):
            return
        stats[flag_name] = True
        if first_name is not None and stats[first_name] < 0:
            stats[first_name] = step
        if count_name is not None:
            stats[count_name] += 1

    mark("crash", "first_crash_step")
    mark("crash_vehicle", "first_crash_vehicle_step", "crash_vehicle_steps")
    mark("out_of_road", "first_out_of_road_step", "out_of_road_steps")
    mark("crash_object")
    mark("crash_building")
    mark("crash_sidewalk")


def evaluate_ppl_once(
    ckpt_path,
    ckpt_index,
    folder_name,
    use_render=False,
    num_ep_in_one_env=5,
    total_env_num=50,
    deterministic=True,
    rss_observer=None,
    rss_shield=False,
):
    """
    Evaluate one PPL checkpoint on `total_env_num` environments,
    each run for `num_ep_in_one_env` episodes.

    Args:
        ckpt_path:        Directory that contains "rl_model_<ckpt_index>_steps.zip",
                          OR a full path to a .zip file.
        ckpt_index:       Checkpoint step index (used to locate the zip file and
                          as a label in the result CSV). Ignored if ckpt_path already
                          ends with ".zip".
        folder_name:      Directory to save result CSV files.
        use_render:       Whether to render the environment.
        num_ep_in_one_env: Number of episodes per environment seed.
        total_env_num:    Number of different environment seeds to evaluate.
        deterministic:    Whether the policy acts deterministically.

    Returns:
        pd.DataFrame with per-episode results, or None on failure.
    """
    # Resolve the zip path
    if ckpt_path.endswith(".zip"):
        zip_path = ckpt_path
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    else:
        zip_path = osp.join(ckpt_path, "rl_model_{}_steps.zip".format(ckpt_index))
        ckpt_name = "checkpoint_{}".format(ckpt_index)

    if not osp.exists(zip_path):
        print("=====\nCheckpoint not found: {}\n=====".format(zip_path))
        return None

    os.makedirs(folder_name, exist_ok=True)

    env = make_metadrive_env(use_render, direct_action_policy=rss_shield)
    try:
        policy_function = PolicyFunction(zip_path, env)
    except Exception as e:
        print(f"Failed to load policy from {zip_path}: {e}")
        env.close()
        return None

    saved_results = []
    ep_velocities = []

    try:
        start = time.time()
        last_time = time.time()
        ep_count = 0
        step_count = 0
        ep_times = []

        env_index = 0
        num_ep_in = 0
        o = reset_eval_env(env, EVAL_ENV_START + env_index)
        episode_event_stats = make_episode_event_stats()
        rss_episode_stats = make_rss_episode_stats() if rss_observer is not None else None
        if rss_observer is not None:
            rss_observer.reset()

        while True:
            action = policy_function(o, deterministic=deterministic)[0]
            rss_shield_info = {}
            if rss_shield and rss_observer is not None:
                action, rss_shield_info = rss_observer.shield_action(
                    env=env,
                    action=action,
                    env_seed=EVAL_ENV_START + env_index,
                    episode=ep_count + 1,
                    step=step_count + 1,
                )
            o, r, d, info = env.step(action)
            step_count += 1
            if rss_shield and rss_observer is not None:
                rss_shield_info.update(
                    rss_observer.verify_shield_step(
                        env=env,
                        shield_info=rss_shield_info,
                        env_seed=EVAL_ENV_START + env_index,
                        episode=ep_count + 1,
                        step=step_count,
                    )
                )
            if rss_observer is not None:
                rss_info = rss_observer.observe(
                    env=env,
                    env_seed=EVAL_ENV_START + env_index,
                    episode=ep_count + 1,
                    step=step_count,
                )
                rss_info.update(rss_shield_info)
                info.update(rss_info)
                update_rss_episode_stats(rss_episode_stats, info, step_count)
            update_episode_event_stats(episode_event_stats, info, step_count)

            if info:
                ep_velocities.append(info.get("velocity", 0))

            if use_render:
                env.render()

            if d or step_count >= 3000:
                ep_times.append(time.time() - last_time)
                last_time = time.time()

                ep_count += 1
                num_ep_in += 1

                env_id_recorded = EVAL_ENV_START + env_index
                num_ep_in_recorded = num_ep_in

                # Collect comprehensive results from both RecorderEnv and raw info
                res = env.get_episode_result()
                res.update(dict(
                    success=info.get("arrive_dest", 0),
                    crash=episode_event_stats["crash"],
                    crash_vehicle=episode_event_stats["crash_vehicle"],
                    out_of_road=episode_event_stats["out_of_road"],
                    crash_object=episode_event_stats["crash_object"],
                    crash_building=episode_event_stats["crash_building"],
                    crash_sidewalk=episode_event_stats["crash_sidewalk"],
                    first_crash_step=episode_event_stats["first_crash_step"],
                    first_crash_vehicle_step=episode_event_stats["first_crash_vehicle_step"],
                    first_out_of_road_step=episode_event_stats["first_out_of_road_step"],
                    crash_vehicle_steps=episode_event_stats["crash_vehicle_steps"],
                    out_of_road_steps=episode_event_stats["out_of_road_steps"],
                    route_completion=info.get("route_completion", 0),
                    velocity_step_mean=np.mean(ep_velocities) if ep_velocities else 0,
                ))
                if rss_episode_stats is not None:
                    res.update(finalize_rss_episode_stats(rss_episode_stats))
                ep_velocities = []

                res["episode"] = ep_count
                res["ckpt_index"] = ckpt_index
                res["env_id"] = env_id_recorded
                res["num_ep_in_one_env"] = num_ep_in_recorded

                saved_results.append(res)
                df = pd.DataFrame(saved_results)

                print(
                    "Env {:3d} | ep_in_env {:2d} | total_ep {:4d} | steps {:5d} | "
                    "ep_time {:.2f}s | total_time {:.2f}s | ckpt: {}".format(
                        env_index, num_ep_in, ep_count, step_count,
                        ep_times[-1], time.time() - start, ckpt_name
                    )
                )
                console_res = {key: value for key, value in res.items() if not key.startswith("rss_")}
                print(pretty_print(console_res))

                # Backup CSV
                tmp_path = osp.join(folder_name, "{}_tmp.csv".format(ckpt_name))
                df.to_csv(tmp_path)

                step_count = 0

                # Advance to next env seed if enough episodes in this one
                if num_ep_in >= num_ep_in_one_env:
                    env_index += 1
                    num_ep_in = 0
                    if env_index >= total_env_num:
                        break

                o = reset_eval_env(env, EVAL_ENV_START + env_index)
                episode_event_stats = make_episode_event_stats()
                rss_episode_stats = make_rss_episode_stats() if rss_observer is not None else None
                if rss_observer is not None:
                    rss_observer.reset()

    except Exception as e:
        raise e
    finally:
        env.close()

    df = pd.DataFrame(saved_results)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Compute key rates
    success_rate = df["success"].mean() * 100 if "success" in df.columns else 0
    crash_rate = df["crash"].mean() * 100 if "crash" in df.columns else 0
    vehicle_crash_rate = df["crash_vehicle"].mean() * 100 if "crash_vehicle" in df.columns else 0
    out_of_road_rate = df["out_of_road"].mean() * 100 if "out_of_road" in df.columns else 0
    avg_route_completion = df["route_completion"].mean() * 100 if "route_completion" in df.columns else 0
    avg_reward = df["episode_reward"].mean() if "episode_reward" in df.columns else 0
    avg_cost = df["episode_cost"].mean() if "episode_cost" in df.columns else 0
    avg_length = df["episode_length"].mean() if "episode_length" in df.columns else 0
    avg_velocity = df["velocity_step_mean"].mean() if "velocity_step_mean" in df.columns else 0

    print("\n" + "=" * 60)
    print("  Evaluation Summary: {}".format(ckpt_name))
    print("=" * 60)
    print("  Episodes:          {}".format(len(df)))
    print("  Success Rate:      {:.1f}%".format(success_rate))
    print("  Crash Rate:        {:.1f}%".format(crash_rate))
    print("  Vehicle Crash Rate:{:.1f}%".format(vehicle_crash_rate))
    print("  Out of Road Rate:  {:.1f}%".format(out_of_road_rate))
    print("  Route Completion:  {:.1f}%".format(avg_route_completion))
    print("  Avg Reward:        {:.2f}".format(avg_reward))
    print("  Avg Cost:          {:.2f}".format(avg_cost))
    print("  Avg Episode Len:   {:.1f}".format(avg_length))
    print("  Avg Velocity:      {:.2f}".format(avg_velocity))
    print("=" * 60 + "\n")

    final_path = osp.join(folder_name, "{}.csv".format(ckpt_name))
    df.to_csv(final_path)
    print("Final results saved to: {}".format(final_path))

    df["model_name"] = ckpt_name
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPL MetaDrive checkpoints.")

    # --- Checkpoint location ---
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help=(
            "Directory containing rl_model_<step>_steps.zip files "
            "(e.g. runs/PPL/PPL_xxxx/models), OR a single .zip file path."
        ),
    )
    parser.add_argument(
        "--ckpt_index",
        type=int,
        default=-1,
        help=(
            "Single checkpoint step to evaluate. "
            "If set, only this checkpoint is evaluated. "
            "Ignored if --path already points to a .zip file."
        ),
    )

    # --- Batch evaluation ---
    parser.add_argument("--start_ckpt", type=int, default=-1, help="First checkpoint step for batch evaluation.")
    parser.add_argument("--num_ckpt", type=int, default=10, help="Number of checkpoints to evaluate.")
    parser.add_argument("--skip", type=int, default=150, help="Step interval between checkpoints (= save_freq).")

    # --- Output ---
    parser.add_argument(
        "--ret_save_folder",
        type=str,
        default="evaluate_results/ppl",
        help="Folder to save evaluation result CSV files.",
    )

    # --- Eval settings ---
    parser.add_argument("--use_render", action="store_true", help="Enable rendering.")
    parser.add_argument("--num_ep_in_one_env", type=int, default=1, help="Episodes per environment seed.")
    parser.add_argument("--total_env_num", type=int, default=50, help="Number of environment seeds.")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy during evaluation (default: deterministic).",
    )
    parser.add_argument("--rss_observe", action="store_true", help="Enable RSS observation and CSV counters.")
    parser.add_argument("--rss_shield", action="store_true", help="Enable longitudinal and lateral RSS shield.")
    parser.add_argument("--rss_response_time", type=float, default=1.0)
    parser.add_argument("--rss_ego_max_accel", type=float, default=2.0)
    parser.add_argument("--rss_ego_min_brake", type=float, default=4.0)
    parser.add_argument("--rss_front_max_brake", type=float, default=6.0)
    parser.add_argument("--rss_disable_auto_dynamics", action="store_true")
    parser.add_argument(
        "--rss_dynamics_mode",
        choices=["manual", "idm", "standard", "measured", "vehicle"],
        default="standard",
    )
    parser.add_argument("--rss_lateral_threshold", type=float, default=2.0)
    parser.add_argument("--rss_max_front_distance", type=float, default=100.0)
    parser.add_argument("--rss_lateral_max_accel", type=float, default=0.8)
    parser.add_argument("--rss_lateral_min_brake", type=float, default=1.5)
    parser.add_argument("--rss_lateral_safe_margin", type=float, default=1.5)
    parser.add_argument("--rss_lateral_longitudinal_threshold", type=float, default=8.0)
    parser.add_argument("--rss_max_lateral_distance", type=float, default=8.0)
    parser.add_argument("--rss_log_every_step", action="store_true")
    parser.add_argument("--rss_debug_interval", type=int, default=0)
    parser.add_argument(
        "--rss_shield_mode",
        choices=["standard", "spring_damper"],
        default="standard",
        help="Shield response mode. 'standard' keeps the original hard RSS behavior.",
    )
    parser.add_argument("--rss_shield_brake", type=float, default=1.0)
    parser.add_argument("--rss_lateral_shield_brake", type=float, default=0.5)
    parser.add_argument("--rss_lateral_steering_scale", type=float, default=0.25)
    parser.add_argument("--rss_spring_longitudinal_buffer", type=float, default=2.0)
    parser.add_argument("--rss_spring_longitudinal_k", type=float, default=0.08)
    parser.add_argument("--rss_damper_longitudinal_k", type=float, default=0.015)
    parser.add_argument("--rss_spring_lateral_buffer", type=float, default=0.4)
    parser.add_argument("--rss_spring_lateral_k", type=float, default=0.05)
    parser.add_argument("--rss_damper_lateral_k", type=float, default=0.02)
    parser.add_argument("--rss_spring_lateral_max_steer", type=float, default=0.04)
    parser.add_argument(
        "--rss_attack",
        choices=["none", "front_range_overestimate", "front_object_removal", "lateral_gap_overestimate"],
        default="none",
        help="Object-level attack applied to RSS monitor inputs.",
    )
    parser.add_argument(
        "--rss_attack_front_distance_delta",
        type=float,
        default=0.0,
        help="Distance added to the selected front vehicle for front_range_overestimate attack, in meters.",
    )
    parser.add_argument(
        "--rss_attack_lateral_gap_delta",
        type=float,
        default=0.0,
        help="Gap added to the selected lateral vehicle for lateral_gap_overestimate attack, in meters.",
    )
    parser.add_argument(
        "--rss_uncertainty",
        choices=["none", "gaussian"],
        default="none",
        help="Perception uncertainty model applied to RSS monitor inputs.",
    )
    parser.add_argument(
        "--rss_noise_level",
        choices=["custom", "small", "medium", "large"],
        default="custom",
        help="Preset Gaussian noise level. small/medium are MetaDrive-scaled; large matches the paper stress setting.",
    )
    parser.add_argument(
        "--rss_noise_front_distance_sigma",
        type=float,
        default=0.0,
        help="Backward-compatible alias for longitudinal position sigma in meters.",
    )
    parser.add_argument(
        "--rss_noise_lateral_gap_sigma",
        type=float,
        default=0.0,
        help="Backward-compatible alias for lateral position sigma in meters.",
    )
    parser.add_argument(
        "--rss_noise_position_x_sigma",
        type=float,
        default=None,
        help="Gaussian sigma for perceived longitudinal position noise, in meters.",
    )
    parser.add_argument(
        "--rss_noise_position_y_sigma",
        type=float,
        default=None,
        help="Gaussian sigma for perceived lateral position noise, in meters.",
    )
    parser.add_argument(
        "--rss_noise_speed_sigma",
        type=float,
        default=0.0,
        help="Gaussian sigma for perceived speed noise, in m/s.",
    )
    parser.add_argument(
        "--rss_noise_heading_sigma",
        type=float,
        default=0.0,
        help="Gaussian sigma for perceived heading noise, in radians.",
    )
    parser.add_argument(
        "--rss_noise_seed",
        type=int,
        default=0,
        help="Random seed for RSS perception noise. Set negative for non-deterministic RandomState.",
    )
    parser.add_argument("--rss_verbose", action="store_true", help="Print RSS trigger/shield logs to terminal.")

    args = parser.parse_args()

    deterministic = not args.stochastic
    rss_dynamics_mode = "manual" if args.rss_disable_auto_dynamics else args.rss_dynamics_mode
    rss_attack = build_rss_attack(
        name=args.rss_attack,
        front_distance_delta=args.rss_attack_front_distance_delta,
        lateral_gap_delta=args.rss_attack_lateral_gap_delta,
    )
    rss_uncertainty = build_rss_uncertainty(
        name=args.rss_uncertainty,
        noise_level=args.rss_noise_level,
        front_distance_sigma=args.rss_noise_front_distance_sigma,
        lateral_gap_sigma=args.rss_noise_lateral_gap_sigma,
        position_x_sigma=args.rss_noise_position_x_sigma,
        position_y_sigma=args.rss_noise_position_y_sigma,
        speed_sigma=args.rss_noise_speed_sigma,
        heading_sigma=args.rss_noise_heading_sigma,
        seed=args.rss_noise_seed,
    )
    rss_observer = RSSObserver(
        response_time=args.rss_response_time,
        ego_max_accel=args.rss_ego_max_accel,
        ego_min_brake=args.rss_ego_min_brake,
        front_max_brake=args.rss_front_max_brake,
        dynamics_mode=rss_dynamics_mode,
        lateral_threshold=args.rss_lateral_threshold,
        max_front_distance=args.rss_max_front_distance,
        lateral_max_accel=args.rss_lateral_max_accel,
        lateral_min_brake=args.rss_lateral_min_brake,
        lateral_safe_margin=args.rss_lateral_safe_margin,
        lateral_longitudinal_threshold=args.rss_lateral_longitudinal_threshold,
        max_lateral_distance=args.rss_max_lateral_distance,
        log_every_step=args.rss_log_every_step,
        debug_interval=args.rss_debug_interval,
        shield_mode=args.rss_shield_mode,
        shield_brake=args.rss_shield_brake,
        lateral_shield_brake=args.rss_lateral_shield_brake,
        lateral_steering_scale=args.rss_lateral_steering_scale,
        spring_longitudinal_buffer=args.rss_spring_longitudinal_buffer,
        spring_longitudinal_k=args.rss_spring_longitudinal_k,
        damper_longitudinal_k=args.rss_damper_longitudinal_k,
        spring_lateral_buffer=args.rss_spring_lateral_buffer,
        spring_lateral_k=args.rss_spring_lateral_k,
        damper_lateral_k=args.rss_damper_lateral_k,
        spring_lateral_max_steer=args.rss_spring_lateral_max_steer,
        attack=rss_attack,
        uncertainty=rss_uncertainty,
        verbose=args.rss_verbose,
    ) if args.rss_observe or args.rss_shield else None
    if rss_observer is not None and args.rss_verbose:
        print(
            "[RSS] observer enabled: response_time={} front_max_distance={} "
            "dynamics_mode={} shield={} shield_mode={} lat_long_threshold={} "
            "max_lat_distance={} debug_interval={} attack={} attack_front_delta={} "
            "attack_lateral_gap_delta={} uncertainty={} noise_level={} noise_x_sigma={} "
            "noise_y_sigma={} noise_v_sigma={} noise_theta_sigma={} noise_seed={}".format(
                args.rss_response_time,
                args.rss_max_front_distance,
                rss_dynamics_mode,
                args.rss_shield,
                args.rss_shield_mode,
                args.rss_lateral_longitudinal_threshold,
                args.rss_max_lateral_distance,
                args.rss_debug_interval,
                args.rss_attack,
                args.rss_attack_front_distance_delta,
                args.rss_attack_lateral_gap_delta,
                args.rss_uncertainty,
                args.rss_noise_level,
                args.rss_noise_position_x_sigma
                if args.rss_noise_position_x_sigma is not None
                else args.rss_noise_front_distance_sigma,
                args.rss_noise_position_y_sigma
                if args.rss_noise_position_y_sigma is not None
                else args.rss_noise_lateral_gap_sigma,
                args.rss_noise_speed_sigma,
                args.rss_noise_heading_sigma,
                args.rss_noise_seed,
            )
        )

    # --- Decide evaluation mode ---
    if args.path.endswith(".zip"):
        # Single explicit zip path
        print("===== Evaluating single checkpoint (zip): {} =====".format(args.path))
        ret = evaluate_ppl_once(
            ckpt_path=args.path,
            ckpt_index=0,
            folder_name=args.ret_save_folder,
            use_render=args.use_render,
            num_ep_in_one_env=args.num_ep_in_one_env,
            total_env_num=args.total_env_num,
            deterministic=deterministic,
            rss_observer=rss_observer,
            rss_shield=args.rss_shield,
        )

    elif args.ckpt_index >= 0:
        # Single checkpoint by step index
        if not args.path:
            parser.error("--path is required when using --ckpt_index")
        print("===== Evaluating checkpoint {} in {} =====".format(args.ckpt_index, args.path))
        ret = evaluate_ppl_once(
            ckpt_path=args.path,
            ckpt_index=args.ckpt_index,
            folder_name=args.ret_save_folder,
            use_render=args.use_render,
            num_ep_in_one_env=args.num_ep_in_one_env,
            total_env_num=args.total_env_num,
            deterministic=deterministic,
            rss_observer=rss_observer,
            rss_shield=args.rss_shield,
        )

    elif args.start_ckpt >= 0:
        # Batch evaluation over a range of checkpoints
        if not args.path:
            parser.error("--path is required when using --start_ckpt")
        all_results = []
        ckpt_indices = list(reversed(range(args.start_ckpt, args.start_ckpt + args.num_ckpt * args.skip, args.skip)))
        print("===== Batch evaluation: {} checkpoints =====".format(len(ckpt_indices)))
        for ckpt_index in ckpt_indices:
            print("\n----- Checkpoint {} -----".format(ckpt_index))
            ret = evaluate_ppl_once(
                ckpt_path=args.path,
                ckpt_index=ckpt_index,
                folder_name=args.ret_save_folder,
                use_render=args.use_render,
                num_ep_in_one_env=args.num_ep_in_one_env,
                total_env_num=args.total_env_num,
                deterministic=deterministic,
                rss_observer=rss_observer,
                rss_shield=args.rss_shield,
            )
            if ret is not None:
                all_results.append(ret)
            else:
                print("Skipped checkpoint {}.".format(ckpt_index))

        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            summary_path = osp.join(args.ret_save_folder, "all_checkpoints_summary.csv")
            combined.to_csv(summary_path)
            print("\n===== All checkpoints summary saved to: {} =====".format(summary_path))
        ret = None  # suppress final print

    else:
        parser.error(
            "Please specify one of: --ckpt_index, --start_ckpt, "
            "or provide a direct .zip path via --path."
        )

    if ret is None and args.ckpt_index < 0 and not args.path.endswith(".zip") and args.start_ckpt < 0:
        print("Evaluation failed.")
    elif ret is not None:
        print("\n\n Evaluation finished successfully.\n")
