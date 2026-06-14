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
import os
import os.path as osp
import time

import numpy as np
import pandas as pd

from ppl.experiments.metadrive.driving_env import DrivingEnv
from ppl.ppl import PPL
from ppl.sb3.td3.policies import TD3Policy
from ppl.utils.metadrive_route_projection import MetaDriveRouteProjector
from ppl.utils.print_dict_utils import pretty_print, RecorderEnv

EVAL_ENV_START = 1000  # Evaluation seeds start from 1000


class RSSObserver:
    """Observer-only longitudinal/lateral RSS checker for runtime evaluation."""

    def __init__(
        self,
        response_time=1.0,
        ego_max_accel=2.0,
        ego_min_brake=4.0,
        front_max_brake=6.0,
        dynamics_mode="measured",
        lateral_threshold=2.0,
        max_front_distance=100.0,
        lateral_max_accel=0.8,
        lateral_min_brake=1.5,
        lateral_safe_margin=0.2,
        lateral_longitudinal_threshold=5.0,
        max_lateral_distance=8.0,
        log_every_step=False,
        debug_interval=0,
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
        self.was_longitudinal_unsafe = False
        self.was_lateral_unsafe = False

    def reset(self):
        self.was_longitudinal_unsafe = False
        self.was_lateral_unsafe = False

    def observe(self, env, env_seed, episode, step):
        raw_env = self._unwrap(env)
        ego = getattr(raw_env, "vehicle", None)
        if ego is None or getattr(ego, "navigation", None) is None:
            return {"rss_unsafe": False}

        try:
            projector = MetaDriveRouteProjector.from_vehicle(ego)
            ego_proj = projector.project_vehicle(ego)
        except Exception as error:
            if self.log_every_step:
                print(f"[RSS] skipped: {error}")
            return {"rss_unsafe": False}

        if ego_proj is None:
            return {"rss_unsafe": False}

        ego_ref = self._reference_coordinates(projector, ego.position, ego_proj.road_key)
        front = None
        lateral = None
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

            proj = projector.project_vehicle(vehicle, use_closest_lane=True)
            if proj is None:
                continue
            stats["route_vehicles"] += 1

            if ego_ref is not None and proj.road_key == ego_proj.road_key and proj.lane_index[2] != ego_proj.lane_index[2]:
                stats["lateral_lane_candidates"] += 1
                candidate = self._lateral_candidate(
                    projector=projector,
                    ego=ego,
                    ego_ref=ego_ref,
                    vehicle=vehicle,
                    proj=proj,
                )
                if candidate == "longitudinal_gap":
                    stats["lateral_long_filtered"] += 1
                elif candidate == "lateral_distance":
                    stats["lateral_far_filtered"] += 1
                elif candidate is not None:
                    stats["lateral_candidates"] += 1
                    lateral = self._select_lateral_candidate(lateral, candidate)

            if proj.lane_index[2] != ego_proj.lane_index[2]:
                continue

            lateral_gap = abs(proj.local_d - ego_proj.local_d)
            if lateral_gap > self.lateral_threshold:
                continue

            distance = proj.route_s - ego_proj.route_s - ego.LENGTH / 2.0 - vehicle.LENGTH / 2.0
            if distance <= 0 or distance > self.max_front_distance:
                continue

            stats["front_candidates"] += 1
            if front is None or distance < front["distance"]:
                front = {"vehicle": vehicle, "proj": proj, "distance": distance}

        longitudinal_info = self._observe_longitudinal(front, ego, ego_proj, env_seed, episode, step)
        lateral_info = self._observe_lateral(lateral, env_seed, episode, step)
        self._debug(env_seed, episode, step, stats, front, lateral, longitudinal_info, lateral_info)
        return {
            "rss_unsafe": longitudinal_info["rss_longitudinal_unsafe"] or lateral_info["rss_lateral_unsafe"],
            **longitudinal_info,
            **lateral_info,
        }

    def _observe_longitudinal(self, front, ego, ego_proj, env_seed, episode, step):
        if front is None:
            if self.was_longitudinal_unsafe:
                print(
                    "[RSS CLEAR] seed={} episode={} step={} no_front_vehicle".format(
                        env_seed, episode, step
                    )
                )
            self.was_longitudinal_unsafe = False
            return {"rss_longitudinal_unsafe": False}

        front_vehicle = front["vehicle"]
        front_proj = front["proj"]
        front_distance = front["distance"]
        ego_max_accel, ego_min_brake, front_max_brake = self._longitudinal_dynamics(ego, front_vehicle)
        safe_distance = self.safe_longitudinal_distance(
            ego.speed,
            front_vehicle.speed,
            ego_max_accel=ego_max_accel,
            ego_min_brake=ego_min_brake,
            front_max_brake=front_max_brake,
        )
        unsafe = front_distance < safe_distance

        if unsafe and (not self.was_longitudinal_unsafe or self.log_every_step):
            print(
                "[RSS TRIGGER] seed={} episode={} step={} "
                "distance={:.2f} safe_distance={:.2f} "
                "ego_acc={:.2f} ego_brake={:.2f} front_brake={:.2f} "
                "ego_v={:.2f} front_v={:.2f} "
                "ego_s={:.2f} front_s={:.2f} "
                "front={}".format(
                    env_seed,
                    episode,
                    step,
                    front_distance,
                    safe_distance,
                    ego_max_accel,
                    ego_min_brake,
                    front_max_brake,
                    ego.speed,
                    front_vehicle.speed,
                    ego_proj.route_s,
                    front_proj.route_s,
                    getattr(front_vehicle, "name", getattr(front_vehicle, "id", "unknown")),
                )
            )
        elif not unsafe and self.was_longitudinal_unsafe:
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
        }

    def _observe_lateral(self, lateral, env_seed, episode, step):
        if lateral is None:
            if self.was_lateral_unsafe:
                print(
                    "[RSS LAT CLEAR] seed={} episode={} step={} no_nearby_vehicle".format(
                        env_seed, episode, step
                    )
                )
            self.was_lateral_unsafe = False
            return {"rss_lateral_unsafe": False}

        unsafe = lateral["unsafe"]
        vehicle = lateral["vehicle"]
        if unsafe and (not self.was_lateral_unsafe or self.log_every_step):
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
        elif not unsafe and self.was_lateral_unsafe:
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
        }

    def _lateral_candidate(self, projector, ego, ego_ref, vehicle, proj):
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

        candidate = {
            "vehicle": vehicle,
            "unsafe": unsafe,
            "violation": violation,
            "lateral_gap": float(lateral_gap),
            "safe_lateral_distance": safe_lateral_distance,
            "longitudinal_gap": longitudinal_gap,
            "ego_d": ego_d,
            "other_d": other_d,
            "ego_lateral_speed": ego_lateral_speed,
            "other_lateral_speed": other_lateral_speed,
        }
        return candidate

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
    def _vehicles(env):
        traffic_manager = getattr(env.engine, "traffic_manager", None)
        if traffic_manager is not None and hasattr(traffic_manager, "vehicles"):
            vehicles = traffic_manager.vehicles
            return vehicles.values() if isinstance(vehicles, dict) else vehicles
        objects = env.engine.get_objects()
        return objects.values() if isinstance(objects, dict) else objects


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


def make_metadrive_env(use_render=False):
    """Build the evaluation environment (no manual control, fixed seed range)."""
    config = dict(
        use_render=use_render,
        manual_control=False,
        start_seed=EVAL_ENV_START,
        horizon=1500,
    )
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


def evaluate_ppl_once(
    ckpt_path,
    ckpt_index,
    folder_name,
    use_render=False,
    num_ep_in_one_env=5,
    total_env_num=50,
    deterministic=True,
    rss_observer=None,
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

    env = make_metadrive_env(use_render)
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
        if rss_observer is not None:
            rss_observer.reset()

        while True:
            action = policy_function(o, deterministic=deterministic)[0]
            o, r, d, info = env.step(action)
            step_count += 1
            if rss_observer is not None:
                rss_info = rss_observer.observe(
                    env=env,
                    env_seed=EVAL_ENV_START + env_index,
                    episode=ep_count + 1,
                    step=step_count,
                )
                info.update(rss_info)

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
                    crash=info.get("crash", False),
                    crash_vehicle=info.get("crash_vehicle", False),
                    out_of_road=info.get("out_of_road", False),
                    route_completion=info.get("route_completion", 0),
                    velocity_step_mean=np.mean(ep_velocities) if ep_velocities else 0,
                ))
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
                print(pretty_print(res))

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
    parser.add_argument("--rss_observe", action="store_true", help="Print observer-only RSS trigger logs.")
    parser.add_argument("--rss_response_time", type=float, default=1.0)
    parser.add_argument("--rss_ego_max_accel", type=float, default=2.0)
    parser.add_argument("--rss_ego_min_brake", type=float, default=4.0)
    parser.add_argument("--rss_front_max_brake", type=float, default=6.0)
    parser.add_argument("--rss_disable_auto_dynamics", action="store_true")
    parser.add_argument("--rss_dynamics_mode", choices=["manual", "idm", "measured", "vehicle"], default="measured")
    parser.add_argument("--rss_lateral_threshold", type=float, default=2.0)
    parser.add_argument("--rss_max_front_distance", type=float, default=100.0)
    parser.add_argument("--rss_lateral_max_accel", type=float, default=0.8)
    parser.add_argument("--rss_lateral_min_brake", type=float, default=1.5)
    parser.add_argument("--rss_lateral_safe_margin", type=float, default=0.2)
    parser.add_argument("--rss_lateral_longitudinal_threshold", type=float, default=5.0)
    parser.add_argument("--rss_max_lateral_distance", type=float, default=8.0)
    parser.add_argument("--rss_log_every_step", action="store_true")
    parser.add_argument("--rss_debug_interval", type=int, default=0)

    args = parser.parse_args()

    deterministic = not args.stochastic
    rss_dynamics_mode = "manual" if args.rss_disable_auto_dynamics else args.rss_dynamics_mode
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
    ) if args.rss_observe else None
    if rss_observer is not None:
        print(
            "[RSS] observer enabled: response_time={} front_max_distance={} "
            "dynamics_mode={} lat_long_threshold={} max_lat_distance={} debug_interval={}".format(
                args.rss_response_time,
                args.rss_max_front_distance,
                rss_dynamics_mode,
                args.rss_lateral_longitudinal_threshold,
                args.rss_max_lateral_distance,
                args.rss_debug_interval,
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
