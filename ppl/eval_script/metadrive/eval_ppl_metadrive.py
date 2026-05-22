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
import traceback

import numpy as np
import pandas as pd

from ppl.experiments.metadrive.driving_env import DrivingEnv
from ppl.ppl import PPL
from ppl.sb3.td3.policies import TD3Policy
from ppl.utils.print_dict_utils import pretty_print, RecorderEnv
from ppl.utils.static_rss_filter import StaticRSSConfig, StaticRSSFilter

EVAL_ENV_START = 1000  # Evaluation seeds start from 1000


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


def make_metadrive_env(use_render=False, eval_env_start=EVAL_ENV_START):
    """Build the evaluation environment (no manual control, fixed seed range)."""
    config = dict(
        use_render=use_render,
        manual_control=False,
        start_seed=eval_env_start,
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


def make_static_rss_step_record(
    ckpt_index,
    env_id,
    episode,
    episode_in_env,
    step_in_episode,
    env_action_nominal,
    env_action_safe,
    internal_action_nominal,
    internal_action_safe,
    rss_info,
    env_info=None,
):
    """Flatten one StaticRSSFilter decision for step-level CSV logging."""
    env_info = env_info or {}
    selected = rss_info.get("selected", {}) if isinstance(rss_info, dict) else {}
    state_debug = rss_info.get("state_debug", {}) if isinstance(rss_info, dict) else {}
    nearest = state_debug.get("nearest_object", {}) if isinstance(state_debug, dict) else {}
    adapter_debug = state_debug.get("adapter_debug", {}) if isinstance(state_debug, dict) else {}
    blocking = rss_info.get("blocking_object", {}) if isinstance(rss_info, dict) else {}
    candidates = rss_info.get("candidates", []) if isinstance(rss_info, dict) else []
    if not isinstance(candidates, list):
        candidates = []

    def candidate_by_mode(mode):
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("mode") == mode:
                return candidate
        return {}

    def candidate_safe(candidate):
        if not candidate:
            return ""
        return candidate.get("safe", "")

    def candidate_projection_failed(candidate):
        if not candidate:
            return ""
        projection_debug = candidate.get("projection_debug", {})
        if not isinstance(projection_debug, dict):
            return ""
        return projection_debug.get("projection_failed", "")

    def candidate_margin(candidate, name):
        if not candidate:
            return np.nan
        margins = candidate.get("margins", {})
        if not isinstance(margins, dict):
            return np.nan
        return margins.get(name, np.nan)

    action_delta = float(np.linalg.norm(np.asarray(env_action_safe) - np.asarray(env_action_nominal)))
    record = dict(
        ckpt_index=ckpt_index,
        env_id=env_id,
        episode=episode,
        episode_in_env=episode_in_env,
        step_in_episode=step_in_episode,
        mode=rss_info.get("mode", "unknown"),
        filter_applied=action_delta > 1e-6 and rss_info.get("mode", "normal") != "adapter_error",
        obstacle_detected=rss_info.get("obstacle_detected", False),
        dynamic_vehicle_detected=rss_info.get("dynamic_vehicle_detected", False),
        step_cost=env_info.get("cost", np.nan),
        total_cost=env_info.get("total_cost", np.nan),
        crash=env_info.get("crash", False),
        crash_vehicle=env_info.get("crash_vehicle", False),
        crash_object=env_info.get("crash_object", False),
        crash_sidewalk=env_info.get("crash_sidewalk", False),
        out_of_road=env_info.get("out_of_road", False),
        route_completion=env_info.get("route_completion", np.nan),
        velocity=env_info.get("velocity", np.nan),
        step_reward=env_info.get("step_reward", np.nan),
        d_obs=rss_info.get("d_obs", np.nan),
        d_brake=rss_info.get("d_brake", np.nan),
        d_front=rss_info.get("d_front", np.nan),
        d_dynamic=rss_info.get("d_dynamic", np.nan),
        clearance_margin=rss_info.get("clearance_margin", np.nan),
        risk_step=rss_info.get("risk_step", np.nan),
        rss_margin=rss_info.get("rss_margin", np.nan),
        left_feasible=rss_info.get("left_feasible", False),
        right_feasible=rss_info.get("right_feasible", False),
        num_static_obstacles=state_debug.get("num_static_obstacles", np.nan),
        num_dynamic_vehicles=state_debug.get("num_dynamic_vehicles", np.nan),
        lidar_fallback_used=adapter_debug.get("observation_lidar_fallback_used", False),
        lidar_fallback_distance=adapter_debug.get("observation_lidar_distance", np.nan),
        lidar_fallback_source=adapter_debug.get("observation_lidar_source", ""),
        nearest_object_type=nearest.get("object_type", ""),
        nearest_object_class=nearest.get("class_name", ""),
        nearest_object_distance=nearest.get("distance", np.nan),
        nearest_object_longitudinal=nearest.get("longitudinal", np.nan),
        nearest_object_lateral=nearest.get("lateral", np.nan),
        nearest_object_speed=nearest.get("speed", np.nan),
        blocking_object_type=blocking.get("object_type", ""),
        blocking_object_class=blocking.get("class_name", ""),
        blocking_object_distance=blocking.get("distance", np.nan),
        blocking_object_longitudinal=blocking.get("longitudinal", np.nan),
        blocking_object_lateral=blocking.get("lateral", np.nan),
        blocking_object_speed=blocking.get("speed", np.nan),
        selected_score=rss_info.get("selected_score", np.nan),
        selected_progress_score=selected.get("progress_score", np.nan),
        selected_intervention_cost=selected.get("intervention_cost", np.nan),
        action_delta=action_delta,
        env_action_nominal_steer=float(env_action_nominal[0]),
        env_action_nominal_throttle_brake=float(env_action_nominal[1]),
        env_action_safe_steer=float(env_action_safe[0]),
        env_action_safe_throttle_brake=float(env_action_safe[1]),
        internal_action_nominal_acc=float(internal_action_nominal[0]),
        internal_action_nominal_steer=float(internal_action_nominal[1]),
        internal_action_safe_acc=float(internal_action_safe[0]),
        internal_action_safe_steer=float(internal_action_safe[1]),
        acc_nominal=rss_info.get("acc_nominal", np.nan),
        acc_safe=rss_info.get("acc_safe", np.nan),
        acc_delta=rss_info.get("acc_delta", np.nan),
        steer_nominal=rss_info.get("steer_nominal", np.nan),
        steer_safe=rss_info.get("steer_safe", np.nan),
        steer_delta=rss_info.get("steer_delta", np.nan),
        adapter_error=rss_info.get("adapter_error", ""),
    )

    if rss_info.get("mode") == "fallback_no_safe_candidate":
        stop_candidate = candidate_by_mode("stop")
        left_candidate = candidate_by_mode("left_bypass")
        right_candidate = candidate_by_mode("right_bypass")
        record.update(
            reason=rss_info.get("reason", ""),
            blocking_object_type=blocking.get("object_type", ""),
            blocking_object_x=blocking.get("x", np.nan),
            blocking_object_y=blocking.get("y", np.nan),
            blocking_object_longitudinal=blocking.get("longitudinal", np.nan),
            blocking_object_lateral=blocking.get("lateral", np.nan),
            d_obs=rss_info.get("d_obs", np.nan),
            d_brake=rss_info.get("d_brake", np.nan),
            rss_margin=rss_info.get("rss_margin", np.nan),
            left_feasible=rss_info.get("left_feasible", ""),
            right_feasible=rss_info.get("right_feasible", ""),
            candidate_count=len(candidates),
            safe_candidate_count=sum(
                1 for candidate in candidates if isinstance(candidate, dict) and candidate.get("safe") is True
            ),
            stop_candidate_safe=candidate_safe(stop_candidate),
            stop_h_stop_min=candidate_margin(stop_candidate, "h_stop_min"),
            stop_projection_failed=candidate_projection_failed(stop_candidate),
            left_candidate_safe=candidate_safe(left_candidate),
            left_projection_failed=candidate_projection_failed(left_candidate),
            right_candidate_safe=candidate_safe(right_candidate),
            right_projection_failed=candidate_projection_failed(right_candidate),
        )

    return record


def summarize_static_rss_episode(step_records):
    """Create episode-level StaticRSSFilter statistics for the main eval CSV."""
    if not step_records:
        return dict(
            static_rss_steps=0,
            static_rss_intervention_steps=0,
            static_rss_intervention_rate=0.0,
            static_rss_obstacle_steps=0,
            static_rss_stop_steps=0,
            static_rss_left_bypass_steps=0,
            static_rss_right_bypass_steps=0,
            static_rss_adapter_error_steps=0,
            static_rss_mean_d_obs=np.nan,
            static_rss_mean_d_brake=np.nan,
            static_rss_mean_rss_margin=np.nan,
        )

    modes = [record["mode"] for record in step_records]
    obstacle_steps = sum(1 for record in step_records if record["obstacle_detected"])
    intervention_steps = sum(1 for record in step_records if record["filter_applied"])
    adapter_error_steps = sum(1 for mode in modes if mode == "adapter_error")

    def _mean_valid(key):
        values = [record[key] for record in step_records if pd.notna(record[key])]
        return float(np.mean(values)) if values else np.nan

    return dict(
        static_rss_steps=len(step_records),
        static_rss_intervention_steps=intervention_steps,
        static_rss_intervention_rate=intervention_steps / max(len(step_records), 1),
        static_rss_obstacle_steps=obstacle_steps,
        static_rss_stop_steps=modes.count("stop"),
        static_rss_left_bypass_steps=modes.count("left_bypass"),
        static_rss_right_bypass_steps=modes.count("right_bypass"),
        static_rss_adapter_error_steps=adapter_error_steps,
        static_rss_mean_d_obs=_mean_valid("d_obs"),
        static_rss_mean_d_brake=_mean_valid("d_brake"),
        static_rss_mean_rss_margin=_mean_valid("rss_margin"),
    )


def evaluate_ppl_once(
    ckpt_path,
    ckpt_index,
    folder_name,
    use_render=False,
    num_ep_in_one_env=5,
    total_env_num=50,
    deterministic=True,
    use_static_rss_filter=False,
    static_rss_assume_adjacent_lanes=False,
    static_rss_static_speed_threshold=0.2,
    static_rss_enable_bypass=False,
    static_rss_intervention_margin=0.0,
    static_rss_reverse_steer=False,
    save_static_rss_step_csv=False,
    static_rss_diagnostic_env_id=-1,
    static_rss_dry_run=False,
    static_rss_disable_clearance_guard=False,
    static_rss_debug_print_every=0,
    eval_max_steps_per_episode=3000,
    eval_env_start=EVAL_ENV_START,
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

    env = make_metadrive_env(use_render, eval_env_start=eval_env_start)
    try:
        policy_function = PolicyFunction(zip_path, env)
    except Exception as e:
        print(f"Failed to load policy from {zip_path}: {e}")
        env.close()
        return None

    rss_filter = None
    if use_static_rss_filter:
        rss_filter = StaticRSSFilter(
            StaticRSSConfig(
                metadrive_assume_adjacent_lanes=static_rss_assume_adjacent_lanes,
                metadrive_static_speed_threshold=static_rss_static_speed_threshold,
                enable_bypass=static_rss_enable_bypass,
                enforce_intervention_margin=True,
                intervention_margin_threshold=static_rss_intervention_margin,
                metadrive_steer_sign=-1.0 if static_rss_reverse_steer else 1.0,
                enable_predictive_clearance_guard=not static_rss_disable_clearance_guard,
            )
        )
        print("[StaticRSSFilter] Enabled. Final evaluation CSV format is unchanged.")
        if static_rss_dry_run:
            print("[StaticRSSFilter] Dry-run mode: filter decisions are logged but nominal actions are executed.")

    saved_results = []
    static_rss_step_records = []
    episode_static_rss_records = []
    ep_velocities = []
    rss_adapter_error_printed = False
    rss_total_steps = 0
    rss_changed_steps = 0
    rss_mode_counts = {}
    rss_static_count_sum = 0
    rss_dynamic_count_sum = 0
    rss_lidar_fallback_steps = 0

    try:
        start = time.time()
        last_time = time.time()
        ep_count = 0
        step_count = 0
        ep_times = []

        env_index = 0
        num_ep_in = 0
        o = reset_eval_env(env, eval_env_start + env_index)

        while True:
            action = policy_function(o, deterministic=deterministic)[0]
            env_action_nominal = np.asarray(action, dtype=np.float32).copy()
            env_action_safe = env_action_nominal.copy()
            internal_action_nominal = [np.nan, np.nan]
            internal_action_safe = [np.nan, np.nan]
            rss_info = {"mode": "normal", "obstacle_detected": False}

            if rss_filter is not None:
                try:
                    rss_state = rss_filter.parse_state_from_metadrive(env)
                    rss_state = rss_filter.augment_state_from_observation(rss_state, o)
                    rss_static_count_sum += len(rss_state.get("static_obstacles", []))
                    rss_dynamic_count_sum += len(rss_state.get("vehicles", []))
                    if rss_state.get("adapter_debug", {}).get("observation_lidar_fallback_used", False):
                        rss_lidar_fallback_steps += 1
                    internal_action_nominal = rss_filter.to_internal_action(env_action_nominal, "metadrive")
                    internal_action_safe, rss_info = rss_filter.filter_action(rss_state, internal_action_nominal)
                    env_action_safe = np.asarray(
                        rss_filter.from_internal_action(internal_action_safe, "metadrive"),
                        dtype=np.float32,
                    )
                    if static_rss_dry_run:
                        action = env_action_nominal
                    else:
                        action = env_action_safe
                except Exception as error:
                    traceback.print_exc()
                    rss_info = {
                        "mode": "adapter_error",
                        "obstacle_detected": False,
                        "adapter_error": str(error),
                    }
                    if not rss_adapter_error_printed:
                        print("[StaticRSSFilter] Adapter failed once; continuing without filtering. Error: {}".format(error))
                        rss_adapter_error_printed = True

                rss_total_steps += 1
                rss_mode = rss_info.get("mode", "unknown")
                rss_mode_counts[rss_mode] = rss_mode_counts.get(rss_mode, 0) + 1
                if np.linalg.norm(env_action_safe - env_action_nominal) > 1e-6 and rss_mode != "adapter_error":
                    rss_changed_steps += 1

            if rss_filter is not None and static_rss_debug_print_every > 0:
                next_step_count = step_count + 1
                if next_step_count % static_rss_debug_print_every == 0:
                    action_delta = float(np.linalg.norm(env_action_safe - env_action_nominal))
                    print("[RSS DEBUG] step={}".format(next_step_count))
                    print("mode={}".format(rss_info.get("mode")))
                    print("reason={}".format(rss_info.get("reason", "")))
                    print("obstacle_detected={}".format(rss_info.get("obstacle_detected")))
                    print("dynamic_vehicle_detected={}".format(rss_info.get("dynamic_vehicle_detected")))
                    print("d_obs={}".format(rss_info.get("d_obs")))
                    print("d_brake={}".format(rss_info.get("d_brake")))
                    print("rss_margin={}".format(rss_info.get("rss_margin")))
                    print("d_dynamic={}".format(rss_info.get("d_dynamic")))
                    print("clearance_margin={}".format(rss_info.get("clearance_margin")))
                    print("risk_step={}".format(rss_info.get("risk_step")))
                    print("left_feasible={}".format(rss_info.get("left_feasible")))
                    print("right_feasible={}".format(rss_info.get("right_feasible")))
                    print("env_action_nominal={}".format(env_action_nominal))
                    print("env_action_safe={}".format(env_action_safe))
                    print("action_delta={}".format(action_delta))
                    print("acc_nominal={}".format(rss_info.get("acc_nominal")))
                    print("acc_safe={}".format(rss_info.get("acc_safe")))
                    print("acc_delta={}".format(rss_info.get("acc_delta")))
                    print("steer_nominal={}".format(rss_info.get("steer_nominal")))
                    print("steer_safe={}".format(rss_info.get("steer_safe")))
                    print("steer_delta={}".format(rss_info.get("steer_delta")))

            o, r, d, info = env.step(action)
            step_count += 1

            if (
                rss_filter is not None
                and save_static_rss_step_csv
                and (static_rss_diagnostic_env_id < 0 or eval_env_start + env_index == static_rss_diagnostic_env_id)
            ):
                record = make_static_rss_step_record(
                    ckpt_index=ckpt_index,
                    env_id=eval_env_start + env_index,
                    episode=ep_count + 1,
                    episode_in_env=num_ep_in + 1,
                    step_in_episode=step_count,
                    env_action_nominal=env_action_nominal,
                    env_action_safe=env_action_safe,
                    internal_action_nominal=internal_action_nominal,
                    internal_action_safe=internal_action_safe,
                    rss_info=rss_info,
                    env_info=info,
                )
                static_rss_step_records.append(record)
                episode_static_rss_records.append(record)

            if info:
                ep_velocities.append(info.get("velocity", 0))

            if use_render:
                env.render()

            if d or step_count >= eval_max_steps_per_episode:
                ep_times.append(time.time() - last_time)
                last_time = time.time()

                ep_count += 1
                num_ep_in += 1

                env_id_recorded = eval_env_start + env_index
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
                episode_static_rss_records = []

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
                if rss_filter is not None and save_static_rss_step_csv and static_rss_step_records:
                    tmp_step_path = osp.join(folder_name, "{}_static_rss_steps_tmp.csv".format(ckpt_name))
                    pd.DataFrame(static_rss_step_records).to_csv(tmp_step_path, index=False)

                step_count = 0

                # Advance to next env seed if enough episodes in this one
                if num_ep_in >= num_ep_in_one_env:
                    env_index += 1
                    num_ep_in = 0
                    if env_index >= total_env_num:
                        break

                o = reset_eval_env(env, eval_env_start + env_index)

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

    if rss_filter is not None and save_static_rss_step_csv:
        step_path = osp.join(folder_name, "{}_static_rss_steps.csv".format(ckpt_name))
        pd.DataFrame(static_rss_step_records).to_csv(step_path, index=False)
        print("Static RSS step-level results saved to: {}".format(step_path))

    if rss_filter is not None:
        changed_rate = rss_changed_steps / max(rss_total_steps, 1)
        avg_static = rss_static_count_sum / max(rss_total_steps, 1)
        avg_dynamic = rss_dynamic_count_sum / max(rss_total_steps, 1)
        print(
            "[StaticRSSFilter] total_steps={} changed_actions={} changed_rate={:.4f} "
            "avg_static_objects={:.2f} avg_dynamic_vehicles={:.2f} lidar_fallback_steps={} modes={}".format(
                rss_total_steps,
                rss_changed_steps,
                changed_rate,
                avg_static,
                avg_dynamic,
                rss_lidar_fallback_steps,
                rss_mode_counts,
            )
        )

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
    parser.add_argument("--eval_start_seed", type=int, default=EVAL_ENV_START, help="First MetaDrive eval env seed.")
    parser.add_argument("--num_ep_in_one_env", type=int, default=1, help="Episodes per environment seed.")
    parser.add_argument("--total_env_num", type=int, default=50, help="Number of environment seeds.")
    parser.add_argument(
        "--eval_max_steps_per_episode",
        type=int,
        default=3000,
        help="Maximum steps per episode before forcing evaluation rollover.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy during evaluation (default: deterministic).",
    )
    parser.add_argument(
        "--static_rss_filter",
        action="store_true",
        help="Enable Progress-Aware RSS Action Projection before env.step(action).",
    )
    parser.add_argument(
        "--static_rss_assume_adjacent_lanes",
        action="store_true",
        help=(
            "If MetaDrive lane introspection cannot find side lanes, still expose "
            "default-width left/right lanes to the filter. Use only for controlled "
            "static-obstacle experiments."
        ),
    )
    parser.add_argument(
        "--static_rss_static_speed_threshold",
        type=float,
        default=0.2,
        help="Objects at or below this speed are treated as static obstacles by the MetaDrive adapter.",
    )
    parser.add_argument(
        "--static_rss_enable_bypass",
        action="store_true",
        help="Allow certified left/right bypass candidates. Default Static RSS filtering is stop-only.",
    )
    parser.add_argument(
        "--static_rss_intervention_margin",
        type=float,
        default=0.0,
        help="Only filter when d_obs - d_brake is at or below this margin.",
    )
    parser.add_argument(
        "--static_rss_reverse_steer",
        action="store_true",
        help="Flip MetaDrive steering sign if bypass direction is opposite in the installed environment.",
    )
    parser.add_argument(
        "--static_rss_diagnostics",
        action="store_true",
        help="Save an extra per-step Static RSS diagnostics CSV. Final result CSV stays unchanged.",
    )
    parser.add_argument(
        "--static_rss_dry_run",
        action="store_true",
        help="Call and log Static RSS filter decisions, but execute nominal policy actions in env.step.",
    )
    parser.add_argument(
        "--static_rss_disable_clearance_guard",
        action="store_true",
        help="Disable predictive clearance guard in StaticRSSFilter.",
    )
    parser.add_argument(
        "--static_rss_debug_print_every",
        type=int,
        default=0,
        help="Print Static RSS debug info every N steps when > 0.",
    )
    parser.add_argument(
        "--static_rss_diagnostic_env_id",
        type=int,
        default=-1,
        help="Only record Static RSS step diagnostics for this env_id. Use -1 to record all envs.",
    )
    parser.add_argument(
        "--no_static_rss_step_csv",
        action="store_true",
        help="When diagnostics are enabled, do not save the extra per-step Static RSS CSV.",
    )

    args = parser.parse_args()

    deterministic = not args.stochastic

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
            use_static_rss_filter=args.static_rss_filter,
            static_rss_assume_adjacent_lanes=args.static_rss_assume_adjacent_lanes,
            static_rss_static_speed_threshold=args.static_rss_static_speed_threshold,
            static_rss_enable_bypass=args.static_rss_enable_bypass,
            static_rss_intervention_margin=args.static_rss_intervention_margin,
            static_rss_reverse_steer=args.static_rss_reverse_steer,
            save_static_rss_step_csv=args.static_rss_diagnostics and not args.no_static_rss_step_csv,
            static_rss_diagnostic_env_id=args.static_rss_diagnostic_env_id,
            static_rss_dry_run=args.static_rss_dry_run,
            static_rss_disable_clearance_guard=args.static_rss_disable_clearance_guard,
            static_rss_debug_print_every=args.static_rss_debug_print_every,
            eval_max_steps_per_episode=args.eval_max_steps_per_episode,
            eval_env_start=args.eval_start_seed,
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
            use_static_rss_filter=args.static_rss_filter,
            static_rss_assume_adjacent_lanes=args.static_rss_assume_adjacent_lanes,
            static_rss_static_speed_threshold=args.static_rss_static_speed_threshold,
            static_rss_enable_bypass=args.static_rss_enable_bypass,
            static_rss_intervention_margin=args.static_rss_intervention_margin,
            static_rss_reverse_steer=args.static_rss_reverse_steer,
            save_static_rss_step_csv=args.static_rss_diagnostics and not args.no_static_rss_step_csv,
            static_rss_diagnostic_env_id=args.static_rss_diagnostic_env_id,
            static_rss_dry_run=args.static_rss_dry_run,
            static_rss_disable_clearance_guard=args.static_rss_disable_clearance_guard,
            static_rss_debug_print_every=args.static_rss_debug_print_every,
            eval_max_steps_per_episode=args.eval_max_steps_per_episode,
            eval_env_start=args.eval_start_seed,
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
                use_static_rss_filter=args.static_rss_filter,
                static_rss_assume_adjacent_lanes=args.static_rss_assume_adjacent_lanes,
                static_rss_static_speed_threshold=args.static_rss_static_speed_threshold,
                static_rss_enable_bypass=args.static_rss_enable_bypass,
                static_rss_intervention_margin=args.static_rss_intervention_margin,
                static_rss_reverse_steer=args.static_rss_reverse_steer,
                save_static_rss_step_csv=args.static_rss_diagnostics and not args.no_static_rss_step_csv,
                static_rss_diagnostic_env_id=args.static_rss_diagnostic_env_id,
                static_rss_dry_run=args.static_rss_dry_run,
                static_rss_disable_clearance_guard=args.static_rss_disable_clearance_guard,
                static_rss_debug_print_every=args.static_rss_debug_print_every,
                eval_max_steps_per_episode=args.eval_max_steps_per_episode,
                eval_env_start=args.eval_start_seed,
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
