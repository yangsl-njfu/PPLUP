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
from collections import defaultdict

import numpy as np
import pandas as pd

from ppl.experiments.metadrive.driving_env import DrivingEnv
from ppl.ppl import PPL
from ppl.sb3.td3.policies import TD3Policy
from ppl.utils.predictive_recovery_filter import (
    RECOVERY_DIAGNOSTIC_FIELDS,
    PredictiveRecoveryConfig,
    PredictiveRecoveryFilter,
)
from ppl.utils.print_dict_utils import pretty_print, RecorderEnv

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


def extract_collision_debug_info(env, info):
    """Safely extract collision and contact information from env/info."""
    result = {
        # From info
        "cost": info.get("cost", 0.0) if info else 0.0,
        "crash": info.get("crash", False) if info else False,
        "crash_vehicle": info.get("crash_vehicle", False) if info else False,
        "crash_object": info.get("crash_object", False) if info else False,
        "crash_sidewalk": info.get("crash_sidewalk", False) if info else False,
        "crash_building": info.get("crash_building", False) if info else False,
        "crash_human": info.get("crash_human", False) if info else False,
        "out_of_road": info.get("out_of_road", False) if info else False,
        "out_of_route": info.get("out_of_route", False) if info else False,
        "on_lane": info.get("on_lane", True) if info else True,
        "route_completion": info.get("route_completion", 0.0) if info else 0.0,
        "velocity": info.get("velocity", 0.0) if info else 0.0,
        "reward": info.get("reward", 0.0) if info else 0.0,
        "arrive_dest": info.get("arrive_dest", False) if info else False,
        # From env state if available
        "contact_results": None,
    }

    # Try to get additional state from env
    try:
        vehicle = getattr(env, "vehicle", None) or getattr(env, "current_track_vehicle", None)
        if vehicle is not None:
            # Get crash flags from vehicle
            for attr in ["crash_vehicle", "crash_object", "crash_sidewalk", "crash_building", "crash_human"]:
                if attr not in result or result.get(attr) is None:
                    val = getattr(vehicle, attr, None)
                    if val is not None:
                        try:
                            result[attr] = bool(val)
                        except Exception:
                            pass

            # Get contact_results
            contact_results = getattr(vehicle, "contact_results", None)
            if contact_results is not None:
                try:
                    result["contact_results"] = str(contact_results)[:500]  # Truncate
                except Exception:
                    pass

            # Get on_lane
            on_lane = getattr(vehicle, "on_lane", None)
            if on_lane is not None:
                try:
                    result["on_lane"] = bool(on_lane)
                except Exception:
                    pass

            # Get out_of_route
            out_of_route = getattr(vehicle, "out_of_route", None)
            if out_of_route is not None:
                try:
                    result["out_of_route"] = bool(out_of_route)
                except Exception:
                    pass

            # Get dist_to_left/right_side
            for side in ["left_side", "right_side"]:
                attr = f"dist_to_{side}"
                val = getattr(vehicle, attr, None)
                if val is not None:
                    try:
                        result[attr] = float(val)
                    except Exception:
                        pass

    except Exception:
        pass

    # Try to get contact_results from env engine
    try:
        root = env
        for _ in range(5):
            if hasattr(root, "engine"):
                engine = getattr(root, "engine")
                if engine is not None:
                    traffic_mgr = getattr(engine, "traffic_manager", None)
                    if traffic_mgr is not None:
                        contact_results = getattr(traffic_mgr, "contact_results", None)
                        if contact_results is not None and result.get("contact_results") is None:
                            result["contact_results"] = str(contact_results)[:500]
                    break
            if hasattr(root, "env"):
                root = getattr(root, "env")
            else:
                break
    except Exception:
        pass

    return result


def get_nearest_objects_info(env, vehicle, ego_pos, max_distance=50.0):
    """Get nearest objects around ego vehicle."""
    result = {
        "nearest_object_type": "",
        "nearest_object_distance": float("inf"),
        "nearest_traffic_object_distance": float("inf"),
        "front_traffic_object_distance": float("inf"),
        "num_nearby_objects": 0,
    }
    if vehicle is None or ego_pos is None:
        return result

    try:
        nav = getattr(vehicle, "navigation", None)
        if nav is not None:
            # Try to get nearby vehicles info
            current_lanes = getattr(nav, "current_ref_lanes", None)
            if current_lanes:
                result["num_nearby_objects"] = len(current_lanes) * 3  # rough estimate
    except Exception:
        pass

    # Simple distance check with lidar if available
    try:
        lidar = getattr(vehicle, "lidar", None)
        if lidar is not None:
            detected = getattr(lidar, "detected_objects", None) or getattr(lidar, "point_cloud", None)
            if detected is not None:
                # Estimate number of nearby objects
                try:
                    if hasattr(detected, "__len__"):
                        result["num_nearby_objects"] = len(detected)
                except Exception:
                    pass
    except Exception:
        pass

    return result


class DiagnosticLogger:
    """Handles all diagnostic logging for evaluation."""

    def __init__(self, folder_name, ckpt_name, log_diagnostics=True, event_window_size=20):
        self.folder_name = folder_name
        self.ckpt_name = ckpt_name
        self.log_diagnostics = log_diagnostics
        self.event_window_size = event_window_size

        # Step diagnostics
        self.step_log_path = None
        self.step_csv_header_written = False
        self.step_rows = []

        # Event window logs
        self.cost_event_log_path = None
        self.crash_event_log_path = None
        self.cost_event_header_written = False
        self.crash_event_header_written = False
        self.cost_event_rows = []
        self.crash_event_rows = []

        # High cost episodes
        self.high_cost_log_path = None
        self.high_cost_rows = []

        if self.log_diagnostics:
            os.makedirs(folder_name, exist_ok=True)
            self.step_log_path = osp.join(folder_name, f"{ckpt_name}_step_diagnostics.csv")
            self.cost_event_log_path = osp.join(folder_name, f"{ckpt_name}_cost_event_windows.csv")
            self.crash_event_log_path = osp.join(folder_name, f"{ckpt_name}_crash_event_windows.csv")
            self.high_cost_log_path = osp.join(folder_name, f"{ckpt_name}_high_cost_episodes.csv")

    def write_step_diagnostic(self, step_data):
        """Write a single step diagnostic row."""
        if not self.log_diagnostics or self.step_log_path is None:
            return
        try:
            file_exists = osp.exists(self.step_log_path)
            with open(self.step_log_path, "a", newline="") as f:
                if not file_exists or not self.step_csv_header_written:
                    writer = csv.DictWriter(f, fieldnames=list(step_data.keys()))
                    writer.writeheader()
                    self.step_csv_header_written = True
                else:
                    writer = csv.DictWriter(f, fieldnames=list(step_data.keys()))
                writer.writerow(step_data)
        except Exception:
            pass

    def add_event_window(self, rows, log_path, header_written_ref):
        """Add event window rows to buffer."""
        if not self.log_diagnostics:
            return
        # Just append to buffer, will write at episode end
        pass

    def write_event_windows(self):
        """Write buffered event windows to CSV."""
        if not self.log_diagnostics:
            return

        # Write cost event windows
        if self.cost_event_rows and self.cost_event_log_path:
            try:
                file_exists = osp.exists(self.cost_event_log_path)
                with open(self.cost_event_log_path, "a", newline="") as f:
                    if not file_exists or not self.cost_event_header_written:
                        writer = csv.DictWriter(f, fieldnames=list(self.cost_event_rows[0].keys()) if self.cost_event_rows else [])
                        writer.writeheader()
                        self.cost_event_header_written = True
                    else:
                        writer = csv.DictWriter(f, fieldnames=list(self.cost_event_rows[0].keys()) if self.cost_event_rows else [])
                    for row in self.cost_event_rows:
                        writer.writerow(row)
                self.cost_event_rows = []
            except Exception:
                pass

        # Write crash event windows
        if self.crash_event_rows and self.crash_event_log_path:
            try:
                file_exists = osp.exists(self.crash_event_log_path)
                with open(self.crash_event_log_path, "a", newline="") as f:
                    if not file_exists or not self.crash_event_header_written:
                        writer = csv.DictWriter(f, fieldnames=list(self.crash_event_rows[0].keys()) if self.crash_event_rows else [])
                        writer.writeheader()
                        self.crash_event_header_written = True
                    else:
                        writer = csv.DictWriter(f, fieldnames=list(self.crash_event_rows[0].keys()) if self.crash_event_rows else [])
                    for row in self.crash_event_rows:
                        writer.writerow(row)
                self.crash_event_rows = []
            except Exception:
                pass

    def add_high_cost_episode(self, episode_data):
        """Add high cost episode to buffer."""
        if not self.log_diagnostics or self.high_cost_log_path is None:
            return
        self.high_cost_rows.append(episode_data)

    def write_high_cost_episodes(self):
        """Write high cost episodes to CSV."""
        if not self.log_diagnostics or not self.high_cost_rows or self.high_cost_log_path is None:
            return
        try:
            df = pd.DataFrame(self.high_cost_rows)
            df = df.sort_values("episode_cost", ascending=False)
            df.to_csv(self.high_cost_log_path, index=False)
            self.high_cost_rows = []
        except Exception:
            pass


def summarize_recovery_episode(step_infos):
    """Summarize recovery diagnostics for one episode."""
    if not step_infos:
        return {}
    summary = {}
    last_info = step_infos[-1]
    for field in RECOVERY_DIAGNOSTIC_FIELDS:
        summary[field] = last_info.get(field)
        numeric_values = []
        for item in step_infos:
            value = item.get(field)
            if isinstance(value, (bool, np.bool_)):
                numeric_values.append(float(value))
            elif isinstance(value, (int, float, np.number)) and np.isfinite(value):
                numeric_values.append(float(value))
        if numeric_values:
            summary["recovery_mean_{}".format(field)] = float(np.mean(numeric_values))
    summary["recovery_num_steps"] = len(step_infos)
    summary["recovery_certified_rate"] = float(np.mean([bool(item.get("recovery_certified", False)) for item in step_infos]))
    summary["recovery_min_hard_safe_candidates"] = int(min(item.get("num_hard_safe_candidates", 0) for item in step_infos))
    summary["recovery_max_filter_time_ms"] = float(max(item.get("filter_time_ms", 0.0) for item in step_infos))

    # 统计接管
    filter_intervened_count = sum(1 for item in step_infos if item.get("filter_intervened", False))
    summary["intervention_count"] = filter_intervened_count
    summary["intervention_rate"] = filter_intervened_count / max(1, len(step_infos))

    # 接管原因统计
    intervention_reasons = [item.get("intervention_reason", "none") for item in step_infos if item.get("filter_intervened", False)]
    summary["intervention_reasons"] = ";".join(intervention_reasons) if intervention_reasons else ""

    # 平均 action 差值
    delta_steer_values = [item.get("safe_raw_steer_delta", 0.0) for item in step_infos if isinstance(item.get("safe_raw_steer_delta"), (int, float))]
    delta_acc_values = [item.get("safe_raw_acc_delta", 0.0) for item in step_infos if isinstance(item.get("safe_raw_acc_delta"), (int, float))]
    summary["mean_safe_raw_steer_delta"] = float(np.mean(delta_steer_values)) if delta_steer_values else 0.0
    summary["mean_safe_raw_acc_delta"] = float(np.mean(delta_acc_values)) if delta_acc_values else 0.0

    # 最小 margin 统计
    vehicle_margin_values = [item.get("vehicle_margin_min", float("inf")) for item in step_infos if item.get("vehicle_margin_min", float("inf")) < float("inf")]
    static_margin_values = [item.get("static_margin_min", float("inf")) for item in step_infos if item.get("static_margin_min", float("inf")) < float("inf")]
    boundary_margin_values = [item.get("boundary_margin_min", float("inf")) for item in step_infos if item.get("boundary_margin_min", float("inf")) < float("inf")]
    summary["min_vehicle_margin"] = float(np.min(vehicle_margin_values)) if vehicle_margin_values else float("inf")
    summary["min_static_margin"] = float(np.min(static_margin_values)) if static_margin_values else float("inf")
    summary["min_boundary_margin"] = float(np.min(boundary_margin_values)) if boundary_margin_values else float("inf")

    # Raw action 对照统计
    raw_safe_count = sum(1 for item in step_infos if item.get("raw_predicted_collision", False) == False and item.get("raw_predicted_out_of_road", False) == False)
    summary["raw_safe_rate"] = raw_safe_count / max(1, len(step_infos))

    # 拒绝原因统计
    all_reject_reasons = [item.get("early_reject_reasons", "") for item in step_infos if item.get("early_reject_reasons", "")]
    summary["all_early_reject_reasons"] = ";".join(all_reject_reasons) if all_reject_reasons else ""

    first_cost_step = -1
    first_hard_risk_step = -1
    first_intervention_step = -1
    low_risk_false_negative_count = 0
    cooldown_false_negative_count = 0
    for idx, item in enumerate(step_infos):
        step_num = int(item.get("step", idx + 1))
        step_cost = float(item.get("step_cost", item.get("cost", 0.0)) or 0.0)
        if first_cost_step < 0 and step_cost > 0:
            first_cost_step = step_num
        if first_hard_risk_step < 0 and bool(item.get("hard_risk", False)):
            first_hard_risk_step = step_num
        if first_intervention_step < 0 and bool(item.get("filter_intervened", False)):
            first_intervention_step = step_num
        if step_cost > 0 and bool(item.get("low_risk_gate_passed", False)):
            low_risk_false_negative_count += 1
        if step_cost > 0 and (
            item.get("fallback_reason", "") == "cooldown_or_passthrough"
            or item.get("intervention_reason", "") == "cooldown"
        ):
            cooldown_false_negative_count += 1

    pre_count = 0
    post_count = 0
    for item in step_infos:
        if not bool(item.get("filter_intervened", False)):
            continue
        step_num = int(item.get("step", 0) or 0)
        if first_cost_step > 0 and step_num >= first_cost_step:
            post_count += 1
        else:
            pre_count += 1

    summary["pre_first_cost_intervention_count"] = pre_count
    summary["post_first_cost_intervention_count"] = post_count
    summary["first_hard_risk_step"] = first_hard_risk_step
    summary["first_intervention_step"] = first_intervention_step
    summary["first_cost_step"] = first_cost_step
    summary["missed_first_cost"] = bool(first_cost_step > 0 and (first_intervention_step < 0 or first_intervention_step >= first_cost_step))
    summary["late_intervention"] = bool(first_cost_step > 0 and first_intervention_step > 0 and first_cost_step - first_intervention_step <= 2)
    summary["low_risk_false_negative_count"] = low_risk_false_negative_count
    summary["cooldown_false_negative_count"] = cooldown_false_negative_count

    return summary


def compact_episode_result_for_terminal(result):
    """Print only the original base evaluation result fields."""
    preferred_keys = [
        "episode",
        "ckpt_index",
        "env_id",
        "num_ep_in_one_env",
        "success",
        "crash",
        "crash_vehicle",
        "out_of_road",
        "route_completion",
        "velocity_step_mean",
        "episode_reward",
        "episode_cost",
        "episode_length",
    ]
    return {key: result.get(key) for key in preferred_keys if key in result}


def contact_results_nonempty(value):
    text = str(value).strip().lower()
    return text not in ("", "none", "nan", "[]", "{}", "set()")


def contact_results_dangerous(value):
    if not contact_results_nonempty(value):
        return False
    text = str(value).lower()
    benign_markers = ("road_line_broken", "broken_single", "broken")
    if any(marker in text for marker in benign_markers):
        return False
    dangerous_markers = (
        "vehicle",
        "traffic_object",
        "traffic_cone",
        "traffic_barrier",
        "road_line_solid",
        "solid_single_white",
        "road_edge",
        "sidewalk",
        "object",
        "crash",
    )
    return any(marker in text for marker in dangerous_markers)


def print_runtime_assurance_validation_stats(step_rows):
    """Print checkpoint-level pre-first-cost runtime assurance diagnostics."""
    if not step_rows:
        print("[Runtime Assurance Validation] No step diagnostics collected.")
        return

    episodes = defaultdict(list)
    for row in step_rows:
        episodes[(row.get("env_id"), row.get("episode"))].append(row)

    first_cost_events = []
    inactive_first_cost = 0
    recall_before_20 = 0
    pre_cost_interventions = 0
    post_cost_interventions = 0
    cooldown_hard_risk_violations = 0
    low_risk_unsafe_violations = 0
    crash_attribution_counts = defaultdict(int)

    for rows in episodes.values():
        rows = sorted(rows, key=lambda item: int(item.get("step", 0) or 0))
        first_cost_idx = -1
        for idx, row in enumerate(rows):
            if float(row.get("step_cost", row.get("cost", 0.0)) or 0.0) > 0.0:
                first_cost_idx = idx
                break
        if first_cost_idx >= 0:
            first_cost_events.append(rows[first_cost_idx])
            first_row = rows[first_cost_idx]
            if first_row.get("recovery_mode") == "inactive_raw_action" or str(first_row.get("selected_candidate_type", "")).startswith("raw_action"):
                inactive_first_cost += 1
            lookback = rows[max(0, first_cost_idx - 20):first_cost_idx + 1]
            if any(bool(item.get("hard_risk", False)) or float(item.get("raw_predicted_cost_risk", 0.0) or 0.0) > 0.0 for item in lookback):
                recall_before_20 += 1
            pre_cost_interventions += sum(1 for item in rows[:first_cost_idx] if bool(item.get("filter_intervened", False)))
            post_cost_interventions += sum(1 for item in rows[first_cost_idx:] if bool(item.get("filter_intervened", False)))

            detected = any(
                bool(item.get("hard_risk", False))
                or bool(item.get("raw_predicted_collision", False))
                or bool(item.get("raw_predicted_out_of_road", False))
                or float(item.get("raw_predicted_cost_risk", 0.0) or 0.0) > 0.0
                for item in lookback
            )
            no_candidate = any(item.get("intervention_reason", "") == "no_safe_candidate" for item in lookback)
            action_limited = any(bool(item.get("action_limited_by_raw_delta", False)) for item in lookback)
            first_intervention_idx = next((i for i, item in enumerate(rows[:first_cost_idx + 1]) if bool(item.get("filter_intervened", False))), -1)
            post_contact = any(bool(item.get("post_contact_intervention", False)) for item in lookback)
            if not detected:
                crash_attribution_counts["risk_not_detected"] += 1
            elif no_candidate:
                crash_attribution_counts["detected_but_no_candidate"] += 1
            elif action_limited:
                crash_attribution_counts["detected_but_action_limited"] += 1
            elif first_intervention_idx >= 0 and first_cost_idx - first_intervention_idx <= 2:
                crash_attribution_counts["detected_but_too_late"] += 1
            elif post_contact:
                crash_attribution_counts["post_contact_stuck"] += 1

        for row in rows:
            if bool(row.get("hard_risk", False)) and (
                row.get("fallback_reason") == "cooldown_or_passthrough"
                or row.get("intervention_reason") == "cooldown"
            ):
                cooldown_hard_risk_violations += 1
            if bool(row.get("low_risk_gate_passed", False)):
                unsafe_low_risk = (
                    contact_results_dangerous(row.get("contact_results"))
                    or float(row.get("min_boundary_margin", float("inf")) or float("inf")) < PredictiveRecoveryConfig.hard_boundary_margin_for_intervention
                    or float(row.get("nearest_static_object_distance", float("inf")) or float("inf")) < PredictiveRecoveryConfig.hard_static_distance_threshold
                    or float(row.get("ttc_vehicle_min", float("inf")) or float("inf")) < PredictiveRecoveryConfig.hard_vehicle_ttc_threshold
                    or float(row.get("ttc_static_min", float("inf")) or float("inf")) < PredictiveRecoveryConfig.hard_static_ttc_threshold
                )
                if unsafe_low_risk:
                    low_risk_unsafe_violations += 1

    total_first_cost = max(1, len(first_cost_events))
    print("\n===== Runtime Assurance Validation =====")
    print("first_cost_events={}".format(len(first_cost_events)))
    print("first_cost_inactive_raw_action_rate={:.3f}".format(inactive_first_cost / total_first_cost))
    print("first_cost_prev20_risk_recall={:.3f}".format(recall_before_20 / total_first_cost))
    print("pre_first_cost_intervention_count={}".format(pre_cost_interventions))
    print("post_first_cost_intervention_count={}".format(post_cost_interventions))
    print("cooldown_hard_risk_violations={}".format(cooldown_hard_risk_violations))
    print("low_risk_unsafe_violations={}".format(low_risk_unsafe_violations))
    if crash_attribution_counts:
        print("crash_attribution={}".format(";".join("{}:{}".format(k, crash_attribution_counts[k]) for k in sorted(crash_attribution_counts))))


def append_recovery_episode_log(path, recovery_summary, episode_meta):
    if not path or not recovery_summary:
        return
    row = {}
    row.update(episode_meta)
    row.update(recovery_summary)
    folder = osp.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    file_exists = osp.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not file_exists, index=False)


import csv


def evaluate_ppl_once(
    ckpt_path,
    ckpt_index,
    folder_name,
    use_render=False,
    num_ep_in_one_env=5,
    total_env_num=50,
    start_env_index=0,
    deterministic=True,
    enable_predictive_recovery=False,
    recovery_config=None,
    progress_interval=0,
    recovery_episode_log_csv="",
    log_step_diagnostics=False,
    log_event_windows=False,
    log_high_cost_episodes=False,
    diagnostic_cost_threshold=20,
):
    """
    Evaluate one PPL checkpoint on `total_env_num` environments.

    Args:
        ckpt_path: Directory containing checkpoint or full path to .zip
        ckpt_index: Checkpoint step index
        folder_name: Directory to save result CSV files
        use_render: Whether to render the environment
        num_ep_in_one_env: Episodes per environment seed
        total_env_num: Number of different environment seeds
        start_env_index: Offset added to EVAL_ENV_START for the first evaluated seed
        deterministic: Whether the policy acts deterministically
        enable_predictive_recovery: Enable runtime assurance filter
        recovery_config: Configuration for the filter
        progress_interval: Print progress every N steps
        recovery_episode_log_csv: Path to save recovery diagnostics
        log_step_diagnostics: Log every step to CSV
        log_event_windows: Log cost/crash event windows
        log_high_cost_episodes: Log high cost episodes
        diagnostic_cost_threshold: Cost threshold for high-cost episode

    Returns:
        pd.DataFrame with per-episode results
    """
    # Resolve zip path
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

    predictive_filter = None
    if enable_predictive_recovery:
        predictive_filter = PredictiveRecoveryFilter(recovery_config or PredictiveRecoveryConfig())
        print("[Runtime Assurance] Enabled.")
        if not recovery_episode_log_csv:
            recovery_episode_log_csv = osp.join(folder_name, "{}_recovery_episode.csv".format(ckpt_name))

    # Initialize diagnostic logger
    diag_logger = DiagnosticLogger(
        folder_name, ckpt_name,
        log_diagnostics=log_step_diagnostics or log_event_windows or log_high_cost_episodes,
        event_window_size=20
    )

    saved_results = []
    ep_velocities = []
    recovery_step_infos = []

    # Step diagnostics for current episode
    current_ep_step_diagnostics = []
    all_step_diagnostics = []

    try:
        start = time.time()
        last_time = time.time()
        ep_count = 0
        step_count = 0
        global_step_count = 0
        ep_times = []

        env_index = int(start_env_index)
        end_env_index = env_index + int(total_env_num)
        num_ep_in = 0
        o = reset_eval_env(env, EVAL_ENV_START + env_index)

        # Reset episode state
        if predictive_filter is not None:
            predictive_filter.reset_episode()

        while True:
            action = policy_function(o, deterministic=deterministic)[0]
            raw_action = action

            # Runtime assurance filter
            if predictive_filter is not None:
                try:
                    safe_action, safety_info = predictive_filter.filter(env, o, raw_action)
                except Exception:
                    print("[Runtime Assurance] PredictiveRecoveryFilter.filter() failed.")
                    print(
                        "episode={} step={} global_step={}".format(
                            ep_count + 1,
                            step_count + 1,
                            global_step_count + 1,
                        )
                    )
                    print("raw_action={}".format(np.asarray(raw_action).tolist()))
                    traceback.print_exc()
                    raise
            else:
                safe_action, safety_info = raw_action, {}

            # Shadow mode: execute raw_action
            if enable_predictive_recovery and recovery_config is not None and recovery_config._debug_shadow_record:
                action_to_env = raw_action
            else:
                action_to_env = safe_action

            # Execute step
            o, r, d, info = env.step(action_to_env)
            global_step_count += 1
            step_count += 1

            if info:
                ep_velocities.append(info.get("velocity", 0))

            # Extract collision debug info
            collision_info = extract_collision_debug_info(env, info)

            # Get nearest objects info
            try:
                vehicle = getattr(env, "vehicle", None) or getattr(env, "current_track_vehicle", None)
                ego_pos = getattr(vehicle, "position", None)
                if ego_pos is not None:
                    import numpy as np
                    ego_pos = np.asarray(ego_pos).reshape(-1)[:2]
                nearest_info = get_nearest_objects_info(env, vehicle, ego_pos)
            except Exception:
                nearest_info = {}

            # Calculate step cost delta
            step_cost = collision_info.get("cost", 0.0)
            try:
                episode_cost_so_far = sum(env.user_data.get("cost", [0.0])) if isinstance(env.user_data.get("cost"), list) else float(env.user_data.get("cost", 0.0))
            except Exception:
                episode_cost_so_far = 0.0

            # Build step diagnostic row
            step_diagnostic = {
                "ckpt_index": ckpt_index,
                "env_id": EVAL_ENV_START + env_index,
                "episode": ep_count + 1,
                "step": step_count,
                "global_step": global_step_count,
                "raw_action_steer": float(raw_action[0]) if hasattr(raw_action, '__getitem__') else float(raw_action),
                "raw_action_throttle": float(raw_action[1]) if hasattr(raw_action, '__getitem__') and len(raw_action) > 1 else 0.0,
                "safe_action_steer": float(safe_action[0]) if hasattr(safe_action, '__getitem__') else float(safe_action),
                "safe_action_throttle": float(safe_action[1]) if hasattr(safe_action, '__getitem__') and len(safe_action) > 1 else 0.0,
                "action_delta_steer": float(safe_action[0] - raw_action[0]) if hasattr(safe_action, '__getitem__') and hasattr(raw_action, '__getitem__') else 0.0,
                "action_delta_throttle": float(safe_action[1] - raw_action[1]) if hasattr(safe_action, '__getitem__') and hasattr(raw_action, '__getitem__') and len(safe_action) > 1 and len(raw_action) > 1 else 0.0,
                "filter_enabled": enable_predictive_recovery,
                "filter_intervened": safety_info.get("filter_intervened", False),
                "recovery_mode": safety_info.get("recovery_mode", ""),
                "selected_candidate_type": safety_info.get("selected_candidate_type", ""),
                "selected_lateral_target": safety_info.get("selected_lateral_target", ""),
                "selected_speed_target": safety_info.get("selected_speed_target", ""),
                "num_candidates": safety_info.get("num_candidates", 0),
                "num_hard_safe_candidates": safety_info.get("num_hard_safe_candidates", 0),
                "fallback_reason": safety_info.get("fallback_reason", ""),
                "cost": step_cost,
                "step_cost": step_cost,
                "episode_cost_so_far": episode_cost_so_far,
                "crash": collision_info.get("crash", False),
                "crash_vehicle": collision_info.get("crash_vehicle", False),
                "crash_object": collision_info.get("crash_object", False),
                "crash_sidewalk": collision_info.get("crash_sidewalk", False),
                "crash_building": collision_info.get("crash_building", False),
                "crash_human": collision_info.get("crash_human", False),
                "out_of_road": collision_info.get("out_of_road", False),
                "out_of_route": collision_info.get("out_of_route", False),
                "on_lane": collision_info.get("on_lane", True),
                "route_completion": collision_info.get("route_completion", 0.0),
                "velocity": collision_info.get("velocity", 0.0),
                "reward": collision_info.get("reward", 0.0),
                "done": d,
                "done_reason": "arrive_dest" if collision_info.get("arrive_dest", False) else ("crash" if collision_info.get("crash", False) else ("out_of_road" if collision_info.get("out_of_road", False) else "max_steps")),
                "front_blocking_object_type": safety_info.get("front_blocking_object_type", ""),
                "front_blocking_object_distance": safety_info.get("front_blocking_object_distance", float("inf")),
                "front_blocking_object_reason": safety_info.get("front_blocking_object_reason", ""),
                "front_blocking_is_dynamic_vehicle": safety_info.get("front_blocking_is_dynamic_vehicle", False),
                "nearest_object_type": nearest_info.get("nearest_object_type", ""),
                "nearest_object_distance": nearest_info.get("nearest_object_distance", float("inf")),
                "nearest_traffic_object_distance": nearest_info.get("nearest_traffic_object_distance", float("inf")),
                "front_traffic_object_distance": nearest_info.get("front_traffic_object_distance", float("inf")),
                "min_vehicle_margin": safety_info.get("vehicle_margin_min", float("inf")),
                "min_static_margin": safety_info.get("static_margin_min", float("inf")),
                "min_boundary_margin": safety_info.get("boundary_margin_min", float("inf")),
                "rss_longitudinal_margin": safety_info.get("rss_longitudinal_margin", float("inf")),
                "rss_lateral_margin": safety_info.get("rss_lateral_margin", float("inf")),
                "rss_risk_score": safety_info.get("rss_risk_score", 0.0),
                "contact_results": collision_info.get("contact_results", ""),
                "actual_action_modified": safety_info.get("actual_action_modified", False),
                "hard_risk": safety_info.get("hard_risk", False),
                "hard_risk_reason": safety_info.get("hard_risk_reason", ""),
                "hard_recovery_moving_bypass": safety_info.get("hard_recovery_moving_bypass", False),
                "hard_recovery_entered_safety_hold": safety_info.get("hard_recovery_entered_safety_hold", False),
                "hard_bypass_side": safety_info.get("hard_bypass_side", ""),
                "hard_bypass_remaining": safety_info.get("hard_bypass_remaining", 0),
                "lane_change_first_required": safety_info.get("lane_change_first_required", False),
                "lane_change_first_selected": safety_info.get("lane_change_first_selected", False),
                "low_risk_gate_passed": safety_info.get("low_risk_gate_passed", False),
                "low_risk_gate_rejected_reason": safety_info.get("low_risk_gate_rejected_reason", ""),
                "safety_hold_active": safety_info.get("safety_hold_active", False),
                "safety_hold_remaining": safety_info.get("safety_hold_remaining", 0),
                "recent_contact_steps": safety_info.get("recent_contact_steps", 0),
                "nearest_static_object_type": safety_info.get("nearest_static_object_type", ""),
                "nearest_static_object_distance": safety_info.get("nearest_static_object_distance", float("inf")),
                "nearest_static_forward_distance": safety_info.get("nearest_static_forward_distance", float("inf")),
                "nearest_static_lateral_gap": safety_info.get("nearest_static_lateral_gap", float("inf")),
                "front_static_blocking_distance": safety_info.get("front_static_blocking_distance", float("inf")),
                "static_distance_immediate_risk": safety_info.get("static_distance_immediate_risk", False),
                "deadlock_escape_candidate": safety_info.get("deadlock_escape_candidate", False),
                "ttc_vehicle_min": safety_info.get("ttc_vehicle_min", float("inf")),
                "ttc_static_min": safety_info.get("ttc_static_min", float("inf")),
                "boundary_closing_rate": safety_info.get("boundary_closing_rate", 0.0),
                "pre_contact_intervention": safety_info.get("pre_contact_intervention", False),
                "post_contact_intervention": safety_info.get("post_contact_intervention", False),
                "first_risk_detected_step": safety_info.get("first_risk_detected_step", -1),
                "risk_to_intervention_delay": safety_info.get("risk_to_intervention_delay", -1),
                "intervention_to_first_cost_delay": safety_info.get("intervention_to_first_cost_delay", ""),
            }

            # Add safety_info fields
            if safety_info:
                for key in ["raw_predicted_collision", "raw_predicted_out_of_road", "raw_predicted_cost_risk",
                            "raw_min_vehicle_margin", "raw_min_static_margin", "raw_min_boundary_margin", "raw_deadlock_risk",
                            "raw_failure_reason",
                            "allow_intervention", "intervention_reason", "intervention_rejected_reason",
                            "candidate_predicted_collision", "candidate_predicted_out_of_road",
                            "candidate_predicted_cost_risk", "candidate_min_vehicle_margin",
                            "candidate_min_static_margin", "candidate_min_boundary_margin", "vehicle_margin_worse",
                            "intervention_score_gain", "exception_type", "exception_message", "exception_traceback"]:
                    if key in safety_info:
                        step_diagnostic[key] = safety_info.get(key)

            # Write step diagnostic
            if log_step_diagnostics:
                diag_logger.write_step_diagnostic(step_diagnostic)

            # Store for episode analysis
            if safety_info:
                safety_info["step"] = step_count
                safety_info["cost"] = step_cost
                safety_info["step_cost"] = step_cost
                safety_info["episode_cost_so_far"] = episode_cost_so_far
                safety_info["contact_results"] = collision_info.get("contact_results", "")
                recovery_step_infos.append(safety_info)

            current_ep_step_diagnostics.append(step_diagnostic)
            all_step_diagnostics.append(step_diagnostic)

            if use_render:
                env.render()

            if progress_interval and step_count % progress_interval == 0:
                print("[EvalProgress] env {} ep_in_env {} step {}".format(env_index, num_ep_in + 1, step_count))

            if d or step_count >= 3000:
                ep_times.append(time.time() - last_time)
                last_time = time.time()

                ep_count += 1
                num_ep_in += 1

                env_id_recorded = EVAL_ENV_START + env_index
                num_ep_in_recorded = num_ep_in

                # Get episode result
                res = env.get_episode_result()
                res.update(dict(
                    success=collision_info.get("arrive_dest", 0),
                    crash=collision_info.get("crash", False),
                    crash_vehicle=collision_info.get("crash_vehicle", False),
                    out_of_road=collision_info.get("out_of_road", False),
                    route_completion=collision_info.get("route_completion", 0),
                    velocity_step_mean=np.mean(ep_velocities) if ep_velocities else 0,
                ))
                recovery_summary = summarize_recovery_episode(recovery_step_infos)
                ep_velocities = []
                recovery_step_infos = []

                # Analyze cost and crash events
                episode_analysis = analyze_episode_events(current_ep_step_diagnostics, safety_info, diagnostic_cost_threshold)

                # Write event windows if needed
                if log_event_windows:
                    write_event_windows(diag_logger, current_ep_step_diagnostics, episode_analysis, ep_count, ckpt_index, env_id_recorded, num_ep_in_recorded)

                # Add high cost episode if needed
                if log_high_cost_episodes and episode_analysis.get("high_cost_episode", False):
                    high_cost_ep = build_high_cost_episode_row(res, episode_analysis, recovery_summary, ep_count, ckpt_index, env_id_recorded, num_ep_in_recorded)
                    diag_logger.add_high_cost_episode(high_cost_ep)

                res["episode"] = ep_count
                res["ckpt_index"] = ckpt_index
                res["env_id"] = env_id_recorded
                res["num_ep_in_one_env"] = num_ep_in_recorded

                # Add analysis to result
                res.update(episode_analysis)
                res.update(recovery_summary)

                saved_results.append(res)
                df = pd.DataFrame(saved_results)

                print(
                    "Env {:3d} | ep_in_env {:2d} | total_ep {:4d} | steps {:5d} | "
                    "ep_time {:.2f}s | total_time {:.2f}s | ckpt: {} | cost: {:.2f}".format(
                        env_index, num_ep_in, ep_count, step_count,
                        ep_times[-1], time.time() - start, ckpt_name,
                        res.get("episode_cost", 0.0)
                    )
                )
                append_recovery_episode_log(
                    recovery_episode_log_csv,
                    recovery_summary,
                    {
                        "episode": ep_count,
                        "ckpt_index": ckpt_index,
                        "env_id": env_id_recorded,
                        "num_ep_in_one_env": num_ep_in_recorded,
                    },
                )
                print(pretty_print(compact_episode_result_for_terminal(res)))

                tmp_path = osp.join(folder_name, "{}_tmp.csv".format(ckpt_name))
                df.to_csv(tmp_path)

                # Clear step diagnostics buffer
                current_ep_step_diagnostics = []
                step_count = 0

                # Advance to next env seed
                if num_ep_in >= num_ep_in_one_env:
                    env_index += 1
                    num_ep_in = 0
                    if env_index >= end_env_index:
                        break

                o = reset_eval_env(env, EVAL_ENV_START + env_index)
                if predictive_filter is not None:
                    predictive_filter.reset_episode()

    except Exception:
        raise
    finally:
        env.close()

    # Write high cost episodes
    diag_logger.write_high_cost_episodes()

    df = pd.DataFrame(saved_results)

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
    print_runtime_assurance_validation_stats(all_step_diagnostics)

    df["model_name"] = ckpt_name
    return df


def analyze_episode_events(step_diagnostics, safety_info, cost_threshold):
    """Analyze cost and crash events in an episode."""
    result = {
        "episode_cost": 0.0,
        "cost_steps_total": 0,
        "cost_streak_current": 0,
        "cost_streak_max": 0,
        "first_cost_step": -1,
        "first_cost_reason": "",
        "first_cost_after_intervention_step": -1,
        "first_crash_step": -1,
        "first_crash_type": "",
        "first_crash_after_intervention_step": -1,
        "first_contact_step": -1,
        "contact_steps_total": 0,
        "contact_streak_max": 0,
        "crash_vehicle_steps": 0,
        "crash_object_steps": 0,
        "crash_sidewalk_steps": 0,
        "crash_building_steps": 0,
        "crash_human_steps": 0,
        "traffic_object_contact_steps": 0,
        "static_object_contact_steps": 0,
        "filter_intervention_count": 0,
        "filter_intervention_rate": 0.0,
        "last_intervention_step": -1,
        "last_intervention_before_first_cost": -1,
        "last_intervention_before_first_crash": -1,
        "cost_after_intervention_steps_5": 0,
        "cost_after_intervention_steps_10": 0,
        "crash_within_5_steps_after_intervention": 0,
        "crash_within_10_steps_after_intervention": 0,
        "out_of_road_after_intervention_steps_5": 0,
        "out_of_road_after_intervention_steps_10": 0,
        "high_cost_episode": False,
        "high_cost_reason_guess": "",
        "cost_started": False,
        # === 新增：first cost 归因 ===
        "first_cost_caused_by_raw_passthrough": False,
        "first_cost_after_action_modified": False,
        "first_cost_contact_type": "",
        "first_cost_raw_predicted_safe": False,
        "first_cost_emergency_check_missed": False,
        "pre_first_cost_intervention_count": 0,
        "post_first_cost_intervention_count": 0,
        "first_hard_risk_step": -1,
        "first_intervention_step": -1,
        "missed_first_cost": False,
        "late_intervention": False,
        "low_risk_false_negative_count": 0,
        "cooldown_false_negative_count": 0,
        "crash_attribution": "",
    }

    if not step_diagnostics:
        return result

    # Calculate episode cost
    episode_cost = step_diagnostics[-1].get("episode_cost_so_far", 0.0) if step_diagnostics else 0.0
    result["episode_cost"] = episode_cost
    result["high_cost_episode"] = episode_cost > cost_threshold

    # Analyze each step
    last_intervention_step = -1
    last_intervention_before_cost = -1
    last_intervention_before_crash = -1

    cost_streak = 0
    contact_streak = 0

    for i, step in enumerate(step_diagnostics):
        step_num = i + 1
        if result["first_hard_risk_step"] < 0 and bool(step.get("hard_risk", False)):
            result["first_hard_risk_step"] = step_num

        # Cost analysis
        if step.get("cost", 0.0) > 0 or step.get("step_cost", 0.0) > 0:
            result["cost_steps_total"] += 1
            cost_streak += 1
            result["cost_streak_current"] = cost_streak
            if bool(step.get("low_risk_gate_passed", False)):
                result["low_risk_false_negative_count"] += 1
            if (
                step.get("fallback_reason", "") == "cooldown_or_passthrough"
                or step.get("intervention_reason", "") == "cooldown"
            ):
                result["cooldown_false_negative_count"] += 1

            if not result["cost_started"]:
                result["cost_started"] = True
                result["first_cost_step"] = step_num

                # Determine first cost reason
                if step.get("crash_vehicle"):
                    result["first_cost_reason"] = "crash_vehicle"
                    result["first_cost_contact_type"] = "vehicle"
                elif step.get("crash_object"):
                    result["first_cost_reason"] = "crash_object"
                    result["first_cost_contact_type"] = "object"
                elif step.get("crash_sidewalk"):
                    result["first_cost_reason"] = "crash_sidewalk"
                    result["first_cost_contact_type"] = "sidewalk"
                elif step.get("crash_building"):
                    result["first_cost_reason"] = "crash_building"
                    result["first_cost_contact_type"] = "building"
                elif step.get("out_of_road"):
                    result["first_cost_reason"] = "out_of_road"
                    result["first_cost_contact_type"] = "out_of_road"
                elif step.get("out_of_route"):
                    result["first_cost_reason"] = "out_of_route"
                    result["first_cost_contact_type"] = "out_of_route"
                else:
                    result["first_cost_reason"] = "unknown_cost_source"
                    result["first_cost_contact_type"] = "unknown"

                # === First cost 归因分析 ===
                # 1. 检查 actual_action_modified
                if step.get("actual_action_modified", False):
                    result["first_cost_after_action_modified"] = True

                # 2. 检查 filter_intervened（兼容旧字段）
                if step.get("filter_intervened", False):
                    result["first_cost_after_action_modified"] = True

                # 3. 检查 raw_action_predicted_safe
                if step.get("selected_candidate_type") in ["raw_action_safe", "raw_action_no_front_blocker"]:
                    result["first_cost_raw_predicted_safe"] = True
                    result["first_cost_caused_by_raw_passthrough"] = True

                # 4. 检查 emergency_check_missed
                if step.get("selected_candidate_type") in ["raw_action_safe", "raw_action_no_front_blocker"]:
                    if step.get("emergency_detected", False) == False:
                        result["first_cost_emergency_check_missed"] = True
                    # 如果 emergency_detected=True 但仍然发生了 cost，说明 emergency check 有漏检
                    elif step.get("emergency_detected", False) == True:
                        result["first_cost_emergency_check_missed"] = False  # 检测到了但没拦住

                # 5. 如果不是 raw_action 被预测安全的情况，则认为是 action 被修改后仍然发生 cost
                if not result["first_cost_raw_predicted_safe"]:
                    if result["first_cost_after_action_modified"]:
                        # action 被修改后仍然发生 cost，说明修复不够强
                        pass
                    else:
                        # 这种情况不应该发生，但如果发生了说明是其他原因
                        result["first_cost_caused_by_raw_passthrough"] = False
        else:
            cost_streak = 0

        result["cost_streak_max"] = max(result["cost_streak_max"], cost_streak)

        # Crash analysis
        if step.get("crash", False):
            if result["first_crash_step"] < 0:
                result["first_crash_step"] = step_num
                if step.get("crash_vehicle"):
                    result["first_crash_type"] = "crash_vehicle"
                elif step.get("crash_object"):
                    result["first_crash_type"] = "crash_object"
                elif step.get("crash_sidewalk"):
                    result["first_crash_type"] = "crash_sidewalk"
                elif step.get("crash_building"):
                    result["first_crash_type"] = "crash_building"
                elif step.get("crash_human"):
                    result["first_crash_type"] = "crash_human"
                else:
                    result["first_crash_type"] = "crash_unknown"

            # Crash type counts
            if step.get("crash_vehicle"):
                result["crash_vehicle_steps"] += 1
            if step.get("crash_object"):
                result["crash_object_steps"] += 1
            if step.get("crash_sidewalk"):
                result["crash_sidewalk_steps"] += 1
            if step.get("crash_building"):
                result["crash_building_steps"] += 1
            if step.get("crash_human"):
                result["crash_human_steps"] += 1

        # Contact analysis
        if contact_results_nonempty(step.get("contact_results")):
            if result["first_contact_step"] < 0:
                result["first_contact_step"] = step_num
            result["contact_steps_total"] += 1
            contact_streak += 1

            # Determine contact type
            contact_str = str(step.get("contact_results", "")).lower()
            if "vehicle" in contact_str or "traffic" in contact_str:
                result["traffic_object_contact_steps"] += 1
            elif "static" in contact_str or "obstacle" in contact_str:
                result["static_object_contact_steps"] += 1

        result["contact_streak_max"] = max(result["contact_streak_max"], contact_streak)

        # Intervention analysis
        if step.get("filter_intervened", False):
            result["filter_intervention_count"] += 1
            last_intervention_step = step_num
            if result["first_intervention_step"] < 0:
                result["first_intervention_step"] = step_num

            # Check if this intervention is before first cost
            if result["first_cost_step"] < 0 or step_num < result["first_cost_step"]:
                last_intervention_before_cost = step_num
                result["pre_first_cost_intervention_count"] += 1
            else:
                result["post_first_cost_intervention_count"] += 1

            # Check if this intervention is before first crash
            if result["first_crash_step"] < 0 or step_num < result["first_crash_step"]:
                last_intervention_before_crash = step_num

            # Look ahead for cost/crash after intervention
            for look_ahead in range(1, 6):
                if i + look_ahead < len(step_diagnostics):
                    future_step = step_diagnostics[i + look_ahead]
                    if future_step.get("cost", 0.0) > 0 or future_step.get("step_cost", 0.0) > 0:
                        result["cost_after_intervention_steps_5"] += 1
                    if future_step.get("crash", False):
                        result["crash_within_5_steps_after_intervention"] += 1
                    if future_step.get("out_of_road", False):
                        result["out_of_road_after_intervention_steps_5"] += 1

            for look_ahead in range(1, 11):
                if i + look_ahead < len(step_diagnostics):
                    future_step = step_diagnostics[i + look_ahead]
                    if future_step.get("cost", 0.0) > 0 or future_step.get("step_cost", 0.0) > 0:
                        result["cost_after_intervention_steps_10"] += 1
                    if future_step.get("crash", False):
                        result["crash_within_10_steps_after_intervention"] += 1
                    if future_step.get("out_of_road", False):
                        result["out_of_road_after_intervention_steps_10"] += 1

    result["last_intervention_step"] = last_intervention_step
    result["last_intervention_before_first_cost"] = last_intervention_before_cost
    result["last_intervention_before_first_crash"] = last_intervention_before_crash
    result["filter_intervention_rate"] = result["filter_intervention_count"] / max(1, len(step_diagnostics))

    # First cost after intervention
    if last_intervention_before_cost > 0 and result["first_cost_step"] > 0:
        if last_intervention_before_cost < result["first_cost_step"]:
            result["first_cost_after_intervention_step"] = result["first_cost_step"]

    # First crash after intervention
    if last_intervention_before_crash > 0 and result["first_crash_step"] > 0:
        if last_intervention_before_crash < result["first_crash_step"]:
            result["first_crash_after_intervention_step"] = result["first_crash_step"]

    result["missed_first_cost"] = bool(
        result["first_cost_step"] > 0
        and (result["first_intervention_step"] < 0 or result["first_intervention_step"] >= result["first_cost_step"])
    )
    result["late_intervention"] = bool(
        result["first_cost_step"] > 0
        and result["first_intervention_step"] > 0
        and result["first_cost_step"] - result["first_intervention_step"] <= 2
    )

    if result["first_crash_step"] > 0 or result["first_cost_step"] > 0:
        event_step = result["first_crash_step"] if result["first_crash_step"] > 0 else result["first_cost_step"]
        lookback = step_diagnostics[max(0, event_step - 21):event_step]
        detected = any(
            bool(item.get("hard_risk", False))
            or bool(item.get("raw_predicted_collision", False))
            or bool(item.get("raw_predicted_out_of_road", False))
            or float(item.get("raw_predicted_cost_risk", 0.0) or 0.0) > 0.0
            for item in lookback
        )
        intervened = any(bool(item.get("filter_intervened", False)) for item in lookback)
        no_candidate = any(item.get("intervention_reason", "") == "no_safe_candidate" for item in lookback)
        action_limited = any(bool(item.get("action_limited_by_raw_delta", False)) for item in lookback)
        post_contact = any(bool(item.get("post_contact_intervention", False)) for item in lookback)
        if not detected:
            result["crash_attribution"] = "risk_not_detected"
        elif no_candidate:
            result["crash_attribution"] = "detected_but_no_candidate"
        elif action_limited:
            result["crash_attribution"] = "detected_but_action_limited"
        elif result["late_intervention"]:
            result["crash_attribution"] = "detected_but_too_late"
        elif post_contact and not intervened:
            result["crash_attribution"] = "post_contact_stuck"
        elif post_contact:
            result["crash_attribution"] = "post_contact_stuck"
        else:
            result["crash_attribution"] = "detected_but_not_prevented"

    # Determine high cost reason
    if result["high_cost_episode"]:
        if result["traffic_object_contact_steps"] > 10:
            result["high_cost_reason_guess"] = "traffic_object_contact_streak"
        elif result["crash_object_steps"] > 5:
            result["high_cost_reason_guess"] = "crash_object_streak"
        elif result["crash_sidewalk_steps"] > 5:
            result["high_cost_reason_guess"] = "crash_sidewalk_streak"
        elif result["crash_building_steps"] > 5:
            result["high_cost_reason_guess"] = "crash_building_streak"
        elif result["crash_vehicle_steps"] > 5:
            result["high_cost_reason_guess"] = "vehicle_collision_streak"
        elif result["cost_streak_max"] > 50:
            result["high_cost_reason_guess"] = "low_speed_cost_stuck"
        else:
            result["high_cost_reason_guess"] = "unknown_cost_source"

    return result


def write_event_windows(diag_logger, step_diagnostics, episode_analysis, ep_num, ckpt_index, env_id, num_ep_in_env):
    """Write event window CSV for cost and crash events."""
    if not step_diagnostics:
        return

    window_size = diag_logger.event_window_size

    # Find first cost step
    first_cost_step = episode_analysis.get("first_cost_step", -1)
    if first_cost_step > 0:
        start_idx = max(0, first_cost_step - window_size - 1)
        end_idx = min(len(step_diagnostics), first_cost_step + window_size)
        cost_window = step_diagnostics[start_idx:end_idx]
        for i, row in enumerate(cost_window):
            row_copy = row.copy()
            row_copy["event_type"] = "cost_window"
            row_copy["window_relative_step"] = start_idx + i + 1 - first_cost_step
            row_copy["is_first_cost_event"] = (start_idx + i + 1) == first_cost_step
            diag_logger.cost_event_rows.append(row_copy)

    # Find first crash step
    first_crash_step = episode_analysis.get("first_crash_step", -1)
    if first_crash_step > 0:
        start_idx = max(0, first_crash_step - window_size - 1)
        end_idx = min(len(step_diagnostics), first_crash_step + window_size)
        crash_window = step_diagnostics[start_idx:end_idx]
        for i, row in enumerate(crash_window):
            row_copy = row.copy()
            row_copy["event_type"] = "crash_window"
            row_copy["window_relative_step"] = start_idx + i + 1 - first_crash_step
            row_copy["is_first_crash_event"] = (start_idx + i + 1) == first_crash_step
            diag_logger.crash_event_rows.append(row_copy)

    # Write event windows
    diag_logger.write_event_windows()


def build_high_cost_episode_row(res, episode_analysis, recovery_summary, ep_num, ckpt_index, env_id, num_ep_in_env):
    """Build a high cost episode row for the summary CSV."""
    row = {
        "ckpt_index": ckpt_index,
        "env_id": env_id,
        "episode": ep_num,
        "episode_cost": episode_analysis.get("episode_cost", 0.0),
        "episode_length": res.get("episode_length", 0),
        "success": res.get("success", 0),
        "crash": res.get("crash", 0),
        "out_of_road": res.get("out_of_road", 0),
        "route_completion": res.get("route_completion", 0.0),
        "velocity_step_mean": res.get("velocity_step_mean", 0.0),
        "first_cost_step": episode_analysis.get("first_cost_step", -1),
        "first_cost_reason": episode_analysis.get("first_cost_reason", ""),
        "cost_streak_max": episode_analysis.get("cost_streak_max", 0),
        "cost_steps_total": episode_analysis.get("cost_steps_total", 0),
        "first_crash_step": episode_analysis.get("first_crash_step", -1),
        "first_crash_type": episode_analysis.get("first_crash_type", ""),
        "contact_steps_total": episode_analysis.get("contact_steps_total", 0),
        "contact_streak_max": episode_analysis.get("contact_streak_max", 0),
        "filter_intervention_count": episode_analysis.get("filter_intervention_count", 0),
        "last_intervention_before_first_cost": episode_analysis.get("last_intervention_before_first_cost", -1),
        "last_intervention_before_first_crash": episode_analysis.get("last_intervention_before_first_crash", -1),
        "pre_first_cost_intervention_count": episode_analysis.get("pre_first_cost_intervention_count", 0),
        "post_first_cost_intervention_count": episode_analysis.get("post_first_cost_intervention_count", 0),
        "first_hard_risk_step": episode_analysis.get("first_hard_risk_step", -1),
        "first_intervention_step": episode_analysis.get("first_intervention_step", -1),
        "missed_first_cost": episode_analysis.get("missed_first_cost", False),
        "late_intervention": episode_analysis.get("late_intervention", False),
        "low_risk_false_negative_count": episode_analysis.get("low_risk_false_negative_count", 0),
        "cooldown_false_negative_count": episode_analysis.get("cooldown_false_negative_count", 0),
        "crash_attribution": episode_analysis.get("crash_attribution", ""),
        "high_cost_reason_guess": episode_analysis.get("high_cost_reason_guess", ""),
    }
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPL MetaDrive checkpoints.")

    # --- Checkpoint location ---
    parser.add_argument("--path", type=str, default="", help="Directory containing checkpoint or full path to .zip")
    parser.add_argument("--ckpt_index", type=int, default=-1, help="Single checkpoint step to evaluate")

    # --- Batch evaluation ---
    parser.add_argument("--start_ckpt", type=int, default=-1, help="First checkpoint step for batch evaluation.")
    parser.add_argument("--num_ckpt", type=int, default=10, help="Number of checkpoints to evaluate.")
    parser.add_argument("--skip", type=int, default=150, help="Step interval between checkpoints.")

    # --- Output ---
    parser.add_argument("--ret_save_folder", type=str, default="evaluate_results/ppl", help="Folder to save evaluation result CSV files.")

    # --- Eval settings ---
    parser.add_argument("--use_render", action="store_true", help="Enable rendering.")
    parser.add_argument("--num_ep_in_one_env", type=int, default=1, help="Episodes per environment seed.")
    parser.add_argument("--total_env_num", type=int, default=50, help="Number of environment seeds.")
    parser.add_argument("--start_env_index", type=int, default=0, help="Start offset from EVAL_ENV_START.")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy during evaluation.")
    parser.add_argument("--progress_interval", type=int, default=0, help="Print progress every N steps. 0 disables.")

    # --- Runtime Assurance ---
    parser.add_argument("--enable_predictive_recovery", action="store_true", help="Enable runtime assurance filter.")
    parser.add_argument("--recovery_intervention_score_margin", type=float, default=PredictiveRecoveryConfig.intervention_score_margin, help="Minimum score improvement for intervention.")
    parser.add_argument("--recovery_max_steer_delta_from_raw", type=float, default=PredictiveRecoveryConfig.max_steer_delta_from_raw, help="Maximum steering delta from raw action.")
    parser.add_argument("--recovery_max_acc_delta_from_raw", type=float, default=PredictiveRecoveryConfig.max_acc_delta_from_raw, help="Maximum acceleration delta from raw action.")
    parser.add_argument("--recovery_log_csv", nargs="?", const="evaluate_results/ppl/recovery_step_log.csv", default="", help="Optional per-step recovery diagnostic CSV path.")
    parser.add_argument("--recovery_episode_log_csv", type=str, default="", help="Optional episode-level recovery diagnostic CSV path.")

    # --- Diagnostic logging ---
    parser.add_argument("--log_step_diagnostics", action="store_true", help="[Diag] Log every step to CSV.")
    parser.add_argument("--log_event_windows", action="store_true", help="[Diag] Log cost/crash event windows to CSV.")
    parser.add_argument("--log_high_cost_episodes", action="store_true", help="[Diag] Log high cost episodes to CSV.")
    parser.add_argument("--diagnostic_cost_threshold", type=float, default=20.0, help="Cost threshold for high-cost episode (default: 20).")

    # --- Debug options ---
    parser.add_argument("--recovery_debug_shadow", action="store_true", help="[Dev] Run filter but execute raw_action.")
    parser.add_argument("--recovery_debug", action="store_true", help="[Dev] Enable verbose recovery fallback diagnostics.")

    args = parser.parse_args()

    deterministic = not args.stochastic
    recovery_config = PredictiveRecoveryConfig(
        intervention_score_margin=args.recovery_intervention_score_margin,
        max_steer_delta_from_raw=args.recovery_max_steer_delta_from_raw,
        max_acc_delta_from_raw=args.recovery_max_acc_delta_from_raw,
        debug=args.recovery_debug,
        log_csv_path=args.recovery_log_csv or "",
        _debug_shadow_record=args.recovery_debug_shadow,
    )

    # --- Decide evaluation mode ---
    if args.path.endswith(".zip"):
        print("===== Evaluating single checkpoint (zip): {} =====".format(args.path))
        ret = evaluate_ppl_once(
            ckpt_path=args.path, ckpt_index=0, folder_name=args.ret_save_folder,
            use_render=args.use_render, num_ep_in_one_env=args.num_ep_in_one_env,
            total_env_num=args.total_env_num, start_env_index=args.start_env_index, deterministic=deterministic,
            enable_predictive_recovery=args.enable_predictive_recovery,
            recovery_config=recovery_config, progress_interval=args.progress_interval,
            recovery_episode_log_csv=args.recovery_episode_log_csv,
            log_step_diagnostics=args.log_step_diagnostics,
            log_event_windows=args.log_event_windows,
            log_high_cost_episodes=args.log_high_cost_episodes,
            diagnostic_cost_threshold=args.diagnostic_cost_threshold,
        )

    elif args.ckpt_index >= 0:
        if not args.path:
            parser.error("--path is required when using --ckpt_index")
        print("===== Evaluating checkpoint {} in {} =====".format(args.ckpt_index, args.path))
        ret = evaluate_ppl_once(
            ckpt_path=args.path, ckpt_index=args.ckpt_index, folder_name=args.ret_save_folder,
            use_render=args.use_render, num_ep_in_one_env=args.num_ep_in_one_env,
            total_env_num=args.total_env_num, start_env_index=args.start_env_index, deterministic=deterministic,
            enable_predictive_recovery=args.enable_predictive_recovery,
            recovery_config=recovery_config, progress_interval=args.progress_interval,
            recovery_episode_log_csv=args.recovery_episode_log_csv,
            log_step_diagnostics=args.log_step_diagnostics,
            log_event_windows=args.log_event_windows,
            log_high_cost_episodes=args.log_high_cost_episodes,
            diagnostic_cost_threshold=args.diagnostic_cost_threshold,
        )

    elif args.start_ckpt >= 0:
        if not args.path:
            parser.error("--path is required when using --start_ckpt")
        all_results = []
        ckpt_indices = list(reversed(range(args.start_ckpt, args.start_ckpt + args.num_ckpt * args.skip, args.skip)))
        print("===== Batch evaluation: {} checkpoints =====".format(len(ckpt_indices)))
        for ckpt_index in ckpt_indices:
            print("\n----- Checkpoint {} -----".format(ckpt_index))
            ret = evaluate_ppl_once(
                ckpt_path=args.path, ckpt_index=ckpt_index, folder_name=args.ret_save_folder,
                use_render=args.use_render, num_ep_in_one_env=args.num_ep_in_one_env,
                total_env_num=args.total_env_num, start_env_index=args.start_env_index, deterministic=deterministic,
                enable_predictive_recovery=args.enable_predictive_recovery,
                recovery_config=recovery_config, progress_interval=args.progress_interval,
                recovery_episode_log_csv=args.recovery_episode_log_csv,
                log_step_diagnostics=args.log_step_diagnostics,
                log_event_windows=args.log_event_windows,
                log_high_cost_episodes=args.log_high_cost_episodes,
                diagnostic_cost_threshold=args.diagnostic_cost_threshold,
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
        ret = None

    else:
        parser.error("Please specify one of: --ckpt_index, --start_ckpt, or provide a direct .zip path via --path.")

    if ret is None and args.ckpt_index < 0 and not args.path.endswith(".zip") and args.start_ckpt < 0:
        print("Evaluation failed.")
    elif ret is not None:
        print("\n\n Evaluation finished successfully.\n")
