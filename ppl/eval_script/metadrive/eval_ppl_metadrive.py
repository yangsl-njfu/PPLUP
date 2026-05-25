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
import traceback

import numpy as np
import pandas as pd

from ppl.experiments.metadrive.driving_env import DrivingEnv
from ppl.ppl import PPL
from ppl.sb3.td3.policies import TD3Policy
from ppl.utils.print_dict_utils import pretty_print, RecorderEnv
from ppl.utils.rss_cbf_filter import RSSCBFConfig, RSSCBFFilter
from ppl.utils.rss_mpc_filter import RSSMPCConfig, RSSMPCFilter
from ppl.utils.train_eval_config import baseline_eval_config

EVAL_ENV_START = baseline_eval_config["start_seed"]  # Evaluation seeds start from the shared eval config.


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


def make_metadrive_env(use_render=None, eval_env_start=EVAL_ENV_START):
    """Build the evaluation environment (no manual control, fixed seed range)."""
    config = copy.deepcopy(baseline_eval_config)
    config.pop("main_exp", None)
    if use_render is None:
        use_render = bool(config.get("use_render", False))
    config["use_render"] = use_render
    config["manual_control"] = False
    config["start_seed"] = eval_env_start
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


def make_rss_cbf_step_record(
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
    method="ppl_rss_cbf",
):
    """Flatten one runtime assurance decision for step-level CSV logging."""
    env_info = env_info or {}
    rss_info = rss_info or {}
    action_delta = float(
        rss_info.get(
            "action_delta",
            np.linalg.norm(np.asarray(internal_action_safe) - np.asarray(internal_action_nominal)),
        )
    )
    return dict(
        ckpt_index=ckpt_index,
        method=method,
        env_id=env_id,
        episode=episode,
        episode_in_env=episode_in_env,
        step=step_in_episode,
        mode=rss_info.get("mode", "unknown"),
        reason=rss_info.get("reason", ""),
        rss_margin=rss_info.get("rss_margin", np.nan),
        rss_margin_current=rss_info.get("rss_margin_current", rss_info.get("rss_margin", np.nan)),
        rss_margin_min_pred=rss_info.get("rss_margin_min_pred", np.nan),
        rss_margin_final_pred=rss_info.get("rss_margin_final_pred", np.nan),
        rss_lateral_clearance_min_pred=rss_info.get("rss_lateral_clearance_min_pred", np.nan),
        rss_lateral_clearance_final_pred=rss_info.get("rss_lateral_clearance_final_pred", np.nan),
        predicted_progress=rss_info.get("predicted_progress", np.nan),
        mpc_success=rss_info.get("mpc_success", np.nan),
        mpc_num_candidates=rss_info.get("mpc_num_candidates", np.nan),
        mpc_num_feasible=rss_info.get("mpc_num_feasible", np.nan),
        mpc_best_cost=rss_info.get("mpc_best_cost", np.nan),
        fallback_used=rss_info.get("fallback_used", False),
        cbf_mode=rss_info.get("cbf_mode", ""),
        cbf_reason=rss_info.get("cbf_reason", ""),
        cbf_action_delta=rss_info.get("cbf_action_delta", np.nan),
        cbf_acc_safe=rss_info.get("cbf_acc_safe", np.nan),
        cbf_steer_safe=rss_info.get("cbf_steer_safe", np.nan),
        cbf_reference_mode=rss_info.get("cbf_reference_mode", ""),
        cbf_reference_action_delta=rss_info.get("cbf_reference_action_delta", np.nan),
        cbf_reference_acc_safe=rss_info.get("cbf_reference_acc_safe", np.nan),
        cbf_reference_steer_safe=rss_info.get("cbf_reference_steer_safe", np.nan),
        cbf_guard_used=rss_info.get("cbf_guard_used", False),
        cbf_guard_delta=rss_info.get("cbf_guard_delta", np.nan),
        selective_mpc_enabled=rss_info.get("selective_mpc_enabled", False),
        mpc_called=rss_info.get("mpc_called", False),
        mpc_call_reason=rss_info.get("mpc_call_reason", ""),
        deadlock_risk=rss_info.get("deadlock_risk", False),
        deadlock_score=rss_info.get("deadlock_score", np.nan),
        deadlock_counter=rss_info.get("deadlock_counter", 0),
        deadlock_window_progress=rss_info.get("deadlock_window_progress", np.nan),
        deadlock_window_avg_speed=rss_info.get("deadlock_window_avg_speed", np.nan),
        deadlock_cbf_active_ratio=rss_info.get("deadlock_cbf_active_ratio", np.nan),
        deadlock_front_active_ratio=rss_info.get("deadlock_front_active_ratio", np.nan),
        deadlock_avg_cbf_delta=rss_info.get("deadlock_avg_cbf_delta", np.nan),
        deadlock_cbf_fallback_ratio=rss_info.get("deadlock_cbf_fallback_ratio", np.nan),
        deadlock_route_completion=rss_info.get("deadlock_route_completion", np.nan),
        deadlock_reason=rss_info.get("deadlock_reason", ""),
        terminal_recoverable=rss_info.get("terminal_recoverable", False),
        terminal_recovery_reason=rss_info.get("terminal_recovery_reason", ""),
        recovery_progress=rss_info.get("recovery_progress", np.nan),
        recovery_margin_improvement=rss_info.get("recovery_margin_improvement", np.nan),
        blocking_object_final=rss_info.get("blocking_object_final", False),
        minimum_risk_stop_used=rss_info.get("minimum_risk_stop_used", False),
        mpc_terminal_feasible=rss_info.get("mpc_terminal_feasible", np.nan),
        mpc_guard_rejected=rss_info.get("mpc_guard_rejected", np.nan),
        cbf_guard_override_used=rss_info.get("cbf_guard_override_used", False),
        cbf_guard_override_reason=rss_info.get("cbf_guard_override_reason", ""),
        certified_recovery_override_used=rss_info.get("certified_recovery_override_used", False),
        certified_recovery_override_reason=rss_info.get("certified_recovery_override_reason", ""),
        guard_reject_reason=rss_info.get("guard_reject_reason", ""),
        recovery_hold_used=rss_info.get("recovery_hold_used", False),
        mpc_failure_reason=rss_info.get("mpc_failure_reason", ""),
        mpc_num_corridors=rss_info.get("mpc_num_corridors", np.nan),
        mpc_num_rss_feasible=rss_info.get("mpc_num_rss_feasible", np.nan),
        mpc_num_road_safe=rss_info.get("mpc_num_road_safe", np.nan),
        mpc_num_terminal_recoverable=rss_info.get("mpc_num_terminal_recoverable", np.nan),
        mpc_num_guard_rejected=rss_info.get("mpc_num_guard_rejected", np.nan),
        mpc_no_rss_feasible_count=rss_info.get("mpc_no_rss_feasible_count", np.nan),
        mpc_no_road_safe_count=rss_info.get("mpc_no_road_safe_count", np.nan),
        mpc_no_terminal_recoverable_count=rss_info.get("mpc_no_terminal_recoverable_count", np.nan),
        mpc_guard_rejected_count=rss_info.get("mpc_guard_rejected_count", np.nan),
        active_recovery_corridor=rss_info.get("active_recovery_corridor", ""),
        selected_recovery_corridor=rss_info.get("selected_recovery_corridor", ""),
        previous_recovery_corridor=rss_info.get("previous_recovery_corridor", ""),
        corridor_switch_used=rss_info.get("corridor_switch_used", False),
        corridor_switch_reason=rss_info.get("corridor_switch_reason", ""),
        left_corridor_available=rss_info.get("left_corridor_available", False),
        right_corridor_available=rss_info.get("right_corridor_available", False),
        forward_corridor_available=rss_info.get("forward_corridor_available", False),
        recenter_corridor_available=rss_info.get("recenter_corridor_available", False),
        drivable_corridor_available=rss_info.get("drivable_corridor_available", False),
        no_left_corridor=rss_info.get("no_left_corridor", True),
        no_right_corridor=rss_info.get("no_right_corridor", True),
        no_forward_corridor=rss_info.get("no_forward_corridor", True),
        no_recenter_corridor=rss_info.get("no_recenter_corridor", True),
        road_boundary_active=rss_info.get("road_boundary_active", False),
        corridor_target_lateral_offset=rss_info.get("corridor_target_lateral_offset", np.nan),
        corridor_target_speed=rss_info.get("corridor_target_speed", np.nan),
        corridor_cost=rss_info.get("corridor_cost", np.nan),
        corridor_terminal_recoverable=rss_info.get("corridor_terminal_recoverable", False),
        first_step_recovery_feasible=rss_info.get("first_step_recovery_feasible", False),
        first_step_recovery_reason=rss_info.get("first_step_recovery_reason", ""),
        recovery_horizon_steps_used=rss_info.get("recovery_horizon_steps_used", np.nan),
        dynamic_vehicle_detected=rss_info.get("dynamic_vehicle_detected", False),
        selected_candidate_family=rss_info.get("selected_candidate_family", ""),
        best_candidate_family=rss_info.get("best_candidate_family", ""),
        num_candidates_brake=rss_info.get("num_candidates_brake", 0),
        num_candidates_creep=rss_info.get("num_candidates_creep", 0),
        num_candidates_left=rss_info.get("num_candidates_left", 0),
        num_candidates_right=rss_info.get("num_candidates_right", 0),
        best_brake_cost=rss_info.get("best_brake_cost", np.nan),
        best_creep_cost=rss_info.get("best_creep_cost", np.nan),
        best_left_cost=rss_info.get("best_left_cost", np.nan),
        best_right_cost=rss_info.get("best_right_cost", np.nan),
        best_left_terminal_recoverable=rss_info.get("best_left_terminal_recoverable", False),
        best_right_terminal_recoverable=rss_info.get("best_right_terminal_recoverable", False),
        best_left_guard_rejected=rss_info.get("best_left_guard_rejected", False),
        best_right_guard_rejected=rss_info.get("best_right_guard_rejected", False),
        left_reject_reason=rss_info.get("left_reject_reason", ""),
        right_reject_reason=rss_info.get("right_reject_reason", ""),
        brake_selected_reason=rss_info.get("brake_selected_reason", ""),
        mpc_time_ms=rss_info.get("mpc_time_ms", np.nan),
        candidate_generation_time_ms=rss_info.get("candidate_generation_time_ms", np.nan),
        candidate_evaluation_time_ms=rss_info.get("candidate_evaluation_time_ms", np.nan),
        num_total_candidates=rss_info.get("num_total_candidates", 0),
        num_random_candidates=rss_info.get("num_random_candidates", 0),
        num_structured_candidates=rss_info.get("num_structured_candidates", 0),
        action_delta=action_delta,
        acc_nominal=rss_info.get("acc_nominal", np.nan),
        acc_safe=rss_info.get("acc_safe", np.nan),
        acc_delta=rss_info.get("acc_delta", np.nan),
        steer_nominal=rss_info.get("steer_nominal", np.nan),
        steer_safe=rss_info.get("steer_safe", np.nan),
        steer_delta=rss_info.get("steer_delta", np.nan),
        step_cost=env_info.get("cost", np.nan),
        cumulative_cost=env_info.get("total_cost", np.nan),
        crash=env_info.get("crash", False),
        crash_vehicle=env_info.get("crash_vehicle", False),
        crash_object=env_info.get("crash_object", False),
        out_of_road=env_info.get("out_of_road", False),
        route_completion=env_info.get("route_completion", np.nan),
        velocity=env_info.get("velocity", np.nan),
        filter_applied=action_delta > 1e-6,
        object_kind=rss_info.get("object_kind", ""),
        d_front=rss_info.get("d_front", np.nan),
        rss_distance=rss_info.get("rss_distance", np.nan),
        env_action_nominal_steer=float(env_action_nominal[0]),
        env_action_nominal_throttle_brake=float(env_action_nominal[1]),
        env_action_safe_steer=float(env_action_safe[0]),
        env_action_safe_throttle_brake=float(env_action_safe[1]),
        internal_action_nominal_acc=float(internal_action_nominal[0]),
        internal_action_nominal_steer=float(internal_action_nominal[1]),
        internal_action_safe_acc=float(internal_action_safe[0]),
        internal_action_safe_steer=float(internal_action_safe[1]),
        adapter_error=rss_info.get("adapter_error", ""),
    )


def summarize_rss_cbf_steps(step_records):
    """Create checkpoint-level runtime assurance diagnostics for console output."""
    if not step_records:
        return dict(
            total_steps=0,
            changed_actions=0,
            changed_rate=0.0,
            modes={},
            cost_by_mode={},
            avg_mpc_num_feasible=0.0,
            fallback_count=0,
            cbf_guard_count=0,
        )

    modes = [record["mode"] for record in step_records]
    mode_counts = {mode: modes.count(mode) for mode in sorted(set(modes))}
    changed_actions = sum(1 for record in step_records if record.get("filter_applied", False))
    cost_by_mode = {}
    for record in step_records:
        cost = record.get("step_cost", np.nan)
        if pd.notna(cost):
            mode = record.get("mode", "unknown")
            cost_by_mode[mode] = cost_by_mode.get(mode, 0.0) + float(cost)

    feasible_values = [
        float(record.get("mpc_num_feasible"))
        for record in step_records
        if pd.notna(record.get("mpc_num_feasible", np.nan))
    ]
    fallback_count = sum(
        1
        for record in step_records
        if record.get("fallback_used", False) or record.get("mode") == "rss_mpc_fallback_to_cbf"
    )
    cbf_guard_count = sum(1 for record in step_records if record.get("cbf_guard_used", False))
    return dict(
        total_steps=len(step_records),
        changed_actions=changed_actions,
        changed_rate=changed_actions / max(len(step_records), 1),
        modes=mode_counts,
        cost_by_mode=cost_by_mode,
        avg_mpc_num_feasible=float(np.mean(feasible_values)) if feasible_values else 0.0,
        fallback_count=fallback_count,
        cbf_guard_count=cbf_guard_count,
    )


def evaluate_ppl_once(
    ckpt_path,
    ckpt_index,
    folder_name,
    use_render=None,
    num_ep_in_one_env=5,
    total_env_num=50,
    deterministic=True,
    rss_cbf=False,
    rss_mpc=False,
    rss_cbf_diagnostics=False,
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

    if rss_cbf and rss_mpc:
        raise ValueError("Use only one runtime assurance mode: --rss_cbf or --rss_mpc")

    os.makedirs(folder_name, exist_ok=True)

    env = make_metadrive_env(use_render, eval_env_start=eval_env_start)
    try:
        policy_function = PolicyFunction(zip_path, env)
    except Exception as e:
        print(f"Failed to load policy from {zip_path}: {e}")
        env.close()
        return None

    method = "ppl_rss_mpc" if rss_mpc else ("ppl_rss_cbf" if rss_cbf else "ppl")
    rss_filter = None
    runtime_label = ""
    step_file_tag = ""
    save_runtime_step_csv = bool(rss_cbf or rss_mpc or rss_cbf_diagnostics)
    if rss_mpc:
        rss_filter = RSSMPCFilter(RSSMPCConfig())
        runtime_label = "RSS-MPC"
        step_file_tag = "rss_mpc"
        print("[RSS-MPC] Runtime assurance enabled. Step diagnostics will be saved.")
    elif rss_cbf:
        rss_filter = RSSCBFFilter(RSSCBFConfig())
        runtime_label = "RSS-CBF"
        step_file_tag = "rss_cbf"
        print("[RSS-CBF] Runtime assurance enabled. Step diagnostics will be saved.")

    saved_results = []
    runtime_step_records = []
    ep_velocities = []
    rss_adapter_error_printed = False
    rss_total_steps = 0
    rss_changed_steps = 0
    rss_mode_counts = {}

    try:
        start = time.time()
        last_time = time.time()
        ep_count = 0
        step_count = 0
        ep_times = []
        runtime_step_flush_interval = 100

        env_index = 0
        num_ep_in = 0
        o = reset_eval_env(env, eval_env_start + env_index)
        if rss_filter is not None and hasattr(rss_filter, "reset"):
            rss_filter.reset()

        while True:
            action = policy_function(o, deterministic=deterministic)[0]
            env_action_nominal = np.asarray(action, dtype=np.float32).copy()
            env_action_safe = env_action_nominal.copy()
            internal_action_nominal = [np.nan, np.nan]
            internal_action_safe = [np.nan, np.nan]
            rss_info = None

            if rss_filter is not None:
                try:
                    rss_state = rss_filter.parse_state_from_metadrive(env)
                    rss_state = rss_filter.augment_state_from_observation(rss_state, o)
                    internal_action_nominal = rss_filter.to_internal_action(env_action_nominal, "metadrive")
                    internal_action_safe, rss_info = rss_filter.filter_action(rss_state, internal_action_nominal)
                    env_action_safe = np.asarray(
                        rss_filter.from_internal_action(internal_action_safe, "metadrive"),
                        dtype=np.float32,
                    )
                    action = env_action_safe
                except Exception as error:
                    traceback.print_exc()
                    internal_action_nominal = rss_filter.to_internal_action(env_action_nominal, "metadrive")
                    internal_action_safe = internal_action_nominal
                    rss_info = {
                        "mode": "fallback_no_safe_candidate",
                        "reason": "adapter_error: {}".format(error),
                        "dynamic_vehicle_detected": False,
                        "rss_margin": np.nan,
                        "action_delta": 0.0,
                        "acc_nominal": internal_action_nominal[0],
                        "acc_safe": internal_action_safe[0],
                        "acc_delta": 0.0,
                        "steer_nominal": internal_action_nominal[1],
                        "steer_safe": internal_action_safe[1],
                        "steer_delta": 0.0,
                        "adapter_error": str(error),
                    }
                    if not rss_adapter_error_printed:
                        print("[{}] Adapter failed once; continuing without filtering. Error: {}".format(runtime_label, error))
                        rss_adapter_error_printed = True

                rss_total_steps += 1
                rss_mode = rss_info.get("mode", "unknown")
                rss_mode_counts[rss_mode] = rss_mode_counts.get(rss_mode, 0) + 1
                if np.linalg.norm(env_action_safe - env_action_nominal) > 1e-6:
                    rss_changed_steps += 1

            o, r, d, info = env.step(action)
            if rss_filter is not None and hasattr(rss_filter, "update_after_step"):
                rss_filter.update_after_step(info)
            step_count += 1

            if rss_filter is not None and save_runtime_step_csv:
                record = make_rss_cbf_step_record(
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
                    method=method,
                )
                runtime_step_records.append(record)
                if step_count % runtime_step_flush_interval == 0:
                    tmp_step_path = osp.join(folder_name, "{}_{}_steps_tmp.csv".format(ckpt_name, step_file_tag))
                    pd.DataFrame(runtime_step_records).to_csv(tmp_step_path, index=False)

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

                res["episode"] = ep_count
                res["ckpt_index"] = ckpt_index
                res["env_id"] = env_id_recorded
                res["num_ep_in_one_env"] = num_ep_in_recorded
                res["method"] = method

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
                if rss_filter is not None and save_runtime_step_csv and runtime_step_records:
                    tmp_step_path = osp.join(folder_name, "{}_{}_steps_tmp.csv".format(ckpt_name, step_file_tag))
                    pd.DataFrame(runtime_step_records).to_csv(tmp_step_path, index=False)

                step_count = 0

                # Advance to next env seed if enough episodes in this one
                if num_ep_in >= num_ep_in_one_env:
                    env_index += 1
                    num_ep_in = 0
                    if env_index >= total_env_num:
                        break

                o = reset_eval_env(env, eval_env_start + env_index)
                if rss_filter is not None and hasattr(rss_filter, "reset"):
                    rss_filter.reset()

    except Exception as e:
        raise e
    finally:
        env.close()

    df = pd.DataFrame(saved_results)
    if not df.empty:
        df["method"] = method

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

    if rss_filter is not None and save_runtime_step_csv:
        step_path = osp.join(folder_name, "{}_{}_steps.csv".format(ckpt_name, step_file_tag))
        pd.DataFrame(runtime_step_records).to_csv(step_path, index=False)
        print("{} step-level results saved to: {}".format(runtime_label, step_path))

    if rss_filter is not None:
        rss_summary = summarize_rss_cbf_steps(runtime_step_records)
        changed_rate = rss_changed_steps / max(rss_total_steps, 1)
        if rss_mpc:
            print(
                "[RSS-MPC] total_steps={} changed_actions={} changed_rate={:.4f} "
                "modes={} cost_by_mode={} avg_mpc_num_feasible={:.2f} "
                "fallback_count={} cbf_guard_count={}".format(
                    rss_total_steps,
                    rss_changed_steps,
                    changed_rate,
                    rss_summary["modes"] if runtime_step_records else rss_mode_counts,
                    rss_summary["cost_by_mode"],
                    rss_summary["avg_mpc_num_feasible"],
                    rss_summary["fallback_count"],
                    rss_summary["cbf_guard_count"],
                )
            )
        else:
            print(
                "[RSS-CBF] total_steps={} changed_actions={} changed_rate={:.4f} "
                "modes={} cost_by_mode={}".format(
                    rss_total_steps,
                    rss_changed_steps,
                    changed_rate,
                    rss_summary["modes"] if runtime_step_records else rss_mode_counts,
                    rss_summary["cost_by_mode"],
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
    render_group = parser.add_mutually_exclusive_group()
    render_group.add_argument(
        "--use_render",
        dest="use_render",
        action="store_true",
        help="Enable rendering. Defaults to baseline_eval_config['use_render'].",
    )
    render_group.add_argument(
        "--no_render",
        dest="use_render",
        action="store_false",
        help="Disable rendering even if baseline_eval_config enables it.",
    )
    parser.set_defaults(use_render=baseline_eval_config.get("use_render", False))
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
        "--rss_cbf",
        action="store_true",
        help="Enable RSS-CBF runtime assurance safety filter.",
    )
    parser.add_argument(
        "--rss_mpc",
        action="store_true",
        help="Enable RSS-MPC runtime assurance safety filter.",
    )
    parser.add_argument(
        "--rss_cbf_diagnostics",
        action="store_true",
        help="Save RSS-CBF step-level diagnostics. Enabled automatically by --rss_cbf.",
    )

    args = parser.parse_args()

    if args.rss_cbf and args.rss_mpc:
        raise ValueError("Use only one runtime assurance mode: --rss_cbf or --rss_mpc")

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
            rss_cbf=args.rss_cbf,
            rss_mpc=args.rss_mpc,
            rss_cbf_diagnostics=args.rss_cbf_diagnostics,
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
            rss_cbf=args.rss_cbf,
            rss_mpc=args.rss_mpc,
            rss_cbf_diagnostics=args.rss_cbf_diagnostics,
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
                rss_cbf=args.rss_cbf,
                rss_mpc=args.rss_mpc,
                rss_cbf_diagnostics=args.rss_cbf_diagnostics,
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
