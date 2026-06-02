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
    decision_mode = rss_info.get("mode", "unknown")
    safety_function_mode = rss_info.get("safety_function_mode", rss_info.get("rss_cbf_variant", ""))
    record_mode = safety_function_mode if method == "ppl_rss_cbf" and safety_function_mode else decision_mode
    return dict(
        ckpt_index=ckpt_index,
        method=method,
        env_id=env_id,
        episode=episode,
        episode_in_env=episode_in_env,
        step=step_in_episode,
        mode=record_mode,
        decision_mode=decision_mode,
        safety_function_mode=safety_function_mode,
        rss_cbf_variant=rss_info.get("rss_cbf_variant", ""),
        reason=rss_info.get("reason", ""),
        min_h_2d=rss_info.get("min_h_2d", np.nan),
        current_h_2d=rss_info.get("current_h_2d", np.nan),
        final_h_2d=rss_info.get("final_h_2d", np.nan),
        nominal_H=rss_info.get("nominal_H", np.nan),
        selected_H=rss_info.get("selected_H", np.nan),
        worst_h=rss_info.get("worst_h", np.nan),
        nominal_min_h_2d=rss_info.get("nominal_min_h_2d", np.nan),
        nominal_final_h_2d=rss_info.get("nominal_final_h_2d", np.nan),
        nominal_hard_safe=rss_info.get("nominal_hard_safe", False),
        nominal_boundary_margin_above_threshold=rss_info.get("nominal_boundary_margin_above_threshold", False),
        rss_2d_safety_margin=rss_info.get("rss_2d_safety_margin", np.nan),
        rss_2d_boundary_margin_threshold=rss_info.get("rss_2d_boundary_margin_threshold", np.nan),
        worst_object_id=rss_info.get("worst_object_id", ""),
        worst_object_type=rss_info.get("worst_object_type", ""),
        worst_object_geometry_type=rss_info.get("worst_object_geometry_type", ""),
        worst_object_kind=rss_info.get("worst_object_kind", ""),
        worst_object_relation=rss_info.get("worst_object_relation", ""),
        worst_delta_s=rss_info.get("worst_delta_s", np.nan),
        worst_delta_l=rss_info.get("worst_delta_l", np.nan),
        coordinate_mode=rss_info.get("coordinate_mode", ""),
        frenet_valid=rss_info.get("frenet_valid", False),
        frenet_fallback_reason=rss_info.get("frenet_fallback_reason", ""),
        ego_ref_lane_valid=rss_info.get("ego_ref_lane_valid", False),
        ego_frenet_valid=rss_info.get("ego_frenet_valid", False),
        object_frenet_valid=rss_info.get("object_frenet_valid", False),
        s_ego=rss_info.get("s_ego", np.nan),
        l_ego=rss_info.get("l_ego", np.nan),
        heading_ref_ego=rss_info.get("heading_ref_ego", np.nan),
        v_ego_s=rss_info.get("v_ego_s", np.nan),
        ego_s=rss_info.get("ego_s", rss_info.get("s_ego", np.nan)),
        ego_l=rss_info.get("ego_l", rss_info.get("l_ego", np.nan)),
        ego_v_s=rss_info.get("ego_v_s", rss_info.get("v_ego_s", np.nan)),
        ego_heading_ref=rss_info.get("ego_heading_ref", rss_info.get("heading_ref_ego", np.nan)),
        worst_s_obj=rss_info.get("worst_s_obj", np.nan),
        worst_l_obj=rss_info.get("worst_l_obj", np.nan),
        worst_v_obj_s=rss_info.get("worst_v_obj_s", np.nan),
        object_s=rss_info.get("object_s", rss_info.get("worst_s_obj", np.nan)),
        object_l=rss_info.get("object_l", rss_info.get("worst_l_obj", np.nan)),
        object_v_s=rss_info.get("object_v_s", rss_info.get("worst_v_obj_s", np.nan)),
        old_delta_s=rss_info.get("old_delta_s", np.nan),
        old_delta_l=rss_info.get("old_delta_l", np.nan),
        old_ego_local_delta_s=rss_info.get("old_ego_local_delta_s", np.nan),
        old_ego_local_delta_l=rss_info.get("old_ego_local_delta_l", np.nan),
        worst_long_clearance=rss_info.get("worst_long_clearance", np.nan),
        worst_lat_clearance=rss_info.get("worst_lat_clearance", np.nan),
        worst_d_s_safe=rss_info.get("worst_d_s_safe", np.nan),
        worst_d_l_safe=rss_info.get("worst_d_l_safe", np.nan),
        left_boundary_margin=rss_info.get("left_boundary_margin", np.nan),
        right_boundary_margin=rss_info.get("right_boundary_margin", np.nan),
        min_boundary_margin=rss_info.get("min_boundary_margin", np.nan),
        predicted_lateral_position=rss_info.get("predicted_lateral_position", np.nan),
        predicted_lane_offset=rss_info.get("predicted_lane_offset", np.nan),
        road_boundary_centering_improvement=rss_info.get("road_boundary_centering_improvement", np.nan),
        selected_min_boundary_margin=rss_info.get("selected_min_boundary_margin", np.nan),
        selected_final_boundary_margin=rss_info.get("selected_final_boundary_margin", np.nan),
        selected_boundary_margin_improvement=rss_info.get("selected_boundary_margin_improvement", np.nan),
        selected_centering_score=rss_info.get("selected_centering_score", np.nan),
        selected_speed_preserve_score=rss_info.get("selected_speed_preserve_score", np.nan),
        selected_selection_score=rss_info.get("selected_selection_score", np.nan),
        safety_object_count=rss_info.get("safety_object_count", np.nan),
        fallback_used_unified=rss_info.get("fallback_used", False),
        excessive_braking=rss_info.get("excessive_braking", False),
        selected_steering_changed=rss_info.get("selected_steering_changed", False),
        rss_2d_acc_candidate_count=rss_info.get("rss_2d_acc_candidate_count", np.nan),
        rss_2d_steer_candidate_count=rss_info.get("rss_2d_steer_candidate_count", np.nan),
        selected_action=rss_info.get("selected_action", internal_action_safe),
        nominal_action=rss_info.get("nominal_action", internal_action_nominal),
        filter_intervened=rss_info.get("filter_intervened", action_delta > 1e-6),
        candidate_reject_reasons=rss_info.get("candidate_reject_reasons", ""),
        rss_2d_candidate_count=rss_info.get("candidate_count", np.nan),
        rss_2d_safe_candidate_count=rss_info.get("safe_candidate_count", np.nan),
        rss_2d_road_safe_candidate_count=rss_info.get("road_safe_candidate_count", np.nan),
        rss_2d_candidate_search_mode=rss_info.get("rss_2d_candidate_search_mode", ""),
        rss_2d_candidate_search_stopped_after_stage=rss_info.get(
            "rss_2d_candidate_search_stopped_after_stage", ""
        ),
        build_safety_objects_time=rss_info.get("build_safety_objects_time", np.nan),
        query_local_objects_time=rss_info.get("query_local_objects_time", np.nan),
        nominal_eval_time=rss_info.get("nominal_eval_time", np.nan),
        candidate_generation_time=rss_info.get("candidate_generation_time", np.nan),
        candidate_eval_time=rss_info.get("candidate_eval_time", np.nan),
        geometry_distance_time=rss_info.get("geometry_distance_time", np.nan),
        total_filter_time=rss_info.get("total_filter_time", np.nan),
        safety_object_count_total=rss_info.get("safety_object_count_total", np.nan),
        safety_object_count_local=rss_info.get("safety_object_count_local", np.nan),
        rss_2d_horizon_steps=rss_info.get("horizon_steps", np.nan),
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
        mpc_trigger_policy=rss_info.get("mpc_trigger_policy", ""),
        mpc_trigger_conditions=rss_info.get("mpc_trigger_conditions", ""),
        mpc_trigger_blockers=rss_info.get("mpc_trigger_blockers", ""),
        mpc_trigger_front_exists=rss_info.get("mpc_trigger_front_exists", False),
        mpc_trigger_front_object_kind=rss_info.get("mpc_trigger_front_object_kind", ""),
        mpc_trigger_cbf_mode=rss_info.get("mpc_trigger_cbf_mode", ""),
        mpc_trigger_cbf_delta=rss_info.get("mpc_trigger_cbf_delta", np.nan),
        mpc_trigger_boundary_risk=rss_info.get("mpc_trigger_boundary_risk", False),
        mpc_trigger_boundary_margin=rss_info.get("mpc_trigger_boundary_margin", np.nan),
        mpc_trigger_boundary_margin_source=rss_info.get("mpc_trigger_boundary_margin_source", ""),
        mpc_trigger_boundary_min_pred=rss_info.get("mpc_trigger_boundary_min_pred", np.nan),
        mpc_trigger_boundary_final_pred=rss_info.get("mpc_trigger_boundary_final_pred", np.nan),
        road_boundary_projected_risk=rss_info.get("road_boundary_projected_risk", False),
        road_boundary_current_violation=rss_info.get("road_boundary_current_violation", False),
        road_boundary_nominal_margin_current=rss_info.get("road_boundary_nominal_margin_current", np.nan),
        road_boundary_nominal_margin_min_pred=rss_info.get("road_boundary_nominal_margin_min_pred", np.nan),
        road_boundary_nominal_margin_final_pred=rss_info.get("road_boundary_nominal_margin_final_pred", np.nan),
        road_boundary_prediction_horizon_steps=rss_info.get("road_boundary_prediction_horizon_steps", np.nan),
        road_boundary_trigger_margin_dynamic=rss_info.get("road_boundary_trigger_margin_dynamic", np.nan),
        mpc_trigger_yellow_line=rss_info.get("mpc_trigger_yellow_line", False),
        mpc_trigger_white_line=rss_info.get("mpc_trigger_white_line", False),
        mpc_trigger_on_lane=rss_info.get("mpc_trigger_on_lane", True),
        mpc_trigger_crash_sidewalk=rss_info.get("mpc_trigger_crash_sidewalk", False),
        mpc_trigger_out_of_route=rss_info.get("mpc_trigger_out_of_route", False),
        deadlock_risk=rss_info.get("deadlock_risk", False),
        brake_only_risk=rss_info.get("brake_only_risk", False),
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
        lateral_guard_reject_reason=rss_info.get("lateral_guard_reject_reason", ""),
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
        road_boundary_mpc_used=rss_info.get("road_boundary_mpc_used", False),
        road_boundary_mpc_no_safe_candidate=rss_info.get("road_boundary_mpc_no_safe_candidate", False),
        road_boundary_guard_used=rss_info.get("road_boundary_guard_used", False),
        road_boundary_guard_reason=rss_info.get("road_boundary_guard_reason", ""),
        road_boundary_guard_side=rss_info.get("road_boundary_guard_side", ""),
        road_boundary_guard_acc=rss_info.get("road_boundary_guard_acc", np.nan),
        road_boundary_guard_steer=rss_info.get("road_boundary_guard_steer", np.nan),
        road_boundary_recovery_hold_active=rss_info.get("road_boundary_recovery_hold_active", False),
        road_boundary_recovery_hold_ttl=rss_info.get("road_boundary_recovery_hold_ttl", 0),
        road_boundary_route_endpoint_margin_enabled=rss_info.get("road_boundary_route_endpoint_margin_enabled", False),
        corridor_target_lateral_offset=rss_info.get("corridor_target_lateral_offset", np.nan),
        corridor_target_speed=rss_info.get("corridor_target_speed", np.nan),
        corridor_cost=rss_info.get("corridor_cost", np.nan),
        corridor_terminal_recoverable=rss_info.get("corridor_terminal_recoverable", False),
        first_step_recovery_feasible=rss_info.get("first_step_recovery_feasible", False),
        first_step_recovery_reason=rss_info.get("first_step_recovery_reason", ""),
        recovery_horizon_steps_used=rss_info.get("recovery_horizon_steps_used", np.nan),
        dynamic_vehicle_detected=rss_info.get("dynamic_vehicle_detected", False),
        candidate_family=rss_info.get("candidate_family", ""),
        selected_candidate_family=rss_info.get("selected_candidate_family", ""),
        best_candidate_family=rss_info.get("best_candidate_family", ""),
        is_lateral_escape=rss_info.get("is_lateral_escape", False),
        lateral_certified_gate=rss_info.get("lateral_certified_gate", rss_info.get("lateral_escape_certified", False)),
        predictive_filter_used=rss_info.get("predictive_filter_used", False),
        hard_safe=rss_info.get("hard_safe", False),
        soft_longitudinal_slack=rss_info.get("soft_longitudinal_slack", np.nan),
        recovery_certified=rss_info.get("recovery_certified", False),
        emergency_veto=rss_info.get("emergency_veto", False),
        total_cost=rss_info.get("total_cost", np.nan),
        predictive_candidate_certified=rss_info.get("predictive_candidate_certified", False),
        certified_lateral_recovery=rss_info.get("certified_lateral_recovery", False),
        predictive_reject_reason=rss_info.get("predictive_reject_reason", ""),
        predictive_hard_safe=rss_info.get("predictive_hard_safe", False),
        predictive_total_cost=rss_info.get("predictive_total_cost", np.nan),
        predictive_action_bounds_safe=rss_info.get("predictive_action_bounds_safe", False),
        predictive_no_immediate_collision=rss_info.get("predictive_no_immediate_collision", False),
        predictive_road_boundary_safe=rss_info.get("predictive_road_boundary_safe", False),
        predictive_lateral_hard_safe=rss_info.get("predictive_lateral_hard_safe", False),
        predictive_lateral_rss_safe_or_improving=rss_info.get("predictive_lateral_rss_safe_or_improving", False),
        predictive_catastrophic_side_conflict=rss_info.get("predictive_catastrophic_side_conflict", False),
        predictive_terminal_recoverable=rss_info.get("predictive_terminal_recoverable", False),
        predictive_terminal_recovery_reason=rss_info.get("predictive_terminal_recovery_reason", ""),
        predictive_longitudinal_soft_slack=rss_info.get("predictive_longitudinal_soft_slack", np.nan),
        predictive_one_step_cbf_slack=rss_info.get("predictive_one_step_cbf_slack", np.nan),
        predictive_soft_longitudinal_slack=rss_info.get("predictive_soft_longitudinal_slack", np.nan),
        predictive_longitudinal_slack_penalty=rss_info.get("predictive_longitudinal_slack_penalty", np.nan),
        predictive_current_front_gap=rss_info.get("predictive_current_front_gap", np.nan),
        predictive_first_step_front_gap=rss_info.get("predictive_first_step_front_gap", np.nan),
        emergency_guard_used=rss_info.get("emergency_guard_used", False),
        emergency_guard_reject_reason=rss_info.get("emergency_guard_reject_reason", ""),
        first_acc=rss_info.get("first_acc", np.nan),
        first_steer=rss_info.get("first_steer", np.nan),
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
        lateral_rss_margin=rss_info.get("lateral_rss_margin", np.nan),
        front_object_path_overlap=rss_info.get("front_object_path_overlap", False),
        front_object_lateral_distance=rss_info.get("front_object_lateral_distance", np.nan),
        front_object_lateral_margin=rss_info.get("front_object_lateral_margin", np.nan),
        front_object_deconflicted=rss_info.get("front_object_deconflicted", False),
        longitudinal_constraint_relaxed_by_lateral_escape=rss_info.get("longitudinal_constraint_relaxed_by_lateral_escape", False),
        initial_lateral_rss_margin=rss_info.get("initial_lateral_rss_margin", np.nan),
        final_lateral_rss_margin=rss_info.get("final_lateral_rss_margin", np.nan),
        predicted_lateral_distance=rss_info.get("predicted_lateral_distance", np.nan),
        predicted_lateral_rss_margin=rss_info.get("predicted_lateral_rss_margin", np.nan),
        predicted_path_overlap=rss_info.get("predicted_path_overlap", False),
        predicted_path_overlap_reducing=rss_info.get("predicted_path_overlap_reducing", False),
        terminal_lateral_separation_safe=rss_info.get("terminal_lateral_separation_safe", False),
        terminal_lateral_deconflicted=rss_info.get("terminal_lateral_deconflicted", False),
        road_boundary_safe=rss_info.get("road_boundary_safe", False),
        road_boundary_margin_current=rss_info.get("road_boundary_margin_current", np.nan),
        road_boundary_margin_min_pred=rss_info.get("road_boundary_margin_min_pred", np.nan),
        road_boundary_margin_final_pred=rss_info.get("road_boundary_margin_final_pred", np.nan),
        road_boundary_margin_improvement=rss_info.get("road_boundary_margin_improvement", np.nan),
        road_boundary_hard_margin=rss_info.get("road_boundary_hard_margin", np.nan),
        road_boundary_target_margin=rss_info.get("road_boundary_target_margin", np.nan),
        road_boundary_high_speed_margin=rss_info.get("road_boundary_high_speed_margin", np.nan),
        road_boundary_high_speed_requires_brake=rss_info.get("road_boundary_high_speed_requires_brake", False),
        road_boundary_nearest_side=rss_info.get("road_boundary_nearest_side", ""),
        road_boundary_steer_toward_boundary=rss_info.get("road_boundary_steer_toward_boundary", False),
        road_boundary_away_steer_required=rss_info.get("road_boundary_away_steer_required", False),
        road_boundary_away_steer_safe=rss_info.get("road_boundary_away_steer_safe", False),
        road_boundary_away_progress_recoverable=rss_info.get("road_boundary_away_progress_recoverable", False),
        road_boundary_low_speed_recovery=rss_info.get("road_boundary_low_speed_recovery", False),
        road_boundary_margin_source=rss_info.get("road_boundary_margin_source", ""),
        ego_dist_to_left_side=rss_info.get("ego_dist_to_left_side", np.nan),
        ego_dist_to_right_side=rss_info.get("ego_dist_to_right_side", np.nan),
        ego_on_lane=rss_info.get("ego_on_lane", True),
        ego_out_of_route=rss_info.get("ego_out_of_route", False),
        ego_crash_sidewalk=rss_info.get("ego_crash_sidewalk", False),
        ego_on_yellow_continuous_line=rss_info.get("ego_on_yellow_continuous_line", False),
        ego_on_white_continuous_line=rss_info.get("ego_on_white_continuous_line", False),
        ego_on_broken_line=rss_info.get("ego_on_broken_line", False),
        ego_left_lane_line_type=rss_info.get("ego_left_lane_line_type", ""),
        ego_right_lane_line_type=rss_info.get("ego_right_lane_line_type", ""),
        ego_left_lane_line_color=rss_info.get("ego_left_lane_line_color", ""),
        ego_right_lane_line_color=rss_info.get("ego_right_lane_line_color", ""),
        ego_left_lane_line_prohibited=rss_info.get("ego_left_lane_line_prohibited", False),
        ego_right_lane_line_prohibited=rss_info.get("ego_right_lane_line_prohibited", False),
        immediate_longitudinal_margin_safe=rss_info.get("immediate_longitudinal_margin_safe", False),
        conservative_longitudinal_margin_safe=rss_info.get("conservative_longitudinal_margin_safe", False),
        critical_longitudinal_margin_safe=rss_info.get("critical_longitudinal_margin_safe", False),
        lateral_rss_improvement=rss_info.get("lateral_rss_improvement", 0.0),
        path_overlap_reduced=rss_info.get("path_overlap_reduced", False),
        terminal_deconflicted=rss_info.get("terminal_deconflicted", False),
        first_step_lateral_margin_improves=rss_info.get("first_step_lateral_margin_improves", False),
        first_step_path_overlap_reduces=rss_info.get("first_step_path_overlap_reduces", False),
        first_step_lateral_distance_increases=rss_info.get("first_step_lateral_distance_increases", False),
        lateral_escape_candidate=rss_info.get("lateral_escape_candidate", False),
        lateral_escape_certified=rss_info.get("lateral_escape_certified", False),
        lateral_escape_certification_reason=rss_info.get("lateral_escape_certification_reason", ""),
        lateral_escape_reject_reason=rss_info.get("lateral_escape_reject_reason", ""),
        lateral_escape_used_relaxed_longitudinal_gate=rss_info.get("lateral_escape_used_relaxed_longitudinal_gate", False),
        lateral_escape_rejected_by_conservative_gate=rss_info.get("lateral_escape_rejected_by_conservative_gate", False),
        lateral_escape_rejected_by_critical_margin=rss_info.get("lateral_escape_rejected_by_critical_margin", False),
        lateral_escape_lateral_rss_safe=rss_info.get("lateral_escape_lateral_rss_safe", False),
        lateral_escape_lateral_margin_improved=rss_info.get("lateral_escape_lateral_margin_improved", False),
        lateral_escape_path_overlap_reduced=rss_info.get("lateral_escape_path_overlap_reduced", False),
        lateral_escape_terminal_recoverable=rss_info.get("lateral_escape_terminal_recoverable", False),
        lateral_escape_terminal_reason=rss_info.get("lateral_escape_terminal_reason", ""),
        lateral_escape_low_speed_creep=rss_info.get("lateral_escape_low_speed_creep", False),
        lateral_escape_no_immediate_collision_risk=rss_info.get("lateral_escape_no_immediate_collision_risk", False),
        lateral_escape_steer_toward_escape=rss_info.get("lateral_escape_steer_toward_escape", False),
        certified_lateral_escape_used=rss_info.get("certified_lateral_escape_used", False),
        certified_lateral_escape_side=rss_info.get("certified_lateral_escape_side", ""),
        certified_lateral_escape_reason=rss_info.get("certified_lateral_escape_reason", ""),
        lateral_escape_guard_pass_through=rss_info.get("lateral_escape_guard_pass_through", False),
        lateral_escape_guard_reject_reason=rss_info.get("lateral_escape_guard_reject_reason", ""),
        guard_rejected_lateral_escape=rss_info.get("guard_rejected_lateral_escape", False),
        mpc_action_before_guard_acc=rss_info.get("mpc_action_before_guard", [np.nan, np.nan])[0] if isinstance(rss_info.get("mpc_action_before_guard"), list) else np.nan,
        mpc_action_before_guard_steer=rss_info.get("mpc_action_before_guard", [np.nan, np.nan])[1] if isinstance(rss_info.get("mpc_action_before_guard"), list) else np.nan,
        action_after_guard_acc=rss_info.get("action_after_guard", [np.nan, np.nan])[0] if isinstance(rss_info.get("action_after_guard"), list) else np.nan,
        action_after_guard_steer=rss_info.get("action_after_guard", [np.nan, np.nan])[1] if isinstance(rss_info.get("action_after_guard"), list) else np.nan,
        right_steer_value_used=rss_info.get("right_steer_value_used", np.nan),
        left_steer_value_used=rss_info.get("left_steer_value_used", np.nan),
        selected_steer_before_guard=rss_info.get("selected_steer_before_guard", np.nan),
        selected_steer_after_guard=rss_info.get("selected_steer_after_guard", np.nan),
        right_road_safe=rss_info.get("right_road_safe", False),
        left_road_safe=rss_info.get("left_road_safe", False),
        right_lateral_rss_safe=rss_info.get("right_lateral_rss_safe", False),
        left_lateral_rss_safe=rss_info.get("left_lateral_rss_safe", False),
        right_escape_available=rss_info.get("right_escape_available", False),
        left_escape_available=rss_info.get("left_escape_available", False),
        right_escape_reject_reason=rss_info.get("right_escape_reject_reason", ""),
        left_escape_reject_reason=rss_info.get("left_escape_reject_reason", ""),
        best_left_lateral_rss_margin=rss_info.get("best_left_lateral_rss_margin", np.nan),
        best_right_lateral_rss_margin=rss_info.get("best_right_lateral_rss_margin", np.nan),
        right_candidate_generated=rss_info.get("right_candidate_generated", False),
        left_candidate_generated=rss_info.get("left_candidate_generated", False),
        certified_lateral_creep_available=rss_info.get("certified_lateral_creep_available", False),
        certified_lateral_creep_used=rss_info.get("certified_lateral_creep_used", False),
        certified_lateral_creep_side=rss_info.get("certified_lateral_creep_side", ""),
        certified_lateral_creep_reason=rss_info.get("certified_lateral_creep_reason", ""),
        brake_selected_despite_certified_creep=rss_info.get("brake_selected_despite_certified_creep", False),
        brake_selected_despite_lateral_escape_available=rss_info.get("brake_selected_despite_lateral_escape_available", False),
        lateral_escape_throttle_suppressed=rss_info.get("lateral_escape_throttle_suppressed", False),
        creep_suppression_reason=rss_info.get("creep_suppression_reason", ""),
        lateral_creep_failure_reason=rss_info.get("lateral_creep_failure_reason", ""),
        selected_acc_before_guard=rss_info.get("selected_acc_before_guard", np.nan),
        selected_acc_after_guard=rss_info.get("selected_acc_after_guard", np.nan),
        selected_throttle_before_guard=rss_info.get("selected_throttle_before_guard", np.nan),
        selected_throttle_after_guard=rss_info.get("selected_throttle_after_guard", np.nan),
        selected_brake_before_guard=rss_info.get("selected_brake_before_guard", np.nan),
        selected_brake_after_guard=rss_info.get("selected_brake_after_guard", np.nan),
        creep_acc_value_used=rss_info.get("creep_acc_value_used", np.nan),
        action_mapping_note=rss_info.get("action_mapping_note", ""),
        invalid_lateral_escape_no_creep=rss_info.get("invalid_lateral_escape_no_creep", False),
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
    enable_2d_rss_cbf=True,
    rss_2d_power=None,
    rss_2d_lateral_margin=None,
    rss_2d_eps=None,
    rss_2d_use_superellipse=True,
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
        rss_config = RSSMPCConfig(enable_2d_rss_cbf=enable_2d_rss_cbf)
        if rss_2d_power is not None:
            rss_config.rss_2d_power = float(rss_2d_power)
        if rss_2d_lateral_margin is not None:
            rss_config.rss_2d_lateral_margin = float(rss_2d_lateral_margin)
        if rss_2d_eps is not None:
            rss_config.rss_2d_eps = float(rss_2d_eps)
        rss_config.rss_2d_use_superellipse = bool(rss_2d_use_superellipse)
        rss_filter = RSSMPCFilter(rss_config)
        runtime_label = "RSS-MPC"
        step_file_tag = "rss_mpc"
        print("[RSS-MPC] Runtime assurance enabled. Step diagnostics will be saved.")
    elif rss_cbf:
        rss_config = RSSCBFConfig(enable_2d_rss_cbf=enable_2d_rss_cbf)
        if rss_2d_power is not None:
            rss_config.rss_2d_power = float(rss_2d_power)
        if rss_2d_lateral_margin is not None:
            rss_config.rss_2d_lateral_margin = float(rss_2d_lateral_margin)
        if rss_2d_eps is not None:
            rss_config.rss_2d_eps = float(rss_2d_eps)
        rss_config.rss_2d_use_superellipse = bool(rss_2d_use_superellipse)
        rss_filter = RSSCBFFilter(rss_config)
        runtime_label = "RSS-CBF"
        step_file_tag = "rss_cbf"
        print(
            "[RSS-CBF] Runtime assurance enabled. Step diagnostics will be saved. "
            "2D RSS-informed CBF={}".format(enable_2d_rss_cbf)
        )

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
    parser.add_argument(
        "--disable_2d_rss_cbf",
        action="store_true",
        help="Use the original longitudinal RSS-CBF plus lateral gate behavior.",
    )
    parser.add_argument(
        "--rss_2d_power",
        type=float,
        default=None,
        help="Power p for the RSS-informed 2D CBF superellipse. Defaults to RSSCBFConfig.",
    )
    parser.add_argument(
        "--rss_2d_lateral_margin",
        type=float,
        default=None,
        help="Lateral safety scale for the RSS-informed 2D CBF in meters.",
    )
    parser.add_argument(
        "--rss_2d_eps",
        type=float,
        default=None,
        help="Numerical epsilon for the RSS-informed 2D CBF denominators.",
    )
    parser.add_argument(
        "--rss_2d_ellipse",
        action="store_true",
        help="Use the p=2 ellipse form instead of the configured superellipse power.",
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
            enable_2d_rss_cbf=not args.disable_2d_rss_cbf,
            rss_2d_power=args.rss_2d_power,
            rss_2d_lateral_margin=args.rss_2d_lateral_margin,
            rss_2d_eps=args.rss_2d_eps,
            rss_2d_use_superellipse=not args.rss_2d_ellipse,
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
            enable_2d_rss_cbf=not args.disable_2d_rss_cbf,
            rss_2d_power=args.rss_2d_power,
            rss_2d_lateral_margin=args.rss_2d_lateral_margin,
            rss_2d_eps=args.rss_2d_eps,
            rss_2d_use_superellipse=not args.rss_2d_ellipse,
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
                enable_2d_rss_cbf=not args.disable_2d_rss_cbf,
                rss_2d_power=args.rss_2d_power,
                rss_2d_lateral_margin=args.rss_2d_lateral_margin,
                rss_2d_eps=args.rss_2d_eps,
                rss_2d_use_superellipse=not args.rss_2d_ellipse,
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
