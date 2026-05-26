"""Lightweight RSS-constrained MPC runtime assurance filter.

The public runtime path keeps PPL as the nominal policy and runs a cooperative
RSS-MPC / RSS-CBF assurance layer:

    u_nom -> CBF safety reference -> RSS-MPC planner -> CBF shield -> u_safe

This first version uses sampling-based random shooting instead of a nonlinear
optimizer. The rollout model is the same lightweight point-mass / heading-rate
approximation used by the existing RSS helpers; it can be replaced later with a
more accurate kinematic bicycle model without changing the evaluation entry
points.
"""

from __future__ import annotations

import copy
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ppl.utils.rss_cbf_filter import RSSCBFConfig, RSSCBFFilter
from ppl.utils.static_rss_filter import Action, State


@dataclass
class RSSMPCConfig(RSSCBFConfig):
    """Stable defaults for RSS-MPC runtime assurance evaluation."""

    horizon_steps: int = 10
    recovery_horizon_steps: int = 20
    recovery_num_candidates: int = 96
    dt: float = 0.1
    num_samples: int = 64
    min_acc: float = -5.0
    max_acc: float = 2.0
    max_steer: float = 1.0
    steer_gain: float = 0.5
    v_max: float = 30.0

    safety_margin_tolerance: float = -0.5
    comfort_brake: float = -2.0
    strong_brake: float = -5.0
    action_change_tolerance: float = 1e-3

    random_seed: int = 0
    random_acc_std: float = 1.5
    random_steer_std: float = 0.25
    random_smoothing_alpha: float = 0.35

    lateral_clearance_buffer: float = 0.2
    w_lateral_clearance: float = 0.3
    stuck_speed_threshold: float = 0.5
    stuck_progress_target: float = 1.0
    w_stall: float = 80.0
    w_cbf_anchor: float = 0.3

    deadlock_window_steps: int = 20
    deadlock_min_window_steps: int = 10
    deadlock_speed_threshold: float = 0.8
    deadlock_progress_threshold: float = 0.5
    deadlock_cbf_active_ratio_threshold: float = 0.7
    deadlock_front_active_ratio_threshold: float = 0.7
    deadlock_cbf_delta_threshold: float = 0.5
    deadlock_cbf_fallback_ratio_threshold: float = 0.5
    deadlock_score_threshold: float = 0.75
    deadlock_counter_threshold: int = 3
    near_goal_route_completion: float = 0.97

    recovery_progress_threshold: float = 1.2
    recovery_margin_improvement_threshold: float = 0.5
    recovery_lateral_clearance_threshold: float = -0.2
    recovery_lateral_improvement_threshold: float = 0.5
    recovery_terminal_speed_threshold: float = 1.0
    guard_delta_cost_weight: float = 25.0
    guard_brake_cost_weight: float = 20.0
    guard_stall_brake_cost: float = 100.0
    guard_eval_top_k: int = 8
    guard_override_margin_buffer: float = 2.0
    guard_override_fallback_margin_buffer: float = 4.0
    guard_override_min_acc: float = 0.2
    recovery_hold_steps: int = 3
    recovery_hold_margin_buffer: float = 4.0
    corridor_lateral_offset: float = 1.2
    corridor_target_speed: float = 2.5
    corridor_target_progress: float = 2.0
    corridor_target_clearance: float = 0.3
    corridor_required_rss_margin: float = 1.0
    corridor_road_boundary_margin: float = 0.15
    corridor_switch_penalty: float = 15.0
    corridor_keep_margin: float = 3.0
    recovery_corridor_ttl: int = 6
    recovery_side_switch_cooldown: int = 4
    w_corridor_lateral: float = 6.0
    w_corridor_clearance: float = 4.0
    w_road_boundary: float = 500.0
    creep_acc: float = 0.6
    nudge_steer: float = 0.35
    nudge_steer_ratio_threshold: float = 0.5
    bypass_steer: float = 0.8

    recovery_w_intervention: float = 0.2
    recovery_w_cbf_anchor: float = 0.05
    recovery_w_progress: float = 8.0
    recovery_w_speed: float = 0.1

    w_intervention: float = 1.0
    w_smooth: float = 0.5
    w_speed: float = 0.2
    w_progress: float = 2.0
    w_rss_violation: float = 1000.0
    w_brake: float = 0.1

    recovery_brake_penalty_weight: float = 5.0
    recovery_lateral_improvement_bonus: float = 15.0
    recovery_creep_lateral_bonus: float = 10.0
    recovery_long_stall_penalty: float = 200.0
    max_structured_escape_candidates: int = 30
    max_random_escape_candidates: int = 8
    mpc_time_warning_ms: float = 100.0
    mpc_time_critical_ms: float = 500.0

    lateral_rss_deconfliction_threshold: float = 0.0
    lateral_rss_terminal_safe_threshold: float = 0.0
    path_overlap_reduction_threshold: float = 0.1
    certified_lateral_escape_min_margin: float = 1.0
    certified_lateral_escape_min_lateral_margin: float = 0.0
    lateral_creep_immediate_margin_threshold: float = -2.0
    lateral_creep_allow_negative_immediate_margin: bool = True
    lateral_creep_no_collision_distance: float = 1.5
    lateral_escape_critical_longitudinal_margin: float = -1.0
    lateral_escape_max_acc: float = 0.6
    lateral_escape_min_acc: float = 0.05
    lateral_escape_min_lateral_margin_improvement: float = 0.05
    lateral_escape_min_overlap_reduction: float = 0.05


class RSSMPCFilter(RSSCBFFilter):
    """Sampling-based RSS-MPC runtime assurance filter."""

    FORMAL_MODES = {
        "normal",
        "rss_mpc_cbf_only",
        "rss_mpc_intervention",
        "rss_mpc_recovery",
        "rss_mpc_cbf_guard",
        "rss_mpc_fallback_to_cbf",
        "minimum_risk_stop",
        "fallback_no_safe_candidate",
    }

    def __init__(self, config: Optional[RSSMPCConfig] = None):
        self.mpc_config = config or RSSMPCConfig()
        super().__init__(self.mpc_config)
        self.rss_cbf_filter = RSSCBFFilter(self.mpc_config)
        self.rng = np.random.default_rng(self.mpc_config.random_seed)
        self.deadlock_history: Deque[Dict[str, Any]] = deque(maxlen=self.mpc_config.deadlock_window_steps)
        self.deadlock_counter = 0
        self._last_route_completion = math.nan
        self._held_recovery_action: Optional[Action] = None
        self._held_recovery_ttl = 0
        self._last_candidate_families: List[str] = []
        self._last_candidate_corridors: List[Dict[str, Any]] = []
        self.previous_recovery_corridor = ""
        self.active_recovery_corridor = ""
        self.recovery_corridor_ttl = 0
        self.recovery_side_switch_cooldown = 0
        self._consecutive_brake_counter = 0

    def reset(self) -> None:
        """Clear rolling deadlock state at episode reset."""
        self.deadlock_history.clear()
        self.deadlock_counter = 0
        self._last_route_completion = math.nan
        self._held_recovery_action = None
        self._held_recovery_ttl = 0
        self._last_candidate_families = []
        self._last_candidate_corridors = []
        self.previous_recovery_corridor = ""
        self.active_recovery_corridor = ""
        self.recovery_corridor_ttl = 0
        self.recovery_side_switch_cooldown = 0
        self._consecutive_brake_counter = 0

    def update_after_step(self, env_info: Optional[Dict[str, Any]]) -> None:
        """Optionally attach post-step route progress to the newest history sample."""
        env_info = env_info or {}
        route_completion = env_info.get("route_completion", math.nan)
        try:
            route_completion = float(route_completion)
        except Exception:
            route_completion = math.nan
        self._last_route_completion = route_completion
        if self.deadlock_history:
            self.deadlock_history[-1]["route_completion"] = route_completion

    def filter_action(self, state: State, u_nom: Sequence[float]) -> Tuple[Action, Dict[str, Any]]:
        """Selective RSS-MPC recovery.

        The normal path is intentionally conservative: safe RL actions pass
        through unchanged, and unsafe non-deadlocked actions use RSS-CBF only.
        MPC is called only for CBF-induced safe-but-stuck recovery.
        """
        self._tick_corridor_memory()
        u_original = self._clip_action(u_nom)
        cbf_safe, cbf_info = self.rss_cbf_filter.filter_action(state, u_original)
        cbf_safe = self._clip_action(cbf_safe)
        front = self._select_front_rss_object(state)
        deadlock_info = self._update_deadlock_detector(state, u_original, cbf_safe, cbf_info, front)

        if front is None:
            return u_original, self._make_mpc_info(
                state=state,
                obj=None,
                object_kind="none",
                mode="normal",
                reason="no_front_rss_object",
                u_original=u_original,
                u_safe=u_original,
                d_front=math.inf,
                d_rss=0.0,
                rss_margin_current=math.inf,
                dynamic_vehicle_detected=False,
                obstacle_detected=False,
                mpc_success=False,
                mpc_num_candidates=0,
                mpc_num_feasible=0,
                mpc_best_cost=math.nan,
                rss_margin_min_pred=math.inf,
                rss_margin_final_pred=math.inf,
                rss_lateral_clearance_min_pred=math.inf,
                rss_lateral_clearance_final_pred=math.inf,
                predicted_progress=0.0,
                fallback_used=False,
                cbf_info=cbf_info,
                cbf_reference_info=cbf_info,
                cbf_guard_used=False,
                cbf_guard_delta=0.0,
                extra_info=deadlock_info,
            )

        object_kind, obj, d_front, d_rss, current_margin, dynamic_detected, static_detected = front
        rl_is_rss_safe = self._cbf_indicates_nominal_safe(cbf_info, u_original, cbf_safe)
        cbf_reference_action = (
            self._minimum_risk_stop(u_original)
            if cbf_info.get("mode") == "fallback_no_safe_candidate"
            else cbf_safe
        )
        if rl_is_rss_safe:
            return u_original, self._make_mpc_info(
                state=state,
                obj=obj,
                object_kind=object_kind,
                mode="normal",
                reason="rl_action_rss_safe",
                u_original=u_original,
                u_safe=u_original,
                d_front=d_front,
                d_rss=d_rss,
                rss_margin_current=current_margin,
                dynamic_vehicle_detected=dynamic_detected,
                obstacle_detected=static_detected,
                mpc_success=False,
                mpc_num_candidates=0,
                mpc_num_feasible=0,
                mpc_best_cost=math.nan,
                rss_margin_min_pred=math.nan,
                rss_margin_final_pred=math.nan,
                rss_lateral_clearance_min_pred=math.nan,
                rss_lateral_clearance_final_pred=math.nan,
                predicted_progress=math.nan,
                fallback_used=False,
                cbf_info=cbf_info,
                cbf_reference_info=cbf_info,
                cbf_guard_used=False,
                cbf_guard_delta=0.0,
                extra_info=deadlock_info,
            )

        if not deadlock_info["deadlock_risk"]:
            held_recovery = self._take_held_recovery_action(current_margin)
            if held_recovery is not None and cbf_info.get("mode") == "fallback_no_safe_candidate":
                return held_recovery, self._make_mpc_info(
                    state=state,
                    obj=obj,
                    object_kind=object_kind,
                    mode="rss_mpc_recovery",
                    reason="mpc_recovery_hold_no_deadlock_large_margin",
                    u_original=u_original,
                    u_safe=held_recovery,
                    d_front=d_front,
                    d_rss=d_rss,
                    rss_margin_current=current_margin,
                    dynamic_vehicle_detected=dynamic_detected,
                    obstacle_detected=static_detected,
                    mpc_success=True,
                    mpc_num_candidates=0,
                    mpc_num_feasible=0,
                    mpc_best_cost=math.nan,
                    rss_margin_min_pred=math.nan,
                    rss_margin_final_pred=math.nan,
                    rss_lateral_clearance_min_pred=math.nan,
                    rss_lateral_clearance_final_pred=math.nan,
                    predicted_progress=math.nan,
                    fallback_used=False,
                    cbf_info=cbf_info,
                    cbf_reference_info=cbf_info,
                    cbf_guard_used=False,
                    cbf_guard_delta=0.0,
                    extra_info=self._merge_recovery_info(
                        deadlock_info,
                        {
                            "mpc_called": False,
                            "mpc_call_reason": "held_recovery_after_deadlock",
                            "cbf_guard_override_used": True,
                            "cbf_guard_override_reason": "held_certified_recovery_action_with_large_rss_margin",
                            "certified_recovery_override_used": True,
                            "certified_recovery_override_reason": "held_certified_recovery_action_with_large_rss_margin",
                            "recovery_hold_used": True,
                        },
                    ),
                )

            if cbf_info.get("mode") == "fallback_no_safe_candidate":
                u_cbf_only = self._minimum_risk_stop(u_original)
                cbf_only_mode = "minimum_risk_stop"
                cbf_only_reason = "rss_cbf_no_safe_candidate_minimum_risk_stop"
                minimum_risk_stop_used = True
            else:
                u_cbf_only = cbf_safe
                cbf_only_mode = "rss_mpc_cbf_only"
                cbf_only_reason = "rss_cbf_shield_no_deadlock"
                minimum_risk_stop_used = False
            return u_cbf_only, self._make_mpc_info(
                state=state,
                obj=obj,
                object_kind=object_kind,
                mode=cbf_only_mode,
                reason=cbf_only_reason,
                u_original=u_original,
                u_safe=u_cbf_only,
                d_front=d_front,
                d_rss=d_rss,
                rss_margin_current=current_margin,
                dynamic_vehicle_detected=dynamic_detected,
                obstacle_detected=static_detected,
                mpc_success=False,
                mpc_num_candidates=0,
                mpc_num_feasible=0,
                mpc_best_cost=math.nan,
                rss_margin_min_pred=math.nan,
                rss_margin_final_pred=math.nan,
                rss_lateral_clearance_min_pred=math.nan,
                rss_lateral_clearance_final_pred=math.nan,
                predicted_progress=math.nan,
                fallback_used=False,
                cbf_info=cbf_info,
                cbf_reference_info=cbf_info,
                cbf_guard_used=False,
                cbf_guard_delta=0.0,
                extra_info=self._merge_recovery_info(
                    deadlock_info,
                    {"minimum_risk_stop_used": minimum_risk_stop_used},
                ),
            )

        t_mpc_start = time.perf_counter()
        recovery_horizon = max(1, int(self.mpc_config.recovery_horizon_steps))
        original_horizon = self.mpc_config.horizon_steps
        self.mpc_config.horizon_steps = recovery_horizon
        try:
            t_gen_start = time.perf_counter()
            escape_pack = self._generate_deadlock_escape_candidates(
                state=state,
                obj=obj,
                object_kind=object_kind,
                current_margin=current_margin,
                u_nom=u_original,
                u_cbf=cbf_reference_action,
            )
            t_gen_end = time.perf_counter()
        finally:
            self.mpc_config.horizon_steps = original_horizon

        candidates = escape_pack["sequences"]
        corridor_info = escape_pack["info"]
        num_structured = sum(1 for f in self._last_candidate_families if f not in ("random_shooting", "corridor_random"))
        num_random = len(candidates) - num_structured
        total_candidates = len(candidates)

        t_eval_start = time.perf_counter()
        feasible_count = 0
        terminal_feasible_count = 0
        no_rss_feasible_count = 0
        no_road_safe_count = 0
        first_step_rejected_count = 0
        no_terminal_recoverable_count = 0
        guard_rejected_count = 0
        terminal_feasible_candidates: List[Dict[str, Any]] = []
        guard_rejected_candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

        def _family_stats_entry() -> Dict[str, Any]:
            return {
                "count": 0,
                "best_cost": math.inf,
                "terminal_recoverable": False,
                "guard_rejected": False,
                "reject_reason": "",
                "lateral_rss_safe": False,
                "lateral_margin_improved": False,
                "path_overlap_reduced": False,
                "road_safe": False,
                "conservative_margin_safe": False,
                "critical_margin_safe": False,
                "relaxed_longitudinal_gate_used": False,
                "rejected_by_critical_margin": False,
                "rejected_by_conservative_gate": False,
                "terminal_reason": "",
                "certified": False,
                "best_lateral_rss_margin": -math.inf,
                "escape_available": False,
            }

        family_stats: Dict[str, Dict[str, Any]] = {
            "brake": _family_stats_entry(),
            "creep": _family_stats_entry(),
            "left": _family_stats_entry(),
            "right": _family_stats_entry(),
        }

        def _set_family_reject_reason(category: str, reason: str) -> None:
            if category in family_stats and not family_stats[category]["reject_reason"]:
                family_stats[category]["reject_reason"] = reason

        conservative_gate_warning_printed = False

        for idx, sequence in enumerate(candidates):
            corridor = self._last_candidate_corridors[idx] if idx < len(self._last_candidate_corridors) else {}
            evaluation = self._evaluate_sequence(
                state=state,
                obj=obj,
                object_kind=object_kind,
                current_margin=current_margin,
                sequence=sequence,
                u_nom=u_original,
                u_cbf=cbf_reference_action,
                corridor=corridor,
                is_deadlock_recovery=True,
            )
            if idx < len(self._last_candidate_families):
                evaluation["recovery_candidate_family"] = self._last_candidate_families[idx]
            else:
                evaluation["recovery_candidate_family"] = "unknown"

            family = evaluation["recovery_candidate_family"]
            evaluation["candidate_family"] = family
            category = self._family_category(family)
            evaluation["candidate_category"] = category
            family_stats[category]["count"] += 1

            rss_feasible, _, _ = self._rss_sequence_feasibility(
                current_margin=current_margin,
                min_margin=evaluation["rss_margin_min_pred"],
                final_margin=evaluation["rss_margin_final_pred"],
            )
            evaluation["rss_feasible"] = bool(rss_feasible)
            if category in ("left", "right"):
                evaluation.update(self._evaluate_lateral_escape_gate(state, evaluation))
            else:
                evaluation.setdefault("lateral_escape_candidate", False)
                evaluation.setdefault("lateral_escape_certified", False)
                evaluation.setdefault("lateral_escape_used_relaxed_longitudinal_gate", False)
                evaluation.setdefault("longitudinal_constraint_relaxed_by_lateral_escape", False)

            lateral_escape_certified = bool(evaluation.get("lateral_escape_certified", False))
            longitudinal_gate_passed = bool(rss_feasible or lateral_escape_certified)
            if evaluation["cost"] < family_stats[category]["best_cost"]:
                family_stats[category]["best_cost"] = evaluation["cost"]
            if evaluation.get("road_boundary_safe", False):
                family_stats[category]["road_safe"] = True
            final_lat_rss = float(evaluation.get("final_lateral_rss_margin", -math.inf))
            if math.isfinite(final_lat_rss) and final_lat_rss > family_stats[category]["best_lateral_rss_margin"]:
                family_stats[category]["best_lateral_rss_margin"] = final_lat_rss
            terminal_lateral_safe = bool(evaluation.get("terminal_lateral_separation_safe", False))
            if terminal_lateral_safe or evaluation.get("lateral_escape_lateral_rss_safe", False):
                family_stats[category]["lateral_rss_safe"] = True
            if evaluation.get("lateral_escape_lateral_margin_improved", False):
                family_stats[category]["lateral_margin_improved"] = True
            if evaluation.get("lateral_escape_path_overlap_reduced", False):
                family_stats[category]["path_overlap_reduced"] = True
            if evaluation.get("conservative_longitudinal_margin_safe", False):
                family_stats[category]["conservative_margin_safe"] = True
            if evaluation.get("critical_longitudinal_margin_safe", False):
                family_stats[category]["critical_margin_safe"] = True
            if evaluation.get("lateral_escape_used_relaxed_longitudinal_gate", False):
                family_stats[category]["relaxed_longitudinal_gate_used"] = True
            if evaluation.get("lateral_escape_rejected_by_critical_margin", False):
                family_stats[category]["rejected_by_critical_margin"] = True
            if evaluation.get("lateral_escape_rejected_by_conservative_gate", False):
                family_stats[category]["rejected_by_conservative_gate"] = True
            if evaluation.get("lateral_escape_certified", False):
                family_stats[category]["certified"] = True
            if evaluation.get("terminal_recoverable", False):
                family_stats[category]["terminal_recoverable"] = True
                if not family_stats[category]["terminal_reason"]:
                    family_stats[category]["terminal_reason"] = str(evaluation.get("terminal_recovery_reason", ""))
            if category in ("left", "right"):
                if evaluation.get("road_boundary_safe", False) and terminal_lateral_safe:
                    family_stats[category]["escape_available"] = True
                reject_reason = str(evaluation.get("lateral_escape_reject_reason", ""))
                if lateral_escape_certified:
                    pass
                elif reject_reason:
                    _set_family_reject_reason(category, reject_reason)
                elif not rss_feasible:
                    _set_family_reject_reason(category, "lateral_escape_rejected_by_conservative_gate")
                elif not evaluation.get("first_step_recovery_feasible", True):
                    _set_family_reject_reason(category, "first_step_recovery_rejected")
                elif not evaluation["terminal_recoverable"]:
                    _set_family_reject_reason(category, "lateral_escape_terminal_not_recoverable")

            if not rss_feasible:
                no_rss_feasible_count += 1
            if (category in ("left", "right")
                    and not longitudinal_gate_passed
                    and not bool(evaluation.get("conservative_longitudinal_margin_safe", False))
                    and bool(evaluation.get("critical_longitudinal_margin_safe", False))
                    and bool(evaluation.get("road_boundary_safe", False))
                    and (bool(evaluation.get("lateral_escape_lateral_rss_safe", False))
                         or bool(evaluation.get("lateral_escape_lateral_margin_improved", False)))
                    and bool(evaluation.get("lateral_escape_path_overlap_reduced", False))
                    and bool(evaluation.get("lateral_escape_terminal_recoverable", False))
                    and bool(evaluation.get("lateral_escape_low_speed_creep", False))
                    and bool(evaluation.get("lateral_escape_no_immediate_collision_risk", False))):
                evaluation["lateral_escape_rejected_by_conservative_gate"] = True
                family_stats[category]["rejected_by_conservative_gate"] = True
                _set_family_reject_reason(category, "lateral_escape_rejected_by_conservative_gate")
                if not conservative_gate_warning_printed:
                    print("[RSS-MPC-DEADLOCK] lateral escape rejected only by conservative longitudinal gate")
                    conservative_gate_warning_printed = True
            if not longitudinal_gate_passed:
                continue
            if not evaluation.get("road_boundary_safe", False):
                no_road_safe_count += 1
                continue
            if not evaluation.get("first_step_recovery_feasible", True):
                first_step_rejected_count += 1
                continue
            if not evaluation["terminal_recoverable"]:
                no_terminal_recoverable_count += 1
                continue
            terminal_feasible_count += 1
            terminal_feasible_candidates.append(evaluation)

        terminal_feasible_candidates.sort(key=lambda item: item["cost"])
        guard_passed: List[Dict[str, Any]] = []
        for evaluation in terminal_feasible_candidates:
            category = self._family_category(evaluation["recovery_candidate_family"])
            guard_eval = self._evaluate_guarded_first_action(state, evaluation)
            if not guard_eval["guard_safe"]:
                guard_rejected_count += 1
                guard_rejected_candidates.append((evaluation, guard_eval))
                if not family_stats[category]["guard_rejected"]:
                    family_stats[category]["guard_rejected"] = True
                    if category in ("left", "right"):
                        family_stats[category]["reject_reason"] = "lateral_escape_guard_suppressed_throttle"
                    else:
                        family_stats[category]["reject_reason"] = guard_eval.get("guard_reject_reason", "")
                continue
            evaluation.update(guard_eval)
            evaluation["cost"] += guard_eval["guard_cost"]
            feasible_count += 1
            guard_passed.append(evaluation)

        t_eval_end = time.perf_counter()

        feasible_by_category: Dict[str, List[Dict[str, Any]]] = {"left": [], "right": [], "creep": [], "brake": []}
        for evaluation in guard_passed:
            cat = self._family_category(evaluation["recovery_candidate_family"])
            feasible_by_category[cat].append(evaluation)

        best: Optional[Dict[str, Any]] = None
        for preferred_category in ("left", "right", "creep", "brake"):
            candidates_in_cat = feasible_by_category[preferred_category]
            if candidates_in_cat:
                candidates_in_cat.sort(key=lambda e: e["cost"])
                best = candidates_in_cat[0]
                break

        t_mpc_end = time.perf_counter()
        mpc_time_ms = (t_mpc_end - t_mpc_start) * 1000.0
        gen_time_ms = (t_gen_end - t_gen_start) * 1000.0
        eval_time_ms = (t_eval_end - t_eval_start) * 1000.0

        if mpc_time_ms > self.mpc_config.mpc_time_critical_ms:
            print("[RSS-MPC-TIMING] CRITICAL: mpc_time_ms={:.1f} > {:.1f}, reducing random samples".format(
                mpc_time_ms, self.mpc_config.mpc_time_critical_ms))
        elif mpc_time_ms > self.mpc_config.mpc_time_warning_ms:
            print("[RSS-MPC-TIMING] WARNING: mpc_time_ms={:.1f} > {:.1f}".format(
                mpc_time_ms, self.mpc_config.mpc_time_warning_ms))

        def _lateral_escape_failure_reason() -> str:
            if family_stats["left"]["count"] <= 0 and family_stats["right"]["count"] <= 0:
                return "lateral_creep_not_generated"
            for side in ("right", "left"):
                stats = family_stats[side]
                if stats["count"] <= 0:
                    continue
                if stats["reject_reason"]:
                    return str(stats["reject_reason"])
                if not stats["road_safe"]:
                    return "lateral_escape_not_road_safe"
                if stats["rejected_by_critical_margin"]:
                    return "lateral_escape_rejected_by_critical_margin"
                if not (stats["lateral_rss_safe"] or stats["lateral_margin_improved"]):
                    return "lateral_escape_lateral_rss_unsafe"
                if not stats["path_overlap_reduced"]:
                    return "lateral_escape_path_overlap_not_reduced"
                if not stats["terminal_recoverable"]:
                    return "lateral_escape_terminal_not_recoverable"
                if stats["guard_rejected"]:
                    return "lateral_escape_guard_suppressed_throttle"
            return ""

        lateral_failure_reason = _lateral_escape_failure_reason()

        if best is not None:
            failure_reason = ""
        elif not corridor_info.get("drivable_corridor_available", False):
            failure_reason = "no_corridor_available"
        elif lateral_failure_reason:
            failure_reason = lateral_failure_reason
        elif no_rss_feasible_count == total_candidates:
            failure_reason = "no_rss_feasible_candidate"
        elif no_road_safe_count > 0 and no_rss_feasible_count + no_road_safe_count == total_candidates:
            failure_reason = "no_road_boundary_safe_candidate"
        elif first_step_rejected_count > 0 and no_rss_feasible_count + no_road_safe_count + first_step_rejected_count == total_candidates:
            failure_reason = "first_step_recovery_rejected"
        elif terminal_feasible_count == 0:
            failure_reason = "no_terminal_recoverable_candidate"
        elif feasible_count == 0:
            failure_reason = "terminal_candidates_guard_rejected"
        else:
            failure_reason = "horizon_too_short_or_no_progress"

        def _diagnostics_block() -> Dict[str, Any]:
            any_lateral_guard_rejected = family_stats["left"]["guard_rejected"] or family_stats["right"]["guard_rejected"]
            lateral_guard_reject_reason = (
                family_stats["right"]["reject_reason"]
                if family_stats["right"]["guard_rejected"]
                else family_stats["left"]["reject_reason"]
            )
            selected_diag = best or {}
            lateral_stats = family_stats["right"] if family_stats["right"]["count"] > 0 else family_stats["left"]
            return {
                "mpc_failure_reason": failure_reason,
                "mpc_no_rss_feasible_count": no_rss_feasible_count,
                "mpc_no_road_safe_count": no_road_safe_count,
                "mpc_no_terminal_recoverable_count": no_terminal_recoverable_count,
                "mpc_guard_rejected_count": guard_rejected_count,
                "mpc_terminal_candidate_count": terminal_feasible_count,
                "mpc_num_corridors": int(corridor_info.get("mpc_num_corridors", 0)),
                "mpc_num_rss_feasible": max(0, total_candidates - no_rss_feasible_count),
                "mpc_num_road_safe": max(0, total_candidates - no_rss_feasible_count - no_road_safe_count),
                "mpc_num_terminal_recoverable": terminal_feasible_count,
                "mpc_num_guard_rejected": guard_rejected_count,
                "recovery_horizon_steps_used": recovery_horizon,
                "candidate_family": best.get("recovery_candidate_family", "") if best else "",
                "selected_candidate_family": best.get("recovery_candidate_family", "") if best else "",
                "best_candidate_family": best.get("recovery_candidate_family", "") if best else "",
                "num_candidates_brake": family_stats["brake"]["count"],
                "num_candidates_creep": family_stats["creep"]["count"],
                "num_candidates_left": family_stats["left"]["count"],
                "num_candidates_right": family_stats["right"]["count"],
                "best_brake_cost": family_stats["brake"]["best_cost"] if family_stats["brake"]["best_cost"] < math.inf else math.nan,
                "best_creep_cost": family_stats["creep"]["best_cost"] if family_stats["creep"]["best_cost"] < math.inf else math.nan,
                "best_left_cost": family_stats["left"]["best_cost"] if family_stats["left"]["best_cost"] < math.inf else math.nan,
                "best_right_cost": family_stats["right"]["best_cost"] if family_stats["right"]["best_cost"] < math.inf else math.nan,
                "best_left_terminal_recoverable": family_stats["left"]["terminal_recoverable"],
                "best_right_terminal_recoverable": family_stats["right"]["terminal_recoverable"],
                "best_left_guard_rejected": family_stats["left"]["guard_rejected"],
                "best_right_guard_rejected": family_stats["right"]["guard_rejected"],
                "left_reject_reason": family_stats["left"]["reject_reason"],
                "right_reject_reason": family_stats["right"]["reject_reason"],
                "best_left_lateral_rss_margin": family_stats["left"]["best_lateral_rss_margin"] if math.isfinite(family_stats["left"]["best_lateral_rss_margin"]) else math.nan,
                "best_right_lateral_rss_margin": family_stats["right"]["best_lateral_rss_margin"] if math.isfinite(family_stats["right"]["best_lateral_rss_margin"]) else math.nan,
                "right_candidate_generated": family_stats["right"]["count"] > 0,
                "left_candidate_generated": family_stats["left"]["count"] > 0,
                "right_road_safe": family_stats["right"]["road_safe"],
                "left_road_safe": family_stats["left"]["road_safe"],
                "right_lateral_rss_safe": family_stats["right"]["lateral_rss_safe"],
                "left_lateral_rss_safe": family_stats["left"]["lateral_rss_safe"],
                "right_escape_available": family_stats["right"]["escape_available"],
                "left_escape_available": family_stats["left"]["escape_available"],
                "right_escape_reject_reason": family_stats["right"]["reject_reason"],
                "left_escape_reject_reason": family_stats["left"]["reject_reason"],
                "conservative_longitudinal_margin_safe": bool(selected_diag.get("conservative_longitudinal_margin_safe", lateral_stats["conservative_margin_safe"])),
                "critical_longitudinal_margin_safe": bool(selected_diag.get("critical_longitudinal_margin_safe", lateral_stats["critical_margin_safe"])),
                "lateral_escape_candidate": bool(selected_diag.get("lateral_escape_candidate", False)),
                "lateral_escape_certified": bool(selected_diag.get("lateral_escape_certified", False)),
                "lateral_escape_certification_reason": selected_diag.get("lateral_escape_certification_reason", ""),
                "lateral_escape_reject_reason": selected_diag.get("lateral_escape_reject_reason", family_stats["right"]["reject_reason"] or family_stats["left"]["reject_reason"]),
                "lateral_escape_used_relaxed_longitudinal_gate": bool(selected_diag.get("lateral_escape_used_relaxed_longitudinal_gate", lateral_stats["relaxed_longitudinal_gate_used"])),
                "lateral_escape_rejected_by_conservative_gate": bool(selected_diag.get("lateral_escape_rejected_by_conservative_gate", lateral_stats["rejected_by_conservative_gate"])),
                "lateral_escape_rejected_by_critical_margin": bool(selected_diag.get("lateral_escape_rejected_by_critical_margin", lateral_stats["rejected_by_critical_margin"])),
                "lateral_escape_lateral_rss_safe": bool(selected_diag.get("lateral_escape_lateral_rss_safe", lateral_stats["lateral_rss_safe"])),
                "lateral_escape_lateral_margin_improved": bool(selected_diag.get("lateral_escape_lateral_margin_improved", lateral_stats["lateral_margin_improved"])),
                "lateral_escape_path_overlap_reduced": bool(selected_diag.get("lateral_escape_path_overlap_reduced", lateral_stats["path_overlap_reduced"])),
                "lateral_escape_terminal_recoverable": bool(selected_diag.get("lateral_escape_terminal_recoverable", lateral_stats["terminal_recoverable"])),
                "lateral_escape_terminal_reason": selected_diag.get("lateral_escape_terminal_reason", lateral_stats["terminal_reason"]),
                "lateral_escape_low_speed_creep": bool(selected_diag.get("lateral_escape_low_speed_creep", False)),
                "lateral_escape_no_immediate_collision_risk": bool(selected_diag.get("lateral_escape_no_immediate_collision_risk", False)),
                "lateral_escape_steer_toward_escape": bool(selected_diag.get("lateral_escape_steer_toward_escape", False)),
                "guard_rejected_lateral_escape": any_lateral_guard_rejected,
                "lateral_guard_reject_reason": lateral_guard_reject_reason if any_lateral_guard_rejected else "",
                "right_steer_value_used": -self.mpc_config.nudge_steer,
                "left_steer_value_used": self.mpc_config.nudge_steer,
                "creep_acc_value_used": self.mpc_config.creep_acc,
                "certified_lateral_creep_available": any(
                    family_stats[cat]["terminal_recoverable"] and family_stats[cat]["escape_available"]
                    for cat in ("left", "right")
                ),
                "certified_lateral_creep_used": False,
                "certified_lateral_creep_side": "",
                "certified_lateral_creep_reason": "",
                "brake_selected_despite_certified_creep": False,
                "lateral_escape_throttle_suppressed": False,
                "lateral_escape_guard_pass_through": bool(selected_diag.get("lateral_escape_guard_pass_through", False)),
                "lateral_escape_guard_reject_reason": selected_diag.get("lateral_escape_guard_reject_reason", ""),
                "creep_suppression_reason": "",
                "selected_acc_before_guard": math.nan,
                "selected_acc_after_guard": math.nan,
                "selected_steer_before_guard": math.nan,
                "selected_steer_after_guard": math.nan,
                "selected_throttle_before_guard": math.nan,
                "selected_throttle_after_guard": math.nan,
                "selected_brake_before_guard": math.nan,
                "selected_brake_after_guard": math.nan,
                "cbf_guard_delta": 0.0,
                "invalid_lateral_escape_no_creep": False,
                "only_steering_no_creep": False,
                "brake_selected_reason": "",
                "lateral_creep_failure_reason": lateral_failure_reason,
                "mpc_time_ms": mpc_time_ms,
                "candidate_generation_time_ms": gen_time_ms,
                "candidate_evaluation_time_ms": eval_time_ms,
                "num_total_candidates": total_candidates,
                "num_random_candidates": num_random,
                "num_structured_candidates": num_structured,
                **corridor_info,
            }

        if best is not None:
            u_mpc = self._clip_action(best["u_mpc_first"])
            u_safe = self._clip_action(best["u_guarded_first"])
            guard_info = best["guard_info"]
            cbf_guard_delta = best["cbf_guard_delta"]
            cbf_guard_used = best["cbf_guard_used"]
            guard_override_used = best.get("cbf_guard_override_used", False)
            best_category = self._family_category(best.get("recovery_candidate_family", ""))
            best_is_minimum_risk = best.get("corridor_type") == "minimum_risk_stop" or best_category == "brake" and best.get("corridor_type") == "minimum_risk_stop"

            if best_is_minimum_risk:
                mode = "minimum_risk_stop"
                reason = "minimum_risk_condition"
            elif guard_override_used:
                mode = "rss_mpc_recovery"
                reason = "mpc_recovery_large_margin_guard_override"
                self._remember_recovery_action(u_safe)
            else:
                mode = "rss_mpc_cbf_guard" if cbf_guard_used else "rss_mpc_recovery"
                reason = "cbf_guard_projected_mpc_recovery" if cbf_guard_used else "mpc_recovery_sequence"
                if best_category in ("left", "right"):
                    reason = "lateral_{}_{}".format(best_category, reason)
                self._remember_recovery_action(u_safe)

            corridor_memory_info = self._commit_recovery_corridor(best.get("corridor", {}))

            if best_category == "brake" and deadlock_info["deadlock_risk"]:
                lateral_terminal = family_stats["left"]["terminal_recoverable"] or family_stats["right"]["terminal_recoverable"]
                if lateral_terminal:
                    self._consecutive_brake_counter += 1
                    print("[RSS-MPC-DEADLOCK] lateral escape feasible but brake selected "
                          "(consecutive={}, left_tr={}, right_tr={}, left_reject='{}', right_reject='{}')".format(
                              self._consecutive_brake_counter,
                              family_stats["left"]["terminal_recoverable"],
                              family_stats["right"]["terminal_recoverable"],
                              family_stats["left"]["reject_reason"],
                              family_stats["right"]["reject_reason"]))
                else:
                    self._consecutive_brake_counter = 0
            else:
                self._consecutive_brake_counter = 0

            brake_selected_reason = ""
            certified_lateral_creep_available = any(
                family_stats[cat]["terminal_recoverable"] and family_stats[cat]["escape_available"]
                for cat in ("left", "right")
            )
            certified_lateral_creep_used = best_category in ("left", "right") and bool(best.get("certified_lateral_escape_used", False))
            certified_lateral_creep_side = best.get("certified_lateral_escape_side", "") if best_category in ("left", "right") else ""
            certified_lateral_creep_reason = best.get("certified_lateral_escape_reason", "") if best_category in ("left", "right") else ""

            mpc_before = best.get("mpc_action_before_guard", [math.nan, math.nan])
            guard_after = best.get("action_after_guard", [math.nan, math.nan])
            selected_acc_before = float(mpc_before[0]) if isinstance(mpc_before, list) else math.nan
            selected_acc_after = float(guard_after[0]) if isinstance(guard_after, list) else math.nan
            selected_steer_before = float(mpc_before[1]) if isinstance(mpc_before, list) else math.nan
            selected_steer_after = float(guard_after[1]) if isinstance(guard_after, list) else math.nan
            selected_throttle_before = float(best.get("selected_throttle_before_guard", math.nan))
            selected_throttle_after = float(best.get("selected_throttle_after_guard", math.nan))
            selected_brake_before = float(best.get("selected_brake_before_guard", math.nan))
            selected_brake_after = float(best.get("selected_brake_after_guard", math.nan))
            only_steering_no_creep = bool(best.get("only_steering_no_creep", False))
            lateral_escape_throttle_suppressed = (
                best_category in ("left", "right")
                and math.isfinite(selected_acc_before) and selected_acc_before > 0
                and math.isfinite(selected_acc_after) and selected_acc_after <= 0
            )
            creep_suppression_reason = ""
            if lateral_escape_throttle_suppressed:
                creep_suppression_reason = best.get("creep_suppression_reason_guard", "guard_overrode_creep_acc_to_brake")
            if only_steering_no_creep and not lateral_escape_throttle_suppressed:
                creep_suppression_reason = "only_steering_no_creep_detected"

            if best_category == "brake":
                if not family_stats["left"]["terminal_recoverable"] and not family_stats["right"]["terminal_recoverable"]:
                    brake_selected_reason = "no_lateral_terminal_recoverable"
                elif family_stats["left"]["guard_rejected"] and family_stats["right"]["guard_rejected"]:
                    brake_selected_reason = "lateral_guard_rejected"
                elif family_stats["left"]["terminal_recoverable"] or family_stats["right"]["terminal_recoverable"]:
                    brake_selected_reason = "brake_selected_despite_lateral_escape_available"
                    if certified_lateral_creep_available:
                        print("[RSS-MPC-DEADLOCK] certified lateral creep feasible but brake selected "
                              "(left_tr={}, right_tr={}, left_avail={}, right_avail={})".format(
                                  family_stats["left"]["terminal_recoverable"],
                                  family_stats["right"]["terminal_recoverable"],
                                  family_stats["left"]["escape_available"],
                                  family_stats["right"]["escape_available"]))
                else:
                    brake_selected_reason = "brake_lower_cost"
            elif best_category in ("left", "right") and only_steering_no_creep:
                brake_selected_reason = "lateral_escape_only_steering_no_creep"

            return u_safe, self._make_mpc_info(
                state=state,
                obj=obj,
                object_kind=object_kind,
                mode=mode,
                reason=reason,
                u_original=u_original,
                u_safe=u_safe,
                d_front=d_front,
                d_rss=d_rss,
                rss_margin_current=current_margin,
                dynamic_vehicle_detected=dynamic_detected,
                obstacle_detected=static_detected,
                mpc_success=not best_is_minimum_risk,
                mpc_num_candidates=total_candidates,
                mpc_num_feasible=feasible_count,
                mpc_best_cost=best["cost"],
                rss_margin_min_pred=best["rss_margin_min_pred"],
                rss_margin_final_pred=best["rss_margin_final_pred"],
                rss_lateral_clearance_min_pred=best["rss_lateral_clearance_min_pred"],
                rss_lateral_clearance_final_pred=best["rss_lateral_clearance_final_pred"],
                predicted_progress=best["predicted_progress"],
                fallback_used=False,
                cbf_info=guard_info,
                cbf_reference_info=cbf_info,
                cbf_guard_used=cbf_guard_used,
                cbf_guard_delta=cbf_guard_delta,
                extra_info=self._merge_recovery_info(
                    deadlock_info,
                    {
                        "mpc_called": True,
                        "mpc_call_reason": "deadlock_risk",
                        **_diagnostics_block(),
                        "terminal_recoverable": best["terminal_recoverable"],
                        "terminal_recovery_reason": best["terminal_recovery_reason"],
                        "recovery_progress": best["recovery_progress"],
                        "recovery_margin_improvement": best["recovery_margin_improvement"],
                        "blocking_object_final": best["blocking_object_final"],
                        "minimum_risk_stop_used": best_is_minimum_risk,
                        "mpc_terminal_feasible": terminal_feasible_count,
                        "mpc_guard_rejected": guard_rejected_count,
                        "cbf_guard_override_used": guard_override_used,
                        "cbf_guard_override_reason": best.get("cbf_guard_override_reason", ""),
                        "certified_recovery_override_used": guard_override_used,
                        "certified_recovery_override_reason": best.get("cbf_guard_override_reason", ""),
                        "guard_reject_reason": best.get("guard_reject_reason", "") or (
                            family_stats["right"]["reject_reason"]
                            if best_category == "brake" and family_stats["right"]["guard_rejected"]
                            else family_stats["left"]["reject_reason"]
                            if best_category == "brake" and family_stats["left"]["guard_rejected"]
                            else ""
                        ),
                        "first_step_recovery_reason": best.get("first_step_recovery_reason", ""),
                        "recovery_candidate_family": best.get("recovery_candidate_family", ""),
                        "selected_recovery_corridor": best.get("corridor_type", ""),
                        "active_recovery_corridor": self.active_recovery_corridor,
                        "previous_recovery_corridor": self.previous_recovery_corridor,
                        "corridor_target_lateral_offset": best.get("corridor_target_lateral_offset", math.nan),
                        "corridor_target_speed": best.get("corridor_target_speed", math.nan),
                        "corridor_cost": best.get("corridor_cost", math.nan),
                        "corridor_terminal_recoverable": best.get("terminal_recoverable", False),
                        "first_step_recovery_feasible": best.get("first_step_recovery_feasible", False),
                        "brake_selected_reason": brake_selected_reason,
                        "selected_steer_before_guard": selected_steer_before,
                        "selected_steer_after_guard": selected_steer_after,
                        "selected_throttle_before_guard": selected_throttle_before,
                        "selected_throttle_after_guard": selected_throttle_after,
                        "selected_brake_before_guard": selected_brake_before,
                        "selected_brake_after_guard": selected_brake_after,
                        "cbf_guard_delta": cbf_guard_delta,
                        "certified_lateral_escape_used": best.get("certified_lateral_escape_used", False),
                        "certified_lateral_escape_side": best.get("certified_lateral_escape_side", ""),
                        "certified_lateral_escape_reason": best.get("certified_lateral_escape_reason", ""),
                        "guard_rejected_lateral_escape": best.get("guard_rejected_lateral_escape", False) or (
                            best_category == "brake"
                            and (family_stats["left"]["guard_rejected"] or family_stats["right"]["guard_rejected"])
                        ),
                        "mpc_action_before_guard": best.get("mpc_action_before_guard", [math.nan, math.nan]),
                        "action_after_guard": best.get("action_after_guard", [math.nan, math.nan]),
                        "initial_lateral_rss_margin": best.get("initial_lateral_rss_margin", math.nan),
                        "final_lateral_rss_margin": best.get("final_lateral_rss_margin", math.nan),
                        "predicted_lateral_distance": best.get("predicted_lateral_distance", math.nan),
                        "predicted_lateral_rss_margin": best.get("predicted_lateral_rss_margin", math.nan),
                        "predicted_path_overlap": best.get("predicted_path_overlap", False),
                        "predicted_path_overlap_reducing": best.get("predicted_path_overlap_reducing", False),
                        "terminal_lateral_separation_safe": best.get("terminal_lateral_separation_safe", False),
                        "terminal_lateral_deconflicted": best.get("terminal_lateral_deconflicted", False),
                        "road_boundary_safe": best.get("road_boundary_safe", False),
                        "immediate_longitudinal_margin_safe": best.get("immediate_longitudinal_margin_safe", False),
                        "lateral_rss_improvement": best.get("lateral_rss_improvement", 0.0),
                        "path_overlap_reduced": best.get("path_overlap_reduced", False),
                        "terminal_deconflicted": best.get("terminal_deconflicted", False),
                        "first_step_lateral_margin_improves": best.get("first_step_lateral_margin_improves", False),
                        "first_step_path_overlap_reduces": best.get("first_step_path_overlap_reduces", False),
                        "first_step_lateral_distance_increases": best.get("first_step_lateral_distance_increases", False),
                        "certified_lateral_creep_available": certified_lateral_creep_available,
                        "certified_lateral_creep_used": certified_lateral_creep_used,
                        "certified_lateral_creep_side": certified_lateral_creep_side,
                        "certified_lateral_creep_reason": certified_lateral_creep_reason,
                        "brake_selected_despite_certified_creep": bool(best_category == "brake" and certified_lateral_creep_available),
                        "lateral_escape_throttle_suppressed": lateral_escape_throttle_suppressed,
                        "creep_suppression_reason": creep_suppression_reason,
                        "selected_acc_before_guard": selected_acc_before,
                        "selected_acc_after_guard": selected_acc_after,
                        "only_steering_no_creep": only_steering_no_creep,
                        "invalid_lateral_escape_no_creep": bool(best.get("invalid_lateral_escape_no_creep", False)),
                        "creep_acc_value_used": self.mpc_config.creep_acc,
                        "right_steer_value_used": -self.mpc_config.nudge_steer,
                        "left_steer_value_used": self.mpc_config.nudge_steer,
                        "action_mapping_note": "internal_acc_steer_positive_acc_is_throttle" if best_category in ("left", "right") else best_category,
                        **corridor_memory_info,
                    },
                ),
            )

        held_recovery = self._take_held_recovery_action(current_margin)
        if held_recovery is not None and cbf_info.get("mode") == "fallback_no_safe_candidate":
            return held_recovery, self._make_mpc_info(
                state=state,
                obj=obj,
                object_kind=object_kind,
                mode="rss_mpc_recovery",
                reason="mpc_recovery_hold_large_margin",
                u_original=u_original,
                u_safe=held_recovery,
                d_front=d_front,
                d_rss=d_rss,
                rss_margin_current=current_margin,
                dynamic_vehicle_detected=dynamic_detected,
                obstacle_detected=static_detected,
                mpc_success=True,
                mpc_num_candidates=total_candidates,
                mpc_num_feasible=0,
                mpc_best_cost=math.nan,
                rss_margin_min_pred=math.nan,
                rss_margin_final_pred=math.nan,
                rss_lateral_clearance_min_pred=math.nan,
                rss_lateral_clearance_final_pred=math.nan,
                predicted_progress=math.nan,
                fallback_used=False,
                cbf_info=cbf_info,
                cbf_reference_info=cbf_info,
                cbf_guard_used=False,
                cbf_guard_delta=0.0,
                extra_info=self._merge_recovery_info(
                    deadlock_info,
                    {
                        "mpc_called": True,
                        "mpc_call_reason": "deadlock_risk",
                        "mpc_terminal_feasible": terminal_feasible_count,
                        "mpc_guard_rejected": guard_rejected_count,
                        "cbf_guard_override_used": True,
                        "cbf_guard_override_reason": "held_certified_recovery_action_with_large_rss_margin",
                        "certified_recovery_override_used": True,
                        "certified_recovery_override_reason": "held_certified_recovery_action_with_large_rss_margin",
                        "recovery_hold_used": True,
                        **_diagnostics_block(),
                    },
                ),
            )

        if (guard_rejected_candidates
                and math.isfinite(current_margin)
                and current_margin >= self.mpc_config.guard_override_fallback_margin_buffer):
            lateral_guard_rejected = [
                (ev, ge) for ev, ge in guard_rejected_candidates
                if self._family_category(ev.get("recovery_candidate_family", "")) in ("left", "right")
            ]
            override_pool = lateral_guard_rejected if lateral_guard_rejected else guard_rejected_candidates
            override_pool.sort(key=lambda item: item[0].get("cost", 1e9))
            for evaluation, guard_eval in override_pool:
                guard_info = guard_eval.get("guard_info", {})
                guard_mode = guard_info.get("mode", "")
                override_used, override_reason = self._guard_override_for_certified_recovery(
                    state=state,
                    evaluation=evaluation,
                    u_mpc=self._clip_action(evaluation["sequence"][0]),
                    u_guarded=guard_eval.get("u_guarded_first", guard_eval.get("u_mpc_first")),
                    guard_mode=guard_mode,
                )
                if override_used:
                    u_safe = self._clip_action(evaluation["sequence"][0])
                    self._remember_recovery_action(u_safe)
                    corridor_memory_info = self._commit_recovery_corridor(evaluation.get("corridor", {}))
                    ev_category = self._family_category(evaluation.get("recovery_candidate_family", ""))
                    return u_safe, self._make_mpc_info(
                        state=state,
                        obj=obj,
                        object_kind=object_kind,
                        mode="rss_mpc_recovery",
                        reason="mpc_recovery_guard_rejected_override" + ("_lateral" if ev_category in ("left", "right") else ""),
                        u_original=u_original,
                        u_safe=u_safe,
                        d_front=d_front,
                        d_rss=d_rss,
                        rss_margin_current=current_margin,
                        dynamic_vehicle_detected=dynamic_detected,
                        obstacle_detected=static_detected,
                        mpc_success=True,
                        mpc_num_candidates=total_candidates,
                        mpc_num_feasible=1,
                        mpc_best_cost=evaluation.get("cost", math.nan),
                        rss_margin_min_pred=evaluation["rss_margin_min_pred"],
                        rss_margin_final_pred=evaluation["rss_margin_final_pred"],
                        rss_lateral_clearance_min_pred=evaluation["rss_lateral_clearance_min_pred"],
                        rss_lateral_clearance_final_pred=evaluation["rss_lateral_clearance_final_pred"],
                        predicted_progress=evaluation["predicted_progress"],
                        fallback_used=False,
                        cbf_info=guard_info,
                        cbf_reference_info=cbf_info,
                        cbf_guard_used=False,
                        cbf_guard_delta=0.0,
                        extra_info=self._merge_recovery_info(
                            deadlock_info,
                            {
                                "mpc_called": True,
                                "mpc_call_reason": "deadlock_risk",
                                **_diagnostics_block(),
                                "terminal_recoverable": evaluation["terminal_recoverable"],
                                "terminal_recovery_reason": evaluation["terminal_recovery_reason"],
                                "recovery_progress": evaluation["recovery_progress"],
                                "recovery_margin_improvement": evaluation["recovery_margin_improvement"],
                                "blocking_object_final": evaluation["blocking_object_final"],
                                "mpc_terminal_feasible": terminal_feasible_count,
                                "mpc_guard_rejected": guard_rejected_count,
                                "cbf_guard_override_used": True,
                                "cbf_guard_override_reason": override_reason,
                                "certified_recovery_override_used": True,
                                "certified_recovery_override_reason": override_reason,
                                "guard_reject_reason": guard_eval.get("guard_reject_reason", ""),
                                "first_step_recovery_reason": evaluation.get("first_step_recovery_reason", ""),
                                "recovery_candidate_family": evaluation.get("recovery_candidate_family", ""),
                                "selected_recovery_corridor": evaluation.get("corridor_type", ""),
                                "active_recovery_corridor": self.active_recovery_corridor,
                                "previous_recovery_corridor": self.previous_recovery_corridor,
                                "corridor_target_lateral_offset": evaluation.get("corridor_target_lateral_offset", math.nan),
                                "corridor_target_speed": evaluation.get("corridor_target_speed", math.nan),
                                "corridor_cost": evaluation.get("corridor_cost", math.nan),
                                "corridor_terminal_recoverable": evaluation.get("terminal_recoverable", False),
                                "first_step_recovery_feasible": evaluation.get("first_step_recovery_feasible", False),
                                "predicted_lateral_distance": evaluation.get("predicted_lateral_distance", math.nan),
                                "predicted_lateral_rss_margin": evaluation.get("predicted_lateral_rss_margin", math.nan),
                                "predicted_path_overlap": evaluation.get("predicted_path_overlap", False),
                                "predicted_path_overlap_reducing": evaluation.get("predicted_path_overlap_reducing", False),
                                "terminal_lateral_separation_safe": evaluation.get("terminal_lateral_separation_safe", False),
                                "terminal_lateral_deconflicted": evaluation.get("terminal_lateral_deconflicted", False),
                                "road_boundary_safe": evaluation.get("road_boundary_safe", False),
                                "immediate_longitudinal_margin_safe": evaluation.get("immediate_longitudinal_margin_safe", False),
                                "conservative_longitudinal_margin_safe": evaluation.get("conservative_longitudinal_margin_safe", False),
                                "critical_longitudinal_margin_safe": evaluation.get("critical_longitudinal_margin_safe", False),
                                "lateral_escape_candidate": evaluation.get("lateral_escape_candidate", False),
                                "lateral_escape_certified": evaluation.get("lateral_escape_certified", False),
                                "lateral_escape_certification_reason": evaluation.get("lateral_escape_certification_reason", ""),
                                "lateral_escape_reject_reason": evaluation.get("lateral_escape_reject_reason", ""),
                                "lateral_escape_used_relaxed_longitudinal_gate": evaluation.get("lateral_escape_used_relaxed_longitudinal_gate", False),
                                "lateral_escape_rejected_by_conservative_gate": evaluation.get("lateral_escape_rejected_by_conservative_gate", False),
                                "lateral_escape_rejected_by_critical_margin": evaluation.get("lateral_escape_rejected_by_critical_margin", False),
                                "lateral_escape_lateral_rss_safe": evaluation.get("lateral_escape_lateral_rss_safe", False),
                                "lateral_escape_lateral_margin_improved": evaluation.get("lateral_escape_lateral_margin_improved", False),
                                "lateral_escape_path_overlap_reduced": evaluation.get("lateral_escape_path_overlap_reduced", False),
                                "lateral_escape_terminal_recoverable": evaluation.get("lateral_escape_terminal_recoverable", False),
                                "lateral_escape_terminal_reason": evaluation.get("lateral_escape_terminal_reason", ""),
                                "lateral_escape_low_speed_creep": evaluation.get("lateral_escape_low_speed_creep", False),
                                "lateral_escape_no_immediate_collision_risk": evaluation.get("lateral_escape_no_immediate_collision_risk", False),
                                "lateral_escape_steer_toward_escape": evaluation.get("lateral_escape_steer_toward_escape", False),
                                "initial_lateral_rss_margin": evaluation.get("initial_lateral_rss_margin", math.nan),
                                "final_lateral_rss_margin": evaluation.get("final_lateral_rss_margin", math.nan),
                                "lateral_rss_improvement": evaluation.get("lateral_rss_improvement", 0.0),
                                "path_overlap_reduced": evaluation.get("path_overlap_reduced", False),
                                "terminal_deconflicted": evaluation.get("terminal_deconflicted", False),
                                "first_step_lateral_margin_improves": evaluation.get("first_step_lateral_margin_improves", False),
                                "first_step_path_overlap_reduces": evaluation.get("first_step_path_overlap_reduces", False),
                                "first_step_lateral_distance_increases": evaluation.get("first_step_lateral_distance_increases", False),
                                **corridor_memory_info,
                            },
                        ),
                    )

        if cbf_info.get("mode") != "fallback_no_safe_candidate":
            u_safe = cbf_safe
            mode = "rss_mpc_fallback_to_cbf"
            reason = "mpc_recovery_failed_fallback_to_cbf"
            minimum_risk_stop_used = False
            if feasible_count == 0 and terminal_feasible_count > 0:
                failure_reason = failure_reason or "terminal_candidates_guard_rejected"
        else:
            u_safe = self._minimum_risk_stop(u_original)
            mode = "minimum_risk_stop"
            reason = "mpc_recovery_failed_minimum_risk_stop"
            minimum_risk_stop_used = True
            if feasible_count == 0 and terminal_feasible_count > 0:
                failure_reason = "fallback_to_minimum_risk_stop"

        if deadlock_info["deadlock_risk"] and minimum_risk_stop_used:
            lateral_terminal = family_stats["left"]["terminal_recoverable"] or family_stats["right"]["terminal_recoverable"]
            if lateral_terminal:
                self._consecutive_brake_counter += 1
                print("[RSS-MPC-DEADLOCK] lateral escape feasible but brake selected "
                      "(consecutive={}, left_tr={}, right_tr={}, left_reject='{}', right_reject='{}')".format(
                          self._consecutive_brake_counter,
                          family_stats["left"]["terminal_recoverable"],
                          family_stats["right"]["terminal_recoverable"],
                          family_stats["left"]["reject_reason"],
                          family_stats["right"]["reject_reason"]))
            else:
                self._consecutive_brake_counter = 0

        info = self._make_mpc_info(
            state=state,
            obj=obj,
            object_kind=object_kind,
            mode=mode,
            reason=reason,
            u_original=u_original,
            u_safe=u_safe,
            d_front=d_front,
            d_rss=d_rss,
            rss_margin_current=current_margin,
            dynamic_vehicle_detected=dynamic_detected,
            obstacle_detected=static_detected,
            mpc_success=False,
            mpc_num_candidates=total_candidates,
            mpc_num_feasible=0,
            mpc_best_cost=math.nan,
            rss_margin_min_pred=math.nan,
            rss_margin_final_pred=math.nan,
            rss_lateral_clearance_min_pred=math.nan,
            rss_lateral_clearance_final_pred=math.nan,
            predicted_progress=math.nan,
            fallback_used=True,
            cbf_info=cbf_info,
            cbf_reference_info=cbf_info,
            cbf_guard_used=False,
            cbf_guard_delta=0.0,
            extra_info=self._merge_recovery_info(
                deadlock_info,
                {
                    "mpc_called": True,
                    "mpc_call_reason": "deadlock_risk",
                    "minimum_risk_stop_used": minimum_risk_stop_used,
                    "mpc_terminal_feasible": terminal_feasible_count,
                    "mpc_guard_rejected": guard_rejected_count,
                    "lateral_creep_failure_reason": lateral_failure_reason,
                    **_diagnostics_block(),
                },
            ),
        )
        info["cbf_info"] = cbf_info
        return u_safe, info

    def _cbf_indicates_nominal_safe(self, cbf_info: Dict[str, Any], u_original: Action, u_cbf: Action) -> bool:
        cbf_delta = float(cbf_info.get("action_delta", self._action_norm(u_cbf, u_original)))
        return cbf_info.get("mode") == "normal" and cbf_delta <= self.mpc_config.action_change_tolerance

    def _minimum_risk_stop(self, u_nom: Action) -> Action:
        return self._clip_action([self.mpc_config.strong_brake, u_nom[1]])

    def _remember_recovery_action(self, action: Sequence[float]) -> None:
        self._held_recovery_action = self._clip_action(action)
        self._held_recovery_ttl = max(0, int(self.mpc_config.recovery_hold_steps))

    def _take_held_recovery_action(self, current_margin: float) -> Optional[Action]:
        if self._held_recovery_action is None or self._held_recovery_ttl <= 0:
            return None
        if not math.isfinite(current_margin) or current_margin < self.mpc_config.recovery_hold_margin_buffer:
            self._held_recovery_action = None
            self._held_recovery_ttl = 0
            return None
        self._held_recovery_ttl -= 1
        return self._clip_action(self._held_recovery_action)

    def _update_deadlock_detector(
        self,
        state: State,
        u_original: Action,
        u_cbf: Action,
        cbf_info: Dict[str, Any],
        front: Optional[Tuple[str, Dict[str, Any], float, float, float, bool, bool]],
    ) -> Dict[str, Any]:
        ego = self._ego(state)
        speed = self._ego_speed(state)
        cbf_delta = float(cbf_info.get("action_delta", self._action_norm(u_cbf, u_original)))
        cbf_active = (cbf_info.get("mode") != "normal") or cbf_delta > self.mpc_config.deadlock_cbf_delta_threshold
        cbf_fallback = cbf_info.get("mode") == "fallback_no_safe_candidate"
        front_exists = front is not None
        route_completion = self._last_route_completion

        sample = {
            "x": float(ego.get("x", 0.0)),
            "y": float(ego.get("y", 0.0)),
            "heading": float(ego.get("heading", 0.0)),
            "speed": speed,
            "cbf_delta": cbf_delta,
            "cbf_active": bool(cbf_active),
            "cbf_fallback": bool(cbf_fallback),
            "front_exists": bool(front_exists),
            "route_completion": route_completion,
        }
        self.deadlock_history.append(sample)
        info = self._deadlock_diagnostics()
        if info["deadlock_candidate"]:
            self.deadlock_counter += 1
        else:
            self.deadlock_counter = max(0, self.deadlock_counter - 1)
        info["deadlock_counter"] = self.deadlock_counter
        info["deadlock_risk"] = bool(info["deadlock_candidate"] and self.deadlock_counter >= self.mpc_config.deadlock_counter_threshold)
        return info

    def _deadlock_diagnostics(self) -> Dict[str, Any]:
        samples = list(self.deadlock_history)
        if len(samples) < self.mpc_config.deadlock_min_window_steps:
            return self._make_deadlock_info(
                risk=False,
                candidate=False,
                score=0.0,
                progress=0.0,
                avg_speed=float(np.mean([s["speed"] for s in samples])) if samples else math.nan,
                active_ratio=0.0,
                reason="insufficient_deadlock_window",
            )

        first = samples[0]
        last = samples[-1]
        dx = last["x"] - first["x"]
        dy = last["y"] - first["y"]
        heading = first.get("heading", 0.0)
        window_progress = dx * math.cos(heading) + dy * math.sin(heading)
        avg_speed = float(np.mean([sample["speed"] for sample in samples]))
        cbf_active_ratio = float(np.mean([1.0 if sample["cbf_active"] else 0.0 for sample in samples]))
        cbf_fallback_ratio = float(np.mean([1.0 if sample.get("cbf_fallback") else 0.0 for sample in samples]))
        front_active_ratio = float(np.mean([1.0 if sample["front_exists"] else 0.0 for sample in samples]))
        avg_cbf_delta = float(np.mean([sample["cbf_delta"] for sample in samples]))

        route_values = [
            sample["route_completion"]
            for sample in samples
            if isinstance(sample.get("route_completion"), (int, float)) and math.isfinite(sample["route_completion"])
        ]
        route_completion = route_values[-1] if route_values else math.nan
        near_goal = math.isfinite(route_completion) and route_completion >= self.mpc_config.near_goal_route_completion

        low_speed_score = max(0.0, min(1.0, 1.0 - avg_speed / max(self.mpc_config.deadlock_speed_threshold, 1e-6)))
        low_progress_score = max(
            0.0,
            min(1.0, 1.0 - max(0.0, window_progress) / max(self.mpc_config.deadlock_progress_threshold, 1e-6)),
        )
        cbf_ratio_score = max(0.0, min(1.0, cbf_active_ratio / self.mpc_config.deadlock_cbf_active_ratio_threshold))
        cbf_delta_score = max(0.0, min(1.0, avg_cbf_delta / self.mpc_config.deadlock_cbf_delta_threshold))
        front_score = max(0.0, min(1.0, front_active_ratio / self.mpc_config.deadlock_front_active_ratio_threshold))
        not_done_score = 0.0 if near_goal else 1.0
        score = float(
            np.mean([low_speed_score, low_progress_score, cbf_ratio_score, cbf_delta_score, front_score, not_done_score])
        )

        hard_candidate = (
            avg_speed <= self.mpc_config.deadlock_speed_threshold
            and window_progress <= self.mpc_config.deadlock_progress_threshold
            and cbf_active_ratio >= self.mpc_config.deadlock_cbf_active_ratio_threshold
            and front_active_ratio >= self.mpc_config.deadlock_front_active_ratio_threshold
            and avg_cbf_delta >= self.mpc_config.deadlock_cbf_delta_threshold
            and not near_goal
        )
        fallback_candidate = (
            avg_speed <= self.mpc_config.deadlock_speed_threshold
            and window_progress <= self.mpc_config.deadlock_progress_threshold
            and front_active_ratio >= self.mpc_config.deadlock_front_active_ratio_threshold
            and cbf_fallback_ratio >= self.mpc_config.deadlock_cbf_fallback_ratio_threshold
            and not near_goal
        )
        soft_candidate = hard_candidate and score >= self.mpc_config.deadlock_score_threshold
        candidate = hard_candidate or fallback_candidate or soft_candidate
        if fallback_candidate:
            reason = "cbf_fallback_deadlock_risk"
        elif candidate:
            reason = "deadlock_risk"
        else:
            reason = "deadlock_conditions_not_met"
        return self._make_deadlock_info(
            risk=False,
            candidate=candidate,
            score=score,
            progress=window_progress,
            avg_speed=avg_speed,
            active_ratio=cbf_active_ratio,
            reason=reason,
            front_ratio=front_active_ratio,
            avg_cbf_delta=avg_cbf_delta,
            fallback_ratio=cbf_fallback_ratio,
            route_completion=route_completion,
        )

    def _make_deadlock_info(
        self,
        risk: bool,
        candidate: bool,
        score: float,
        progress: float,
        avg_speed: float,
        active_ratio: float,
        reason: str,
        front_ratio: float = 0.0,
        avg_cbf_delta: float = 0.0,
        fallback_ratio: float = 0.0,
        route_completion: float = math.nan,
    ) -> Dict[str, Any]:
        return {
            "selective_mpc_enabled": True,
            "mpc_called": False,
            "mpc_call_reason": "",
            "deadlock_risk": bool(risk),
            "deadlock_candidate": bool(candidate),
            "deadlock_score": float(score),
            "deadlock_counter": self.deadlock_counter,
            "deadlock_window_progress": float(progress),
            "deadlock_window_avg_speed": float(avg_speed),
            "deadlock_cbf_active_ratio": float(active_ratio),
            "deadlock_front_active_ratio": float(front_ratio),
            "deadlock_avg_cbf_delta": float(avg_cbf_delta),
            "deadlock_cbf_fallback_ratio": float(fallback_ratio),
            "deadlock_route_completion": float(route_completion),
            "deadlock_reason": reason,
            "terminal_recoverable": False,
            "terminal_recovery_reason": "",
            "recovery_progress": math.nan,
            "recovery_margin_improvement": math.nan,
            "blocking_object_final": False,
            "minimum_risk_stop_used": False,
        }

    def _merge_recovery_info(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        merged.update(updates)
        return merged

    def _tick_corridor_memory(self) -> None:
        if self.recovery_corridor_ttl > 0:
            self.recovery_corridor_ttl -= 1
        elif self.active_recovery_corridor:
            self.previous_recovery_corridor = self.active_recovery_corridor
            self.active_recovery_corridor = ""
        if self.recovery_side_switch_cooldown > 0:
            self.recovery_side_switch_cooldown -= 1

    def _commit_recovery_corridor(self, corridor: Dict[str, Any]) -> Dict[str, Any]:
        selected = str(corridor.get("corridor_type", ""))
        previous = self.active_recovery_corridor
        if selected == "minimum_risk_stop":
            if previous:
                self.previous_recovery_corridor = previous
                self.active_recovery_corridor = ""
                self.recovery_corridor_ttl = 0
            return {
                "corridor_switch_used": bool(previous),
                "corridor_switch_reason": "minimum_risk_stop_no_drivable_corridor" if previous else "",
            }
        switch_used = bool(selected and previous and selected != previous)
        switch_reason = ""
        if switch_used:
            switch_reason = "better_recovery_corridor_selected"
            if {selected, previous} == {"left_offset", "right_offset"}:
                self.recovery_side_switch_cooldown = max(
                    self.recovery_side_switch_cooldown,
                    int(self.mpc_config.recovery_side_switch_cooldown),
                )
        if selected:
            self.previous_recovery_corridor = previous
            self.active_recovery_corridor = selected
            self.recovery_corridor_ttl = max(0, int(self.mpc_config.recovery_corridor_ttl))
        return {
            "corridor_switch_used": switch_used,
            "corridor_switch_reason": switch_reason,
        }

    def _family_category(self, family: str) -> str:
        if any(k in family for k in ("minimum_risk", "cbf_brake", "strong_brake", "comfort_brake")):
            return "brake"
        if any(k in family for k in ("left",)):
            return "left"
        if any(k in family for k in ("right",)):
            return "right"
        return "creep"

    def _generate_deadlock_escape_candidates(
        self,
        state: State,
        obj: Dict[str, Any],
        object_kind: str,
        current_margin: float,
        u_nom: Action,
        u_cbf: Action,
    ) -> Dict[str, Any]:
        cfg = self.mpc_config
        horizon = max(1, int(cfg.recovery_horizon_steps))
        max_structured = max(1, int(cfg.max_structured_escape_candidates))
        max_random = max(0, int(cfg.max_random_escape_candidates))
        sequences: List[np.ndarray] = []
        families: List[str] = []
        corridor_meta: List[Dict[str, Any]] = []

        acc_cbf, steer_cbf = self._clip_action(u_cbf)
        lane_width = max(self.config.default_lane_width, self._current_lane_width(state))
        lanes = state.get("lanes", {}) or {}
        left_lane = lanes.get("left", {}) or {}
        right_lane = lanes.get("right", {}) or {}
        left_available = bool(left_lane.get("available", False) and left_lane.get("drivable", True))
        right_available = bool(right_lane.get("available", False) and right_lane.get("drivable", True))
        left_boundary_feasible = self._corridor_boundary_feasible(state, cfg.corridor_lateral_offset)
        right_boundary_feasible = self._corridor_boundary_feasible(state, -cfg.corridor_lateral_offset)
        left_feasible = left_available and left_boundary_feasible
        right_feasible = right_available and right_boundary_feasible
        road_margin = self._current_road_boundary_margin(state)
        road_boundary_active = math.isfinite(road_margin) and road_margin < max(0.35, cfg.corridor_road_boundary_margin * 2.0)

        def _make_corridor_meta(corridor_type: str, target_offset: float, target_speed: float) -> Dict[str, Any]:
            return {
                "corridor_type": corridor_type,
                "target_lateral_offset": float(target_offset),
                "target_speed": float(target_speed),
                "target_progress": float(cfg.corridor_target_progress),
                "target_clearance": float(cfg.corridor_target_clearance),
                "required_rss_margin": float(cfg.corridor_required_rss_margin),
                "road_boundary_margin": float(cfg.corridor_road_boundary_margin),
                "available": True,
                "reason": "deadlock_escape",
                "priority": 0.5,
                "road_boundary_active": bool(road_boundary_active),
            }

        def add(acc_values, steer_values, family: str, corridor_type: str, target_offset: float = 0.0, target_speed: float = 0.0) -> None:
            if len(sequences) >= max_structured:
                return
            sequence = np.asarray(
                [self._clip_action([acc, steer]) for acc, steer in zip(acc_values, steer_values)],
                dtype=np.float64,
            )
            if sequence.shape == (horizon, 2):
                sequences.append(sequence)
                families.append(family)
                corridor_meta.append(_make_corridor_meta(corridor_type, target_offset, target_speed))

        # --- Family 1: cbf_brake ---
        add(np.full(horizon, acc_cbf, dtype=np.float64), np.full(horizon, steer_cbf, dtype=np.float64),
            "cbf_brake", "creep_forward", 0.0, cfg.corridor_target_speed)

        # --- Family 2: minimum_risk_stop ---
        add(np.full(horizon, cfg.strong_brake, dtype=np.float64), np.full(horizon, u_nom[1], dtype=np.float64),
            "minimum_risk_stop", "minimum_risk_stop")
        add(np.full(horizon, cfg.strong_brake, dtype=np.float64), np.zeros(horizon, dtype=np.float64),
            "minimum_risk_stop_zero_steer", "minimum_risk_stop")

        # --- Family 3: creep_forward ---
        add(np.full(horizon, cfg.creep_acc, dtype=np.float64), np.full(horizon, u_nom[1], dtype=np.float64),
            "creep_forward", "creep_forward", 0.0, cfg.corridor_target_speed)
        add(np.full(horizon, cfg.creep_acc, dtype=np.float64), np.zeros(horizon, dtype=np.float64),
            "creep_forward_straight", "creep_forward", 0.0, cfg.corridor_target_speed)

        # --- Family 4: explicit left escape candidates ---
        add(np.full(horizon, cfg.creep_acc, dtype=np.float64), np.full(horizon, cfg.nudge_steer, dtype=np.float64),
            "left_creep_escape", "left_offset", cfg.corridor_lateral_offset, cfg.corridor_target_speed)
        add(np.full(horizon, cfg.creep_acc, dtype=np.float64),
            np.full(horizon, cfg.nudge_steer * 1.5, dtype=np.float64),
            "left_nudge_escape", "left_offset", cfg.corridor_lateral_offset, cfg.corridor_target_speed)

        # --- Family 5: explicit right escape candidates ---
        add(np.full(horizon, cfg.creep_acc, dtype=np.float64), np.full(horizon, -cfg.nudge_steer, dtype=np.float64),
            "right_creep_escape", "right_offset", -cfg.corridor_lateral_offset, cfg.corridor_target_speed)
        add(np.full(horizon, cfg.creep_acc, dtype=np.float64),
            np.full(horizon, -cfg.nudge_steer * 1.5, dtype=np.float64),
            "right_nudge_escape", "right_offset", -cfg.corridor_lateral_offset, cfg.corridor_target_speed)

        # --- Family 6: brake_then_creep_left ---
        if horizon >= 2:
            add(
                np.concatenate([[cfg.comfort_brake], np.full(horizon - 1, cfg.creep_acc, dtype=np.float64)]),
                np.concatenate([[0.0], np.full(horizon - 1, cfg.nudge_steer, dtype=np.float64)]),
                "brake_then_creep_left", "left_offset", cfg.corridor_lateral_offset, cfg.corridor_target_speed,
            )

        # --- Family 7: brake_then_creep_right ---
        if horizon >= 2:
            add(
                np.concatenate([[cfg.comfort_brake], np.full(horizon - 1, cfg.creep_acc, dtype=np.float64)]),
                np.concatenate([[0.0], np.full(horizon - 1, -cfg.nudge_steer, dtype=np.float64)]),
                "brake_then_creep_right", "right_offset", -cfg.corridor_lateral_offset, cfg.corridor_target_speed,
            )

        # --- Family 8: left_then_straight_escape ---
        half = max(1, horizon // 2)
        add(
            np.full(horizon, cfg.creep_acc, dtype=np.float64),
            np.concatenate([np.full(half, cfg.nudge_steer, dtype=np.float64),
                            np.linspace(cfg.nudge_steer, 0.0, horizon - half)]),
            "left_then_straight_escape", "left_offset", cfg.corridor_lateral_offset, cfg.corridor_target_speed,
        )

        # --- Family 9: right_then_straight_escape ---
        add(
            np.full(horizon, cfg.creep_acc, dtype=np.float64),
            np.concatenate([np.full(half, -cfg.nudge_steer, dtype=np.float64),
                            np.linspace(-cfg.nudge_steer, 0.0, horizon - half)]),
            "right_then_straight_escape", "right_offset", -cfg.corridor_lateral_offset, cfg.corridor_target_speed,
        )

        # --- Family 10: left_bypass_low_speed ---
        add(np.full(horizon, cfg.creep_acc * 0.5, dtype=np.float64), np.full(horizon, cfg.bypass_steer, dtype=np.float64),
            "left_bypass_low_speed", "left_offset", cfg.corridor_lateral_offset, cfg.corridor_target_speed)
        add(
            np.full(horizon, cfg.creep_acc, dtype=np.float64),
            np.concatenate([np.full(half, cfg.bypass_steer, dtype=np.float64),
                            np.linspace(cfg.bypass_steer, 0.0, horizon - half)]),
            "left_bypass_then_straight", "left_offset", cfg.corridor_lateral_offset, cfg.corridor_target_speed,
        )

        # --- Family 11: right_bypass_low_speed ---
        add(np.full(horizon, cfg.creep_acc * 0.5, dtype=np.float64), np.full(horizon, -cfg.bypass_steer, dtype=np.float64),
            "right_bypass_low_speed", "right_offset", -cfg.corridor_lateral_offset, cfg.corridor_target_speed)
        add(
            np.full(horizon, cfg.creep_acc, dtype=np.float64),
            np.concatenate([np.full(half, -cfg.bypass_steer, dtype=np.float64),
                            np.linspace(-cfg.bypass_steer, 0.0, horizon - half)]),
            "right_bypass_then_straight", "right_offset", -cfg.corridor_lateral_offset, cfg.corridor_target_speed,
        )

        # --- Supplement: lightweight random candidates ---
        num_random = min(max_random, max(0, max_structured + max_random - len(sequences)))
        for _ in range(num_random):
            sequences.append(self._sample_random_sequence(u_nom, horizon))
            families.append("random_shooting")
            corridor_meta.append(_make_corridor_meta("creep_forward", 0.0, cfg.corridor_target_speed))

        self._last_candidate_families = families
        self._last_candidate_corridors = corridor_meta

        drivable_available = left_feasible or right_feasible or True
        corridor_info = {
            "left_corridor_available": left_feasible,
            "right_corridor_available": right_feasible,
            "forward_corridor_available": True,
            "recenter_corridor_available": True,
            "drivable_corridor_available": drivable_available,
            "road_boundary_active": bool(road_boundary_active),
            "no_left_corridor": not left_feasible,
            "no_right_corridor": not right_feasible,
            "no_forward_corridor": False,
            "no_recenter_corridor": False,
            "mpc_num_corridors": 4,
            "active_recovery_corridor": self.active_recovery_corridor,
            "previous_recovery_corridor": self.previous_recovery_corridor,
        }
        return {
            "sequences": sequences,
            "info": corridor_info,
        }

    def _generate_corridor_candidate_sequences(
        self,
        state: State,
        obj: Dict[str, Any],
        object_kind: str,
        current_margin: float,
        u_nom: Action,
        u_cbf: Action,
    ) -> Dict[str, Any]:
        corridors, corridor_info = self._generate_recovery_corridors(state, obj, object_kind, current_margin)
        horizon = max(1, int(self.mpc_config.horizon_steps))
        max_candidates = max(1, int(self.mpc_config.num_samples))
        sequences: List[np.ndarray] = []
        families: List[str] = []
        corridor_meta: List[Dict[str, Any]] = []

        recovery_corridors = [corridor for corridor in corridors if corridor.get("corridor_type") != "minimum_risk_stop"]
        stop_corridors = [corridor for corridor in corridors if corridor.get("corridor_type") == "minimum_risk_stop"]
        ordered_corridors = recovery_corridors + stop_corridors
        if not ordered_corridors:
            ordered_corridors = [self._minimum_risk_corridor(state, "no_corridor_available")]

        per_corridor = max(4, max_candidates // max(1, len(ordered_corridors)))
        for corridor in ordered_corridors:
            for sequence, family in self._sequences_for_corridor(corridor, u_nom, u_cbf, horizon, per_corridor):
                if len(sequences) >= max_candidates:
                    break
                sequences.append(sequence)
                families.append(family)
                corridor_meta.append(corridor)
            if len(sequences) >= max_candidates:
                break

        while len(sequences) < max_candidates:
            sequence = self._sample_corridor_random_sequence(ordered_corridors[0], u_nom, horizon)
            sequences.append(sequence)
            families.append("corridor_random")
            corridor_meta.append(ordered_corridors[0])

        self._last_candidate_families = families[:max_candidates]
        self._last_candidate_corridors = corridor_meta[:max_candidates]
        corridor_info = dict(corridor_info)
        corridor_info["mpc_num_corridors"] = len(corridors)
        return {
            "sequences": sequences[:max_candidates],
            "corridors": corridors,
            "info": corridor_info,
        }

    def _generate_recovery_corridors(
        self,
        state: State,
        obj: Dict[str, Any],
        object_kind: str,
        current_margin: float,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        cfg = self.mpc_config
        lane_width = max(self.config.default_lane_width, self._current_lane_width(state))
        lanes = state.get("lanes", {}) or {}
        left_lane = lanes.get("left", {}) or {}
        right_lane = lanes.get("right", {}) or {}
        left_available = bool(left_lane.get("available", False) and left_lane.get("drivable", True))
        right_available = bool(right_lane.get("available", False) and right_lane.get("drivable", True))
        road_margin = self._current_road_boundary_margin(state)
        road_boundary_active = math.isfinite(road_margin) and road_margin < max(0.35, cfg.corridor_road_boundary_margin * 2.0)
        forward_available = bool(current_margin >= cfg.corridor_required_rss_margin)
        recenter_available = road_boundary_active or abs(self._ego(state).get("lateral", 0.0)) > lane_width * 0.35

        corridors: List[Dict[str, Any]] = []

        def make_corridor(
            corridor_type: str,
            target_lateral_offset: float,
            target_speed: float,
            target_progress: float,
            target_clearance: float,
            available: bool,
            reason: str,
            priority: float,
        ) -> Dict[str, Any]:
            return {
                "corridor_type": corridor_type,
                "target_lateral_offset": float(target_lateral_offset),
                "target_speed": float(target_speed),
                "target_progress": float(target_progress),
                "target_clearance": float(target_clearance),
                "required_rss_margin": float(cfg.corridor_required_rss_margin),
                "road_boundary_margin": float(cfg.corridor_road_boundary_margin),
                "available": bool(available),
                "reason": reason,
                "priority": float(priority),
                "road_boundary_active": bool(road_boundary_active),
            }

        if recenter_available:
            corridors.append(make_corridor(
                "recenter",
                0.0,
                min(cfg.corridor_target_speed, 2.0),
                cfg.corridor_target_progress,
                cfg.corridor_target_clearance,
                True,
                "road_boundary_or_large_lateral_offset",
                0.8,
            ))

        if left_available and self._corridor_boundary_feasible(state, cfg.corridor_lateral_offset):
            corridors.append(make_corridor(
                "left_offset",
                cfg.corridor_lateral_offset,
                cfg.corridor_target_speed,
                cfg.corridor_target_progress,
                cfg.corridor_target_clearance,
                True,
                "left_drivable_space_available",
                0.5,
            ))

        if right_available and self._corridor_boundary_feasible(state, -cfg.corridor_lateral_offset):
            corridors.append(make_corridor(
                "right_offset",
                -cfg.corridor_lateral_offset,
                cfg.corridor_target_speed,
                cfg.corridor_target_progress,
                cfg.corridor_target_clearance,
                True,
                "right_drivable_space_available",
                0.5,
            ))

        if forward_available and self._corridor_boundary_feasible(state, 0.0):
            corridors.append(make_corridor(
                "creep_forward",
                0.0,
                min(cfg.corridor_target_speed, 1.5),
                max(0.8, cfg.corridor_target_progress * 0.5),
                cfg.corridor_target_clearance,
                True,
                "front_rss_margin_allows_creep",
                0.6,
            ))

        drivable_available = any(c.get("corridor_type") != "minimum_risk_stop" for c in corridors)
        if not drivable_available:
            corridors.append(self._minimum_risk_corridor(state, "no_road_boundary_safe_recovery_corridor"))

        # Keep the active corridor near the front unless it is no longer represented.
        if self.active_recovery_corridor:
            corridors.sort(
                key=lambda c: (
                    0 if c.get("corridor_type") == self.active_recovery_corridor else 1,
                    c.get("priority", 1.0),
                )
            )
        else:
            corridors.sort(key=lambda c: c.get("priority", 1.0))

        info = {
            "left_corridor_available": any(c.get("corridor_type") == "left_offset" for c in corridors),
            "right_corridor_available": any(c.get("corridor_type") == "right_offset" for c in corridors),
            "forward_corridor_available": any(c.get("corridor_type") == "creep_forward" for c in corridors),
            "recenter_corridor_available": any(c.get("corridor_type") == "recenter" for c in corridors),
            "drivable_corridor_available": drivable_available,
            "road_boundary_active": bool(road_boundary_active),
            "no_left_corridor": not any(c.get("corridor_type") == "left_offset" for c in corridors),
            "no_right_corridor": not any(c.get("corridor_type") == "right_offset" for c in corridors),
            "no_forward_corridor": not any(c.get("corridor_type") == "creep_forward" for c in corridors),
            "no_recenter_corridor": not any(c.get("corridor_type") == "recenter" for c in corridors),
            "active_recovery_corridor": self.active_recovery_corridor,
            "previous_recovery_corridor": self.previous_recovery_corridor,
        }
        return corridors, info

    def _minimum_risk_corridor(self, state: State, reason: str) -> Dict[str, Any]:
        return {
            "corridor_type": "minimum_risk_stop",
            "target_lateral_offset": 0.0,
            "target_speed": 0.0,
            "target_progress": 0.0,
            "target_clearance": 0.0,
            "required_rss_margin": 0.0,
            "road_boundary_margin": float(self.mpc_config.corridor_road_boundary_margin),
            "available": True,
            "reason": reason,
            "priority": 99.0,
            "road_boundary_active": bool(self._current_road_boundary_margin(state) < self.mpc_config.corridor_road_boundary_margin),
        }

    def _current_road_boundary_margin(self, state: State) -> float:
        return self._road_boundary_margin_for_state(state, state)

    def _road_boundary_margin_for_state(self, reference_state: State, rollout_state: State) -> float:
        initial_ego = self._ego(reference_state)
        _, lateral = self._relative_position(initial_ego, self._ego(rollout_state))
        current_width = self._current_lane_width(reference_state)
        lanes = reference_state.get("lanes", {}) or {}
        left_width = self._lane_available_width(reference_state, lanes.get("left")) if lanes.get("left") else 0.0
        right_width = self._lane_available_width(reference_state, lanes.get("right")) if lanes.get("right") else 0.0
        left_available = bool((lanes.get("left") or {}).get("available", False))
        right_available = bool((lanes.get("right") or {}).get("available", False))
        upper = current_width / 2.0 + (left_width if left_available else 0.0) - self.config.lane_margin
        lower = -current_width / 2.0 - (right_width if right_available else 0.0) + self.config.lane_margin
        return min(upper - lateral, lateral - lower)

    def _corridor_boundary_feasible(self, state: State, target_lateral_offset: float) -> bool:
        current_width = self._current_lane_width(state)
        lanes = state.get("lanes", {}) or {}
        left_width = self._lane_available_width(state, lanes.get("left")) if lanes.get("left") else 0.0
        right_width = self._lane_available_width(state, lanes.get("right")) if lanes.get("right") else 0.0
        left_available = bool((lanes.get("left") or {}).get("available", False))
        right_available = bool((lanes.get("right") or {}).get("available", False))
        upper = current_width / 2.0 + (left_width if left_available else 0.0) - self.mpc_config.corridor_road_boundary_margin
        lower = -current_width / 2.0 - (right_width if right_available else 0.0) + self.mpc_config.corridor_road_boundary_margin
        return lower <= float(target_lateral_offset) <= upper

    def _sequences_for_corridor(
        self,
        corridor: Dict[str, Any],
        u_nom: Action,
        u_cbf: Action,
        horizon: int,
        limit: int,
    ) -> List[Tuple[np.ndarray, str]]:
        cfg = self.mpc_config
        corridor_type = str(corridor.get("corridor_type", "creep_forward"))
        target_speed = float(corridor.get("target_speed", cfg.corridor_target_speed))
        target_offset = float(corridor.get("target_lateral_offset", 0.0))
        direction = 0.0 if abs(target_offset) < self.config.small_tolerance else math.copysign(1.0, target_offset)
        steer_mag = min(cfg.max_steer, max(cfg.nudge_steer, abs(target_offset) / max(cfg.corridor_lateral_offset, 1e-6) * cfg.bypass_steer))
        steer_target = direction * steer_mag
        base_acc = max(cfg.creep_acc, min(cfg.max_acc, target_speed * 0.5))
        results: List[Tuple[np.ndarray, str]] = []

        def add(acc_values: Sequence[float], steer_values: Sequence[float], family: str) -> None:
            if len(results) >= limit:
                return
            sequence = np.asarray(
                [self._clip_action([acc, steer]) for acc, steer in zip(acc_values, steer_values)],
                dtype=np.float64,
            )
            if sequence.shape == (horizon, 2):
                results.append((sequence, family))

        if corridor_type == "minimum_risk_stop":
            add(np.full(horizon, cfg.strong_brake), np.full(horizon, u_nom[1]), "minimum_risk_stop")
            add(np.full(horizon, cfg.strong_brake), np.zeros(horizon), "minimum_risk_stop_zero_steer")
            return results

        if corridor_type == "recenter":
            recenter_steer = -math.copysign(min(cfg.nudge_steer, abs(u_nom[1])), u_nom[1]) if abs(u_nom[1]) > 0.05 else 0.0
            add(np.full(horizon, cfg.creep_acc), np.linspace(u_nom[1], 0.0, horizon), "recenter_smooth")
            add(np.full(horizon, base_acc), np.full(horizon, recenter_steer), "recenter_inward")
            add(np.linspace(cfg.creep_acc, base_acc, horizon), np.linspace(recenter_steer, 0.0, horizon), "recenter_then_straight")
        elif corridor_type in {"left_offset", "right_offset"}:
            half = max(1, horizon // 2)
            add(np.full(horizon, base_acc), np.full(horizon, steer_target), "{}_constant".format(corridor_type))
            add(
                np.full(horizon, base_acc),
                np.concatenate([np.full(half, steer_target), np.linspace(steer_target, 0.0, horizon - half)]),
                "{}_then_straight".format(corridor_type),
            )
            add(
                np.linspace(cfg.creep_acc, base_acc, horizon),
                np.concatenate([np.full(half, steer_target * 0.75), np.full(horizon - half, steer_target * 0.25)]),
                "{}_smooth".format(corridor_type),
            )
            add(
                np.concatenate([[cfg.comfort_brake], np.full(horizon - 1, base_acc)]),
                np.concatenate([[0.0], np.full(horizon - 1, steer_target)]),
                "{}_brake1_then_offset".format(corridor_type),
            )
        else:
            add(np.full(horizon, cfg.creep_acc), np.full(horizon, u_nom[1]), "creep_nominal_steer")
            add(np.full(horizon, base_acc), np.zeros(horizon), "creep_straight")
            add(np.linspace(cfg.creep_acc, base_acc, horizon), np.linspace(u_nom[1], 0.0, horizon), "creep_smooth")
            for steer in (cfg.nudge_steer, -cfg.nudge_steer):
                add(np.full(horizon, cfg.creep_acc), np.full(horizon, steer), "creep_nudge")

        while len(results) < limit:
            results.append((self._sample_corridor_random_sequence(corridor, u_nom, horizon), "corridor_random"))
        return results[:limit]

    def _sample_corridor_random_sequence(self, corridor: Dict[str, Any], u_nom: Action, horizon: int) -> np.ndarray:
        cfg = self.mpc_config
        target_offset = float(corridor.get("target_lateral_offset", 0.0))
        direction = 0.0 if abs(target_offset) < self.config.small_tolerance else math.copysign(1.0, target_offset)
        steer_mean = direction * min(cfg.max_steer, cfg.nudge_steer + 0.25 * abs(direction))
        acc_mean = max(cfg.creep_acc, min(cfg.max_acc, float(corridor.get("target_speed", 1.0)) * 0.45))
        nominal = np.asarray([acc_mean, steer_mean], dtype=np.float64)
        prev = nominal.copy()
        sequence = []
        alpha = max(0.0, min(1.0, float(cfg.random_smoothing_alpha)))
        for _ in range(horizon):
            raw = self.rng.normal(
                loc=nominal,
                scale=np.asarray([max(0.4, cfg.random_acc_std * 0.4), max(0.08, cfg.random_steer_std * 0.6)]),
            )
            smoothed = prev + alpha * (raw - prev)
            clipped = np.asarray(self._clip_action(smoothed), dtype=np.float64)
            sequence.append(clipped)
            prev = clipped
        return np.asarray(sequence, dtype=np.float64)

    def _generate_recovery_candidate_sequences(self, u_nom: Action, u_cbf: Optional[Action] = None) -> List[np.ndarray]:
        horizon = max(1, int(self.mpc_config.horizon_steps))
        max_candidates = max(1, int(self.mpc_config.num_samples))
        sequences: List[np.ndarray] = []
        families: List[str] = []

        def add_sequence(acc_values: Sequence[float], steer_values: Sequence[float], family: str = "") -> None:
            if len(sequences) >= max_candidates:
                return
            sequence = np.asarray(
                [self._clip_action([acc, steer]) for acc, steer in zip(acc_values, steer_values)],
                dtype=np.float64,
            )
            if sequence.shape == (horizon, 2):
                sequences.append(sequence)
                families.append(family)

        acc_nom, steer_nom = u_nom
        acc_cbf, steer_cbf = self._clip_action(u_cbf if u_cbf is not None else u_nom)
        cfg = self.mpc_config

        # --- Baseline deterministic candidates ---
        add_sequence(
            np.full(horizon, acc_cbf, dtype=np.float64),
            np.full(horizon, steer_cbf, dtype=np.float64),
            "cbf_continuous",
        )
        add_sequence(
            np.linspace(acc_nom, acc_cbf, horizon),
            np.linspace(steer_nom, steer_cbf, horizon),
            "interpolate_to_cbf",
        )
        add_sequence(
            np.full(horizon, cfg.comfort_brake, dtype=np.float64),
            np.full(horizon, steer_nom, dtype=np.float64),
            "comfort_brake",
        )
        add_sequence(
            np.full(horizon, cfg.strong_brake, dtype=np.float64),
            np.full(horizon, 0.0, dtype=np.float64),
            "strong_brake_straight",
        )

        # --- Creep candidates ---
        add_sequence(
            np.full(horizon, cfg.creep_acc, dtype=np.float64),
            np.full(horizon, steer_nom, dtype=np.float64),
            "creep_nominal",
        )
        for steer in (cfg.nudge_steer, -cfg.nudge_steer):
            add_sequence(
                np.full(horizon, cfg.creep_acc, dtype=np.float64),
                np.full(horizon, steer, dtype=np.float64),
                "creep_nudge",
            )
        for steer in (cfg.bypass_steer, -cfg.bypass_steer):
            add_sequence(
                np.full(horizon, cfg.creep_acc, dtype=np.float64),
                np.full(horizon, steer, dtype=np.float64),
                "creep_bypass",
            )
            straighten = np.concatenate(
                [
                    np.full(max(1, horizon // 2), steer, dtype=np.float64),
                    np.linspace(steer, 0.0, horizon - max(1, horizon // 2)),
                ]
            )
            add_sequence(
                np.full(horizon, cfg.creep_acc, dtype=np.float64),
                straighten,
                "bypass_then_straighten",
            )

        # --- Maneuver-biased recovery candidates (deadlock recovery only) ---
        # Steer first half, then creep straight (steer-before-creep)
        for steer in (cfg.bypass_steer, -cfg.bypass_steer):
            half = max(1, horizon // 2)
            resteer = np.concatenate([
                np.full(half, steer, dtype=np.float64),
                np.full(horizon - half, 0.0, dtype=np.float64),
            ])
            add_sequence(np.full(horizon, cfg.creep_acc, dtype=np.float64), resteer, "steer_first_then_creep")

        # Brake one step, then bypass (create space before turning)
        for steer in (cfg.bypass_steer, -cfg.bypass_steer):
            if horizon >= 2:
                acc_seq = np.concatenate([
                    [cfg.strong_brake],
                    np.full(horizon - 1, cfg.creep_acc, dtype=np.float64),
                ])
                steer_seq = np.concatenate([
                    [0.0],
                    np.full(horizon - 1, steer, dtype=np.float64),
                ])
                add_sequence(acc_seq, steer_seq, "brake1_then_bypass")

        # Brake two steps, then bypass
        for steer in (cfg.bypass_steer, -cfg.bypass_steer):
            if horizon >= 3:
                acc_seq = np.concatenate([
                    np.full(2, cfg.strong_brake, dtype=np.float64),
                    np.full(horizon - 2, cfg.creep_acc, dtype=np.float64),
                ])
                steer_seq = np.concatenate([
                    np.full(2, 0.0, dtype=np.float64),
                    np.full(horizon - 2, steer, dtype=np.float64),
                ])
                add_sequence(acc_seq, steer_seq, "brake2_then_bypass")

        # Recenter maneuver (gradually steer to zero, creep forward)
        if abs(steer_nom) > cfg.action_change_tolerance:
            add_sequence(
                np.full(horizon, cfg.creep_acc, dtype=np.float64),
                np.linspace(steer_nom, 0.0, horizon),
                "recenter_maneuver",
            )

        # Slow nudge with comfort brake
        for steer in (cfg.nudge_steer, -cfg.nudge_steer):
            add_sequence(
                np.full(horizon, cfg.comfort_brake, dtype=np.float64),
                np.full(horizon, steer, dtype=np.float64),
                "comfort_brake_nudge",
            )

        while len(sequences) < max_candidates:
            sequences.append(self._sample_random_sequence(u_nom, horizon))
            families.append("random_shooting")

        # Attach family metadata as a non-numeric attribute via object array trick
        # Store family info on each sequence container
        sequences_with_families = sequences[:max_candidates]
        self._last_candidate_families = families[:max_candidates]

        return sequences_with_families

    def _sample_random_sequence(self, u_nom: Action, horizon: int) -> np.ndarray:
        alpha = max(0.0, min(1.0, float(self.mpc_config.random_smoothing_alpha)))
        nominal = np.asarray(u_nom, dtype=np.float64)
        prev = nominal.copy()
        sequence = []

        for _ in range(horizon):
            raw = self.rng.normal(
                loc=nominal,
                scale=np.asarray(
                    [self.mpc_config.random_acc_std, self.mpc_config.random_steer_std],
                    dtype=np.float64,
                ),
            )
            smoothed = prev + alpha * (raw - prev)
            clipped = np.asarray(self._clip_action(smoothed), dtype=np.float64)
            sequence.append(clipped)
            prev = clipped

        return np.asarray(sequence, dtype=np.float64)

    def _evaluate_sequence(
        self,
        state: State,
        obj: Dict[str, Any],
        object_kind: str,
        current_margin: float,
        sequence: np.ndarray,
        u_nom: Action,
        u_cbf: Action,
        corridor: Optional[Dict[str, Any]] = None,
        is_deadlock_recovery: bool = False,
    ) -> Dict[str, Any]:
        corridor = corridor or {}
        rollout_state = copy.deepcopy(state)
        rollout_obj = copy.deepcopy(obj)
        margins: List[float] = []
        lateral_clearance_margins: List[float] = []
        lateral_rss_margins: List[float] = []
        lateral_distances: List[float] = []
        longitudinal_distances: List[float] = []
        road_boundary_margins: List[float] = []
        speeds: List[float] = []
        path_overlaps: List[bool] = []
        path_overlap_amounts: List[float] = []
        initial_lateral_clearance = self._lateral_clearance_margin(state, obj)
        initial_lateral_metrics = self.compute_lateral_rss_metrics(state, obj)
        initial_lateral_rss_margin = float(initial_lateral_metrics["lateral_rss_margin"])
        initial_lateral_distance = float(initial_lateral_metrics["lateral_distance"])
        initial_longitudinal_distance = float(initial_lateral_metrics["longitudinal_distance"])
        initial_path_overlap = bool(initial_lateral_metrics["path_overlap"])
        initial_path_overlap_amount = float(initial_lateral_metrics["path_overlap_amount"])
        initial_ego = self._ego(state)

        for action in sequence:
            rollout_state, rollout_obj = self._simulate_next_state_and_object(
                rollout_state, rollout_obj, object_kind, action
            )
            margin = self._rss_margin_for_object(rollout_state, rollout_obj, object_kind)
            lateral_clearance_margin = self._lateral_clearance_margin(rollout_state, rollout_obj)
            lateral_metrics = self.compute_lateral_rss_metrics(rollout_state, rollout_obj)
            lateral_rss_margin = float(lateral_metrics["lateral_rss_margin"])
            road_boundary_margin = self._road_boundary_margin_for_state(state, rollout_state)
            margins.append(float(margin))
            lateral_clearance_margins.append(float(lateral_clearance_margin))
            lateral_rss_margins.append(float(lateral_rss_margin))
            lateral_distances.append(float(lateral_metrics["lateral_distance"]))
            longitudinal_distances.append(float(lateral_metrics["longitudinal_distance"]))
            road_boundary_margins.append(float(road_boundary_margin))
            speeds.append(self._ego_speed(rollout_state))
            path_overlaps.append(bool(lateral_metrics["path_overlap"]))
            path_overlap_amounts.append(float(lateral_metrics["path_overlap_amount"]))

        final_margin = margins[-1] if margins else current_margin
        min_margin = min([current_margin] + margins) if margins else current_margin
        final_lateral_clearance = lateral_clearance_margins[-1] if lateral_clearance_margins else -math.inf
        min_lateral_clearance = min(lateral_clearance_margins) if lateral_clearance_margins else -math.inf
        final_lateral_rss_margin = lateral_rss_margins[-1] if lateral_rss_margins else -math.inf
        min_lateral_rss_margin = min(lateral_rss_margins) if lateral_rss_margins else -math.inf
        final_lateral_distance = lateral_distances[-1] if lateral_distances else initial_lateral_distance
        first_lateral_distance = lateral_distances[0] if lateral_distances else initial_lateral_distance
        final_longitudinal_distance = longitudinal_distances[-1] if longitudinal_distances else initial_longitudinal_distance
        first_longitudinal_distance = longitudinal_distances[0] if longitudinal_distances else initial_longitudinal_distance
        min_longitudinal_distance = min([initial_longitudinal_distance] + longitudinal_distances) if longitudinal_distances else initial_longitudinal_distance
        first_lateral_rss_margin = lateral_rss_margins[0] if lateral_rss_margins else initial_lateral_rss_margin
        final_path_overlap = path_overlaps[-1] if path_overlaps else initial_path_overlap
        first_path_overlap = path_overlaps[0] if path_overlaps else initial_path_overlap
        final_path_overlap_amount = path_overlap_amounts[-1] if path_overlap_amounts else initial_path_overlap_amount
        first_path_overlap_amount = path_overlap_amounts[0] if path_overlap_amounts else initial_path_overlap_amount
        lateral_rss_improvement = (final_lateral_rss_margin - initial_lateral_rss_margin) if (
            math.isfinite(final_lateral_rss_margin) and math.isfinite(initial_lateral_rss_margin)
        ) else 0.0
        path_overlap_reduced = (
            initial_path_overlap and not final_path_overlap
        ) or final_path_overlap_amount + self.config.small_tolerance < initial_path_overlap_amount
        first_step_path_overlap_reduces = (
            initial_path_overlap and not first_path_overlap
        ) or first_path_overlap_amount + self.config.small_tolerance < initial_path_overlap_amount
        first_step_lateral_margin_improves = (
            math.isfinite(first_lateral_rss_margin)
            and math.isfinite(initial_lateral_rss_margin)
            and first_lateral_rss_margin > initial_lateral_rss_margin + self.config.small_tolerance
        )
        first_step_lateral_distance_increases = first_lateral_distance > initial_lateral_distance + self.config.small_tolerance
        terminal_lateral_separation_safe = (
            math.isfinite(final_lateral_rss_margin)
            and final_lateral_rss_margin >= self.mpc_config.lateral_rss_terminal_safe_threshold
        )
        terminal_deconflicted = self._is_lateral_deconflicted(rollout_state, rollout_obj) or terminal_lateral_separation_safe
        terminal_lateral_deconflicted = bool(terminal_deconflicted and terminal_lateral_separation_safe)
        min_road_boundary_margin = min(road_boundary_margins) if road_boundary_margins else self._current_road_boundary_margin(state)
        final_road_boundary_margin = road_boundary_margins[-1] if road_boundary_margins else self._current_road_boundary_margin(state)
        progress = self._longitudinal_progress(initial_ego, self._ego(rollout_state))
        final_lateral_offset = self._lateral_progress(initial_ego, self._ego(rollout_state))
        final_speed = speeds[-1] if speeds else self._ego_speed(state)
        blocking_object_final = self._front_object_blocks_predicted_path(rollout_state, rollout_obj)
        road_boundary_safe = min_road_boundary_margin >= float(corridor.get("road_boundary_margin", self.mpc_config.corridor_road_boundary_margin))
        first_margin = margins[0] if margins else current_margin
        conservative_longitudinal_margin_safe = first_margin >= self.mpc_config.safety_margin_tolerance
        critical_threshold = float(self.mpc_config.lateral_escape_critical_longitudinal_margin)

        def _critical_margin_safe(margin: float) -> bool:
            return margin == math.inf or (math.isfinite(margin) and margin >= critical_threshold)

        critical_longitudinal_margin_safe = bool(
            _critical_margin_safe(float(current_margin))
            and _critical_margin_safe(float(first_margin))
            and _critical_margin_safe(float(min_margin))
        )
        immediate_longitudinal_margin_safe = conservative_longitudinal_margin_safe

        rss_feasible, recovery_used, reason = self._rss_sequence_feasibility(
            current_margin=current_margin,
            min_margin=min_margin,
            final_margin=final_margin,
        )
        recovery_margin_improvement = final_margin - current_margin
        terminal_recoverable, terminal_recovery_reason = self._terminal_recoverability(
            rss_feasible=rss_feasible,
            progress=progress,
            current_margin=current_margin,
            final_margin=final_margin,
            final_lateral_clearance=final_lateral_clearance,
            initial_lateral_clearance=initial_lateral_clearance,
            final_speed=final_speed,
            blocking_object_final=blocking_object_final,
            corridor=corridor,
            final_lateral_offset=final_lateral_offset,
            initial_road_boundary_margin=self._current_road_boundary_margin(state),
            final_road_boundary_margin=final_road_boundary_margin,
            final_lateral_rss_margin=final_lateral_rss_margin,
            initial_lateral_rss_margin=initial_lateral_rss_margin,
            lateral_rss_improvement=lateral_rss_improvement,
            path_overlap_reduced=path_overlap_reduced,
            terminal_deconflicted=terminal_deconflicted,
            final_path_overlap=final_path_overlap,
            min_margin=min_margin,
            road_boundary_safe=road_boundary_safe,
            terminal_lateral_separation_safe=terminal_lateral_separation_safe,
            predicted_path_overlap_reducing=path_overlap_reduced,
            immediate_longitudinal_margin_safe=immediate_longitudinal_margin_safe,
            critical_longitudinal_margin_safe=critical_longitudinal_margin_safe,
        )
        current_speed = self._ego_speed(state)
        first_acc = float(sequence[0][0]) if len(sequence) else self.mpc_config.strong_brake
        first_steer = float(sequence[0][1]) if len(sequence) else 0.0
        candidate_family = str(corridor.get("corridor_type", ""))
        is_lateral_escape_family = any(
            k in candidate_family for k in ("left_offset", "right_offset")
        ) or abs(float(corridor.get("target_lateral_offset", 0.0))) > self.config.small_tolerance
        invalid_lateral_escape_no_creep = bool(
            is_lateral_escape_family
            and first_acc <= 0.0
        )
        first_step_recovery_feasible = True
        first_step_recovery_reason = ""
        if current_speed <= self.mpc_config.stuck_speed_threshold:
            lateral_improves = (
                math.isfinite(final_lateral_clearance)
                and math.isfinite(initial_lateral_clearance)
                and float(final_lateral_clearance) - float(initial_lateral_clearance) > self.config.small_tolerance
            )
            target_offset = float(corridor.get("target_lateral_offset", 0.0))
            corridor_tracking_improves = (
                abs(target_offset) > self.config.small_tolerance
                and abs(target_offset - final_lateral_offset) + self.config.small_tolerance < abs(target_offset)
            )
            road_margin_improves = final_road_boundary_margin + self.config.small_tolerance > self._current_road_boundary_margin(state)
            is_lateral_recovery = terminal_recovery_reason in {
                "lateral_clearance_improved",
                "bypass_clearance_forming",
                "blocking_object_cleared",
                "corridor_tracking_improved",
                "recentered_from_boundary",
                "lateral_rss_margin_improved",
                "path_overlap_reduced",
                "front_object_deconflicted",
                "lateral_escape_safe",
                "right_escape_deconflicted",
                "left_escape_deconflicted",
            }
            steer_toward_corridor = (
                abs(target_offset) > self.config.small_tolerance
                and first_steer * target_offset > self.config.small_tolerance
            )
            has_significant_steer = (
                abs(first_steer) >= self.mpc_config.nudge_steer * self.mpc_config.nudge_steer_ratio_threshold
                and (steer_toward_corridor or abs(target_offset) <= self.config.small_tolerance)
            )
            if first_acc >= self.mpc_config.guard_override_min_acc:
                first_step_recovery_feasible = True
                first_step_recovery_reason = "first_acc_sufficient"
            elif has_significant_steer:
                first_step_recovery_feasible = True
                first_step_recovery_reason = "first_steer_toward_escape"
            elif first_step_lateral_margin_improves:
                first_step_recovery_feasible = True
                first_step_recovery_reason = "first_step_lateral_rss_margin_improves"
            elif first_step_path_overlap_reduces:
                first_step_recovery_feasible = True
                first_step_recovery_reason = "first_step_path_overlap_reduces"
            elif first_step_lateral_distance_increases:
                first_step_recovery_feasible = True
                first_step_recovery_reason = "first_step_lateral_distance_increases"
            elif lateral_improves:
                first_step_recovery_feasible = True
                first_step_recovery_reason = "lateral_clearance_improves"
            elif corridor_tracking_improves:
                first_step_recovery_feasible = True
                first_step_recovery_reason = "corridor_tracking_improves"
            elif road_margin_improves:
                first_step_recovery_feasible = True
                first_step_recovery_reason = "road_boundary_margin_improves"
            elif is_lateral_recovery:
                first_step_recovery_feasible = True
                first_step_recovery_reason = "terminal_lateral_recovery"
            elif terminal_deconflicted:
                first_step_recovery_feasible = True
                first_step_recovery_reason = "first_step_lateral_deconflicted"
            else:
                first_step_recovery_feasible = False
                first_step_recovery_reason = "first_step_no_progress"

        speed_feasible = all(
            -self.config.small_tolerance <= speed <= self.mpc_config.v_max + self.config.small_tolerance
            for speed in speeds
        )
        progress_feasible = progress >= -self.config.small_tolerance
        feasible = bool(
            rss_feasible
            and speed_feasible
            and road_boundary_safe
            and progress_feasible
            and terminal_recoverable
            and first_step_recovery_feasible
        )

        cost = self._sequence_cost(
            sequence=sequence,
            u_nom=u_nom,
            u_cbf=u_cbf,
            current_speed=current_speed,
            speeds=speeds,
            margins=margins,
            lateral_clearance_margins=lateral_clearance_margins,
            progress=progress,
            corridor=corridor,
            final_lateral_offset=final_lateral_offset,
            min_road_boundary_margin=min_road_boundary_margin,
            is_deadlock_recovery=is_deadlock_recovery,
            initial_lateral_clearance=initial_lateral_clearance,
        )

        if not first_step_recovery_feasible:
            reason = "recovery_first_step_no_progress"
        elif not road_boundary_safe:
            reason = "rollout_road_boundary_unsafe"
        elif not speed_feasible:
            reason = "rollout_speed_out_of_bounds"
        elif not progress_feasible:
            reason = "rollout_negative_progress"

        return {
            "sequence": sequence,
            "feasible": feasible,
            "cost": cost,
            "corridor_cost": cost,
            "rss_margin_current": current_margin,
            "rss_margin_min_pred": min_margin,
            "rss_margin_final_pred": final_margin,
            "rss_lateral_clearance_min_pred": min_lateral_clearance,
            "rss_lateral_clearance_final_pred": final_lateral_clearance,
            "road_boundary_margin_min_pred": min_road_boundary_margin,
            "road_boundary_margin_final_pred": final_road_boundary_margin,
            "road_boundary_safe": road_boundary_safe,
            "initial_lateral_clearance": initial_lateral_clearance,
            "final_lateral_offset": final_lateral_offset,
            "final_speed": final_speed,
            "predicted_progress": progress,
            "recovery_used": recovery_used,
            "rss_feasible": rss_feasible,
            "terminal_recoverable": terminal_recoverable,
            "terminal_recovery_reason": terminal_recovery_reason,
            "recovery_progress": progress,
            "recovery_margin_improvement": recovery_margin_improvement,
            "blocking_object_final": blocking_object_final,
            "reason": reason,
            "first_step_recovery_feasible": first_step_recovery_feasible,
            "first_step_recovery_reason": first_step_recovery_reason,
            "initial_lateral_distance": initial_lateral_distance,
            "predicted_lateral_distance": final_lateral_distance,
            "first_step_lateral_distance": first_lateral_distance,
            "initial_longitudinal_distance": initial_longitudinal_distance,
            "first_step_longitudinal_distance": first_longitudinal_distance,
            "predicted_longitudinal_distance": final_longitudinal_distance,
            "min_predicted_longitudinal_distance": min_longitudinal_distance,
            "initial_lateral_rss_margin": initial_lateral_rss_margin,
            "final_lateral_rss_margin": final_lateral_rss_margin,
            "predicted_lateral_rss_margin": final_lateral_rss_margin,
            "min_lateral_rss_margin": min_lateral_rss_margin,
            "lateral_rss_improvement": lateral_rss_improvement,
            "initial_path_overlap": initial_path_overlap,
            "final_path_overlap": final_path_overlap,
            "predicted_path_overlap": final_path_overlap,
            "initial_path_overlap_amount": initial_path_overlap_amount,
            "predicted_path_overlap_amount": final_path_overlap_amount,
            "path_overlap_reduced": path_overlap_reduced,
            "predicted_path_overlap_reducing": path_overlap_reduced,
            "terminal_deconflicted": terminal_deconflicted,
            "terminal_lateral_separation_safe": terminal_lateral_separation_safe,
            "terminal_lateral_deconflicted": terminal_lateral_deconflicted,
            "first_step_lateral_margin_improves": first_step_lateral_margin_improves if current_speed <= self.mpc_config.stuck_speed_threshold else False,
            "first_step_path_overlap_reduces": first_step_path_overlap_reduces if current_speed <= self.mpc_config.stuck_speed_threshold else False,
            "first_step_lateral_distance_increases": first_step_lateral_distance_increases if current_speed <= self.mpc_config.stuck_speed_threshold else False,
            "first_step_rss_margin_pred": first_margin,
            "immediate_longitudinal_margin_safe": immediate_longitudinal_margin_safe,
            "conservative_longitudinal_margin_safe": conservative_longitudinal_margin_safe,
            "critical_longitudinal_margin_safe": critical_longitudinal_margin_safe,
            "corridor": corridor,
            "corridor_type": corridor.get("corridor_type", ""),
            "corridor_target_lateral_offset": corridor.get("target_lateral_offset", math.nan),
            "corridor_target_speed": corridor.get("target_speed", math.nan),
            "invalid_lateral_escape_no_creep": invalid_lateral_escape_no_creep,
        }

    def _evaluate_guarded_first_action(self, state: State, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Run the first MPC action through RSS-CBF before accepting a recovery."""
        u_mpc = self._clip_action(evaluation["sequence"][0])
        u_guarded, guard_info = self.rss_cbf_filter.filter_action(state, u_mpc)
        u_guarded = self._clip_action(u_guarded)
        guard_delta = self._action_norm(u_guarded, u_mpc)
        guard_mode = guard_info.get("mode", "")
        guard_fallback = guard_mode == "fallback_no_safe_candidate"
        certified_lateral_escape_used, certified_lateral_escape_side, certified_lateral_escape_reason = (
            self._check_certified_lateral_escape(state, evaluation, u_mpc, u_guarded, guard_mode)
        )
        if certified_lateral_escape_used:
            guard_override_used = True
            guard_override_reason = certified_lateral_escape_reason
        else:
            guard_override_used, guard_override_reason = self._guard_override_for_certified_recovery(
                state=state,
                evaluation=evaluation,
                u_mpc=u_mpc,
                u_guarded=u_guarded,
                guard_mode=guard_mode,
            )

        u_effective = u_mpc if (guard_override_used or certified_lateral_escape_used) else u_guarded

        guarded_acc = float(u_effective[0])
        current_speed = self._ego_speed(state)
        guard_cost = self.mpc_config.guard_delta_cost_weight * guard_delta * guard_delta
        guard_cost += self.mpc_config.guard_brake_cost_weight * max(0.0, -float(u_guarded[0])) ** 2
        if current_speed <= self.mpc_config.stuck_speed_threshold and guarded_acc <= self.mpc_config.comfort_brake:
            guard_cost += self.mpc_config.guard_stall_brake_cost
        if guard_override_used or certified_lateral_escape_used:
            guard_cost *= 0.1

        guard_safe = guard_override_used or certified_lateral_escape_used or not guard_fallback
        if guard_safe and not (guard_override_used or certified_lateral_escape_used) and current_speed <= self.mpc_config.stuck_speed_threshold:
            progress_guarded = guarded_acc >= self.mpc_config.guard_override_min_acc
            lateral_recovery = abs(float(u_effective[1])) >= min(0.25, self.mpc_config.nudge_steer)
            lateral_clearance_improves = (
                math.isfinite(evaluation.get("rss_lateral_clearance_final_pred", -math.inf))
                and math.isfinite(evaluation.get("rss_lateral_clearance_min_pred", -math.inf))
                and float(evaluation["rss_lateral_clearance_final_pred"])
                - float(evaluation["rss_lateral_clearance_min_pred"]) > self.config.small_tolerance
            )
            is_lateral_terminal_recovery = evaluation.get("terminal_recovery_reason") in {
                "lateral_clearance_improved",
                "bypass_clearance_forming",
                "blocking_object_cleared",
                "corridor_tracking_improved",
                "recentered_from_boundary",
                "lateral_rss_margin_improved",
                "path_overlap_reduced",
                "front_object_deconflicted",
                "lateral_escape_safe",
                "right_escape_deconflicted",
                "left_escape_deconflicted",
            }
            lateral_rss_improves = float(evaluation.get("lateral_rss_improvement", 0.0)) > self.config.small_tolerance
            path_overlap_reducing = bool(evaluation.get("path_overlap_reduced", False))
            terminal_deconflicted = bool(evaluation.get("terminal_deconflicted", False))
            first_steer_toward_escape = (
                abs(float(u_mpc[1])) >= self.mpc_config.nudge_steer * self.mpc_config.nudge_steer_ratio_threshold
            )
            lateral_distance_increases = bool(evaluation.get("first_step_lateral_distance_increases", False))
            guard_safe = (progress_guarded or lateral_recovery
                          or lateral_clearance_improves or is_lateral_terminal_recovery
                          or lateral_rss_improves or path_overlap_reducing or terminal_deconflicted
                          or (first_steer_toward_escape and (lateral_rss_improves or path_overlap_reducing or lateral_distance_increases)))

        mpc_acc = float(u_mpc[0])
        effective_acc = float(u_effective[0])
        mpc_throttle = max(0.0, mpc_acc / max(self.config.max_acc, 1e-6))
        effective_throttle = max(0.0, effective_acc / max(self.config.max_acc, 1e-6))
        mpc_brake = max(0.0, -mpc_acc / max(abs(self.config.min_acc), 1e-6))
        effective_brake = max(0.0, -effective_acc / max(abs(self.config.min_acc), 1e-6))
        candidate_category = self._family_category(str(evaluation.get("recovery_candidate_family", "")))
        only_steering_no_creep = bool(
            candidate_category in ("left", "right")
            and abs(float(u_effective[1])) > 0.05
            and effective_acc <= 0.0
            and mpc_acc > 0.0
        )
        lateral_escape_throttle_suppressed_guard = bool(
            candidate_category in ("left", "right")
            and mpc_acc > 0.0
            and effective_acc <= 0.0
        )
        creep_suppression_reason_guard = ""
        if lateral_escape_throttle_suppressed_guard:
            if guard_fallback:
                creep_suppression_reason_guard = "cbf_fallback_no_safe_candidate_suppressed_creep"
            elif guard_mode == "rss_cbf_recovery":
                creep_suppression_reason_guard = "cbf_recovery_projected_acc_to_brake"
            elif guard_mode == "rss_cbf_intervention":
                creep_suppression_reason_guard = "cbf_intervention_projected_acc_to_zero"
            else:
                creep_suppression_reason_guard = "guard_overrode_creep_acc_to_brake"

        guard_reject_reason = ""
        if not guard_safe:
            lateral_category = self._family_category(str(evaluation.get("recovery_candidate_family", "")))
            if guard_fallback:
                guard_reject_reason = (
                    "{}_escape_guard_rejected".format(lateral_category)
                    if lateral_category in ("left", "right")
                    else "cbf_fallback_no_safe_candidate"
                )
            elif lateral_category in ("left", "right"):
                guard_reject_reason = "{}_escape_guard_rejected".format(lateral_category)
            else:
                guard_reject_reason = "first_step_recovery_rejected_after_cbf_guard"

        return {
            "guard_safe": bool(guard_safe),
            "guard_cost": float(guard_cost),
            "u_mpc_first": u_mpc,
            "u_guarded_first": u_effective,
            "guard_info": guard_info,
            "cbf_guard_used": (not guard_override_used and not certified_lateral_escape_used) and guard_delta > self.mpc_config.action_change_tolerance,
            "cbf_guard_delta": float(guard_delta),
            "cbf_guard_override_used": bool(guard_override_used),
            "cbf_guard_override_reason": guard_override_reason,
            "certified_recovery_override_used": bool(guard_override_used),
            "certified_recovery_override_reason": guard_override_reason,
            "certified_lateral_escape_used": bool(certified_lateral_escape_used),
            "certified_lateral_escape_side": certified_lateral_escape_side,
            "certified_lateral_escape_reason": certified_lateral_escape_reason,
            "lateral_escape_guard_pass_through": bool(certified_lateral_escape_used),
            "lateral_escape_guard_reject_reason": guard_reject_reason,
            "guard_rejected_lateral_escape": bool(
                not guard_safe
                and self._family_category(str(evaluation.get("recovery_candidate_family", ""))) in ("left", "right")
            ),
            "guard_reject_reason": guard_reject_reason,
            "mpc_action_before_guard": [float(u_mpc[0]), float(u_mpc[1])],
            "action_after_guard": [float(u_effective[0]), float(u_effective[1])],
            "selected_throttle_before_guard": float(mpc_throttle),
            "selected_throttle_after_guard": float(effective_throttle),
            "selected_brake_before_guard": float(mpc_brake),
            "selected_brake_after_guard": float(effective_brake),
            "only_steering_no_creep": only_steering_no_creep,
            "lateral_escape_throttle_suppressed_guard": lateral_escape_throttle_suppressed_guard,
            "creep_suppression_reason_guard": creep_suppression_reason_guard,
        }

    def _guard_override_for_certified_recovery(
        self,
        state: State,
        evaluation: Dict[str, Any],
        u_mpc: Action,
        u_guarded: Action,
        guard_mode: str,
    ) -> Tuple[bool, str]:
        """Permit MPC-certified creep when one-step CBF is overly conservative.

        RSS-CBF requires the margin to be non-decreasing, which can freeze a
        stopped vehicle behind a static blocker even when the RSS buffer remains
        large. In deadlock recovery only, the MPC horizon can certify that a
        small forward/bypass action keeps all predicted RSS margins positive.
        """
        current_speed = self._ego_speed(state)
        if current_speed > self.mpc_config.stuck_speed_threshold:
            return False, ""
        candidate_category = self._family_category(str(evaluation.get("recovery_candidate_family", "")))
        if not bool(evaluation.get("terminal_recoverable", False)):
            return False, ""
        if not bool(evaluation.get("road_boundary_safe", False)):
            return False, ""
        if str((evaluation.get("corridor") or {}).get("corridor_type", "")) == "minimum_risk_stop":
            return False, ""
        if guard_mode not in {"rss_cbf_intervention", "rss_cbf_recovery", "fallback_no_safe_candidate"}:
            return False, ""

        if candidate_category not in ("left", "right"):
            if float(u_mpc[0]) < self.mpc_config.guard_override_min_acc:
                return False, ""
            if guard_mode != "fallback_no_safe_candidate" and float(u_guarded[0]) >= self.mpc_config.guard_override_min_acc:
                return False, ""

        margin_buffer = (
            self.mpc_config.guard_override_fallback_margin_buffer
            if guard_mode == "fallback_no_safe_candidate"
            else self.mpc_config.guard_override_margin_buffer
        )
        min_margin = float(evaluation.get("rss_margin_min_pred", -math.inf))
        final_margin = float(evaluation.get("rss_margin_final_pred", -math.inf))
        current_margin = float(evaluation.get("rss_margin_current", math.inf))

        if candidate_category in ("left", "right"):
            lateral_rss_improvement = float(evaluation.get("lateral_rss_improvement", 0.0))
            path_overlap_reduced = bool(evaluation.get("path_overlap_reduced", False))
            terminal_deconflicted = bool(evaluation.get("terminal_deconflicted", False))
            lateral_improving = evaluation.get("terminal_recovery_reason") in {
                "blocking_object_cleared",
                "bypass_clearance_forming",
                "lateral_clearance_improved",
                "corridor_tracking_improved",
                "recentered_from_boundary",
                "lateral_rss_margin_improved",
                "path_overlap_reduced",
                "front_object_deconflicted",
                "lateral_escape_safe",
                "right_escape_deconflicted",
                "left_escape_deconflicted",
            }
            if not lateral_improving:
                return False, ""
            if bool(evaluation.get("lateral_escape_certified", False)):
                prefix = "mpc_certified_cbf_fallback" if guard_mode == "fallback_no_safe_candidate" else "mpc_certified"
                reason = "{}_{}_lateral_creep_override".format(prefix, candidate_category)
                if path_overlap_reduced:
                    reason += "_path_overlap_reduced"
                elif terminal_deconflicted:
                    reason += "_deconflicted"
                elif lateral_rss_improvement > self.config.small_tolerance:
                    reason += "_lateral_rss_improved"
                return True, reason
            certified_margin = (
                math.isfinite(min_margin)
                and math.isfinite(final_margin)
                and min_margin >= -self.mpc_config.safety_margin_tolerance
                and current_margin >= -self.mpc_config.safety_margin_tolerance
            )
            if not certified_margin:
                return False, ""
            if float(evaluation.get("road_boundary_margin_min_pred", -math.inf)) < self.mpc_config.corridor_road_boundary_margin:
                return False, ""
            prefix = "mpc_certified_cbf_fallback" if guard_mode == "fallback_no_safe_candidate" else "mpc_certified"
            reason = "{}_{}_lateral_creep_override".format(prefix, candidate_category)
            if path_overlap_reduced:
                reason += "_path_overlap_reduced"
            elif terminal_deconflicted:
                reason += "_deconflicted"
            elif lateral_rss_improvement > self.config.small_tolerance:
                reason += "_lateral_rss_improved"
            return True, reason

        certified_margin = (
            math.isfinite(min_margin)
            and math.isfinite(final_margin)
            and min_margin >= margin_buffer
            and final_margin >= margin_buffer
            and current_margin >= margin_buffer
        )
        if not certified_margin:
            return False, ""
        if float(evaluation.get("road_boundary_margin_min_pred", -math.inf)) < self.mpc_config.corridor_road_boundary_margin:
            return False, ""

        progress = float(evaluation.get("predicted_progress", 0.0))
        lateral_clearance = float(evaluation.get("rss_lateral_clearance_final_pred", -math.inf))
        lateral_improving = evaluation.get("terminal_recovery_reason") in {
            "blocking_object_cleared",
            "bypass_clearance_forming",
            "lateral_clearance_improved",
            "corridor_tracking_improved",
            "recentered_from_boundary",
            "lateral_rss_margin_improved",
            "path_overlap_reduced",
            "front_object_deconflicted",
            "lateral_escape_safe",
            "right_escape_deconflicted",
            "left_escape_deconflicted",
        }
        if progress <= self.config.small_tolerance and not lateral_improving:
            return False, ""

        if lateral_improving:
            prefix = "mpc_certified_cbf_fallback" if guard_mode == "fallback_no_safe_candidate" else "mpc_certified"
            return True, "{}_lateral_recovery_with_large_rss_margin".format(prefix)
        if lateral_clearance >= self.mpc_config.recovery_lateral_clearance_threshold:
            prefix = "mpc_certified_cbf_fallback" if guard_mode == "fallback_no_safe_candidate" else "mpc_certified"
            return True, "{}_clearance_recovery_with_large_rss_margin".format(prefix)
        prefix = "mpc_certified_cbf_fallback" if guard_mode == "fallback_no_safe_candidate" else "mpc_certified"
        return True, "{}_creep_with_large_rss_margin".format(prefix)

    def _evaluate_lateral_escape_gate(self, state: State, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        family = str(evaluation.get("candidate_family", evaluation.get("recovery_candidate_family", "")))
        candidate_side = self._family_category(family)
        is_lateral_escape = candidate_side in ("left", "right")
        result: Dict[str, Any] = {
            "lateral_escape_candidate": False,
            "lateral_escape_side": "",
            "lateral_escape_certified": False,
            "lateral_escape_certification_reason": "",
            "lateral_escape_reject_reason": "",
            "lateral_escape_used_relaxed_longitudinal_gate": False,
            "lateral_escape_rejected_by_conservative_gate": False,
            "lateral_escape_rejected_by_critical_margin": False,
            "lateral_escape_lateral_rss_safe": False,
            "lateral_escape_lateral_margin_improved": False,
            "lateral_escape_path_overlap_reduced": False,
            "lateral_escape_terminal_recoverable": False,
            "lateral_escape_terminal_reason": "",
            "lateral_escape_low_speed_creep": False,
            "lateral_escape_no_immediate_collision_risk": False,
            "lateral_escape_steer_toward_escape": False,
        }
        if not is_lateral_escape:
            return result

        sequence = evaluation.get("sequence")
        if sequence is not None and len(sequence):
            first_acc = float(sequence[0][0])
            first_steer = float(sequence[0][1])
        else:
            first_acc = self.mpc_config.strong_brake
            first_steer = 0.0

        current_speed = self._ego_speed(state)
        max_escape_speed = max(self.mpc_config.stuck_speed_threshold * 3.0, self.config.small_tolerance)
        low_speed = current_speed <= max_escape_speed
        min_acc = float(self.mpc_config.lateral_escape_min_acc)
        max_acc = float(self.mpc_config.lateral_escape_max_acc)
        positive_creep = first_acc > 0.0
        low_speed_creep = bool(low_speed and min_acc <= first_acc <= max_acc)
        invalid_no_creep = bool(not positive_creep)
        steer_threshold = self.mpc_config.nudge_steer * self.mpc_config.nudge_steer_ratio_threshold
        steer_toward_escape = bool(
            abs(first_steer) >= steer_threshold
            and ((candidate_side == "right" and first_steer < -self.config.small_tolerance)
                 or (candidate_side == "left" and first_steer > self.config.small_tolerance))
        )

        final_lateral_rss_margin = float(evaluation.get("final_lateral_rss_margin", -math.inf))
        lateral_rss_safe = bool(
            math.isfinite(final_lateral_rss_margin)
            and final_lateral_rss_margin >= self.mpc_config.lateral_rss_terminal_safe_threshold
        )
        lateral_margin_improved = bool(
            float(evaluation.get("lateral_rss_improvement", 0.0))
            >= float(self.mpc_config.lateral_escape_min_lateral_margin_improvement)
        )
        initial_overlap_amount = float(evaluation.get("initial_path_overlap_amount", math.inf))
        final_overlap_amount = float(evaluation.get("predicted_path_overlap_amount", math.inf))
        overlap_reduction = initial_overlap_amount - final_overlap_amount
        path_overlap_reduced = bool(
            evaluation.get("path_overlap_reduced", False)
            or overlap_reduction >= float(self.mpc_config.lateral_escape_min_overlap_reduction)
            or (bool(evaluation.get("initial_path_overlap", False)) and not bool(evaluation.get("final_path_overlap", True)))
        )
        road_safe = bool(evaluation.get("road_boundary_safe", False))
        terminal_lateral_safe = bool(evaluation.get("terminal_lateral_separation_safe", False))
        terminal_recoverable = bool(evaluation.get("terminal_recoverable", False))
        terminal_reason = str(evaluation.get("terminal_recovery_reason", ""))
        conservative_safe = bool(evaluation.get(
            "conservative_longitudinal_margin_safe",
            evaluation.get("immediate_longitudinal_margin_safe", False),
        ))
        critical_safe = bool(evaluation.get("critical_longitudinal_margin_safe", False))
        initial_longitudinal_distance = float(evaluation.get("initial_longitudinal_distance", math.inf))
        first_longitudinal_distance = float(evaluation.get("first_step_longitudinal_distance", math.inf))
        immediate_distance_clear = bool(
            (not math.isfinite(initial_longitudinal_distance) or initial_longitudinal_distance > self.config.small_tolerance)
            and (not math.isfinite(first_longitudinal_distance) or first_longitudinal_distance > self.config.small_tolerance)
        )
        no_immediate_collision_risk = bool(critical_safe and immediate_distance_clear)
        lateral_condition = bool(lateral_rss_safe or lateral_margin_improved)

        result.update({
            "lateral_escape_candidate": True,
            "lateral_escape_side": candidate_side,
            "lateral_escape_lateral_rss_safe": lateral_rss_safe,
            "lateral_escape_lateral_margin_improved": lateral_margin_improved,
            "lateral_escape_path_overlap_reduced": path_overlap_reduced,
            "lateral_escape_terminal_recoverable": terminal_recoverable,
            "lateral_escape_terminal_reason": terminal_reason,
            "lateral_escape_low_speed_creep": low_speed_creep,
            "lateral_escape_no_immediate_collision_risk": no_immediate_collision_risk,
            "lateral_escape_steer_toward_escape": steer_toward_escape,
            "invalid_lateral_escape_no_creep": bool(evaluation.get("invalid_lateral_escape_no_creep", False) or invalid_no_creep),
        })

        reject_reason = ""
        if invalid_no_creep:
            reject_reason = "invalid_lateral_escape_no_creep"
        elif not low_speed:
            reject_reason = "lateral_escape_not_low_speed"
        elif first_acc < min_acc:
            reject_reason = "invalid_lateral_escape_no_creep"
        elif first_acc > max_acc:
            reject_reason = "lateral_escape_acc_too_aggressive"
        elif not steer_toward_escape:
            reject_reason = "lateral_escape_wrong_steer_direction"
        elif not road_safe:
            reject_reason = "lateral_escape_not_road_safe"
        elif not critical_safe:
            reject_reason = "lateral_escape_rejected_by_critical_margin"
        elif not no_immediate_collision_risk:
            reject_reason = "lateral_escape_immediate_collision_risk"
        elif not lateral_condition:
            reject_reason = "lateral_escape_lateral_rss_unsafe"
        elif not path_overlap_reduced:
            reject_reason = "lateral_escape_path_overlap_not_reduced"
        elif not terminal_lateral_safe:
            reject_reason = "lateral_escape_terminal_not_recoverable"
        elif not terminal_recoverable:
            reject_reason = "lateral_escape_terminal_not_recoverable"

        if reject_reason:
            result["lateral_escape_reject_reason"] = reject_reason
            result["lateral_escape_rejected_by_critical_margin"] = reject_reason == "lateral_escape_rejected_by_critical_margin"
            return result

        if candidate_side == "right":
            reason = "right_escape_deconflicted"
        elif candidate_side == "left":
            reason = "left_escape_deconflicted"
        else:
            reason = "lateral_escape_safe"
        if path_overlap_reduced:
            reason = "path_overlap_reduced" if candidate_side not in ("left", "right") else reason
        elif lateral_margin_improved:
            reason = "lateral_rss_margin_improved"

        result.update({
            "lateral_escape_certified": True,
            "lateral_escape_certification_reason": reason,
            "lateral_escape_used_relaxed_longitudinal_gate": bool(
                not bool(evaluation.get("rss_feasible", False)) or not conservative_safe
            ),
            "longitudinal_constraint_relaxed_by_lateral_escape": bool(
                not bool(evaluation.get("rss_feasible", False)) or not conservative_safe
            ),
        })
        return result

    def _check_certified_lateral_escape(
        self,
        state: State,
        evaluation: Dict[str, Any],
        u_mpc: Action,
        u_guarded: Action,
        guard_mode: str,
    ) -> Tuple[bool, str, str]:
        current_speed = self._ego_speed(state)
        if current_speed > self.mpc_config.stuck_speed_threshold * 3:
            return False, "", ""
        candidate_side = self._family_category(str(evaluation.get("recovery_candidate_family", "")))
        if candidate_side not in ("left", "right"):
            return False, "", ""
        if not bool(evaluation.get("lateral_escape_certified", False)):
            return False, "", ""
        if not bool(evaluation.get("terminal_recoverable", False)):
            return False, "", ""
        if not bool(evaluation.get("road_boundary_safe", False)):
            return False, "", ""
        if not bool(evaluation.get("critical_longitudinal_margin_safe", False)):
            return False, "", ""
        if not bool(evaluation.get("lateral_escape_no_immediate_collision_risk", False)):
            return False, "", ""
        if guard_mode not in {"rss_cbf_intervention", "rss_cbf_recovery", "fallback_no_safe_candidate"}:
            return False, "", ""

        path_overlap_reduced = bool(evaluation.get("lateral_escape_path_overlap_reduced", False))
        terminal_deconflicted = bool(evaluation.get("terminal_lateral_deconflicted", evaluation.get("terminal_deconflicted", False)))
        lateral_rss_improvement = float(evaluation.get("lateral_rss_improvement", 0.0))
        lateral_safe_or_improving = bool(
            evaluation.get("lateral_escape_lateral_rss_safe", False)
            or evaluation.get("lateral_escape_lateral_margin_improved", False)
        )
        if not lateral_safe_or_improving or not path_overlap_reduced:
            return False, "", ""
        if not bool(evaluation.get("terminal_lateral_separation_safe", False)):
            return False, "", ""

        min_road_boundary = float(evaluation.get("road_boundary_margin_min_pred", -math.inf))
        if min_road_boundary < self.mpc_config.corridor_road_boundary_margin:
            return False, "", ""

        acc = float(u_mpc[0])
        if acc < self.mpc_config.lateral_escape_min_acc - self.config.small_tolerance:
            return False, "", ""
        if acc > self.mpc_config.lateral_escape_max_acc + self.config.small_tolerance:
            return False, "", ""
        steer_threshold = self.mpc_config.nudge_steer * self.mpc_config.nudge_steer_ratio_threshold
        if abs(float(u_mpc[1])) < steer_threshold:
            return False, "", ""
        if candidate_side == "right" and float(u_mpc[1]) >= -self.config.small_tolerance:
            return False, "", ""
        if candidate_side == "left" and float(u_mpc[1]) <= self.config.small_tolerance:
            return False, "", ""

        reason = "certified_{}_lateral_creep".format(candidate_side)
        if path_overlap_reduced:
            reason += "_path_overlap_reduced"
        elif terminal_deconflicted:
            reason += "_deconflicted"
        elif lateral_rss_improvement > self.config.small_tolerance:
            reason += "_lateral_rss_improved"

        return True, candidate_side, reason

    def _rss_sequence_feasibility(
        self,
        current_margin: float,
        min_margin: float,
        final_margin: float,
    ) -> Tuple[bool, bool, str]:
        tolerance = float(self.mpc_config.safety_margin_tolerance)
        if current_margin >= -self.config.small_tolerance:
            return min_margin >= tolerance, False, "mpc_feasible_rss_sequence"

        improvement = final_margin - current_margin
        improves_enough = improvement + self.config.small_tolerance >= self.config.recovery_margin_improvement
        holds_margin = (
            self.config.recovery_allow_equal_margin
            and final_margin + self.config.small_tolerance >= current_margin
        )
        recovery = bool(improves_enough or holds_margin)
        return recovery, recovery, "mpc_recovery_sequence" if recovery else "mpc_recovery_not_improving"

    def _sequence_cost(
        self,
        sequence: np.ndarray,
        u_nom: Action,
        u_cbf: Action,
        current_speed: float,
        speeds: Sequence[float],
        margins: Sequence[float],
        lateral_clearance_margins: Sequence[float],
        progress: float,
        corridor: Optional[Dict[str, Any]] = None,
        final_lateral_offset: float = 0.0,
        min_road_boundary_margin: float = math.inf,
        is_deadlock_recovery: bool = False,
        initial_lateral_clearance: float = -math.inf,
    ) -> float:
        cfg = self.mpc_config
        corridor = corridor or {}
        nominal = np.asarray(u_nom, dtype=np.float64)
        deltas = sequence - nominal
        intervention_cost = float(np.sum(np.sum(deltas * deltas, axis=1)))

        previous = np.vstack([nominal.reshape(1, 2), sequence[:-1]])
        smooth_deltas = sequence - previous
        smooth_cost = float(np.sum(np.sum(smooth_deltas * smooth_deltas, axis=1)))
        cbf_reference = np.asarray(self._clip_action(u_cbf), dtype=np.float64)
        cbf_anchor_cost = 0.0
        if self._action_norm(cbf_reference, nominal) > self.mpc_config.action_change_tolerance:
            cbf_deltas = sequence - cbf_reference
            cbf_anchor_cost = float(np.sum(np.sum(cbf_deltas * cbf_deltas, axis=1)))

        v_ref = min(float(current_speed) + 2.0, 12.0)
        speed_cost = float(np.sum([(speed - v_ref) ** 2 for speed in speeds]))

        tolerance = float(cfg.safety_margin_tolerance)
        rss_violation_penalty = 0.0
        for margin in margins:
            if math.isfinite(margin) and margin < tolerance:
                rss_violation_penalty += (tolerance - margin) ** 2

        brake_penalty = float(
            np.sum([max(0.0, cfg.comfort_brake - float(action[0])) ** 2 for action in sequence])
        )
        stall_penalty = 0.0
        if current_speed <= cfg.stuck_speed_threshold:
            progress_gap = max(0.0, cfg.stuck_progress_target - float(progress))
            stall_penalty = progress_gap ** 2
        lateral_clearance_reward = 0.0
        if lateral_clearance_margins:
            finite_clearances = [
                margin for margin in lateral_clearance_margins if math.isfinite(float(margin))
            ]
            if finite_clearances:
                lateral_clearance_reward = max(0.0, float(finite_clearances[-1]))
        target_offset = float(corridor.get("target_lateral_offset", 0.0))
        corridor_lateral_cost = (float(final_lateral_offset) - target_offset) ** 2
        target_clearance = float(corridor.get("target_clearance", cfg.corridor_target_clearance))
        final_clearance = lateral_clearance_margins[-1] if lateral_clearance_margins else -math.inf
        clearance_cost = max(0.0, target_clearance - float(final_clearance)) ** 2 if math.isfinite(final_clearance) else 0.0
        road_cost = max(0.0, float(corridor.get("road_boundary_margin", cfg.corridor_road_boundary_margin)) - float(min_road_boundary_margin)) ** 2
        corridor_switch_cost = 0.0
        corridor_type = str(corridor.get("corridor_type", ""))
        if self.active_recovery_corridor and corridor_type and corridor_type != self.active_recovery_corridor:
            corridor_switch_cost += cfg.corridor_switch_penalty
            if {corridor_type, self.active_recovery_corridor} == {"left_offset", "right_offset"}:
                corridor_switch_cost += cfg.corridor_switch_penalty * max(1, self.recovery_side_switch_cooldown)
        if corridor_type == self.active_recovery_corridor and corridor_type:
            corridor_switch_cost -= cfg.corridor_keep_margin

        cost = (
            cfg.recovery_w_intervention * intervention_cost
            + cfg.w_smooth * smooth_cost
            + cfg.recovery_w_cbf_anchor * cbf_anchor_cost
            + cfg.recovery_w_speed * speed_cost
            - cfg.recovery_w_progress * float(progress)
            + cfg.w_rss_violation * rss_violation_penalty
            + cfg.w_brake * brake_penalty
            + cfg.w_stall * stall_penalty
            - cfg.w_lateral_clearance * lateral_clearance_reward
            + cfg.w_corridor_lateral * corridor_lateral_cost
            + cfg.w_corridor_clearance * clearance_cost
            + cfg.w_road_boundary * road_cost
            + corridor_switch_cost
        )

        if is_deadlock_recovery:
            recovery_brake_penalty = float(
                np.sum([max(0.0, -float(action[0])) ** 2 for action in sequence])
            )
            cost += cfg.recovery_brake_penalty_weight * recovery_brake_penalty

            if lateral_clearance_margins:
                final_lc = lateral_clearance_margins[-1] if math.isfinite(lateral_clearance_margins[-1]) else -math.inf
                init_lc = initial_lateral_clearance if math.isfinite(initial_lateral_clearance) else -math.inf
                if math.isfinite(final_lc) and math.isfinite(init_lc):
                    lateral_improvement = final_lc - init_lc
                    if lateral_improvement > 0:
                        cost -= cfg.recovery_lateral_improvement_bonus * lateral_improvement

            has_creep = any(float(action[0]) > 0 for action in sequence[:max(1, len(sequence) // 3)])
            has_lateral = abs(final_lateral_offset) > 0.3
            if has_creep and has_lateral:
                cost -= cfg.recovery_creep_lateral_bonus

            if current_speed <= cfg.stuck_speed_threshold:
                avg_speed = float(np.mean(speeds)) if speeds else 0.0
                if avg_speed <= cfg.stuck_speed_threshold:
                    cost += cfg.recovery_long_stall_penalty

        return float(cost)

    def _terminal_recoverability(
        self,
        rss_feasible: bool,
        progress: float,
        current_margin: float,
        final_margin: float,
        final_lateral_clearance: float,
        initial_lateral_clearance: float,
        final_speed: float,
        blocking_object_final: bool,
        corridor: Optional[Dict[str, Any]] = None,
        final_lateral_offset: float = 0.0,
        initial_road_boundary_margin: float = math.inf,
        final_road_boundary_margin: float = math.inf,
        final_lateral_rss_margin: float = -math.inf,
        initial_lateral_rss_margin: float = -math.inf,
        lateral_rss_improvement: float = 0.0,
        path_overlap_reduced: bool = False,
        terminal_deconflicted: bool = False,
        final_path_overlap: bool = True,
        min_margin: float = -math.inf,
        road_boundary_safe: bool = False,
        terminal_lateral_separation_safe: bool = False,
        predicted_path_overlap_reducing: bool = False,
        immediate_longitudinal_margin_safe: bool = False,
        critical_longitudinal_margin_safe: bool = False,
    ) -> Tuple[bool, str]:
        corridor = corridor or {}
        if corridor.get("corridor_type") == "minimum_risk_stop":
            if rss_feasible and final_speed <= max(self.mpc_config.stuck_speed_threshold, 0.5):
                return True, "minimum_risk_condition"
            return False, "minimum_risk_stop_not_reached"

        margin_improvement = final_margin - current_margin
        lateral_improvement = final_lateral_clearance - initial_lateral_clearance
        target_offset = float(corridor.get("target_lateral_offset", 0.0))
        corridor_tracking_improvement = abs(target_offset) - abs(target_offset - float(final_lateral_offset))
        road_margin_improvement = float(final_road_boundary_margin) - float(initial_road_boundary_margin)
        is_lateral_escape = abs(target_offset) > self.config.small_tolerance or abs(final_lateral_offset) > 0.3
        escape_side = "right" if final_lateral_offset < -self.config.small_tolerance else ("left" if final_lateral_offset > self.config.small_tolerance else "")
        lateral_margin_improved = lateral_rss_improvement > self.config.small_tolerance
        lateral_escape_certified = bool(
            is_lateral_escape
            and road_boundary_safe
            and critical_longitudinal_margin_safe
            and (math.isfinite(min_margin) or min_margin == math.inf)
            and lateral_margin_improved
            and predicted_path_overlap_reducing
            and terminal_lateral_separation_safe
            and terminal_deconflicted
        )

        if lateral_escape_certified:
            if escape_side == "right":
                return True, "right_escape_deconflicted"
            if escape_side == "left":
                return True, "left_escape_deconflicted"
            return True, "front_object_deconflicted"

        if is_lateral_escape and road_boundary_safe and terminal_lateral_separation_safe:
            if lateral_margin_improved and predicted_path_overlap_reducing:
                if escape_side == "right":
                    return True, "right_escape_deconflicted"
                if escape_side == "left":
                    return True, "left_escape_deconflicted"
                return True, "lateral_escape_safe"
            if path_overlap_reduced and terminal_deconflicted:
                if escape_side == "right":
                    return True, "right_escape_deconflicted"
                if escape_side == "left":
                    return True, "left_escape_deconflicted"
                return True, "lateral_escape_safe"

        if not rss_feasible:
            return False, "rss_not_feasible"

        if progress >= self.mpc_config.recovery_progress_threshold:
            return True, "progress_recovered"
        if margin_improvement >= self.mpc_config.recovery_margin_improvement_threshold:
            return True, "rss_margin_improved"
        if not blocking_object_final:
            if terminal_lateral_separation_safe and terminal_deconflicted:
                return True, "front_object_deconflicted"
            return True, "blocking_object_cleared"

        if lateral_rss_improvement > self.mpc_config.path_overlap_reduction_threshold:
            return True, "lateral_rss_margin_improved"
        if path_overlap_reduced:
            return True, "path_overlap_reduced"
        if terminal_deconflicted and not final_path_overlap:
            return True, "front_object_deconflicted"

        if is_lateral_escape and terminal_deconflicted:
            if final_lateral_rss_margin >= self.mpc_config.lateral_rss_terminal_safe_threshold:
                if escape_side == "right":
                    return True, "right_escape_deconflicted"
                elif escape_side == "left":
                    return True, "left_escape_deconflicted"
                return True, "lateral_escape_safe"

        if final_lateral_clearance >= self.mpc_config.recovery_lateral_clearance_threshold:
            return True, "bypass_clearance_forming"
        if lateral_improvement >= self.mpc_config.recovery_lateral_improvement_threshold:
            return True, "lateral_clearance_improved"
        if abs(target_offset) > self.config.small_tolerance and corridor_tracking_improvement >= self.mpc_config.recovery_lateral_improvement_threshold:
            return True, "corridor_tracking_improved"
        if corridor.get("corridor_type") == "recenter" and road_margin_improvement > self.config.small_tolerance:
            return True, "recentered_from_boundary"
        if final_margin >= self.mpc_config.guard_override_margin_buffer and final_speed >= self.mpc_config.recovery_terminal_speed_threshold:
            return True, "handoff_to_rl_safe"
        if final_speed >= self.mpc_config.recovery_terminal_speed_threshold and final_margin >= self.mpc_config.safety_margin_tolerance:
            return True, "terminal_speed_recovered"
        if initial_lateral_clearance < 0 and lateral_improvement > self.config.small_tolerance:
            return True, "lateral_clearance_improving_from_negative"
        if (math.isfinite(final_margin) and final_margin >= self.mpc_config.guard_override_margin_buffer
                and margin_improvement >= -self.config.small_tolerance):
            return True, "rss_margin_maintained_large_buffer"
        return False, "safe_but_no_recovery_progress"

    def _simulate_next_state_and_object(
        self,
        state: State,
        obj: Dict[str, Any],
        object_kind: str,
        action: Sequence[float],
    ) -> Tuple[State, Dict[str, Any]]:
        next_state = self._simulate_next_state(state, action)
        next_obj = copy.deepcopy(obj)

        if object_kind == "dynamic":
            heading = float(next_obj.get("heading", self._ego(next_state).get("heading", 0.0)))
            speed = max(0.0, float(next_obj.get("speed", 0.0)))
            next_obj["x"] = float(next_obj.get("x", 0.0)) + speed * math.cos(heading) * self.mpc_config.dt
            next_obj["y"] = float(next_obj.get("y", 0.0)) + speed * math.sin(heading) * self.mpc_config.dt

        return next_state, next_obj

    def _rss_margin_for_object(self, state: State, obj: Dict[str, Any], object_kind: str) -> float:
        if not self._front_object_blocks_predicted_path(state, obj):
            return math.inf
        return self._distance_to_obstacle_front(state, obj) - self._rss_cbf_distance(state, obj, object_kind)

    def _front_object_blocks_predicted_path(self, state: State, obj: Dict[str, Any]) -> bool:
        longitudinal, _ = self._relative_position(self._ego(state), obj)
        if longitudinal <= 0.0:
            return False
        if self.config.enable_lateral_rss:
            metrics = self.compute_lateral_rss_metrics(state, obj)
            if (
                float(metrics["lateral_rss_margin"]) >= self.mpc_config.lateral_rss_deconfliction_threshold
                and not bool(metrics["path_overlap"])
            ):
                return False
        lateral_margin = self._lateral_clearance_margin(state, obj)
        if lateral_margin >= 0.0:
            return False
        return True

    def _front_object_path_overlap(self, state: State, obj: Dict[str, Any]) -> bool:
        _, lateral = self._relative_position(self._ego(state), obj)
        half_ego = self.config.vehicle_width / 2.0 + self.config.obstacle_margin
        half_obj = self._object_width(obj, self.config.vehicle_width) / 2.0
        return abs(lateral) < (half_ego + half_obj)

    def _lateral_rss_margin_for_object(self, state: State, obj: Dict[str, Any]) -> float:
        return self.compute_lateral_rss_margin(state, obj)

    def _is_lateral_deconflicted(self, state: State, obj: Dict[str, Any]) -> bool:
        lateral_margin = self._lateral_clearance_margin(state, obj)
        if lateral_margin >= self.mpc_config.lateral_rss_deconfliction_threshold:
            return True
        lat_rss_margin = self._lateral_rss_margin_for_object(state, obj)
        return lat_rss_margin >= self.mpc_config.lateral_rss_terminal_safe_threshold

    def _lateral_clearance_margin(self, state: State, obj: Dict[str, Any]) -> float:
        _, lateral = self._relative_position(self._ego(state), obj)
        required_clearance = (
            self.config.vehicle_width / 2.0
            + self._object_width(obj, self.config.vehicle_width) / 2.0
            + self.config.obstacle_margin
            + self.mpc_config.lateral_clearance_buffer
        )
        return abs(lateral) - required_clearance

    def _make_mpc_info(
        self,
        state: State,
        obj: Optional[Dict[str, Any]],
        object_kind: str,
        mode: str,
        reason: str,
        u_original: Action,
        u_safe: Action,
        d_front: float,
        d_rss: float,
        rss_margin_current: float,
        dynamic_vehicle_detected: bool,
        obstacle_detected: bool,
        mpc_success: bool,
        mpc_num_candidates: int,
        mpc_num_feasible: int,
        mpc_best_cost: float,
        rss_margin_min_pred: float,
        rss_margin_final_pred: float,
        rss_lateral_clearance_min_pred: float,
        rss_lateral_clearance_final_pred: float,
        predicted_progress: float,
        fallback_used: bool,
        cbf_info: Optional[Dict[str, Any]] = None,
        cbf_reference_info: Optional[Dict[str, Any]] = None,
        cbf_guard_used: bool = False,
        cbf_guard_delta: float = 0.0,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if mode not in self.FORMAL_MODES:
            mode = "rss_mpc_intervention"

        info = {
            "mode": mode,
            "reason": reason,
            "mpc_success": bool(mpc_success),
            "mpc_num_candidates": int(mpc_num_candidates),
            "mpc_num_feasible": int(mpc_num_feasible),
            "mpc_best_cost": float(mpc_best_cost),
            "rss_margin": rss_margin_current,
            "rss_margin_current": rss_margin_current,
            "rss_margin_min_pred": rss_margin_min_pred,
            "rss_margin_final_pred": rss_margin_final_pred,
            "rss_lateral_clearance_min_pred": rss_lateral_clearance_min_pred,
            "rss_lateral_clearance_final_pred": rss_lateral_clearance_final_pred,
            "predicted_progress": predicted_progress,
            "fallback_used": bool(fallback_used),
            "cbf_mode": (cbf_info or {}).get("mode", ""),
            "cbf_reason": (cbf_info or {}).get("reason", ""),
            "cbf_action_delta": (cbf_info or {}).get("action_delta", math.nan),
            "cbf_acc_safe": (cbf_info or {}).get("acc_safe", math.nan),
            "cbf_steer_safe": (cbf_info or {}).get("steer_safe", math.nan),
            "cbf_reference_mode": (cbf_reference_info or {}).get("mode", ""),
            "cbf_reference_action_delta": (cbf_reference_info or {}).get("action_delta", math.nan),
            "cbf_reference_acc_safe": (cbf_reference_info or {}).get("acc_safe", math.nan),
            "cbf_reference_steer_safe": (cbf_reference_info or {}).get("steer_safe", math.nan),
            "cbf_guard_used": bool(cbf_guard_used),
            "cbf_guard_delta": float(cbf_guard_delta),
            "rss_distance": d_rss,
            "d_front": d_front,
            "d_dynamic": d_rss if object_kind == "dynamic" else math.nan,
            "d_obs": d_front if object_kind == "static" else math.nan,
            "d_brake": d_rss,
            "object_kind": object_kind,
            "dynamic_vehicle_detected": bool(dynamic_vehicle_detected),
            "obstacle_detected": bool(obstacle_detected),
            "left_feasible": False,
            "right_feasible": False,
            "state_debug": self._state_debug(state),
            "blocking_object": self._object_debug(obj, state) if obj is not None else {},
        }
        info.update(
            {
                "selective_mpc_enabled": True,
                "mpc_called": False,
                "mpc_call_reason": "",
                "deadlock_risk": False,
                "deadlock_score": 0.0,
                "deadlock_counter": self.deadlock_counter,
                "deadlock_window_progress": 0.0,
                "deadlock_window_avg_speed": math.nan,
                "deadlock_cbf_active_ratio": 0.0,
                "deadlock_cbf_fallback_ratio": 0.0,
                "deadlock_reason": "",
                "terminal_recoverable": False,
                "terminal_recovery_reason": "",
                "recovery_progress": math.nan,
                "recovery_margin_improvement": math.nan,
                "blocking_object_final": False,
                "minimum_risk_stop_used": False,
                "mpc_terminal_feasible": 0,
                "mpc_guard_rejected": 0,
                "cbf_guard_override_used": False,
                "cbf_guard_override_reason": "",
                "certified_recovery_override_used": False,
                "certified_recovery_override_reason": "",
                "guard_reject_reason": "",
                "lateral_guard_reject_reason": "",
                "recovery_hold_used": False,
                "mpc_failure_reason": "",
                "mpc_num_corridors": 0,
                "mpc_num_rss_feasible": 0,
                "mpc_num_road_safe": 0,
                "mpc_num_terminal_recoverable": 0,
                "mpc_num_guard_rejected": 0,
                "mpc_no_rss_feasible_count": 0,
                "mpc_no_road_safe_count": 0,
                "mpc_no_terminal_recoverable_count": 0,
                "mpc_guard_rejected_count": 0,
                "mpc_terminal_candidate_count": 0,
                "active_recovery_corridor": self.active_recovery_corridor,
                "selected_recovery_corridor": "",
                "previous_recovery_corridor": self.previous_recovery_corridor,
                "corridor_switch_used": False,
                "corridor_switch_reason": "",
                "left_corridor_available": False,
                "right_corridor_available": False,
                "forward_corridor_available": False,
                "recenter_corridor_available": False,
                "drivable_corridor_available": False,
                "no_left_corridor": True,
                "no_right_corridor": True,
                "no_forward_corridor": True,
                "no_recenter_corridor": True,
                "road_boundary_active": False,
                "corridor_target_lateral_offset": math.nan,
                "corridor_target_speed": math.nan,
                "corridor_cost": math.nan,
                "corridor_terminal_recoverable": False,
                "first_step_recovery_feasible": False,
                "first_step_recovery_reason": "",
                "recovery_horizon_steps_used": self.mpc_config.horizon_steps,
                "candidate_family": "",
                "recovery_candidate_family": "",
                "selected_candidate_family": "",
                "best_candidate_family": "",
                "num_candidates_brake": 0,
                "num_candidates_creep": 0,
                "num_candidates_left": 0,
                "num_candidates_right": 0,
                "best_brake_cost": math.nan,
                "best_creep_cost": math.nan,
                "best_left_cost": math.nan,
                "best_right_cost": math.nan,
                "best_left_terminal_recoverable": False,
                "best_right_terminal_recoverable": False,
                "best_left_guard_rejected": False,
                "best_right_guard_rejected": False,
                "left_reject_reason": "",
                "right_reject_reason": "",
                "brake_selected_reason": "",
                "mpc_time_ms": math.nan,
                "candidate_generation_time_ms": math.nan,
                "candidate_evaluation_time_ms": math.nan,
                "num_total_candidates": 0,
                "num_random_candidates": 0,
                "num_structured_candidates": 0,
                "lateral_rss_margin": math.nan,
                "lateral_rss_safe_distance": math.nan,
                "longitudinal_distance": math.nan,
                "front_object_path_overlap": False,
                "front_object_path_overlap_reducing": False,
                "front_object_lateral_distance": math.nan,
                "front_object_lateral_margin": math.nan,
                "front_object_deconflicted": False,
                "longitudinal_constraint_relaxed_by_lateral_escape": False,
                "initial_lateral_rss_margin": math.nan,
                "final_lateral_rss_margin": math.nan,
                "predicted_lateral_distance": math.nan,
                "initial_longitudinal_distance": math.nan,
                "first_step_longitudinal_distance": math.nan,
                "predicted_longitudinal_distance": math.nan,
                "min_predicted_longitudinal_distance": math.nan,
                "predicted_lateral_rss_margin": math.nan,
                "predicted_path_overlap": False,
                "predicted_path_overlap_reducing": False,
                "terminal_lateral_separation_safe": False,
                "terminal_lateral_deconflicted": False,
                "road_boundary_safe": False,
                "immediate_longitudinal_margin_safe": False,
                "conservative_longitudinal_margin_safe": False,
                "critical_longitudinal_margin_safe": False,
                "lateral_rss_improvement": 0.0,
                "initial_path_overlap": False,
                "final_path_overlap": False,
                "path_overlap_reduced": False,
                "terminal_deconflicted": False,
                "first_step_lateral_margin_improves": False,
                "first_step_path_overlap_reduces": False,
                "first_step_lateral_distance_increases": False,
                "lateral_escape_candidate": False,
                "lateral_escape_side": "",
                "lateral_escape_certified": False,
                "lateral_escape_certification_reason": "",
                "lateral_escape_reject_reason": "",
                "lateral_escape_used_relaxed_longitudinal_gate": False,
                "lateral_escape_rejected_by_conservative_gate": False,
                "lateral_escape_rejected_by_critical_margin": False,
                "lateral_escape_lateral_rss_safe": False,
                "lateral_escape_lateral_margin_improved": False,
                "lateral_escape_path_overlap_reduced": False,
                "lateral_escape_terminal_recoverable": False,
                "lateral_escape_terminal_reason": "",
                "lateral_escape_low_speed_creep": False,
                "lateral_escape_no_immediate_collision_risk": False,
                "lateral_escape_steer_toward_escape": False,
                "lateral_escape_guard_pass_through": False,
                "lateral_escape_guard_reject_reason": "",
                "certified_lateral_escape_used": False,
                "certified_lateral_escape_side": "",
                "certified_lateral_escape_reason": "",
                "guard_rejected_lateral_escape": False,
                "mpc_action_before_guard": [math.nan, math.nan],
                "action_after_guard": [math.nan, math.nan],
                "right_steer_value_used": math.nan,
                "left_steer_value_used": math.nan,
                "selected_steer_before_guard": math.nan,
                "selected_steer_after_guard": math.nan,
                "right_road_safe": False,
                "left_road_safe": False,
                "right_lateral_rss_safe": False,
                "left_lateral_rss_safe": False,
                "right_escape_available": False,
                "left_escape_available": False,
                "right_escape_reject_reason": "",
                "left_escape_reject_reason": "",
                "certified_lateral_creep_available": False,
                "certified_lateral_creep_used": False,
                "certified_lateral_creep_side": "",
                "certified_lateral_creep_reason": "",
                "brake_selected_despite_certified_creep": False,
                "lateral_escape_throttle_suppressed": False,
                "creep_suppression_reason": "",
                "selected_acc_before_guard": math.nan,
                "selected_acc_after_guard": math.nan,
                "selected_steer_before_guard": math.nan,
                "selected_steer_after_guard": math.nan,
                "selected_throttle_before_guard": math.nan,
                "selected_throttle_after_guard": math.nan,
                "selected_brake_before_guard": math.nan,
                "selected_brake_after_guard": math.nan,
                "creep_acc_value_used": math.nan,
                "action_mapping_note": "",
                "invalid_lateral_escape_no_creep": False,
                "only_steering_no_creep": False,
                "lateral_creep_failure_reason": "",
            }
        )
        if obj is not None and self.config.enable_lateral_rss:
            lat_metrics = self.compute_lateral_rss_metrics(state, obj)
            path_overlap, lat_dist, lat_mrg = self.check_path_overlap(state, obj)
            lat_margin = float(lat_metrics["lateral_rss_margin"])
            lateral_horizon = self._rss_cbf_lateral_horizon_metrics(state, obj, object_kind, u_safe)
            info.update({
                "lateral_rss_margin": lat_margin,
                "lateral_rss_safe_distance": float(lat_metrics["lateral_rss_safe_distance"]),
                "longitudinal_distance": float(lat_metrics["longitudinal_distance"]),
                "front_object_path_overlap": bool(path_overlap),
                "front_object_path_overlap_reducing": bool(lateral_horizon.get("path_overlap_reducing", False)),
                "front_object_lateral_distance": float(lat_dist),
                "front_object_lateral_margin": float(lat_mrg),
                "front_object_deconflicted": bool(self.check_lateral_deconflicted(state, obj, lat_margin)),
                "longitudinal_constraint_relaxed_by_lateral_escape": bool(
                    self._front_object_lateral_constraint_relaxed(state, obj)
                ),
            })
        if extra_info:
            info.update(extra_info)
        if self.deadlock_history:
            self.deadlock_history[-1]["mode"] = mode
        return self._with_action_debug(info, u_original, u_safe)

    def _action_norm(self, action: Sequence[float], reference: Sequence[float]) -> float:
        action = np.asarray(self._clip_action(action), dtype=np.float64)
        reference = np.asarray(self._clip_action(reference), dtype=np.float64)
        return float(np.linalg.norm(action - reference))
