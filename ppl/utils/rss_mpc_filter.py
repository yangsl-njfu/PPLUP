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

    def reset(self) -> None:
        """Clear rolling deadlock state at episode reset."""
        self.deadlock_history.clear()
        self.deadlock_counter = 0
        self._last_route_completion = math.nan
        self._held_recovery_action = None
        self._held_recovery_ttl = 0
        self._last_candidate_families = []

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

        recovery_horizon = max(1, int(self.mpc_config.recovery_horizon_steps))
        original_horizon = self.mpc_config.horizon_steps
        self.mpc_config.horizon_steps = recovery_horizon
        try:
            candidates = self._generate_recovery_candidate_sequences(u_original, cbf_reference_action)
        finally:
            self.mpc_config.horizon_steps = original_horizon

        best: Optional[Dict[str, Any]] = None
        feasible_count = 0
        terminal_feasible_count = 0
        no_rss_feasible_count = 0
        no_terminal_recoverable_count = 0
        guard_rejected_count = 0
        terminal_feasible_candidates: List[Dict[str, Any]] = []
        guard_rejected_candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

        for idx, sequence in enumerate(candidates):
            evaluation = self._evaluate_sequence(
                state=state,
                obj=obj,
                object_kind=object_kind,
                current_margin=current_margin,
                sequence=sequence,
                u_nom=u_original,
                u_cbf=cbf_reference_action,
            )
            if idx < len(self._last_candidate_families):
                evaluation["recovery_candidate_family"] = self._last_candidate_families[idx]
            else:
                evaluation["recovery_candidate_family"] = "unknown"
            rss_feasible, _, _ = self._rss_sequence_feasibility(
                current_margin=current_margin,
                min_margin=evaluation["rss_margin_min_pred"],
                final_margin=evaluation["rss_margin_final_pred"],
            )
            if not rss_feasible:
                no_rss_feasible_count += 1
                continue
            if not evaluation["terminal_recoverable"]:
                no_terminal_recoverable_count += 1
                continue
            terminal_feasible_count += 1
            terminal_feasible_candidates.append(evaluation)

        guard_eval_limit = max(1, int(self.mpc_config.guard_eval_top_k))
        terminal_feasible_candidates.sort(key=lambda item: item["cost"])
        for evaluation in terminal_feasible_candidates[:guard_eval_limit]:
            guard_eval = self._evaluate_guarded_first_action(state, evaluation)
            if not guard_eval["guard_safe"]:
                guard_rejected_count += 1
                guard_rejected_candidates.append((evaluation, guard_eval))
                continue
            evaluation.update(guard_eval)
            evaluation["cost"] += guard_eval["guard_cost"]
            feasible_count += 1
            if best is None or evaluation["cost"] < best["cost"]:
                best = evaluation
        if best is None and len(terminal_feasible_candidates) > guard_eval_limit:
            for evaluation in terminal_feasible_candidates[guard_eval_limit:]:
                guard_eval = self._evaluate_guarded_first_action(state, evaluation)
                if not guard_eval["guard_safe"]:
                    guard_rejected_count += 1
                    guard_rejected_candidates.append((evaluation, guard_eval))
                    continue
                evaluation.update(guard_eval)
                evaluation["cost"] += guard_eval["guard_cost"]
                feasible_count += 1
                if best is None or evaluation["cost"] < best["cost"]:
                    best = evaluation

        total_candidates = len(candidates)
        if best is not None:
            failure_reason = ""
        elif no_rss_feasible_count == total_candidates:
            failure_reason = "no_rss_feasible_candidate"
        elif terminal_feasible_count == 0:
            failure_reason = "no_terminal_recoverable_candidate"
        elif feasible_count == 0:
            failure_reason = "terminal_candidates_guard_rejected"
        elif feasible_count == 0 and terminal_feasible_count == 0:
            failure_reason = "no_drivable_recovery_corridor"
        else:
            failure_reason = "horizon_too_short_or_no_progress"

        def _diagnostics_block() -> Dict[str, Any]:
            return {
                "mpc_failure_reason": failure_reason,
                "mpc_no_rss_feasible_count": no_rss_feasible_count,
                "mpc_no_terminal_recoverable_count": no_terminal_recoverable_count,
                "mpc_guard_rejected_count": guard_rejected_count,
                "mpc_terminal_candidate_count": terminal_feasible_count,
                "recovery_horizon_steps_used": recovery_horizon,
            }

        if best is not None:
            u_mpc = self._clip_action(best["u_mpc_first"])
            u_safe = self._clip_action(best["u_guarded_first"])
            guard_info = best["guard_info"]
            cbf_guard_delta = best["cbf_guard_delta"]
            cbf_guard_used = best["cbf_guard_used"]
            guard_override_used = best.get("cbf_guard_override_used", False)
            if guard_override_used:
                mode = "rss_mpc_recovery"
                reason = "mpc_recovery_large_margin_guard_override"
                self._remember_recovery_action(u_safe)
            else:
                mode = "rss_mpc_cbf_guard" if cbf_guard_used else "rss_mpc_recovery"
                reason = "cbf_guard_projected_mpc_recovery" if cbf_guard_used else "mpc_recovery_sequence"
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
                mpc_success=True,
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
                        "terminal_recoverable": best["terminal_recoverable"],
                        "terminal_recovery_reason": best["terminal_recovery_reason"],
                        "recovery_progress": best["recovery_progress"],
                        "recovery_margin_improvement": best["recovery_margin_improvement"],
                        "blocking_object_final": best["blocking_object_final"],
                        "mpc_terminal_feasible": terminal_feasible_count,
                        "mpc_guard_rejected": guard_rejected_count,
                        "cbf_guard_override_used": guard_override_used,
                        "cbf_guard_override_reason": best.get("cbf_guard_override_reason", ""),
                        "first_step_recovery_reason": best.get("first_step_recovery_reason", ""),
                        "recovery_candidate_family": best.get("recovery_candidate_family", ""),
                        **_diagnostics_block(),
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
                        "recovery_hold_used": True,
                        **_diagnostics_block(),
                    },
                ),
            )

        # Try guard override on guard-rejected terminal candidates with large margin
        if (guard_rejected_candidates
                and math.isfinite(current_margin)
                and current_margin >= self.mpc_config.guard_override_fallback_margin_buffer):
            guard_rejected_candidates.sort(
                key=lambda item: item[0].get("cost", 1e9)
            )
            for evaluation, guard_eval in guard_rejected_candidates:
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
                    return u_safe, self._make_mpc_info(
                        state=state,
                        obj=obj,
                        object_kind=object_kind,
                        mode="rss_mpc_recovery",
                        reason="mpc_recovery_guard_rejected_override",
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
                                "terminal_recoverable": evaluation["terminal_recoverable"],
                                "terminal_recovery_reason": evaluation["terminal_recovery_reason"],
                                "recovery_progress": evaluation["recovery_progress"],
                                "recovery_margin_improvement": evaluation["recovery_margin_improvement"],
                                "blocking_object_final": evaluation["blocking_object_final"],
                                "mpc_terminal_feasible": terminal_feasible_count,
                                "mpc_guard_rejected": guard_rejected_count,
                                "cbf_guard_override_used": True,
                                "cbf_guard_override_reason": override_reason,
                                "first_step_recovery_reason": evaluation.get("first_step_recovery_reason", ""),
                                "recovery_candidate_family": evaluation.get("recovery_candidate_family", ""),
                                **_diagnostics_block(),
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
    ) -> Dict[str, Any]:
        rollout_state = copy.deepcopy(state)
        rollout_obj = copy.deepcopy(obj)
        margins: List[float] = []
        lateral_clearance_margins: List[float] = []
        speeds: List[float] = []
        initial_lateral_clearance = self._lateral_clearance_margin(state, obj)

        for action in sequence:
            rollout_state, rollout_obj = self._simulate_next_state_and_object(
                rollout_state, rollout_obj, object_kind, action
            )
            margin = self._rss_margin_for_object(rollout_state, rollout_obj, object_kind)
            lateral_clearance_margin = self._lateral_clearance_margin(rollout_state, rollout_obj)
            margins.append(float(margin))
            lateral_clearance_margins.append(float(lateral_clearance_margin))
            speeds.append(self._ego_speed(rollout_state))

        final_margin = margins[-1] if margins else current_margin
        min_margin = min([current_margin] + margins) if margins else current_margin
        final_lateral_clearance = lateral_clearance_margins[-1] if lateral_clearance_margins else -math.inf
        min_lateral_clearance = min(lateral_clearance_margins) if lateral_clearance_margins else -math.inf
        progress = self._longitudinal_progress(self._ego(state), self._ego(rollout_state))
        final_speed = speeds[-1] if speeds else self._ego_speed(state)
        blocking_object_final = self._front_object_blocks_predicted_path(rollout_state, rollout_obj)

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
        )
        current_speed = self._ego_speed(state)
        first_acc = float(sequence[0][0]) if len(sequence) else self.mpc_config.strong_brake
        first_steer = float(sequence[0][1]) if len(sequence) else 0.0
        first_step_recovery_feasible = True
        first_step_recovery_reason = ""
        if current_speed <= self.mpc_config.stuck_speed_threshold:
            lateral_improves = (
                math.isfinite(final_lateral_clearance)
                and math.isfinite(initial_lateral_clearance)
                and float(final_lateral_clearance) - float(initial_lateral_clearance) > self.config.small_tolerance
            )
            is_lateral_recovery = terminal_recovery_reason in {
                "lateral_bypass_improving",
                "bypass_clearance_forming",
                "blocking_object_cleared",
            }
            has_significant_steer = abs(first_steer) >= self.mpc_config.nudge_steer * self.mpc_config.nudge_steer_ratio_threshold
            if first_acc >= self.mpc_config.guard_override_min_acc:
                first_step_recovery_feasible = True
                first_step_recovery_reason = "first_acc_sufficient"
            elif has_significant_steer:
                first_step_recovery_feasible = True
                first_step_recovery_reason = "first_steer_significant"
            elif lateral_improves:
                first_step_recovery_feasible = True
                first_step_recovery_reason = "lateral_clearance_improves"
            elif is_lateral_recovery:
                first_step_recovery_feasible = True
                first_step_recovery_reason = "terminal_lateral_recovery"
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
        )

        if not first_step_recovery_feasible:
            reason = "recovery_first_step_no_progress"
        elif not speed_feasible:
            reason = "rollout_speed_out_of_bounds"
        elif not progress_feasible:
            reason = "rollout_negative_progress"

        return {
            "sequence": sequence,
            "feasible": feasible,
            "cost": cost,
            "rss_margin_current": current_margin,
            "rss_margin_min_pred": min_margin,
            "rss_margin_final_pred": final_margin,
            "rss_lateral_clearance_min_pred": min_lateral_clearance,
            "rss_lateral_clearance_final_pred": final_lateral_clearance,
            "initial_lateral_clearance": initial_lateral_clearance,
            "final_speed": final_speed,
            "predicted_progress": progress,
            "recovery_used": recovery_used,
            "terminal_recoverable": terminal_recoverable,
            "terminal_recovery_reason": terminal_recovery_reason,
            "recovery_progress": progress,
            "recovery_margin_improvement": recovery_margin_improvement,
            "blocking_object_final": blocking_object_final,
            "reason": reason,
            "first_step_recovery_reason": first_step_recovery_reason,
        }

    def _evaluate_guarded_first_action(self, state: State, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Run the first MPC action through RSS-CBF before accepting a recovery."""
        u_mpc = self._clip_action(evaluation["sequence"][0])
        u_guarded, guard_info = self.rss_cbf_filter.filter_action(state, u_mpc)
        u_guarded = self._clip_action(u_guarded)
        guard_delta = self._action_norm(u_guarded, u_mpc)
        guard_mode = guard_info.get("mode", "")
        guard_fallback = guard_mode == "fallback_no_safe_candidate"
        guard_override_used, guard_override_reason = self._guard_override_for_certified_recovery(
            state=state,
            evaluation=evaluation,
            u_mpc=u_mpc,
            u_guarded=u_guarded,
            guard_mode=guard_mode,
        )
        u_effective = u_mpc if guard_override_used else u_guarded

        guarded_acc = float(u_effective[0])
        current_speed = self._ego_speed(state)
        guard_cost = self.mpc_config.guard_delta_cost_weight * guard_delta * guard_delta
        guard_cost += self.mpc_config.guard_brake_cost_weight * max(0.0, -float(u_guarded[0])) ** 2
        if current_speed <= self.mpc_config.stuck_speed_threshold and guarded_acc <= self.mpc_config.comfort_brake:
            guard_cost += self.mpc_config.guard_stall_brake_cost
        if guard_override_used:
            guard_cost *= 0.1

        guard_safe = guard_override_used or not guard_fallback
        if guard_safe and current_speed <= self.mpc_config.stuck_speed_threshold:
            progress_guarded = guarded_acc >= self.mpc_config.guard_override_min_acc
            lateral_recovery = abs(float(u_effective[1])) >= min(0.25, self.mpc_config.nudge_steer)
            lateral_clearance_improves = (
                math.isfinite(evaluation.get("rss_lateral_clearance_final_pred", -math.inf))
                and math.isfinite(evaluation.get("rss_lateral_clearance_min_pred", -math.inf))
                and float(evaluation["rss_lateral_clearance_final_pred"])
                - float(evaluation["rss_lateral_clearance_min_pred"]) > self.config.small_tolerance
            )
            is_lateral_terminal_recovery = evaluation.get("terminal_recovery_reason") in {
                "lateral_bypass_improving",
                "bypass_clearance_forming",
                "blocking_object_cleared",
            }
            guard_safe = (progress_guarded or lateral_recovery
                          or lateral_clearance_improves or is_lateral_terminal_recovery)

        return {
            "guard_safe": bool(guard_safe),
            "guard_cost": float(guard_cost),
            "u_mpc_first": u_mpc,
            "u_guarded_first": u_effective,
            "guard_info": guard_info,
            "cbf_guard_used": (not guard_override_used) and guard_delta > self.mpc_config.action_change_tolerance,
            "cbf_guard_delta": float(guard_delta),
            "cbf_guard_override_used": bool(guard_override_used),
            "cbf_guard_override_reason": guard_override_reason,
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
        if guard_mode not in {"rss_cbf_intervention", "rss_cbf_recovery", "fallback_no_safe_candidate"}:
            return False, ""
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
        certified_margin = (
            math.isfinite(min_margin)
            and math.isfinite(final_margin)
            and min_margin >= margin_buffer
            and final_margin >= margin_buffer
            and current_margin >= margin_buffer
        )
        if not certified_margin:
            return False, ""

        progress = float(evaluation.get("predicted_progress", 0.0))
        lateral_clearance = float(evaluation.get("rss_lateral_clearance_final_pred", -math.inf))
        lateral_improving = evaluation.get("terminal_recovery_reason") in {
            "blocking_object_cleared",
            "bypass_clearance_forming",
            "lateral_bypass_improving",
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
    ) -> float:
        cfg = self.mpc_config
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

        return (
            cfg.recovery_w_intervention * intervention_cost
            + cfg.w_smooth * smooth_cost
            + cfg.recovery_w_cbf_anchor * cbf_anchor_cost
            + cfg.recovery_w_speed * speed_cost
            - cfg.recovery_w_progress * float(progress)
            + cfg.w_rss_violation * rss_violation_penalty
            + cfg.w_brake * brake_penalty
            + cfg.w_stall * stall_penalty
            - cfg.w_lateral_clearance * lateral_clearance_reward
        )

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
    ) -> Tuple[bool, str]:
        if not rss_feasible:
            return False, "rss_not_feasible"

        margin_improvement = final_margin - current_margin
        lateral_improvement = final_lateral_clearance - initial_lateral_clearance
        if progress >= self.mpc_config.recovery_progress_threshold:
            return True, "progress_recovered"
        if margin_improvement >= self.mpc_config.recovery_margin_improvement_threshold:
            return True, "rss_margin_improved"
        if not blocking_object_final:
            return True, "blocking_object_cleared"
        if final_lateral_clearance >= self.mpc_config.recovery_lateral_clearance_threshold:
            return True, "bypass_clearance_forming"
        if lateral_improvement >= self.mpc_config.recovery_lateral_improvement_threshold:
            return True, "lateral_bypass_improving"
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
        return self._lateral_clearance_margin(state, obj) < 0.0

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
                "recovery_hold_used": False,
                "mpc_failure_reason": "",
                "mpc_no_rss_feasible_count": 0,
                "mpc_no_terminal_recoverable_count": 0,
                "mpc_guard_rejected_count": 0,
                "mpc_terminal_candidate_count": 0,
                "first_step_recovery_reason": "",
                "recovery_horizon_steps_used": self.mpc_config.horizon_steps,
                "recovery_candidate_family": "",
            }
        )
        if extra_info:
            info.update(extra_info)
        if self.deadlock_history:
            self.deadlock_history[-1]["mode"] = mode
        return self._with_action_debug(info, u_original, u_safe)

    def _action_norm(self, action: Sequence[float], reference: Sequence[float]) -> float:
        action = np.asarray(self._clip_action(action), dtype=np.float64)
        reference = np.asarray(self._clip_action(reference), dtype=np.float64)
        return float(np.linalg.norm(action - reference))
