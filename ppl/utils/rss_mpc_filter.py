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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ppl.utils.rss_cbf_filter import RSSCBFConfig, RSSCBFFilter
from ppl.utils.static_rss_filter import Action, State


@dataclass
class RSSMPCConfig(RSSCBFConfig):
    """Stable defaults for RSS-MPC runtime assurance evaluation."""

    horizon_steps: int = 10
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
        "rss_mpc_intervention",
        "rss_mpc_recovery",
        "rss_mpc_cbf_guard",
        "rss_mpc_fallback_to_cbf",
        "fallback_no_safe_candidate",
    }

    def __init__(self, config: Optional[RSSMPCConfig] = None):
        self.mpc_config = config or RSSMPCConfig()
        super().__init__(self.mpc_config)
        self.rss_cbf_filter = RSSCBFFilter(self.mpc_config)
        self.rng = np.random.default_rng(self.mpc_config.random_seed)

    def filter_action(self, state: State, u_nom: Sequence[float]) -> Tuple[Action, Dict[str, Any]]:
        """Return the first action of the best RSS-feasible MPC sequence."""
        u_original = self._clip_action(u_nom)
        cbf_safe, cbf_info = self.rss_cbf_filter.filter_action(state, u_original)
        cbf_safe = self._clip_action(cbf_safe)
        front = self._select_front_rss_object(state)

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
                mpc_success=True,
                mpc_num_candidates=1,
                mpc_num_feasible=1,
                mpc_best_cost=0.0,
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
            )

        object_kind, obj, d_front, d_rss, current_margin, dynamic_detected, static_detected = front
        candidates = self._generate_candidate_sequences(u_original, cbf_safe)
        best: Optional[Dict[str, Any]] = None
        feasible_count = 0

        for sequence in candidates:
            evaluation = self._evaluate_sequence(
                state=state,
                obj=obj,
                object_kind=object_kind,
                current_margin=current_margin,
                sequence=sequence,
                u_nom=u_original,
                u_cbf=cbf_safe,
            )
            if not evaluation["feasible"]:
                continue
            feasible_count += 1
            if best is None or evaluation["cost"] < best["cost"]:
                best = evaluation

        if best is not None:
            u_mpc = self._clip_action(best["sequence"][0])
            u_safe, guard_info = self.rss_cbf_filter.filter_action(state, u_mpc)
            u_safe = self._clip_action(u_safe)
            cbf_guard_delta = self._action_norm(u_safe, u_mpc)
            cbf_guard_used = cbf_guard_delta > self.mpc_config.action_change_tolerance
            action_delta = self._action_norm(u_safe, u_original)
            if current_margin < -self.config.small_tolerance and best.get("recovery_used", False):
                mode = "rss_mpc_recovery"
            elif cbf_guard_used:
                mode = "rss_mpc_cbf_guard"
            elif action_delta <= self.mpc_config.action_change_tolerance:
                mode = "normal"
            else:
                mode = "rss_mpc_intervention"
            reason = best.get("reason", "mpc_feasible_sequence_selected")
            if cbf_guard_used:
                reason = "cbf_guard_projected_mpc_action"

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
                mpc_num_candidates=len(candidates),
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
            )

        if cbf_info.get("mode") == "fallback_no_safe_candidate":
            u_safe = u_original
            mode = "fallback_no_safe_candidate"
            reason = "mpc_no_feasible_sequence_cbf_no_safe_candidate"
        else:
            u_safe = self._clip_action(cbf_safe)
            mode = "rss_mpc_fallback_to_cbf"
            reason = "mpc_no_feasible_sequence"

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
            mpc_num_candidates=len(candidates),
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
            cbf_guard_used=mode == "rss_mpc_fallback_to_cbf",
            cbf_guard_delta=self._action_norm(u_safe, u_original),
        )
        info["cbf_info"] = cbf_info
        return u_safe, info

    def _generate_candidate_sequences(self, u_nom: Action, u_cbf: Optional[Action] = None) -> List[np.ndarray]:
        horizon = max(1, int(self.mpc_config.horizon_steps))
        max_candidates = max(1, int(self.mpc_config.num_samples))
        sequences: List[np.ndarray] = []

        def add_sequence(acc_values: Sequence[float], steer_values: Sequence[float]) -> None:
            if len(sequences) >= max_candidates:
                return
            sequence = np.asarray(
                [self._clip_action([acc, steer]) for acc, steer in zip(acc_values, steer_values)],
                dtype=np.float64,
            )
            if sequence.shape == (horizon, 2):
                sequences.append(sequence)

        acc_nom, steer_nom = u_nom

        add_sequence(
            np.full(horizon, acc_nom, dtype=np.float64),
            np.full(horizon, steer_nom, dtype=np.float64),
        )

        if u_cbf is not None:
            acc_cbf, steer_cbf = self._clip_action(u_cbf)
            add_sequence(
                np.full(horizon, acc_cbf, dtype=np.float64),
                np.full(horizon, steer_cbf, dtype=np.float64),
            )
            add_sequence(
                np.linspace(acc_nom, acc_cbf, horizon),
                np.linspace(steer_nom, steer_cbf, horizon),
            )
            for steer_bias in (-0.25, 0.25):
                add_sequence(
                    np.full(horizon, acc_cbf, dtype=np.float64),
                    np.full(horizon, steer_cbf + steer_bias, dtype=np.float64),
                )

        for target_acc in (self.mpc_config.comfort_brake, self.mpc_config.strong_brake):
            add_sequence(
                np.linspace(acc_nom, target_acc, horizon),
                np.full(horizon, steer_nom, dtype=np.float64),
            )

        for steer_bias in (-0.25, -0.1, 0.1, 0.25):
            for acc_value in (acc_nom, 0.0, self.mpc_config.comfort_brake):
                add_sequence(
                    np.full(horizon, acc_value, dtype=np.float64),
                    np.full(horizon, steer_nom + steer_bias, dtype=np.float64),
                )

        for steer_target in (-1.0, -0.8, -0.5, 0.5, 0.8, 1.0):
            for acc_value in (self.mpc_config.max_acc, 1.0, 0.0):
                add_sequence(
                    np.full(horizon, acc_value, dtype=np.float64),
                    np.full(horizon, steer_target, dtype=np.float64),
                )

        for steer_target in (-1.0, 1.0):
            add_sequence(
                np.linspace(max(acc_nom, 0.0), self.mpc_config.max_acc, horizon),
                np.linspace(steer_nom, steer_target, horizon),
            )

        while len(sequences) < max_candidates:
            sequences.append(self._sample_random_sequence(u_nom, horizon))

        return sequences[:max_candidates]

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

        rss_feasible, recovery_used, reason = self._rss_sequence_feasibility(
            current_margin=current_margin,
            min_margin=min_margin,
            final_margin=final_margin,
        )
        speed_feasible = all(
            -self.config.small_tolerance <= speed <= self.mpc_config.v_max + self.config.small_tolerance
            for speed in speeds
        )
        progress_feasible = progress >= -self.config.small_tolerance
        feasible = bool(rss_feasible and speed_feasible and progress_feasible)

        cost = self._sequence_cost(
            sequence=sequence,
            u_nom=u_nom,
            u_cbf=u_cbf,
            current_speed=self._ego_speed(state),
            speeds=speeds,
            margins=margins,
            lateral_clearance_margins=lateral_clearance_margins,
            progress=progress,
        )

        if not speed_feasible:
            reason = "rollout_speed_out_of_bounds"
        elif not progress_feasible:
            reason = "rollout_negative_progress"

        return {
            "sequence": sequence,
            "feasible": feasible,
            "cost": cost,
            "rss_margin_min_pred": min_margin,
            "rss_margin_final_pred": final_margin,
            "rss_lateral_clearance_min_pred": min_lateral_clearance,
            "rss_lateral_clearance_final_pred": final_lateral_clearance,
            "predicted_progress": progress,
            "recovery_used": recovery_used,
            "reason": reason,
        }

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
            cfg.w_intervention * intervention_cost
            + cfg.w_smooth * smooth_cost
            + cfg.w_cbf_anchor * cbf_anchor_cost
            + cfg.w_speed * speed_cost
            - cfg.w_progress * float(progress)
            + cfg.w_rss_violation * rss_violation_penalty
            + cfg.w_brake * brake_penalty
            + cfg.w_stall * stall_penalty
            - cfg.w_lateral_clearance * lateral_clearance_reward
        )

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
        return self._with_action_debug(info, u_original, u_safe)

    def _action_norm(self, action: Sequence[float], reference: Sequence[float]) -> float:
        action = np.asarray(self._clip_action(action), dtype=np.float64)
        reference = np.asarray(self._clip_action(reference), dtype=np.float64)
        return float(np.linalg.norm(action - reference))
