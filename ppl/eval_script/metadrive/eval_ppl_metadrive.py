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


def summarize_recovery_episode(step_infos):
    """Keep exact last-step diagnostics and add episode means for numeric fields."""
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

    # 统计接管相关
    filter_intervened_count = sum(1 for item in step_infos if item.get("filter_intervened", False))
    summary["intervention_count"] = filter_intervened_count
    summary["intervention_rate"] = filter_intervened_count / max(1, len(step_infos))

    # 接管原因统计
    intervention_reasons = [item.get("intervention_reason", "none") for item in step_infos if item.get("filter_intervened", False)]
    summary["intervention_reasons"] = ";".join(intervention_reasons) if intervention_reasons else ""

    # 平均分数差值
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

    # 拒绝原因统计
    all_reject_reasons = []
    for item in step_infos:
        reasons_str = item.get("early_reject_reasons", "")
        if reasons_str:
            all_reject_reasons.append(reasons_str)
    summary["all_early_reject_reasons"] = ";".join(all_reject_reasons) if all_reject_reasons else ""

    # Raw action 对照
    raw_safe_count = sum(1 for item in step_infos if item.get("raw_hard_safe", True))
    summary["raw_safe_rate"] = raw_safe_count / max(1, len(step_infos))

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


def evaluate_ppl_once(
    ckpt_path,
    ckpt_index,
    folder_name,
    use_render=False,
    num_ep_in_one_env=5,
    total_env_num=50,
    deterministic=True,
    enable_predictive_recovery=False,
    recovery_config=None,
    progress_interval=0,
    recovery_episode_log_csv="",
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

    predictive_filter = None
    if enable_predictive_recovery:
        predictive_filter = PredictiveRecoveryFilter(recovery_config or PredictiveRecoveryConfig())
        print("[PredictiveRecovery] Enabled.")
        if not recovery_episode_log_csv:
            recovery_episode_log_csv = osp.join(folder_name, "{}_recovery_episode.csv".format(ckpt_name))

    saved_results = []
    ep_velocities = []
    recovery_step_infos = []

    try:
        start = time.time()
        last_time = time.time()
        ep_count = 0
        step_count = 0
        ep_times = []

        env_index = 0
        num_ep_in = 0
        o = reset_eval_env(env, EVAL_ENV_START + env_index)

        while True:
            action = policy_function(o, deterministic=deterministic)[0]
            raw_action = action
            if predictive_filter is not None:
                safe_action, safety_info = predictive_filter.filter(env, o, raw_action)
            else:
                safe_action, safety_info = raw_action, {}

            o, r, d, info = env.step(safe_action)
            if safety_info:
                recovery_step_infos.append(safety_info)
            step_count += 1

            if info:
                ep_velocities.append(info.get("velocity", 0))

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
                recovery_summary = summarize_recovery_episode(recovery_step_infos)
                ep_velocities = []
                recovery_step_infos = []

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
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=0,
        help="Print an evaluation progress line every N environment steps. 0 disables step progress logs.",
    )
    parser.add_argument(
        "--enable_predictive_recovery",
        action="store_true",
        help="Enable runtime assurance filter for predictive collision/departure prevention.",
    )
    # 运行时保障可选参数
    parser.add_argument(
        "--recovery_intervention_score_margin",
        type=float,
        default=PredictiveRecoveryConfig.intervention_score_margin,
        help="Minimum score improvement required for filter to intervene (default: 1.5).",
    )
    parser.add_argument(
        "--recovery_max_steer_delta_from_raw",
        type=float,
        default=PredictiveRecoveryConfig.max_steer_delta_from_raw,
        help="Maximum steering delta from raw action (default: 0.25).",
    )
    parser.add_argument(
        "--recovery_max_acc_delta_from_raw",
        type=float,
        default=PredictiveRecoveryConfig.max_acc_delta_from_raw,
        help="Maximum acceleration delta from raw action (default: 0.35).",
    )
    parser.add_argument(
        "--recovery_log_csv",
        nargs="?",
        const="evaluate_results/ppl/recovery_step_log.csv",
        default="",
        help="Optional per-step recovery diagnostic CSV path.",
    )
    parser.add_argument(
        "--recovery_episode_log_csv",
        type=str,
        default="",
        help="Optional episode-level recovery diagnostic CSV path.",
    )
    # --- Debug / Developer options ---
    parser.add_argument(
        "--recovery_debug_shadow",
        action="store_true",
        help="[Dev] Run filter but execute raw_action, record what filter would do.",
    )
    parser.add_argument(
        "--recovery_debug",
        action="store_true",
        help="[Dev] Enable verbose recovery fallback diagnostics.",
    )

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
            enable_predictive_recovery=args.enable_predictive_recovery,
            recovery_config=recovery_config,
            progress_interval=args.progress_interval,
            recovery_episode_log_csv=args.recovery_episode_log_csv,
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
            enable_predictive_recovery=args.enable_predictive_recovery,
            recovery_config=recovery_config,
            progress_interval=args.progress_interval,
            recovery_episode_log_csv=args.recovery_episode_log_csv,
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
                enable_predictive_recovery=args.enable_predictive_recovery,
                recovery_config=recovery_config,
                progress_interval=args.progress_interval,
                recovery_episode_log_csv=args.recovery_episode_log_csv,
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
