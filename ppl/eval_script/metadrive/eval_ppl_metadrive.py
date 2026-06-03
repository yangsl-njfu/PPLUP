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
    return summary


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
                info.update(safety_info)
                recovery_step_infos.append(safety_info)
            step_count += 1

            if info:
                ep_velocities.append(info.get("velocity", 0))

            if use_render:
                env.render()

            if progress_interval and step_count % progress_interval == 0:
                if safety_info:
                    print(
                        "[EvalProgress] env {} ep_in_env {} step {} recovery={} certified={} "
                        "filter_ms={:.1f} hard_safe={}/{}".format(
                            env_index,
                            num_ep_in + 1,
                            step_count,
                            safety_info.get("recovery_mode", ""),
                            safety_info.get("recovery_certified", False),
                            float(safety_info.get("filter_time_ms", 0.0)),
                            int(safety_info.get("num_hard_safe_candidates", 0)),
                            int(safety_info.get("num_candidates", 0)),
                        )
                    )
                else:
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
                res.update(summarize_recovery_episode(recovery_step_infos))
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
        help="Enable RSS-risk-guided predictive runtime recovery.",
    )
    parser.add_argument(
        "--recovery_horizon",
        type=float,
        default=PredictiveRecoveryConfig.horizon,
        help="Predictive recovery horizon in seconds.",
    )
    parser.add_argument(
        "--recovery_dt",
        type=float,
        default=PredictiveRecoveryConfig.dt,
        help="Predictive recovery rollout time step in seconds.",
    )
    parser.add_argument(
        "--recovery_num_lateral_targets",
        type=int,
        default=PredictiveRecoveryConfig.num_lateral_targets,
        help="Number of lateral target offsets used to generate candidates.",
    )
    parser.add_argument(
        "--recovery_num_speed_targets",
        type=int,
        default=PredictiveRecoveryConfig.num_speed_targets,
        help="Number of speed targets used to generate candidates.",
    )
    parser.add_argument(
        "--recovery_max_objects",
        type=int,
        default=PredictiveRecoveryConfig.max_objects,
        help="Maximum number of nearby scene objects considered per step.",
    )
    parser.add_argument(
        "--recovery_max_candidates",
        type=int,
        default=PredictiveRecoveryConfig.max_candidates,
        help="Maximum number of trajectory candidates evaluated per step.",
    )
    parser.add_argument(
        "--recovery_max_rollout_steps",
        type=int,
        default=PredictiveRecoveryConfig.max_rollout_steps,
        help="Maximum sampled rollout points per candidate.",
    )
    parser.add_argument(
        "--recovery_object_scan_limit",
        type=int,
        default=PredictiveRecoveryConfig.object_scan_limit,
        help="Maximum raw environment objects scanned before nearest-object filtering.",
    )
    parser.add_argument(
        "--recovery_debug",
        action="store_true",
        help="Enable verbose recovery fallback diagnostics.",
    )
    parser.add_argument(
        "--recovery_log_csv",
        nargs="?",
        const="evaluate_results/ppl/recovery_step_log.csv",
        default="",
        help="Optional per-step recovery diagnostic CSV path.",
    )

    args = parser.parse_args()

    deterministic = not args.stochastic
    recovery_config = PredictiveRecoveryConfig(
        horizon=args.recovery_horizon,
        dt=args.recovery_dt,
        num_lateral_targets=args.recovery_num_lateral_targets,
        num_speed_targets=args.recovery_num_speed_targets,
        max_objects=args.recovery_max_objects,
        max_candidates=args.recovery_max_candidates,
        max_rollout_steps=args.recovery_max_rollout_steps,
        object_scan_limit=args.recovery_object_scan_limit,
        debug=args.recovery_debug,
        log_csv_path=args.recovery_log_csv or "",
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
