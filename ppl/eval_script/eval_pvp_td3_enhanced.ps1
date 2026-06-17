# PPL Batch Evaluation Script for PowerShell
# Author: Auto-generated script for evaluating all PPL checkpoints

param(
    [Alias("rss_observe")]
    [switch]$RssObserve,

    [Alias("rss_shield")]
    [switch]$RssShield,

    [Alias("rss_shield_mode")]
    [ValidateSet("standard", "spring_damper")]
    [string]$RssShieldMode = "standard",

    [Alias("rss_debug_interval")]
    [int]$RssDebugInterval = 0,

    [Alias("rss_attack")]
    [ValidateSet("none", "front_range_overestimate", "front_object_removal", "lateral_gap_overestimate")]
    [string]$RssAttack = "none",

    [Alias("rss_attack_front_distance_delta")]
    [double]$RssAttackFrontDistanceDelta = 0.0,

    [Alias("rss_attack_lateral_gap_delta")]
    [double]$RssAttackLateralGapDelta = 0.0,

    [Alias("rss_uncertainty")]
    [ValidateSet("none", "gaussian")]
    [string]$RssUncertainty = "none",

    [Alias("rss_noise_level")]
    [ValidateSet("small", "medium", "large")]
    [string]$RssNoiseLevel = "medium"
)

# ========== Configuration ==========
$MODEL_DIR = "E:\CodeProject\CodexExp01\PPL-main\runs\PPL\PPL_de15a333\models"
$RESULT_DIR = "evaluation_results\PPL_de15a333_SpringDamper_GaussianNoise"
$START_STEP = 6000
$END_STEP = 8000
$STEP_INTERVAL = 200
$NUM_EP_IN_ONE_ENV = 1
$TOTAL_ENV_NUM = 50

# Fixed perception uncertainty settings for Gaussian RSS-noise experiments.
# medium is the recommended MetaDrive setting; large matches the paper's
# high-covariance highway setting and is mainly for stress tests.
if ($RssNoiseLevel -eq "small") {
    $RSS_NOISE_POSITION_X_SIGMA = 0.50
    $RSS_NOISE_POSITION_Y_SIGMA = 0.15
    $RSS_NOISE_SPEED_SIGMA = 0.50
    $RSS_NOISE_HEADING_SIGMA = 0.02
} elseif ($RssNoiseLevel -eq "large") {
    $RSS_NOISE_POSITION_X_SIGMA = 1.87
    $RSS_NOISE_POSITION_Y_SIGMA = 0.54
    $RSS_NOISE_SPEED_SIGMA = 2.64
    $RSS_NOISE_HEADING_SIGMA = 0.10
} else {
    $RSS_NOISE_POSITION_X_SIGMA = 1.00
    $RSS_NOISE_POSITION_Y_SIGMA = 0.30
    $RSS_NOISE_SPEED_SIGMA = 1.00
    $RSS_NOISE_HEADING_SIGMA = 0.05
}
$RSS_NOISE_SEED = 7

# ========== Start Evaluation ==========
Write-Host "===== PPL Batch Evaluation Script =====" -ForegroundColor Cyan
Write-Host "Model Directory: $MODEL_DIR" -ForegroundColor White
Write-Host "Result Directory: $RESULT_DIR" -ForegroundColor White
Write-Host "Step Range: $START_STEP -> $END_STEP (interval $STEP_INTERVAL)" -ForegroundColor White
Write-Host "Episodes per Env: $NUM_EP_IN_ONE_ENV, Total Envs: $TOTAL_ENV_NUM" -ForegroundColor White
Write-Host "RSS Observe: $RssObserve, RSS Shield: $RssShield, RSS Shield Mode: $RssShieldMode, RSS Debug Interval: $RssDebugInterval" -ForegroundColor White
Write-Host "RSS Attack: $RssAttack, Front Distance Delta: $RssAttackFrontDistanceDelta m, Lateral Gap Delta: $RssAttackLateralGapDelta m" -ForegroundColor White
Write-Host "RSS Uncertainty: $RssUncertainty, Noise Level: $RssNoiseLevel, SigmaX: $RSS_NOISE_POSITION_X_SIGMA m, SigmaY: $RSS_NOISE_POSITION_Y_SIGMA m, SigmaV: $RSS_NOISE_SPEED_SIGMA m/s, SigmaTheta: $RSS_NOISE_HEADING_SIGMA rad, Noise Seed: $RSS_NOISE_SEED" -ForegroundColor White
Write-Host ""

# Create result directory
New-Item -ItemType Directory -Force -Path $RESULT_DIR | Out-Null

# Merged CSV files
$MERGED_CSV = "$RESULT_DIR\all_checkpoints_evaluation.csv"  # 详细数据
$SUMMARY_CSV = "$RESULT_DIR\all_checkpoints_summary.csv"    # 汇总数据
$is_first_file = $true

$total_ckpts = [Math]::Floor(($END_STEP - $START_STEP) / $STEP_INTERVAL) + 1
$current_count = 0

for ($step = $START_STEP; $step -le $END_STEP; $step += $STEP_INTERVAL) {
    $current_count++
    $model_path = "$MODEL_DIR\rl_model_${step}_steps.zip"

    if (Test-Path $model_path) {
        try {
            Write-Host "[$current_count/$total_ckpts] Evaluating checkpoint $step ..." -ForegroundColor Green
            $eval_args = @(
                "-m", "ppl.eval_script.metadrive.eval_ppl_metadrive",
                "--path", "$MODEL_DIR",
                "--ckpt_index", "$step",
                "--ret_save_folder", "$RESULT_DIR",
                "--num_ep_in_one_env", "$NUM_EP_IN_ONE_ENV",
                "--total_env_num", "$TOTAL_ENV_NUM"
            )
            if ($RssObserve) {
                $eval_args += "--rss_observe"
            }
            if ($RssShield) {
                $eval_args += "--rss_shield"
                $eval_args += @("--rss_shield_mode", "$RssShieldMode")
            }
            if ($RssDebugInterval -gt 0) {
                $eval_args += @("--rss_debug_interval", "$RssDebugInterval")
            }
            if ($RssAttack -ne "none") {
                $eval_args += @("--rss_attack", "$RssAttack")
                $eval_args += @("--rss_attack_front_distance_delta", "$RssAttackFrontDistanceDelta")
                $eval_args += @("--rss_attack_lateral_gap_delta", "$RssAttackLateralGapDelta")
            }
            if ($RssUncertainty -ne "none") {
                $eval_args += @("--rss_uncertainty", "$RssUncertainty")
                $eval_args += @("--rss_noise_level", "$RssNoiseLevel")
                $eval_args += @("--rss_noise_position_x_sigma", "$RSS_NOISE_POSITION_X_SIGMA")
                $eval_args += @("--rss_noise_position_y_sigma", "$RSS_NOISE_POSITION_Y_SIGMA")
                $eval_args += @("--rss_noise_speed_sigma", "$RSS_NOISE_SPEED_SIGMA")
                $eval_args += @("--rss_noise_heading_sigma", "$RSS_NOISE_HEADING_SIGMA")
                $eval_args += @("--rss_noise_seed", "$RSS_NOISE_SEED")
            }
            python @eval_args
            if ($LASTEXITCODE -ne 0) {
                throw "Python evaluation failed for checkpoint $step with exit code $LASTEXITCODE"
            }
            
            # Merge CSV files
            $ckpt_csv = "$RESULT_DIR\checkpoint_${step}.csv"
            if (Test-Path $ckpt_csv) {
                # 1. 合并详细数据
                if ($is_first_file) {
                    Copy-Item $ckpt_csv $MERGED_CSV
                    $is_first_file = $false
                    Write-Host "Created merged file: $MERGED_CSV" -ForegroundColor Cyan
                } else {
                    Get-Content $ckpt_csv | Select-Object -Skip 1 | Add-Content $MERGED_CSV
                    Write-Host "Appended checkpoint $step data to merged file" -ForegroundColor Cyan
                }
                
                # 2. 计算汇总统计数据
                $ckpt_csv_py = $ckpt_csv -replace '\\', '/'
                $summary_csv_py = $SUMMARY_CSV -replace '\\', '/'
                python -c @"
import pandas as pd
import os
try:
    df = pd.read_csv('$ckpt_csv_py')
    summary = {
        'ckpt_index': $step,
        'rss_uncertainty': '$RssUncertainty',
        'rss_noise_level': '$RssNoiseLevel',
        'rss_noise_sigma_x': $RSS_NOISE_POSITION_X_SIGMA,
        'rss_noise_sigma_y': $RSS_NOISE_POSITION_Y_SIGMA,
        'rss_noise_sigma_v': $RSS_NOISE_SPEED_SIGMA,
        'rss_noise_sigma_theta': $RSS_NOISE_HEADING_SIGMA,
        'success_rate': df['success'].mean(),
        'crash_rate': df['crash'].mean() if 'crash' in df.columns else 0.0,
        'crash_count': int(df['crash'].sum()) if 'crash' in df.columns else 0,
        'vehicle_crash_rate': df['crash_vehicle'].mean() if 'crash_vehicle' in df.columns else 0.0,
        'vehicle_crash_count': int(df['crash_vehicle'].sum()) if 'crash_vehicle' in df.columns else 0,
        'out_of_road_rate': df['out_of_road'].mean() if 'out_of_road' in df.columns else 0.0,
        'out_of_road_count': int(df['out_of_road'].sum()) if 'out_of_road' in df.columns else 0,
        'avg_reward': df['episode_reward'].mean(), 
        'avg_episode_length': df['episode_length'].mean(),
        'avg_velocity': df['velocity_step_mean'].mean(),
                'avg_cost': df['episode_cost'].mean(),
                'avg_route_completion': df['route_completion'].mean() if 'route_completion' in df.columns else 0.0,
                'avg_rss_attack_steps': df['rss_attack_steps'].mean() if 'rss_attack_steps' in df.columns else 0.0,
                'avg_rss_front_attack_steps': df['rss_front_attack_steps'].mean() if 'rss_front_attack_steps' in df.columns else 0.0,
                'avg_rss_lateral_attack_steps': df['rss_lateral_attack_steps'].mean() if 'rss_lateral_attack_steps' in df.columns else 0.0,
                'avg_rss_uncertainty_steps': df['rss_uncertainty_steps'].mean() if 'rss_uncertainty_steps' in df.columns else 0.0,
                'avg_rss_front_uncertainty_steps': df['rss_front_uncertainty_steps'].mean() if 'rss_front_uncertainty_steps' in df.columns else 0.0,
                'avg_rss_lateral_uncertainty_steps': df['rss_lateral_uncertainty_steps'].mean() if 'rss_lateral_uncertainty_steps' in df.columns else 0.0,
                'avg_rss_attack_delta': df['rss_attack_delta_mean'].mean() if 'rss_attack_delta_mean' in df.columns else 0.0,
                'avg_rss_lateral_attack_delta': df['rss_lateral_attack_delta_mean'].mean() if 'rss_lateral_attack_delta_mean' in df.columns else 0.0,
                'avg_rss_front_distance_noise': df['rss_front_distance_noise_mean'].mean() if 'rss_front_distance_noise_mean' in df.columns else 0.0,
                'avg_rss_front_speed_noise': df['rss_front_speed_noise_mean'].mean() if 'rss_front_speed_noise_mean' in df.columns else 0.0,
                'avg_rss_lateral_gap_noise': df['rss_lateral_gap_noise_mean'].mean() if 'rss_lateral_gap_noise_mean' in df.columns else 0.0,
                'avg_rss_front_distance_raw': df['rss_front_distance_raw_mean'].mean() if 'rss_front_distance_raw_mean' in df.columns else 0.0,
                'avg_rss_front_distance_attacked': df['rss_front_distance_attacked_mean'].mean() if 'rss_front_distance_attacked_mean' in df.columns else 0.0,
                'avg_rss_front_distance_noisy': df['rss_front_distance_noisy_mean'].mean() if 'rss_front_distance_noisy_mean' in df.columns else 0.0,
                'avg_rss_front_distance_used': df['rss_front_distance_used_mean'].mean() if 'rss_front_distance_used_mean' in df.columns else 0.0,
                'avg_rss_lateral_gap_raw': df['rss_lateral_gap_raw_mean'].mean() if 'rss_lateral_gap_raw_mean' in df.columns else 0.0,
                'avg_rss_lateral_gap_attacked': df['rss_lateral_gap_attacked_mean'].mean() if 'rss_lateral_gap_attacked_mean' in df.columns else 0.0,
                'avg_rss_lateral_gap_noisy': df['rss_lateral_gap_noisy_mean'].mean() if 'rss_lateral_gap_noisy_mean' in df.columns else 0.0,
                'avg_rss_lateral_gap_used': df['rss_lateral_gap_used_mean'].mean() if 'rss_lateral_gap_used_mean' in df.columns else 0.0,
                'num_episodes': len(df)
    }
    header = not os.path.exists('$summary_csv_py')
    pd.DataFrame([summary]).to_csv('$summary_csv_py', mode='a', header=header, index=False)
    print('✅ Computed summary statistics for checkpoint ${step}')
except Exception as e:
    print('❌ Error processing checkpoint ${step}:', str(e))
"@
            }
            
            Write-Host "Checkpoint $step evaluation completed!" -ForegroundColor Green
            Write-Host ""
        } catch {
            Write-Host "❌ FATAL ERROR processing checkpoint ${step}:" -ForegroundColor Red
            Write-Host $_.Exception.Message -ForegroundColor Red
        }
    } else {
        Write-Host "[$current_count/$total_ckpts] Skipped: Model not found $model_path" -ForegroundColor Yellow
        Write-Host ""
    }
}

Write-Host "===== All checkpoints evaluation completed! =====" -ForegroundColor Cyan

# 汇总数据统计
if (Test-Path $SUMMARY_CSV) {
    $summary_count = (Get-Content $SUMMARY_CSV | Measure-Object -Line).Lines - 1
    Write-Host "Summary data saved to: $SUMMARY_CSV" -ForegroundColor Green
    Write-Host "Summary file contains $summary_count rows (one per checkpoint)" -ForegroundColor White
}

# 详细数据统计
if (Test-Path $MERGED_CSV) {
    $line_count = (Get-Content $MERGED_CSV | Measure-Object -Line).Lines
    Write-Host "Detailed data saved to: $MERGED_CSV" -ForegroundColor Green
    Write-Host "Detailed file contains $line_count lines (including header)" -ForegroundColor White
}

# 清理临时的单独检查点文件
Write-Host "`nCleaning up temporary checkpoint files..." -ForegroundColor Cyan
$temp_files = Get-ChildItem -Path $RESULT_DIR -Filter "checkpoint_*.csv"
$deleted_count = 0
foreach ($file in $temp_files) {
    Remove-Item $file.FullName -Force
    $deleted_count++
}
Write-Host "Deleted $deleted_count temporary files" -ForegroundColor Green

Write-Host "`n===== Generated Files =====" -ForegroundColor Cyan
Write-Host "1. Detailed data (for wandb with variance): $MERGED_CSV" -ForegroundColor Yellow
Write-Host "2. Summary data (averaged per checkpoint): $SUMMARY_CSV" -ForegroundColor Yellow
