# PPL Batch Evaluation Script for PowerShell
# Author: Auto-generated script for evaluating all PPL checkpoints

# ========== Configuration ==========
$MODEL_DIR = "E:\CodeProject\CodexExp01\PPL-main\runs\PPL\PPL_0ee03603\models"
$RESULT_DIR = "evaluation_results\PPL_0ee03603"
$START_STEP = 6000
$END_STEP = 10000
$STEP_INTERVAL = 200
$NUM_EP_IN_ONE_ENV = 1
$TOTAL_ENV_NUM = 50

# ========== Start Evaluation ==========
Write-Host "===== PPL Batch Evaluation Script =====" -ForegroundColor Cyan
Write-Host "Model Directory: $MODEL_DIR" -ForegroundColor White
Write-Host "Result Directory: $RESULT_DIR" -ForegroundColor White
Write-Host "Step Range: $START_STEP -> $END_STEP (interval $STEP_INTERVAL)" -ForegroundColor White
Write-Host "Episodes per Env: $NUM_EP_IN_ONE_ENV, Total Envs: $TOTAL_ENV_NUM" -ForegroundColor White
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
            python -m ppl.eval_script.metadrive.eval_ppl_metadrive --path "$MODEL_DIR" --ckpt_index $step --ret_save_folder "$RESULT_DIR" --num_ep_in_one_env $NUM_EP_IN_ONE_ENV --total_env_num $TOTAL_ENV_NUM
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
        'success_rate': df['success'].mean(),
        'crash_rate': df['crash'].mean() if 'crash' in df.columns else 0.0,
        'crash_count': int(df['crash'].sum()) if 'crash' in df.columns else 0,
        'out_of_road_rate': df['out_of_road'].mean() if 'out_of_road' in df.columns else 0.0,
        'out_of_road_count': int(df['out_of_road'].sum()) if 'out_of_road' in df.columns else 0,
        'avg_reward': df['episode_reward'].mean(), 
        'avg_episode_length': df['episode_length'].mean(),
        'avg_velocity': df['velocity_step_mean'].mean(),
                'avg_cost': df['episode_cost'].mean(),
                'avg_route_completion': df['route_completion'].mean() if 'route_completion' in df.columns else 0.0,
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
