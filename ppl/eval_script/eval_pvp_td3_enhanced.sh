#!/bin/bash

# ========== PPL 批量评估脚本 (Bash) ==========
# 用于批量评估所有 PPL 训练检查点

# ========== 配置参数 ==========
MODEL_DIR="runs/PPL/PPL_HUMAN_SEED_0_2026-04-12_xx-xx-xx/models"
RESULT_DIR="evaluation_results/PPL_HUMAN_SEED_0"
START_STEP=200
END_STEP=10000
STEP_INTERVAL=200
NUM_EP_IN_ONE_ENV=1
TOTAL_ENV_NUM=50

# ========== 开始评估 ==========
echo "===== PPL Batch Evaluation Script ====="
echo "Model Directory: $MODEL_DIR"
echo "Result Directory: $RESULT_DIR"
echo "Step Range: $START_STEP -> $END_STEP (interval $STEP_INTERVAL)"
echo "Episodes per Env: $NUM_EP_IN_ONE_ENV, Total Envs: $TOTAL_ENV_NUM"
echo ""

# 创建结果目录
mkdir -p "$RESULT_DIR"

# Merged CSV files
MERGED_CSV="$RESULT_DIR/all_checkpoints_evaluation.csv"  # 详细数据
SUMMARY_CSV="$RESULT_DIR/all_checkpoints_summary.csv"    # 汇总数据
is_first_file=true

total_ckpts=$(( ($END_STEP - $START_STEP) / $STEP_INTERVAL + 1 ))
current_count=0

# 循环评估每个检查点
for step in $(seq $START_STEP $STEP_INTERVAL $END_STEP); do
    current_count=$((current_count + 1))
    model_path="$MODEL_DIR/rl_model_${step}_steps.zip"
    
    if [ -f "$model_path" ]; then
        echo "[$current_count/$total_ckpts] Evaluating checkpoint $step ..."
        
        # Run evaluation using eval_ppl_metadrive.py
        python -m ppl.eval_script.metadrive.eval_ppl_metadrive \
            --path "$MODEL_DIR" \
            --ckpt_index $step \
            --ret_save_folder "$RESULT_DIR" \
            --num_ep_in_one_env $NUM_EP_IN_ONE_ENV \
            --total_env_num $TOTAL_ENV_NUM
        
        # Merge CSV files
        ckpt_csv="$RESULT_DIR/checkpoint_${step}.csv"
        if [ -f "$ckpt_csv" ]; then
            # 1. 合并详细数据
            if [ "$is_first_file" = true ]; then
                cp "$ckpt_csv" "$MERGED_CSV"
                is_first_file=false
                echo "Created merged file: $MERGED_CSV"
            else
                tail -n +2 "$ckpt_csv" >> "$MERGED_CSV"
                echo "Appended checkpoint $step data to merged file"
            fi
            
            # 2. 计算汇总统计数据（使用Python）
            python -c "
import pandas as pd
df = pd.read_csv('$ckpt_csv')
summary = {
    'ckpt_index': $step,
    'success_rate': df['success'].mean() if 'success' in df.columns else 0.0,
    'crash_rate': df['crash'].mean() if 'crash' in df.columns else 0.0,
    'crash_count': int(df['crash'].sum()) if 'crash' in df.columns else 0,
    'out_of_road_rate': df['out_of_road'].mean() if 'out_of_road' in df.columns else 0.0,
    'out_of_road_count': int(df['out_of_road'].sum()) if 'out_of_road' in df.columns else 0,
    'avg_reward': df['episode_reward'].mean() if 'episode_reward' in df.columns else 0.0,
    'avg_episode_length': df['episode_length'].mean() if 'episode_length' in df.columns else 0.0,
    'avg_velocity': df['velocity_step_mean'].mean() if 'velocity_step_mean' in df.columns else 0.0,
    'avg_cost': df['episode_cost'].mean() if 'episode_cost' in df.columns else 0.0,
    'num_episodes': len(df)
}
# 追加到汇总文件
import os
if not os.path.exists('$SUMMARY_CSV'):
    pd.DataFrame([summary]).to_csv('$SUMMARY_CSV', index=False)
else:
    pd.DataFrame([summary]).to_csv('$SUMMARY_CSV', mode='a', header=False, index=False)
print('Computed summary statistics for checkpoint $step')
"
        fi
        
        echo "Checkpoint $step evaluation completed!"
        echo ""
    else
        echo "[$current_count/$total_ckpts] Skipped: Model not found $model_path"
        echo ""
    fi
done

echo "===== All checkpoints evaluation completed! ====="

# 详细数据统计
echo "Detailed data saved to: $MERGED_CSV"
if [ -f "$MERGED_CSV" ]; then
    line_count=$(wc -l < "$MERGED_CSV")
    echo "Detailed file contains $line_count lines (including header)"
fi

# 汇总数据统计
if [ -f "$SUMMARY_CSV" ]; then
    summary_count=$(wc -l < "$SUMMARY_CSV")
    summary_count=$((summary_count - 1))  # 减去表头
    echo "Summary data saved to: $SUMMARY_CSV"
    echo "Summary file contains $summary_count rows (one per checkpoint)"
fi

# 清理临时的单独检查点文件（checkpoint_*.csv）
echo ""
echo "Cleaning up temporary checkpoint files..."
deleted_count=$(find "$RESULT_DIR" -name "checkpoint_*.csv" -type f | wc -l)
find "$RESULT_DIR" -name "checkpoint_*.csv" -type f -delete
echo "Deleted $deleted_count temporary files (checkpoint_*.csv and *_tmp.csv)"

echo ""
echo "===== Generated Files ====="
echo "1. Detailed data (for wandb with variance): $MERGED_CSV"
echo "2. Summary data (averaged per checkpoint): $SUMMARY_CSV"
