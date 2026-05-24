# PPL MetaDrive 评估

本项目在 MetaDrive 环境中评估 PPL/TD3 策略。当前评估对比两种方法：

- `ppl`：原始 PPL 策略动作，无运行时安全保障。
- `ppl_rss_cbf`：经 RSS-CBF 运行时安全保障过滤的 PPL 策略动作。

## 运行时安全保障模式

仅通过以下方式启用 RSS-CBF 运行时安全保障：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1 -RSSCBF
```

基线 PPL 评估命令为：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1
```

单 checkpoint 评估示例：

```powershell
python -m ppl.eval_script.metadrive.eval_ppl_metadrive --path runs/PPL/PPL_0ee03603/models --ckpt_index 6200 --ret_save_folder evaluation_results/debug_ppl --total_env_num 1 --num_ep_in_one_env 1
```

```powershell
python -m ppl.eval_script.metadrive.eval_ppl_metadrive --path runs/PPL/PPL_0ee03603/models --ckpt_index 6200 --ret_save_folder evaluation_results/debug_ppl_rss_cbf --rss_cbf --total_env_num 1 --num_ep_in_one_env 1
```

启用 RSS-CBF 后，每步诊断信息将写入：

```text
checkpoint_XXXX_rss_cbf_steps.csv
```

## PPL vs RSS-CBF 结果对比

数据来源：

- `evaluation_results/PPL_0ee03603/all_checkpoints_summary.csv`
- `evaluation_results/PPL_0ee03603_rss_cbf/all_checkpoints_summary.csv`
- `evaluation_results/PPL_0ee03603/all_checkpoints_evaluation.csv`
- `evaluation_results/PPL_0ee03603_rss_cbf/all_checkpoints_evaluation.csv`

摘要对比使用 `6000` 至 `10000` 的 checkpoint 级别结果。
RSS-CBF 摘要文件中 checkpoint `8000` 存在一行重复数据；
以下对比将 `ckpt_index` 视为唯一值并删除重复项。

| 指标均值 | PPL | PPL + RSS-CBF | 变化 |
|---|---:|---:|---:|
| 成功率 | 0.7143 | 0.7371 | +0.0229 |
| 碰撞率 | 0.0714 | 0.0371 | -0.0343 |
| 出界率 | 0.2857 | 0.2095 | -0.0762 |
| 平均代价 | 3.5991 | 1.9867 | -1.6124 |
| 平均奖励 | 324.12 | 322.78 | -1.34 |
| 平均路线完成度 | 0.8781 | 0.8732 | -0.0049 |
| 平均速度 | 11.94 | 9.40 | -2.54 |
| 平均回合长度 | 259.93 | 397.28 | +137.35 |

Checkpoint 级别结果统计：

| 指标 | 改善 | 持平 | 变差 |
|---|---:|---:|---:|
| 成功率 | 13/21 | 2/21 | 6/21 |
| 碰撞率 | 17/21 | 2/21 | 2/21 |
| 出界率 | 16/21 | 2/21 | 3/21 |
| 平均代价 | 21/21 | 0/21 | 0/21 |
| 平均奖励 | 10/21 | 0/21 | 11/21 |
| 平均路线完成度 | 9/21 | 0/21 | 12/21 |

各方法最优 checkpoint：

| 评判标准 | PPL | PPL + RSS-CBF |
|---|---:|---:|
| 最高成功率 | ckpt 8600, 0.86 | ckpt 9800, 0.86 |
| 最高平均奖励 | ckpt 8600, 345.996 | ckpt 9800, 342.352 |
| 最高路线完成度 | ckpt 8600, 0.9305 | ckpt 9800, 0.9249 |
| 最低平均代价 | ckpt 8600, 2.52 | ckpt 8200, 1.42 |

对于回合级别的 `all_checkpoints_evaluation.csv` 文件，重叠的
checkpoint 范围仅为 `8000` 至 `10000`。在此重叠范围内，两种方法均有
550 个评估回合。

| 重叠指标均值 | PPL | PPL + RSS-CBF | 变化 |
|---|---:|---:|---:|
| 成功率 | 0.7291 | 0.7545 | +0.0255 |
| 碰撞率 | 0.0564 | 0.0255 | -0.0309 |
| 出界率 | 0.2709 | 0.1891 | -0.0818 |
| 回合代价 | 3.8836 | 2.1509 | -1.7327 |
| 回合奖励 | 328.82 | 326.55 | -2.27 |
| 路线完成度 | 0.8892 | 0.8833 | -0.0060 |

## 结果解读

RSS-CBF 运行时安全保障显著改善了安全指标：

- 每个 checkpoint 的 `平均代价` 均有所下降。
- 大多数 checkpoint 的 `碰撞率` 和 `出界率` 均有所下降。
- `成功率` 平均而言有所改善。

代价是驾驶策略更为保守：

- 平均速度降低，
- 回合长度增加，
- 平均奖励略有下降，
- 平均路线完成度略有下降。

简而言之，当前 RSS-CBF 配置作为安全过滤器是有效的，
但会降低策略速度，因此需要同时结合安全性指标和进度指标进行评估。

## 数据说明

- RSS-CBF 的 `all_checkpoints_summary.csv` 目前存在重复的 `8000` 行。
- RSS-CBF 的 `all_checkpoints_evaluation.csv` 目前仅覆盖
  `8000` 至 `10000` 的 checkpoint，而 PPL 评估 CSV 覆盖
  `6000` 至 `10000`。
- 在用于最终报告之前，请重新生成 RSS-CBF 合并后的
  CSV 文件，以确保摘要级别和回合级别的覆盖范围一致。
