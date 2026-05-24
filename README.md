# PPL MetaDrive RSS-CBF 运行时安全保障

本分支将 CBF-RSS 工作整合为一条清晰的 RSS-CBF 运行时安全保障评估路径，
用于 MetaDrive 中的 PPL/TD3 策略。

本分支保留两种官方评估方法：

- `ppl`：原始 PPL 策略，无运行时安全保障。
- `ppl_rss_cbf`：带有 RSS-CBF 运行时安全保障动作过滤的 PPL 策略。

## 本分支新增内容

当前分支变更：

- 在 `ppl/utils/rss_cbf_filter.py` 中新增 `RSSCBFConfig` 和 `RSSCBFFilter`。
- 保留现有的 MetaDrive 状态/动作适配函数：
  - `parse_state_from_metadrive(env)`
  - `to_internal_action(...)`
  - `from_internal_action(...)`
  - `detect_dynamic_vehicle_ahead(...)`
  - `compute_rss_distance(...)`
  - `filter_action(state, u_nom)`
- 将正式的 RSS-CBF 输出模式限定为：
  - `normal`（正常）
  - `rss_cbf_intervention`（RSS-CBF 干预）
  - `rss_cbf_recovery`（RSS-CBF 恢复）
  - `fallback_no_safe_candidate`（无安全候选回退）
- 移除旧的公开评估开关，包括静态 RSS 实验、旁路、
  间隙保护、干运行、调试打印和诊断环境选择。
- 仅保留官方的 RSS-CBF 用户开关：
  - PowerShell: `-RSSCBF`
  - Python: `--rss_cbf`
- 将 RSS-CBF 步骤诊断信息保存为：
  - `checkpoint_XXXX_rss_cbf_steps.csv`
- 在评估输出中新增 `method` 字段：
  - `ppl`
  - `ppl_rss_cbf`
- 从 `ppl/utils/train_eval_config.py` 加载基线评估设置。
- 在构建 `DrivingEnv` 前过滤遗留的 `main_exp` 键，因为
  当前 MetaDrive 配置不接受该键。
- 渲染默认遵循 `baseline_eval_config["use_render"]`。
  Python 评估可通过 `--use_render` 或 `--no_render` 覆盖此设置。

## RSS-CBF 过滤器行为

运行时路径为：

```text
策略动作 u_nom -> RSS-CBF 运行时安全保障 -> u_safe
```

当前 RSS-CBF 过滤器主要是一个前向 RSS 安全距离过滤器。
它主要修改纵向加速度并保留策略转向：

```text
u_safe = [safe_acc, u_nom_steer]
```

RSS 距离公式为：

```text
d_rss = v * rho
      + 0.5 * a_max * rho^2
      + (v + a_max * rho)^2 / (2 * b_min)
      + margin
```

安全裕度为：

```text
h_rss = d_front - d_rss
```

如果当前 RSS 裕度安全且名义动作不会恶化裕度，
过滤器返回 `normal` 并保留策略动作。如果裕度不安全或名义动作增加风险，
过滤器将加速度投影到更安全的动作。当系统已经处于不安全状态时，
恢复模式接受保持或改善当前裕度的动作，而不是要求一步内立即完全恢复。

当前官方版本不主动执行旁路、变道或轨迹规划。这些功能保留给未来的 MPC 工作。

## 配置默认值

`StaticRSSConfig` 和 `RSSCBFConfig` 使用本分支所需的稳定默认值：

```python
enable_predictive_clearance_guard = False
preserve_steer_on_stop = True
fallback_to_brake = False
enable_recovery_mode = True
intervention_margin_threshold = 0.0
```

原理说明：

- 预测间隙保护默认禁用，因为之前它会导致频繁的 `clearance_stop` 行为和车辆停滞。
- RSS-CBF 干预期间保留转向。
- 无安全候选回退默认保留名义动作，而不是强制最大制动。
- 启用恢复模式以处理已经处于不安全状态的 RSS 状态。

## 启动命令

### 批量评估

原始 PPL：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1
```

带有 RSS-CBF 运行时安全保障的 PPL：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1 -RSSCBF
```

PowerShell 脚本输出：

```text
evaluation_results\PPL_0ee03603-enhanced
evaluation_results\PPL_0ee03603_rss_cbf
```

如果启用了 RSS-CBF，每个 checkpoint 的步骤诊断信息将保存为：

```text
checkpoint_XXXX_rss_cbf_steps.csv
```

临时的 checkpoint 回合 CSV 文件会被清理，但 RSS-CBF 步骤诊断信息不会被删除。

### 单 Checkpoint 评估

原始 PPL：

```powershell
python -m ppl.eval_script.metadrive.eval_ppl_metadrive --path runs/PPL/PPL_0ee03603/models --ckpt_index 6200 --ret_save_folder evaluation_results/debug_ppl --total_env_num 1 --num_ep_in_one_env 1
```

带有 RSS-CBF 的 PPL：

```powershell
python -m ppl.eval_script.metadrive.eval_ppl_metadrive --path runs/PPL/PPL_0ee03603/models --ckpt_index 6200 --ret_save_folder evaluation_results/debug_ppl_rss_cbf --rss_cbf --total_env_num 1 --num_ep_in_one_env 1
```

强制开启渲染：

```powershell
python -m ppl.eval_script.metadrive.eval_ppl_metadrive --path runs/PPL/PPL_0ee03603/models --ckpt_index 6200 --ret_save_folder evaluation_results/debug_ppl_rss_cbf_render --rss_cbf --total_env_num 1 --num_ep_in_one_env 1 --use_render
```

强制关闭渲染：

```powershell
python -m ppl.eval_script.metadrive.eval_ppl_metadrive --path runs/PPL/PPL_0ee03603/models --ckpt_index 6200 --ret_save_folder evaluation_results/debug_ppl_rss_cbf_no_render --rss_cbf --total_env_num 1 --num_ep_in_one_env 1 --no_render
```

## 如何确认 RSS-CBF 已启用

当 RSS-CBF 激活时，控制台输出包括：

```text
[RSS-CBF] Runtime assurance enabled. Step diagnostics will be saved.
[RSS-CBF] total_steps=... changed_actions=... changed_rate=... modes={...} cost_by_mode={...}
```

结果 CSV 应包含：

```text
method = ppl_rss_cbf
```

步骤诊断文件应包含以下字段：

```text
step
mode
reason
rss_margin
dynamic_vehicle_detected
action_delta
acc_nominal
acc_safe
acc_delta
steer_nominal
steer_safe
steer_delta
step_cost
cumulative_cost
crash
crash_vehicle
crash_object
out_of_road
route_completion
velocity
```

示例检查命令：

```powershell
Import-Csv evaluation_results\debug_ppl_rss_cbf\checkpoint_6200_rss_cbf_steps.csv |
  Select-Object -First 20
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

## 停滞 / 超时说明

现有回合级别结果表明，RSS-CBF 提高了安全性，但仍存在少量停滞尾部：

- 原始 PPL 有 `0/1050` 个回合的 `episode_length == 1500`。
- RSS-CBF 有 `31/550` 个回合的 `episode_length == 1500`。
- 所有 31 个 RSS-CBF 超时的回合都具有低平均速度、无碰撞、
  无出界终止、无成功。

这表明当前 RSS-CBF 过滤器仍可能在少数回合中导致低速停滞。
可能的原因是官方候选集较窄：过滤器调整加速度，但不主动执行
旁路、变道或轨迹规划。

未来的 MPC 工作应在单独分支上开发，并视为一种新方法，例如：

```text
ppl_mpc_rss_cbf
```

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

当前结论：

```text
RSS-CBF 作为运行时安全过滤器是有效的，但它不是规划器。
它在提高安全性的同时，引入了少量低速停滞的风险。
```

## 数据说明

- RSS-CBF 的 `all_checkpoints_summary.csv` 目前存在重复的 `8000` 行。
- RSS-CBF 的 `all_checkpoints_evaluation.csv` 目前仅覆盖
  `8000` 至 `10000` 的 checkpoint，而 PPL 评估 CSV 覆盖
  `6000` 至 `10000`。
- 在用于最终报告之前，请重新生成 RSS-CBF 合并后的
  CSV 文件，以确保摘要级别和回合级别的覆盖范围一致。
