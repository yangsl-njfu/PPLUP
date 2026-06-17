# PPL-MetaDrive 运行时 RSS 安全层实验说明

本仓库当前用于评估 PPL/PVP 驾驶策略在 MetaDrive 随机场景中的运行时安全保障。安全层只在评估或部署运行阶段工作，不参与训练，也不修改策略网络。

当前重点包括：

- 纯 RSS 运行时安全屏障。
- RSS + Spring-Damper 软硬结合安全层。
- 感知不确定性下的 RSS 评估，即给 RSS monitor 的目标级输入加入高斯噪声。

## 实验结果记录

以下结果来自三个已有评估目录：

- `evaluation_results/PPL_de15a333`
- `evaluation_results/PPL_de15a333_RSS`
- `evaluation_results/PPL_de15a333_SpringDamper`

统计设置：

- 每组包含 11 个 checkpoint。
- 每个 checkpoint 评估 50 个 episode。
- 每组共 550 个 episode。
- 平均速度单位为 `m/s`，括号中给出 `km/h`。
- 当前评估配置启用了碰撞终止回合，因此一次碰撞不会在后续 step 中被重复累计 cost。
- 静态障碍物已关闭，当前重点比较车辆交互下的运行时安全保障效果。

| 方法 | 成功率 | 总碰撞率 | 车辆碰撞率 | 出路率 | 平均速度 | 平均 cost | 路线完成率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 无 RSS | 68.00% | 26.36% | 21.45% | 10.55% | 8.588 m/s (30.917 km/h) | 0.3200 | 87.55% |
| 纯 RSS | 93.09% | 6.91% | 6.73% | 0.18% | 8.070 m/s (29.054 km/h) | 0.0691 | 96.78% |
| RSS + Spring-Damper | 95.09% | 4.73% | 4.18% | 0.73% | 8.050 m/s (28.981 km/h) | 0.0491 | 97.32% |

速度变化：

- 纯 RSS 相比无 RSS：平均速度下降 0.518 m/s，约下降 6.03%。
- RSS + Spring-Damper 相比无 RSS：平均速度下降 0.538 m/s，约下降 6.26%。
- RSS + Spring-Damper 相比纯 RSS：平均速度下降 0.020 m/s，约下降 0.25%。

当前结论：

纯 RSS 显著提高安全性与任务完成率，但带来约 6% 的速度下降。RSS + Spring-Damper 在速度上与纯 RSS 基本持平，同时进一步降低车辆碰撞率并提高成功率。当前结果更适合表述为：在几乎不额外牺牲速度的情况下，Spring-Damper 软层进一步改善了安全性。

## PowerShell 启动参数

主要脚本：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1
```

### `-RssObserve`

开启 RSS 观测与统计。

开启后会记录 RSS 的状态，例如：

- RSS 是否 unsafe。
- 纵向 RSS unsafe 步数。
- 横向 RSS unsafe 步数。
- RSS shield 触发步数。
- 前车距离、安全距离。
- 横向间隙、横向安全距离。
- 感知噪声相关统计。

这些信息会写入 CSV。

如果只传这个参数，不传 `-RssShield`，则只观察 RSS，不接管动作。

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1 -RssObserve
```

### `-RssShield`

开启 RSS 运行时接管。

如果 RSS 判断当前策略动作不安全，就会修改 policy 输出动作。例如：

- 纵向风险时执行 RSS 制动。
- 横向风险时抑制朝危险车辆方向的转向。
- 在严重横向风险下执行制动。

通常与 `-RssObserve` 一起使用：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1 -RssObserve -RssShield
```

### `-RssShieldMode`

选择 RSS shield 的响应模式。

可选值：

```text
standard
spring_damper
```

`standard` 表示纯 RSS 硬安全屏障：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1 -RssObserve -RssShield -RssShieldMode standard
```

`spring_damper` 表示 RSS 硬安全屏障 + Spring-Damper 软干预：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1 -RssObserve -RssShield -RssShieldMode spring_damper
```

注意：`spring_damper` 不改变 RSS 的纵向/横向安全距离，不改变 RSS unsafe 判定，也不改变 RSS unsafe 后的 proper response。它只在 RSS safe 但接近安全边界时进行轻量软修正。

### `-RssUncertainty`

控制是否给 RSS monitor 的输入加入感知不确定性。

可选值：

```text
none
gaussian
```

`none` 表示关闭感知噪声，RSS 使用无噪声目标状态：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1 -RssObserve -RssShield -RssShieldMode standard -RssUncertainty none
```

这就是纯 RSS clean setting。

`gaussian` 表示开启高斯感知噪声：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1 -RssObserve -RssShield -RssShieldMode standard -RssUncertainty gaussian
```

### `-RssNoiseLevel`

选择高斯感知噪声档位。只有 `-RssUncertainty gaussian` 开启时生效。

可选值：

```text
small
medium
large
```

默认值是 `medium`，作为 MetaDrive 主实验设置。三档参数固定在 `eval_pvp_td3_enhanced.ps1` 中：

| 档位 | 定位 | sigma_x | sigma_y | sigma_v | sigma_theta |
|---|---|---:|---:|---:|---:|
| `small` | MetaDrive 温和感知误差 | 0.50 m | 0.15 m | 0.50 m/s | 0.02 rad |
| `medium` | MetaDrive 主实验默认值 | 1.00 m | 0.30 m | 1.00 m/s | 0.05 rad |
| `large` | 论文 highway large covariance / 压力测试 | 1.87 m | 0.54 m | 2.64 m/s | 0.10 rad |

示例：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1 -RssObserve -RssShield -RssShieldMode spring_damper -RssUncertainty gaussian -RssNoiseLevel medium
```

随机种子固定为 `7`。

噪声作用于 RSS monitor 看到的目标车感知状态 `(x, y, v, theta)`。RSS 会基于 noisy 后的目标车状态重新做 route/Frenet 投影、前车筛选、横向候选筛选和安全距离计算；它不改变 MetaDrive 真实车辆位置，也不改变 policy observation。

## 常用实验命令

无 RSS：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1
```

只观察 RSS，不接管动作：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1 -RssObserve
```

纯 RSS，无遮挡、无噪声：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1 -RssObserve -RssShield -RssShieldMode standard -RssUncertainty none
```

RSS + Spring-Damper，无遮挡、无噪声：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1 -RssObserve -RssShield -RssShieldMode spring_damper -RssUncertainty none
```

纯 RSS，高斯感知噪声：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1 -RssObserve -RssShield -RssShieldMode standard -RssUncertainty gaussian -RssNoiseLevel medium
```

RSS + Spring-Damper，高斯感知噪声：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1 -RssObserve -RssShield -RssShieldMode spring_damper -RssUncertainty gaussian -RssNoiseLevel medium
```

## 感知不确定性设定

当前实现与 “Risk-Based Safety Envelopes for Autonomous Vehicles Under Perception Uncertainty” 的问题设定对齐：RSS monitor 不再假设感知输入完全准确，而是接收带噪声的目标级状态估计。

对每个目标车构造感知状态：

```text
x_obs     = x_true     + epsilon_x
y_obs     = y_true     + epsilon_y
v_obs     = max(0, v_true + epsilon_v)
theta_obs = theta_true + epsilon_theta
```

其中：

```text
epsilon_x     ~ N(0, sigma_x^2)       m
epsilon_y     ~ N(0, sigma_y^2)       m
epsilon_v     ~ N(0, sigma_v^2)       m/s
epsilon_theta ~ N(0, sigma_theta^2)   rad
```

在实现中，`epsilon_x` 沿目标车当前航向方向加入，`epsilon_y` 沿目标车横向方向加入。随后 RSS 使用 noisy 目标车状态重新计算：

```text
raw target state
  -> Gaussian perception uncertainty on (x, y, v, theta)
  -> route/Frenet projection and candidate selection
  -> RSS / Spring-Damper
```

代码模块：

- `ppl/utils/rss_uncertainty.py`：可插拔感知不确定性模块。
- `ppl/eval_script/metadrive/eval_ppl_metadrive.py`：RSSObserver 接入噪声后的目标级输入。
- `ppl/eval_script/eval_pvp_td3_enhanced.ps1`：实验启动脚本与固定噪声配置。

## RSS 实现概述

RSS 安全层位于：

```text
ppl/eval_script/metadrive/eval_ppl_metadrive.py
```

核心类：

```text
RSSObserver
```

RSS 当前包含纵向与横向两部分。

### 纵向 RSS

纵向 RSS 判断自车与前车之间是否满足安全距离。安全距离形式为：

```text
d_safe =
ego_v * rho
+ 0.5 * ego_max_accel * rho^2
+ (ego_v + rho * ego_max_accel)^2 / (2 * ego_min_brake)
- front_v^2 / (2 * front_max_brake)
```

当：

```text
distance < d_safe
```

系统进入纵向 RSS unsafe 状态，并触发硬制动响应。

### 横向 RSS

横向 RSS 判断相邻车辆或横向接近车辆是否存在横向安全风险。横向安全距离形式为：

```text
d_lat_safe =
lateral_safe_margin
+ closing_lateral_speed * rho
+ 0.5 * lateral_max_accel * rho^2
+ (closing_lateral_speed + lateral_max_accel * rho)^2 / (2 * lateral_min_brake)
```

当：

```text
lateral_gap < d_lat_safe
```

系统进入横向 RSS unsafe 状态。若策略动作正在朝风险车辆方向转向，则削弱该方向的转向并施加横向 RSS 制动；若横向间隙已经非常小，则直接触发硬制动。

## Spring-Damper 实现概述

Spring-Damper 不是替代 RSS，也不改变 RSS 的安全距离、unsafe 判定或 proper response。它只在 RSS 仍然 safe 但已经接近安全边界时生效，相当于一个运行时软预干预层。

整体结构：

```text
policy action
  -> Spring-Damper soft correction
  -> RSS hard safety check
  -> final action
```

更准确地说：

- 一旦纵向 RSS unsafe，立即使用原来的 RSS 硬制动。
- 一旦横向 RSS unsafe，立即使用原来的 RSS 横向硬响应。
- 只有 RSS safe 且进入 buffer 区域时，才允许 Spring-Damper 进行轻微动作修正。

纵向软边界：

```text
d_soft = d_safe + spring_longitudinal_buffer
```

横向软边界：

```text
d_lat_soft = d_lat_safe + spring_lateral_buffer
```

当前默认参数：

```text
spring_longitudinal_buffer = 2.0
spring_longitudinal_k = 0.08
damper_longitudinal_k = 0.015

spring_lateral_buffer = 0.4
spring_lateral_k = 0.05
damper_lateral_k = 0.02
spring_lateral_max_steer = 0.04
```

## 论文定位

推荐将当前工作表述为：

```text
面向学习型驾驶策略的感知不确定性下 RSS 运行时安全层。
```

与 Risk-Based Safety Envelope 类工作的区别可以放在后续方法中：

- 对方依赖显式噪声模型或风险分布建模。
- 后续方法可以使用共形预测从校准数据中估计误差分位数。
- 共形预测不要求感知误差服从特定分布。
- Spring-Damper 用于缓解风险裕度或共形裕度带来的保守性与动作突变。

建议主线：

```text
perception uncertainty
  -> noisy RSS 可能产生误判
  -> conformal calibration 给出分布无关的保守裕度
  -> RSS 提供硬安全边界
  -> Spring-Damper 缓解效率下降与动作突变
```
