# PPL-MetaDrive 运行时 RSS 与弹簧阻尼安全层实验记录

本仓库当前关注的是在 MetaDrive 随机场景中，为 PPL/PVP 类驾驶策略增加运行时安全保障。这里的安全层只作用在评估或部署运行阶段，不参与训练，也不修改策略网络本身。

当前实现包含三种评估设置：

- 无 RSS：直接执行策略输出动作。
- 纯 RSS：使用纵向 RSS 与横向 RSS 作为硬安全屏障。
- RSS + 弹簧阻尼：保持 RSS 硬约束完全不变，在 RSS 触发前增加一个软性的弹簧阻尼预干预层。

## 实验配置

对比结果来自以下三个目录：

- `evaluation_results/PPL_de15a333`
- `evaluation_results/PPL_de15a333_RSS`
- `evaluation_results/PPL_de15a333_SpringDamper`

统计方式：

- 每组包含 11 个 checkpoint。
- 每个 checkpoint 评估 50 个 episode。
- 每组共 550 个 episode。
- 平均速度单位为 `m/s`，括号中给出 `km/h`。
- 当前评估配置启用了碰撞终止回合，因此碰撞率与 cost 不会因为一次碰撞在后续 step 中被重复累计。
- 静态障碍物已关闭，当前重点比较车辆交互下的运行时安全保障效果。

## 结果对比

| 方法 | 成功率 | 总碰撞率 | 车辆碰撞率 | 出路率 | 平均速度 | 平均 cost | 路线完成率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 无 RSS | 68.00% | 26.36% | 21.45% | 10.55% | 8.588 m/s (30.917 km/h) | 0.3200 | 87.55% |
| 纯 RSS | 93.09% | 6.91% | 6.73% | 0.18% | 8.070 m/s (29.054 km/h) | 0.0691 | 96.78% |
| RSS + 弹簧阻尼 | 95.09% | 4.73% | 4.18% | 0.73% | 8.050 m/s (28.981 km/h) | 0.0491 | 97.32% |

速度变化：

- 纯 RSS 相比无 RSS：平均速度下降 0.518 m/s，约下降 6.03%。
- RSS + 弹簧阻尼相比无 RSS：平均速度下降 0.538 m/s，约下降 6.26%。
- RSS + 弹簧阻尼相比纯 RSS：平均速度下降 0.020 m/s，约下降 0.25%。

安全变化：

- 纯 RSS 将车辆碰撞率从 21.45% 降到 6.73%。
- RSS + 弹簧阻尼进一步将车辆碰撞率降到 4.18%。
- 纯 RSS 将成功率从 68.00% 提升到 93.09%。
- RSS + 弹簧阻尼进一步将成功率提升到 95.09%。

当前结论：

RSS 显著提高安全性与任务完成率，但带来约 6% 的速度下降。RSS + 弹簧阻尼在速度上与纯 RSS 基本持平，只比纯 RSS 低约 0.25%，同时进一步降低车辆碰撞率并提高成功率。因此当前结果更适合表述为：在几乎不额外牺牲速度的情况下，弹簧阻尼软层进一步改善了安全性。若论文目标强调“缓解 RSS 保守性并提升效率”，还需要继续调参，使平均速度高于纯 RSS。

## 启动方式

无 RSS：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1
```

只观察 RSS，不接管动作：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1 -RssObserve
```

纯 RSS 硬屏障：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1 -RssObserve -RssShield -RssShieldMode standard
```

RSS + 弹簧阻尼：

```powershell
.\ppl\eval_script\eval_pvp_td3_enhanced.ps1 -RssObserve -RssShield -RssShieldMode spring_damper
```

注意：运行前建议在脚本中手动修改结果目录名称，避免不同实验结果写入同一个目录。

## 纯 RSS 实现

RSS 安全层位于：

```text
ppl/eval_script/metadrive/eval_ppl_metadrive.py
```

核心类为：

```text
RSSObserver
```

RSS 当前包含纵向与横向两部分。

### 纵向 RSS

纵向 RSS 用于判断自车与前车之间的安全距离。安全距离计算形式为：

```text
d_safe =
ego_v * rho
+ 0.5 * ego_max_accel * rho^2
+ (ego_v + rho * ego_max_accel)^2 / (2 * ego_min_brake)
- front_v^2 / (2 * front_max_brake)
```

其中：

- `rho` 为响应时间。
- `ego_v` 为自车速度。
- `front_v` 为前车速度。
- `ego_max_accel` 为响应时间内自车可能继续加速的最大加速度。
- `ego_min_brake` 为自车承诺可以达到的最小制动能力。
- `front_max_brake` 为前车可能采取的最大制动能力。

当实际距离小于安全距离时：

```text
distance < d_safe
```

系统进入纵向 RSS unsafe 状态，并触发硬制动响应。这个响应在 `standard` 模式和 `spring_damper` 模式中完全一致。

### 横向 RSS

横向 RSS 用于判断相邻车辆或横向接近车辆是否存在横向安全风险。横向安全距离根据横向闭合速度、响应时间、横向最大加速度、横向最小制动能力和安全裕度计算。

当前横向安全距离形式为：

```text
d_lat_safe =
lateral_safe_margin
+ closing_lateral_speed * rho
+ 0.5 * lateral_max_accel * rho^2
+ (closing_lateral_speed + lateral_max_accel * rho)^2 / (2 * lateral_min_brake)
```

当横向间隙小于横向安全距离时：

```text
lateral_gap < d_lat_safe
```

系统进入横向 RSS unsafe 状态。若策略动作正在朝风险车辆方向转向，则削弱该方向的转向并施加横向 RSS 制动；若横向间隙已经非常小，则直接触发硬制动。这个横向 RSS 硬响应在 `standard` 模式和 `spring_damper` 模式中也保持一致。

## 弹簧阻尼实现

弹簧阻尼不是替代 RSS，也不改变 RSS 的安全距离、unsafe 判定或 proper response。它只在 RSS 仍然安全但已经接近安全边界时生效，相当于一个运行时软预干预层。

可以把整体结构理解为：

```text
策略动作
  -> 弹簧阻尼软修正，只有 spring_damper 模式启用
  -> RSS 硬安全检查
  -> 最终执行动作
```

更准确地说，代码中优先保证 RSS 硬约束：

- 一旦纵向 RSS unsafe，立即使用原来的 RSS 硬制动。
- 一旦横向 RSS unsafe，立即使用原来的 RSS 横向硬响应。
- 只有 RSS safe 且进入 buffer 区域时，才允许弹簧阻尼进行轻微动作修正。

### 纵向弹簧阻尼

纵向软边界为：

```text
d_soft = d_safe + spring_longitudinal_buffer
```

当车辆还没有违反 RSS，但已经进入软边界：

```text
d_safe <= distance < d_soft
```

并且自车相对前车存在闭合速度时，弹簧阻尼层会降低策略给出的正 throttle。当前形式为：

```text
penetration_ratio = (d_soft - distance) / spring_longitudinal_buffer
closing_speed = max(0, ego_v - front_v)
throttle_reduction =
    spring_longitudinal_k * penetration_ratio
    + damper_longitudinal_k * closing_speed
```

其中：

- 弹簧项由距离软边界的侵入程度决定，越靠近 RSS 安全边界，修正越强。
- 阻尼项由闭合速度决定，接近速度越快，修正越强。
- 该层只降低正 throttle，不直接替代 RSS 制动。
- 如果已经违反 RSS，则不再使用软修正，而是交给 RSS hard brake。

当前默认参数：

```text
spring_longitudinal_buffer = 2.0
spring_longitudinal_k = 0.08
damper_longitudinal_k = 0.015
```

### 横向弹簧阻尼

横向软边界为：

```text
d_lat_soft = d_lat_safe + spring_lateral_buffer
```

当横向间隙仍然满足 RSS，但已经进入横向软边界：

```text
d_lat_safe <= lateral_gap < d_lat_soft
```

弹簧阻尼层会给策略动作增加一个远离风险车辆的轻微转向修正。当前形式为：

```text
soft_penetration = d_lat_soft - lateral_gap
closing_lateral_speed = max(0, 横向闭合速度)

desired_away =
    spring_lateral_k * (soft_penetration / spring_lateral_buffer)
    + damper_lateral_k * closing_lateral_speed
```

随后将 `desired_away` 限制在最大转向修正范围内：

```text
desired_away <= spring_lateral_max_steer
```

其中：

- 弹簧项由横向软边界侵入程度决定。
- 阻尼项由横向闭合速度决定。
- 修正方向始终远离风险车辆。
- 横向弹簧阻尼不主动制动，只做轻微转向修正。
- 如果已经违反横向 RSS，则立即交给 RSS 横向硬响应。

当前默认参数：

```text
spring_lateral_buffer = 0.4
spring_lateral_k = 0.05
damper_lateral_k = 0.02
spring_lateral_max_steer = 0.04
```

## 设计原则

本实现遵循以下原则：

1. RSS 是硬安全边界。
2. 弹簧阻尼只在 RSS safe 区域内生效。
3. 弹簧阻尼不能改变 RSS 的纵向和横向判定。
4. 弹簧阻尼不能改变 RSS unsafe 后的 proper response。
5. 弹簧阻尼的作用是提前、轻量、连续地修正动作，减少突然触发硬 RSS 的情况。

因此，论文中更稳妥的表述是：

```text
一种面向学习型驾驶策略的 RSS 边界驱动弹簧阻尼运行时安全层。
```

不建议直接写成“首次提出 RSS + 弹簧力”，因为 RSS 与势场、虚拟力结合的相关工作已经存在。更合适的创新点是强调：

- 作用对象是学习型驾驶策略的运行时动作输出。
- RSS 保持为不可放松的硬安全边界。
- 弹簧阻尼只作为 RSS safe 区域内的软预干预。
- 目标是在不明显牺牲效率的情况下减少碰撞和硬触发。

## 当前局限与下一步

当前 RSS + 弹簧阻尼结果在安全性上优于纯 RSS，但平均速度略低于纯 RSS 0.25%。这个差距很小，说明弹簧阻尼没有明显额外牺牲速度；但如果论文主张是“提升效率”，还需要继续调参。

下一步建议：

1. 保持 RSS 参数不变。
2. 优先减弱纵向弹簧阻尼，避免过早压低 throttle。
3. 保留或增强轻微横向弹簧阻尼，让方法更多通过柔性横向避让减少后续硬制动。
4. 用同一批 seed 和 checkpoint 比较纯 RSS 与 RSS + 弹簧阻尼。
5. 同时报告成功率、车辆碰撞率、平均速度、路线完成率和 cost。

推荐论文表述方向：

```text
在保持 RSS 硬安全约束不变的前提下，引入弹簧阻尼软预干预机制，使车辆在接近安全边界时提前产生连续、轻量的动作修正，从而降低碰撞风险并减少对策略动作的突兀覆盖。
```
