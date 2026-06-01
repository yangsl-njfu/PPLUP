# RSS-MPC / RSS-CBF 当前死锁问题总结

## 问题现象

在 MetaDrive RSS-MPC 评估中，部分场景会出现车辆低速卡死：

- steering 有持续输出
- throttle 接近 0 或没有
- brake 有轻微输出
- speed 接近 0

表现为车辆原地或极低速转向，但无法给出小正加速度绕行。

## 当前定位到的卡死样本

当前截图对应的 CSV 是：

```text
evaluation_results/PPL_0ee03603_rss_mpc_1/checkpoint_6200_rss_mpc_steps_tmp.csv
```

截图中的 `Total Step:5044` 映射到：

```text
checkpoint = 6200
env_id = 1024
step = 1304
```

不是 `checkpoint_6000/env_id=1010`。

## 关键诊断结果

卡死点处 RSS-MPC 已经进入 recovery/deadlock 路径：

```text
mode = rss_mpc_fallback_to_cbf
mpc_called = True
mpc_call_reason = deadlock_risk
deadlock_risk = True
deadlock_score ≈ 0.9997
deadlock_window_avg_speed ≈ 0.0012
```

MPC 确实生成了左右 lateral escape 候选：

```text
right_candidate_generated = True
left_candidate_generated = True
num_candidates_right = 6
num_candidates_left = 6
```

左右 escape 在道路边界和终端恢复上看起来可行：

```text
road_boundary_safe = True
lateral_escape_terminal_recoverable = True
right_escape_available = True
left_escape_available = True
lateral_escape_lateral_rss_safe = True
lateral_escape_lateral_margin_improved = True
lateral_escape_path_overlap_reduced = True
```

但 lateral certified gate 没有通过：

```text
lateral_certified_gate = False
critical_longitudinal_margin_safe = False
right_escape_reject_reason = lateral_escape_rejected_by_critical_margin
left_escape_reject_reason = lateral_escape_rejected_by_critical_margin
lateral_escape_reject_reason = lateral_escape_rejected_by_critical_margin
```

因此没有候选进入最终选择：

```text
selected_candidate_family = ""
mpc_num_feasible = 0
mpc_num_rss_feasible = 0
mpc_no_rss_feasible_count = 25
mpc_guard_rejected_count = 0
```

这说明问题不是 final RSS-CBF guard 把已经认证的 creep 压成 brake，因为根本没有 selected lateral candidate 进入 guard。

## 当前直接原因

卡死时被选中的前向 blocking object 是 `static`，不是动态车辆：

```text
object_kind = static
dynamic_vehicle_detected = False
d_front ≈ 2.36 m
rss_distance ≈ 3.43 m
rss_margin_current ≈ -1.06 m
```

也就是说，车辆距离前方 static object 小于当前 RSS 静态安全距离，导致：

```text
critical_longitudinal_margin_safe = False
```

而 lateral escape gate 把 `critical_longitudinal_margin_safe=True` 作为硬条件。即使左右绕行能减少 overlap、道路安全、终端可恢复，也会被拒绝。

最终系统只能 fallback 到 RSS-CBF：

```text
acc_safe ≈ -0.02
steer_safe ≈ -0.756
velocity ≈ 0.001
env_action_safe_throttle_brake ≈ -0.005
```

这就是渲染里看到的：

```text
escape-side steering + brake / zero throttle
```

而不是：

```text
small positive throttle + escape-side steering
```

## 当前系统问题

当前系统的问题可以概括为：

RSS-MPC 能检测 deadlock，也能生成左右 lateral escape；但 lateral escape 认证逻辑过于依赖前向 critical longitudinal margin。当 static blocking object 距离略小于 RSS 静态安全距离时，系统不允许低速小正加速度横向绕行，导致所有 escape 候选在进入 final guard 前被拒，最终退回 RSS-CBF 的 steer + brake/zero throttle。

简单把 MPC 接在 CBF 后面并不能真正解决死锁，因为 CBF 的一步纵向安全判据会覆盖 MPC 的多步横向恢复证据。

核心链路是：

```text
static front object
-> rss_margin_current < 0
-> critical_longitudinal_margin_safe = False
-> lateral_certified_gate = False
-> left/right escape rejected
-> mpc_num_feasible = 0
-> fallback_to_cbf
-> steer + zero throttle / tiny brake
```

## 不是哪类问题

当前证据显示，这不是：

- MPC 完全没有生成绕行候选
- final RSS-CBF guard 压掉了已经认证的 creep
- road boundary unsafe
- terminal not recoverable
- dynamic vehicle 纵向距离过长

当前更像是：

- static object / static blocker 的前向 RSS margin 触发过严
- lateral escape gate 对 `critical_longitudinal_margin_safe` 的硬约束过严
- 或 static object 的来源/类型需要进一步确认

## 仍需确认的问题

目前 CSV 只能确认 blocking object 是 `static`，还不能确认它具体是什么。

它可能是：

- 静止车辆被归类成 static obstacle
- MetaDrive 的静态障碍物、barrier、cone、traffic object
- observation lidar fallback 生成的 `front_lidar` 伪障碍
- 车道边界、路面对象或其他环境对象被解析/扫描成 static front object

下一步应补充最小诊断字段：

```text
blocking_object_type
blocking_object_class_name
blocking_object_id
blocking_object_speed
adapter_static_count
adapter_vehicle_count
observation_lidar_fallback_used
observation_lidar_source
observation_lidar_distance
```

这样才能判断 static object 是否真的是车辆、障碍物、lidar fallback，还是车道/路面相关对象。

## 当前修复方向

现在不再继续修补旧的：

```text
RSS-CBF shield -> MPC recovery -> final RSS-CBF veto
```

因为这个结构的根本问题是：MPC 给出 horizon-level lateral recovery，但 final RSS-CBF / lateral gate 仍会用 one-step longitudinal RSS/CBF 判据裁掉小正加速度，最后变成 `steer + brake`。

新的 deadlock recovery 分支改为：

```text
MPC 内部评估 RSS/CBF constraints
-> 选择 certified recovery action
-> final guard 只做 emergency veto
```

normal 路径仍保持不变：

1. RL action safe：直接执行 RL action。
2. RL action unsafe 但不是 deadlock-risk：走现有 RSS-CBF。
3. deadlock-risk / brake-only-risk：进入 Predictive RSS-CBF-MPC。

在 Predictive RSS-CBF-MPC 中：

- immediate collision、road boundary、action bounds、catastrophic lateral conflict 是硬约束。
- conservative longitudinal RSS margin 和 one-step longitudinal CBF margin 不再一票否决，而是作为 soft constraint 进入代价，并附加大惩罚。
- lateral recovery 只要满足小正加速度、escape-side steering、道路安全、无立即碰撞、横向 RSS 安全或改善、path overlap 减少、终端可恢复，就可以被认证。
- final guard 对 certified lateral recovery 不能再把 `positive acc + escape-side steer` 改成 `brake/zero acc + escape-side steer`。
- final guard 只允许因 immediate collision、road boundary violation、catastrophic lateral RSS violation、action out of bound 做 emergency veto。

deadlock-risk 分支中的 MPC candidate 统一输出：

```text
hard_safe
soft_longitudinal_slack
recovery_certified
emergency_veto
total_cost
```

其中 `hard_safe` 只包含立即碰撞、道路边界、横向 RSS/终端横向安全、灾难性侧向冲突、动作边界。纵向 RSS/CBF margin 不再进入 hard reject，只进入 `soft_longitudinal_slack` 和 cost penalty。

修复后的目标输出是：

```text
small positive acc/throttle + escape-side steering
```

而不是：

```text
escape-side steering + brake / zero throttle
```

如果最终仍然 brake，CSV 必须能明确说明原因是：

- no_recovery_certified_candidate
- emergency_veto
- immediate_collision_risk
- road_boundary_violation
- catastrophic_lateral_rss_violation
- action_out_of_bounds
