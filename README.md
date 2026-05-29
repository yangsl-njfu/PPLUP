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

## 修复方向

不要大改 MPC，也不要重写 RSS-CBF。

优先做最小修复：

1. 先补齐 blocking object 诊断字段，确认 static object 来源。
2. 如果 static object 是误检，例如 lidar fallback 或车道/路面对象，应修 static object 过滤/分类。
3. 如果 static object 是真实 blocker，但 lateral escape 已满足：

```text
road_boundary_safe = True
lateral_escape_terminal_recoverable = True
lateral_escape_lateral_rss_safe = True
lateral_escape_path_overlap_reduced = True
first_acc > 0
first_acc <= lateral_escape_max_acc
```

则应考虑让 certified low-speed lateral creep 使用更合理的 gate，而不是被 `critical_longitudinal_margin_safe=False` 一票否决。

修复目标是：

当 lateral escape 被证明道路安全、横向安全、终端可恢复，并且第一步是受限的小正加速度时，系统不应继续输出：

```text
escape-side steering + brake / zero throttle
```

而应输出：

```text
small positive acc/throttle + escape-side steering
```

如果最终仍然 brake，CSV 必须能明确说明原因是：

- critical margin unsafe
- road unsafe
- lateral_certified_gate failed
- terminal not recoverable
- first step rejected
- final guard rejected
