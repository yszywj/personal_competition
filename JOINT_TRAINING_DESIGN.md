# 联合强化学习训练接口与实现契约

本文描述 `personal_train` 当前联合训练实现的实际行为。目标是让一套策略同时决定每枚红方飞行器的部署/发射、目标、再指派、机动和卫星请求，并明确 Actor 可以使用的数据、动作合法性、奖励和回合边界。

## 1. 已采用的总体方案

- 每枚红方高/中/低速飞行器对应一个稳定单位槽位，共享同一个单位策略网络。
- 卫星不是独立 Agent。同一策略使用团队 pointer/STOP 头选择请求导弹。
  适配器自动区分当前 `team_global` 和旧 `per_unit` 后端，并按对应的额度、
  激活窗口与回执语义维护账本。
- 目标使用固定离散槽位；策略输出目标槽位，不直接输出实体 ID。
- 部署发生在游戏进行过程中。对 `STAGED` 单位，激活动作同时给出 `(x, y)`、首个目标和初始机动；适配器先修改位置，再在同一仿真步发送发射及机动命令。
- 规划 Actor 按 `激活 → 目标 → 部署位置/首次机动` 自回归采样；换目标时，本步机动以新选目标为条件。
- 目标情报采用集中合法共享：只融合所有红方实体在隔离观测中合法获得的 `detectInfo`。任一红方实体发现目标后，该目标可以被所有红方单位选择。
- Actor 同时使用本单位特征和所有实际单位特征的团队均值池化上下文；plan、motion 和 sensor 三个 Critic 使用同源的非特权上下文。

当前实现可以在单一场景容量下训练。覆盖 final20 的跨场景统一
`U_MAX=420`、`O_MAX=48` 张量及 padding 尚未实现，这是当前主要剩余结构限制，
详见第 10 节。

## 2. 仿真可见数据边界

### 2.1 Actor 合法输入

Actor 只从场景公开初始信息、受控红方实体自己的记录和控制器历史构造输入。适配器从完整帧中只提取每个受控实体的以下字段：

- `step`
- `self.position`、`self.pos_ecf`
- `self.stage`、`self.health`、`self.isVisible`
- `self.type`、`self.side`
- `self.detectInfo`
- `self.commRangeInfo`

实体 ID 仅用于内部稳定槽位和命令翻译，不作为数值特征进入网络。

目标航迹来自所有红方隔离观测中的 `detectInfo` 并集中融合。`DetectInfo` 可提供：

- `detect_from`
- `time`
- `entity_id`
- `entity_type`
- `nameChn`
- `lla`
- `pos_ecf`
- `vel_ecf`

航迹本身不提供目标当前血量、存活状态或权威可见性。训练输入不得从完整环境观测中读取尚未探测的蓝方坐标、速度或血量。

### 2.2 集中合法探测共享

当前实现采用团队情报共享，具体规则如下：

1. 遍历所有红方实体自己的隔离观测。
2. 只读取这些观测里的 `detectInfo`。
3. 对同一目标按严格递增的源时间戳更新航迹。
4. 任一红方实体合法发现的目标会进入团队目标表和全局目标 mask。
5. 未经任一红方隔离观测发现的隐藏目标仍保持 `known=0`、`valid=0`，其几何和运动字段为零。

因此，全局目标 mask 是有意采用的团队通信语义，而不是从完整蓝方真值泄漏信息。`9400`、`9600` 可按场景公开信息初始化；`9500` 不在初始化舰船观测中，只有经过合法探测后才变为可选。

## 3. 容量与槽位

### 3.1 当前场景容量

已知场景上限如下：

| 场景 | 红方单位数 | 计分目标数 | 拦截弹总数 | 自动 `O` | 观测 `D` |
|---|---:|---:|---:|---:|---:|
| legacy nine（最大） | 304 | 14 | 432 | 18 | 382 |
| final24（最大） | 360 | 18 | 324 | 18 | 382 |
| final20 easy | 210 | 24 | 189 | 24 | 496 |
| final20 medium | 300 | 36 | 270 | 36 | 724 |
| final20 hard | 420 | 48 | 378 | 48 | 952 |

正式时限不按场景名称硬编码，必须以加载后的 `case_info` 和
`reward_policy.max_steps` 为准；legacy E01 为 1200 步，final20 为 3000 步。

目标槽位未显式配置时自动取
`O=max(18, len(RewardPolicy.objective_ids))`。显式的 `--objective-slots` 仍可覆盖，
但小于场景计分目标数会在构造时失败。单位维使用场景实际数量 `U_actual`：

```text
observation.shape == [U_actual, 40 + 19*O]
action_mask.unit_count == U_actual
```

`--scenario E01` 仍解析到 legacy easy/E01。final20 必须使用带套件的 selector
`final20/easy/E01` 或显式路径：

```text
/home/amax/ry/competition/competition_envs/scenarios/cases/final20/easy/E01/scenario.json
```

### 3.2 单位槽位

环境创建时收集红方高/中/低速飞行器，先按高/中/低速类型索引排序，再在同一类型内按数值实体 ID 升序排列。该映射在一个环境实例和回合内保持稳定：

```text
unit_slot -> red entity_id
```

实体死亡后槽位不会被复用，其相位变为 `TERMINAL`。当前没有额外 padding 槽。

### 3.3 目标槽位

目标槽位按 `RewardPolicy.objective_ids` 的固定顺序建立，不足当前 `O` 时在末尾填空：

```text
target_slot -> objective entity_id | None
```

目标 ID 只用于适配器内部维护槽位和下发坐标，不直接编码给 Actor。已知公开目标或被合法探测的目标填入观测；隐藏且未发现的目标虽然在内部保留稳定槽位，其 Actor 字段仍是全零且 mask 为假。

## 4. 单位状态机与动作语义

每个单位槽位使用四态状态机：

```text
STAGED -> PENDING -> ACTIVE -> TERMINAL
   ^          |
   +----------+  （命令拒绝或超时）
```

- `STAGED`：尚未部署/发射。
- `PENDING`：激活命令已经提交，等待同步回执；当前适配器通常在同一步确认，所以该态一般不出现在下一帧。
- `ACTIVE`：已经发射并参与仿真。
- `TERMINAL`：已经死亡、不可继续控制，或团队回合已经结束。

### 4.1 激活即部署并发射

对 `STAGED` 单位，策略一次采样：

- `activate ∈ {0,1}`
- `deploy_x ∈ [-1,1]`
- `deploy_y ∈ [-1,1]`
- `objective_slot ∈ [0,O-1]`
- `movement ∈ {left, hold, right}`

若 `activate=1`，适配器把归一化部署坐标映射到场景区域，先调用 `modify_simulator_position`，再在同一 `engine.step` 的命令列表中发送发射和机动命令。未激活时，其余激活参数不进入有效 log-prob。

采样顺序为 `activate → objective → placement/movement`。目标槽先通过合法 mask
采样，随后将该目标的槽位 embedding 与单位 Actor 特征融合，以条件化的
高斯分布生成部署坐标，并以条件化的三分类分布生成首次机动。

### 4.2 发射后动作

对 `ACTIVE` 单位，策略可输出：

- `retarget ∈ {0,1}`
- `objective_slot ∈ [0,O-1]`
- `movement ∈ {left, hold, right}`

部署坐标和再次激活动作被 mask。再指派目标只允许选择当前团队合法目标；当前目标会从可再指派槽位中排除。
若 `retarget=1`，先采样新目标，再用新目标的 embedding 条件化本步
movement；保持目标时使用普通 movement 头。因此“换到哪个目标”可以立即影响机动，
不需要等下一帧才反映在策略输入中。

卫星是独立的团队 pointer/STOP 分支。`team_global` 允许它选择步首的
`STAGED` 或 `ACTIVE` 槽作 requester，因此第一步可以把部署、发射、首次机动和
全局卫星请求放在同一次 Engine step。旧 `per_unit` 语义只允许步首已是
`ACTIVE` 的 requester。

### 4.3 机动映射

三值机动映射为单轴过载命令：

```text
left  -> -a_max
hold  -> 0
right -> +a_max
```

网络输出的是三分类分布，避免连续动作在环境中被硬阈值切分。

## 5. 当前观测编码

### 5.1 总体形状

策略每步接收一个固定顺序的单位向量 tuple；堆叠后的逻辑形状为：

```text
[U_actual, D]
D = 40 + 19 * O
legacy/final24: O=18, D=382
final20 easy/medium/hard: O=24/36/48, D=496/724/952
```

每个向量只包含合法红方信息和团队融合航迹。所有归一化值都会裁剪到约定范围，未知值同时由 `known`/`valid` 位区分。

### 5.2 全局字段：5 维

每个单位向量开头重复同一份团队全局字段：

| 字段 | 维度 | 归一化 |
|---|---:|---|
| `time` | 1 | `step / configured_max_steps` |
| `time_remaining` | 1 | `(configured_max_steps-step) / configured_max_steps` |
| `sensor_available` | 1 | 团队剩余额度 / 团队总额度 |
| `sensor_pending` | 1 | 本步待确认请求数 / 团队总额度 |
| `sensor_ready` | 1 | 协调层和 backend 都允许新请求时为 1 |

新 `team_global` 后端由 `SimulatorFactory` 为全队维护 100 次额度和默认
3 分钟活动窗口。窗口激活或额度耗尽时，`sensor_ready=0` 且所有 requester
均被 mask。旧 `per_unit` 后端仍按每枚导弹的独立额度和窗口计算。
`--sensor-capacity` 只能在真实 backend 上限内进一步收紧协调容量。

### 5.3 相位、自身与历史字段：23 维

| 字段 | 维度 | 含义或归一化 |
|---|---:|---|
| `phase_one_hot` | 4 | `STAGED/PENDING/ACTIVE/TERMINAL` |
| `position_known` | 1 | 自身位置是否可用 |
| `x, y` | 2 | 经度/纬度按场景 `mapArea` 线性归一化到 `[-1,1]` |
| `altitude` | 1 | 在 `[0,40000]` 范围线性映射到 `[-1,1]` 后裁剪 |
| `velocity_known` | 1 | 自身 ENU 平面速度是否可用 |
| `vx, vy` | 2 | ECF 速度转换为本地 ENU 后除以 `max_speed=3000` |
| `health` | 1 | 自身健康度裁剪到 `[0,1]` |
| `visible` | 1 | 自身 `isVisible` |
| `activation_age` | 1 | 激活后步数 / 场景时限 |
| `objective_change_age` | 1 | 上次目标改变后的步数 / 场景时限 |
| `target_progress_reference` | 1 | 当前距离势函数的参考距离 / 地图对角线 |
| `target_progress_fraction` | 1 | `(d_ref-d)/d_ref`，裁剪到 `[-1,1]` |
| `last_movement_one_hot` | 3 | 上次左/停/右动作 |
| `unit_type_one_hot` | 3 | 高速/中速/低速 |

`STAGED` 单位的自身位置保持未知，`position_known=0`，自身位置载荷为零。第一次进入 `ACTIVE` 的帧没有可靠速度，`velocity_known=0`；获得连续 ACTIVE 位置后才提供速度。后端 ECF 速度在进入网络前转换为本地 ENU，避免把坐标系分量误当经纬方向速度。
进度两维与训练中的 PBRS 内部状态一致，让 motion Critic 能区分“相同当前距离、
不同分配起点”。未处于有效 `ACTIVE+已分配目标` 状态时两者为零。

### 5.4 威胁摘要：11 维

当前编码保留该单位隔离观测中的探测总数和最近一个拦截弹航迹：

| 字段 | 维度 | 含义或归一化 |
|---|---:|---|
| `threat_count` | 1 | `team_global` 除以场景初始敌方拦截弹数；`per_unit` 仍为 `min(count,32)/32` |
| `threat_known` | 1 | 最近威胁是否存在 |
| `rel_x, rel_y` | 2 | 相对场景经纬跨度归一化 |
| `distance` | 1 | 平面距离 / 场景对角线 |
| `bearing_sin, bearing_cos` | 2 | 相对方位角 |
| `velocity_known` | 1 | 威胁 ENU 速度是否可用 |
| `vx, vy` | 2 | ENU 平面速度 / `max_speed` |
| `age` | 1 | 航迹年龄 / `max_track_age_steps=300` 后裁剪 |

这里的航迹源是本单位隔离观测。新 `team_global` 卫星在活动时探测全部
存活且由环境标记为可见的 24000 型拦截弹，Engine 将它们共享给每个红方导弹群。适配器丢弃航迹年龄
大于 `max_track_age_steps` 的记录。团队均值池化让其他单位的威胁摘要可以影响当前单位
决策，而不读取蓝方完整状态。

### 5.5 当前目标：`O+1` 维

当前目标用 one-hot 编码：`O` 个目标槽位加 1 个“无当前目标”类别。

### 5.6 每个目标槽位：18 维

| 字段 | 维度 | 含义或归一化 |
|---|---:|---|
| `valid` | 1 | 当前可被动作选择 |
| `known` | 1 | 已有公开或合法探测位置 |
| `rel_x, rel_y` | 2 | 相对参考点按经纬跨度归一化 |
| `distance` | 1 | 平面距离 / 场景对角线 |
| `bearing_sin, bearing_cos` | 2 | 相对方位角 |
| `velocity_known` | 1 | 目标 ENU 速度是否可用 |
| `vx, vy` | 2 | ENU 平面速度 / `max_speed` |
| `age` | 1 | 航迹年龄 / `max_track_age_steps=300` 后裁剪 |
| `type_one_hot` | 3 | `9400/9600/9500` |
| `assigned_total` | 1 | 以全部红方单位数为分母的当前 `ACTIVE` 分配比例 |
| `assigned_high` | 1 | 以全部 H 型单位数为分母的当前 `ACTIVE` H 分配比例 |
| `assigned_medium` | 1 | 以全部 M 型单位数为分母的当前 `ACTIVE` M 分配比例 |
| `assigned_low` | 1 | 以全部 L 型单位数为分母的当前 `ACTIVE` L 分配比例 |

对 `ACTIVE` 单位，目标几何以自身位置为参考。对 `STAGED` 单位，自身位置尚不存在，目标几何改用场景 `mapArea` 中心为参考；因此待部署策略在首步就能根据合法目标方向和距离选择部署点。目标速度同样先从 ECF 转为本地 ENU。

源时间戳必须严格增加才会刷新航迹；重复时间戳不会把陈旧数据错误地伪装为新探测。目标航迹年龄按当前步与最近一次被接受更新所在步之差计算。

`assigned_*` 是“显式目标负载”。它们只根据 lifecycle tracker 中我方 `ACTIVE`
单位的当前目标计数，因此会随激活回执、换目标和死亡自动更新。它们表达己方计划中的
火力分布，不读取敌方血量，也不作为硬性容量上限。

### 5.7 不可观测值约定

- 连续值未知时载荷为 `0`，同时对应 `known=0`。
- 目标未知时 `valid=0`、`known=0`，其几何、速度和年龄均为 `0`。
- 空目标槽位始终全零且不可选择。
- `TERMINAL` 单位保留固定槽位，动作只允许 no-op。
- 不把目标 ID 当连续数值输入，防止网络学习无意义的 ID 大小关系。

## 6. 动作 mask

mask 在采样前进入分布，非法动作不参与采样和 log-prob。

### 6.1 STAGED

- `activate={0,1}`。
- 若不激活，其余动作对环境无效。
- 若激活，目标只允许团队全局合法目标槽位。
- 激活时同时采样部署坐标和初始机动。
- 再指派动作不可用。`team_global` 的卫星 requester 可以选该 STAGED 槽；
  `per_unit` 的卫星请求仍不可用。

### 6.2 ACTIVE

- 再次激活和部署坐标不可用。
- 当前合法团队目标可以用于首次目标或再指派。
- 再指派时排除当前目标，避免无效重复命令。
- 左/停/右机动均可用。
- 卫星请求必须同时满足当前 backend 剩余额度大于零、当前不在对应
  卫星激活窗口、且本步未超过请求上限。`team_global` 的窗口会屏蔽全队。

### 6.3 PENDING、TERMINAL 和未来 padding

`PENDING` 和 `TERMINAL` 槽位只有 no-op 合法。未来加入跨场景 padding 后，padding 槽也必须使用同样的 `TERMINAL + no-op` 约定，并从 Actor/Critic 损失及团队池化中排除。

## 7. 卫星资源

卫星控制不单独训练网络。请求通过某个 `STAGED` 或 `ACTIVE` 导弹携带的命令接口发出，
但当前 `pku@32c08cd` 在 `SimulatorFactory` 中统一处理：

- `red_sat_use_count / red_sat_max_use_count`：全队共享，默认上限 100；
- `is_using_satellite()`：全局活动窗口，默认每次 3 分钟；
- 活动时由 `9202` 卫星探测全部存活可见的 `24000` 拦截弹，Engine 将航迹
  共享给所有红方导弹群；
- 卫星活动期间，H 型 `21000` 对 `9400/9600` 的命中率提升到 100%。

联合适配器把这种后端记为 `team_global`。请求后通过工厂计数器的前后差
确认接受；活动窗口内所有 requester 均被 mask，避免重复消耗额度来重置同一窗口。
不显式给出 `--sensor-capacity` 时，协调容量使用真实工厂上限；显式值只能更小。
新后端无论配置如何，有效每步上限都为 1。

为复现旧模型，适配器仍支持 `per_unit` 后端：它逐导弹读取旧版
`RED_SAT_USE_COUNT / RED_SAT_MAX_USE_COUNT / is_using_satellite()`，并用请求导弹的
计数增量确认回执。训练合同会区分 `team_global` 与 `per_unit`，不允许将两者
的 checkpoint 混用。

`team_global` 请求的 executor 身份不影响全局效果，因此 STAGED 槽可在自己
部署/发射的同一步作 requester。`per_unit` 的额度附着在单枚导弹上，仍需等到
后续 ACTIVE 步才可申请。

## 8. 奖励

### 8.1 官方分差

基础任务信号来自官方评分增量：

```text
team_reward_t = reward_scale * max(score_t - score_(t-1), 0) / 100
```

官方评分按新增毁伤计算，范围为 0 到 100，目标权重为目标舰 5、阵地 2、
无人船 1。回合 reset 后重新建立评分锚点，避免跨回合差分。后续三个奖励通道
以它为共同的任务目标，但分别向规划、机动和卫星分支提供适合其时间尺度的 credit。

### 8.2 规划专用终局 credit

激活、部署位置、首次目标和再指派共用 plan 分支。对本局曾被仿真回执确认
激活的单位，局末直接构造未折扣的规划目标：

```text
raw_local_credit_i = 按目标责任分配的守恒份额
learning_local_credit_i = clip(eligible_unit_count * raw_local_credit_i, 0, 1)
plan_return_i = 0.7 * final_official_score/100
              + 0.3 * learning_local_credit_i
```

默认权重由 `--planning-team-weight 0.7` 和 `--planning-local-weight 0.3`
控制，两者必须之和为 1。从未成功激活的单位 plan target 为零。直接使用终局
return，而不在完整 1200/3000 步 horizon 上再做 GAE，可避免部署与终局毁伤之间的长时域折扣衰减。

原始局部 credit 先将每个目标的 `weight * damage_fraction / total_weight`
作为可分配总量，再按控制器记录的单位—目标参与度按比例分配。当前仿真没有
可靠的命中归因回执，所以实现使用合法的己方目标分配持续时间作为 fallback
责任量。终局步使用环境将全队统一转为 `TERMINAL` 之前保存的
`assignment_states` 快照，不会丢掉最后一步的目标责任。对任一目标，分给所有单位的局部 credit
不超过该目标的官方得分贡献；
没有参与者的部分明确记为 unallocated，不会广播给无关单位。目标终局血量只用于训练
target 和诊断，不进入 Actor/Critic 观测。

为了使局部项与广播到每个合格单位的 team score 保持可比量级，进入
plan target 前将每个单位的 raw credit 乘以本局 `eligible_unit_count`，再裁剪到 `[0,1]`。
这份缩放并裁剪的 learning copy 不是全局守恒量；只有未缩放的 raw credit
满足按目标守恒，并在报告中
单独保存 allocated/unallocated 数据，便于审计。

同一单位在一局中可产生大量 `activate=NO` 或 `retarget=NO` 规划样本。训练时按
unit-episode 对 plan 样本做逆次数加权，再在整个 rollout 上将平均有效权重归一为 1；
该权重同时用于 plan policy loss、plan entropy 和 plan advantage 归一化。因此每个单位每局
贡献相同总规划权重，不会因为等待或存活更久而在 plan PPO 中自动占更大比例。

### 8.3 机动奖励

movement 分支使用逐步官方分差加距离 PBRS：

```text
d_ref = max(激活或换目标时的距离, 25 km)
p_i,t = clip((d_ref - d_i,t) / d_ref, -1, 1)
motion_reward_i,t = team_reward_t
                    + 0.05 * (gamma * p_i,t+1 - p_i,t)
```

换目标、死亡或团队真实终止时会闭合旧势函数，防止凭空兑现进度奖励。
motion 分支以自己的 Critic 计算逐步 GAE，不再与激活、部署和目标选择的
log-prob 合并成一个 PPO ratio。观测中显式给出 `d_ref` 的归一化值和 `p_i,t`，
使价值函数能看到这个奖励内部状态。

### 8.4 卫星信息奖励

卫星 pointer/STOP 分支使用：

```text
sensor_reward_t = team_reward_t + gamma * Phi_info(s_t+1) - Phi_info(s_t)
```

`Phi_info` 必须与卫星 backend 的真实功能对齐：

```text
team_global:
  freshness_j = max(0, 1 - age_j / max_track_age)
  Phi_info(s) = 0.015 * min(1, sum_j freshness_j / initial_interceptor_count)

per_unit (legacy):
  Phi_info(s) = 0.015 * (已合法知道位置的目标权重 / 所有计分目标权重)
```

`team_global` 中的 `j` 遍历受控红方合法 `detectInfo` 中按 ID 去重的 24000 型
航迹；同一 ID 取最新时间戳，丢弃超过 `max_track_age` 的记录。分母是场景
初始敌方拦截弹数，只作为 reward 与 Actor `threat_count` 的归一化分母，
不把隐藏坐标或血量送入 Actor/Critic。
`per_unit` 的旧目标势函只为准确复现旧后端保留。真实终止时以下一势函为零闭合；
debug truncation 保留下一状态势函和 value bootstrap。默认 scale 为 `0.015`，
暂不另外引入卫星请求成本。

### 8.5 三路 PPO 信号

plan、motion、sensor 分别保存 value、reward/return 和 advantage，分别求自己的联合
log-prob ratio。三路分开计算 policy loss、value loss、entropy、approximate KL 和
clip fraction，优化时再汇总总损失。这样可以区分“目标/部署规划出问题”、
“飞行机动出问题”和“卫星调度出问题”，不再只有一个混合的 unit advantage。

## 9. 回合结束、截断与 bootstrap

### 9.1 正式场景时限

正式场景使用 `case_info` 和 `reward_policy.max_steps` 给出的完整 horizon；legacy
E01 为 1200 步，final20 为 3000 步。代码不根据套件名称推断或硬编码时限。

到达该正式 horizon 是任务真实结束，返回 `terminated=True`、`truncated=False`。
motion 和 sensor GAE 在该边界使用零 bootstrap；plan 直接使用已完结的整局 return。

### 9.2 调试短时限

`max_steps_override` 只用于比正式 horizon 更短的 smoke/debug 回合。到达该短时限且没有先发生真实终止时，返回：

```text
terminated=False
truncated=True
```

motion 和 sensor GAE 的一步 TD 误差使用各自的 `V(next_obs)` bootstrap，
同时在该边界停止向更早轨迹继续递归优势。这样短回合不会把仍有价值的状态误当作零价值终态。
由于 debug 回合没有真正的终局任务结果，其 plan credit 只适合检查数据通路，不应用来
判断完整 horizon 的规划质量。

### 9.3 其他真实终止

以下情况返回 `terminated=True` 并零 bootstrap：

- 仿真报告自然结束。
- 配置启用 `terminate_on_all_objectives_destroyed` 且官方任务目标已经全部完成。
- 正式 horizon 到达。

团队真实终止时所有尚未终态的单位一起闭合为 `TERMINAL`。单位在此前死亡时可先独立进入 `TERMINAL`，不再贡献后续 Actor/Critic 样本。

## 10. 网络输入与跨场景限制

每个单位的 `D=40+19*O` 维向量先经过共享实体编码器。Actor 将本单位编码与所有
实际单位编码的均值团队上下文拼接。激活/再指派分支先产生目标分布，选中目标的
embedding 再通过 target-condition encoder 与 Actor 特征融合，产生部署位置和
目标相关的首次/换目标机动。三个 Critic 分别预测 plan 终局 return、motion
逐步 return 和 sensor 团队 return。三者都使用同源非特权特征和团队上下文。

当前环境的 `JointSpace.unit_count` 等于场景实际单位数，策略和 checkpoint 元数据会校验这个精确容量。
网络输入层还绑定 `O` 与 `D`，因此 382/496/724/952 维的 checkpoint 互不兼容。

要支持一个 checkpoint 跨场景训练，仍需：

1. 统一固定 `U_MAX=420` 和 `O_MAX=48`。
2. 小场景在单位维和目标维末尾补 padding 槽。
3. 单位 padding 槽使用 `TERMINAL` 观测和仅 no-op 的 mask，目标 padding 始终 invalid。
4. Actor 团队均值和 Critic 团队池化改为 masked mean，排除 padding。
5. PPO 的 policy/value/entropy 统计排除 padding。
6. checkpoint 记录并校验统一后的 `U_MAX/O_MAX` 与 padding 协议。

在这些步骤完成前，应按场景分别创建 policy 和 checkpoint。

## 11. 仿真接口事实与边界

以下限制来自仿真接口，适配器已经按当前可用信息处理，但后续改后端时需要重新验证：

1. 普通 `TrainingEnv.step` 没有公开的动态部署动作；当前必须在 `engine.step` 前调用 `modify_simulator_position`。
2. 部署、发射、再指派、机动和卫星 handler 没有统一正式回执。当前以“命令无异常且模拟器存在”为同步接受条件，并在步后校验发射内部状态、卫星计数差等可见字段。
3. 发射和再指派接口传目标坐标，不传目标 ID；目标槽位到坐标的解析必须由适配器维护。
4. `STAGED` 实体在后端可能仍是 `isVisible=True`；策略状态必须以适配器状态机为准，不能仅以可见性判断是否已发射。
5. `DetectInfo` 没有目标健康和权威存活字段，所以 Actor 不编码目标血量；任务得分由正式 scoring 路径计算。
6. 后端复位未必重置各模拟器的 `sim_time`；当前环境沿用时钟快照恢复，后续新增模拟器类型时需要覆盖 smoke test。
7. 当前部署区域只接受严格轴对齐矩形：`_bbox` 会验证输入点集合恰好等于矩形四角，空区域、零面积或非矩形区域都会立即抛错并停止初始化。通过验证后才把 `[-1,1]^2` 线性映射到该矩形，因此不会静默把任意多边形退化为包围盒。

## 12. PPO rollout、checkpoint 与报告

默认 `--rollout-episodes 4`。策略参数在这一组完整回合采集期间保持冻结，随后将轨迹
拼接并做一次 PPO 更新。motion/sensor GAE 会在每个回合的 terminated/truncated
边界停止递推，plan 则按回合计算直接终局 return，所以
拼接不会把一个回合的回报传播到下一个回合。若总轮数不是 4 的整数倍，最后的完整
回合也会作为较小批次更新，不会丢弃。

`best.pt` 在 PPO 更新前保存，因此它精确对应产生该最佳正式得分的行为策略。
`latest.pt` 和周期 checkpoint 只在完整 rollout 更新后保存。训练中断时，已采集但尚未
更新的部分 rollout 会记录在状态文件里，但不会伪装成已经训练进 checkpoint 的样本。
SIGINT/SIGTERM 若在多个 minibatch 的 PPO 更新中到达，会先记下停止请求，待这次更新、
报告和安全 checkpoint 完整提交后再退出。异常若使 PPO 只更新了一部分，则不会写出
`failed.pt` 来冒充一致状态。

结果目录包含：

- `rounds.csv` 与 `rounds.jsonl`：逐回合得分、plan/motion/sensor return、rollout
  归属，以及激活、目标、部署、机动和卫星请求诊断。JSONL 还保留每目标的
  毁伤贡献、已分配/未分配局部 credit、分配持续时间和每单位 plan target。
- `updates.csv` 与 `updates.jsonl`：逐 PPO 更新记录覆盖的回合和样本步数，并对
  plan、motion、sensor 分别记录 policy/value loss、entropy、approximate KL、
  clip fraction、advantage 摘要和有效决策数。
- `run_config.json`、`status.json`、`training.log`：可复现配置、状态和完整日志。

仿真器运行时需要的源码符号链接及原生空文件位于
`personal_train/.runtime/training_runs/`，不属于实验结果。正式结果目录只保留报告和
训练产物的引用信息。

checkpoint 策略 schema 为 `v3`。旧 `per_unit` 后端保留环境合同 v3；
当前 `team_global` 后端使用环境合同 v4，sensor 部分明确写入：

- `backend=team_global`；
- `backend_capacity_team=100`；
- `backend_active_minutes=3`；
- `effective_max_requests_per_step=1`；
- `observation.detected_threat_count_normalizer=场景初始敌方拦截弹数`。

合同同时绑定场景哈希、单位槽、目标槽、观测宽度和 sensor reward source。
因此旧 per-unit checkpoint 不能在新 team-global 环境上 `--resume` 或
`--init-from`；不同 `O/D` 的 final20 checkpoint 也不兼容。需要复现旧模型时，
应将 `COMPETITION_REPO_ROOT` 指向保留的 `glibc-2.38/runtime/pku` 旧快照。

只有带 `resume_safe=true` 的 `latest.pt` 和周期 checkpoint 可用于
`--resume`；它会恢复优化器、计数器、策略 RNG，以及原 seed、蓝方策略、rollout 大小、
debug horizon 和游戏配置。新进程不能恢复蓝方对象内部已经推进的随机流，因此续跑保持
训练配置和策略状态一致，但不是整个仿真进程的逐位重放。

### 12.1 固定多种子评估

[`eval_joint_ppo.py`](eval_joint_ppo.py) 不更新策略，会对 checkpoint 与场景完整契约
做严格校验，并对每个种子重建一个仿真器。例如用固定种子做确定性评估：

```bash
source /home/amax/ry/competition/personal_train/activate_competition_rl.sh
export COMPETITION_REPO_ROOT=/home/amax/ry/competition/competition_envs
cd /home/amax/ry/competition

/home/amax/ry/competition/glibc-2.38/python3.11-glibc238 \
  personal_train/eval_joint_ppo.py \
  --checkpoint /home/amax/ry/competition/personal_train/models/<run>/best.pt \
  --scenario /home/amax/ry/competition/competition_envs/scenarios/cases/easy/E01/scenario.json \
  --seeds 11 23 37 53 71 \
  --episodes-per-seed 1 \
  --deterministic \
  --device cuda:0 \
  --run-id e01_joint_v3_eval_det
```

要测量采样策略的方差，使用相同的 checkpoint、场景和种子，将 `--deterministic`
替换为 `--stochastic`，并提高 `--episodes-per-seed`。每次评估在 `results/<eval-run>/`
写入 `per_episode.csv/jsonl`、`per_seed.csv`、`aggregate.json`、
`evaluation_config.json`、`status.json` 和 `evaluation.log`。`aggregate.json` 同时给出所有
episode 得分及各 seed 平均得分的 mean/std/median/quantile/lower-CVaR，可用于
比较 best/latest 或不同训练设置。确定性和随机评估是两个独立 run，不应写入同一目录。
省略 `--seeds` 时使用预置且固定的 `1..10`。

### 12.2 final20 多场景调度

`train_joint_multi_scenario.py` 用 `--suite legacy|final24|final20` 选择套件，默认仍为
legacy。省略 `--scenarios` 时分别枚举 9/24/20 个场景；也可以只选部分 final20：

```bash
export COMPETITION_REPO_ROOT=/home/amax/ry/competition/competition_envs
/home/amax/ry/competition/glibc-2.38/python3.11-glibc238 \
  personal_train/train_joint_multi_scenario.py \
  --suite final20 \
  --scenarios E01 M06 H08 \
  --rounds 100 \
  --gpu-ids 1,2,3 \
  --max-parallel 3 \
  --dry-run
```

调度器会将这三个 ID 解析为 `final20/easy/E01`、`final20/medium/M06` 和
`final20/hard/H08`，非 legacy 的输出名包含 suite 前缀。续训会严格校验 suite、
selector 和绝对场景路径。确认 dry-run JSON 后删除 `--dry-run` 才启动训练。

## 13. 冒烟训练判定

新架构已通过回归测试和短时限真实训练/评估 smoke。不应把短 smoke
的得分当成最终仿真结果；完整 horizon 和 200 回合效果仍需实际训练确认。关键通路包括：

- `STAGED` 单位以部署区中心计算目标几何，不再丢失首步目标方向。
- 自身和目标 ECF 速度转换到本地 ENU；首个 ACTIVE 帧标记速度未知。
- 航迹只接受严格更新的源时间戳，陈旧重复数据不会续期。
- 卫星按 `team_global/per_unit` 后端的各自额度、窗口和回执语义记账。
- Actor 使用本单位特征与团队池化上下文。
- 目标负载只统计己方 ACTIVE 分配，并随激活、换目标和死亡更新。
- 部署与首次机动以已选目标为条件，换目标后本步机动以新目标为条件。
- plan/motion/sensor 的 value、advantage、ratio 和诊断完全分开。
- 目标 mask 采用集中合法探测共享，未探测隐藏目标仍不可选择。
- 正式 horizon 为真实终止并零 bootstrap；调试短 horizon 为截断并 bootstrap。

最小 smoke 应验证：

1. reset 后形状严格为 `[U_actual,40+19*O]`，所有值有限；final20
   easy/medium/hard 分别校验 496/724/952 维。
2. `STAGED` 单位自身位置未知，但对公开目标已有相对部署区中心的几何。
3. 激活命令同一步完成位置修改、发射和初始机动，随后进入 `ACTIVE`。
4. 首个 ACTIVE 帧 `velocity_known=0`，后续可用帧才为 1。
5. 任一红方合法探测隐藏目标后，全队对应目标 mask 变真；探测前保持假。
6. `team_global` 在首步可用 STAGED requester 与部署/发射同步，请求只使
   工厂全局计数加一，活动窗口内全部 requester 被 mask 且 `sensor_ready=0`；
   `per_unit` 仅允许 ACTIVE requester，仍只扣请求实体的旧额度。
7. 官方得分增加时团队奖励为对应差分，reset 后不存在跨回合奖励尖峰。
8. raw 目标 credit 的 allocated+unallocated 与每目标官方贡献守恒，学习副本的
   `eligible_unit_count` 尺度校正被单独标记。
9. `team_global` 的新鲜 24000 航迹使 sensor 势函按年龄和初始拦截弹数归一化增加，
   陈旧航迹衰减/过期；`per_unit` 仍按合法目标权重增加。两者在真实终止时正确闭合。
10. 正式 horizon 返回 terminated 且 bootstrap 为零；短 debug horizon 返回 truncated 且使用下一状态价值。
11. 死亡/终态单位只产生 no-op，不进入后续单位 PPO 样本。

跨场景固定 `U_MAX/O_MAX` 和 padding 是当前明确保留的结构工作。当前可按单一
场景运行 smoke 和正式训练，完整 horizon 与 200 回合效果需要通过实际训练和固定多种子评估确认。
