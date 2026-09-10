# 联合强化学习仿真适配设计

本文只描述 `personal_train` 如何适配只读的 pku 仿真环境。当前训练与审计对象是：

```text
/home/amax/ry/competition/competition_envs  (pku@32c08cd)
```

`COMPETITION_REPO_ROOT` 应指向上述目录。
`/home/amax/ry/competition/glibc-2.38/runtime/pku` 是旧 pku 快照，只用于
复现旧 checkpoint 的 per-unit 卫星语义，不代表当前 live 环境。

证据标记如下：

- **[源码已验证]**：结论可以直接从当前 pku 源码得到。
- **[运行已验证]**：已用本地 glibc 2.38 Python 包装器实例化环境并核对运行对象，未推进仿真。
- **[强制仿真已验证]**：已在原生 Engine 上构造确定动作，实际推进仿真并核对回执或 Simulator 状态。
- **[设计决定]**：建议在 `personal_train` 中采用的行为，不代表上游环境已经提供此接口。
- **[待验证]**：仅凭源码不能确认原生模型在该用法下长期稳定，必须用烟雾测试确认。

## 最终实现状态（以本节为准）

> **当前可训练实现已经确定，不再采用早期的 proxy Agent 方案。** 后文保留上游接口审计依据；若历史建议与本节冲突，以本节和当前代码为准。

- 红方由 [`JointGameEnv`](/home/amax/ry/competition/personal_train/joint_game_env.py:256) 直接编排。它包装只读 `TrainingEnv`，固定全部 `sideId == 0` 且类型为 `21000/21001/21002` 的单位槽，直接完成动作校验、部署副作用、命令适配、一次 `engine.step()`、回执、奖励和终止；不向 `AgentManager` 注册每弹 proxy，也不调用 `red_model_deploy()`。
- 蓝方不需要训练器提交动作。想定中的 `DefendCommanderModelSimulator` 在 Engine 内部接收雷达航迹、调用 `BLUE_POLICY` 选定的防御策略并发送拦截命令（[`DefendCommanderModelSimulator.py` L30–60](/home/amax/ry/competition/competition_envs/core/envengine/simulator/simlulator_impl/DefendCommanderModelSimulator.py:30)、[`DefendCommanderModelSimulator.py` L76–155](/home/amax/ry/competition/competition_envs/core/envengine/simulator/simlulator_impl/DefendCommanderModelSimulator.py:76)）。训练入口只在构造环境前设置 `BLUE_POLICY` 和种子（[`train_joint_ppo.py` L801–833](/home/amax/ry/competition/personal_train/train_joint_ppo.py:801)）。
- 红方采用**集中共享本方合法探测**：适配器只遍历受控红方导弹自身的 `detectInfo`，把其中已合法出现的计分目标并入团队目标槽；从未出现在红方合法观测中的隐藏坐标不会进入网络。所有单位共享这一团队航迹目录和目标 mask，这是明确的集中指挥假设（[`joint_game_env.py` L746–778](/home/amax/ry/competition/personal_train/joint_game_env.py:746)）。
- Critic 是**非特权 Critic**。plan/motion 逐弹 value 和 sensor 团队 value
  都由同一组合法红方编码观测产生；sensor value 只池化这些实体特征，
  不读取 `_get_observation()` 中未探测蓝方的完整真值。完整原始态势只用于仿真推进后的
  官方计分、奖励 target、终止判断和诊断。
- 卫星没有独立 Agent。它由同一联合策略的团队 pointer/STOP 头选择 requester。
  适配器在运行时识别两种后端：当前 `team_global` 后端的 100 次由
  `SimulatorFactory` 全队共享，活动窗口内屏蔽全部重复请求；旧 `per_unit`
  后端继续逐导弹检查计数和窗口。两种后端默认每步最多接受 1 个请求。
- 终止语义已经固定：正式想定 horizon 到达记为 `terminated=True`；仅 `--debug-max-steps` 产生的短 horizon 在没有更早真实终止时记为 `truncated=True`。自然结束和配置启用的“全部计分目标摧毁”也属于真实终止（[`joint_game_env.py` L662–678](/home/amax/ry/competition/personal_train/joint_game_env.py:662)）。
- **[强制仿真已验证]** 已绕过随机初始化偏置，分别强制选择 H/M/L，验证动态 placement、同一步 launch 和首个 maneuver 均由原生模型接受。
  旧后端的逐弹卫星回执也已实测；当前后端则以 `SimulatorFactory.red_sat_use_count`
  前后差确认团队请求。两种语义不得混用，checkpoint 环境合同会绑定 backend。

仍需保留的边界包括：staged Simulator 在上游从开局即存在，可能参与弹间通信；当前部署区域只接受轴对齐矩形；强制烟雾测试不能替代所有场景的完整 1200 步回归。详见第 11、12 节。

## 1. 结论

当前实现使用“一个共享联合策略 + 一个直接编排原生 Engine 的 `JointGameEnv`”。每枚红方导弹对应固定的策略槽，但不是一个注册到 `AgentManager` 的 Python Agent。卫星由同一策略的团队头调度，蓝方由 Engine 内的 DefendCommander 自行运行。

策略中的 `activate` 不是单独的部署阶段。它表示一个原子操作：在当前仿真步把该导弹放到策略选择的位置，并立即向策略选择的目标发射。`STAGED/PENDING/ACTIVE/TERMINAL` 只是控制器内部的动作掩码状态；其中 `PENDING` 只用于同一步执行回执，不是一个游戏阶段。

一次联合动作应覆盖全部红方导弹槽位：

```text
每枚导弹：activate(2) + placement(x,y) + objective(K)
          + retarget(2) + maneuver(left/stop/right)
全队协调：satellite requester pointer/STOP（默认每步最多 1 个）
```

Actor 读取全部受控红方单位特征、公开初始目标、红方合法探测的团队并集和控制器历史。
plan、motion、sensor 三个 Critic 与 Actor 共用这套非特权输入，只增加团队池化，
不读取隐藏蓝方真值。

## 2. Profile 与 TrainingEnv 的构造

### 2.1 最小构造链

**[源码已验证]** `Profile` 只有 `environmentProfile` 和 `imagineProfile` 两个 dataclass 字段（[`profile.py` L8–14](/home/amax/ry/competition/competition_envs/core/envengine/sdk/base_struct/profile/profile.py:8)）。本地场景的官方入口也是 `json.load()` 后调用 `Profile.from_dict()`（[`core/main.py` L102–123](/home/amax/ry/competition/competition_envs/core/main.py:102)）。适配器应沿用同一方式：

```python
scenario_path = Path(...).resolve()
profile = Profile.from_dict(json.loads(scenario_path.read_text(encoding="utf-8")))
reward_policy = load_reward_policy(scenario_path)
env = JointGameEnv(
    profile,
    reward_policy=reward_policy,
    render_mode=None,
)
```

`TrainingEnv(profile)` 会立即创建 `Engine`、全部 Simulator 和一个空 `AgentManager`（[`training_env.py` L29–55](/home/amax/ry/competition/competition_envs/core/envengine/environment/training_env.py:29)）。`Engine` 由 `simEndLogicTime // simStep` 得到总步数（[`engine.py` L31–46](/home/amax/ry/competition/competition_envs/core/envengine/engine/engine.py:31)）。最终构造会断言 `case_info.max_steps` 与 Engine horizon 一致；显式的更短 `--debug-max-steps` 只用于调试截断，不能覆盖正式任务定义。

**[最终实现]** `JointGameEnv` 采用组合而不是继承：内部创建 `TrainingEnv(profile)`，随后直接编排其 Engine。`train_joint_ppo.py` 每步从联合策略采样一次，再调用 `JointGameEnv.step(action)`；pku 文件保持不变（[`train_joint_ppo.py` L490–524](/home/amax/ry/competition/personal_train/train_joint_ppo.py:490)）。

### 2.2 当前 E01 的实际规模

**[运行已验证]** 使用当前训练脚本缩写 `E01` 对应的 `scenarios/cases/easy/E01/scenario.json` 实例化后得到：

| 项目 | 数值 |
|---|---:|
| 全部实体 | 323 |
| 红方 `21000` | 30 |
| 红方 `21001` | 44 |
| 红方 `21002` | 90 |
| 可控红方导弹总数 | 164 |
| `_get_init_ship_observation()` 返回的目标 | 5（3 个 `9400`、2 个 `9600`） |
| 环境步数 | 1200 |

场景元数据也记录了 323 个实体、164 枚红方导弹和 1200 步时限（[`case_info.json` L107–134](/home/amax/ry/competition/competition_envs/scenarios/cases/easy/E01/case_info.json:107)）。九个旧版 E/M/H 场景中最大可控规模为 304、最大计分目标数为 14；`final24` 目录中最大可控规模为 360、最大计分目标数为 18。新增
`final20` 按难度分别有 210/300/420 枚可控红方导弹和 24/36/48 个计分目标，
正式时限均为 3000 步。

训练器使用场景实际单位数 `N`；目标槽容量 `K` 在未显式指定时取
`max(18, len(RewardPolicy.objective_ids))`，多余槽由 mask 关闭。因此 legacy/final24
通常仍为 `K=18`，final20 easy/medium/hard 为 `K=24/36/48`。若网络必须跨场景
共用一个 checkpoint，还必须先给 `JointSpaceSpec`、mask、编码器和 trajectory
增加 `unit_present[N]` 及 padding 协议；当前核心不能直接用零填充冒充不存在的导弹。

## 3. 直接编排与实体槽位

### 3.1 发现全部红方导弹，但不注册 proxy Agent

上游示例会为 `21000/21001/21002` 创建 `AttackMissileAgent` 并注册（[`core/main.py` L215–231](/home/amax/ry/competition/competition_envs/core/main.py:215)）。这证明这些类型是红方动作执行实体，但最终联合实现不复用该 per-Agent 动作路径。

`JointGameEnv` 直接遍历 Simulator，要求 `sideId == 0` 且类型在上述集合内，再按“类型、实体 ID”稳定排序并建立 `unit_ids`、`unit_slot_by_id` 和 `unit_types`（[`joint_game_env.py` L312–352](/home/amax/ry/competition/personal_train/joint_game_env.py:312)）。策略内部使用从 0 开始的 `unit_slot`，原始 `entity_id` 只用于命令翻译、日志和回执。

场景中的“增援”也是开局就由 `entityList` 创建的静态 Simulator，并不是运行时动态新增实体；工厂初始化时会遍历全部 `entityList`（[`simulator_factory.py` L51–67](/home/amax/ry/competition/competition_envs/core/envengine/simulator/simulator_factory.py:51)）。因此构造时的一次扫描会包含增援单位。

### 3.2 为什么最终绕过 AgentManager

上游 `collect_actions_from_agents()` 会先对各 Agent 填观测、再逐个调用 `get_action()`，并在单 Agent 出错时捕获异常后继续（[`agent_manager.py` L92–121](/home/amax/ry/competition/competition_envs/core/envengine/agent_manager/agent_manager.py:92)）。它还按实体只记录第一条动作，无法完整保存同一步 launch、maneuver 和 satellite 组合。联合 PPO 需要一次采样、一份不可变 mask/trace，以及整组命令共同推进一次 Engine，所以 `JointGameEnv` 直接执行完整事务更清楚。

早期 proxy 方案还需要绕开 `agent_id=0` 的映射缺陷：`AgentManager.get_agent_by_entity()` 使用真假判断，ID 0 会被误判为不存在（[`agent_manager.py` L40–45](/home/amax/ry/competition/competition_envs/core/envengine/agent_manager/agent_manager.py:40)）。最终实现完全不建立这层 ID，因而不存在该问题。

最终实现不注册 `AttackMissileAgent`、`DeployAgent` 或卫星 Agent，也不调用 `TrainingEnv.red_model_deploy()` / `TrainingEnv.step()`。它仅复用 `TrainingEnv.reset()`、观测生成和底层 Engine；红方命令直接经 `CommandAdapter` 送入 Engine。蓝方 DefendCommander 作为原生 Simulator 随 Engine 自动更新。

## 4. 策略动作与引擎动作的精确映射

### 4.1 策略层动作

对单位槽 `i`：

| 分支 | 形状 | 条件 |
|---|---|---|
| `activate[i]` | categorical(2) | `STAGED` 时有效 |
| `placement[i]` | 连续 2 维，范围 `[-1,1]²` | `STAGED && activate=YES` |
| `objective[i]` | categorical(K) | 激活，或 `ACTIVE && retarget=YES` |
| `retarget[i]` | categorical(2) | `ACTIVE` 时有效 |
| `maneuver[i]` | categorical(3) | `ACTIVE`，以及本步 `activate=YES` 时有效 |
| `shared_sensor.requester_slots` | 团队 pointer/STOP 序列 | `team_global` 允许步首 `STAGED/ACTIVE` 槽，`per_unit` 只允许 `ACTIVE`；当前 backend 额度与窗口还必须合法 |

规划分支的采样是自回归的。`STAGED && activate=YES` 先选 objective，
再将目标槽 embedding 与当前单位 Actor 特征融合，从条件分布采样 placement
和首次 movement。`ACTIVE && retarget=YES` 也先选新 objective，本步 movement
使用新目标的条件分布。评估 log-prob 时使用已存储动作以同样顺序重建这些分布。

`maneuver` 的三值定义固定为：

```text
0 = left     -> acc_z = -1.0
1 = stop     -> acc_z =  0.0
2 = right    -> acc_z = +1.0
```

**[最终实现]** 激活步同时采样并执行 `maneuver`。引擎会在所有 Simulator 更新前同步分发整组 AI 指令（[`engine.py` L48–58](/home/amax/ry/competition/competition_envs/core/envengine/engine/engine.py:48)），因此“发射后首个机动”在同一个 `simStep` 生效；`joint_rl_core` 已把该分支纳入 mask、intent、branch activity 和 PPO log-prob。

### 4.2 普通动作数组

`CommandConverter` 只识别下列 0–3 四元动作（[`command_converter.py` L11–62](/home/amax/ry/competition/competition_envs/core/envengine/environment/command_converter.py:11)）：

| 意图 | `np.float64` 动作行 | 对应 action class | 引擎 command id |
|---|---|---|---:|
| 左/停/右机动 | `[0, entity_id, -1/0/+1, 0]` | `SetDesiredAccZ` | 3007 |
| 发射 | `[1, entity_id, target_lon, target_lat]` | `MissileLaunchAction` | 200 |
| 修改目标 | `[2, entity_id, target_lon, target_lat]` | `ChangeTargetAction` | 3014 |
| 请求卫星 | `[3, entity_id, 0, 0]` | `UseSatelliteAction` | 3013 |

具体 dataclass 字段可见：

- 发射：`executor_id` 与 `target: Vector3d`（[`missile_launch.py` L11–25](/home/amax/ry/competition/competition_envs/core/envengine/agent_manager/actions/aircraft_action/missile_launch.py:11)）；
- 改目标：相同的坐标字段（[`change_target.py` L11–25](/home/amax/ry/competition/competition_envs/core/envengine/agent_manager/actions/aircraft_action/change_target.py:11)）；
- 机动：`acc_z`（[`set_desired_acc_z.py` L11–24](/home/amax/ry/competition/competition_envs/core/envengine/agent_manager/actions/aircraft_action/set_desired_acc_z.py:11)）；
- 卫星：`requested=True`（[`use_satellite.py` L11–24](/home/amax/ry/competition/competition_envs/core/envengine/agent_manager/actions/aircraft_action/use_satellite.py:11)）。

发射和改目标接口不接受目标 ID，只接受经纬度；converter 还会把目标高度固定为 0（[`command_converter.py` L28–51](/home/amax/ry/competition/competition_envs/core/envengine/environment/command_converter.py:28)）。因此网络输出目标槽，适配器保存 `slot → target_id` 元数据，并在发送时用该槽当前合法的 `lon/lat` 转换成动作。原始 ID 不应作为连续数值输入网络。

### 4.3 placement 不能走普通动作 converter

**[源码已验证]** 原部署 Agent 返回的是五元数组：

```text
[0, entity_id, lon, lat, alt]       # 设置部署位置
[1, -1, 0, 0, 0]                   # 整体部署完成
```

格式和高度规则见 [`deploy_agent.py` L28–78](/home/amax/ry/competition/competition_envs/core/user_agents/deploy_agent.py:28)：`21000/21001` 使用 `coordinatesHM` 且高度 0；`21002` 使用 `coordinates` 且高度 10000。

这些五元动作只在 `red_model_deploy()` 的专用循环中解释，然后直接调用 `modify_simulator_position()`（[`training_env.py` L105–158](/home/amax/ry/competition/competition_envs/core/envengine/environment/training_env.py:105)）。普通 `CommandConverter` 中动作 0 已经表示侧向加速度，不能把五元部署动作混入 `step()`。

**[设计决定]** 对 `STAGED && activate=YES` 的单位，协调器按以下顺序执行：

1. 把 `placement ∈ [-1,1]²` 映射到该类型的合法部署多边形；
2. 调用 `simulator_factory.modify_simulator_position(entity_id, {x: lon, y: lat, z: alt})`；
3. 向当前引擎步加入 launch 行；
4. 紧接着加入 maneuver 行；
5. 对当前 mask 中另行选出的 requester 加入 satellite 行；`team_global`
   可以选本步同时激活的 `STAGED` 槽，`per_unit` 只允许步首已经 `ACTIVE` 的槽；
6. 所有单位处理完后，只调用一次 `engine.step(...)`。

当前位置修改方法会调用对应 Simulator 的 `set_lla()`（[`simulator_factory.py` L473–483](/home/amax/ry/competition/competition_envs/core/envengine/simulator/simulator_factory.py:473)）。它的成功路径没有显式 `return True`，实际返回 `None`；适配器不能用其返回值的真假判断成功。应先确认实体存在，并把无异常完成记录为 placement 成功。

部署区域是 GeoJSON 风格多边形。E01 的 `coordinatesHM` 和 `coordinates` 范围可见 [`scenario.json` L12948–12990](/home/amax/ry/competition/competition_envs/scenarios/cases/easy/E01/scenario.json:12948)。不要只对边界框 clamp 后假设点在任意多边形内；矩形可以线性映射，非矩形应做点内判断并投影或重采样。

### 4.4 命令顺序、回执和重复指令

`SimulatorFactory.process_ai_commands()` 按列表顺序处理命令：普通命令立即调用目标
Simulator 的 `command_received()`，卫星命令则交给工厂级 handler
（[`simulator_factory.py` L378–416](/home/amax/ry/competition/competition_envs/core/envengine/simulator/simulator_factory.py:378)）。
所以推荐顺序固定为：

```text
placement side effect -> launch -> retarget（仅既有 ACTIVE）-> maneuver -> satellite
```

H/M/L 的 launch handler 分别在收到发射命令后设置内部 launch 状态；重复发射只打印提示并返回，不抛异常（H：[`CompCruiseMissileHSimulator.py` L218–240](/home/amax/ry/competition/competition_envs/core/envengine/simulator/simlulator_impl/CompCruiseMissileHSimulator.py:218)，M：[`CompCruiseMissileMSimulator.py` L214–233](/home/amax/ry/competition/competition_envs/core/envengine/simulator/simlulator_impl/CompCruiseMissileMSimulator.py:214)，L：[`CompCruiseMissileLSimulator.py` L212–231](/home/amax/ry/competition/competition_envs/core/envengine/simulator/simlulator_impl/CompCruiseMissileLSimulator.py:212)）。因此生命周期 mask 必须阻止二次 launch。

环境没有统一的 launch、retarget、maneuver 或 satellite 回执。**[最终实现]** `engine.step()` 返回后，activation 通过 H/M 的 `launch != 0`、L 的 `launch >= 0` 确认；卫星回执按 backend 读取计数器前后差：
`team_global` 读 `SimulatorFactory.red_sat_use_count`，`per_unit` 读 requester Simulator
的旧版计数。适配器不把这些后端字段当作特权目标情报送入 Actor，
并在 `CommandAdapter` 丢弃任一联合命令时立即失败。

## 5. 动作条件掩码

推荐状态与分支如下：

| 状态 | 合法分支 |
|---|---|
| `STAGED, activate=NO` | activation；其余分支不计 log-prob |
| `STAGED, activate=YES` | activation、placement、objective、movement |
| `PENDING` | 无动作，等待同一步回执 |
| `ACTIVE, retarget=NO` | retarget、movement、satellite |
| `ACTIVE, retarget=YES` | retarget、objective、movement、satellite |
| `TERMINAL` | 全部 no-op |

最终 mask 已同时覆盖激活步 movement、生命周期 no-op、当前目标不可重复 retarget、
卫星 backend 额度/窗口和团队每步请求上限。`team_global` 一旦处于卫星活动窗口，
会屏蔽所有 requester，避免在同一窗口内浪费全队次数去重置结束时间。

目标 mask 使用全队一份 `objective_valid[K]`。其来源不是完整蓝方真值，而是公开初始目标与全部受控红方单位合法 `detectInfo` 的并集。一个单位发现的目标因此可由集中策略分配给任意红方单位；这是当前实现明确采用的团队指挥信息共享。上游 Engine 本身只在通信簇内融合探测（[`engine.py` L172–220](/home/amax/ry/competition/competition_envs/core/envengine/engine/engine.py:172)），所以若未来需要模拟断联，必须把当前团队 mask 改为 `objective_valid[N,K]`，不能把现实现误称为逐弹隔离。

不要仅按毁伤效率把 `21002 → 9400/9600` 屏蔽，因为低性能单位仍可能作为诱饵飞向这些点。把命中/毁伤适配性编码成目标特征即可。当前引擎表明 H/M 对 `9500` 命中率为 0，`21002` 对 `9400/9600` 伤害为 0，但后者仍有航路与诱导价值（[`simulator_factory.py` L164–224](/home/amax/ry/competition/competition_envs/core/envengine/simulator/simulator_factory.py:164)）。

## 6. 观测设计

### 6.1 环境提供的两种观测

**[源码已验证]** `_get_observation()` 返回所有 Simulator 的完整全局态势。每个实体含：

```text
nameChn, position(lon/lat/alt), pos_ecf(x/y/z), stage,
health, isVisible, type, side, detectInfo, commRangeInfo
```

对应实现见 [`training_env.py` L346–378](/home/amax/ry/competition/competition_envs/core/envengine/environment/training_env.py:346)。这份字典包含未被红方探测的蓝方实体。最终实现只用完整字典定位受控红方自身记录、计算官方奖励、判断终止和诊断；隐藏蓝方真值不进入 Actor，也不进入 Critic。

`AgentManager.extract_observation_for_agent()` 只在实体存活且可见时返回：

```python
{
    "step": step,
    "entity_id": entity_id,
    "agent_id": agent_id,
    "self": full_observation["entities"][entity_id],
}
```

实体死亡或不可见时返回空字典（[`agent_manager.py` L64–90](/home/amax/ry/competition/competition_envs/core/envengine/agent_manager/agent_manager.py:64)）。最终实现不调用该 AgentManager 方法，但遵循相同信息边界：只读取每个 `unit_id` 对应红方实体的自身字段。其 `detectInfo` 是目标 ID 到 `DetectInfo` 的映射；每条航迹含来源、时间、ID、类型、LLA、ECF 位置和 ECF 速度（[`DetectInfo.py` L8–29](/home/amax/ry/competition/competition_envs/core/envengine/sdk/base_struct/Basic/DetectInfo.py:8)）。运行对象可能是 dataclass 而不是普通 dict，解析器同时支持属性和 mapping 访问。

### 6.2 Actor 输入

当前实现每个单位槽编码：

- 当前步比例和剩余时间；
- 控制器生命周期 one-hot，而不是从 `stage` 猜测是否已发射；
- 类型 one-hot：`21000/21001/21002`；
- 已部署单位的归一化 LLA、健康、可见性；
- 由连续两帧 `pos_ecf` 估计的速度、航向和爬升率；全局观测本身没有导弹速度字段；
- 当前目标槽、上次改目标时间、上次机动；
- 当前目标距离势函数的参考距离和进度比例，使 motion Critic 能观测
  PBRS 依赖的控制器内部状态；
- 团队协调卫星的可用量、pending 和 ready 状态；`team_global` 的 ready 会同时反映
  工厂级活动窗口，`per_unit` 按合法 requester 的旧计数得出；
- 隔离 `detectInfo` 中的拦截弹数量、最近距离/方位、航迹年龄等威胁摘要。
  `team_global` 的数量用场景初始敌方拦截弹数归一化，`per_unit` 为保持旧观测
  契约仍使用固定分母 32。

每个目标槽编码：

- `present/valid/known`；
- 目标类型 one-hot：`9400/9500/9600`；
- 相对经纬方向、距离、方位正余弦；
- 航迹速度与 `velocity_known`；
- 航迹年龄；
- 是否为当前目标；
- 是否已由任一受控红方单位合法发现该目标航迹。
- 当前己方 `ACTIVE` 单位对该目标的 `assigned_total/high/medium/low`
  归一化分配负载。

原始 `target_id` 只留在 `TargetSlotRegistry` 元数据中。目标绝对坐标可以作为已知输入，但输出仍为离散目标槽。对于 `STAGED` 单位，场景中的占位坐标不是策略已经选择的部署位置，Actor 应使用“未部署”标记并将自身位置特征清零，另行提供类型对应部署区域的归一化描述。

负载只由 lifecycle tracker 内己方 ACTIVE 单位的当前目标汇总，不读取蓝方
血量或存活真值。`total` 以全部红方单位数为分母，H/M/L 分别以该类型的
全队总数为分母。它们是观测，不会屏蔽高负载目标或构成硬性容量约束。

单位观测宽度为 `D=40+19*O`。`O=18` 时由旧的 308 维变为 382 维；
final20 的 `O=24/36/48` 分别为 496/724/952 维。当前网络对全部合法单位 token
做均值池化。每个单位 Actor 由“本单位编码 + 团队池化上下文”产生动作；
plan/motion 逐弹 Critic 使用同一组合，sensor Critic 使用团队池化上下文。
因此这是集中但非特权的 Actor-Critic：集中体现在团队合法观测共享与池化，
不是访问蓝方隐藏真值。

`TrainingEnv` 保存 observation 时只是浅拷贝（[`training_env.py` L94–96](/home/amax/ry/competition/competition_envs/core/envengine/environment/training_env.py:94)），而 `DetectInfo` 内含可变向量对象。trajectory 在引擎推进前必须把需要的数据转换成独立的 NumPy 数值副本，不能长期持有原始观测对象引用。

### 6.3 初始目标目录存在的源码差异

`_get_init_ship_observation()` 注释列出 `9400/9500/9600`，但实际过滤条件只有 `(9400, 9600)`（[`training_env.py` L308–344](/home/amax/ry/competition/competition_envs/core/envengine/environment/training_env.py:308)）。

**[运行已验证]** 旧版 E01 的 `case_info.json` 声明 7 个目标全部公开，包括两个 `9500`（[`case_info.json` L18–26](/home/amax/ry/competition/competition_envs/scenarios/cases/easy/E01/case_info.json:18)、[`case_info.json` L87–105](/home/amax/ry/competition/competition_envs/scenarios/cases/easy/E01/case_info.json:87)），但环境初始接口只返回 3 个 `9400` 和 2 个 `9600`。

最终实现只把 `_get_init_ship_observation()` 返回的 `9400/9600` 作为已知初始坐标，再从所有受控红方单位的合法 `detectInfo` 动态补入 `9500` 等目标。`RewardPolicy.objective_ids` 仅预留稳定槽位并用于计分，不会提前把隐藏目标坐标送入网络。训练 metadata 固定记录 `central_detection_sharing=true` 和 `ground_truth_actor_access=false`（[`train_joint_ppo.py` L894–912](/home/amax/ry/competition/personal_train/train_joint_ppo.py:894)）。

## 7. 单位槽与目标槽生命周期

### 7.1 单位槽

- 环境构造后，收集并排序全部红方 `entity_id`；
- `unit_slot` 在一局内固定，不因死亡回收；
- 不创建 Agent ID，也不注册 proxy；
- 原始实体 ID 只用于动作翻译、日志和回执；
- 死亡或不可见后标为 `TERMINAL`，其后所有动作 mask 为 no-op。

### 7.2 目标槽

1. 构造时按 `RewardPolicy.objective_ids` 的稳定顺序预留计分目标槽，其余补到固定容量 `K`；预留 ID 本身不等于坐标已知；
2. reset 时只把 `_get_init_ship_observation()` 实际返回的公开目标标为 known；
3. 每步从全部受控红方自身 `detectInfo` 收集 `9400/9500/9600`，按来源时间只接受更新航迹，并把合法发现集中共享；
4. 一局内槽位不回收，动态目标一旦合法发现便保留最后已知位置及航迹年龄；
5. 发射/改目标时再次验证 slot 已知并把该槽的最新 `lon/lat` 写入命令，实体 ID 只留在适配器元数据。

若目标容量溢出，应立即把 overflow 数量写入 `info` 并使训练失败；静默丢目标会改变任务定义。

## 8. 卫星接口现状与建模

### 8.1 不应训练独立卫星 Agent

当前场景包含红方 `9202` 卫星，但它没有策略可控的轨道或运动接口：

- 真实轨道解算被注释，`set_lla()` 和 `set_speed()` 为空实现；
- 只有卫星全局窗口激活时，`OrbitModelSimulator` 才以 1000 ms 内部步长更新
  所有存活且可见的 `entityType == 24000` 拦截弹航迹
  （[`OrbitModelSimulator.py` L103–135](/home/amax/ry/competition/competition_envs/core/envengine/simulator/simlulator_impl/OrbitModelSimulator.py:103)）；
- Engine 把这些红方卫星航迹加入每个红方导弹群的融合结果，因此卫星情报是全队共享的。

因此给卫星单独建网络没有更多可执行的物理动作。联合策略的 sensor 分支只决定
是否在当前时刻发起服务请求；requester 槽仅用于把命令送进现有的导弹命令接口。

### 8.2 当前 `team_global` 后端

`[3, missile_id, 0, 0]` 转换成 `EXECUTE_SATELLITE_DETECTION` 后，
`SimulatorFactory.process_ai_commands()` 不再把它交给单枚导弹的
`command_received()`，而是由工厂统一处理
（[`simulator_factory.py` L378–416](/home/amax/ry/competition/competition_envs/core/envengine/simulator/simulator_factory.py:378)）。
工厂维护：

- `red_sat_use_count`：全队已用次数；
- `red_sat_max_use_count`：想定级全队上限，默认 100；
- `is_using_satellite()`：由工厂当前时间和全局结束时间计算的活动窗口。

一个请求会消耗全队一次额度，并把窗口结束时间设为“当前仿真时间 +
`satelliteUseMinutes`”。原生后端本身仍允许窗口内重复请求并重置时间；联合适配器会
屏蔽全部 requester，直到窗口结束，防止这种无意义的额度消耗。

卫星生效时还使 H 型 `21000` 对 `9400/9600` 的命中率提升到 100%
（[`simulator_factory.py` L246–261](/home/amax/ry/competition/competition_envs/core/envengine/simulator/simulator_factory.py:246)）。
M/L 型仍使用各自的本地探测规则；卫星的新全局情报负载是 24000 型拦截弹，
不再是旧后端中的逐导弹目标/无人船探测扩展。

### 8.3 适配器兼容契约

同一联合策略仍使用团队 pointer/STOP 头选 requester。`JointGameEnv` 通过卫星状态所在对象自动识别 backend：

- `team_global`：容量、活动状态和回执都读取 `SimulatorFactory`；步首的
  `STAGED` 和 `ACTIVE` 槽都可作 API requester，因此新部署/发射可与卫星请求在
  同一步执行；活动窗口使所有槽不 eligible，一步最多提交一个全局请求；
- `per_unit`：用于旧 runtime，继续逐槽检查导弹计数和窗口。

`--sensor-capacity` 可以进一步收紧协调层额度，但不能突破真实后端上限。
不显式指定时，`team_global` 使用工厂的 100 次上限，`per_unit` 使用所有
导弹后端上限之和。新 `team_global` 环境合同 v4 写入 backend、全队容量、
活动分钟数、有效每步请求上限，以及观测中的拦截弹数量归一化分母。旧
`per_unit` 保留环境合同 v3；因此旧 per-unit checkpoint 不能在新 team-global
后端上 resume 或 init-from。

`team_global` 的 requester 身份不改变卫星效果，但保留该槽使现有
pointer trace、命令格式和 PPO 契约不需另建网络。`per_unit` 仍要求 requester
在步首已是 `ACTIVE`，不允许以 staged 槽提前消耗该导弹的旧独立额度。

## 9. reset 与 step 的确定顺序

### 9.1 初始化一次

1. 加载 Profile 和 RewardPolicy；
2. 构造本地 `JointGameEnv`，由它内部创建原生 `TrainingEnv`；
3. 直接扫描受控红方 Simulator 并固定单位槽；
4. 创建一套共享 `JointPPOPolicy`；
5. 在构造环境前通过 `BLUE_POLICY` 选择 Engine 内部 DefendCommander 的蓝方策略；
6. 不注册任何红方 proxy、DeployAgent 或卫星 Agent，不调用 `red_model_deploy()`。

### 9.2 每局 reset

1. 调用 `TrainingEnv.reset()`；它先 reset Engine，再 reset 全部 Agent，随后获取初始全局观测（[`training_env.py` L66–103](/home/amax/ry/competition/competition_envs/core/envengine/environment/training_env.py:66)）。
2. 把每个 Simulator 的 `sim_time` 恢复到 `profile.imagineProfile.simTime`。
   `team_global` 后端的 `SimulatorFactory.reset_all()` 还会把全局卫星计数与窗口归零；
   适配器在每步由 Engine 提供的工厂时钟上判断活动状态。
3. lifecycle tracker、轨迹缓存、速度历史和本地卫星协调状态清零一次。
4. 校验构造时固定的实体槽仍对应同一批 Simulator。
5. 建立公开目标与合法红方探测并集对应的团队目标 mask。
6. 保存独立数值化的 `s0`、同源非特权 Critic 输入和初始官方分数。

### 9.3 每个联合 step

```text
s_t raw simulator snapshot
  -> 只提取受控红方自身字段与本方合法 detectInfo
  -> 集中合并合法目标航迹并更新单位 terminal 状态
  -> 构造共享 actor/非特权 critic tokens 和条件 masks
  -> policy 只采样一次 JointAction，并保存精确 trace/log-prob
  -> tracker.apply() 产生 activation/retarget/movement/satellite intents
  -> activation placement 直接 set_lla
  -> intents 转成按 slot 排序的引擎命令
  -> CommandAdapter.common_adapter(commands)
  -> engine.step(commands)，只调用一次
  -> current_step += 1
  -> 读取 s_{t+1}
  -> 确认同步 receipts，标记个体 terminal
  -> 在团队终止时先保存 assignment_states，再将剩余单位统一终态化
  -> 计算 team score delta、motion reward 和 sensor information PBRS
  -> 保存一条带三路 value/reward 契约的 JointTransition
  -> 局末按官方毁伤与己方分配历史回填 plan return
```

这与上游 `TrainingEnv.step()` 的“动作 → engine.step → 新观测 → 奖励 → done”主顺序一致（[`training_env.py` L160–232](/home/amax/ry/competition/competition_envs/core/envengine/environment/training_env.py:160)），但最终 rollout 不调用该方法，也不依赖其 per-Agent 历史。上游在一个实体有多条命令时只保存第一条匹配命令（[`training_env.py` L208–220](/home/amax/ry/competition/competition_envs/core/envengine/environment/training_env.py:208)），会丢失同一步的 maneuver/satellite；本地 trajectory 保存采样时的完整 `JointAction`、mask、分支 log-prob 和团队卫星 pointer trace。

上游 `AgentManager` 还会捕获动作异常、只写日志并继续（[`agent_manager.py` L99–119](/home/amax/ry/competition/competition_envs/core/envengine/agent_manager/agent_manager.py:99)）。直接编排路径会先验证完整联合动作，并要求 `CommandAdapter` 输出数量与输入命令数量一致；动作或适配失败时不调用 `engine.step()`，避免把“采样了但未执行”的动作写成行为策略数据。

## 10. 奖励与终止

### 10.1 奖励

不要使用基类示例奖励。它只对存活 Agent 每步加 1，其余战斗和团队项仍是 TODO（[`training_env.py` L406–448](/home/amax/ry/competition/competition_envs/core/envengine/environment/training_env.py:406)），会强烈鼓励不发射、一直保留导弹。

当前实现首先对每个联合 transition 计算团队分差：

```text
r_t = official_score(s_{t+1}) / 100 - official_score(s_t) / 100
```

官方分数是按目标初始血量的加权毁伤比例：目标/阵地/无人船权重分别为 5/2/1（[`reward.py` L12–18](/home/amax/ry/competition/competition_envs/scenarios/cases/reward.py:12)），计算过程见 [`reward.py` L144–224](/home/amax/ry/competition/competition_envs/scenarios/cases/reward.py:144)。逐步差分的回报和等于最终归一化分数，不需要终局再重复加分。

计分明确与完成时间无关（[`reward.py` L150–155](/home/amax/ry/competition/competition_envs/scenarios/cases/reward.py:150)），
当前不加时间或“存活一帧”奖励。这一团队信号被拆到三条专用通道：

- **plan**：对曾成功激活的单位，局末直接使用 `0.7 * final_score/100 +
  0.3 * learning_local_credit_i`。每个目标的 raw local credit 按该目标的官方
  加权毁伤贡献建立，再按己方目标分配持续时间分配，保证 raw allocated +
  unallocated 与目标贡献守恒。学习用副本为
  `clip(eligible_unit_count * raw_local_credit_i, 0, 1)`，使其与广播 team score
  同量级；该缩放并裁剪的副本不再是全局守恒量。plan 直接学终局 return，不做长
  horizon GAE。
- **motion**：使用 `team_delta + 0.05*(gamma*Phi_distance(s')-Phi_distance(s))`，
  通过独立 motion Critic 计算 GAE。换目标、死亡和真实终止会闭合距离势函数。
- **sensor**：使用 `team_delta + gamma*Phi_info(s')-Phi_info(s)`，但 `Phi_info`
  按后端效果选择。`team_global` 对受控红方 `detectInfo` 中的 24000 航迹按
  拦截弹 ID 去重并取最新航迹，丢弃超过 `max_track_age` 的记录。对每个航迹使用
  `max(0, 1-age/max_track_age)` 线性新鲜度，求和后除以场景初始敌方拦截弹数并裁剪到 1，
  最后乘以 scale 得到 fresh-interceptor-track potential；`per_unit` 为复现旧后端，继续使用
  `已合法发现目标的公开权重/总权重`。两种势函均不读取隐藏蓝方真值，
  上限都由默认 `0.015` 的 scale 控制，并由独立 sensor Critic 计算 GAE。

plan、motion、sensor 分别形成 PPO ratio、advantage、policy/value loss、entropy、
KL 和 clip fraction。一个单位一局内的 plan 样本按逆次数加权，权重再在
整个 rollout 上归一化，同时用于 plan policy loss、entropy 和 advantage 归一化。

Actor/Critic 都不读取隐藏目标血量；环境层可用完整仿真态势计算官方 score
delta 和局末 credit，因为奖励 target 不是策略观测。

### 10.2 terminated 与 truncated

基类 `get_is_done()` 在 `current_step >= max_steps` 时结束；否则只要任意 `21000/21001/21002/24000` 仍可见就继续（[`training_env.py` L234–255](/home/amax/ry/competition/competition_envs/core/envengine/environment/training_env.py:234)）。它只返回一个 `done`。

最终适配器按“正式任务终点”和“人为调试截断”拆分：

```python
debug_limit = is_debug_horizon and current_step >= effective_max_steps
official_limit = not is_debug_horizon and current_step >= official_max_steps
terminated = official_limit or natural_done or objectives_completed
truncated = debug_limit and not terminated
done = terminated or truncated
```

因此 E01 正式第 1200 步是任务定义内的有限时域终点，记 `team_terminated=True` 并令 bootstrap 为零；只有显式 `--debug-max-steps` 造成的提前停止才记 `team_truncated=True`，允许按截断语义 bootstrap。若在调试上限同一步已经发生自然终止或全部目标摧毁，真实 `terminated` 优先。实现见 [`joint_game_env.py` L662–700](/home/amax/ry/competition/personal_train/joint_game_env.py:662)。

个体单位在 health `<= 0` 或不可见时进入 `TERMINAL`。未发射的 `STAGED` 储备仍在原生 Simulator 中存在，因而会影响上游 `natural_done`；当前还可由 `terminate_on_all_objectives_destroyed` 控制是否在计分目标全部摧毁时提前真实终止。

## 11. 动态部署的验证结果与已知边界

上游官方路径只在仿真正式步进前调用 `modify_simulator_position()`，源码中没有提供延迟激活范例。**[强制仿真已验证]** 当前本地适配路径已经在多步 GPU smoke 中发生真实延迟激活，并另用确定动作强制覆盖 H/M/L 三种类型：运行中的 Simulator 接受新位置，H/M 以高度 0、L 以高度 10000 重置原生模型，同一步 launch 与首个 maneuver 生效；H/M 的 `launch != 0`、L 的 `launch >= 0` 回执判定均成功。旧 runtime 的确定 requester 实测证明了 per-unit 回执；更新后的 live 路径改为校验工厂级计数。因此动态部署/发射不再属于统一的“未验证接口”。

该结果是 E01 的短程集成验证，仍不能证明 legacy、`final24` 和
`final20` 在完整 horizon 下都长期稳定。开始大规模训练后仍应监控原生位置/速度
是否有限、发射后阶段转换、异常终止和跨 reset 污染。

另一个边界是 staged Simulator 从场景加载时已经存在且通常 `isVisible=True`。蓝方雷达会额外过滤 `stage >= 3`（[`RadarModelSimulator.py` L78–90](/home/amax/ry/competition/competition_envs/core/envengine/simulator/simlulator_impl/RadarModelSimulator.py:78)，未发射导弹通常不会进入该阶段），但弹间通信分群按存活红方导弹计算，没有过滤是否发射或可见（[`engine.py` L179–220](/home/amax/ry/competition/competition_envs/core/envengine/engine/engine.py:179)）。所以“尚未部署的储备完全不存在于战场”无法通过公开接口严格表达。

当前实现与限制：

- 将 STAGED 坐标和探测字段从 Actor 输入中屏蔽；
- 不修改 staged 实体健康，不用临时置零等高风险技巧；
- 把可能的通信桥接影响记录为已知仿真限制；
- 部署坐标适配器只接受轴对齐矩形，多边形想定会在构造时直接报错，避免把边界框外点误当合法点；
- `modify_simulator_position()` 是先于 `CommandAdapter` 的副作用。联合动作和目标会在此前校验，但若后续发生极少见的适配器内部异常，原生接口没有通用 rollback；这类局应丢弃并 reset；
- 对比“立即全部激活”和“延迟激活”时的 `commRangeInfo/detectInfo`，继续评估 staged 通信桥接的实际影响。

若严格储备隔离是比赛规则的硬要求，则需要上游提供正式的 spawn/activate 接口；仅在 `personal_train` 猜测性修改原生实体可见性和生命值不够安全。

## 12. 接入验收与后续回归

动态部署/发射和卫星 requester 已通过原生强制验证；变长目标观测、目标条件分支、
三路 reward/PPO 与双卫星 backend 契约需在每次上游更新后继续做真实 smoke。
短 smoke 不代表已完成完整 horizon 或 200 回合仿真。测试均应使用固定 seed、
`render_mode=None`，且不写 pku 目录：

1. **构造测试**：E01 得到 164 个连续单位槽、无重复 entity ID、目标容量 18、
   5 个环境初始目标槽，观测形状为 `[164,382]`。
2. **三类型原子激活（已强制实测）**：分别激活一枚 H/M/L；验证部署区、H/M 高度 0、L 高度 10000，并确认同一步 launch 与 maneuver 生效。
3. **延迟激活（已在多步 GPU smoke 发生）**：在 Engine 已推进后再激活，并继续检查位置和观测有限、无原生崩溃；完整 horizon 仍需回归。
4. **同一步多命令**：新激活单位包含 launch + maneuver；`team_global` 还可以
   以该 STAGED 槽作 requester 同步发 satellite。trajectory 保存全部原始分支，而不是
   上游的第一条 command。
5. **二次发射 mask**：ACTIVE 单位永远不能再产生 launch；无“重复发射”输出。
6. **retarget**：目标槽翻译出的目标 ID 只出现在 metadata，实际引擎命令是正确的最新 lon/lat。
7. **隐藏信息边界**：构造未出现在任何受控红方 `detectInfo` 的隐藏目标，Actor 与 Critic 均不得包含其坐标；一旦任一本方单位合法发现，可按当前集中共享口径进入全部团队 token 和目标 mask。
8. **卫星双后端**：`team_global` 在首步允许 STAGED requester 与部署/发射同步，
   请求仅使工厂计数加一，窗口内全部 requester 均被 mask，并且 24000 航迹
   对红方全队可见；`per_unit` 仅允许 ACTIVE requester，仍只扣请求导弹的旧额度。
9. **reset 两局**：槽位、时钟、对应 backend 的卫星次数/窗口、速度历史、
   lifecycle、trajectory 全部复位，第二局首帧不含第一局末帧数据。
10. **奖励守恒**：整局 `sum(score_delta)` 与最终 `score/100` 在浮点误差内一致；
    raw local allocated+unallocated 逐目标等于其官方毁伤贡献，并与乘以合格单位数的
    learning copy 分开记录。
11. **三路 PPO**：plan/motion/sensor 分别只聚合自己的有效 log-prob，三路
    advantage、value loss、entropy、KL、clip fraction 和决策数均有独立记录。
12. **条件动作**：改变已选目标会改变部署/首次机动分布；换目标本步的
    movement log-prob 按新目标重建。
13. **目标负载**：只有 ACTIVE 单位计入 total/H/M/L，激活、换目标和死亡后
    下一观测立即更新，不受敌方血量影响。
14. **终止语义**：正式想定 horizon、物理自然结束和启用的目标全毁条件标为 `terminated`；只有显式缩短的 debug horizon 标为 `truncated`；个体死亡只关闭对应槽。
15. **变长目标槽**：final20 easy/medium/hard 自动得到 24/36/48 槽和
    496/724/952 维观测，不发生 overflow。
16. **失败原子性**：非法坐标、溢出目标槽或动作转换异常时不推进 Engine，也不写 PPO transition。

策略 schema 为 v3；`per_unit` 环境合同为 v3，`team_global` 环境合同为 v4。
环境合同还绑定 backend、backend 容量/窗口、威胁计数归一化、目标槽和观测宽度。
308 维观测的旧 checkpoint、per-unit 后端 checkpoint 和不同 `O` 的 final20
checkpoint 不能直接 resume 或 init-from。完整 horizon 之后应使用
`eval_joint_ppo.py` 对相同场景与固定 seed 集分别运行 deterministic/stochastic
评估（省略 `--seeds` 时默认为 `1..10`），检查 `per_episode.csv/jsonl`、
`per_seed.csv` 和 `aggregate.json`。

环境本身不是可并发共享的对象；后续多环境采样应使用独立进程和独立运行目录，
不能让多个 rank 同时操作同一个 `TrainingEnv`。完整场景训练应保留上述长期回归、
固定多种子评估和监控。
