# personal_train 强化学习训练

## 端到端联合 PPO（当前推荐）

[`train_joint_ppo.py`](train_joint_ppo.py) 已把部署/发射、目标选择、发射后机动和
卫星调度放进同一套可训练策略，不再调用 R9 高层规划。每枚红方导弹占一个稳定槽位：

- 尚未发射时输出 `是否激活 + 部署 (x,y) + 目标槽位 + 首次机动`；部署、发射和
  首次机动在同一个仿真 step 执行。
- 发射后输出 `是否换目标 + 目标槽位 + 左/停/右机动`，部署与再次发射会被 mask。
- 卫星不单独训练 Agent；同一网络使用一个团队指针头，从当前可用导弹中选择请求者
  或 STOP。新版 `pku@32c08cd` 的卫星是 `SimulatorFactory` 管理的全队共享资源：
  全队共用 100 次，卫星生效窗口内屏蔽重复请求。生效时卫星探测全部存活
  24000 型拦截弹并向红方全队共享航迹，同时把 H 型对 9400/9600 的命中率提升到
  100%。这个全局请求可以以 STAGED 或 ACTIVE 槽作 API requester，因此能与首步
  部署/发射同步。适配器也保留对旧版
  逐导弹后端的兼容。

Actor 只接收红方受控导弹拥有的合法观测。各导弹的探测结果在红方团队内集中共享，
隐藏目标只有被己方发现后才进入目标 mask；完整蓝方坐标和血量只用于官方计分，
不会进入 Actor 或 Critic。每个目标槽还包含当前己方 `ACTIVE` 导弹的
`total/H/M/L` 归一化分配负载；每个单位包含当前距离势函数的参考距离和进度比例。
这些字段全部来自控制器历史和合法观测。目标血量仍不是策略输入。单位观测维度为
`D = 40 + 19 * O`。目标槽数 `O` 默认至少为 18，并会按当前场景的全部计分目标
自动扩展。因此 legacy/final24 的 `O=18` 对应 382 维；final20 的 easy、medium、
hard 分别为 `O=24/36/48`，对应 `D=496/724/952`。

策略的规划分支是自回归的：激活时先选目标，部署位置和首次机动再以该目标为
条件；`ACTIVE` 单位选择换目标时，本步机动也以新目标为条件。网络分别使用
plan、motion 和 sensor 三个 Critic，计算三组优势与 PPO 比率，报告中也分别
记录 loss、entropy、KL、clip fraction 和决策数。

三路奖励与该分支的时间尺度对应：

- plan 在局末学习 `0.7 * 官方终局分 + 0.3 * 局部目标 credit`。原始局部
  credit 按每个目标守恒分配；学习用副本为
  `clip(eligible_unit_count * raw_local_credit, 0, 1)`，并与原始守恒值分开记录。
  每个 unit-episode 的规划样本总权重相同。
- motion 学习逐步官方分差加距离 PBRS，由 motion Critic 计算 GAE。
- sensor 学习团队分差加后端匹配的信息 PBRS。新的全队卫星后端使用
  合法可见的新鲜拦截弹航迹势差：按拦截弹 ID 去重，航迹随年龄线性衰减，
  并以场景初始敌方拦截弹数归一化。旧的逐导弹后端仍使用已知计分目标势差。
  两者默认势函上限均为 `0.015`，由 sensor Critic 计算 GAE。

本机 Conda 环境位于 `personal_train/.conda/competition-rl`，私有 glibc 2.38
启动器已经接好 Linux 原生导弹库。服务器不需要 Docker，也不要用 `torchrun`。
当前训练源码以 `/home/amax/ry/competition/competition_envs` 的
`pku@32c08cd` 为准；`bootstrap.py` 会优先发现这个平级目录，正式命令仍显式设置
`COMPETITION_REPO_ROOT` 以便复现。`glibc-2.38/runtime/pku` 保留为旧后端快照，只用于
复现和评估旧的 per-unit 卫星模型。

checkpoint 策略 schema 仍为 `v3`。旧 `per_unit` 环境合同保留 v3，新
`team_global` 环境合同为 v4，并写入全队容量 100、活动时间 3 分钟和有效
每步最多 1 个请求，以及观测中拦截弹数量的场景归一化分母。合同还会绑定
场景、单位/目标槽和观测维度。308 维 checkpoint、旧 per-unit backend 上产生的
checkpoint，以及目标槽数不同的 final20 checkpoint 都不能直接用于新后端，
需要从新随机初始化。
下面命令在 tmux 中开始 E01 的 200 回合训练：

```bash
tmux new-session -d -s e01-joint-env32-200 \
  "bash -lc 'source /home/amax/ry/competition/personal_train/activate_competition_rl.sh && \
  export COMPETITION_REPO_ROOT=/home/amax/ry/competition/competition_envs && \
  export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 && \
  cd /home/amax/ry/competition && \
  exec /home/amax/ry/competition/glibc-2.38/python3.11-glibc238 \
  personal_train/train_joint_ppo.py \
  --scenario /home/amax/ry/competition/competition_envs/scenarios/cases/easy/E01/scenario.json \
  --rounds 200 \
  --device cuda:0 \
  --learning-rate 1e-4 \
  --learning-rate-final 1e-5 \
  --learning-rate-decay-updates 50 \
  --target-kl 0.01 \
  --rollout-episodes 4 \
  --minibatch-size 128 \
  --sensor-capacity 100 \
  --planning-team-weight 0.7 \
  --planning-local-weight 0.3 \
  --sensor-information-potential-scale 0.015 \
  --run-id e01_joint_env32_200'"

tmux attach -t e01-joint-env32-200
```

命令显式指向常规 `scenarios/cases/easy/E01`，避免与同名场景混淆。缩写
`--scenario E01` 也会解析到这个场景。如果要训练
`final24` 版 E01，必须显式传入
`--scenario /home/amax/ry/competition/competition_envs/scenarios/cases/final24/easy/E01/scenario.json`；
训练 final20 版 E01 则显式传入
`--scenario /home/amax/ry/competition/competition_envs/scenarios/cases/final20/easy/E01/scenario.json`。
不要为 final20 手工设置 `--objective-slots 18`；省略该参数时训练器会根据
`RewardPolicy.objective_ids` 自动扩展。这些场景的实体数、目标数和观测宽度可能不同，
checkpoint 不通用。

`--rollout-episodes 4` 会用冻结的行为网络收集 4 个
完整回合，再合并做一次 PPO 更新；最后不足 4 回合的批次也会更新。这样增加每次更新
的有效样本量。200 回合正好产生 50 次更新，因此命令让学习率在本次运行内由
`1e-4` 线性降到接近 `1e-5`；entropy 保留默认的慢衰减以维持稀疏奖励下的探索。
KL 阈值和卫星团队额度也使用较保守设置。
新架构尚未完成 200 回合或完整 horizon 的结果验证。启动前可先检查 GPU：

```bash
source /home/amax/ry/competition/personal_train/activate_competition_rl.sh
/home/amax/ry/competition/glibc-2.38/python3.11-glibc238 \
  -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_CUDA")'
```

先做短验证可显式缩短 horizon：

```bash
export COMPETITION_REPO_ROOT=/home/amax/ry/competition/competition_envs
/home/amax/ry/competition/glibc-2.38/python3.11-glibc238 \
  personal_train/train_joint_ppo.py \
  --scenario /home/amax/ry/competition/competition_envs/scenarios/cases/easy/E01/scenario.json \
  --rounds 1 \
  --debug-max-steps 20 \
  --hidden-dim 64 \
  --update-epochs 1 \
  --minibatch-size 20 \
  --device cuda:0 \
  --run-id e01_joint_smoke
```

正常训练不传 `--debug-max-steps`，会使用 E01 的正式 1200 步有限时域。调试 horizon
属于 truncation 并进行 value bootstrap；正式时限和自然结束属于 termination，
bootstrap 为零。checkpoint 会绑定场景文件、实体槽、目标槽、观测 schema、horizon、
探测共享、卫星和奖励配置，因此不能拿短验证 checkpoint 直接续接正式训练。

需要区分两种 checkpoint 用法：

- `--init-from best.pt` 只加载兼容的 v3 网络权重，重新创建优化器、计数器和随机数流，并采用本次
  命令的 PPO、奖励和卫星团队额度。调整训练稳定性参数时应使用它。
- `--resume latest.pt` 用于完整策略状态续跑，会恢复网络、优化器、调度器、计数器、
  策略随机数状态以及原 PPO、游戏和 rollout 配置；命令中的新超参数不会替换
  checkpoint 配置。

续跑时 `--rounds` 表示本次再训练的轮数：

```bash
export COMPETITION_REPO_ROOT=/home/amax/ry/competition/competition_envs
/home/amax/ry/competition/glibc-2.38/python3.11-glibc238 \
  personal_train/train_joint_ppo.py \
  --scenario E01 \
  --rounds 100 \
  --device cuda:0 \
  --resume /home/amax/ry/competition/personal_train/models/<run>/latest.pt \
  --run-id e01_joint_ppo_resume
```

只有当前 v3 版本写出的 `latest.pt` 和 `checkpoints/round_*.pt` 带有安全更新边界标记，
可以传给 `--resume`。同一 v3 schema 的 `best.pt`、`interrupted.pt`、`failed.pt`
及没有安全边界标记的 checkpoint 应传给 `--init-from`；任何 v1/v2 checkpoint 都会被拒绝。
续跑会恢复记录的 seed、蓝方策略、rollout 大小和
debug horizon；新进程中的蓝方独立随机流仍从该 seed 重新开始，因此它不是逐位一致的
整个仿真进程重放。

运行报告写入 `results/<run>/`，模型写入 `models/<run>/`。`rounds.csv/jsonl` 每回合
记录得分、plan/motion/sensor return、raw/学习用局部 credit，以及激活、目标、
部署、机动和卫星动作统计；`updates.csv/jsonl` 对 plan、motion、sensor 分别记录
policy/value loss、entropy、KL、clip fraction、advantage 摘要和决策数。`best.pt`
是产生最佳得分的更新前行为网络，
`latest.pt` 和周期 checkpoint 是完成一个完整 rollout 更新后的状态。
SIGINT/SIGTERM 在 PPO 更新期间会延迟到该更新、报告和安全 checkpoint 提交完成后处理，
避免保存只有部分 minibatch 生效的网络。

当前实现会在 rollout 收集时按分支批量采样所有单位动作，并在 PPO 更新前只做一次
轨迹张量打包；每个 minibatch 的 mask、log-prob、entropy 和 value loss 都在设备上
批量计算。完整 rollout 的观测、动作、mask、return 和 advantage 保留在 CPU，只把当前
minibatch 搬到 GPU，因此显存峰值由 `minibatch_size` 而不是场景的完整 transition 数量
决定。观测编码、同一步 action mask 和不可变轨迹快照也会复用。网络结构、v3
checkpoint、rollout 大小和 PPO 超参数均未改变；随机采样仍来自相同的条件分布，但批量
抽样改变了随机数的消费顺序，因此从旧实现的 checkpoint 续训不会逐位复现旧轨迹。

## 多场景联合 PPO 独立并发

[`train_joint_multi_scenario.py`](train_joint_multi_scenario.py) 是原生宿主机调度器，
使用 `--suite legacy|final24|final20` 选择场景套件，默认是 legacy。省略
`--scenarios` 时会分别创建 9/24/20 个相互隔离的
`train_joint_ppo.py` 进程。每个场景拥有独立策略、optimizer、随机数流、私有 simulator
runtime、结果、模型和 worker 日志；不同场景的经验不会混入同一策略梯度，因此不改变
单场景 PPO 的训练定义。

final20 可以先预览部分场景：

```bash
cd /home/amax/ry/competition
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

它们会解析为 `final20/easy/E01`、`final20/medium/M06` 和
`final20/hard/H08`。非 legacy 运行名包含 suite 前缀；续训会校验 suite、
selector 和绝对场景路径。

若 GPU0 正被单场景训练占用，最稳妥的预览方式是只列出七张空闲卡；九个任务
全部进入队列，最多七个同时运行，任一结束后自动补入下一项：

```bash
cd /home/amax/ry/competition
export COMPETITION_REPO_ROOT=/home/amax/ry/competition/competition_envs

/home/amax/ry/competition/glibc-2.38/python3.11-glibc238 \
  personal_train/train_joint_multi_scenario.py \
  --rounds 100 \
  --gpu-ids 1,2,3,4,5,6,7 \
  --max-parallel 7 \
  --rollout-episodes 4 \
  --dry-run
```

确认 JSON 计划无误后删除 `--dry-run` 才会真正启动。`--dry-run` 不创建目录，也不启动
子进程。为保证不影响当前 E01，需等它结束后再删除 `--dry-run`；即使避开 GPU0，正式
worker 仍会竞争主机 CPU、内存带宽和 I/O。脚本要求显式给出 `--gpu-ids`；GPU0 默认
受保护，重复 GPU 默认也会被拒绝。
若必须在 GPU0 仍被占用时同时启动九个 worker，只能显式共享两张空闲卡：

```bash
export COMPETITION_REPO_ROOT=/home/amax/ry/competition/competition_envs
/home/amax/ry/competition/glibc-2.38/python3.11-glibc238 \
  personal_train/train_joint_multi_scenario.py \
  --rounds 100 \
  --gpu-ids 1,2,3,4,5,6,7,1,2 \
  --allow-gpu-sharing \
  --max-parallel 9 \
  --dry-run
```

共享 GPU 能提高环境总并发，但通常会降低共享卡上每个 PPO 更新的速度；优先使用七并发
队列模式。当前训练结束后可使用 `--gpu-ids 0,1,2,3,4,5,6,7`，并同时传入
`--allow-gpu-zero --max-parallel 8`。可选的 `--numa-nodes`、`--cpu-sets` 和
`--threads-per-worker` 按 device slot 绑定 CPU/内存；`--resume-batch` 会从上一批九个
场景各自的 `latest.pt` 恢复，或用重复的 `--resume-from CASE=/path/latest.pt` 单独指定。
调度状态和汇总写入 `launcher_runs/joint_multi_<timestamp>/`；发送退出信号时，调度器只
通知本批次的独立进程组，并等待 trainer 在安全 PPO 更新边界保存后退出。

使用 [`eval_joint_ppo.py`](eval_joint_ppo.py) 对同一组固定种子做严格的确定性评估：

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

采样评估使用同样的 checkpoint、场景和种子，将 `--deterministic` 换成
`--stochastic`，并可增大 `--episodes-per-seed`。评估会严格比对场景和 checkpoint
契约，在 `results/<eval-run>/` 写入 `per_episode.csv/jsonl`、`per_seed.csv`、
`aggregate.json`、`evaluation_config.json`、`status.json` 和 `evaluation.log`。
`aggregate.json` 包含 mean/std/median/quantile/lower-CVaR，用于比较不同 checkpoint。
省略 `--seeds` 时使用固定的 `1..10`。

仿真器要求工作目录中存在一组上游源码链接和原生 `Results` 临时文件。这些实现文件
现在统一放在 Git 忽略的 `personal_train/.runtime/training_runs/`，不会再混入正式结果
目录。`run_config.json` 中的 `runtime_dir` 可用于定位对应运行沙箱。

历史目录 `results/joint_forced_hml_satellite_20260909_114546_104287` 是一次 3 step 的
H/M/L 激活和旧 per-unit 卫星接口验证，并不是正式训练结果，也不能当作
新 team-global 后端的验证证据。其中 23 个“代码文件”都是指向只读
上游运行库的符号链接，624 个原生结果文件均为空。它已整体迁至
`.runtime/validation/joint_forced_hml_satellite_20260909_114546_104287`；历史训练目录中
同类 `runtime/` 也已迁出，映射记录在 `.runtime/runtime_migration_20260909.json`。
当前 checkpoint 严格对应一个场景的实际实体槽数；要用一个 checkpoint 混训不同规模
场景，还需要增加 `unit_present` padding/mask。

联合核心的动作、状态和轨迹契约见 [JOINT_RL_CORE.md](JOINT_RL_CORE.md)，仿真适配
与观测/奖励设计见 [JOINT_ADAPTER_DESIGN.md](JOINT_ADAPTER_DESIGN.md) 和
[JOINT_TRAINING_DESIGN.md](JOINT_TRAINING_DESIGN.md)。

## R9 + PPO 旧架构与迁移记录（保留）

当前目录已独立为 `/home/ry/competition/personal_train`，上游为平级的
`competition-platform-env`。通用路径、只读挂载、设备依赖及容器基础环境已适配，
新版 `pku` 的加权损伤评分接口也已适配。完整训练状态及仍待验证项目见
迁移记录。
请先阅读 [迁移与验证记录](MIGRATION.md)。

当前九场景并行入口会为每个场景创建独立容器、原生输出 tmpfs，以及直接位于
`results/` 和 `models/` 下的扁平单场景目录。推荐从指定历史批次中九个场景各自的
`best.pt` 初始化：

```bash
cd /home/ry/competition/personal_train
python3 train_r9_multi_scenario.py \
  --rounds 100 \
  --scenarios E01 E02 E03 M01 M02 M03 H01 H02 H03 \
  --resume-batch r9_multi_20260906_151415_061752 \
  --seed 3 \
  --gpu-ids 0,1 \
  --max-parallel 9 \
  --allow-gpu-sharing \
  --threads-per-worker 1 \
  --image personal-competition:runtime
```

下方旧的手工 Docker 命令保留为迁移前历史说明，仍含失效路径，不要直接执行。

本目录提供一个不修改竞赛项目其他文件的 R9 + PPO 训练入口。Actor 的竞赛接口保持不变：

- 观测：90 维（原 85 维基础观测 + R9 的 5 维任务上下文）
- 动作：3 维离散动作，左机动 / 停止机动 / 右机动
- 高层规划：原项目 `r9_hierarchical_learning`
- PPO 与网络：从原项目复制到 `personal_train/ppo_policy.py` 后独立改造；
  Actor-Critic 层名和结构保持兼容，训练算法改为逐 Agent 轨迹 GAE-PPO
- 默认网络：`90 -> 128 -> 128`，共享骨干后接 `Actor -> 3` 与 `Critic -> 1`

原文件 `policies/red/learning/ppo_policy.py` 没有被本训练器修改。本训练器不再在
单个 Agent 提交 transition 时更新：默认整轮冻结行为策略，按 164 条独立轨迹
计算 GAE 后统一更新。还加入 value clipping、KL early-stop、梯度裁剪、学习率与
熵系数退火。默认值为 `gamma=0.999`、`gae_lambda=0.995`、`lr=1e-4 -> 2e-5`、
`entropy=0.002 -> 0.0002`、`epochs=2`、`minibatch=4096`。

## 已在训练侧修复的问题

1. 使用本目录的 `ResetAwareAgentManager`，每轮将所有 Agent 重置，并按对象身份把共享 Commander 只重置一次。
2. 从连续两帧、单弹合法观测中的 `pos_ecf` 和 step 差分得到 ECEF 速度，再转换为本地速度、天向速度和航向；没有读取全局引擎速度真值。
3. 用球面初始方位角计算目标方位，并把相对方位规范到 `[-pi, pi]`。
4. 按稳定的 `entity_type == 24000` 统计拦截弹，不依赖名称前缀“标6”。
5. 把 `DetectInfo.time` 的绝对毫秒逻辑时转换成回合 step 后再计算航迹年龄。
6. PPO 每次发出动作后，更新下一状态使用的 `maneuver_state`。
7. 编码下一状态时重新传入当前目标槽索引。
8. 原项目的目标融合器会丢弃新发现的 9500；本目录的 `PersonalR9Commander` 只从红方隔离观测的合法 `detectInfo` 动态加入 9500，并在下一回合清空这些动态航迹。
9. 在统一隔离观测快照到达后先完成一次原 R9 重规划，并冻结本 step 的发射比例上下文，避免 Agent 的 Python 遍历顺序改变同一步观测，也保证 PPO 保存的下一状态目标与下一次 Actor 使用的一致。
10. 新版全队卫星后端由 Commander 每步仲裁：活动窗口内不重复申请，同一步多个
    H leader 只提交一次请求；90 维观测中原有的卫星位改为团队窗口状态。缺少新版
    顶层字段的旧环境继续使用逐弹行为。

观测修复只改变原 90 个槽位中错误值的生成方式，没有增加观测、改变槽位顺序或扩展动作。Commander 扩展也仍调用原 R9 的分配算法，只把合法探测到的 9500 加入其目标目录，并把原先发生在首个 Agent 查询时的同一步规划提前到统一观测阶段。

## 奖励

训练奖励与赛事最终评分严格分开。最终评分由当前只读上游的 `RewardTracker`
按加权损伤比例计算：普通目标、拦截阵地、无人船的权重分别为 `5:2:1`，
每个目标按实际损失生命值比例计分，不含时间项。训练奖励不会写回 Actor 观测。

默认训练奖励改为“正式分增量为主、有界 shaping 为辅”：

- 每次目标生命值下降时，直接通过上游 `RewardTracker` 的相邻状态分差产生
  精确正式评分增量。每枚存活弹得到
  该值除以固定初始队伍规模，而不是除以当时幸存数，避免少数幸存者突然收到
  数十倍奖励；原始正式分增量另记为 `raw_official_score_delta`。
- 实际掉血 shaping 降为 `1 * 目标权重 * 健康损失比例`。全场理论上限仅为
  总目标权重（旧九场景 E01 为 21）；同时命中时仍按名义有效伤害近似归因，但总量只来自
  真实 HP 下降。
- 距离项改为与 PPO 折扣一致的 `gamma*Phi(s')-Phi(s)`，其中势函数相对
  “本次目标分配初始距离”定义，`Phi` 本身严格限制在 `[-0.1, +0.1]`；
  与训练使用相同 gamma 的折扣累计不会改变原任务的最优策略。切换目标时重新建锚点，
  并结清旧势函数；死亡和时间上限也以终端势函数 0 结清。
- 取消逐飞行 step 成本，动作切换成本降为 `0.0002`，避免“越早死亡越少扣分”。
- 无有效目标毁伤时，提前死亡和活到未完成任务的时间上限使用同一成本：
  H `-1.0`、M `-0.35`、L `-0.10`；产生过真实目标掉血的来源免除此项。

旧训练中距离项占全部正奖励约 92%，现在单弹最多 `+0.1`，不再能够盖过
目标正式分和全弹失败成本。不同弹种仍通过真实伤害和不同失败成本区分。

当前没有为“消耗一枚蓝方拦截弹”设置直接正奖励：L 弹作为诱饵是通过更低的损失代价隐式表达的。这样可以避免 PPO 学成集体送死；如后续实验证明确实需要显式诱饵奖励，应再基于拦截关系以远小于目标毁伤的尺度加入。

## Docker GPU 训练（历史说明，当前服务器禁止使用）

推荐使用 GPU 镜像内自带的 `/app` 核心代码，只挂载本目录（可写）和九个场景（只读）。原生导弹库必须在容器内写 `./Results/...` 临时文件，因此不要把整个 `/app` 或源码中的 `core` 只读覆盖；这些原生临时文件会随 `--rm` 删除，宿主机仍只有 `personal_train` 会被写入：

```bash
cd /home/ry/competition-platform-env

TRAIN_UID="$(id -u)"
TRAIN_GID="$(id -g)"
TRAIN_USER="$(id -un)"
docker run --rm -it \
  --name r9-ppo-train \
  --gpus all \
  --ipc=host \
  --user "${TRAIN_UID}:${TRAIN_GID}" \
  --tmpfs "/app/Results:rw,uid=${TRAIN_UID},gid=${TRAIN_GID},mode=0775" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e USER="${TRAIN_USER}" \
  -e LOGNAME="${TRAIN_USER}" \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
  -v /home/ry/competition-platform-env/personal_train:/app/personal_train:rw \
  -v /home/ry/competition-platform-env/scenarios:/app/scenarios:ro \
  -w /app \
  competition:ppo \
  python3 personal_train/train_r9_ppo.py \
    --scenario E01 \
    --rounds 100 \
    --device cuda
```

未指定 `--run-id` 时，E01 会自动生成单层目录
`results/e01_r9_ppo_<时间戳>` 和 `models/e01_r9_ppo_<时间戳>`。如果要添加实验标签，
可传入例如 `--run-id e01_r9_ppo_lr_test`；时间仍会追加在同一个目录名中。

`--device cuda` 是默认值：容器看不到 GPU 时会直接报错，不会悄悄退回 CPU。可先验证：

```bash
docker run --rm --gpus all competition:ppo \
  python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

本机现有 `competition:v1` 是未包含 Torch 的旧镜像；GPU 训练应使用已经构建好的 `competition:ppo`。

继续训练时传入先前模型（路径必须是容器内路径）：

```bash
cd /home/ry/competition-platform-env

TRAIN_UID="$(id -u)"
TRAIN_GID="$(id -g)"
TRAIN_USER="$(id -un)"
docker run --rm -it \
  --name r9-ppo-resume \
  --gpus all \
  --ipc=host \
  --user "${TRAIN_UID}:${TRAIN_GID}" \
  --tmpfs "/app/Results:rw,uid=${TRAIN_UID},gid=${TRAIN_GID},mode=0775" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e USER="${TRAIN_USER}" \
  -e LOGNAME="${TRAIN_USER}" \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
  -v /home/ry/competition-platform-env/personal_train:/app/personal_train:rw \
  -v /home/ry/competition-platform-env/scenarios:/app/scenarios:ro \
  -w /app \
  competition:ppo \
  python3 personal_train/train_r9_ppo.py \
    --scenario E01 \
    --rounds 100 \
    --device cuda \
    --run-id e01_r9_ppo_resume \
    --resume personal_train/models/e01_r9_ppo_<时间戳>/latest.pt
```

当前格式的 `latest.pt` 和编号 checkpoint 会恢复网络、优化器、计数器、PPO 自身随机数
状态和 PPO 超参数；`--device` 决定本次设备，`--seed` 仍用于新一轮仿真、Commander
和蓝方初始化，但不会覆盖完整续训时恢复的 PPO 采样随机流。观测/动作维度不是
90/3 时会拒绝加载。实际采用的 `resume_mode` 和 `policy_rng_restored` 会写入
`run_config.json`。

checkpoint 内部格式现为 v3，并绑定新版卫星仲裁与观测语义，防止把旧环境下的
checkpoint 错当成可完整续训的模型；这个内部兼容字段不会出现在新结果目录名中。
v1/v2 checkpoint 默认拒绝；显式加入 `--allow-legacy-resume` 时只把兼容的网络权重
作为初始化，不恢复旧 Adam 状态、旧计数器或旧超参数。续训会新建一套结果曲线
并从 round 1 编号。

默认 `--update-mode episode`，整轮轨迹一次更新，保证一轮内行为策略冻结且
`best.pt` 正是产生该轮训练分数的网络。也提供 `--update-mode rollout`，此时只会
在完整联合 step 提交后、达到 `--rollout-size` 才更新；该模式下单轮会混合多个
行为策略，`best.pt` 只能视为候选。查看全部参数：

```bash
docker run --rm \
  -v /home/ry/competition-platform-env/personal_train:/app/personal_train:ro \
  -v /home/ry/competition-platform-env/scenarios:/app/scenarios:ro \
  -w /app \
  competition:ppo \
  python3 personal_train/train_r9_ppo.py --help
```

## 输出文件

每次运行使用独立时间戳，不覆盖旧实验：

```text
personal_train/
├── results/<run-id>_<timestamp>/
│   ├── round_scores.txt          # 每轮最终正式得分，逐轮立即刷新
│   ├── round_scores.csv
│   ├── round_scores.svg          # 每轮正式得分曲线
│   ├── training_dashboard.svg    # 得分、K/T、训练回报、红方存活图
│   ├── ppo_metrics.csv
│   ├── round_metrics.jsonl
│   ├── summary_round_0001.json   # 每轮完整汇总
│   ├── latest_summary.json
│   ├── run_config.json
│   └── training.log
└── models/<run-id>_<timestamp>/
    ├── latest.pt                 # 每轮刷新
    ├── best.pt                   # episode模式下为产生最佳训练轮分数的行为网络
    ├── model_metadata.json
    └── checkpoints/round_XXXX.pt
```

按 Ctrl-C、`docker stop` 或调度器发送 SIGTERM 时会尽力保存 `interrupted.pt`
和 `interruption.json`；其他异常会写 `failure.json` 和 `failed.pt`。中途文件不保存
未完成 rollout，因此恢复时只作为权重初始化；正常的 `latest.pt` 和编号 checkpoint
是在回合更新完成后的安全边界保存，可完整恢复优化器、计数器与随机数状态。

SVG 是标准浏览器可直接打开的图表，不依赖 matplotlib，适合离线服务器。

`ppo_metrics.csv` 额外记录 GAE 更新的 `approx_kl`、clip fraction、explained
variance、当前学习率/熵系数、实际 epoch 数和 KL 是否提前停止。episode 模式下
每行就是该轮完整轨迹的一次更新统计。

## 九场景独立并发训练

`train_r9_multi_scenario.py` 是宿主机调度器。它默认把九个场景全部加入队列、每个
场景训练 100 轮。推荐用 `--resume-batch` 明确指定一个历史批次，此时 E01 到 H03
分别读取该批次各自场景目录内的 `best.pt`；每个场景仍拥有独立 PPO、optimizer、
RNG、结果和模型，不会把九个场景的经验混入同一网络。

必须由宿主机运行这个脚本，而不是先进入一个训练容器。原生仿真库会在固定的
`./Results/<模型>/<实体ID>/` 下写文件，而不同场景会复用实体 ID；调度器因此为
每个场景启动一个独立 Docker 容器和独立 `/app/Results` tmpfs，避免文件竞争。

先只验证完整计划，不创建目录或启动训练：

```bash
cd /home/ry/competition/personal_train
python3 train_r9_multi_scenario.py \
  --rounds 100 \
  --scenarios E01 E02 E03 M01 M02 M03 H01 H02 H03 \
  --resume-batch r9_multi_20260906_151415_061752 \
  --seed 3 \
  --gpu-ids 0,1 \
  --max-parallel 9 \
  --allow-gpu-sharing \
  --threads-per-worker 1 \
  --image personal-competition:runtime \
  --dry-run
```

确认计划后，删除最后一行的 `--dry-run` 即可正式启动。`--resume-batch` 会验证九个
场景的 checkpoint、元数据和来源，并把实际路径及哈希写入调度清单，便于复现实验。
旧接口 `--e01-resume /绝对路径/best.pt` 仍保留，用于只给 E01 指定初始化权重、其余
场景随机初始化；它不能与 `--resume-batch` 同时使用。两者都省略时，调度器仍可扫描
`models` 选择兼容的 E01 模型，但九场景续训不推荐依赖这种隐式选择。

当前服务器有两张 GPU，默认并发数也是 2：九个任务同时进入队列，每次运行两个，
完成后自动补入下一个。每个容器只看见分配给自己的单张 GPU。可以显式使用
`--max-parallel 9 --allow-gpu-sharing` 让九个全开，但会让多个进程争用两张 GPU、CPU
和内存，通常更慢，也更容易显存不足。

同一批次共享一次时间戳，但九个场景的结果和模型都是各自直接位于根目录下的
单层目录；批次级状态、日志和汇总单独放入 `launcher_runs/`：

```text
personal_train/
├── results/
│   ├── e01_r9_ppo_<timestamp>/ # E01 的文字、JSON、CSV、SVG
│   ├── e02_r9_ppo_<timestamp>/
│   └── ... h03_r9_ppo_<timestamp>/
├── models/
│   ├── e01_r9_ppo_<timestamp>/ # best.pt、latest.pt、checkpoints/
│   ├── e02_r9_ppo_<timestamp>/
│   └── ... h03_r9_ppo_<timestamp>/
└── launcher_runs/r9_multi_<timestamp>/
    ├── batch_config.json
    ├── batch_status.json
    ├── batch_scores.txt
    ├── batch_summary.csv
    ├── batch_summary.json
    ├── batch_dashboard.svg
    └── launcher_logs/E01.log ... H03.log
```

使用 `--resume-batch` 时，每个场景自己的源 `best.pt` 提供网络权重和保存时的 PPO
超参数；optimizer、计数器与随机流重新初始化，再训练 100 轮，新回合从 1 编号。
批次汇总分别保留历史初始分、本批次最佳分和两者中的最高分；
源 checkpoint 不会被覆盖。某个场景失败不会终止其他场景，错误码和日志会写入
批次状态。按 Ctrl-C 或向调度器发送 SIGTERM 时，它会先向所有活动容器转发
SIGTERM，给单场景训练器保存 `interrupted.pt` 的机会。

2026-09-07 实际启动批次为 `r9_multi_20260907_151202_805457`，九个 worker 均已
启动。其九个源 checkpoint 全部来自
`models/r9_multi_20260906_151415_061752/<SCENE>/best.pt`；已中止的
`r9_multi_20260907_seed3_100r` 不作为任何场景的续训来源。

上述主命令已经使用宿主 UID/GID 写 `personal_train`，新结果会直接属于当前用户；`/app/Results` 则是随容器删除的可写 tmpfs。这个组合适配本服务器的 root-squash 文件系统。不要再用容器 root 写宿主结果，也不要把 `/app` 整体设成只读。

## 验证

在已经包含依赖的镜像内运行：

```bash
docker run --rm --network none \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src=/home/ry/competition/personal_train,dst=/app/personal_train \
  --mount type=bind,src=/home/ry/competition/competition-platform-env,dst=/opt/competition-platform-env,readonly \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e COMPETITION_REPO_ROOT=/opt/competition-platform-env \
  personal-competition:runtime \
  python3 -m unittest -v \
    personal_train.test_layout \
    personal_train.test_personal_train \
    personal_train.test_multi_scenario
```

当前完整容器回归测试共 68 项。

## 有意保留的竞赛边界

- `target_slots=5` 未扩展；这是赛事方尚未回答的接口边界。
- 因此 R9 只保留给旧九场景；final20 的 24/36/48 个目标应使用本页开头的联合 PPO。
- 9500 无人船必须先由合法探测发现；本训练器不会从 `case_info` 提前读取其坐标。发现后 Commander 可以分配 L 弹，但因为 `target_slots=5` 不扩展，Actor 对该船没有独立详细槽位，只能使用 R9 的“船目标”任务上下文和其余合法自身/探测特征。
- 本训练器不启动后台规划进程。
- 修复后的 checkpoint 应继续通过本目录的 Agent/Encoder 进行评估。直接改回原 `main.py` 会恢复原来错误的特征生成和状态传递逻辑。
