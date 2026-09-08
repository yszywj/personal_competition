# R9 PPO 独立训练器

## 与仿真器解耦的联合强化学习核心

新增的 [`joint_rl_core`](JOINT_RL_CORE.md) 包含实体生命周期、条件混合动作
掩码、全队共享传感资源、定长观测编码和原子轨迹校验。它不依赖旧 R9 目录、
Torch、Gymnasium 或竞赛仿真器，可以单独验证接口与 PPO 数据语义。具体用途、
集成边界和测试命令见链接文档。

本机 Conda 环境位于 `personal_train/.conda/competition-rl`。使用下面的
项目内激活脚本后，终端提示符显示为简短的 `(competition-rl)`：

```bash
source /home/amax/ry/competition/personal_train/activate_competition_rl.sh
```

该环境当前使用 Python 3.11、NumPy 2.4.6 和 PyTorch 2.11.0+cu126。
依赖检查、联合核心测试和实际 RTX 4090 运算均已通过。完整仿真仍会因上游缺少
Linux 版 `_CompCruiseMissileHPy.so` 而在导入时停止；这不是 Conda 依赖问题。

## 目录迁移状态（2026-09-07）

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

## Docker GPU 训练

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

checkpoint 内部仍保留格式版本标记，用于防止把旧的一步 TD checkpoint 错当成
可完整续训的模型；这个内部兼容字段不会出现在新结果目录名中。旧的一步 TD
checkpoint 默认拒绝；显式加入 `--allow-legacy-resume` 时只把兼容的网络权重
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
- 9500 无人船必须先由合法探测发现；本训练器不会从 `case_info` 提前读取其坐标。发现后 Commander 可以分配 L 弹，但因为 `target_slots=5` 不扩展，Actor 对该船没有独立详细槽位，只能使用 R9 的“船目标”任务上下文和其余合法自身/探测特征。
- 本训练器不启动后台规划进程。
- 修复后的 checkpoint 应继续通过本目录的 Agent/Encoder 进行评估。直接改回原 `main.py` 会恢复原来错误的特征生成和状态传递逻辑。
