# 平级目录迁移与验证

状态：平级目录、只读上游挂载、新版加权损伤 reward 和训练侧回合重置兼容均已完成。
68 项容器内回归测试通过，九场景并行训练已完成启动验证。

## 当前目录约定

```text
/home/ry/competition/
├── competition-platform-env/   # 上游 Git 仓库，只读依赖
└── personal_train/             # 本仓库、模型、结果和运行输出
```

`bootstrap.py` 优先发现平级的 `competition-platform-env`，也可以通过
`COMPETITION_REPO_ROOT` 明确指定。指定路径失效会直接报错，不会自动回退镜像里的旧代码。
旧的嵌套/扁平布局仍可被定位以便检查，但当前写入保护要求个人输出位于上游之外；
运行应使用上述平级布局，或下述容器挂载布局。

所有结果/模型目录在创建前进行路径解析和边界验证。结果目录、模型目录或其父目录
若是指向上游的符号链接，会在写入前拒绝。源码导入禁用 Python 字节码缓存。
相对资源从个人结果目录下的 `runtime/` 访问；原生 `Results` 与源码资源分开，容器中
使用独立 tmpfs。此目录准备逻辑已经过文件系统单元测试和九个真实场景的启动验证。

## Docker 基础环境

检查现有镜像得到：

| 镜像 | 依赖结论 |
|---|---|
| `competition:ppo` | Torch/CUDA 可用，缺部分 X11 动态库 |
| `competition:ppo-ui` | Torch/CUDA 和 X11 动态库可用 |
| `competition:v1` | 没有 Torch，不满足个人工具依赖 |

没有需要迁移的上述镜像驻留容器；其他业务容器未操作。

已基于本机 `competition:ppo-ui` 离线构建 `personal-competition:runtime`。
这是依赖镜像，不是上游源码快照；本次没有重新下载 Torch、CUDA 或 apt 包。
它默认只打印提示，不会自动运行训练。需要重新构建时：

```bash
cd /home/ry/competition/personal_train
docker build --network=none --pull=false \
  -f Dockerfile.runtime -t personal-competition:runtime .
```

`.dockerignore` 将模型、结果和个人源码排除在该依赖镜像的构建上下文之外。
容器中的目录约定：

| 宿主目录/资源 | 容器路径 | 权限 |
|---|---|---|
| 上游仓库 | `/opt/competition-platform-env` | 只读 |
| 个人仓库 | `/app/personal_train` | 需要产生输出时可写 |
| 原生临时输出 | `/app/Results` | 每容器独立 tmpfs |

`COMPETITION_REPO_ROOT=/opt/competition-platform-env` 保证导入此处的 core/policies/scenarios，
而非基础镜像历史 `/app` 副本。已确认实际 `envengine.__file__` 和 `policies.__file__`
分别位于该只读上游目录。Torch 为 `2.11.0+cu126`，两张 RTX 2080 Ti 可见。
镜像仍包含其基础镜像的历史文件，但这些文件不应作为当前应用源码使用。

## 验证

宿主机运行不依赖 Torch 的单元测试：

```bash
cd /home/ry/competition
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  personal_train.test_layout personal_train.test_multi_scenario
```

完整回归测试可在依赖容器内执行；其中调度器测试使用模拟子进程，实际训练另按
下文的九场景命令验证：

```bash
docker run --rm --network none \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src=/home/ry/competition/personal_train,dst=/app/personal_train \
  --mount type=bind,src=/home/ry/competition/competition-platform-env,dst=/opt/competition-platform-env,readonly \
  personal-competition:runtime \
  python3 -m unittest -v \
    personal_train.test_layout \
    personal_train.test_personal_train \
    personal_train.test_multi_scenario
```

## 新版评分接口迁移

`personal_env.py` 已移除旧版 `DESTROYED_TARGET_WEIGHT` 和
`TIME_EFFICIENCY_WEIGHT` 依赖。正式分奖励不在首次摧毁时自行重算，而是逐步读取
当前只读上游 `RewardTracker` 的精确分差。因此，部分掉血也立即按 `5:2:1`
目标类型权重计分，时间不再影响奖励；`raw_official_score_delta` 保存未缩放的
0--100 正式分变化。因果毁伤、距离势函数和分弹种失败成本仍作为独立 shaping。

对应回归测试覆盖部分掉血、重复帧不重复计分和后续多目标掉血增量。

## 仿真器回合时钟

当前上游 `SimulatorFactory.reset_all()` 只重置模型，没有把各 simulator 的私有
`sim_time` 恢复到想定起始时间。个人环境现在在 `super().reset()` 返回后、首个动作
下发前，把所有 simulator 的时钟统一恢复为 `profile.imagineProfile.simTime`。
该覆盖完全位于 `personal_train`，上游目录保持不变，并有独立回归测试。

原 `AttackMissileAgent.reset()` 遗留的 `latest_observation` 也由
`PersonalR9PPOAttackAgent.reset()` 在个人层清空；共享 Commander 仍通过
`ResetAwareAgentManager` 按对象身份每回合只重置一次。

## 当前训练

2026-09-07 已在 `tmux` 会话 `ppo_daodan` 启动批次
`r9_multi_20260907_151202_805457`：E01/E02/E03/M01/M02/M03/H01/H02/H03 各
100 轮，统一种子 3，九个独立容器共享两张 GPU。九个场景都从长时间训练批次
`r9_multi_20260906_151415_061752` 中各自场景的 `best.pt` 初始化，而不是共享某一个
场景的模型。具体源路径和哈希保存在本次调度清单中。

已中止的 `r9_multi_20260907_seed3_100r` 明确不作为本次任何场景的 checkpoint
来源；其旧结果已归档到 `results/results_past/`，文档不再把它描述为运行中的批次。

结果和模型采用严格的单层场景命名，批次级调度文件与训练结果分开：

```text
/home/ry/competition/personal_train/results/e01_r9_ppo_20260907_151202_805457/
/home/ry/competition/personal_train/results/e02_r9_ppo_20260907_151202_805457/
... /home/ry/competition/personal_train/results/h03_r9_ppo_20260907_151202_805457/
/home/ry/competition/personal_train/models/e01_r9_ppo_20260907_151202_805457/
/home/ry/competition/personal_train/models/e02_r9_ppo_20260907_151202_805457/
... /home/ry/competition/personal_train/models/h03_r9_ppo_20260907_151202_805457/
/home/ry/competition/personal_train/launcher_runs/r9_multi_20260907_151202_805457/
```

九个 worker 均已启动。每个场景的新模型/结果目录使用同一个时间戳，但不存在
`results/r9_multi_.../<SCENE>/...` 或在场景目录内再套一层时间目录的布局。

## 仍待运行完成后验证

1. 当前批次的最终每场 100 条评分、最终 checkpoint 哈希和批次汇总只能
   在九个 worker 全部结束后验收。
2. UI 展示不属于本次无界面训练启动范围，尚未重新验证。

本次没有修改上游或桌面备份，也没有改写历史结果/模型。所有代码变更留在本仓库，
没有自动提交或推送 Git。
