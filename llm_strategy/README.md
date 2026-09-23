# LLM 一次性全局作战规划模块（V0）

LLM 是唯一策略制定者：每局在部署完成后调用**一次** GLM，输出整局完整
作战计划；Validator 只做 PASS/REJECT（绝不修正）；Executor 只按计划
字面执行（绝不再决策）；全部红方导弹底层机动固定**直行**（不发送任何
横向加速度指令）；计分继续完全由上游 `RunSummary` / `RewardTracker`
产生。不引入 R9、PPO、MAPPO 或任何人工启发式策略辅助 LLM。

## 权限映射（重要）

任务清单原要求把模块建在 `competition_envs/` 下并修改
`run.py` / `core/main.py`。由于 `competition_envs` 是**只读环境代码、禁止
修改**，本模块按本目录既有模式（同 `joint_game_env.py` /
`train_r9_ppo.py`）落在 `personal_train` 内，通过
`bootstrap.install_project_paths()` 只读复用上游，不改动上游任何一行：

| 清单要求 | 实际位置 |
|---|---|
| `competition_envs/llm_strategy/` | `personal_train/llm_strategy/`（内部结构一致） |
| 修改 `run.py` + `core/main.py` 新增 `llm_plan` 模式 | 独立入口 `personal_train/run_llm_plan.py`（复刻 `core/main.py` 主循环，红方替换为 LLM 规划；上游 R0--R9/PPO/MAPPO 行为零影响，已用 `git status` 验证上游无改动） |

其余条目（信息边界、schema、validator 语义、executor 语义、审计、
官方计分复用）全部按清单实现。

## 目录

```text
llm_strategy/
├── agent.py          # LLMPlanAgent(BaseAgent)：仅转发计划命令
├── commander.py      # LLMPlanCommander：首次 begin_step 一次性规划
├── state_builder.py  # 合法 BattleState（只整理事实）
├── plan_schema.py    # V0 严格 schema（entity/coordinate/event_entity）
├── validator.py      # 纯 PASS/REJECT
├── executor.py       # 逐字执行 + 事件绑定，不替换不补救
├── glm_client.py     # GLM_API_KEY/GLM_BASE_URL/GLM_MODEL + Mock
├── audit.py          # state/raw/plan(+sha256)/report/trace/metrics
├── prompts/system_prompt.md
└── tests/            # 55 项测试（含 E01 mock 端到端冒烟）
```

## 运行

```bash
# 无 API Key 的完整链路冒烟（Mock 计划走同一 Schema/Validator/Executor）
/home/amax/ry/competition/glibc-2.38/python3.11-glibc238 \
  personal_train/run_llm_plan.py \
    --scenario easy/E01 --seed 1 --run-id llm_e01 \
    --rounds 1 --render-mode none \
    --mock-plan path/to/plan.json

# 真实 GLM（每局恰好 1 次调用）
export GLM_API_KEY=...            # 必填
export GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4   # 可选
export GLM_MODEL=glm-4.5                                    # 可选
# 去掉 --mock-plan 重新运行上列命令
```

计划非法（解析或验证失败）时该局以 `termination_reason=invalid_plan`
终止并记入审计，不重试、不修复；运行退出码非 0。

## 信息边界（实现要点）

- `9400/9600` 仅来自 `TrainingEnv._get_init_ship_observation()`（当前引擎
  实际只放这两类）。
- `9500` 绝不从 scenario/case_info/manifest/`_get_observation()` 获取；
  只有出现在某红方平台 `self.detectInfo` 后才成为合法事件目标。
- StateBuilder 输入只有三类：开局目录、`begin_step` 收到的红方隔离观测、
  公开配置与武器规则；`detectInfo` 兼容 dataclass 与 dict。
- 实体识别只用数值 `type`，测试用随机/误导 `nameChn` 验证。
- 完整蓝方态势只被赛后 `RunSummary`/`RewardTracker` 使用，从不进入模型。

## 计划语义（V0）

- 每平台：`launch(at_step|never)` + `initial_target` + 固定步 `retarget_orders`
  + `satellite_steps`；`motion` 必须为 `"straight"`，否则整计划 INVALID。
- 目标引用三模式：`entity`（仅规划时合法已知的 id）、`coordinate`（直接
  经纬度）、`event_entity`（仅限 `new_detection` 规则内，绑定触发事件的
  实体及其合法轨迹坐标）。
- 触发器：`at_step`、`new_detection(entity_type, occurrence)`（按首次合法
  发现计次，同一步内按 id 升序）、`launched_steps_ago(platform_id, steps)`。
- 天眼：可在任意合法 step 请求（含非发射步）；校验每步至多 1 次、总量
  不超过 `satelliteMaxUseCount`。
- 平台死亡/未发射/目标不可解析 → 记录 execution failure，不换弹、不重规划。

## 审计（每局）

`results/<run-id>_<时间戳>/round_NNN/` 下：
`state_input.json`、`glm_raw_response.json`、`accepted_plan.json`（原样
保存 + `plan_sha256.txt`）、`validation_report.json`、
`execution_trace.jsonl`（step/platform_id/plan_rule_id/trigger/
planned_action/engine_action/success/failure_reason）、`llm_metrics.json`。
Runner 校验每局 `api_calls == 1`，否则整run标记失败。

## 测试

```bash
cd /home/amax/ry/competition
/home/amax/ry/competition/glibc-2.38/python3.11-glibc238 \
  -m unittest discover -s personal_train/llm_strategy/tests \
     -t /home/amax/ry/competition -p "test_*.py"
```

覆盖清单第 19 节全部 15 项（55 个用例；`test_smoke_e01` 用 mock 计划
端到端跑 E01 两局 × 40 步：30/44/10 枚分波发射、2 次非发射步天眼、
plan sha256 一致、trace 无横向机动指令、第 2 局不继承第 1 局状态）。
