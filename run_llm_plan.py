#!/usr/bin/env python3
"""Standalone runner for the LLM one-shot global planner (red mode `llm_plan`).

The upstream checkout (``competition_envs``) is read-only, so instead of
patching ``run.py``/``core/main.py`` this entry point reproduces their round
loop verbatim with two substitutions:

    RedBaselineCommander/AttackMissileAgent  ->  LLMPlanCommander/LLMPlanAgent
    (everything else, including DeployAgent, RunSummary and RewardTracker,
     is the unmodified upstream code path via TrainingEnv)

Usage (host runtime is the project's glibc-2.38 Python):

    /home/amax/ry/competition/glibc-2.38/python3.11-glibc238 \
      personal_train/run_llm_plan.py \
        --scenario easy/E01 \
        --seed 1 \
        --run-id llm_e01 \
        --rounds 1 \
        [--mock-plan path/to/plan.json] \
        [--render-mode none]

Real GLM mode requires GLM_API_KEY (+ optional GLM_BASE_URL, GLM_MODEL).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True

PERSONAL_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = PERSONAL_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from personal_train.bootstrap import (  # noqa: E402
    REPOSITORY_ROOT,
    install_project_paths,
    prepare_runtime_directory,
    training_runtime_path,
    validate_personal_output_path,
)

install_project_paths()

import numpy as np  # noqa: E402

from envengine import Profile, TrainingEnv  # noqa: E402
from envengine.sdk.log import LogManager  # noqa: E402
from envengine.sdk.writer import WriteConfig, get_writer, init_writer, write_immediately  # noqa: E402
from evaluation import RunSummary  # noqa: E402
from scenarios.cases import RewardTracker, load_reward_policy  # noqa: E402
from user_agents import DeployAgent  # noqa: E402

from personal_train.llm_strategy.agent import LLMPlanAgent  # noqa: E402
from personal_train.llm_strategy.audit import AuditWriter  # noqa: E402
from personal_train.llm_strategy.commander import LLMPlanCommander  # noqa: E402
from personal_train.llm_strategy.glm_client import GLMClient, LLMClientError, MockGLMClient  # noqa: E402
from personal_train.llm_strategy.state_builder import EnvironmentRules  # noqa: E402

LOGGER = logging.getLogger("personal_train.llm_strategy.run")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SCENARIO_ID_PATTERN = re.compile(r"^[EMH]\d{2}$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Run one or more episodes with the LLM one-shot planner.",
    )
    parser.add_argument(
        "--scenario",
        default="easy/E01",
        help=(
            "Case id such as easy/E01, medium/M01, hard/H01, a suite path "
            "like final20/easy/E01, or an explicit scenario.json path."
        ),
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--run-id", default="llm_plan")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--blue-policy", default="b0_fixed_ratio_random")
    parser.add_argument(
        "--mock-plan",
        type=Path,
        default=None,
        help="Plan JSON file used in place of a real GLM call (same pipeline).",
    )
    parser.add_argument("--render-mode", choices=("none", "human"), default="none")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Optional step cap for smoke tests; cannot exceed the official "
            "scenario horizon and never changes the official scoring."
        ),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=PERSONAL_ROOT / "llm_strategy" / "results",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        parser.error("--run-id has invalid characters")
    return args


def resolve_scenario(value: str) -> Path:
    supplied = Path(value).expanduser()
    candidates: list[Path] = []
    if supplied.is_absolute():
        candidates.append(supplied)
    else:
        candidates.append(Path.cwd() / supplied)
        candidates.append(REPOSITORY_ROOT / supplied)
    candidates.append(REPOSITORY_ROOT / "scenarios" / "cases" / supplied / "scenario.json")
    label = value.upper()
    if SCENARIO_ID_PATTERN.fullmatch(label):
        difficulty = {"E": "easy", "M": "medium", "H": "hard"}[label[0]]
        candidates.append(
            REPOSITORY_ROOT / "scenarios" / "cases" / difficulty / label / "scenario.json"
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot resolve scenario '{value}'. Tried:\n  {rendered}")


def load_profile(path: Path) -> Profile:
    with path.open("r", encoding="utf-8") as stream:
        return Profile.from_dict(json.load(stream))


def environment_rules(profile: Profile, reward_policy) -> EnvironmentRules:
    imagine = profile.imagineProfile
    map_area = imagine.mapArea
    return EnvironmentRules(
        max_steps=int(reward_policy.max_steps),
        sim_step_ms=int(imagine.simStep),
        map_lon_min=float(map_area.lonMin),
        map_lon_max=float(map_area.lonMax),
        map_lat_min=float(map_area.latMin),
        map_lat_max=float(map_area.latMax),
        satellite_max_use_count=int(
            getattr(imagine, "satelliteMaxUseCount", 100)
        ),
        satellite_active_minutes=float(
            getattr(imagine, "satelliteUseMinutes", 3.0)
        ),
        hit_increase_time_interval_minutes=float(
            getattr(imagine, "missileRateIncreaseTimeIntervalMinutes", 2.0)
        ),
        hit_increase_min_angle_deg=float(
            getattr(imagine, "missileRateIncreaseMinAngle", 30.0)
        ),
        hit_increase_max_fraction=float(
            getattr(imagine, "missileRateIncreaseMaxValue", 0.2)
        ),
        hit_decrease_time_interval_minutes=float(
            getattr(imagine, "missileRateDecreaseTimeIntervalMinutes", 2.0)
        ),
        hit_decrease_max_angle_deg=float(
            getattr(imagine, "missileRateDecreaseMaxAngle", 10.0)
        ),
        hit_decrease_max_fraction=float(
            getattr(imagine, "missileRateDecreaseMaxValue", 0.4)
        ),
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    invocation_cwd = Path.cwd()
    args = parse_args(argv)
    scenario_path = resolve_scenario(args.scenario)
    reward_policy = load_reward_policy(scenario_path)
    if reward_policy is None:
        raise ValueError(f"{scenario_path} has no case_info.json; not a scored case.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = validate_personal_output_path(
        args.result_root / f"{args.run_id}_{timestamp}"
    )
    run_root.mkdir(parents=True, exist_ok=False)
    audit = AuditWriter(run_root)

    LogManager(color_enabled=False)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(run_root / "llm_run.log", encoding="utf-8")],
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # ---- runtime sandbox (native library writes ./Results) ----------------
    runtime_dir = prepare_runtime_directory(training_runtime_path(run_root))
    os.chdir(runtime_dir)
    writer_ready = False
    environment: TrainingEnv | None = None
    exit_code = 0
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        os.environ["SIMULATION_SEED"] = str(args.seed)
        os.environ["RED_POLICY_SEED"] = str(args.seed)
        os.environ["BLUE_POLICY_SEED"] = str(args.seed)
        os.environ["BLUE_POLICY"] = str(args.blue_policy)

        init_writer(
            WriteConfig(
                output_dir=str(run_root / "simulator_writer"),
                verbose=False,
                batch_size=100,
                enable_config=False,
                enable_state=False,
                enable_event=False,
                enable_ai_action=False,
            )
        )
        writer_ready = True

        profile = load_profile(scenario_path)
        rules = environment_rules(profile, reward_policy)
        environment = TrainingEnv(profile, render_mode=args.render_mode)
        if int(environment.max_steps) != int(reward_policy.max_steps):
            raise ValueError(
                "engine horizon and case_info max_steps disagree: "
                f"{environment.max_steps} != {reward_policy.max_steps}"
            )
        step_limit = (
            int(environment.max_steps)
            if args.max_steps is None
            else min(int(args.max_steps), int(environment.max_steps))
        )

        init_observation_ship = environment._get_init_ship_observation()
        if args.mock_plan is not None:
            client = MockGLMClient(args.mock_plan)
        else:
            client = GLMClient.from_env()

        commander = LLMPlanCommander(
            init_ship_observation=init_observation_ship,
            client=client,
            rules=rules,
            audit=audit,
        )

        simulators = environment.engine.simulator_factory.get_all_simulators()
        red_simulators = sorted(
            (
                simulator
                for simulator in simulators
                if int(simulator.entity_ext.entity.entityType) in (21000, 21001, 21002)
                and int(getattr(simulator.entity_ext.entity, "sideId", 0)) == 0
            ),
            key=lambda simulator: int(simulator.entity_ext.entity.id),
        )
        if not red_simulators:
            raise RuntimeError("scenario contains no controllable red platforms")
        for index, simulator in enumerate(red_simulators):
            entity_id = int(simulator.entity_ext.entity.id)
            agent = LLMPlanAgent(
                index + 1,
                entity_id,
                init_observation_ship,
                commander=commander,
            )
            environment.agent_manager.register_agent(agent)
            commander.register_platform(entity_id)
        LOGGER.info(
            "registered %d LLM plan agents (red platforms)", len(red_simulators)
        )

        imagine = profile.imagineProfile
        deploy_agent = DeployAgent(
            -1,
            -1,
            {},
            imagine.redArea.coordinates,
            imagine.redArea.coordinatesHM,
        )
        environment.agent_manager.register_agent(deploy_agent)

        _atomic_json(
            run_root / "llm_run_config.json",
            {
                "created_at": datetime.now().astimezone().isoformat(),
                "scenario": reward_policy.scenario_id,
                "scenario_path": str(scenario_path),
                "rounds": int(args.rounds),
                "seed": int(args.seed),
                "blue_policy": str(args.blue_policy),
                "red_policy": "llm_plan",
                "red_motion": "straight",
                "mock_plan": (
                    str(args.mock_plan) if args.mock_plan is not None else None
                ),
                "model": getattr(client, "model", None),
                "step_limit": int(step_limit),
                "official_max_steps": int(reward_policy.max_steps),
                "audit_root": str(run_root),
                "runtime_dir": str(runtime_dir),
            },
        )

        invalid_plan_encountered = False
        api_call_violation = False
        for round_index in range(1, args.rounds + 1):
            LOGGER.info("round %d/%d", round_index, args.rounds)
            commander.reset()
            initial_observation = environment.reset()
            run_summary = RunSummary(
                scenario=reward_policy.scenario_id,
                policies={
                    "red": "llm_plan",
                    "red_motion": "straight",
                    "blue": str(args.blue_policy),
                },
                reward_tracker=RewardTracker(reward_policy),
            )
            run_summary.start(initial_observation)
            final_observation = initial_observation
            termination_reason = "time_limit"

            # Red deployment exactly as the upstream loop does it.
            environment.red_model_deploy()

            for step in range(1, step_limit + 1):
                obs, _rewards, done, _info = environment.step()
                final_observation = obs
                run_summary.update(step, obs)
                if commander.plan_rejected:
                    termination_reason = "invalid_plan"
                    invalid_plan_encountered = True
                    LOGGER.error(
                        "round %d aborted: %s", round_index, commander.rejection_reason
                    )
                    break
                if done:
                    termination_reason = "environment_done"
                    break

            red_launched = commander.launched_platform_count
            summary = run_summary.build(
                final_observation,
                termination_reason=termination_reason,
                red_launched=red_launched,
            )
            summary["llm_plan"] = {
                "api_calls": int(commander.api_call_count),
                "accepted_plan_sha256": commander.accepted_plan_sha256,
                "plan_rejected": bool(commander.plan_rejected),
                "rejection_reason": commander.rejection_reason,
            }
            filename = (
                "summary.json"
                if args.rounds == 1
                else f"summary_round_{round_index}.json"
            )
            summary_path = run_summary.write(run_root, summary, filename)
            print("FINAL_SUMMARY " + json.dumps(summary, ensure_ascii=False))
            LOGGER.info("round %d summary written to %s", round_index, summary_path)

            if commander.api_call_count != 1:
                api_call_violation = True
                LOGGER.error(
                    "round %d made %d LLM calls (must be exactly 1)",
                    round_index,
                    commander.api_call_count,
                )
            score_value = summary.get("score")
            final_score = (
                float(score_value.get("score", 0.0))
                if isinstance(score_value, dict)
                else float(score_value or 0.0)
            )
            commander.write_round_metrics(
                extra={
                    "termination_reason": termination_reason,
                    "final_score": final_score,
                }
            )
            write_immediately()

        status = "complete"
        if invalid_plan_encountered or api_call_violation:
            status = "failed"
            exit_code = 1
        _atomic_json(
            run_root / "llm_run_status.json",
            {
                "status": status,
                "rounds": int(args.rounds),
                "invalid_plan_encountered": invalid_plan_encountered,
                "api_call_violation": api_call_violation,
                "finished_at": datetime.now().astimezone().isoformat(),
            },
        )
        LOGGER.info("run finished with status=%s (audit: %s)", status, run_root)
    except LLMClientError as error:
        LOGGER.error("LLM client error: %s", error)
        _atomic_json(
            run_root / "llm_run_status.json",
            {
                "status": "failed",
                "error": str(error),
                "finished_at": datetime.now().astimezone().isoformat(),
            },
        )
        exit_code = 2
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception:
                LOGGER.exception("environment close failed")
        if writer_ready:
            try:
                get_writer().close()
            except Exception:
                LOGGER.exception("simulator writer close failed")
        os.chdir(invocation_cwd)

    print(f"LLM_AUDIT_DIR {run_root}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
