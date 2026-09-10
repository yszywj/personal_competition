#!/usr/bin/env python3
"""Evaluate one joint PPO checkpoint over fixed, reproducible seed streams.

The evaluator never updates the policy.  It rebuilds the simulator once per
seed, verifies the checkpoint's complete environment contract, and writes only
compact evaluation reports below ``personal_train/results``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True

PERSONAL_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PERSONAL_ROOT / "results"
_PACKAGE_PARENT = PERSONAL_ROOT.parent
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from personal_train.bootstrap import (  # noqa: E402
    install_project_paths,
    prepare_runtime_directory,
    training_runtime_path,
    validate_personal_output_path,
)

install_project_paths()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from envengine.sdk.log import LogManager  # noqa: E402
from envengine.sdk.writer import (  # noqa: E402
    WriteConfig,
    get_writer,
    init_writer,
    write_immediately,
)
from scenarios.cases import load_reward_policy  # noqa: E402

from personal_train.joint_diagnostics import EpisodeActionDiagnostics  # noqa: E402
from personal_train.joint_game_env import JointGameEnv, JointGameStep  # noqa: E402
from personal_train.joint_policy import JointPPOPolicy  # noqa: E402
from personal_train.joint_rl_core import UnitPhase  # noqa: E402
from personal_train.train_joint_ppo import (  # noqa: E402
    _checkpoint_contract,
    _debug_max_steps_from_checkpoint_contract,
    _environment_contract,
    _episode_planning_credit,
    _game_config_from_checkpoint_contract,
    _sha256_file,
    load_profile,
    resolve_scenario,
)


LOGGER = logging.getLogger("personal_train.joint_eval")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_DEFAULT_SEEDS = tuple(range(1, 11))
_EPISODE_COLUMNS = (
    "seed",
    "episode",
    "global_episode",
    "mode",
    "score",
    "steps",
    "termination_reason",
    "elapsed_seconds",
    "team_return",
    "planning_reward_sum",
    "planning_reward_mean",
    "planning_eligible_units",
    "planning_local_credit_allocated",
    "planning_local_credit_unallocated",
    "motion_return",
    "sensor_return",
    "unit_reward_sum",
    "accepted_activations",
    "accepted_sensor_requests",
    "activation_yes_rate",
    "activation_rejected",
    "retarget_yes_rate",
    "movement_negative_fraction",
    "movement_neutral_fraction",
    "movement_positive_fraction",
    "movement_switch_rate",
    "placement_edge_fraction",
    "dominant_objective_slot",
    "dominant_objective_fraction",
    "sensor_stop_rate",
    "sensor_rejected",
)
_SEED_COLUMNS = (
    "seed",
    "episodes",
    "score_mean",
    "score_std",
    "score_median",
    "score_min",
    "score_max",
    "score_q05",
    "score_q25",
    "score_q75",
    "score_q95",
    "score_cvar_lower",
    "score_cvar_alpha",
    "score_cvar_tail_count",
    "steps_mean",
    "team_return_mean",
    "planning_reward_sum_mean",
    "planning_reward_mean",
    "planning_eligible_units_mean",
    "planning_local_credit_allocated_mean",
    "planning_local_credit_unallocated_mean",
    "motion_return_mean",
    "sensor_return_mean",
    "unit_reward_sum_mean",
    "accepted_activations_mean",
    "accepted_sensor_requests_mean",
)


def _parse_seeds(raw_values: Sequence[str]) -> tuple[int, ...]:
    values: list[int] = []
    for raw in raw_values:
        for token in str(raw).split(","):
            stripped = token.strip()
            if not stripped:
                raise ValueError("seed list contains an empty value")
            try:
                seed = int(stripped)
            except ValueError as error:
                raise ValueError(f"invalid seed: {stripped!r}") from error
            if not 0 <= seed <= 2**32 - 1:
                raise ValueError("seeds must be between 0 and 4294967295")
            values.append(seed)
    if not values:
        raise ValueError("at least one evaluation seed is required")
    if len(set(values)) != len(values):
        raise ValueError("evaluation seeds must be unique")
    return tuple(values)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a joint PPO checkpoint without training. The selected "
            "scenario must exactly match the checkpoint environment contract."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        required=True,
        help=(
            "Competition case ID/path or explicit scenario.json. Use an explicit "
            "case path when several case families share an ID such as E01."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=[",".join(str(value) for value in _DEFAULT_SEEDS)],
        metavar="SEED",
        help="Fixed seeds separated by spaces or commas (default: 1..10).",
    )
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Use the maximum-probability action at every policy branch (default).",
    )
    mode.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Sample policy actions from the learned distributions.",
    )
    parser.set_defaults(deterministic=True)
    parser.add_argument(
        "--blue-policy",
        default=None,
        help="Optional evaluation opponent; default is the checkpoint trainer contract.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--cvar-alpha", type=float, default=0.2)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--render-mode", choices=("none", "human"), default="none")
    parser.add_argument("--render-fps", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--disable-log-color", action="store_true")
    args = parser.parse_args(argv)

    if args.episodes_per_seed <= 0:
        parser.error("--episodes-per-seed must be positive")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if args.render_fps <= 0:
        parser.error("--render-fps must be positive")
    if not math.isfinite(args.cvar_alpha) or not 0.0 < args.cvar_alpha <= 1.0:
        parser.error("--cvar-alpha must be in (0, 1]")
    try:
        args.seeds = _parse_seeds(args.seeds)
    except ValueError as error:
        parser.error(str(error))
    if args.run_id is not None and not _RUN_ID.fullmatch(args.run_id):
        parser.error(
            "--run-id must start with an alphanumeric and contain only "
            "A-Z, a-z, 0-9, _, ., -"
        )
    if args.blue_policy is not None:
        args.blue_policy = str(args.blue_policy).strip()
        if not args.blue_policy:
            parser.error("--blue-policy must be non-empty")
    return args


def _contract_differences(saved: Any, current: Any, prefix: str = "") -> tuple[str, ...]:
    """Return stable leaf paths that differ between two nested contracts."""

    if isinstance(saved, Mapping) and isinstance(current, Mapping):
        differences: list[str] = []
        for key in sorted(set(saved) | set(current), key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in saved or key not in current:
                differences.append(child)
            else:
                differences.extend(
                    _contract_differences(saved[key], current[key], child)
                )
        return tuple(differences)
    if isinstance(saved, (tuple, list)) and isinstance(current, (tuple, list)):
        if len(saved) != len(current):
            return (prefix,)
        differences = []
        for index, (left, right) in enumerate(zip(saved, current)):
            differences.extend(
                _contract_differences(left, right, f"{prefix}[{index}]")
            )
        return tuple(differences)
    return () if saved == current else (prefix or "<root>",)


def _validate_environment_contract(
    saved: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    differences = _contract_differences(saved, current)
    if differences:
        preview = ", ".join(differences[:20])
        if len(differences) > 20:
            preview += f", ... (+{len(differences) - 20} more)"
        raise ValueError(
            "Checkpoint environment contract differs from the evaluation environment: "
            + preview
        )


def _distribution_summary(
    values: Sequence[float], *, cvar_alpha: float
) -> dict[str, float | int]:
    array = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("summary requires at least one value")
    if not bool(np.isfinite(array).all()):
        raise ValueError("summary values must be finite")
    if not math.isfinite(cvar_alpha) or not 0.0 < cvar_alpha <= 1.0:
        raise ValueError("cvar_alpha must be in (0, 1]")
    ordered = np.sort(array)
    tail_count = max(1, int(math.ceil(cvar_alpha * int(array.size))))
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "median": float(np.median(array)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "cvar_lower": float(np.mean(ordered[:tail_count])),
        "cvar_alpha": float(cvar_alpha),
        "cvar_tail_count": int(tail_count),
    }


def _seed_summary(
    seed: int,
    episodes: Sequence[Mapping[str, Any]],
    *,
    cvar_alpha: float,
) -> dict[str, Any]:
    score = _distribution_summary(
        [float(item["score"]) for item in episodes], cvar_alpha=cvar_alpha
    )
    row: dict[str, Any] = {"seed": int(seed), "episodes": len(episodes)}
    row.update({f"score_{name}": value for name, value in score.items() if name != "count"})
    row["planning_reward_mean"] = float(
        np.mean([float(item["planning_reward_mean"]) for item in episodes])
    )
    for name in (
        "steps",
        "team_return",
        "planning_reward_sum",
        "planning_eligible_units",
        "planning_local_credit_allocated",
        "planning_local_credit_unallocated",
        "motion_return",
        "sensor_return",
        "unit_reward_sum",
        "accepted_activations",
        "accepted_sensor_requests",
    ):
        row[f"{name}_mean"] = float(
            np.mean([float(item[name]) for item in episodes])
        )
    return row


def _resolve_result_directory(
    *, scenario_id: str, checkpoint: Path, run_id: str | None, timestamp: str
) -> tuple[str, Path]:
    label = run_id or f"{scenario_id.lower()}_joint_eval_{checkpoint.stem}"
    run_name = f"{label}_{timestamp}"
    result = validate_personal_output_path(RESULTS_ROOT / run_name)
    results_root = RESULTS_ROOT.resolve()
    if results_root not in result.parents:
        raise ValueError(f"Evaluation output must be below {results_root}")
    return run_name, result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(
    path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in columns})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_episode(result_dir: Path, record: Mapping[str, Any]) -> None:
    csv_path = result_dir / "per_episode.csv"
    with csv_path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_EPISODE_COLUMNS, extrasaction="ignore")
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow({name: record.get(name, "") for name in _EPISODE_COLUMNS})
        stream.flush()
        os.fsync(stream.fileno())
    with (result_dir / "per_episode.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(record), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _configure_logging(
    result_dir: Path, *, color: bool, verbose: bool
) -> logging.Handler:
    LogManager(color_enabled=color)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.FileHandler(result_dir / "evaluation.log", encoding="utf-8")
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    return handler


def _set_seed_stream(seed: int, *, blue_policy: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["SIMULATION_SEED"] = str(seed)
    os.environ["RED_POLICY_SEED"] = str(seed)
    os.environ["BLUE_POLICY_SEED"] = str(seed)
    os.environ["BLUE_POLICY"] = blue_policy


def _termination_reason(environment: JointGameEnv, final: JointGameStep) -> str:
    if final.team_truncated:
        return "debug_horizon" if environment.is_debug_horizon else "time_limit"
    if final.team_terminated:
        if (
            not environment.is_debug_horizon
            and environment.current_step >= environment.max_steps
        ):
            return "official_horizon"
        return "environment_done"
    raise RuntimeError("terminal joint step has no termination reason")


def _evaluate_episode(
    policy: JointPPOPolicy,
    environment: JointGameEnv,
    *,
    deterministic: bool,
    seed: int,
    episode: int,
    global_episode: int,
    progress_every: int,
) -> dict[str, Any]:
    observations = environment.reset()
    diagnostics = EpisodeActionDiagnostics(
        environment.space,
        objective_ids=environment.objective_ids,
        unit_ids=environment.unit_ids,
        unit_types=environment.unit_types,
    )
    team_return = 0.0
    unit_reward_sum = 0.0
    sensor_return = 0.0
    accepted_activations = 0
    accepted_sensor_requests = 0
    eligible_units = [False] * environment.space.unit_count
    assignment_duration = np.zeros(
        (environment.space.unit_count, environment.space.objective_count),
        dtype=np.int64,
    )
    final: JointGameStep | None = None
    start = time.perf_counter()
    for _ in range(environment.max_steps):
        states = environment.states
        mask = environment.action_mask()
        action, trace = policy.sample(
            observations, states, mask, deterministic=deterministic
        )
        decision_step = environment.current_step
        outcome = environment.step(action)
        for slot in outcome.accepted_activations:
            eligible_units[int(slot)] = True
        # Match the trainer's planning responsibility exactly: an assignment
        # accrues only while the post-step controller state is ACTIVE.
        assignment_states = (
            outcome.assignment_states
            if outcome.assignment_states is not None
            else environment.states
        )
        for slot, state in enumerate(assignment_states):
            objective_slot = int(state.current_objective_slot)
            if (
                state.phase == UnitPhase.ACTIVE
                and 0 <= objective_slot < environment.space.objective_count
            ):
                assignment_duration[slot, objective_slot] += 1
        sensor_reward = (
            float(outcome.team_reward)
            if outcome.sensor_reward is None
            else float(outcome.sensor_reward)
        )
        diagnostics.observe(
            states, mask, action, trace, outcome, step=decision_step
        )
        observations = outcome.observations
        final = outcome
        team_return += float(outcome.team_reward)
        unit_reward_sum += float(sum(outcome.rewards))
        sensor_return += sensor_reward
        accepted_activations += len(outcome.accepted_activations)
        accepted_sensor_requests += len(outcome.accepted_sensor_requests)
        if environment.current_step % progress_every == 0:
            LOGGER.info(
                "seed=%d episode=%d progress=%d/%d score=%.6f",
                seed,
                episode,
                environment.current_step,
                environment.max_steps,
                outcome.score,
            )
        if outcome.done:
            break
    elapsed = time.perf_counter() - start
    if final is None:
        raise RuntimeError("joint environment produced an empty evaluation episode")
    if not final.done:
        raise RuntimeError("joint environment reached max_steps without a terminal marker")
    planning = _episode_planning_credit(
        environment,
        final_observation=final.raw_observation,
        final_score=float(final.score),
        eligible_units=eligible_units,
        assignment_duration=assignment_duration,
    )
    planning_reward_sum = float(sum(planning.rewards_by_unit))
    action_diagnostics = diagnostics.finalize()
    record: dict[str, Any] = {
        "seed": int(seed),
        "episode": int(episode),
        "global_episode": int(global_episode),
        "mode": "deterministic" if deterministic else "stochastic",
        "score": float(final.score),
        "steps": int(environment.current_step),
        "termination_reason": _termination_reason(environment, final),
        "elapsed_seconds": float(elapsed),
        "team_return": float(team_return),
        "planning_reward_sum": planning_reward_sum,
        "planning_reward_mean": (
            planning_reward_sum / environment.space.unit_count
        ),
        "planning_eligible_units": int(
            planning.diagnostics["eligible_unit_count"]
        ),
        "planning_local_credit_allocated": float(
            planning.diagnostics["total_local_credit_allocated"]
        ),
        "planning_local_credit_unallocated": float(
            planning.diagnostics["total_local_credit_unallocated"]
        ),
        "motion_return": float(unit_reward_sum),
        "sensor_return": float(sensor_return),
        "unit_reward_sum": float(unit_reward_sum),
        "accepted_activations": int(accepted_activations),
        "accepted_sensor_requests": int(accepted_sensor_requests),
        "planning_credit": planning.diagnostics,
        "action_diagnostics": action_diagnostics,
    }
    record.update(action_diagnostics.get("csv_scalars", {}))
    return record


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Not a compatible joint PPO checkpoint: {path}")
    if checkpoint.get("algorithm") != JointPPOPolicy.ALGORITHM:
        raise ValueError(f"Checkpoint uses a different algorithm: {path}")
    if int(checkpoint.get("schema_version", -1)) != JointPPOPolicy.CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Checkpoint uses an unsupported schema: {path}")
    if not isinstance(checkpoint.get("joint_env_contract"), dict):
        raise ValueError("Checkpoint has no environment contract")
    trainer = checkpoint.get("trainer_contract")
    if trainer is not None and not isinstance(trainer, dict):
        raise ValueError("Checkpoint trainer contract is invalid")
    return {
        "algorithm": checkpoint["algorithm"],
        "schema_version": int(checkpoint["schema_version"]),
        "checkpoint_role": checkpoint.get("checkpoint_role"),
        "resume_safe": bool(checkpoint.get("resume_safe", False)),
        "update_count": int(checkpoint.get("update_count", 0)),
        "episode_count": int(checkpoint.get("episode_count", 0)),
        "transition_count": int(checkpoint.get("transition_count", 0)),
        "trainer_contract": dict(trainer) if isinstance(trainer, dict) else None,
    }


def _aggregate(
    episode_records: Sequence[Mapping[str, Any]],
    seed_records: Sequence[Mapping[str, Any]],
    *,
    cvar_alpha: float,
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "episode_score": _distribution_summary(
            [float(item["score"]) for item in episode_records],
            cvar_alpha=cvar_alpha,
        ),
        "seed_mean_score": _distribution_summary(
            [float(item["score_mean"]) for item in seed_records],
            cvar_alpha=cvar_alpha,
        ),
    }
    for name in (
        "steps",
        "team_return",
        "planning_reward_sum",
        "planning_reward_mean",
        "planning_eligible_units",
        "planning_local_credit_allocated",
        "planning_local_credit_unallocated",
        "motion_return",
        "sensor_return",
        "unit_reward_sum",
        "accepted_activations",
        "accepted_sensor_requests",
    ):
        aggregate[name] = _distribution_summary(
            [float(item[name]) for item in episode_records],
            cvar_alpha=cvar_alpha,
        )
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    invocation_cwd = Path.cwd()
    args = parse_args(argv)
    checkpoint_path = args.checkpoint.expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = invocation_cwd / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {checkpoint_path}")

    checkpoint_metadata = _checkpoint_metadata(checkpoint_path)
    saved_contract = dict(_checkpoint_contract(checkpoint_path))
    scenario_path = resolve_scenario(args.scenario, invocation_cwd)
    reward_policy = load_reward_policy(scenario_path)
    if reward_policy is None:
        raise ValueError(
            f"{scenario_path} has no case_info.json; only scored competition cases are supported."
        )
    trainer_contract = checkpoint_metadata.get("trainer_contract")
    saved_blue_policy = (
        trainer_contract.get("blue_policy")
        if isinstance(trainer_contract, Mapping)
        else None
    )
    blue_policy = args.blue_policy or saved_blue_policy
    if not isinstance(blue_policy, str) or not blue_policy.strip():
        raise ValueError(
            "Checkpoint has no valid blue-policy trainer contract; supply --blue-policy"
        )
    blue_policy = blue_policy.strip()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name, result_dir = _resolve_result_directory(
        scenario_id=reward_policy.scenario_id,
        checkpoint=checkpoint_path,
        run_id=args.run_id,
        timestamp=timestamp,
    )
    if result_dir.exists():
        raise FileExistsError(f"Evaluation output must be new: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=False)

    environment: JointGameEnv | None = None
    writer_initialized = False
    log_handler: logging.Handler | None = None
    episode_records: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []
    original_cwd = Path.cwd()
    started = time.perf_counter()
    try:
        log_handler = _configure_logging(
            result_dir,
            color=not args.disable_log_color,
            verbose=args.verbose,
        )
        runtime_dir = prepare_runtime_directory(training_runtime_path(result_dir))
        os.chdir(runtime_dir)
        init_writer(
            WriteConfig(
                output_dir=str(result_dir / "simulator_writer"),
                verbose=False,
                batch_size=100,
                enable_config=False,
                enable_state=False,
                enable_event=False,
                enable_ai_action=False,
            )
        )
        writer_initialized = True

        policy = JointPPOPolicy.from_checkpoint(
            checkpoint_path, device=args.device, load_optimizer=False
        )
        policy.set_training(False)
        game_config = _game_config_from_checkpoint_contract(saved_contract)
        max_steps_override = _debug_max_steps_from_checkpoint_contract(saved_contract)

        config_record = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "run_name": run_name,
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _sha256_file(checkpoint_path),
                **checkpoint_metadata,
            },
            "scenario": reward_policy.scenario_id,
            "scenario_path": str(scenario_path),
            "scenario_sha256": _sha256_file(scenario_path),
            "seeds": list(args.seeds),
            "episodes_per_seed": int(args.episodes_per_seed),
            "seed_protocol": (
                "each seed constructs a fresh simulator and resets Python/NumPy/Torch; "
                "episodes within that seed advance the same seeded random stream"
            ),
            "mode": "deterministic" if args.deterministic else "stochastic",
            "blue_policy": blue_policy,
            "blue_policy_source": "cli" if args.blue_policy is not None else "checkpoint",
            "device": str(policy.device),
            "cvar": {
                "alpha": float(args.cvar_alpha),
                "definition": "arithmetic mean of the lowest ceil(alpha * n) scores",
                "higher_score_is_better": True,
            },
            "checkpoint_environment_contract": saved_contract,
        }
        _atomic_json(result_dir / "evaluation_config.json", config_record)
        _atomic_json(
            result_dir / "status.json",
            {
                "status": "running",
                "completed_episodes": 0,
                "total_episodes": len(args.seeds) * args.episodes_per_seed,
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )

        global_episode = 0
        for seed in args.seeds:
            _set_seed_stream(seed, blue_policy=blue_policy)
            profile = load_profile(scenario_path)
            environment = JointGameEnv(
                profile,
                reward_policy=reward_policy,
                config=game_config,
                render_mode=None if args.render_mode == "none" else "human",
                render_fps=args.render_fps,
                max_steps_override=max_steps_override,
            )
            current_contract = _environment_contract(
                environment,
                scenario_id=reward_policy.scenario_id,
                scenario_path=scenario_path,
            )
            _validate_environment_contract(saved_contract, current_contract)
            if policy.space != environment.space:
                raise ValueError("Checkpoint joint space differs from evaluation environment")
            if policy.config.observation_dim != environment.observation_dim:
                raise ValueError(
                    "Checkpoint observation size differs from evaluation environment"
                )

            current_seed_records: list[dict[str, Any]] = []
            try:
                for episode in range(1, args.episodes_per_seed + 1):
                    global_episode += 1
                    record = _evaluate_episode(
                        policy,
                        environment,
                        deterministic=args.deterministic,
                        seed=seed,
                        episode=episode,
                        global_episode=global_episode,
                        progress_every=args.progress_every,
                    )
                    episode_records.append(record)
                    current_seed_records.append(record)
                    _append_episode(result_dir, record)
                    _atomic_json(
                        result_dir / "status.json",
                        {
                            "status": "running",
                            "completed_episodes": len(episode_records),
                            "total_episodes": len(args.seeds) * args.episodes_per_seed,
                            "last_seed": int(seed),
                            "last_episode": int(episode),
                            "updated_at": datetime.now().astimezone().isoformat(),
                        },
                    )
                    LOGGER.info(
                        "EVAL_RESULT seed=%d episode=%d score=%.6f steps=%d "
                        "launches=%d sensor=%d elapsed=%.3fs",
                        seed,
                        episode,
                        record["score"],
                        record["steps"],
                        record["accepted_activations"],
                        record["accepted_sensor_requests"],
                        record["elapsed_seconds"],
                    )
            finally:
                environment.close()
                environment = None
            seed_records.append(
                _seed_summary(seed, current_seed_records, cvar_alpha=args.cvar_alpha)
            )
            _atomic_csv(result_dir / "per_seed.csv", _SEED_COLUMNS, seed_records)

        aggregate = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "run_name": run_name,
            "scenario": reward_policy.scenario_id,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "mode": "deterministic" if args.deterministic else "stochastic",
            "blue_policy": blue_policy,
            "seeds": list(args.seeds),
            "episodes_per_seed": int(args.episodes_per_seed),
            "seed_count": len(args.seeds),
            "episode_count": len(episode_records),
            "statistics": _aggregate(
                episode_records, seed_records, cvar_alpha=args.cvar_alpha
            ),
        }
        _atomic_json(result_dir / "aggregate.json", aggregate)
        _atomic_json(
            result_dir / "status.json",
            {
                "status": "complete",
                "completed_episodes": len(episode_records),
                "total_episodes": len(episode_records),
                "aggregate": str(result_dir / "aggregate.json"),
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )
        LOGGER.info(
            "Evaluation complete: episodes=%d score_mean=%.6f score_std=%.6f",
            len(episode_records),
            aggregate["statistics"]["episode_score"]["mean"],
            aggregate["statistics"]["episode_score"]["std"],
        )
        print(f"RESULT_DIR {result_dir}")
        return 0
    except KeyboardInterrupt:
        _atomic_json(
            result_dir / "status.json",
            {
                "status": "interrupted",
                "completed_episodes": len(episode_records),
                "total_episodes": len(args.seeds) * args.episodes_per_seed,
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )
        LOGGER.warning("Evaluation interrupted after %d episodes", len(episode_records))
        return 130
    except Exception as error:
        LOGGER.exception("Joint evaluation failed: %s", error)
        _atomic_json(
            result_dir / "failure.json",
            {
                "time": datetime.now().astimezone().isoformat(),
                "completed_episodes": len(episode_records),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        _atomic_json(
            result_dir / "status.json",
            {
                "status": "failed",
                "completed_episodes": len(episode_records),
                "total_episodes": len(args.seeds) * args.episodes_per_seed,
                "error_type": type(error).__name__,
                "error": str(error),
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )
        raise
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception:
                LOGGER.exception("Environment close failed")
        if writer_initialized:
            try:
                write_immediately()
                get_writer().close()
            except Exception:
                LOGGER.exception("Writer close failed")
        os.chdir(original_cwd)
        if log_handler is not None:
            logging.getLogger().removeHandler(log_handler)
            log_handler.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "parse_args",
    "_aggregate",
    "_contract_differences",
    "_distribution_summary",
    "_evaluate_episode",
    "_parse_seeds",
    "_resolve_result_directory",
    "_seed_summary",
    "_validate_environment_contract",
]
