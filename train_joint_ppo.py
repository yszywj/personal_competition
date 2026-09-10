#!/usr/bin/env python3
"""Train one masked joint PPO policy against the competition simulator.

The policy owns deployment/launch, target selection, retargeting, three-way
movement and the shared satellite-request schedule.  The upstream simulator is
used as a read-only dependency; every generated file stays below
``personal_train``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import signal
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True

# Support both ``python -m personal_train.train_joint_ppo`` and direct use.
PERSONAL_ROOT = Path(__file__).resolve().parent
_PACKAGE_PARENT = PERSONAL_ROOT.parent
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from personal_train.bootstrap import (  # noqa: E402
    REPOSITORY_ROOT,
    install_project_paths,
    prepare_runtime_directory,
    training_runtime_path,
    validate_personal_output_path,
)

install_project_paths()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from envengine import Profile  # noqa: E402
from envengine.sdk.log import LogManager  # noqa: E402
from envengine.sdk.writer import (  # noqa: E402
    WriteConfig,
    get_writer,
    init_writer,
    write_immediately,
)
from scenarios.cases import load_reward_policy  # noqa: E402

from personal_train.joint_diagnostics import EpisodeActionDiagnostics  # noqa: E402
from personal_train.joint_game_env import (  # noqa: E402
    JointGameConfig,
    JointGameEnv,
    JointGameStep,
)
from personal_train.joint_policy import JointPPOConfig, JointPPOPolicy  # noqa: E402
from personal_train.joint_reward_credit import (  # noqa: E402
    ObjectiveDamageRecord,
    ObjectiveParticipation,
    allocate_local_objective_credit,
    compute_objective_damage_contributions,
    mix_planning_rewards,
)
from personal_train.joint_rl_core import (  # noqa: E402
    JointTrajectoryBuffer,
    JointTransition,
    UnitPhase,
)


LOGGER = logging.getLogger("personal_train.joint")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SCENARIO_ID = re.compile(r"^[EMH]0[1-3]$")
_RESUME_SAFE_CHECKPOINT_ROLES = frozenset({"latest", "periodic"})
_ROUND_COLUMNS = (
    "round",
    "rollout_group",
    "rollout_episode_index",
    "rollout_episode_count",
    "rollout_transitions",
    "ppo_update_applied",
    "pending_rollout_episodes",
    "score",
    "steps",
    "termination_reason",
    "elapsed_seconds",
    "ppo_update_seconds",
    "round_compute_seconds",
    "team_return",
    "planning_reward_sum",
    "planning_reward_mean",
    "planning_eligible_units",
    "planning_local_credit_allocated",
    "planning_local_credit_unallocated",
    "motion_return",
    "sensor_return",
    "unit_reward_sum",
    "unit_reward_mean",
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
    "dominant_objective_id",
    "dominant_objective_fraction",
    "sensor_stop_rate",
    "sensor_rejected",
    "update_count",
    "episode_count",
    "transition_count",
    "policy_loss",
    "plan_policy_loss",
    "motion_policy_loss",
    "sensor_policy_loss",
    "value_loss",
    "plan_value_loss",
    "motion_value_loss",
    "sensor_value_loss",
    "entropy",
    "plan_entropy",
    "motion_entropy",
    "sensor_entropy",
    "approx_kl",
    "plan_approx_kl",
    "motion_approx_kl",
    "sensor_approx_kl",
    "max_approx_kl",
    "max_plan_approx_kl",
    "max_motion_approx_kl",
    "max_sensor_approx_kl",
    "early_stop_kl",
    "clip_fraction",
    "plan_clip_fraction",
    "motion_clip_fraction",
    "sensor_clip_fraction",
    "plan_decisions",
    "motion_decisions",
    "sensor_decisions",
    "plan_value_samples",
    "motion_value_samples",
    "sensor_value_samples",
    "plan_advantage_mean",
    "motion_advantage_mean",
    "sensor_advantage_mean",
    "grad_norm",
    "learning_rate",
    "early_stopped",
)

_UPDATE_COLUMNS = (
    "update_count",
    "round_start",
    "round_end",
    "episodes",
    "transitions",
    "elapsed_seconds",
    "policy_loss",
    "plan_policy_loss",
    "motion_policy_loss",
    "unit_policy_loss",
    "sensor_policy_loss",
    "value_loss",
    "plan_value_loss",
    "motion_value_loss",
    "sensor_value_loss",
    "entropy",
    "plan_entropy",
    "motion_entropy",
    "sensor_entropy",
    "approx_kl",
    "plan_approx_kl",
    "motion_approx_kl",
    "sensor_approx_kl",
    "max_approx_kl",
    "max_plan_approx_kl",
    "max_motion_approx_kl",
    "max_sensor_approx_kl",
    "early_stop_kl",
    "clip_fraction",
    "plan_clip_fraction",
    "motion_clip_fraction",
    "sensor_clip_fraction",
    "plan_decisions",
    "motion_decisions",
    "sensor_decisions",
    "plan_value_samples",
    "motion_value_samples",
    "sensor_value_samples",
    "plan_advantage_mean",
    "motion_advantage_mean",
    "sensor_advantage_mean",
    "grad_norm",
    "learning_rate",
    "entropy_coef",
    "epochs_ran",
    "minibatch_updates",
    "early_stopped",
)


@dataclass(frozen=True)
class EpisodeResult:
    """Collected episode and reporting values before the PPO update."""

    buffer: JointTrajectoryBuffer
    score: float
    steps: int
    termination_reason: str
    elapsed_seconds: float
    team_return: float
    planning_rewards: tuple[float, ...]
    motion_return: float
    sensor_return: float
    unit_reward_sum: float
    accepted_activations: int
    accepted_sensor_requests: int
    planning_credit: Mapping[str, Any]
    action_diagnostics: Mapping[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Train the end-to-end masked joint PPO controller. This is a "
            "single-process trainer; launch it with python, not torchrun."
        )
    )
    parser.add_argument(
        "--scenario",
        default="E01",
        help=(
            "Competition case ID (E01..H03), suite-qualified case path such "
            "as final20/easy/E01, "
            "or an explicit scenario.json path."
        ),
    )
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional output label; a timestamp is appended for normal runs.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Exact new result directory below personal_train (requires --model-dir).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Exact new model directory below personal_train (requires --result-dir).",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--blue-policy", default="b0_fixed_ratio_random")
    parser.add_argument(
        "--device",
        default="cuda",
        help="PPO device: cuda, cuda:N, auto, or cpu.",
    )
    checkpoint_source = parser.add_mutually_exclusive_group()
    checkpoint_source.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Safe update-boundary continuation checkpoint. Network, optimizer, "
            "policy RNG, counters, PPO settings and recorded game/trainer settings "
            "are restored; use latest.pt or a numbered checkpoint."
        ),
    )
    checkpoint_source.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help=(
            "Warm-start from checkpoint network weights while using this run's fresh "
            "optimizer, counters, RNG and CLI hyperparameters."
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--rollout-episodes",
        type=int,
        default=4,
        help="Complete episodes collected with one frozen policy before each PPO update.",
    )
    parser.add_argument(
        "--debug-max-steps",
        type=int,
        default=None,
        help="Short explicit smoke-test horizon; omitted means the official case horizon.",
    )

    # Joint PPO hyperparameters. Observation size is derived from the adapter.
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--learning-rate-final", type=float, default=1e-5)
    parser.add_argument("--learning-rate-decay-updates", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.20)
    parser.add_argument("--value-clip-ratio", type=float, default=0.20)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--sensor-policy-coef", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--entropy-final-coef", type=float, default=0.0002)
    parser.add_argument("--entropy-decay-updates", type=int, default=500)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--value-inference-batch-size", type=int, default=128)
    parser.add_argument("--unit-team-reward-weight", type=float, default=0.0)
    parser.add_argument("--placement-log-std-min", type=float, default=-5.0)
    parser.add_argument("--placement-log-std-max", type=float, default=1.0)

    # Game-facing observation, sensor and reward settings.
    parser.add_argument(
        "--objective-slots",
        type=int,
        default=None,
        help=(
            "Stable objective slots. By default this is max(18, the scored "
            "objective count), which supports final20 automatically."
        ),
    )
    parser.add_argument("--max-track-age-steps", type=int, default=300)
    parser.add_argument("--max-speed-mps", type=float, default=3000.0)
    parser.add_argument(
        "--sensor-capacity",
        type=int,
        default=100,
        help=(
            "Coordinated team cap per episode (default: 100, matching the "
            "current global competition limit)."
        ),
    )
    parser.add_argument("--sensor-max-requests-per-step", type=int, default=1)
    parser.add_argument("--sensor-cooldown-steps", type=int, default=0)
    parser.add_argument("--official-reward-scale", type=float, default=1.0)
    parser.add_argument("--progress-potential-scale", type=float, default=0.05)
    parser.add_argument(
        "--planning-team-weight",
        type=float,
        default=0.7,
        help="Weight of the final team score in each eligible unit's planning target.",
    )
    parser.add_argument(
        "--planning-local-weight",
        type=float,
        default=0.3,
        help=(
            "Weight of target-local credit after per-episode unit-count scale "
            "correction and clipping to [0, 1]; the unscaled conserved credit "
            "is retained in diagnostics."
        ),
    )
    parser.add_argument(
        "--sensor-information-potential-scale",
        type=float,
        default=0.015,
        help=(
            "Potential scale for fresh interceptor-track coverage on the current "
            "global backend (legacy backends retain objective-discovery shaping)."
        ),
    )
    parser.add_argument(
        "--no-early-stop-on-completion",
        dest="terminate_on_all_objectives_destroyed",
        action="store_false",
        default=True,
        help="Continue to the official horizon after all scored objectives are destroyed.",
    )

    parser.add_argument("--render-mode", choices=("none", "human"), default="none")
    parser.add_argument("--render-fps", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--disable-log-color", action="store_true")
    args = parser.parse_args(argv)

    positive_integer_names = (
        "rounds",
        "rollout_episodes",
        "checkpoint_every",
        "progress_every",
        "hidden_dim",
        "learning_rate_decay_updates",
        "entropy_decay_updates",
        "update_epochs",
        "minibatch_size",
        "value_inference_batch_size",
        "max_track_age_steps",
        "sensor_max_requests_per_step",
        "render_fps",
    )
    for name in positive_integer_names:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.objective_slots is not None and args.objective_slots <= 0:
        parser.error("--objective-slots must be positive")
    if args.debug_max_steps is not None and args.debug_max_steps <= 0:
        parser.error("--debug-max-steps must be positive")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 4294967295")
    if args.sensor_capacity is not None and args.sensor_capacity < 0:
        parser.error("--sensor-capacity must be non-negative")
    if args.sensor_cooldown_steps < 0:
        parser.error("--sensor-cooldown-steps must be non-negative")

    finite_names = (
        "learning_rate",
        "learning_rate_final",
        "gamma",
        "gae_lambda",
        "clip_ratio",
        "value_clip_ratio",
        "value_coef",
        "sensor_policy_coef",
        "entropy_coef",
        "entropy_final_coef",
        "target_kl",
        "max_grad_norm",
        "unit_team_reward_weight",
        "placement_log_std_min",
        "placement_log_std_max",
        "max_speed_mps",
        "official_reward_scale",
        "progress_potential_scale",
        "planning_team_weight",
        "planning_local_weight",
        "sensor_information_potential_scale",
    )
    if not all(math.isfinite(float(getattr(args, name))) for name in finite_names):
        parser.error("floating-point parameters must be finite")
    if args.learning_rate <= 0.0 or args.learning_rate_final <= 0.0:
        parser.error("learning rates must be positive")
    if args.learning_rate_final > args.learning_rate:
        parser.error("--learning-rate-final cannot exceed --learning-rate")
    if not 0.0 <= args.gamma <= 1.0 or not 0.0 <= args.gae_lambda <= 1.0:
        parser.error("--gamma and --gae-lambda must be between 0 and 1")
    if args.clip_ratio <= 0.0 or args.value_clip_ratio < 0.0:
        parser.error("PPO clipping ratios are invalid")
    for name in (
        "value_coef",
        "sensor_policy_coef",
        "entropy_coef",
        "entropy_final_coef",
        "target_kl",
        "unit_team_reward_weight",
        "official_reward_scale",
        "progress_potential_scale",
        "planning_team_weight",
        "planning_local_weight",
        "sensor_information_potential_scale",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if not math.isclose(
        args.planning_team_weight + args.planning_local_weight,
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        parser.error("--planning-team-weight and --planning-local-weight must sum to 1")
    if args.entropy_final_coef > args.entropy_coef:
        parser.error("--entropy-final-coef cannot exceed --entropy-coef")
    if args.max_grad_norm <= 0.0 or args.max_speed_mps <= 0.0:
        parser.error("--max-grad-norm and --max-speed-mps must be positive")
    if args.placement_log_std_min >= args.placement_log_std_max:
        parser.error("--placement-log-std-min must be below --placement-log-std-max")
    if args.run_id is not None and not _RUN_ID.fullmatch(args.run_id):
        parser.error(
            "--run-id must start with an alphanumeric and contain only A-Z, a-z, 0-9, _, ., -"
        )
    args.blue_policy = str(args.blue_policy).strip()
    if not args.blue_policy:
        parser.error("--blue-policy must be non-empty")
    if (args.result_dir is None) != (args.model_dir is None):
        parser.error("--result-dir and --model-dir must be supplied together")
    return args


def resolve_scenario(value: str, invocation_cwd: Path) -> Path:
    supplied = Path(value).expanduser()
    candidates: list[Path] = []
    if supplied.is_absolute():
        candidates.append(supplied)
    else:
        candidates.extend((invocation_cwd / supplied, REPOSITORY_ROOT / supplied))

    label = value.upper()
    if _SCENARIO_ID.fullmatch(label):
        difficulty = {"E": "easy", "M": "medium", "H": "hard"}[label[0]]
        candidates.append(
            REPOSITORY_ROOT
            / "scenarios"
            / "cases"
            / difficulty
            / label
            / "scenario.json"
        )
    candidates.append(
        REPOSITORY_ROOT / "scenarios" / "cases" / value / "scenario.json"
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Cannot resolve competition scenario '{value}'. Tried:\n  {rendered}"
    )


def load_profile(path: Path) -> Profile:
    with path.open("r", encoding="utf-8") as stream:
        return Profile.from_dict(json.load(stream))


def _resolve_run_directories(
    *,
    scenario_id: str,
    run_id: str | None,
    timestamp: str,
    result_dir: Path | None,
    model_dir: Path | None,
    invocation_cwd: Path,
) -> tuple[str, Path, Path]:
    if result_dir is None and model_dir is None:
        label = run_id or f"{scenario_id.lower()}_joint_ppo"
        run_name = f"{label}_{timestamp}"
        return (
            run_name,
            validate_personal_output_path(PERSONAL_ROOT / "results" / run_name),
            validate_personal_output_path(PERSONAL_ROOT / "models" / run_name),
        )
    if result_dir is None or model_dir is None:
        raise ValueError("result_dir and model_dir must be supplied together")

    def resolve(path: Path) -> Path:
        expanded = path.expanduser()
        return (
            expanded.resolve()
            if expanded.is_absolute()
            else (invocation_cwd / expanded).resolve()
        )

    exact_result = validate_personal_output_path(resolve(result_dir))
    exact_model = validate_personal_output_path(resolve(model_dir))
    if (
        exact_result == exact_model
        or exact_result in exact_model.parents
        or exact_model in exact_result.parents
    ):
        raise ValueError("Result and model directories must be separate, non-nested paths")
    return run_id or scenario_id.lower(), exact_result, exact_model


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


def _append_round_records(result_dir: Path, row: Mapping[str, Any]) -> None:
    """Durably append one compact CSV row and the complete JSON record."""

    csv_path = result_dir / "rounds.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_ROUND_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in _ROUND_COLUMNS})
        stream.flush()
        os.fsync(stream.fileno())
    jsonl_path = result_dir / "rounds.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _append_update_records(result_dir: Path, row: Mapping[str, Any]) -> None:
    """Durably record one PPO update over one or more complete episodes."""

    csv_path = result_dir / "updates.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_UPDATE_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in _UPDATE_COLUMNS})
        stream.flush()
        os.fsync(stream.fileno())
    jsonl_path = result_dir / "updates.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _configure_logging(
    result_dir: Path, *, color: bool, verbose: bool
) -> logging.Handler:
    LogManager(color_enabled=color)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.FileHandler(result_dir / "training.log", encoding="utf-8")
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    return handler


def _fresh_policy_config(args: argparse.Namespace, observation_dim: int) -> JointPPOConfig:
    return JointPPOConfig(
        observation_dim=observation_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        learning_rate_final=args.learning_rate_final,
        learning_rate_decay_updates=args.learning_rate_decay_updates,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_clip_ratio=args.value_clip_ratio,
        value_coef=args.value_coef,
        sensor_policy_coef=args.sensor_policy_coef,
        entropy_coef=args.entropy_coef,
        entropy_final_coef=args.entropy_final_coef,
        entropy_decay_updates=args.entropy_decay_updates,
        target_kl=args.target_kl,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        value_inference_batch_size=args.value_inference_batch_size,
        unit_team_reward_weight=args.unit_team_reward_weight,
        placement_log_std_min=args.placement_log_std_min,
        placement_log_std_max=args.placement_log_std_max,
        seed=args.seed,
        device=args.device,
    )


def _checkpoint_config(path: Path) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("algorithm") != JointPPOPolicy.ALGORITHM
        or int(checkpoint.get("schema_version", -1))
        != JointPPOPolicy.CHECKPOINT_SCHEMA_VERSION
        or not isinstance(checkpoint.get("config"), dict)
    ):
        raise ValueError(f"Not a compatible joint PPO checkpoint: {path}")
    return checkpoint["config"]


def _training_contract(
    *, seed: int, blue_policy: str, rollout_episodes: int
) -> dict[str, Any]:
    """Settings that define episode generation and PPO batch boundaries."""

    return {
        "version": 1,
        "seed": int(seed),
        "blue_policy": str(blue_policy),
        "rollout_episodes": int(rollout_episodes),
    }


def _resume_training_contract(path: Path) -> Mapping[str, Any]:
    """Validate that a checkpoint was committed at a resumable update boundary."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Not a compatible joint PPO checkpoint: {path}")
    role = checkpoint.get("checkpoint_role")
    if (
        checkpoint.get("resume_safe") is not True
        or role not in _RESUME_SAFE_CHECKPOINT_ROLES
    ):
        raise ValueError(
            "--resume requires latest.pt or a numbered checkpoint written at a "
            "complete PPO update boundary; use --init-from for best, interrupted, "
            "failed, or older unmarked checkpoints"
        )
    contract = checkpoint.get("trainer_contract")
    if not isinstance(contract, dict) or contract.get("version") != 1:
        raise ValueError("Resume checkpoint has no supported trainer contract")
    seed = contract.get("seed")
    rollout_episodes = contract.get("rollout_episodes")
    blue_policy = contract.get("blue_policy")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 2**32 - 1
        or isinstance(rollout_episodes, bool)
        or not isinstance(rollout_episodes, int)
        or rollout_episodes <= 0
        or not isinstance(blue_policy, str)
        or not blue_policy
    ):
        raise ValueError("Resume checkpoint trainer contract is invalid")
    return contract


def _game_config(
    args: argparse.Namespace,
    *,
    gamma: float,
    required_objective_slots: int = 0,
) -> JointGameConfig:
    if (
        isinstance(required_objective_slots, bool)
        or not isinstance(required_objective_slots, int)
        or required_objective_slots < 0
    ):
        raise ValueError("required_objective_slots must be a non-negative integer")
    objective_slots = (
        max(18, required_objective_slots)
        if args.objective_slots is None
        else int(args.objective_slots)
    )
    return JointGameConfig(
        objective_slots=objective_slots,
        max_track_age_steps=args.max_track_age_steps,
        max_speed_mps=args.max_speed_mps,
        sensor_capacity=args.sensor_capacity,
        sensor_max_requests_per_step=args.sensor_max_requests_per_step,
        sensor_cooldown_steps=args.sensor_cooldown_steps,
        official_reward_scale=args.official_reward_scale,
        progress_potential_scale=args.progress_potential_scale,
        planning_team_weight=args.planning_team_weight,
        planning_local_weight=args.planning_local_weight,
        sensor_information_potential_scale=(
            args.sensor_information_potential_scale
        ),
        gamma=gamma,
        terminate_on_all_objectives_destroyed=args.terminate_on_all_objectives_destroyed,
    )


def _game_config_from_checkpoint_contract(
    contract: Mapping[str, Any],
) -> JointGameConfig:
    """Restore game-facing settings for a full policy-state continuation."""

    try:
        space = contract["space"]
        observation = contract["observation"]
        sensor = contract["sensor"]
        reward = contract["reward"]
        return JointGameConfig(
            objective_slots=int(space["objective_count"]),
            max_track_age_steps=int(observation["max_track_age_steps"]),
            max_speed_mps=float(observation["max_speed_mps"]),
            sensor_capacity=int(sensor["coordinated_team_capacity"]),
            sensor_max_requests_per_step=int(sensor["max_requests_per_step"]),
            sensor_cooldown_steps=int(sensor["cooldown_steps"]),
            official_reward_scale=float(reward["official_reward_scale"]),
            progress_potential_scale=float(reward["progress_potential_scale"]),
            planning_team_weight=float(reward["planning_team_weight"]),
            planning_local_weight=float(reward["planning_local_weight"]),
            sensor_information_potential_scale=float(
                reward["sensor_information_potential_scale"]
            ),
            gamma=float(reward["gamma"]),
            terminate_on_all_objectives_destroyed=bool(
                reward["terminate_on_all_objectives_destroyed"]
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Resume checkpoint does not contain a complete game configuration"
        ) from error


def _debug_max_steps_from_checkpoint_contract(
    contract: Mapping[str, Any],
) -> int | None:
    """Restore a debug horizon; official-horizon runs use no override."""

    try:
        debug_horizon = contract["debug_horizon"]
        max_steps = contract["max_steps"]
        if not isinstance(debug_horizon, bool):
            raise TypeError("debug_horizon must be boolean")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise TypeError("max_steps must be a positive integer")
        return max_steps if debug_horizon else None
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Resume checkpoint does not contain a valid horizon configuration"
        ) from error


def _warm_start_contract_differences(
    saved: Mapping[str, Any], current: Mapping[str, Any]
) -> tuple[str, ...]:
    """Require the same tensor/schema contract while permitting new game tuning."""

    paths = (
        ("version",),
        ("scenario_id",),
        ("scenario_sha256",),
        ("objective_ids",),
        ("unit_ids",),
        ("unit_types",),
        ("space",),
        ("observation_dim",),
        ("feature_names",),
        ("max_steps",),
        ("debug_horizon",),
        ("central_detection_sharing",),
        ("observation",),
        ("sensor", "backend"),
        ("sensor", "backend_capacity_per_unit"),
        ("sensor", "backend_capacity_team"),
        ("sensor", "backend_active_minutes"),
        ("sensor", "effective_max_requests_per_step"),
        ("reward", "sensor_information_source"),
    )

    def value_at(source: Mapping[str, Any], path: tuple[str, ...]) -> Any:
        value: Any = source
        for name in path:
            if not isinstance(value, Mapping) or name not in value:
                return None
            value = value[name]
        return value

    return tuple(
        ".".join(path)
        for path in paths
        if value_at(saved, path) != value_at(current, path)
    )


@dataclass(frozen=True)
class _PlanningCreditResult:
    rewards_by_unit: tuple[float, ...]
    diagnostics: Mapping[str, Any]


def _episode_planning_credit(
    environment: JointGameEnv,
    *,
    final_observation: Mapping[str, Any],
    final_score: float,
    eligible_units: Sequence[bool],
    assignment_duration: np.ndarray,
) -> _PlanningCreditResult:
    """Build auditable end-of-episode targets for planning decisions.

    Target health is consumed only after an episode ends.  It is never added
    to policy observations.  Local objective credit is allocated by legal
    controller-owned assignment history and therefore conserves every
    target's weighted contribution.
    """

    unit_count = int(environment.space.unit_count)
    objective_count = int(environment.space.objective_count)
    eligible = tuple(bool(value) for value in eligible_units)
    if len(eligible) != unit_count:
        raise ValueError("planning eligibility must contain one value per unit")
    durations = np.asarray(assignment_duration, dtype=np.float64)
    if durations.shape != (unit_count, objective_count):
        raise ValueError("assignment duration has an invalid shape")
    if not bool(np.isfinite(durations).all()) or bool((durations < 0.0).any()):
        raise ValueError("assignment duration must be finite and non-negative")

    reward_policy = environment.reward_policy
    weight_by_id = dict(reward_policy.objective_weights) or {
        int(entity_id): 1.0 for entity_id in reward_policy.objective_ids
    }
    initial_health_by_id = dict(reward_policy.objective_initial_health) or {
        int(entity_id): 1.0 for entity_id in reward_policy.objective_ids
    }
    entities = final_observation.get("entities") or {}
    records: list[ObjectiveDamageRecord] = []
    objective_id_by_slot: dict[int, int] = {}
    for slot, entity_id_raw in enumerate(environment.objective_ids):
        if entity_id_raw is None:
            continue
        entity_id = int(entity_id_raw)
        initial_health = float(initial_health_by_id[entity_id])
        entity = entities.get(entity_id, entities.get(str(entity_id)))
        # Missing entities are scored as undamaged by the official tracker
        # (its +inf sentinel clips to zero damage), represented here by the
        # finite initial health required by ObjectiveDamageRecord.
        final_health = (
            float(entity.get("health", initial_health))
            if isinstance(entity, Mapping)
            else initial_health
        )
        records.append(
            ObjectiveDamageRecord(
                objective_slot=slot,
                weight=float(weight_by_id[entity_id]),
                initial_health=initial_health,
                final_health=final_health,
            )
        )
        objective_id_by_slot[slot] = entity_id

    contributions = compute_objective_damage_contributions(records)
    contribution_slots = {item.objective_slot for item in contributions}
    participations = tuple(
        ObjectiveParticipation(
            unit_slot=unit_slot,
            objective_slot=objective_slot,
            fallback_responsibility=float(durations[unit_slot, objective_slot]),
        )
        for unit_slot in range(unit_count)
        for objective_slot in sorted(contribution_slots)
        if durations[unit_slot, objective_slot] > 0.0
    )
    allocation = allocate_local_objective_credit(
        contributions,
        participations,
        unit_count=unit_count,
    )
    team_score_fraction = float(final_score) / 100.0
    if not math.isclose(
        allocation.total_contribution,
        team_score_fraction,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            "planning objective contributions do not reproduce the official score"
        )
    # The raw allocation is deliberately conserved and remains the audit
    # quantity below.  Its per-unit mean is team_score / eligible_count,
    # whereas the team term is broadcast at team_score to every eligible unit.
    # Rescale only the learning copy so the configured 0.7/0.3 mixture has the
    # stated order of magnitude instead of shrinking the local term by 1/N.
    eligible_unit_count = int(sum(eligible))
    local_learning_scale = float(max(eligible_unit_count, 1))
    learning_local_credit_unclipped = tuple(
        float(value) * local_learning_scale for value in allocation.credit_by_unit
    )
    # Concentrated credit can otherwise grow with the full team size and
    # overwhelm the plan critic through the shared encoder.  The normalized
    # official outcome is in [0, 1], so keep the learning copy on that scale.
    learning_local_credit = tuple(
        min(1.0, value) for value in learning_local_credit_unclipped
    )
    planning_rewards = mix_planning_rewards(
        team_score_fraction=team_score_fraction,
        local_credit_by_unit=learning_local_credit,
        eligible_units=eligible,
        team_weight=environment.config.planning_team_weight,
        local_weight=environment.config.planning_local_weight,
    )

    contribution_by_slot = {
        item.objective_slot: item for item in contributions
    }
    allocated_by_slot = dict(allocation.allocated_by_objective)
    unallocated_by_slot = dict(allocation.unallocated_by_objective)
    objective_rows = [
        {
            "objective_slot": slot,
            "objective_id": objective_id_by_slot[slot],
            "damage_fraction": contribution_by_slot[slot].damage_fraction,
            "normalized_contribution": (
                contribution_by_slot[slot].normalized_contribution
            ),
            "allocated_credit": allocated_by_slot[slot],
            "unallocated_credit": unallocated_by_slot[slot],
            "used_assignment_fallback": slot
            in allocation.fallback_objective_slots,
        }
        for slot in sorted(contribution_by_slot)
    ]
    assignment_rows = [
        {
            "unit_slot": unit_slot,
            "objective_slot": objective_slot,
            "objective_id": objective_id_by_slot[objective_slot],
            "steps": int(durations[unit_slot, objective_slot]),
        }
        for unit_slot in range(unit_count)
        for objective_slot in sorted(contribution_slots)
        if durations[unit_slot, objective_slot] > 0.0
    ]
    diagnostics: dict[str, Any] = {
        "team_score_fraction": team_score_fraction,
        "team_weight": float(environment.config.planning_team_weight),
        "local_weight": float(environment.config.planning_local_weight),
        "responsibility_mode": "assignment_duration_fallback",
        "eligible_unit_slots": [
            slot for slot, value in enumerate(eligible) if value
        ],
        "eligible_unit_count": eligible_unit_count,
        "planning_reward_by_unit": list(planning_rewards),
        "local_credit_learning_scale": local_learning_scale,
        "local_credit_learning_clipped_units": int(
            sum(value > 1.0 for value in learning_local_credit_unclipped)
        ),
        "learning_local_credit_by_unit": list(learning_local_credit),
        "local_credit_by_unit": list(allocation.credit_by_unit),
        "total_objective_contribution": allocation.total_contribution,
        "total_local_credit_allocated": allocation.total_allocated,
        "total_local_credit_unallocated": allocation.total_unallocated,
        "fallback_objective_slots": list(allocation.fallback_objective_slots),
        "objectives": objective_rows,
        "assignment_duration": assignment_rows,
    }
    return _PlanningCreditResult(
        rewards_by_unit=planning_rewards,
        diagnostics=diagnostics,
    )


def _collect_episode(
    policy: JointPPOPolicy,
    environment: JointGameEnv,
    *,
    round_index: int,
    progress_every: int,
) -> EpisodeResult:
    observations = environment.reset()
    buffer = JointTrajectoryBuffer(environment.space, environment.observation_dim)
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
    diagnostics = EpisodeActionDiagnostics(
        environment.space,
        objective_ids=environment.objective_ids,
        unit_ids=environment.unit_ids,
        unit_types=environment.unit_types,
    )
    final: JointGameStep | None = None
    start = time.perf_counter()
    for _ in range(environment.max_steps):
        states = environment.states
        mask = environment.action_mask()
        action, trace = policy.sample(observations, states, mask)
        decision_step = environment.current_step
        outcome = environment.step(action)
        accepted_activation_slots = frozenset(outcome.accepted_activations)
        for slot in accepted_activation_slots:
            eligible_units[slot] = True
        # Responsibility follows the post-step tracker state.  Requested
        # activations that the simulator rejected remain STAGED and therefore
        # cannot receive target credit.
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
            states,
            mask,
            action,
            trace,
            outcome,
            step=decision_step,
        )
        buffer.append_collected(
            JointTransition(
                observations=observations,
                states=states,
                mask=mask,
                action=action,
                trace=trace,
                rewards=outcome.rewards,
                team_reward=outcome.team_reward,
                next_observations=outcome.observations,
                terminated=outcome.terminated,
                truncated=outcome.truncated,
                team_terminated=outcome.team_terminated,
                team_truncated=outcome.team_truncated,
                plan_rewards=None,
                motion_rewards=outcome.rewards,
                sensor_reward=sensor_reward,
            )
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
                "Round %d progress: %d/%d score=%.6f launched=%d sensor=%d",
                round_index,
                environment.current_step,
                environment.max_steps,
                outcome.score,
                environment.launch_count,
                environment.sensor_request_count,
            )
        if outcome.done:
            break
    elapsed = time.perf_counter() - start
    if final is None:
        raise RuntimeError("joint environment produced an empty episode")
    if not final.done:
        raise RuntimeError("joint environment reached max_steps without a terminal marker")
    if final.team_truncated:
        reason = "debug_horizon" if environment.is_debug_horizon else "time_limit"
    elif final.team_terminated:
        reason = (
            "official_horizon"
            if not environment.is_debug_horizon
            and environment.current_step >= environment.max_steps
            else "environment_done"
        )
    else:
        raise RuntimeError("terminal joint step has no termination reason")
    planning = _episode_planning_credit(
        environment,
        final_observation=final.raw_observation,
        final_score=float(final.score),
        eligible_units=eligible_units,
        assignment_duration=assignment_duration,
    )
    finalized_buffer = buffer.with_episode_plan_rewards(planning.rewards_by_unit)
    buffer.clear()
    return EpisodeResult(
        buffer=finalized_buffer,
        score=float(final.score),
        steps=len(finalized_buffer),
        termination_reason=reason,
        elapsed_seconds=elapsed,
        team_return=team_return,
        planning_rewards=planning.rewards_by_unit,
        motion_return=unit_reward_sum,
        sensor_return=sensor_return,
        unit_reward_sum=unit_reward_sum,
        accepted_activations=accepted_activations,
        accepted_sensor_requests=accepted_sensor_requests,
        planning_credit=planning.diagnostics,
        action_diagnostics=diagnostics.finalize(),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_record(
    path: Path, *, expected_contract: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("algorithm") != JointPPOPolicy.ALGORITHM
        or int(checkpoint.get("schema_version", -1))
        != JointPPOPolicy.CHECKPOINT_SCHEMA_VERSION
        or not isinstance(checkpoint.get("network"), dict)
        or (
            expected_contract is not None
            and checkpoint.get("joint_env_contract") != dict(expected_contract)
        )
    ):
        raise ValueError(f"Saved checkpoint failed validation: {path}")
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "checkpoint_role": checkpoint.get("checkpoint_role"),
        "resume_safe": bool(checkpoint.get("resume_safe", False)),
        "update_count": int(checkpoint.get("update_count", 0)),
        "episode_count": int(checkpoint.get("episode_count", 0)),
    }


def _checkpoint_contract(path: Path) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    contract = checkpoint.get("joint_env_contract") if isinstance(checkpoint, dict) else None
    if not isinstance(contract, dict):
        raise ValueError(
            f"Joint PPO checkpoint has no environment contract and cannot be resumed: {path}"
        )
    return contract


def _environment_contract(
    environment: JointGameEnv,
    *,
    scenario_id: str,
    scenario_path: Path,
) -> dict[str, Any]:
    sensor = environment.tracker.sensor_state.config
    sensor_contract: dict[str, Any] = {
        "backend": str(environment.sensor_backend),
        "coordinated_team_capacity": int(sensor.capacity),
        "max_requests_per_step": int(sensor.max_requests_per_step),
        "cooldown_steps": int(sensor.cooldown_steps),
    }
    if environment.sensor_backend == "team_global":
        sensor_contract.update(
            {
                "backend_capacity_team": int(
                    environment.sensor_backend_capacity_team
                ),
                "backend_active_minutes": float(
                    environment.sensor_backend_active_minutes
                ),
                "effective_max_requests_per_step": 1,
            }
        )
    else:
        sensor_contract["backend_capacity_per_unit"] = int(
            environment.sensor_backend_capacity_per_unit
        )
    reward_contract: dict[str, Any] = {
        "official_reward_scale": float(environment.config.official_reward_scale),
        "progress_potential_scale": float(
            environment.config.progress_potential_scale
        ),
        "planning_team_weight": float(
            environment.config.planning_team_weight
        ),
        "planning_local_weight": float(
            environment.config.planning_local_weight
        ),
        "sensor_information_potential_scale": float(
            environment.config.sensor_information_potential_scale
        ),
        "gamma": float(environment.config.gamma),
        "terminate_on_all_objectives_destroyed": bool(
            environment.config.terminate_on_all_objectives_destroyed
        ),
    }
    if environment.sensor_backend == "team_global":
        reward_contract["sensor_information_source"] = str(
            environment.sensor_information_source
        )
    return {
        # Version 3 exactly describes the legacy per-unit satellite runtime.
        # Version 4 records the updated factory-global behavior while retaining
        # the same policy tensor schema and observation dimension.
        "version": 4 if environment.sensor_backend == "team_global" else 3,
        "scenario_id": str(scenario_id),
        "scenario_sha256": _sha256_file(scenario_path),
        "objective_ids": tuple(environment.objective_ids),
        "unit_ids": tuple(environment.unit_ids),
        "unit_types": tuple(environment.unit_types),
        "space": asdict(environment.space),
        "observation_dim": int(environment.observation_dim),
        "feature_names": tuple(environment.encoder.feature_names),
        "max_steps": int(environment.max_steps),
        "debug_horizon": bool(environment.is_debug_horizon),
        "central_detection_sharing": True,
        "official_scoring": {
            "objective_weights": tuple(environment.reward_policy.objective_weights),
            "objective_initial_health": tuple(
                environment.reward_policy.objective_initial_health
            ),
        },
        "observation": {
            "max_track_age_steps": int(environment.config.max_track_age_steps),
            "max_speed_mps": float(environment.config.max_speed_mps),
            **(
                {
                    "detected_threat_count_normalizer": float(
                        environment.encoder.config.detected_threat_count_normalizer
                    )
                }
                if environment.sensor_backend == "team_global"
                else {}
            ),
        },
        "sensor": sensor_contract,
        "reward": reward_contract,
    }


def _save_policy(
    policy: JointPPOPolicy,
    path: Path,
    *,
    environment_contract: Mapping[str, Any],
    trainer_contract: Mapping[str, Any],
    checkpoint_role: str,
    resume_safe: bool,
) -> None:
    """Atomically attach the simulator contract to every policy checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.policy-{os.getpid()}")
    temporary = path.with_name(f".{path.name}.contract-{os.getpid()}")
    try:
        policy.save(staging)
        checkpoint = torch.load(staging, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("Joint PPO save produced a non-dictionary checkpoint")
        checkpoint["joint_env_contract"] = dict(environment_contract)
        checkpoint["trainer_contract"] = dict(trainer_contract)
        checkpoint["checkpoint_role"] = str(checkpoint_role)
        checkpoint["resume_safe"] = bool(resume_safe)
        torch.save(checkpoint, temporary)
        verified = torch.load(temporary, map_location="cpu", weights_only=True)
        if (
            not isinstance(verified, dict)
            or verified.get("algorithm") != policy.ALGORITHM
            or verified.get("joint_env_contract") != dict(environment_contract)
            or verified.get("trainer_contract") != dict(trainer_contract)
            or verified.get("checkpoint_role") != str(checkpoint_role)
            or verified.get("resume_safe") is not bool(resume_safe)
            or not isinstance(verified.get("network"), dict)
        ):
            raise ValueError(f"Joint PPO checkpoint verification failed: {path}")
        os.replace(temporary, path)
    finally:
        staging.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def _write_model_metadata(
    model_dir: Path,
    *,
    policy: JointPPOPolicy,
    latest_round: int,
    best_round: int,
    best_score: float,
    status: str,
) -> None:
    _atomic_json(
        model_dir / "model_metadata.json",
        {
            "algorithm": policy.ALGORITHM,
            "schema_version": policy.CHECKPOINT_SCHEMA_VERSION,
            "latest_round": int(latest_round),
            "best_round": int(best_round),
            "best_official_score": float(best_score),
            "update_count": int(policy.update_count),
            "transition_count": int(policy.transition_count),
            "episode_count": int(policy.episode_count),
            "device": str(policy.device),
            "status": status,
        },
    )


def _resolve_optional_path(path: Path | None, invocation_cwd: Path) -> Path | None:
    if path is None:
        return None
    expanded = path.expanduser()
    return (
        expanded.resolve()
        if expanded.is_absolute()
        else (invocation_cwd / expanded).resolve()
    )


def _recovery_checkpoint(
    model_dir: Path,
    *,
    resume_path: Path | None,
    init_from_path: Path | None,
) -> tuple[Path | None, str | None]:
    """Choose the newest known-consistent checkpoint after an aborted run."""

    candidates = (
        (model_dir / "latest.pt", "resume"),
        (resume_path, "resume"),
        (model_dir / "best.pt", "init_from"),
        (init_from_path, "init_from"),
    )
    for candidate, mode in candidates:
        if candidate is not None and candidate.is_file():
            return candidate, mode
    return None, None


def main(argv: Sequence[str] | None = None) -> int:
    invocation_cwd = Path.cwd()
    args = parse_args(argv)
    scenario_path = resolve_scenario(args.scenario, invocation_cwd)
    reward_policy = load_reward_policy(scenario_path)
    if reward_policy is None:
        raise ValueError(
            f"{scenario_path} has no case_info.json; only scored competition cases are supported."
        )
    resume_path = _resolve_optional_path(args.resume, invocation_cwd)
    init_from_path = _resolve_optional_path(args.init_from, invocation_cwd)
    source_path = resume_path or init_from_path
    resume_config: Mapping[str, Any] | None = None
    saved_environment_contract: Mapping[str, Any] | None = None
    saved_trainer_contract: Mapping[str, Any] | None = None
    if source_path is not None:
        if not source_path.is_file():
            label = "Resume" if resume_path is not None else "Initialization"
            raise FileNotFoundError(f"{label} checkpoint does not exist: {source_path}")
        resume_config = _checkpoint_config(source_path)
        saved_environment_contract = _checkpoint_contract(source_path)
        if resume_path is not None:
            saved_trainer_contract = _resume_training_contract(resume_path)
    effective_seed = (
        int(saved_trainer_contract["seed"])
        if saved_trainer_contract is not None
        else int(args.seed)
    )
    effective_blue_policy = (
        str(saved_trainer_contract["blue_policy"])
        if saved_trainer_contract is not None
        else str(args.blue_policy)
    )
    effective_rollout_episodes = (
        int(saved_trainer_contract["rollout_episodes"])
        if saved_trainer_contract is not None
        else int(args.rollout_episodes)
    )
    effective_debug_max_steps = (
        _debug_max_steps_from_checkpoint_contract(saved_environment_contract)
        if resume_path is not None and saved_environment_contract is not None
        else args.debug_max_steps
    )
    effective_gamma = (
        float(resume_config["gamma"])
        if resume_path is not None and resume_config is not None
        else float(args.gamma)
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name, result_dir, model_dir = _resolve_run_directories(
        scenario_id=reward_policy.scenario_id,
        run_id=args.run_id,
        timestamp=timestamp,
        result_dir=args.result_dir,
        model_dir=args.model_dir,
        invocation_cwd=invocation_cwd,
    )
    if result_dir.exists() or model_dir.exists():
        raise FileExistsError(
            "Training output must be new; refusing to overwrite "
            f"result={result_dir} model={model_dir}"
        )
    result_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)

    environment: JointGameEnv | None = None
    policy: JointPPOPolicy | None = None
    environment_contract: dict[str, Any] | None = None
    trainer_contract: dict[str, Any] | None = None
    writer_initialized = False
    log_handler: logging.Handler | None = None
    completed_rounds = 0
    best_round = 0
    best_score = float("-inf")
    active_round = 0
    latest_update_round = 0
    pending_rollout_episodes = 0
    pending_rollout_transitions = 0
    interrupted = False
    update_in_progress = False
    round_commit_in_progress = False
    policy_state_consistent = True
    termination_signal: int | None = None
    old_handlers = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal termination_signal
        termination_signal = int(signum)
        # Finish a completed round's report transaction before honoring a stop.
        # PPO updates include several optimizer steps, so this also prevents a
        # half-updated network from being presented as resumable.
        if update_in_progress or round_commit_in_progress:
            return
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        log_handler = _configure_logging(
            result_dir,
            color=not args.disable_log_color,
            verbose=args.verbose,
        )
        runtime_dir = prepare_runtime_directory(training_runtime_path(result_dir))
        os.chdir(runtime_dir)

        random.seed(effective_seed)
        np.random.seed(effective_seed)
        torch.manual_seed(effective_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(effective_seed)
        os.environ["SIMULATION_SEED"] = str(effective_seed)
        os.environ["RED_POLICY_SEED"] = str(effective_seed)
        os.environ["BLUE_POLICY_SEED"] = str(effective_seed)
        os.environ["BLUE_POLICY"] = effective_blue_policy

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

        profile = load_profile(scenario_path)
        game_config = (
            _game_config_from_checkpoint_contract(saved_environment_contract)
            if resume_path is not None and saved_environment_contract is not None
            else _game_config(
                args,
                gamma=effective_gamma,
                required_objective_slots=len(reward_policy.objective_ids),
            )
        )
        environment = JointGameEnv(
            profile,
            reward_policy=reward_policy,
            config=game_config,
            render_mode=None if args.render_mode == "none" else "human",
            render_fps=args.render_fps,
            max_steps_override=effective_debug_max_steps,
        )
        environment_contract = _environment_contract(
            environment,
            scenario_id=reward_policy.scenario_id,
            scenario_path=scenario_path,
        )
        if resume_path is not None and saved_environment_contract != environment_contract:
            differing = sorted(
                key
                for key in set(saved_environment_contract or {}) | set(environment_contract)
                if (saved_environment_contract or {}).get(key)
                != environment_contract.get(key)
            )
            raise ValueError(
                "Resume checkpoint environment contract differs from this run: "
                + ", ".join(differing)
            )
        if init_from_path is not None and saved_environment_contract is not None:
            differing = _warm_start_contract_differences(
                saved_environment_contract, environment_contract
            )
            if differing:
                raise ValueError(
                    "Warm-start checkpoint tensor/schema contract differs from this run: "
                    + ", ".join(differing)
                )
        if source_path is None:
            policy = JointPPOPolicy(
                environment.space,
                _fresh_policy_config(args, environment.observation_dim),
            )
            resume_mode = "fresh"
        elif resume_path is not None:
            policy = JointPPOPolicy.from_checkpoint(
                resume_path,
                device=args.device,
                load_optimizer=True,
            )
            if policy.space != environment.space:
                raise ValueError(
                    "Resume checkpoint joint space differs from the selected scenario/config"
                )
            if policy.config.observation_dim != environment.observation_dim:
                raise ValueError(
                    "Resume checkpoint observation size differs from the game adapter"
                )
            resume_mode = "full_policy_state"
            if not math.isclose(args.gamma, policy.config.gamma):
                LOGGER.warning(
                    "Resume uses checkpoint gamma %.8f; requested --gamma %.8f is ignored",
                    policy.config.gamma,
                    args.gamma,
                )
        else:
            policy = JointPPOPolicy(
                environment.space,
                _fresh_policy_config(args, environment.observation_dim),
            )
            policy.load_weights(init_from_path)
            resume_mode = "weights_only_warm_start"
        policy.set_training(True)
        trainer_contract = _training_contract(
            seed=effective_seed,
            blue_policy=effective_blue_policy,
            rollout_episodes=effective_rollout_episodes,
        )
        if saved_trainer_contract is not None:
            requested_settings = (
                ("--seed", args.seed, effective_seed),
                ("--blue-policy", args.blue_policy, effective_blue_policy),
                (
                    "--rollout-episodes",
                    args.rollout_episodes,
                    effective_rollout_episodes,
                ),
                (
                    "--debug-max-steps",
                    args.debug_max_steps,
                    effective_debug_max_steps,
                ),
                (
                    "--planning-team-weight",
                    args.planning_team_weight,
                    game_config.planning_team_weight,
                ),
                (
                    "--planning-local-weight",
                    args.planning_local_weight,
                    game_config.planning_local_weight,
                ),
                (
                    "--sensor-information-potential-scale",
                    args.sensor_information_potential_scale,
                    game_config.sensor_information_potential_scale,
                ),
            )
            for option, requested, restored in requested_settings:
                if requested != restored:
                    LOGGER.warning(
                        "Resume restores %s=%r from checkpoint; requested value %r is ignored",
                        option,
                        restored,
                        requested,
                    )

        run_config = {
            "created_at": datetime.now().astimezone().isoformat(),
            "run_name": run_name,
            "scenario": reward_policy.scenario_id,
            "scenario_path": str(scenario_path),
            "rounds": args.rounds,
            "rollout_episodes": effective_rollout_episodes,
            "seed": effective_seed,
            "blue_policy": effective_blue_policy,
            "red_policy": "joint_masked_ppo",
            "launcher": "single_process_python",
            "unit_count": environment.space.unit_count,
            "objective_slots": environment.space.objective_count,
            "observation_dim": environment.observation_dim,
            "official_max_steps": int(reward_policy.max_steps),
            "effective_max_steps": environment.max_steps,
            "debug_horizon": environment.is_debug_horizon,
            "central_detection_sharing": True,
            "sensor_backend": environment.sensor_backend,
            "sensor_backend_capacity_per_unit": (
                environment.sensor_backend_capacity_per_unit
            ),
            "sensor_backend_capacity_team": environment.sensor_backend_capacity_team,
            "sensor_backend_active_minutes": environment.sensor_backend_active_minutes,
            "sensor_information_source": environment.sensor_information_source,
            "detected_threat_count_normalizer": (
                environment.encoder.config.detected_threat_count_normalizer
            ),
            "sensor_coordinated_max_requests_per_step": (
                game_config.sensor_max_requests_per_step
            ),
            "action_semantics": {
                "activation": "deployment position + launch target + first movement in one simulator step",
                "active_unit": "optional retarget + three-way movement",
                "satellite": (
                    "one masked team scheduling head; the simulator executes one "
                    "team-global window per accepted request"
                    if environment.sensor_backend == "team_global"
                    else "one masked team scheduling head over per-unit satellite windows"
                ),
            },
            "observation_semantics": {
                "hidden_targets": "become valid only through controlled-unit detections",
                "detection_sharing": "central union across the controlled red team",
                "ground_truth_actor_access": False,
            },
            "reward_semantics": {
                "planning": (
                    "terminal official outcome plus target-local credit; raw local "
                    "credit is conserved for audit, then unit-count scaled and clipped "
                    "to [0, 1] for learning; assigned only to units with an accepted "
                    "activation"
                ),
                "motion": "official score delta plus distance potential shaping",
                "sensor": (
                    "official score delta plus legally observed fresh-interceptor-track "
                    "potential"
                    if environment.sensor_backend == "team_global"
                    else "official score delta plus legal known-objective information potential"
                ),
                "true_terminal_potential": "closed_to_zero",
                "debug_truncation_potential": "bootstrapped",
            },
            "game": asdict(game_config),
            "joint_env_contract": environment_contract,
            "trainer_contract": trainer_contract,
            "ppo": asdict(policy.config),
            "result_dir": str(result_dir),
            "model_dir": str(model_dir),
            "runtime_dir": str(runtime_dir),
            "resume": str(resume_path) if resume_path else None,
            "init_from": str(init_from_path) if init_from_path else None,
            "resume_mode": resume_mode,
            "requested_resume_overrides": (
                {
                    "seed": args.seed,
                    "blue_policy": args.blue_policy,
                    "rollout_episodes": args.rollout_episodes,
                    "debug_max_steps": args.debug_max_steps,
                    "planning_team_weight": args.planning_team_weight,
                    "planning_local_weight": args.planning_local_weight,
                    "sensor_information_potential_scale": (
                        args.sensor_information_potential_scale
                    ),
                }
                if resume_path is not None
                else None
            ),
            "externally_managed_output": args.result_dir is not None,
            "render_mode": args.render_mode,
            "render_fps": args.render_fps,
        }
        _atomic_json(result_dir / "run_config.json", run_config)
        _atomic_json(
            result_dir / "status.json",
            {
                "status": "running",
                "completed_rounds": 0,
                "active_round": 0,
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )
        LOGGER.info(
            "Scenario=%s rounds=%d units=%d objectives=%d horizon=%d%s",
            reward_policy.scenario_id,
            args.rounds,
            environment.space.unit_count,
            environment.space.objective_count,
            environment.max_steps,
            " (debug)" if environment.is_debug_horizon else "",
        )
        LOGGER.info(
            "Joint PPO device=%s observation_dim=%d hidden_dim=%d",
            policy.device,
            policy.config.observation_dim,
            policy.config.hidden_dim,
        )
        LOGGER.info(
            "Stable rollout: episodes_per_update=%d learning_rate=%.8f target_kl=%.4f "
            "minibatch=%d sensor_capacity=%d",
            effective_rollout_episodes,
            policy.config.learning_rate,
            policy.config.target_kl,
            policy.config.minibatch_size,
            game_config.sensor_capacity,
        )
        LOGGER.info(
            "Sensor backend=%s backend_team_capacity=%d active_minutes=%s "
            "threat_count_normalizer=%.1f",
            environment.sensor_backend,
            environment.sensor_backend_capacity_team,
            environment.sensor_backend_active_minutes,
            environment.encoder.config.detected_threat_count_normalizer,
        )
        LOGGER.info("Results: %s", result_dir)
        LOGGER.info("Models: %s", model_dir)

        rollout_buffer = JointTrajectoryBuffer(
            environment.space, environment.observation_dim
        )
        rollout_start_round = 1
        last_checkpoint_round = 0
        for round_index in range(1, args.rounds + 1):
            active_round = round_index
            LOGGER.info("Round %d/%d reset", round_index, args.rounds)
            episode = _collect_episode(
                policy,
                environment,
                round_index=round_index,
                progress_every=args.progress_every,
            )
            if not math.isfinite(episode.score) or not 0.0 <= episode.score <= 100.0:
                raise ValueError(f"Official score is invalid: {episode.score!r}")
            round_commit_in_progress = True
            write_immediately()

            is_new_best = episode.score > best_score
            if is_new_best:
                # This is the exact behavior network that generated the score.
                _save_policy(
                    policy,
                    model_dir / "best.pt",
                    environment_contract=environment_contract,
                    trainer_contract=trainer_contract,
                    checkpoint_role="best_behavior",
                    resume_safe=False,
                )
                best_score = episode.score
                best_round = round_index

            if pending_rollout_episodes == 0:
                rollout_start_round = round_index
            # Episode collection has already defensively copied and validated
            # every transition.  The snapshots are immutable, so the rollout
            # buffer can share them instead of copying all observation/mask
            # arrays for a third time.
            rollout_buffer.extend_snapshots(episode.buffer)
            episode.buffer.clear()
            pending_rollout_episodes += 1
            pending_rollout_transitions = len(rollout_buffer)
            rollout_group = (round_index - 1) // effective_rollout_episodes + 1
            rollout_episode_index = pending_rollout_episodes
            rollout_episode_count = min(
                effective_rollout_episodes, args.rounds - rollout_start_round + 1
            )
            ppo_update_applied = bool(
                pending_rollout_episodes >= effective_rollout_episodes
                or round_index == args.rounds
            )
            ppo_metrics: Mapping[str, float] = {}
            ppo_update_seconds = 0.0
            if ppo_update_applied:
                update_in_progress = True
                policy_state_consistent = False
                batch_episodes = pending_rollout_episodes
                batch_transitions = len(rollout_buffer)
                LOGGER.info(
                    "PPO update starting: rounds=%d-%d episodes=%d transitions=%d "
                    "epochs=%d minibatch=%d",
                    rollout_start_round,
                    round_index,
                    batch_episodes,
                    batch_transitions,
                    policy.config.update_epochs,
                    policy.config.minibatch_size,
                )
                update_started = time.perf_counter()
                ppo_metrics = policy.finish_rollout(
                    rollout_buffer, episode_count=batch_episodes
                )
                policy_state_consistent = True
                ppo_update_seconds = time.perf_counter() - update_started
                latest_update_round = round_index
                pending_rollout_episodes = 0
                pending_rollout_transitions = 0
                # Commit the newly completed optimizer state before report I/O.
                # If CSV/JSON logging then fails, recovery still retains this
                # complete rollout update instead of falling back one batch.
                _save_policy(
                    policy,
                    model_dir / "latest.pt",
                    environment_contract=environment_contract,
                    trainer_contract=trainer_contract,
                    checkpoint_role="latest",
                    resume_safe=True,
                )
                _append_update_records(
                    result_dir,
                    {
                        "update_count": policy.update_count,
                        "round_start": rollout_start_round,
                        "round_end": round_index,
                        "episodes": batch_episodes,
                        "transitions": batch_transitions,
                        "elapsed_seconds": ppo_update_seconds,
                        **{key: float(value) for key, value in ppo_metrics.items()},
                    },
                )
                LOGGER.info(
                    "PPO update %d finished in %.3fs for rounds %d-%d",
                    policy.update_count,
                    ppo_update_seconds,
                    rollout_start_round,
                    round_index,
                )
            row: dict[str, Any] = {
                "round": round_index,
                "rollout_group": rollout_group,
                "rollout_episode_index": rollout_episode_index,
                "rollout_episode_count": rollout_episode_count,
                "rollout_transitions": (
                    int(ppo_metrics.get("joint_steps", 0.0))
                    if ppo_update_applied
                    else pending_rollout_transitions
                ),
                "ppo_update_applied": ppo_update_applied,
                "pending_rollout_episodes": pending_rollout_episodes,
                "score": episode.score,
                "steps": episode.steps,
                "termination_reason": episode.termination_reason,
                "elapsed_seconds": episode.elapsed_seconds,
                "ppo_update_seconds": ppo_update_seconds,
                "round_compute_seconds": episode.elapsed_seconds + ppo_update_seconds,
                "team_return": episode.team_return,
                "planning_reward_sum": float(sum(episode.planning_rewards)),
                "planning_reward_mean": (
                    float(sum(episode.planning_rewards))
                    / environment.space.unit_count
                ),
                "planning_eligible_units": int(
                    episode.planning_credit["eligible_unit_count"]
                ),
                "planning_local_credit_allocated": float(
                    episode.planning_credit["total_local_credit_allocated"]
                ),
                "planning_local_credit_unallocated": float(
                    episode.planning_credit["total_local_credit_unallocated"]
                ),
                "motion_return": episode.motion_return,
                "sensor_return": episode.sensor_return,
                "unit_reward_sum": episode.unit_reward_sum,
                "unit_reward_mean": episode.unit_reward_sum / environment.space.unit_count,
                "accepted_activations": episode.accepted_activations,
                "accepted_sensor_requests": episode.accepted_sensor_requests,
                "planning_credit": episode.planning_credit,
                "action_diagnostics": episode.action_diagnostics,
                **dict(episode.action_diagnostics["csv_scalars"]),
                "dominant_objective_id": episode.action_diagnostics["target"][
                    "dominant_objective_id"
                ],
                "update_count": policy.update_count,
                "episode_count": policy.episode_count,
                "transition_count": policy.transition_count,
                **{key: float(value) for key, value in ppo_metrics.items()},
            }
            _append_round_records(result_dir, row)
            completed_rounds = round_index

            if ppo_update_applied:
                crossed_checkpoint_interval = (
                    round_index // args.checkpoint_every
                    > last_checkpoint_round // args.checkpoint_every
                )
                if crossed_checkpoint_interval or round_index == args.rounds:
                    _save_policy(
                        policy,
                        model_dir / "checkpoints" / f"round_{round_index:04d}.pt",
                        environment_contract=environment_contract,
                        trainer_contract=trainer_contract,
                        checkpoint_role="periodic",
                        resume_safe=True,
                    )
                    last_checkpoint_round = round_index
            _write_model_metadata(
                model_dir,
                policy=policy,
                latest_round=latest_update_round,
                best_round=best_round,
                best_score=best_score,
                status="running" if round_index < args.rounds else "complete",
            )
            _atomic_json(
                result_dir / "status.json",
                {
                    "status": "running" if round_index < args.rounds else "complete",
                    "completed_rounds": completed_rounds,
                    "active_round": round_index,
                    "best_round": best_round,
                    "best_score": best_score,
                    "update_count": policy.update_count,
                    "pending_rollout_episodes": pending_rollout_episodes,
                    "pending_rollout_transitions": pending_rollout_transitions,
                    "updated_at": datetime.now().astimezone().isoformat(),
                },
            )
            LOGGER.info(
                "ROUND_RESULT round=%d score=%.6f steps=%d launches=%d sensor=%d "
                "ppo_update=%s policy_loss=%.6f value_loss=%.6f collect=%.3fs update=%.3fs",
                round_index,
                episode.score,
                episode.steps,
                episode.accepted_activations,
                episode.accepted_sensor_requests,
                ppo_update_applied,
                float(ppo_metrics.get("policy_loss", 0.0)),
                float(ppo_metrics.get("value_loss", 0.0)),
                episode.elapsed_seconds,
                ppo_update_seconds,
            )
            update_in_progress = False
            round_commit_in_progress = False
            if termination_signal is not None:
                raise KeyboardInterrupt

        _atomic_json(
            model_dir / "checkpoint_validation.json",
            {
                "algorithm": policy.ALGORITHM,
                "schema_version": policy.CHECKPOINT_SCHEMA_VERSION,
                "checkpoints": {
                    "best": _checkpoint_record(
                        model_dir / "best.pt",
                        expected_contract=environment_contract,
                    ),
                    "latest": _checkpoint_record(
                        model_dir / "latest.pt",
                        expected_contract=environment_contract,
                    ),
                },
            },
        )
    except KeyboardInterrupt:
        interrupted = True
        recovery_path, recovery_mode = _recovery_checkpoint(
            model_dir,
            resume_path=resume_path,
            init_from_path=init_from_path,
        )
        if policy_state_consistent:
            LOGGER.warning("Training interrupted; preserving the current joint policy")
        else:
            LOGGER.warning(
                "Training interrupted inside an incomplete PPO update; the partial "
                "network will not be saved"
            )
        interruption = {
            "time": datetime.now().astimezone().isoformat(),
            "completed_rounds": completed_rounds,
            "active_round": active_round,
            "active_step": environment.current_step if environment is not None else 0,
            "pending_rollout_episodes": pending_rollout_episodes,
            "pending_rollout_transitions": pending_rollout_transitions,
            "policy_state_consistent": policy_state_consistent,
            "signal": termination_signal,
        }
        if (
            policy is not None
            and environment_contract is not None
            and trainer_contract is not None
            and policy_state_consistent
        ):
            try:
                interrupted_path = model_dir / "interrupted.pt"
                _save_policy(
                    policy,
                    interrupted_path,
                    environment_contract=environment_contract,
                    trainer_contract=trainer_contract,
                    checkpoint_role="interrupted",
                    resume_safe=False,
                )
                if recovery_path is None:
                    recovery_path = interrupted_path
                    recovery_mode = "init_from"
                _write_model_metadata(
                    model_dir,
                    policy=policy,
                    latest_round=latest_update_round,
                    best_round=best_round,
                    best_score=best_score if math.isfinite(best_score) else 0.0,
                    status="interrupted",
                )
            except Exception:
                LOGGER.exception("Could not save the interrupted joint PPO checkpoint")
        interruption.update(
            {
                "recovery_checkpoint": str(recovery_path) if recovery_path else None,
                "recovery_mode": recovery_mode,
            }
        )
        _atomic_json(result_dir / "interruption.json", interruption)
        _atomic_json(
            result_dir / "status.json",
            {
                "status": "interrupted",
                "completed_rounds": completed_rounds,
                "active_round": active_round,
                "pending_rollout_episodes": pending_rollout_episodes,
                "pending_rollout_transitions": pending_rollout_transitions,
                "policy_state_consistent": policy_state_consistent,
                "recovery_checkpoint": str(recovery_path) if recovery_path else None,
                "recovery_mode": recovery_mode,
                "signal": termination_signal,
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )
    except Exception as error:
        LOGGER.exception("Joint training failed: %s", error)
        recovery_path, recovery_mode = _recovery_checkpoint(
            model_dir,
            resume_path=resume_path,
            init_from_path=init_from_path,
        )
        failure = {
            "time": datetime.now().astimezone().isoformat(),
            "completed_rounds": completed_rounds,
            "active_round": active_round,
            "active_step": environment.current_step if environment is not None else 0,
            "pending_rollout_episodes": pending_rollout_episodes,
            "pending_rollout_transitions": pending_rollout_transitions,
            "policy_state_consistent": policy_state_consistent,
            "recovery_checkpoint": str(recovery_path) if recovery_path else None,
            "recovery_mode": recovery_mode,
            "termination_signal": termination_signal,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if (
            policy is not None
            and environment_contract is not None
            and trainer_contract is not None
            and policy_state_consistent
        ):
            try:
                failed_path = model_dir / "failed.pt"
                _save_policy(
                    policy,
                    failed_path,
                    environment_contract=environment_contract,
                    trainer_contract=trainer_contract,
                    checkpoint_role="failed",
                    resume_safe=False,
                )
                if recovery_path is None:
                    recovery_path = failed_path
                    recovery_mode = "init_from"
                _write_model_metadata(
                    model_dir,
                    policy=policy,
                    latest_round=latest_update_round,
                    best_round=best_round,
                    best_score=best_score if math.isfinite(best_score) else 0.0,
                    status="failed",
                )
            except Exception:
                LOGGER.exception("Could not save the failed joint PPO checkpoint")
        elif policy is not None and not policy_state_consistent:
            LOGGER.error("Skipped failed.pt because PPO state may be partially updated")
        failure.update(
            {
                "recovery_checkpoint": str(recovery_path) if recovery_path else None,
                "recovery_mode": recovery_mode,
            }
        )
        _atomic_json(result_dir / "failure.json", failure)
        _atomic_json(result_dir / "status.json", {"status": "failed", **failure})
        raise
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception:
                LOGGER.exception("Environment close failed")
        if writer_initialized:
            try:
                get_writer().close()
            except Exception:
                LOGGER.exception("Simulator writer close failed")
        os.chdir(invocation_cwd)
        for signum, old_handler in old_handlers.items():
            signal.signal(signum, old_handler)
        if log_handler is not None:
            root = logging.getLogger()
            root.removeHandler(log_handler)
            log_handler.close()

    LOGGER.info(
        "Training finished: completed_rounds=%d best_round=%d best_score=%.6f interrupted=%s",
        completed_rounds,
        best_round,
        best_score if math.isfinite(best_score) else 0.0,
        interrupted,
    )
    print(f"RESULT_DIR {result_dir}")
    print(f"MODEL_DIR {model_dir}")
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
