#!/usr/bin/env python3
"""Standalone, competition-compatible PPO trainer for the R9 hierarchy."""

from __future__ import annotations

import argparse
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
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

# Support both ``python -m personal_train.train_r9_ppo`` and direct execution.
PERSONAL_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PERSONAL_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from personal_train.bootstrap import CORE_ROOT, install_project_paths  # noqa: E402

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
from evaluation import RunSummary  # noqa: E402
from policies.red import initial_targets_from_observation  # noqa: E402
from scenarios.cases import RewardTracker, load_reward_policy  # noqa: E402
from user_agents import DeployAgent  # noqa: E402

from personal_train.personal_agent import PersonalR9PPOAttackAgent  # noqa: E402
from personal_train.personal_commander import PersonalR9Commander  # noqa: E402
from personal_train.personal_env import (  # noqa: E402
    RED_MISSILE_TYPES,
    PersonalR9TrainingEnv,
    R9RewardConfig,
)
from personal_train.ppo_policy import PPOConfig, PPOSharedPolicy  # noqa: E402
from personal_train.reporting import TrainingReporter  # noqa: E402


LOGGER = logging.getLogger("personal_train")
R9_POLICY = "r9_hierarchical_learning"
OBSERVATION_DIM = 90
ACTION_DIM = 3
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
CHECKPOINT_SCHEMA = {
    "name": "personal-r9-ppo",
    "version": 2,
    "algorithm": "personal_ppo_gae_v2",
    "reward": "official_score_delta_v2",
    "observation_dim": OBSERVATION_DIM,
    "action_dim": ACTION_DIM,
    "target_slots": 5,
    "agent_id_scheme": "red-contiguous-1-based",
    "commander": "r9-with-detected-9500",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the personal PPO policy under R9 without editing project code."
        )
    )
    parser.add_argument(
        "--scenario",
        default="E01",
        help="Competition case ID (E01..H03), case path such as easy/E01, or scenario.json path.",
    )
    parser.add_argument("--rounds", type=int, default=100, help="Number of training rounds.")
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional output label. By default it is derived from the scenario, "
            "for example e01_r9_ppo. The timestamp is appended to the same directory name."
        ),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help=(
            "Exact result directory for a batch launcher. It must not already "
            "exist and must be supplied together with --model-dir."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=(
            "Exact model directory for a batch launcher. It must not already "
            "exist and must be supplied together with --result-dir."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--blue-policy",
        default="b0_fixed_ratio_random",
        help="Existing blue policy selected through the original environment variable interface.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PPO device: cuda, cuda:N, auto, or cpu. Default deliberately requires GPU.",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Optional PPO checkpoint to resume.")
    parser.add_argument(
        "--allow-legacy-resume",
        action="store_true",
        help=(
            "Allow an old/incompatible 90-D checkpoint as network weights only; "
            "legacy optimizer state and hyperparameters are discarded."
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--learning-rate-final", type=float, default=2e-5)
    parser.add_argument("--learning-rate-decay-updates", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.995)
    parser.add_argument("--clip-ratio", type=float, default=0.15)
    parser.add_argument("--value-clip-ratio", type=float, default=0.20)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--entropy-final-coef", type=float, default=0.0002)
    parser.add_argument("--entropy-decay-updates", type=int, default=100)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--rollout-size", type=int, default=65536)
    parser.add_argument(
        "--update-mode",
        choices=("episode", "rollout"),
        default="episode",
        help=(
            "episode keeps one behaviour policy frozen for a whole round; "
            "rollout updates at joint-step boundaries after --rollout-size transitions"
        ),
    )
    parser.add_argument("--value-inference-batch-size", type=int, default=16384)
    parser.add_argument("--render-mode", choices=("none", "human"), default="none")
    parser.add_argument(
        "--render-fps",
        type=int,
        default=10,
        help=(
            "Pygame/VNC refresh rate in human mode; this does not skip or alter "
            "simulation steps"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--disable-log-color", action="store_true")
    args = parser.parse_args(argv)
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if args.render_fps <= 0:
        parser.error("--render-fps must be positive")
    if (
        args.hidden_dim <= 0
        or args.minibatch_size <= 0
        or args.rollout_size <= 0
        or args.value_inference_batch_size <= 0
    ):
        parser.error("PPO dimensions and batch sizes must be positive")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 4294967295")
    if (
        args.update_epochs <= 0
        or args.learning_rate_decay_updates <= 0
        or args.entropy_decay_updates <= 0
    ):
        parser.error("--update-epochs must be positive")
    finite_values = (
        args.learning_rate,
        args.learning_rate_final,
        args.gamma,
        args.gae_lambda,
        args.clip_ratio,
        args.value_clip_ratio,
        args.value_coef,
        args.entropy_coef,
        args.entropy_final_coef,
        args.target_kl,
        args.max_grad_norm,
    )
    if not all(math.isfinite(value) for value in finite_values):
        parser.error("PPO floating-point parameters must be finite")
    if (
        args.learning_rate <= 0.0
        or args.learning_rate_final <= 0.0
        or args.max_grad_norm <= 0.0
    ):
        parser.error("--learning-rate and --max-grad-norm must be positive")
    if args.learning_rate_final > args.learning_rate:
        parser.error("--learning-rate-final cannot exceed --learning-rate")
    if not 0.0 <= args.gamma <= 1.0 or not 0.0 <= args.gae_lambda <= 1.0:
        parser.error("--gamma and --gae-lambda must be between 0 and 1")
    if args.clip_ratio <= 0.0 or args.value_clip_ratio < 0.0:
        parser.error("PPO clipping ratios are invalid")
    if (
        args.value_coef < 0.0
        or args.entropy_coef < 0.0
        or args.entropy_final_coef < 0.0
        or args.target_kl < 0.0
    ):
        parser.error("--value-coef and --entropy-coef must be non-negative")
    if args.entropy_final_coef > args.entropy_coef:
        parser.error("--entropy-final-coef cannot exceed --entropy-coef")
    if args.run_id is not None and not _RUN_ID.fullmatch(args.run_id):
        parser.error("--run-id must start with an alphanumeric and contain only A-Z, a-z, 0-9, _, ., -")
    if (args.result_dir is None) != (args.model_dir is None):
        parser.error("--result-dir and --model-dir must be supplied together")
    return args


def _run_directories(
    scenario_id: str,
    run_id: str | None,
    timestamp: str,
) -> tuple[str, Path, Path]:
    """Return paired, single-level result/model directories for a new run."""

    label = run_id or f"{scenario_id.lower()}_r9_ppo"
    run_name = f"{label}_{timestamp}"
    return (
        run_name,
        PERSONAL_ROOT / "results" / run_name,
        PERSONAL_ROOT / "models" / run_name,
    )


def _resolve_run_directories(
    *,
    scenario_id: str,
    run_id: str | None,
    timestamp: str,
    result_dir: Path | None,
    model_dir: Path | None,
    invocation_cwd: Path,
) -> tuple[str, Path, Path]:
    """Resolve the normal layout or exact leaf paths selected by a launcher."""

    if result_dir is None and model_dir is None:
        return _run_directories(scenario_id, run_id, timestamp)
    if result_dir is None or model_dir is None:
        raise ValueError("result_dir and model_dir must be supplied together")

    def resolve(path: Path) -> Path:
        expanded = path.expanduser()
        return (
            expanded.resolve()
            if expanded.is_absolute()
            else (invocation_cwd / expanded).resolve()
        )

    exact_result = resolve(result_dir)
    exact_model = resolve(model_dir)
    if (
        exact_result == exact_model
        or exact_result in exact_model.parents
        or exact_model in exact_result.parents
    ):
        raise ValueError("Result and model directories must be separate, non-nested paths")
    return run_id or scenario_id.lower(), exact_result, exact_model


def resolve_scenario(value: str, invocation_cwd: Path) -> Path:
    supplied = Path(value).expanduser()
    candidates = []
    if supplied.is_absolute():
        candidates.append(supplied)
    else:
        candidates.extend((invocation_cwd / supplied, REPOSITORY_ROOT / supplied))

    label = value.upper()
    if re.fullmatch(r"[EMH]0[1-3]", label):
        difficulty = {"E": "easy", "M": "medium", "H": "hard"}[label[0]]
        candidates.append(REPOSITORY_ROOT / "scenarios" / "cases" / difficulty / label / "scenario.json")
    candidates.append(REPOSITORY_ROOT / "scenarios" / "cases" / value / "scenario.json")

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot resolve competition scenario '{value}'. Tried:\n  {rendered}")


def load_profile(path: Path) -> Profile:
    with path.open("r", encoding="utf-8") as stream:
        return Profile.from_dict(json.load(stream))


def _configure_logging(result_dir: Path, *, color: bool, verbose: bool) -> None:
    LogManager(color_enabled=color)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler = logging.FileHandler(result_dir / "training.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(file_handler)


def _save_policy(policy: PPOSharedPolicy, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    policy.save(str(temporary))
    checkpoint = torch.load(temporary, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Original PPO save() produced a non-dictionary checkpoint")
    checkpoint["personal_train_schema"] = dict(CHECKPOINT_SCHEMA)
    torch.save(checkpoint, temporary)
    verified = torch.load(temporary, map_location="cpu", weights_only=False)
    if (
        not isinstance(verified, dict)
        or verified.get("personal_train_schema") != CHECKPOINT_SCHEMA
        or verified.get("algorithm") != PPOSharedPolicy.ALGORITHM
        or not isinstance(verified.get("network"), dict)
    ):
        raise ValueError(f"PPO checkpoint verification failed: {path}")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_checkpoint_record(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("personal_train_schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("algorithm") != PPOSharedPolicy.ALGORITHM
        or not isinstance(checkpoint.get("network"), dict)
    ):
        raise ValueError(f"Invalid final PPO checkpoint: {path}")
    config = checkpoint.get("config")
    if (
        not isinstance(config, dict)
        or int(config.get("observation_dim", -1)) != OBSERVATION_DIM
        or int(config.get("action_dim", -1)) != ACTION_DIM
    ):
        raise ValueError(f"Final PPO checkpoint violates the locked interface: {path}")
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "safe_training_boundary": bool(checkpoint.get("safe_training_boundary", False)),
    }


def _write_checkpoint_validation(model_dir: Path) -> None:
    validation = {
        "format_version": 1,
        "algorithm": PPOSharedPolicy.ALGORITHM,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoints": {
            name: _validated_checkpoint_record(model_dir / name)
            for name in ("best.pt", "latest.pt")
        },
    }
    path = model_dir / "checkpoint_validation.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_model_metadata(
    model_dir: Path,
    *,
    policy: PPOSharedPolicy,
    latest_round: int,
    best_round: int,
    best_score: float,
    interrupted: bool = False,
) -> None:
    metadata = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "algorithm": PPOSharedPolicy.ALGORITHM,
        "latest_round": int(latest_round),
        "best_round": int(best_round),
        "best_official_score": float(best_score),
        "update_count": int(policy.update_count),
        "transition_count": int(policy.transition_count),
        "episode_count": int(policy.episode_count),
        "device": str(policy.device),
        "interrupted": bool(interrupted),
    }
    path = model_dir / "model_metadata.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _register_agents(
    environment: PersonalR9TrainingEnv,
    profile: Profile,
    policy: PPOSharedPolicy,
    commander: PersonalR9Commander,
    init_observation: dict,
) -> list[PersonalR9PPOAttackAgent]:
    simulators = environment.engine.simulator_factory.get_all_simulators()
    red_count = sum(
        int(sim.entity_ext.entity.entityType) in RED_MISSILE_TYPES
        for sim in simulators
    )
    environment.learning_team_size = red_count
    agents: list[PersonalR9PPOAttackAgent] = []
    next_agent_id = 1
    for simulator in simulators:
        entity = simulator.entity_ext.entity
        if int(entity.entityType) not in RED_MISSILE_TYPES:
            continue
        agent = PersonalR9PPOAttackAgent(
            next_agent_id,
            int(entity.id),
            init_observation,
            commander,
            policy,
            learning_max_steps=environment.max_steps,
            team_size=red_count,
            initial_sim_time_ms=float(profile.imagineProfile.simTime),
            sim_step_ms=float(profile.imagineProfile.simStep),
        )
        environment.agent_manager.register_agent(agent)
        commander.register_platform(int(entity.id))
        agents.append(agent)
        next_agent_id += 1

    deploy_agent = DeployAgent(
        -1,
        -1,
        {},
        profile.imagineProfile.redArea.coordinates,
        profile.imagineProfile.redArea.coordinatesHM,
    )
    environment.agent_manager.register_agent(deploy_agent)
    return agents


def _policy_config(
    args: argparse.Namespace, resume_path: Path | None = None
) -> PPOConfig:
    fresh = PPOConfig(
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        learning_rate_final=args.learning_rate_final,
        learning_rate_decay_updates=args.learning_rate_decay_updates,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_clip_ratio=args.value_clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        entropy_final_coef=args.entropy_final_coef,
        entropy_decay_updates=args.entropy_decay_updates,
        target_kl=args.target_kl,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        rollout_size=args.rollout_size,
        update_mode=args.update_mode,
        value_inference_batch_size=args.value_inference_batch_size,
        seed=args.seed,
        device=args.device,
    )
    if resume_path is None:
        return _validate_policy_config(fresh)

    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(saved, dict):
        raise ValueError(f"Checkpoint has no usable PPO config: {resume_path}")
    schema = checkpoint.get("personal_train_schema")
    current_schema = bool(
        schema == CHECKPOINT_SCHEMA
        and isinstance(checkpoint, dict)
        and checkpoint.get("algorithm") == PPOSharedPolicy.ALGORITHM
    )
    if not current_schema:
        if not args.allow_legacy_resume:
            raise ValueError(
                "Checkpoint is not a current personal PPO checkpoint. Start fresh, "
                "use a checkpoint created by this trainer, or explicitly pass "
                "--allow-legacy-resume for a weights-only initialization."
            )
        LOGGER.warning(
            "Loading legacy checkpoint weights only; optimizer/counters and old "
            "one-step-TD hyperparameters will not be restored: %s",
            resume_path,
        )
    merged = asdict(fresh)
    valid_fields = set(PPOConfig.__dataclass_fields__)
    if current_schema:
        merged.update({key: value for key, value in saved.items() if key in valid_fields})
    else:
        # ActorCritic names/shapes are deliberately compatible.  Only structural
        # fields are inherited; all unstable legacy optimization settings stay
        # replaced by the new command-line defaults.
        merged.update(
            {
                key: saved[key]
                for key in ("observation_dim", "action_dim", "hidden_dim")
                if key in saved
            }
        )
    merged["device"] = args.device
    merged["seed"] = args.seed
    if int(merged["observation_dim"]) != OBSERVATION_DIM or int(merged["action_dim"]) != ACTION_DIM:
        raise ValueError(
            "Resume checkpoint is incompatible with the locked R9 interface: "
            f"obs={merged['observation_dim']}, action={merged['action_dim']}"
        )
    return _validate_policy_config(PPOConfig(**merged))


def _validate_policy_config(config: PPOConfig) -> PPOConfig:
    if config.observation_dim != OBSERVATION_DIM or config.action_dim != ACTION_DIM:
        raise ValueError("PPO config violates the locked 90-D observation / 3-action interface")
    if (
        config.hidden_dim <= 0
        or config.minibatch_size <= 0
        or config.rollout_size <= 0
        or config.value_inference_batch_size <= 0
    ):
        raise ValueError("PPO hidden/batch/rollout dimensions must be positive")
    if (
        config.update_epochs <= 0
        or config.learning_rate_decay_updates <= 0
        or config.entropy_decay_updates <= 0
    ):
        raise ValueError("PPO update_epochs must be positive")
    finite_values = (
        config.learning_rate,
        config.learning_rate_final,
        config.gamma,
        config.gae_lambda,
        config.clip_ratio,
        config.value_clip_ratio,
        config.value_coef,
        config.entropy_coef,
        config.entropy_final_coef,
        config.target_kl,
        config.max_grad_norm,
    )
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("PPO floating-point parameters must be finite")
    if (
        config.learning_rate <= 0.0
        or config.learning_rate_final <= 0.0
        or config.max_grad_norm <= 0.0
    ):
        raise ValueError("PPO learning_rate and max_grad_norm must be positive")
    if config.learning_rate_final > config.learning_rate:
        raise ValueError("PPO final learning rate cannot exceed initial learning rate")
    if (
        not 0.0 <= config.gamma <= 1.0
        or not 0.0 <= config.gae_lambda <= 1.0
        or config.clip_ratio <= 0.0
        or config.value_clip_ratio < 0.0
    ):
        raise ValueError("PPO gamma/GAE/clipping settings are invalid")
    if (
        config.value_coef < 0.0
        or config.entropy_coef < 0.0
        or config.entropy_final_coef < 0.0
        or config.target_kl < 0.0
    ):
        raise ValueError("PPO value_coef and entropy_coef must be non-negative")
    if config.entropy_final_coef > config.entropy_coef:
        raise ValueError("PPO final entropy coefficient cannot exceed initial coefficient")
    if config.update_mode not in {"episode", "rollout"}:
        raise ValueError("PPO update_mode must be episode or rollout")
    return config


def _resume_loads_optimizer(resume_path: Path) -> bool:
    return _resume_mode(resume_path) == "full_state_resume"


def _resume_mode(resume_path: Path) -> str:
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    is_current = bool(
        isinstance(checkpoint, dict)
        and checkpoint.get("personal_train_schema") == CHECKPOINT_SCHEMA
        and checkpoint.get("algorithm") == PPOSharedPolicy.ALGORITHM
    )
    # ``best.pt`` is captured as the behaviour network that produced the best
    # official score.  It is an initialization checkpoint, never an exact
    # optimizer/RNG continuation point—even if a rollout-mode save happens to
    # find an empty buffer and marks the boundary safe.
    if is_current and resume_path.name == "best.pt":
        return "current_weights_only"
    if (
        is_current
        and bool(checkpoint.get("safe_training_boundary", False))
    ):
        return "full_state_resume"
    if is_current:
        return "current_weights_only"
    return "legacy_weights_only"


def main(argv: list[str] | None = None) -> int:
    invocation_cwd = Path.cwd()
    args = parse_args(argv)
    scenario_path = resolve_scenario(args.scenario, invocation_cwd)
    resume_path = None
    resume_mode = "fresh"
    if args.resume is not None:
        supplied_resume = args.resume.expanduser()
        resume_path = (
            supplied_resume.resolve()
            if supplied_resume.is_absolute()
            else (invocation_cwd / supplied_resume).resolve()
        )
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
    reward_policy = load_reward_policy(scenario_path)
    if reward_policy is None:
        raise ValueError(
            f"{scenario_path} has no case_info.json; this trainer accepts only scored competition cases."
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
    reporter: TrainingReporter | None = None
    policy: PPOSharedPolicy | None = None
    environment: PersonalR9TrainingEnv | None = None
    writer_initialized = False
    agents: list[PersonalR9PPOAttackAgent] = []
    best_score = float("-inf")
    best_round = 0
    completed_rounds = 0
    interrupted = False
    termination_signal: int | None = None
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def _request_graceful_stop(signum, _frame) -> None:
        nonlocal termination_signal
        termination_signal = int(signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _request_graceful_stop)
    try:
        _configure_logging(
            result_dir,
            color=not args.disable_log_color,
            verbose=args.verbose,
        )
        reporter = TrainingReporter(result_dir)

        # Existing simulator models use relative package resources.  This is
        # /app in the image and <repository>/core in a source checkout.
        os.chdir(CORE_ROOT)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        os.environ["SIMULATION_SEED"] = str(args.seed)
        os.environ["RED_POLICY_SEED"] = str(args.seed)
        os.environ["BLUE_POLICY_SEED"] = str(args.seed)
        os.environ["BLUE_POLICY"] = args.blue_policy

        writer_config = WriteConfig(
            output_dir=str(result_dir / "simulator_writer"),
            verbose=False,
            batch_size=100,
            enable_config=False,
            enable_state=False,
            enable_event=False,
            enable_ai_action=False,
        )
        init_writer(writer_config)
        writer_initialized = True

        policy_config = _policy_config(args, resume_path)
        policy = PPOSharedPolicy(policy_config)
        if resume_path is not None:
            resume_mode = _resume_mode(resume_path)
            load_optimizer = resume_mode == "full_state_resume"
            policy.load(str(resume_path), load_optimizer=load_optimizer)
            LOGGER.info(
                "PPO checkpoint mode=%s path=%s",
                resume_mode,
                resume_path,
            )
        policy.set_training(True)

        reward_config = R9RewardConfig(progress_discount=policy.config.gamma)
        profile = load_profile(scenario_path)
        environment = PersonalR9TrainingEnv(
            profile,
            reward_policy=reward_policy,
            reward_config=reward_config,
            render_mode=None if args.render_mode == "none" else "human",
            render_fps=args.render_fps,
        )
        environment.set_shared_learning_policy(policy)
        initial_targets = environment._get_init_ship_observation()
        commander = PersonalR9Commander(
            initial_targets_from_observation(initial_targets),
            seed=args.seed,
        )
        agents = _register_agents(environment, profile, policy, commander, initial_targets)
        if not agents:
            raise RuntimeError("Scenario contains no red missile agents")

        run_config: dict[str, Any] = {
            "created_at": datetime.now().astimezone().isoformat(),
            "run_name": run_name,
            "run_label": args.run_id or f"{reward_policy.scenario_id.lower()}_r9_ppo",
            "scenario": reward_policy.scenario_id,
            "scenario_path": str(scenario_path),
            "rounds": args.rounds,
            "seed": args.seed,
            "red_policy": R9_POLICY,
            "commander": "PersonalR9Commander (R9 + legal dynamic 9500 discovery)",
            "red_motion_policy": "ppo",
            "blue_policy": args.blue_policy,
            "competition_max_steps": environment.max_steps,
            "red_agents": len(agents),
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "ppo": asdict(policy.config),
            "reward": reward_config.to_dict(),
            "result_dir": str(result_dir),
            "model_dir": str(model_dir),
            "externally_managed_output": args.result_dir is not None,
            "resume": str(resume_path) if resume_path else None,
            "resume_mode": resume_mode,
            "policy_rng_restored": resume_mode == "full_state_resume",
            "allow_legacy_resume": bool(args.allow_legacy_resume),
            "render_mode": args.render_mode,
            "render_fps": args.render_fps,
        }
        reporter.write_run_config(run_config)
        LOGGER.info(
            "Scenario=%s, rounds=%d, red_agents=%d",
            reward_policy.scenario_id,
            args.rounds,
            len(agents),
        )
        LOGGER.info(
            "PPO network: 90 -> %d -> %d -> actor(3) / critic(1)",
            policy.config.hidden_dim,
            policy.config.hidden_dim,
        )
        LOGGER.info("PPO device: %s", policy.device)
        LOGGER.info("Results: %s", result_dir)
        LOGGER.info("Models: %s", model_dir)

        for round_index in range(1, args.rounds + 1):
            LOGGER.info("Round %d/%d reset", round_index, args.rounds)
            initial_observation = environment.reset()
            summary_tracker = RewardTracker(reward_policy)
            run_summary = RunSummary(
                scenario=reward_policy.scenario_id,
                policies={
                    "red": R9_POLICY,
                    "red_motion": "ppo",
                    "blue": args.blue_policy,
                },
                reward_tracker=summary_tracker,
            )
            run_summary.start(initial_observation)
            environment.red_model_deploy()
            environment.prepare_round_after_deploy()

            start = time.perf_counter()
            final_observation = initial_observation
            termination_reason = "time_limit"
            for _ in range(environment.max_steps):
                observation, _, done, _ = environment.step()
                final_observation = observation
                run_summary.update(environment.current_step, observation)
                if environment.current_step % args.progress_every == 0:
                    LOGGER.info(
                        "Round %d progress: %d/%d",
                        round_index,
                        environment.current_step,
                        environment.max_steps,
                    )
                if done:
                    termination_reason = (
                        "time_limit"
                        if environment.current_step >= environment.max_steps
                        else "environment_done"
                    )
                    break
            elapsed = time.perf_counter() - start

            write_immediately()
            red_launched = sum(agent.launch_step >= 0 for agent in agents)
            summary = run_summary.build(
                final_observation,
                termination_reason=termination_reason,
                red_launched=red_launched,
            )
            summary["interception"] = environment.interception_metrics()
            score_value = float((summary.get("score") or {}).get("score", 0.0))
            if not math.isfinite(score_value) or not 0.0 <= score_value <= 100.0:
                raise ValueError(f"Official score is invalid: {score_value!r}")
            is_new_best = score_value > best_score
            if is_new_best:
                # In the default episode mode no update happened during this
                # round, so this is the exact behaviour network that generated
                # the score.  Its unflushed rollout is intentionally not part of
                # the inference checkpoint.
                _save_policy(policy, model_dir / "best.pt")

            # GAE is computed from the complete, per-Agent episode before the
            # reporting row captures this round's optimizer diagnostics.
            policy.finish_episode()
            action_counts: Counter[int] = Counter()
            action_switches = 0
            for agent in agents:
                action_counts.update(agent.action_counts)
                action_switches += agent.action_switch_count
            row = reporter.record_round(
                round_index=round_index,
                summary=summary,
                reward_metrics=environment.reward_metrics(),
                policy=policy,
                elapsed_seconds=elapsed,
                action_counts=action_counts,
                action_switches=action_switches,
            )
            # The round is durably reported at this point.  Checkpoint I/O
            # failure must not make failure.json claim the scored round never
            # completed.
            completed_rounds = round_index

            # ``latest`` and numbered checkpoints are saved after the update and
            # are therefore safe for exact optimizer/counter resume.
            _save_policy(policy, model_dir / "latest.pt")
            if is_new_best:
                best_score = float(row["score"])
                best_round = round_index
            if round_index % args.checkpoint_every == 0 or round_index == args.rounds:
                _save_policy(
                    policy,
                    model_dir / "checkpoints" / f"round_{round_index:04d}.pt",
                )
            _write_model_metadata(
                model_dir,
                policy=policy,
                latest_round=completed_rounds,
                best_round=best_round,
                best_score=best_score,
            )
            LOGGER.info(
                "ROUND_RESULT round=%d score=%.6f K=%.6f T=%.6f steps=%d elapsed=%.3fs",
                round_index,
                row["score"],
                row["K"],
                row["T"],
                row["steps"],
                elapsed,
            )
        _write_checkpoint_validation(model_dir)
    except KeyboardInterrupt:
        interrupted = True
        LOGGER.warning("Training interrupted; preserving the most recent network state")
        (result_dir / "interruption.json").write_text(
            json.dumps(
                {
                    "time": datetime.now().astimezone().isoformat(),
                    "completed_rounds": completed_rounds,
                    "signal": termination_signal,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if policy is not None:
            try:
                _save_policy(policy, model_dir / "interrupted.pt")
                _write_model_metadata(
                    model_dir,
                    policy=policy,
                    latest_round=completed_rounds,
                    best_round=best_round,
                    best_score=best_score if math.isfinite(best_score) else 0.0,
                    interrupted=True,
                )
            except Exception:
                LOGGER.exception("Could not save the interrupted PPO checkpoint")
    except Exception as error:
        LOGGER.exception("Training failed: %s", error)
        failure = {
            "time": datetime.now().astimezone().isoformat(),
            "completed_rounds": completed_rounds,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        (result_dir / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if policy is not None:
            try:
                _save_policy(policy, model_dir / "failed.pt")
                _write_model_metadata(
                    model_dir,
                    policy=policy,
                    latest_round=completed_rounds,
                    best_round=best_round,
                    best_score=best_score if math.isfinite(best_score) else 0.0,
                    interrupted=False,
                )
            except Exception:
                LOGGER.exception("Could not save the failed PPO checkpoint")
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
        signal.signal(signal.SIGTERM, previous_sigterm_handler)

    LOGGER.info(
        "Training finished: completed_rounds=%d best_round=%d best_score=%.6f interrupted=%s",
        completed_rounds,
        best_round,
        best_score if best_score != float("-inf") else 0.0,
        interrupted,
    )
    print(f"RESULT_DIR {result_dir}")
    print(f"MODEL_DIR {model_dir}")
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
