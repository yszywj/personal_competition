#!/usr/bin/env python3
"""Launch independent R9 PPO jobs for several competition scenarios.

This is deliberately a host-side Docker launcher.  Every scenario receives a
separate container and a separate ``/app/Results`` tmpfs because the native
simulator libraries write entity-ID-based files below that fixed directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import pwd
import re
import secrets
import shlex
import signal
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


if __package__:
    from .bootstrap import (
        PERSONAL_ROOT, REPOSITORY_ROOT, SCENARIOS_ROOT, validate_personal_output_path,
    )
else:
    from bootstrap import (
        PERSONAL_ROOT, REPOSITORY_ROOT, SCENARIOS_ROOT, validate_personal_output_path,
    )
RESULTS_ROOT = PERSONAL_ROOT / "results"
MODELS_ROOT = PERSONAL_ROOT / "models"
LAUNCHER_RUNS_ROOT = PERSONAL_ROOT / "launcher_runs"
TRAINER_IN_CONTAINER = "/app/personal_train/train_r9_ppo.py"
PERSONAL_IN_CONTAINER = Path("/app/personal_train")
ALL_SCENARIOS = ("E01", "E02", "E03", "M01", "M02", "M03", "H01", "H02", "H03")
CURRENT_SCHEMA_NAME = "personal-r9-ppo"
CURRENT_SCHEMA_VERSION = 3
CURRENT_ALGORITHM = "personal_ppo_gae_v2"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_GPU_ID = re.compile(r"^(?:[0-9]+|GPU-[A-Fa-f0-9-]+)$")
_RUN_TIMESTAMP = re.compile(r"^[0-9]{8}_[0-9]{6}_[0-9]{6}$")


@dataclass(frozen=True)
class ScenarioCheckpoint:
    path: Path
    recorded_score: float | None
    selection: str
    scenario: str = "E01"
    source_batch: str | None = None
    latest_round: int | None = None
    best_round: int | None = None
    interrupted: bool = False
    sha256: str | None = None


# Kept as an import-compatible name for callers that explicitly select E01.
E01Checkpoint = ScenarioCheckpoint


@dataclass(frozen=True)
class JobPlan:
    scenario: str
    result_dir: Path
    model_dir: Path
    resume: Path | None = None
    initial_score: float | None = None

    @property
    def run_name(self) -> str:
        return self.result_dir.name

    @property
    def initialization(self) -> str:
        return "checkpoint_weights" if self.resume is not None else "fresh_random"


@dataclass
class RunningJob:
    plan: JobPlan
    device: str
    slot: int
    container_name: str
    cidfile: Path
    process: subprocess.Popen
    log_stream: Any
    started_monotonic: float


def _positive(parser: argparse.ArgumentParser, option: str, value: int) -> None:
    if value <= 0:
        parser.error(f"{option} must be positive")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent R9 PPO training jobs in isolated Docker containers. "
            "A resume batch initializes every selected scenario from its own best checkpoint."
        )
    )
    parser.add_argument("--rounds", type=int, default=100, help="Rounds per scenario.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(ALL_SCENARIOS),
        metavar="CASE",
        help="Subset of E01 E02 E03 M01 M02 M03 H01 H02 H03.",
    )
    parser.add_argument(
        "--resume-batch",
        type=Path,
        default=None,
        help=(
            "Existing models batch directory or its direct child name below personal_train/models. "
            "Every selected scenario is initialized from <batch>/<scenario>/best.pt."
        ),
    )
    parser.add_argument(
        "--e01-resume",
        type=Path,
        default=None,
        help=(
            "Legacy E01-only checkpoint option. It cannot be combined with --resume-batch."
        ),
    )
    parser.add_argument("--seed", type=int, default=1, help="One comparable seed per scenario.")
    parser.add_argument("--blue-policy", default="b0_fixed_ratio_random")
    parser.add_argument("--image", default="personal-competition:runtime", help="Docker runtime image.")
    parser.add_argument(
        "--gpu-ids",
        default="0,1",
        help="Host GPU IDs, for example 0,1. Use cpu for CPU-only jobs.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Concurrent containers; default is one per listed GPU.",
    )
    parser.add_argument(
        "--allow-gpu-sharing",
        action="store_true",
        help="Allow more concurrent workers than GPU IDs (normally slower and less stable).",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="OMP/MKL/OpenBLAS threads exposed to each independent worker.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Launcher control-run ID; default: r9_multi_<timestamp>. Business outputs stay flat.",
    )
    parser.add_argument("--docker", default="docker", help="Docker CLI executable.")
    parser.add_argument("--stop-timeout", type=int, default=60)
    parser.add_argument("--verbose-workers", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the complete plan without creating directories or containers.",
    )
    args = parser.parse_args(argv)

    for name in ("rounds", "threads_per_worker", "checkpoint_every", "progress_every", "stop_timeout"):
        _positive(parser, f"--{name.replace('_', '-')}", int(getattr(args, name)))
    if args.max_parallel is not None:
        _positive(parser, "--max-parallel", args.max_parallel)
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 4294967295")

    scenarios = [str(value).upper() for value in args.scenarios]
    invalid = [value for value in scenarios if value not in ALL_SCENARIOS]
    if invalid:
        parser.error(f"Unknown scenarios: {', '.join(invalid)}")
    if len(set(scenarios)) != len(scenarios):
        parser.error("--scenarios cannot contain duplicates")
    args.scenarios = scenarios

    if args.batch_id is not None and not _SAFE_ID.fullmatch(args.batch_id):
        parser.error("--batch-id must use only A-Z, a-z, 0-9, _, ., - and be at most 64 characters")
    if args.resume_batch is not None and args.e01_resume is not None:
        parser.error("--resume-batch and --e01-resume cannot be combined")

    try:
        args.devices = parse_devices(args.gpu_ids)
    except ValueError as error:
        parser.error(str(error))
    if args.max_parallel is None:
        args.max_parallel = len(args.devices)
    if args.max_parallel > len(args.devices) and not args.allow_gpu_sharing:
        parser.error(
            "--max-parallel exceeds the number of device slots; "
            "pass --allow-gpu-sharing explicitly"
        )
    # Distinguish this launcher invocation from another checkout/process that
    # happens to reuse the same user-supplied batch ID and Docker names.
    args.ownership_token = secrets.token_hex(16)
    return args


def parse_devices(value: str) -> tuple[str, ...]:
    tokens = tuple(item.strip() for item in value.split(",") if item.strip())
    if not tokens:
        raise ValueError("--gpu-ids must contain at least one GPU ID or cpu")
    lowered = tuple(token.lower() for token in tokens)
    if "cpu" in lowered:
        if len(tokens) != 1:
            raise ValueError("cpu cannot be combined with GPU IDs")
        return ("cpu",)
    if any(not _GPU_ID.fullmatch(token) for token in tokens):
        raise ValueError("GPU IDs must be numeric indices or NVIDIA GPU-... UUIDs")
    kinds = {"index" if token.isdigit() else "uuid" for token in tokens}
    if len(kinds) > 1:
        raise ValueError("Do not mix numeric GPU indices and GPU UUIDs in one batch")
    normalized = tuple(str(int(token)) if token.isdigit() else token for token in tokens)
    if len({token.lower() for token in normalized}) != len(normalized):
        raise ValueError(
            "GPU IDs cannot repeat; list each once, then set a larger --max-parallel "
            "with --allow-gpu-sharing"
        )
    return normalized


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _current_checkpoint_metadata(metadata: Any, *, allow_interrupted: bool) -> bool:
    if not isinstance(metadata, dict):
        return False
    schema = metadata.get("checkpoint_schema")
    try:
        version = int(schema.get("version", -1)) if isinstance(schema, dict) else -1
    except (TypeError, ValueError):
        return False
    return bool(
        isinstance(schema, dict)
        and schema.get("name") == CURRENT_SCHEMA_NAME
        and version == CURRENT_SCHEMA_VERSION
        and metadata.get("algorithm") == CURRENT_ALGORITHM
        and (allow_interrupted or not bool(metadata.get("interrupted", False)))
    )


def _current_e01_metadata(metadata: Any) -> bool:
    """Backward-compatible strict predicate used for completed job outputs."""

    return _current_checkpoint_metadata(metadata, allow_interrupted=False)


def _result_dirs_for_model_metadata(metadata_path: Path) -> tuple[Path, ...]:
    try:
        relative_dir = metadata_path.parent.relative_to(MODELS_ROOT.resolve())
    except ValueError:
        return ()
    # Historical results may be archived without moving their paired models.
    return (
        RESULTS_ROOT / relative_dir,
        RESULTS_ROOT / "results_past" / relative_dir,
    )


def _model_scenario(metadata_path: Path) -> str | None:
    for result_dir in _result_dirs_for_model_metadata(metadata_path):
        try:
            value = json.loads((result_dir / "run_config.json").read_text(encoding="utf-8")).get(
                "scenario"
            )
            if value:
                return str(value).upper()
        except (OSError, ValueError, AttributeError, json.JSONDecodeError):
            continue
    name = metadata_path.parent.name.upper()
    if name in ALL_SCENARIOS:
        return name
    match = re.match(r"^([emh]0[1-3])_r9_ppo_", metadata_path.parent.name, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def _model_run_is_complete(metadata_path: Path, metadata: dict[str, Any]) -> bool:
    for result_dir in _result_dirs_for_model_metadata(metadata_path):
        try:
            run_config = json.loads((result_dir / "run_config.json").read_text(encoding="utf-8"))
            requested_rounds = int(run_config["rounds"])
            latest_round = int(metadata["latest_round"])
            best_round = int(metadata["best_round"])
            best_score = float(metadata["best_official_score"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if (
            requested_rounds > 0
            and latest_round == requested_rounds
            and 1 <= best_round <= latest_round
            and math.isfinite(best_score)
            and 0.0 <= best_score <= 100.0
            and not (result_dir / "failure.json").exists()
            and not (result_dir / "interruption.json").exists()
        ):
            return True
    return False


def find_best_e01_checkpoint() -> E01Checkpoint:
    candidates: list[tuple[float, int, Path]] = []
    for metadata_path in MODELS_ROOT.rglob("model_metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            score = float(metadata["best_official_score"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        checkpoint = metadata_path.parent / "best.pt"
        if (
            _current_e01_metadata(metadata)
            and _model_scenario(metadata_path) == "E01"
            and _model_run_is_complete(metadata_path, metadata)
            and math.isfinite(score)
            and checkpoint.is_file()
        ):
            candidates.append((score, checkpoint.stat().st_mtime_ns, checkpoint.resolve()))
    if not candidates:
        raise FileNotFoundError(
            "No compatible E01 best.pt was found. Pass --e01-resume explicitly."
        )
    score, _, path = max(candidates)
    return E01Checkpoint(
        path=path,
        recorded_score=score,
        selection="highest_compatible_recorded_score",
        scenario="E01",
        source_batch=path.parent.parent.name,
        sha256=_sha256_file(path),
    )


def _score_next_to_checkpoint(path: Path) -> float | None:
    metadata_path = path.parent / "model_metadata.json"
    try:
        score = float(json.loads(metadata_path.read_text(encoding="utf-8"))["best_official_score"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    return score if math.isfinite(score) else None


def resolve_e01_checkpoint(value: Path | None, invocation_cwd: Path) -> E01Checkpoint:
    if value is None:
        return find_best_e01_checkpoint()
    expanded = value.expanduser()
    path = expanded.resolve() if expanded.is_absolute() else (invocation_cwd / expanded).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"E01 checkpoint does not exist: {path}")
    _container_personal_path(path)
    if path.name != "best.pt":
        raise ValueError("--e01-resume must point to a best.pt checkpoint")
    metadata_path = path.parent / "model_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            f"E01 checkpoint metadata is missing or invalid: {metadata_path}"
        ) from error
    if not _current_e01_metadata(metadata):
        raise ValueError(
            "--e01-resume must use a non-interrupted current-schema PPO checkpoint"
        )
    if _model_scenario(metadata_path) != "E01":
        raise ValueError("--e01-resume metadata does not identify an E01 training run")
    if not _model_run_is_complete(metadata_path, metadata):
        raise ValueError("--e01-resume must come from a completed, non-failed E01 run")
    return E01Checkpoint(
        path=path,
        recorded_score=_score_next_to_checkpoint(path),
        selection="explicit",
        scenario="E01",
        source_batch=path.parent.parent.name,
        latest_round=int(metadata["latest_round"]),
        best_round=int(metadata["best_round"]),
        interrupted=False,
        sha256=_sha256_file(path),
    )


def _resolve_resume_batch_root(value: Path, invocation_cwd: Path) -> Path:
    supplied = value.expanduser()
    if supplied.is_absolute():
        root = supplied.resolve()
    elif len(supplied.parts) == 1 and supplied.name not in {".", ".."}:
        root = (MODELS_ROOT / supplied).resolve()
    else:
        root = (invocation_cwd / supplied).resolve()
    models_root = MODELS_ROOT.resolve()
    if root.parent != models_root:
        raise ValueError(
            f"--resume-batch must identify one direct batch directory below {models_root}: {root}"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"Resume models batch does not exist: {root}")
    return root


def _source_run_config(source_root: Path, scenario: str) -> tuple[Path, dict[str, Any]]:
    relative_dir = source_root.relative_to(MODELS_ROOT.resolve()) / scenario
    candidates = (
        RESULTS_ROOT / relative_dir / "run_config.json",
        RESULTS_ROOT / "results_past" / relative_dir / "run_config.json",
    )
    for path in candidates:
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Source run_config.json is invalid: {path}") from error
        if not isinstance(config, dict):
            raise ValueError(f"Source run_config.json must contain an object: {path}")
        actual = str(config.get("scenario", "")).upper()
        if actual != scenario:
            raise ValueError(
                f"Resume checkpoint scenario mismatch for {scenario}: run_config identifies {actual or 'none'}"
            )
        return path, config
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Cannot verify the source scenario for {scenario}; run_config.json not found in: {rendered}"
    )


def _validate_recorded_checkpoint(checkpoint: Path, validation_path: Path) -> None:
    if not validation_path.is_file():
        return
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        record = validation["checkpoints"][checkpoint.name]
        expected_size = int(record["size_bytes"])
        expected_digest = str(record["sha256"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"Checkpoint validation manifest is invalid: {validation_path}") from error
    if (
        expected_size != checkpoint.stat().st_size
        or not re.fullmatch(r"[0-9a-f]{64}", expected_digest)
        or expected_digest != _sha256_file(checkpoint)
    ):
        raise ValueError(f"Checkpoint does not match its validation manifest: {checkpoint}")


def resolve_resume_batch(
    value: Path,
    scenarios: Sequence[str],
    invocation_cwd: Path,
) -> dict[str, ScenarioCheckpoint]:
    """Resolve one coherent historical models batch into per-scenario best weights."""

    source_root = _resolve_resume_batch_root(value, invocation_cwd)
    selected: dict[str, ScenarioCheckpoint] = {}
    for scenario in scenarios:
        model_dir = source_root / scenario
        metadata_path = model_dir / "model_metadata.json"
        checkpoint = model_dir / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist for {scenario}: {checkpoint}")
        resolved_checkpoint = checkpoint.resolve()
        if resolved_checkpoint.parent != model_dir.resolve():
            raise ValueError(f"Resume checkpoint must not redirect outside its scenario directory: {checkpoint}")
        _container_personal_path(resolved_checkpoint)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Checkpoint metadata is missing or invalid: {metadata_path}") from error
        if not _current_checkpoint_metadata(metadata, allow_interrupted=True):
            raise ValueError(f"Checkpoint metadata has an incompatible schema: {metadata_path}")
        try:
            latest_round = int(metadata["latest_round"])
            best_round = int(metadata["best_round"])
            recorded_score = float(metadata["best_official_score"])
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"Checkpoint metadata has invalid best/latest fields: {metadata_path}") from error
        if (
            latest_round <= 0
            or not 1 <= best_round <= latest_round
            or not math.isfinite(recorded_score)
            or not 0.0 <= recorded_score <= 100.0
        ):
            raise ValueError(f"Checkpoint metadata has invalid best/latest fields: {metadata_path}")
        _source_run_config(source_root, scenario)
        if _model_scenario(metadata_path) != scenario:
            raise ValueError(f"Resume checkpoint metadata does not identify scenario {scenario}: {metadata_path}")
        _validate_recorded_checkpoint(checkpoint, model_dir / "checkpoint_validation.json")
        selected[scenario] = ScenarioCheckpoint(
            path=resolved_checkpoint,
            recorded_score=recorded_score,
            selection="explicit_resume_batch",
            scenario=scenario,
            source_batch=source_root.name,
            latest_round=latest_round,
            best_round=best_round,
            interrupted=bool(metadata.get("interrupted", False)),
            sha256=_sha256_file(checkpoint),
        )
    return selected


def _container_personal_path(host_path: Path) -> str:
    resolved = host_path.resolve()
    try:
        relative = resolved.relative_to(PERSONAL_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"Path must be below {PERSONAL_ROOT} because only personal_train is mounted: {resolved}"
        ) from error
    return str(PERSONAL_IN_CONTAINER / relative)


def make_run_timestamp(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")


def make_batch_id(
    value: str | None,
    now: datetime | None = None,
    *,
    timestamp: str | None = None,
) -> str:
    if value is not None:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("Invalid batch ID")
        return value
    timestamp = timestamp or make_run_timestamp(now)
    if not _RUN_TIMESTAMP.fullmatch(timestamp):
        raise ValueError(f"Invalid run timestamp: {timestamp}")
    return f"r9_multi_{timestamp}"


def _checkpoint_mapping(
    checkpoints: Mapping[str, ScenarioCheckpoint] | ScenarioCheckpoint | None,
) -> dict[str, ScenarioCheckpoint]:
    if checkpoints is None:
        return {}
    if isinstance(checkpoints, ScenarioCheckpoint):
        return {checkpoints.scenario: checkpoints}
    return {str(scenario).upper(): checkpoint for scenario, checkpoint in checkpoints.items()}


def build_job_plans(
    scenarios: Sequence[str],
    timestamp: str,
    checkpoints: Mapping[str, ScenarioCheckpoint] | ScenarioCheckpoint | None,
) -> list[JobPlan]:
    if not _RUN_TIMESTAMP.fullmatch(timestamp):
        raise ValueError("Output timestamp must use YYYYMMDD_HHMMSS_microseconds")
    checkpoint_by_scenario = _checkpoint_mapping(checkpoints)
    plans = []
    for scenario in scenarios:
        checkpoint = checkpoint_by_scenario.get(scenario)
        if scenario == "E01" and checkpoint is None:
            raise ValueError("E01 requires an existing best checkpoint")
        if checkpoint is not None and checkpoint.scenario != scenario:
            raise ValueError(
                f"Checkpoint scenario mismatch: plan={scenario} checkpoint={checkpoint.scenario}"
            )
        run_name = f"{scenario.lower()}_r9_ppo_{timestamp}"
        plans.append(
            JobPlan(
                scenario=scenario,
                result_dir=RESULTS_ROOT / run_name,
                model_dir=MODELS_ROOT / run_name,
                resume=checkpoint.path if checkpoint else None,
                initial_score=checkpoint.recorded_score if checkpoint else None,
            )
        )
    return plans


def _launcher_run_root(batch_id: str) -> Path:
    if not _SAFE_ID.fullmatch(batch_id):
        raise ValueError(f"Invalid batch ID: {batch_id}")
    return validate_personal_output_path(LAUNCHER_RUNS_ROOT / batch_id)


def validate_batch_outputs(batch_id: str, plans: Sequence[JobPlan]) -> Path:
    launcher_root = _launcher_run_root(batch_id)
    if launcher_root.exists():
        raise FileExistsError(f"Launcher run already exists: {launcher_root}")
    resolved_results_root = RESULTS_ROOT.resolve()
    resolved_models_root = MODELS_ROOT.resolve()
    seen: set[Path] = set()
    for plan in plans:
        expected_name = re.compile(
            rf"^{re.escape(plan.scenario.lower())}_r9_ppo_[0-9]{{8}}_[0-9]{{6}}_[0-9]{{6}}$"
        )
        result_dir = validate_personal_output_path(plan.result_dir)
        model_dir = validate_personal_output_path(plan.model_dir)
        if result_dir.parent != resolved_results_root or model_dir.parent != resolved_models_root:
            raise ValueError(
                "Scenario outputs must be direct children of personal_train/results and personal_train/models"
            )
        if result_dir.name != model_dir.name or not expected_name.fullmatch(result_dir.name):
            raise ValueError(
                f"Invalid scenario output name for {plan.scenario}: {result_dir.name} / {model_dir.name}"
            )
        if result_dir in seen or model_dir in seen:
            raise ValueError("Scenario output paths must be unique")
        seen.update((result_dir, model_dir))
        existing = [str(path) for path in (result_dir, model_dir) if path.exists()]
        if existing:
            raise FileExistsError("Training output already exists: " + ", ".join(existing))
        if plan.resume is not None and not plan.resume.resolve().is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {plan.resume}")
    return launcher_root


def validate_batch_roots(
    batch_id: str, plans: Sequence[JobPlan] | None = None
) -> tuple[Path, Path] | Path:
    """Compatibility wrapper; new callers validate all flat output leaves."""

    if plans is not None:
        return validate_batch_outputs(batch_id, plans)
    launcher_root = _launcher_run_root(batch_id)
    if launcher_root.exists():
        raise FileExistsError(f"Launcher run already exists: {launcher_root}")
    return launcher_root, launcher_root


def _container_name(batch_id: str, scenario: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9_.-]", "-", batch_id)[:32].strip("-.")
    digest = hashlib.sha256(batch_id.encode("utf-8")).hexdigest()[:8]
    return f"r9-{compact}-{digest}-{scenario.lower()}"


def docker_command(
    *,
    args: argparse.Namespace,
    plan: JobPlan,
    device: str,
    uid: int,
    gid: int,
    username: str,
    launcher_root: Path | None = None,
) -> list[str]:
    control_root = launcher_root or _launcher_run_root(args.batch_id)
    cidfile = control_root / "launcher_logs" / f"{plan.scenario}.cid"
    command = [
        args.docker,
        "run",
        "--rm",
        "--name",
        _container_name(args.batch_id, plan.scenario),
        "--cidfile",
        str(cidfile),
        "--stop-timeout",
        str(args.stop_timeout),
        "--user",
        f"{uid}:{gid}",
        "--tmpfs",
        f"/app/Results:rw,nosuid,nodev,uid={uid},gid={gid},mode=0775",
        "--label",
        f"personal_train.batch={args.batch_id}",
        "--label",
        f"personal_train.scenario={plan.scenario}",
        "--label",
        f"personal_train.owner={args.ownership_token}",
    ]
    trainer_device = "cpu"
    if device != "cpu":
        command.extend(("--gpus", f"device={device}"))
        trainer_device = "cuda:0"
    command.extend(
        (
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            f"USER={username}",
            "-e",
            f"LOGNAME={username}",
            "-e",
            "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor",
            "-e",
            "COMPETITION_REPO_ROOT=/opt/competition-platform-env",
            "-e",
            "PERSONAL_NATIVE_RESULTS=/app/Results",
            "-e",
            f"OMP_NUM_THREADS={args.threads_per_worker}",
            "-e",
            f"MKL_NUM_THREADS={args.threads_per_worker}",
            "-e",
            f"OPENBLAS_NUM_THREADS={args.threads_per_worker}",
            "--mount",
            f"type=bind,src={PERSONAL_ROOT.resolve()},dst=/app/personal_train",
            "--mount",
            f"type=bind,src={REPOSITORY_ROOT.resolve()},dst=/opt/competition-platform-env,readonly",
            "-w",
            "/app",
            args.image,
            "python3",
            TRAINER_IN_CONTAINER,
            "--scenario",
            plan.scenario,
            "--rounds",
            str(args.rounds),
            "--seed",
            str(args.seed),
            "--blue-policy",
            args.blue_policy,
            "--device",
            trainer_device,
            "--render-mode",
            "none",
            "--run-id",
            plan.run_name,
            "--result-dir",
            _container_personal_path(plan.result_dir),
            "--model-dir",
            _container_personal_path(plan.model_dir),
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--progress-every",
            str(args.progress_every),
            "--disable-log-color",
        )
    )
    if plan.resume is not None:
        command.extend(("--resume", _container_personal_path(plan.resume)))
    if args.verbose_workers:
        command.append("--verbose")
    return command


def _empty_state(
    plans: Sequence[JobPlan], launcher_root: Path | None = None
) -> dict[str, dict[str, Any]]:
    control_root = launcher_root or LAUNCHER_RUNS_ROOT / "unassigned"
    return {
        plan.scenario: {
            "status": "pending",
            "initialization": plan.initialization,
            "resume": str(plan.resume) if plan.resume else None,
            "initial_recorded_score": plan.initial_score,
            "result_dir": str(plan.result_dir),
            "model_dir": str(plan.model_dir),
            "device": None,
            "container": None,
            "container_id": None,
            "cidfile": str(control_root / "launcher_logs" / f"{plan.scenario}.cid"),
            "pid": None,
            "started_at": None,
            "finished_at": None,
            "elapsed_seconds": None,
            "return_code": None,
            "error": None,
        }
        for plan in plans
    }


def _read_scores(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except (OSError, csv.Error, UnicodeError):
        return []


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_scores(result_dir: Path) -> list[tuple[int, float]]:
    numeric: list[tuple[int, float]] = []
    for row in _read_scores(result_dir / "round_scores.csv"):
        score = _safe_float(row.get("score"))
        try:
            round_index = int(row.get("round", 0))
        except (TypeError, ValueError):
            continue
        if score is not None:
            numeric.append((round_index, score))
    return numeric


def validate_job_outputs(plan: JobPlan, expected_rounds: int) -> list[str]:
    errors = []
    completed = _numeric_scores(plan.result_dir)
    if len(completed) != expected_rounds:
        errors.append(f"expected {expected_rounds} scored rounds, found {len(completed)}")
    elif [round_index for round_index, _ in completed] != list(range(1, expected_rounds + 1)):
        errors.append("scored round indices are not exactly 1..expected_rounds")
    required = (
        plan.result_dir / "round_scores.txt",
        plan.result_dir / "round_scores.svg",
        plan.result_dir / "training_dashboard.svg",
        plan.result_dir / "run_config.json",
        plan.model_dir / "latest.pt",
        plan.model_dir / "best.pt",
        plan.model_dir / "model_metadata.json",
        plan.model_dir / "checkpoint_validation.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        errors.append("missing required outputs: " + ", ".join(missing))
    empty = [str(path) for path in required if path.is_file() and path.stat().st_size == 0]
    if empty:
        errors.append("empty required outputs: " + ", ".join(empty))
    unexpected_markers = [
        str(path)
        for path in (plan.result_dir / "failure.json", plan.result_dir / "interruption.json")
        if path.exists()
    ]
    if unexpected_markers:
        errors.append("training left failure/interruption markers: " + ", ".join(unexpected_markers))

    try:
        run_config = json.loads((plan.result_dir / "run_config.json").read_text(encoding="utf-8"))
        if (
            str(run_config.get("scenario", "")).upper() != plan.scenario
            or int(run_config.get("rounds", -1)) != expected_rounds
        ):
            errors.append("run_config.json scenario/round count does not match the job")
    except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
        if (plan.result_dir / "run_config.json").is_file():
            errors.append("run_config.json is invalid")

    try:
        metadata = json.loads((plan.model_dir / "model_metadata.json").read_text(encoding="utf-8"))
        if not _current_e01_metadata(metadata):
            errors.append("model_metadata.json has an incompatible or interrupted schema")
        if int(metadata.get("latest_round", -1)) != expected_rounds:
            errors.append("model_metadata.json latest_round does not match the job")
        best_round = int(metadata.get("best_round", -1))
        best_score = _safe_float(metadata.get("best_official_score"))
        if not 1 <= best_round <= expected_rounds:
            errors.append("model_metadata.json best_round is outside the completed rounds")
        if best_score is None or not 0.0 <= best_score <= 100.0:
            errors.append("model_metadata.json best_official_score is invalid")
        elif completed:
            expected_best_round, expected_best_score = max(completed, key=lambda item: item[1])
            if best_round != expected_best_round or not math.isclose(
                best_score, expected_best_score, rel_tol=1e-9, abs_tol=1e-9
            ):
                errors.append("model_metadata.json best score/round does not match round_scores.csv")
    except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
        if (plan.model_dir / "model_metadata.json").is_file():
            errors.append("model_metadata.json is invalid")

    validation_path = plan.model_dir / "checkpoint_validation.json"
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if (
            not isinstance(validation, dict)
            or int(validation.get("format_version", -1)) != 1
            or validation.get("algorithm") != CURRENT_ALGORITHM
            or not isinstance(validation.get("checkpoints"), dict)
        ):
            raise ValueError("invalid checkpoint validation header")
        for name in ("best.pt", "latest.pt"):
            record = validation["checkpoints"].get(name)
            checkpoint_path = plan.model_dir / name
            if (
                not isinstance(record, dict)
                or record.get("file") != name
                or int(record.get("size_bytes", -1)) != checkpoint_path.stat().st_size
                or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))
                or record["sha256"] != _sha256_file(checkpoint_path)
            ):
                raise ValueError(f"invalid or stale validation record for {name}")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError):
        if validation_path.is_file():
            errors.append("checkpoint_validation.json is invalid or does not match the checkpoints")
    return errors


def batch_summary(plans: Sequence[JobPlan], states: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for plan in plans:
        numeric = _numeric_scores(plan.result_dir)
        if numeric:
            best_round, best_score = max(numeric, key=lambda item: item[1])
            final_round, final_score = numeric[-1]
            mean_score = sum(score for _, score in numeric) / len(numeric)
        else:
            best_round = final_round = 0
            best_score = final_score = mean_score = None
        initial = plan.initial_score
        available = [value for value in (initial, best_score) if value is not None]
        rows.append(
            {
                "scenario": plan.scenario,
                "status": states[plan.scenario]["status"],
                "initialization": plan.initialization,
                "rounds_completed": len(numeric),
                "initial_recorded_score": initial,
                "best_round_in_batch": best_round,
                "best_score_in_batch": best_score,
                "final_round": final_round,
                "final_score": final_score,
                "mean_score": mean_score,
                "best_score_including_initial": max(available) if available else None,
                "return_code": states[plan.scenario]["return_code"],
                "elapsed_seconds": states[plan.scenario]["elapsed_seconds"],
            }
        )
    return rows


def _batch_svg(rows: Sequence[dict[str, Any]]) -> str:
    width, height = 1040, 520
    left, right, top, bottom = 80, 35, 55, 80
    plot_width, plot_height = width - left - right, height - top - bottom
    group_width = plot_width / max(1, len(rows))
    bar_width = min(30.0, group_width * 0.30)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:"DejaVu Sans",Arial,sans-serif;fill:#172033}</style>',
        '<text x="520" y="30" text-anchor="middle" font-size="21" font-weight="bold">R9 PPO multi-scenario batch scores</text>',
    ]
    for tick in range(0, 101, 20):
        y = top + plot_height * (1.0 - tick / 100.0)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dbe2ea"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="12">{tick}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#475569"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" stroke="#475569"/>')
    for index, row in enumerate(rows):
        center = left + group_width * (index + 0.5)
        best = _safe_float(row.get("best_score_in_batch")) or 0.0
        final = _safe_float(row.get("final_score")) or 0.0
        for offset, value, color in ((-bar_width, best, "#2563eb"), (0.0, final, "#f97316")):
            value = max(0.0, min(100.0, value))
            bar_height = plot_height * value / 100.0
            x = center + offset
            y = top + plot_height - bar_height
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" rx="2"/>')
        parts.append(f'<text x="{center:.1f}" y="{top+plot_height+25}" text-anchor="middle" font-size="13">{html.escape(str(row["scenario"]))}</text>')
    legend_y = height - 24
    parts.extend(
        (
            f'<rect x="365" y="{legend_y-12}" width="16" height="12" fill="#2563eb"/><text x="388" y="{legend_y}" font-size="13">Best in batch</text>',
            f'<rect x="535" y="{legend_y-12}" width="16" height="12" fill="#f97316"/><text x="558" y="{legend_y}" font-size="13">Final</text>',
            "</svg>",
        )
    )
    return "\n".join(parts) + "\n"


def write_batch_reports(batch_root: Path, rows: Sequence[dict[str, Any]]) -> None:
    _atomic_json(batch_root / "batch_summary.json", list(rows))
    fields = tuple(rows[0].keys()) if rows else (
        "scenario",
        "status",
        "initialization",
        "rounds_completed",
        "best_score_in_batch",
        "final_score",
    )
    csv_tmp = batch_root / "batch_summary.csv.tmp"
    with csv_tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    csv_tmp.replace(batch_root / "batch_summary.csv")
    lines = ["R9 PPO multi-scenario batch summary"]
    for row in rows:
        best = row.get("best_score_in_batch")
        final = row.get("final_score")
        lines.append(
            f"scenario={row['scenario']} status={row['status']} "
            f"rounds={row['rounds_completed']} "
            f"best={'NA' if best is None else f'{best:.6f}'} "
            f"final={'NA' if final is None else f'{final:.6f}'}"
        )
    _atomic_text(batch_root / "batch_scores.txt", "\n".join(lines) + "\n")
    _atomic_text(batch_root / "batch_dashboard.svg", _batch_svg(rows))


def _status_document(
    *,
    batch_id: str,
    created_at: str,
    states: dict[str, dict[str, Any]],
    interrupted: bool,
) -> dict[str, Any]:
    values = [state["status"] for state in states.values()]
    return {
        "batch_id": batch_id,
        "created_at": created_at,
        "updated_at": datetime.now().astimezone().isoformat(),
        "interrupted": interrupted,
        "counts": {status: values.count(status) for status in sorted(set(values))},
        "jobs": states,
    }


def _container_id_from_cidfile(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value if re.fullmatch(r"[0-9a-fA-F]{12,64}", value) else None


def _container_identity(args: argparse.Namespace, running: RunningJob) -> tuple[str, str | None]:
    """Return owned/absent/foreign/unknown and a safe Docker reference."""

    container_id = _container_id_from_cidfile(running.cidfile)
    if container_id is not None:
        return "owned", container_id
    try:
        inspected = subprocess.run(
            [
                args.docker,
                "inspect",
                "--format",
                "{{json .Config.Labels}}",
                running.container_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
            check=False,
        )
        if inspected.returncode != 0:
            detail = (inspected.stderr or "").lower()
            if "no such object" in detail or "no such container" in detail:
                try:
                    labeled = subprocess.run(
                        [
                            args.docker,
                            "ps",
                            "-aq",
                            "--filter",
                            f"label=personal_train.batch={args.batch_id}",
                            "--filter",
                            f"label=personal_train.scenario={running.plan.scenario}",
                            "--filter",
                            f"label=personal_train.owner={args.ownership_token}",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=3,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    return "unknown", None
                if labeled.returncode != 0:
                    return "unknown", None
                matches = [line.strip() for line in labeled.stdout.splitlines() if line.strip()]
                return ("owned", matches[0]) if matches else ("absent", None)
            return "unknown", None
        labels = json.loads(inspected.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError, json.JSONDecodeError):
        return "unknown", None
    if not isinstance(labels, dict):
        return "foreign", None
    if (
        labels.get("personal_train.batch") == args.batch_id
        and labels.get("personal_train.scenario") == running.plan.scenario
        and labels.get("personal_train.owner") == args.ownership_token
    ):
        return "owned", running.container_name
    return "foreign", None


def _container_is_stopped(args: argparse.Namespace, reference: str) -> bool:
    try:
        inspected = subprocess.run(
            [args.docker, "inspect", "--format", "{{.State.Running}}", reference],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if inspected.returncode == 0:
        return inspected.stdout.strip().lower() == "false"
    detail = (inspected.stderr or "").lower()
    return "no such object" in detail or "no such container" in detail


def _stop_containers(args: argparse.Namespace, active: dict[str, RunningJob]) -> bool:
    references = []
    unresolved = []
    for running in active.values():
        identity, reference = _container_identity(args, running)
        if identity == "owned" and reference is not None:
            references.append(reference)
        elif identity == "foreign" and running.process.poll() is not None:
            continue
        else:
            unresolved.append(running)

    # Give an in-flight ``docker run`` a bounded opportunity to publish its
    # cidfile/name, then stop its client and perform one final ownership check.
    for _ in range(4):
        if not unresolved:
            break
        time.sleep(0.25)
        retry = []
        for running in unresolved:
            identity, reference = _container_identity(args, running)
            if identity == "owned" and reference is not None:
                references.append(reference)
            elif identity in {"absent", "foreign"} and running.process.poll() is not None:
                continue
            else:
                retry.append(running)
        unresolved = retry

    for running in unresolved:
        if running.process.poll() is None:
            running.process.terminate()
            try:
                running.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                running.process.kill()
                try:
                    running.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
    final_unknown = []
    for running in unresolved:
        if running.process.poll() is None:
            final_unknown.append(running)
            continue
        identity, reference = _container_identity(args, running)
        if identity == "absent":
            time.sleep(0.25)
            identity, reference = _container_identity(args, running)
        if identity == "owned" and reference is not None:
            references.append(reference)
        elif identity in {"absent", "foreign"}:
            continue
        else:
            final_unknown.append(running)
    if not references:
        return not final_unknown
    try:
        completed = subprocess.run(
            [
                args.docker,
                "stop",
                "--signal",
                "TERM",
                "--timeout",
                str(args.stop_timeout),
                *references,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=args.stop_timeout + 30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        completed = None
    if completed is None or completed.returncode != 0:
        try:
            subprocess.run(
                [args.docker, "kill", *references],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    return not final_unknown and all(
        _container_is_stopped(args, reference) for reference in references
    )


def _username_for_uid(uid: int) -> str:
    """Numeric Docker users need not have an entry in /etc/passwd."""
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return f"uid{uid}"


def _checkpoint_document(checkpoint: ScenarioCheckpoint) -> dict[str, Any]:
    digest = _sha256_file(checkpoint.path) if checkpoint.path.is_file() else None
    if checkpoint.sha256 is not None and digest != checkpoint.sha256:
        raise ValueError(f"Resume checkpoint changed after validation: {checkpoint.path}")
    return {
        "scenario": checkpoint.scenario,
        "path": str(checkpoint.path.resolve()),
        "source_batch": checkpoint.source_batch,
        "selection": checkpoint.selection,
        "recorded_best_score": checkpoint.recorded_score,
        "recorded_best_round": checkpoint.best_round,
        "recorded_latest_round": checkpoint.latest_round,
        "source_run_interrupted": checkpoint.interrupted,
        "sha256": digest,
        "resume_mode_expected": "current_weights_only",
        "resume_semantics": "network weights initialization; optimizer, counters, and RNG are not restored",
    }


def run_batch(
    args: argparse.Namespace,
    plans: Sequence[JobPlan],
    checkpoints: Mapping[str, ScenarioCheckpoint] | ScenarioCheckpoint | None,
) -> int:
    checkpoint_by_scenario = _checkpoint_mapping(checkpoints)
    planned_resumes = {plan.scenario: plan.resume for plan in plans if plan.resume is not None}
    if set(planned_resumes) != set(checkpoint_by_scenario):
        raise ValueError("Every planned resume must have exactly one provenance record")
    for scenario, resume in planned_resumes.items():
        if resume is None or resume.resolve() != checkpoint_by_scenario[scenario].path.resolve():
            raise ValueError(f"Resume provenance path does not match the {scenario} job plan")
    launcher_root = validate_batch_outputs(args.batch_id, plans)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    LAUNCHER_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    launcher_root.mkdir(parents=True, exist_ok=False)
    logs_root = launcher_root / "launcher_logs"
    logs_root.mkdir()

    uid, gid = os.getuid(), os.getgid()
    username = _username_for_uid(uid)
    created_at = datetime.now().astimezone().isoformat()
    states = _empty_state(plans, launcher_root)
    resume_documents = {
        scenario: _checkpoint_document(checkpoint)
        for scenario, checkpoint in checkpoint_by_scenario.items()
    }
    config = {
        "batch_id": args.batch_id,
        "created_at": created_at,
        "rounds_per_scenario": args.rounds,
        "seed_per_scenario": args.seed,
        "scenarios": [plan.scenario for plan in plans],
        "blue_policy": args.blue_policy,
        "docker_image": args.image,
        "launcher_ownership_token": args.ownership_token,
        "device_slots": list(args.devices),
        "max_parallel": args.max_parallel,
        "gpu_sharing": args.max_parallel > len(args.devices),
        "threads_per_worker": args.threads_per_worker,
        "results_root": str(RESULTS_ROOT.resolve()),
        "models_root": str(MODELS_ROOT.resolve()),
        "launcher_run_dir": str(launcher_root),
        "native_results_isolation": "one Docker tmpfs /app/Results per scenario",
        "resume_batches": sorted(
            {
                checkpoint.source_batch
                for checkpoint in checkpoint_by_scenario.values()
                if checkpoint.source_batch is not None
            }
        ),
        "resume_checkpoints": resume_documents,
        "jobs": [
            {
                **asdict(plan),
                "result_dir": str(plan.result_dir),
                "model_dir": str(plan.model_dir),
                "resume": str(plan.resume) if plan.resume else None,
                "initialization": plan.initialization,
                "source_checkpoint": resume_documents.get(plan.scenario),
            }
            for plan in plans
        ],
    }
    _atomic_json(launcher_root / "batch_config.json", config)
    _atomic_json(
        launcher_root / "batch_status.json",
        _status_document(batch_id=args.batch_id, created_at=created_at, states=states, interrupted=False),
    )
    write_batch_reports(launcher_root, batch_summary(plans, states))
    print(f"LAUNCHER_RUN_DIR {launcher_root}", flush=True)
    for plan in plans:
        print(
            f"OUTPUT {plan.scenario} result={plan.result_dir} model={plan.model_dir} resume={plan.resume}",
            flush=True,
        )

    pending = deque(plans)
    active: dict[str, RunningJob] = {}
    free_slots = deque(range(args.max_parallel))
    requested_signal: int | None = None
    launcher_error: str | None = None

    def request_stop(signum, _frame) -> None:
        nonlocal requested_signal
        requested_signal = int(signum)

    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    previous_handlers = {signum: signal.getsignal(signum) for signum in handled_signals}
    for signum in handled_signals:
        signal.signal(signum, request_stop)

    def persist(interrupted: bool = False) -> None:
        _atomic_json(
            launcher_root / "batch_status.json",
            _status_document(
                batch_id=args.batch_id,
                created_at=created_at,
                states=states,
                interrupted=interrupted,
            ),
        )
        write_batch_reports(launcher_root, batch_summary(plans, states))

    try:
        while pending or active:
            if requested_signal is not None:
                while pending:
                    plan = pending.popleft()
                    states[plan.scenario]["status"] = "cancelled"
                    states[plan.scenario]["finished_at"] = datetime.now().astimezone().isoformat()
                    states[plan.scenario]["error"] = f"launcher received signal {requested_signal}"
                persist(interrupted=True)
                _stop_containers(args, active)

            while requested_signal is None and pending and free_slots:
                plan = pending.popleft()
                slot = free_slots.popleft()
                device = args.devices[slot % len(args.devices)]
                name = _container_name(args.batch_id, plan.scenario)
                log_path = logs_root / f"{plan.scenario}.log"
                started_at = datetime.now().astimezone().isoformat()
                state = states[plan.scenario]
                state.update(
                    {
                        "status": "starting",
                        "device": device,
                        "container": name,
                        "started_at": started_at,
                    }
                )
                log_stream = None
                try:
                    if plan.resume is not None:
                        expected_digest = resume_documents[plan.scenario]["sha256"]
                        if _sha256_file(plan.resume) != expected_digest:
                            raise ValueError(
                                f"Resume checkpoint changed before {plan.scenario} launch: {plan.resume}"
                            )
                    command = docker_command(
                        args=args,
                        plan=plan,
                        device=device,
                        uid=uid,
                        gid=gid,
                        username=username,
                        launcher_root=launcher_root,
                    )
                    log_stream = log_path.open("w", encoding="utf-8", buffering=1)
                    log_stream.write(f"COMMAND {shlex.join(command)}\n")
                    process = subprocess.Popen(
                        command,
                        cwd=REPOSITORY_ROOT,
                        stdout=log_stream,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=True,
                    )
                except Exception as error:
                    if log_stream is not None:
                        try:
                            log_stream.write(f"LAUNCH_ERROR {type(error).__name__}: {error}\n")
                        except OSError:
                            pass
                        finally:
                            log_stream.close()
                    state.update(
                        {
                            "status": "failed",
                            "device": device,
                            "container": name,
                            "started_at": started_at,
                            "finished_at": datetime.now().astimezone().isoformat(),
                            "return_code": 127,
                            "elapsed_seconds": 0.0,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    free_slots.append(slot)
                    persist()
                    continue
                active[plan.scenario] = RunningJob(
                    plan=plan,
                    device=device,
                    slot=slot,
                    container_name=name,
                    cidfile=logs_root / f"{plan.scenario}.cid",
                    process=process,
                    log_stream=log_stream,
                    started_monotonic=time.monotonic(),
                )
                states[plan.scenario].update(
                    {
                        "status": "running",
                        "device": device,
                        "container": name,
                        "container_id": _container_id_from_cidfile(
                            logs_root / f"{plan.scenario}.cid"
                        ),
                        "pid": process.pid,
                        "started_at": started_at,
                    }
                )
                print(f"START {plan.scenario} device={device} container={name} log={log_path}", flush=True)
                persist()

            finished = []
            cleanup_failures = []
            for scenario, running in active.items():
                return_code = running.process.poll()
                if return_code is None:
                    continue
                cleanup_confirmed = True
                if return_code != 0:
                    # An attached Docker client can fail before its container
                    # exits (daemon/terminal disruption). Stop only the
                    # container ID/labels proven to belong to this batch before
                    # reusing the GPU slot.
                    cleanup_confirmed = _stop_containers(args, {scenario: running})
                running.log_stream.close()
                elapsed = time.monotonic() - running.started_monotonic
                state = states[scenario]
                state["container_id"] = _container_id_from_cidfile(running.cidfile)
                if return_code == 0:
                    output_errors = validate_job_outputs(running.plan, args.rounds)
                    status = "failed" if output_errors else "succeeded"
                elif return_code == 130 or requested_signal is not None:
                    output_errors = []
                    status = "interrupted"
                else:
                    output_errors = []
                    status = "failed"
                if not cleanup_confirmed:
                    output_errors.append("owned container cleanup could not be confirmed")
                    status = "failed"
                    cleanup_failures.append(scenario)
                state.update(
                    {
                        "status": status,
                        "finished_at": datetime.now().astimezone().isoformat(),
                        "elapsed_seconds": round(elapsed, 3),
                        "return_code": int(return_code),
                        "error": (
                            "; ".join(output_errors)
                            if output_errors
                            else None if return_code == 0 else f"container exited with code {return_code}"
                        ),
                    }
                )
                print(f"DONE {scenario} status={status} code={return_code} elapsed={elapsed:.1f}s", flush=True)
                if cleanup_confirmed:
                    free_slots.append(running.slot)
                    finished.append(scenario)
            for scenario in finished:
                del active[scenario]
            if finished:
                persist(interrupted=requested_signal is not None)
            if cleanup_failures:
                raise RuntimeError(
                    "Cannot safely reuse a device slot after unconfirmed container cleanup: "
                    + ", ".join(cleanup_failures)
                )
            if requested_signal is not None and not active:
                break
            if pending or active:
                time.sleep(0.5)
    except BaseException as error:
        launcher_error = f"{type(error).__name__}: {error}"
        raise
    finally:
        if active:
            cleanup_confirmed = _stop_containers(args, active)
            for scenario, running in active.items():
                try:
                    running.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    running.process.terminate()
                    try:
                        running.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        running.process.kill()
                        running.process.wait(timeout=5)
                running.log_stream.close()
                return_code = running.process.poll()
                elapsed = time.monotonic() - running.started_monotonic
                state = states[scenario]
                state.update(
                    {
                        "status": "interrupted" if requested_signal is not None else "failed",
                        "finished_at": datetime.now().astimezone().isoformat(),
                        "elapsed_seconds": round(elapsed, 3),
                        "return_code": int(return_code) if return_code is not None else None,
                        "error": (
                            f"launcher received signal {requested_signal}"
                            if requested_signal is not None
                            else launcher_error or "launcher stopped the active container"
                        ),
                    }
                )
                if not cleanup_confirmed:
                    state["error"] += "; owned container cleanup could not be confirmed"
        while pending:
            plan = pending.popleft()
            state = states[plan.scenario]
            state.update(
                {
                    "status": "cancelled",
                    "finished_at": datetime.now().astimezone().isoformat(),
                    "error": (
                        f"launcher received signal {requested_signal}"
                        if requested_signal is not None
                        else launcher_error or "launcher ended before this job started"
                    ),
                }
            )
        for state in states.values():
            if state["status"] == "starting":
                state.update(
                    {
                        "status": "failed",
                        "finished_at": datetime.now().astimezone().isoformat(),
                        "error": launcher_error or "launcher failed while starting this job",
                    }
                )
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        persist(interrupted=requested_signal is not None)

    if requested_signal is not None:
        return 130
    return 0 if all(state["status"] == "succeeded" for state in states.values()) else 1


def _dry_run_document(
    args: argparse.Namespace,
    plans: Sequence[JobPlan],
    checkpoints: Mapping[str, ScenarioCheckpoint] | ScenarioCheckpoint | None,
) -> dict[str, Any]:
    uid, gid = os.getuid(), os.getgid()
    username = _username_for_uid(uid)
    checkpoint_by_scenario = _checkpoint_mapping(checkpoints)
    launcher_root = _launcher_run_root(args.batch_id)
    resume_documents = {
        scenario: _checkpoint_document(checkpoint)
        for scenario, checkpoint in checkpoint_by_scenario.items()
    }
    jobs = []
    for index, plan in enumerate(plans):
        slot = index % args.max_parallel
        device = args.devices[slot % len(args.devices)]
        jobs.append(
            {
                "scenario": plan.scenario,
                "initialization": plan.initialization,
                "device_slot_preview": device,
                "result_dir": str(plan.result_dir),
                "model_dir": str(plan.model_dir),
                "resume": str(plan.resume) if plan.resume else None,
                "source_checkpoint": resume_documents.get(plan.scenario),
                "command": docker_command(
                    args=args,
                    plan=plan,
                    device=device,
                    uid=uid,
                    gid=gid,
                    username=username,
                    launcher_root=launcher_root,
                ),
            }
        )
    return {
        "dry_run": True,
        "batch_id": args.batch_id,
        "rounds_per_scenario": args.rounds,
        "max_parallel": args.max_parallel,
        "device_slots": list(args.devices),
        "launcher_run_dir": str(launcher_root),
        "resume_checkpoints": resume_documents,
        "jobs": jobs,
    }


def _cleanup_preflight_container(
    args: argparse.Namespace, container_name: str, ownership_token: str
) -> bool:
    try:
        inspected = subprocess.run(
            [args.docker, "inspect", "--format", "{{json .Config.Labels}}", container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if inspected.returncode != 0:
        detail = (inspected.stderr or "").lower()
        return "no such object" in detail or "no such container" in detail
    try:
        labels = json.loads(inspected.stdout)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(labels, dict) or labels.get("personal_train.preflight") != ownership_token:
        # A name collision is foreign state and must never be stopped here.
        return True
    try:
        stopped = subprocess.run(
            [args.docker, "stop", "--signal", "TERM", "--timeout", "5", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        stopped = None
    if stopped is None or stopped.returncode != 0:
        try:
            subprocess.run(
                [args.docker, "kill", container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    return _container_is_stopped(args, container_name)


def preflight_docker(args: argparse.Namespace) -> None:
    def interrupt_preflight(signum, _frame) -> None:
        raise KeyboardInterrupt(f"preflight received signal {signum}")

    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    previous_handlers = {signum: signal.getsignal(signum) for signum in handled_signals}
    for signum in handled_signals:
        signal.signal(signum, interrupt_preflight)
    try:
        try:
            completed = subprocess.run(
                [args.docker, "image", "inspect", args.image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
                check=False,
            )
        except FileNotFoundError as error:
            raise RuntimeError(f"Docker CLI was not found: {args.docker}") from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("Docker image inspection timed out") from error
        if completed.returncode != 0:
            detail = (completed.stderr or "").strip()
            raise RuntimeError(
                f"Docker image is unavailable or the daemon cannot be used: {args.image}"
                + (f"\n{detail}" if detail else "")
            )
        for device in args.devices:
            if device == "cpu":
                continue
            ownership_token = hashlib.sha256(
                f"{args.ownership_token}:{device}:{os.getpid()}".encode("utf-8")
            ).hexdigest()
            digest = ownership_token[:8]
            container_name = f"r9-preflight-{os.getpid()}-{digest}"
            command = [
                args.docker,
                "run",
                "--rm",
                "--network",
                "none",
                "--name",
                container_name,
                "--label",
                f"personal_train.preflight={ownership_token}",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--gpus",
                f"device={device}",
                args.image,
                "python3",
                "-c",
                (
                    "import sys, torch; "
                    "sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() == 1 else 3)"
                ),
            ]
            try:
                gpu_check = subprocess.run(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=60,
                    check=False,
                )
            except BaseException as error:
                cleaned = _cleanup_preflight_container(args, container_name, ownership_token)
                suffix = "" if cleaned else "; container cleanup could not be confirmed"
                if isinstance(error, subprocess.TimeoutExpired):
                    raise RuntimeError(
                        f"GPU preflight timed out for host device {device}{suffix}"
                    ) from error
                if isinstance(error, OSError):
                    raise RuntimeError(
                        f"GPU preflight could not start for host device {device}{suffix}"
                    ) from error
                if not cleaned:
                    raise RuntimeError(
                        f"GPU preflight was interrupted for host device {device}{suffix}"
                    ) from error
                raise
            if gpu_check.returncode != 0:
                detail = (gpu_check.stderr or "").strip()
                cleaned = _cleanup_preflight_container(args, container_name, ownership_token)
                cleanup_detail = "" if cleaned else "\nContainer cleanup could not be confirmed."
                raise RuntimeError(
                    f"Docker cannot expose host GPU {device} as one CUDA device"
                    + (f"\n{detail}" if detail else "")
                    + cleanup_detail
                )
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main(argv: list[str] | None = None) -> int:
    invocation_cwd = Path.cwd()
    args = parse_args(argv)
    missing_scenarios = [
        scenario
        for scenario in args.scenarios
        if not (
            SCENARIOS_ROOT
            / "cases"
            / {"E": "easy", "M": "medium", "H": "hard"}[scenario[0]]
            / scenario
            / "scenario.json"
        ).is_file()
    ]
    if missing_scenarios:
        raise FileNotFoundError(
            f"Scenario files are missing for: {', '.join(missing_scenarios)}"
        )
    output_timestamp = make_run_timestamp()
    args.batch_id = make_batch_id(args.batch_id, timestamp=output_timestamp)
    if args.resume_batch is not None:
        checkpoints: dict[str, ScenarioCheckpoint] = resolve_resume_batch(
            args.resume_batch, args.scenarios, invocation_cwd
        )
    else:
        checkpoints = {}
        if "E01" in args.scenarios:
            checkpoint = resolve_e01_checkpoint(args.e01_resume, invocation_cwd)
            checkpoints["E01"] = checkpoint
    plans = build_job_plans(args.scenarios, output_timestamp, checkpoints)
    validate_batch_outputs(args.batch_id, plans)
    if args.dry_run:
        print(
            json.dumps(
                _dry_run_document(args, plans, checkpoints),
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return 0
    preflight_docker(args)
    return run_batch(args, plans, checkpoints)


if __name__ == "__main__":
    raise SystemExit(main())
