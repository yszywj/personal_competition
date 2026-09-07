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
from typing import Any, Sequence


PERSONAL_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PERSONAL_ROOT.parent
SCENARIOS_ROOT = REPOSITORY_ROOT / "scenarios"
RESULTS_ROOT = PERSONAL_ROOT / "results"
MODELS_ROOT = PERSONAL_ROOT / "models"
TRAINER_IN_CONTAINER = "/app/personal_train/train_r9_ppo.py"
PERSONAL_IN_CONTAINER = Path("/app/personal_train")
ALL_SCENARIOS = ("E01", "E02", "E03", "M01", "M02", "M03", "H01", "H02", "H03")
CURRENT_SCHEMA_NAME = "personal-r9-ppo"
CURRENT_SCHEMA_VERSION = 2
CURRENT_ALGORITHM = "personal_ppo_gae_v2"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_GPU_ID = re.compile(r"^(?:[0-9]+|GPU-[A-Fa-f0-9-]+)$")


@dataclass(frozen=True)
class E01Checkpoint:
    path: Path
    recorded_score: float | None
    selection: str


@dataclass(frozen=True)
class JobPlan:
    scenario: str
    result_dir: Path
    model_dir: Path
    resume: Path | None = None
    initial_score: float | None = None

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
            "E01 starts from an existing best checkpoint; all other cases start fresh."
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
        "--e01-resume",
        type=Path,
        default=None,
        help=(
            "E01 best.pt on the host. If omitted, the compatible E01 best with "
            "the highest recorded score below personal_train/models is selected."
        ),
    )
    parser.add_argument("--seed", type=int, default=1, help="One comparable seed per scenario.")
    parser.add_argument("--blue-policy", default="b0_fixed_ratio_random")
    parser.add_argument("--image", default="competition:ppo", help="Docker training image.")
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
        help="Exact common directory name; default: r9_multi_<timestamp>.",
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


def _current_e01_metadata(metadata: Any) -> bool:
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
        and not bool(metadata.get("interrupted", False))
    )


def _model_scenario(metadata_path: Path) -> str | None:
    try:
        relative_dir = metadata_path.parent.relative_to(MODELS_ROOT)
    except ValueError:
        return None
    run_config = RESULTS_ROOT / relative_dir / "run_config.json"
    try:
        value = json.loads(run_config.read_text(encoding="utf-8")).get("scenario")
        return str(value).upper() if value else None
    except (OSError, ValueError, AttributeError):
        pass
    lowered = relative_dir.name.lower()
    if lowered == "e01" or lowered.startswith("e01_"):
        return "E01"
    return None


def _model_run_is_complete(metadata_path: Path, metadata: dict[str, Any]) -> bool:
    try:
        relative_dir = metadata_path.parent.relative_to(MODELS_ROOT)
        result_dir = RESULTS_ROOT / relative_dir
        run_config = json.loads((result_dir / "run_config.json").read_text(encoding="utf-8"))
        requested_rounds = int(run_config["rounds"])
        latest_round = int(metadata["latest_round"])
        best_round = int(metadata["best_round"])
        best_score = float(metadata["best_official_score"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return bool(
        requested_rounds > 0
        and latest_round == requested_rounds
        and 1 <= best_round <= latest_round
        and math.isfinite(best_score)
        and 0.0 <= best_score <= 100.0
        and not (result_dir / "failure.json").exists()
        and not (result_dir / "interruption.json").exists()
    )


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
    return E01Checkpoint(path=path, recorded_score=score, selection="highest_compatible_recorded_score")


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
    )


def _container_personal_path(host_path: Path) -> str:
    resolved = host_path.resolve()
    try:
        relative = resolved.relative_to(PERSONAL_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"Path must be below {PERSONAL_ROOT} because only personal_train is mounted: {resolved}"
        ) from error
    return str(PERSONAL_IN_CONTAINER / relative)


def make_batch_id(value: str | None, now: datetime | None = None) -> str:
    if value is not None:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("Invalid batch ID")
        return value
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
    return f"r9_multi_{timestamp}"


def build_job_plans(
    scenarios: Sequence[str],
    batch_id: str,
    e01_checkpoint: E01Checkpoint | None,
) -> list[JobPlan]:
    plans = []
    for scenario in scenarios:
        checkpoint = e01_checkpoint if scenario == "E01" else None
        if scenario == "E01" and checkpoint is None:
            raise ValueError("E01 requires an existing best checkpoint")
        plans.append(
            JobPlan(
                scenario=scenario,
                result_dir=RESULTS_ROOT / batch_id / scenario,
                model_dir=MODELS_ROOT / batch_id / scenario,
                resume=checkpoint.path if checkpoint else None,
                initial_score=checkpoint.recorded_score if checkpoint else None,
            )
        )
    return plans


def validate_batch_roots(batch_id: str) -> tuple[Path, Path]:
    result_root = RESULTS_ROOT / batch_id
    model_root = MODELS_ROOT / batch_id
    if result_root.exists() or model_root.exists():
        raise FileExistsError(
            f"Batch output already exists; choose another --batch-id: {batch_id}"
        )
    return result_root, model_root


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
) -> list[str]:
    cidfile = plan.result_dir.parent / "launcher_logs" / f"{plan.scenario}.cid"
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
            f"OMP_NUM_THREADS={args.threads_per_worker}",
            "-e",
            f"MKL_NUM_THREADS={args.threads_per_worker}",
            "-e",
            f"OPENBLAS_NUM_THREADS={args.threads_per_worker}",
            "-v",
            f"{PERSONAL_ROOT.resolve()}:/app/personal_train:rw",
            "-v",
            f"{SCENARIOS_ROOT.resolve()}:/app/scenarios:ro",
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
            plan.scenario.lower(),
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


def _empty_state(plans: Sequence[JobPlan]) -> dict[str, dict[str, Any]]:
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
            "cidfile": str(plan.result_dir.parent / "launcher_logs" / f"{plan.scenario}.cid"),
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


def run_batch(args: argparse.Namespace, plans: Sequence[JobPlan], checkpoint: E01Checkpoint | None) -> int:
    result_root, model_root = validate_batch_roots(args.batch_id)
    result_root.mkdir(parents=True, exist_ok=False)
    model_root.mkdir(parents=True, exist_ok=False)
    logs_root = result_root / "launcher_logs"
    logs_root.mkdir()

    uid, gid = os.getuid(), os.getgid()
    username = pwd.getpwuid(uid).pw_name
    created_at = datetime.now().astimezone().isoformat()
    states = _empty_state(plans)
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
        "results_root": str(result_root),
        "models_root": str(model_root),
        "native_results_isolation": "one Docker tmpfs /app/Results per scenario",
        "e01_checkpoint": (
            {
                "path": str(checkpoint.path),
                "recorded_score": checkpoint.recorded_score,
                "selection": checkpoint.selection,
                "resume_semantics": (
                    "network weights and saved PPO hyperparameters; "
                    "new optimizer, counters, and RNG for best.pt"
                ),
            }
            if checkpoint is not None
            else None
        ),
        "jobs": [
            {
                **asdict(plan),
                "result_dir": str(plan.result_dir),
                "model_dir": str(plan.model_dir),
                "resume": str(plan.resume) if plan.resume else None,
                "initialization": plan.initialization,
            }
            for plan in plans
        ],
    }
    _atomic_json(result_root / "batch_config.json", config)
    _atomic_json(
        result_root / "batch_status.json",
        _status_document(batch_id=args.batch_id, created_at=created_at, states=states, interrupted=False),
    )
    write_batch_reports(result_root, batch_summary(plans, states))
    print(f"BATCH_RESULT_DIR {result_root}", flush=True)
    print(f"BATCH_MODEL_DIR {model_root}", flush=True)

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
            result_root / "batch_status.json",
            _status_document(
                batch_id=args.batch_id,
                created_at=created_at,
                states=states,
                interrupted=interrupted,
            ),
        )
        write_batch_reports(result_root, batch_summary(plans, states))

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
                    command = docker_command(
                        args=args,
                        plan=plan,
                        device=device,
                        uid=uid,
                        gid=gid,
                        username=username,
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
    checkpoint: E01Checkpoint | None,
) -> dict[str, Any]:
    uid, gid = os.getuid(), os.getgid()
    username = pwd.getpwuid(uid).pw_name
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
                "command": docker_command(
                    args=args,
                    plan=plan,
                    device=device,
                    uid=uid,
                    gid=gid,
                    username=username,
                ),
            }
        )
    return {
        "dry_run": True,
        "batch_id": args.batch_id,
        "rounds_per_scenario": args.rounds,
        "max_parallel": args.max_parallel,
        "device_slots": list(args.devices),
        "e01_checkpoint": asdict(checkpoint) if checkpoint is not None else None,
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
    args.batch_id = make_batch_id(args.batch_id)
    checkpoint = resolve_e01_checkpoint(args.e01_resume, invocation_cwd) if "E01" in args.scenarios else None
    plans = build_job_plans(args.scenarios, args.batch_id, checkpoint)
    validate_batch_roots(args.batch_id)
    if args.dry_run:
        print(json.dumps(_dry_run_document(args, plans, checkpoint), ensure_ascii=False, indent=2, default=str))
        return 0
    preflight_docker(args)
    return run_batch(args, plans, checkpoint)


if __name__ == "__main__":
    raise SystemExit(main())
