#!/usr/bin/env python3
"""Launch independent joint-PPO trainers for several competition scenarios.

The launcher intentionally does *not* combine experience from different cases.
Each case owns one Python process, simulator runtime, policy, optimizer, RNG,
result directory, model directory, and log.  Host GPUs are hidden per worker via
``CUDA_VISIBLE_DEVICES`` so every trainer can consistently use ``cuda:0``.

Unlike :mod:`train_r9_multi_scenario`, this launcher does not need Docker: the
joint trainer already creates a unique private simulator runtime for every
result directory.  Process groups, atomic status files, checkpoint provenance,
bounded signal handling, and strict output validation follow the same safety
principles as that launcher.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


if __package__:
    from .bootstrap import PERSONAL_ROOT, REPOSITORY_ROOT, validate_personal_output_path
else:
    from bootstrap import PERSONAL_ROOT, REPOSITORY_ROOT, validate_personal_output_path


RESULTS_ROOT = PERSONAL_ROOT / "results"
MODELS_ROOT = PERSONAL_ROOT / "models"
LAUNCHER_RUNS_ROOT = PERSONAL_ROOT / "launcher_runs"
TRAINER = PERSONAL_ROOT / "train_joint_ppo.py"
_GLIBC_PYTHON = PERSONAL_ROOT.parent / "glibc-2.38" / "python3.11-glibc238"
DEFAULT_PYTHON = str(_GLIBC_PYTHON if _GLIBC_PYTHON.is_file() else Path(sys.executable))

SCENARIO_SUITES: Mapping[str, tuple[str, ...]] = {
    "legacy": ("E01", "E02", "E03", "M01", "M02", "M03", "H01", "H02", "H03"),
    "final24": tuple(
        f"{difficulty}{index:02d}"
        for difficulty in ("E", "M", "H")
        for index in range(1, 9)
    ),
    "final20": tuple(
        [f"E{index:02d}" for index in range(1, 7)]
        + [f"M{index:02d}" for index in range(1, 7)]
        + [f"H{index:02d}" for index in range(1, 9)]
    ),
}
# Backward-compatible name for callers that launch the original nine cases.
ALL_SCENARIOS = SCENARIO_SUITES["legacy"]
LAUNCHER_SCHEMA = "personal-joint-ppo-multi"
LAUNCHER_SCHEMA_VERSION = 2
_COMPATIBLE_LAUNCHER_SCHEMA_VERSIONS = frozenset({1, LAUNCHER_SCHEMA_VERSION})
JOINT_ALGORITHM = "personal_joint_masked_ppo"
JOINT_CHECKPOINT_SCHEMA_VERSION = 3

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_GPU_ID = re.compile(r"^(?:[0-9]+|GPU-[A-Fa-f0-9-]+)$")
_CPU_SET = re.compile(r"^[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*$")
_TIMESTAMP = re.compile(r"^[0-9]{8}_[0-9]{6}_[0-9]{6}$")
_RESUME_FILENAME = re.compile(r"^(?:latest\.pt|round_[0-9]+\.pt)$")
_PROTECTED_TRAINER_OPTIONS = frozenset(
    {
        "--scenario",
        "--rounds",
        "--run-id",
        "--result-dir",
        "--model-dir",
        "--seed",
        "--blue-policy",
        "--device",
        "--resume",
        "--init-from",
        "--checkpoint-every",
        "--progress-every",
        "--rollout-episodes",
        "--render-mode",
        "--verbose",
        "--disable-log-color",
    }
)


@dataclass(frozen=True)
class ResumeCheckpoint:
    scenario: str
    path: Path
    sha256: str
    selection: str
    source_batch: str | None = None
    suite: str = "legacy"
    scenario_selector: str | None = None


@dataclass(frozen=True)
class JobPlan:
    scenario: str
    result_dir: Path
    model_dir: Path
    resume: ResumeCheckpoint | None = None
    suite: str = "legacy"
    scenario_selector: str | None = None
    scenario_file: Path | None = None

    @property
    def run_name(self) -> str:
        return self.result_dir.name

    @property
    def initialization(self) -> str:
        return "full_policy_state" if self.resume is not None else "fresh_random"

    @property
    def selector(self) -> str:
        return self.scenario_selector or scenario_selector(self.suite, self.scenario)

    @property
    def job_key(self) -> str:
        return self.scenario if self.suite == "legacy" else f"{self.suite}:{self.scenario}"

    @property
    def output_label(self) -> str:
        return self.scenario.lower() if self.suite == "legacy" else f"{self.suite}_{self.scenario.lower()}"


@dataclass(frozen=True)
class WorkerSlot:
    index: int
    device: str
    numa_node: int | None = None
    cpu_set: str | None = None

    @property
    def trainer_device(self) -> str:
        return "cpu" if self.device == "cpu" else "cuda:0"


@dataclass
class RunningJob:
    plan: JobPlan
    slot: WorkerSlot
    process: subprocess.Popen[Any]
    log_stream: Any
    started_monotonic: float
    stop_deadline: float | None = None
    kill_sent: bool = False


def scenario_selector(suite: str, scenario: str) -> str:
    """Return the unambiguous path selector understood by the single-case trainer."""

    if suite not in SCENARIO_SUITES:
        raise ValueError(f"Unknown scenario suite: {suite}")
    case_id = str(scenario).upper()
    if case_id not in SCENARIO_SUITES[suite]:
        raise ValueError(f"Unknown scenario for suite {suite}: {case_id}")
    if suite == "legacy":
        return case_id
    difficulty = {"E": "easy", "M": "medium", "H": "hard"}[case_id[0]]
    return f"{suite}/{difficulty}/{case_id}"


def scenario_file_path(repository_root: Path, suite: str, scenario: str) -> Path:
    selector = scenario_selector(suite, scenario)
    if suite == "legacy":
        difficulty = {"E": "easy", "M": "medium", "H": "hard"}[scenario[0].upper()]
        relative = Path(difficulty) / scenario.upper()
    else:
        relative = Path(selector)
    return (
        repository_root.expanduser().resolve()
        / "scenarios"
        / "cases"
        / relative
        / "scenario.json"
    )


def _positive(parser: argparse.ArgumentParser, option: str, value: int | float) -> None:
    if (isinstance(value, float) and not math.isfinite(value)) or value <= 0:
        parser.error(f"{option} must be positive and finite")


def parse_devices(value: str, *, allow_gpu_zero: bool, allow_gpu_sharing: bool) -> tuple[str, ...]:
    """Parse explicit worker device slots.

    A repeated GPU means two concurrent trainers may use it, so repetition is
    rejected unless the caller acknowledges sharing.  Repeated ``cpu`` slots
    are harmless and represent independent CPU worker slots.
    """

    tokens = tuple(item.strip() for item in value.split(",") if item.strip())
    if not tokens:
        raise ValueError("--gpu-ids must contain at least one explicit GPU ID or cpu")
    normalized: list[str] = []
    gpu_kinds: set[str] = set()
    for token in tokens:
        if token.lower() == "cpu":
            normalized.append("cpu")
            continue
        if not _GPU_ID.fullmatch(token):
            raise ValueError("GPU IDs must be numeric indices, NVIDIA GPU-... UUIDs, or cpu")
        value_normalized = str(int(token)) if token.isdigit() else token
        if value_normalized == "0" and not allow_gpu_zero:
            raise ValueError(
                "GPU 0 is protected because the current E01 trainer uses it; "
                "omit 0 or pass --allow-gpu-zero explicitly"
            )
        gpu_kinds.add("index" if value_normalized.isdigit() else "uuid")
        normalized.append(value_normalized)
    if len(gpu_kinds) > 1:
        raise ValueError("Do not mix numeric GPU indices and GPU UUIDs in one launch")
    gpu_values = [value.lower() for value in normalized if value != "cpu"]
    if len(set(gpu_values)) != len(gpu_values) and not allow_gpu_sharing:
        raise ValueError("Repeated GPU slots require --allow-gpu-sharing")
    return tuple(normalized)


def parse_numa_nodes(value: str | None, slot_count: int) -> tuple[int | None, ...]:
    if value is None:
        return (None,) * slot_count
    tokens = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not tokens:
        raise ValueError("--numa-nodes cannot be empty")
    if len(tokens) == 1 and slot_count > 1:
        tokens = tokens * slot_count
    if len(tokens) != slot_count:
        raise ValueError("--numa-nodes must contain one value per device slot (or one broadcast value)")
    nodes: list[int | None] = []
    for token in tokens:
        if token in {"none", "auto"}:
            nodes.append(None)
        elif token.isdigit():
            nodes.append(int(token))
        else:
            raise ValueError("NUMA nodes must be non-negative integers, none, or auto")
    return tuple(nodes)


def _expand_cpu_set(value: str) -> set[int]:
    if not _CPU_SET.fullmatch(value):
        raise ValueError(f"Invalid CPU set: {value}")
    result: set[int] = set()
    for item in value.split(","):
        if "-" in item:
            start_text, stop_text = item.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            if stop < start:
                raise ValueError(f"Invalid descending CPU range: {item}")
            result.update(range(start, stop + 1))
        else:
            result.add(int(item))
    return result


def parse_cpu_sets(
    values: Sequence[str] | None,
    slot_count: int,
    *,
    allow_overlap: bool,
) -> tuple[str | None, ...]:
    if values is None:
        return (None,) * slot_count
    if len(values) != slot_count:
        raise ValueError("--cpu-sets must contain one whitespace-separated value per device slot")
    parsed: list[str | None] = []
    occupied: set[int] = set()
    for raw in values:
        value = raw.strip().lower()
        if value in {"none", "auto"}:
            parsed.append(None)
            continue
        cpus = _expand_cpu_set(value)
        overlap = occupied & cpus
        if overlap and not allow_overlap:
            rendered = ",".join(str(cpu) for cpu in sorted(overlap))
            raise ValueError(f"CPU sets overlap on {rendered}; pass --allow-cpu-overlap to acknowledge")
        occupied.update(cpus)
        parsed.append(value)
    return tuple(parsed)


def _validate_extra_trainer_args(parser: argparse.ArgumentParser, values: Sequence[str]) -> tuple[str, ...]:
    extras = tuple(values)
    for token in extras:
        option = token.split("=", 1)[0]
        if option in _PROTECTED_TRAINER_OPTIONS:
            parser.error(
                f"{option} is managed by the multi-scenario launcher and cannot appear in --trainer-args"
            )
        if option.startswith("--") and any(
            protected.startswith(option) for protected in _PROTECTED_TRAINER_OPTIONS
        ):
            parser.error(
                f"{option} abbreviates an option managed by the multi-scenario launcher "
                "and cannot appear in --trainer-args"
            )
    return extras


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Run independent joint-PPO jobs with explicit GPU/CPU slots. "
            "No device is selected implicitly, and numeric GPU 0 is protected by default."
        )
    )
    parser.add_argument("--rounds", type=int, default=100, help="Additional rounds per scenario.")
    parser.add_argument(
        "--suite",
        choices=tuple(SCENARIO_SUITES),
        default="legacy",
        help=(
            "Scenario suite. Omitting --scenarios selects every case in the suite; "
            "the default legacy suite preserves the original nine-case behavior."
        ),
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        metavar="CASE",
        help="Optional subset of case IDs available in the selected suite.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--blue-policy", default="b0_fixed_ratio_random")
    parser.add_argument("--rollout-episodes", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--gpu-ids",
        required=True,
        help=(
            "Explicit concurrent worker slots, e.g. 1,2,3 or cpu. For nine jobs on seven "
            "free GPUs use 1,2,3,4,5,6,7,1,2 with --allow-gpu-sharing."
        ),
    )
    parser.add_argument(
        "--allow-gpu-zero",
        action="store_true",
        help="Acknowledge that a new worker may contend with the existing GPU-0 training job.",
    )
    parser.add_argument(
        "--allow-gpu-sharing",
        action="store_true",
        help="Allow a GPU ID to appear in more than one concurrent worker slot.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum active jobs; defaults to the number of explicit device slots.",
    )
    parser.add_argument(
        "--numa-nodes",
        default=None,
        help="Comma-separated NUMA node per device slot, or one broadcast node.",
    )
    parser.add_argument(
        "--cpu-sets",
        nargs="+",
        default=None,
        metavar="CPUSET",
        help="One CPU list per slot, e.g. 0-7 8-15; quote comma-containing lists.",
    )
    parser.add_argument("--allow-cpu-overlap", action="store_true")
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--python", default=DEFAULT_PYTHON, help="Trainer Python executable.")
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--numactl", default="numactl")
    parser.add_argument(
        "--resume-batch",
        type=Path,
        default=None,
        help="Prior joint launcher directory/batch ID; each selected case uses its latest.pt.",
    )
    parser.add_argument(
        "--resume-from",
        action="append",
        default=[],
        metavar="CASE=CHECKPOINT",
        help="Explicit per-case safe checkpoint; repeat for multiple cases.",
    )
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--stop-timeout", type=float, default=3600.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--verbose-workers", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the complete plan without creating directories or starting workers.",
    )
    parser.add_argument(
        "--trainer-args",
        nargs=argparse.REMAINDER,
        default=(),
        help="Additional non-protected train_joint_ppo.py arguments; this option must be last.",
    )
    args = parser.parse_args(argv)

    for name in (
        "rounds",
        "rollout_episodes",
        "checkpoint_every",
        "progress_every",
        "threads_per_worker",
        "stop_timeout",
        "poll_interval",
    ):
        _positive(parser, f"--{name.replace('_', '-')}", getattr(args, name))
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 4294967295")
    available_scenarios = SCENARIO_SUITES[args.suite]
    scenarios = (
        available_scenarios
        if args.scenarios is None
        else tuple(str(value).upper() for value in args.scenarios)
    )
    invalid = [scenario for scenario in scenarios if scenario not in available_scenarios]
    if invalid:
        parser.error(f"Unknown scenarios for suite {args.suite}: " + ", ".join(invalid))
    if len(set(scenarios)) != len(scenarios):
        parser.error("--scenarios cannot contain duplicates")
    args.scenarios = scenarios
    if args.batch_id is not None and not _SAFE_ID.fullmatch(args.batch_id):
        parser.error("--batch-id must use only A-Z, a-z, 0-9, _, ., - and be at most 64 characters")
    if args.resume_batch is not None and args.resume_from:
        parser.error("--resume-batch and --resume-from cannot be combined")
    try:
        args.devices = parse_devices(
            args.gpu_ids,
            allow_gpu_zero=args.allow_gpu_zero,
            allow_gpu_sharing=args.allow_gpu_sharing,
        )
        args.numa_node_values = parse_numa_nodes(args.numa_nodes, len(args.devices))
        args.cpu_set_values = parse_cpu_sets(
            args.cpu_sets,
            len(args.devices),
            allow_overlap=args.allow_cpu_overlap,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.max_parallel is None:
        args.max_parallel = len(args.devices)
    _positive(parser, "--max-parallel", args.max_parallel)
    if args.max_parallel > len(args.devices):
        parser.error(
            "--max-parallel cannot exceed the explicit device-slot count; repeat GPU IDs "
            "with --allow-gpu-sharing when intentional"
        )
    args.trainer_args = _validate_extra_trainer_args(parser, args.trainer_args)
    args.repository_root = args.repository_root.expanduser().resolve()
    return args


def make_timestamp(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")


def make_batch_id(value: str | None, *, timestamp: str) -> str:
    if not _TIMESTAMP.fullmatch(timestamp):
        raise ValueError(f"Invalid timestamp: {timestamp}")
    result = value or f"joint_multi_{timestamp}"
    if not _SAFE_ID.fullmatch(result):
        raise ValueError(f"Invalid batch ID: {result}")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_file(value: str | Path, invocation_cwd: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (invocation_cwd / path).resolve()


def _validate_resume_path(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    if not _RESUME_FILENAME.fullmatch(path.name):
        raise ValueError(
            "Joint --resume accepts only latest.pt or a numbered round_*.pt checkpoint; "
            f"got {path.name}"
        )


def resolve_explicit_resumes(
    values: Sequence[str],
    scenarios: Sequence[str],
    invocation_cwd: Path,
    *,
    suite: str = "legacy",
) -> dict[str, ResumeCheckpoint]:
    selected = set(scenarios)
    result: dict[str, ResumeCheckpoint] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--resume-from must use CASE=CHECKPOINT")
        scenario_text, path_text = value.split("=", 1)
        scenario = scenario_text.strip().upper()
        if scenario not in selected:
            raise ValueError(f"Resume scenario is not selected: {scenario}")
        if scenario in result:
            raise ValueError(f"Duplicate resume checkpoint for {scenario}")
        path = _resolve_file(path_text.strip(), invocation_cwd)
        _validate_resume_path(path)
        result[scenario] = ResumeCheckpoint(
            scenario=scenario,
            path=path,
            sha256=_sha256_file(path),
            selection="explicit",
            suite=suite,
            scenario_selector=scenario_selector(suite, scenario),
        )
    return result


def _resolve_launcher_root(value: Path, invocation_cwd: Path) -> Path:
    expanded = value.expanduser()
    candidates = (
        expanded.resolve() if expanded.is_absolute() else (invocation_cwd / expanded).resolve(),
        (LAUNCHER_RUNS_ROOT / expanded).resolve(),
    )
    for candidate in candidates:
        if (candidate / "batch_plan.json").is_file():
            return candidate
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Cannot find prior joint batch_plan.json; tried: {rendered}")


def resolve_resume_batch(
    value: Path,
    scenarios: Sequence[str],
    invocation_cwd: Path,
    *,
    suite: str = "legacy",
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, ResumeCheckpoint]:
    root = _resolve_launcher_root(value, invocation_cwd)
    try:
        document = json.loads((root / "batch_plan.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid prior batch plan: {root / 'batch_plan.json'}") from error
    try:
        schema_version = int(document.get("schema_version", -1)) if isinstance(document, dict) else -1
    except (TypeError, ValueError, OverflowError):
        schema_version = -1
    if (
        not isinstance(document, dict)
        or document.get("schema") != LAUNCHER_SCHEMA
        or schema_version not in _COMPATIBLE_LAUNCHER_SCHEMA_VERSIONS
        or not isinstance(document.get("jobs"), list)
    ):
        raise ValueError("--resume-batch is not a compatible joint multi-scenario batch")
    document_suite = str(document.get("suite", "legacy")).lower()
    if document_suite != suite:
        raise ValueError(
            f"Prior batch suite is {document_suite}, but the selected suite is {suite}"
        )
    jobs: dict[str, Mapping[str, Any]] = {}
    for raw in document["jobs"]:
        if not isinstance(raw, dict):
            continue
        scenario = str(raw.get("scenario", "")).upper()
        raw_suite = str(raw.get("suite", document_suite)).lower()
        if scenario not in SCENARIO_SUITES.get(raw_suite, ()):
            continue
        expected_selector = scenario_selector(raw_suite, scenario)
        raw_selector = str(raw.get("scenario_selector", expected_selector))
        expected_path = scenario_file_path(repository_root, raw_suite, scenario).resolve()
        raw_path_text = str(raw.get("scenario_path", ""))
        path_matches = (
            schema_version == 1
            or (
                bool(raw_path_text)
                and Path(raw_path_text).expanduser().resolve() == expected_path
            )
        )
        if (
            raw_suite == suite
            and raw_selector == expected_selector
            and path_matches
        ):
            jobs[scenario] = raw
    result: dict[str, ResumeCheckpoint] = {}
    for scenario in scenarios:
        raw = jobs.get(scenario)
        if raw is None:
            raise ValueError(f"Prior batch has no plan for selected scenario {scenario}")
        model_dir = Path(str(raw.get("model_dir", ""))).expanduser().resolve()
        checkpoint = model_dir / "latest.pt"
        _validate_resume_path(checkpoint)
        result[scenario] = ResumeCheckpoint(
            scenario=scenario,
            path=checkpoint,
            sha256=_sha256_file(checkpoint),
            selection="resume_batch_latest",
            source_batch=str(document.get("batch_id") or root.name),
            suite=suite,
            scenario_selector=scenario_selector(suite, scenario),
        )
    return result


def build_job_plans(
    scenarios: Sequence[str],
    timestamp: str,
    resumes: Mapping[str, ResumeCheckpoint] | None = None,
    *,
    suite: str = "legacy",
    repository_root: Path = REPOSITORY_ROOT,
) -> list[JobPlan]:
    if not _TIMESTAMP.fullmatch(timestamp):
        raise ValueError("Output timestamp must use YYYYMMDD_HHMMSS_microseconds")
    resume_by_scenario = dict(resumes or {})
    unexpected = set(resume_by_scenario) - set(scenarios)
    if unexpected:
        raise ValueError("Resume checkpoints supplied for unselected scenarios: " + ", ".join(sorted(unexpected)))
    plans: list[JobPlan] = []
    for scenario in scenarios:
        selector = scenario_selector(suite, scenario)
        resume = resume_by_scenario.get(scenario)
        if resume is not None and (
            resume.scenario != scenario
            or resume.suite != suite
            or (resume.scenario_selector or selector) != selector
        ):
            raise ValueError(
                "Resume scenario mismatch: "
                f"plan={suite}/{scenario} checkpoint={resume.suite}/{resume.scenario}"
            )
        output_label = scenario.lower() if suite == "legacy" else f"{suite}_{scenario.lower()}"
        run_name = f"{output_label}_joint_ppo_{timestamp}"
        plans.append(
            JobPlan(
                scenario=scenario,
                result_dir=RESULTS_ROOT / run_name,
                model_dir=MODELS_ROOT / run_name,
                resume=resume,
                suite=suite,
                scenario_selector=selector,
                scenario_file=scenario_file_path(repository_root, suite, scenario),
            )
        )
    return plans


def build_worker_slots(args: argparse.Namespace) -> tuple[WorkerSlot, ...]:
    return tuple(
        WorkerSlot(index=index, device=device, numa_node=node, cpu_set=cpus)
        for index, (device, node, cpus) in enumerate(
            zip(args.devices, args.numa_node_values, args.cpu_set_values, strict=True)
        )
    )


def _launcher_root(batch_id: str) -> Path:
    if not _SAFE_ID.fullmatch(batch_id):
        raise ValueError(f"Invalid batch ID: {batch_id}")
    return validate_personal_output_path(LAUNCHER_RUNS_ROOT / batch_id)


def validate_outputs_available(batch_id: str, plans: Sequence[JobPlan]) -> Path:
    launcher_root = _launcher_root(batch_id)
    if launcher_root.exists():
        raise FileExistsError(f"Launcher run already exists: {launcher_root}")
    results_root = RESULTS_ROOT.resolve()
    models_root = MODELS_ROOT.resolve()
    seen: set[Path] = set()
    for plan in plans:
        expected = re.compile(
            rf"^{re.escape(plan.output_label)}_joint_ppo_[0-9]{{8}}_[0-9]{{6}}_[0-9]{{6}}$"
        )
        result_dir = validate_personal_output_path(plan.result_dir)
        model_dir = validate_personal_output_path(plan.model_dir)
        if result_dir.parent != results_root or model_dir.parent != models_root:
            raise ValueError("Scenario outputs must be direct children of results and models")
        if result_dir.name != model_dir.name or not expected.fullmatch(result_dir.name):
            raise ValueError(f"Invalid output name for {plan.scenario}: {result_dir.name}")
        if result_dir in seen or model_dir in seen:
            raise ValueError("Scenario output paths must be unique")
        seen.update((result_dir, model_dir))
        existing = [str(path) for path in (result_dir, model_dir) if path.exists()]
        if existing:
            raise FileExistsError("Training output already exists: " + ", ".join(existing))
        if plan.resume is not None:
            _validate_resume_path(plan.resume.path)
    return launcher_root


def _python_executable(value: str) -> str | None:
    expanded = Path(value).expanduser()
    if expanded.is_absolute() or "/" in value:
        return str(expanded.resolve()) if expanded.is_file() and os.access(expanded, os.X_OK) else None
    return shutil.which(value)


def preflight(args: argparse.Namespace, slots: Sequence[WorkerSlot]) -> None:
    executable = _python_executable(args.python)
    if executable is None:
        raise FileNotFoundError(f"Trainer Python is missing or not executable: {args.python}")
    args.python = executable
    if not TRAINER.is_file():
        raise FileNotFoundError(f"Joint trainer is missing: {TRAINER}")
    cases = args.repository_root / "scenarios" / "cases"
    if not cases.is_dir():
        raise FileNotFoundError(f"Repository has no scenarios/cases: {args.repository_root}")
    missing_scenarios = [
        scenario_file_path(args.repository_root, args.suite, scenario)
        for scenario in args.scenarios
        if not scenario_file_path(args.repository_root, args.suite, scenario).is_file()
    ]
    if missing_scenarios:
        raise FileNotFoundError(
            "Selected scenario files are missing: "
            + ", ".join(str(path) for path in missing_scenarios)
        )
    if any(slot.numa_node is not None or slot.cpu_set is not None for slot in slots):
        if shutil.which(args.numactl) is None:
            raise FileNotFoundError(f"NUMA/CPU binding requested but numactl was not found: {args.numactl}")
    requested_nodes = {slot.numa_node for slot in slots if slot.numa_node is not None}
    missing_nodes = sorted(
        node
        for node in requested_nodes
        if not Path(f"/sys/devices/system/node/node{node}").is_dir()
    )
    if missing_nodes:
        raise ValueError("Requested NUMA nodes are unavailable: " + ", ".join(map(str, missing_nodes)))
    if hasattr(os, "sched_getaffinity"):
        allowed_cpus = set(os.sched_getaffinity(0))
        for slot in slots:
            if slot.cpu_set is None:
                continue
            unavailable = sorted(_expand_cpu_set(slot.cpu_set) - allowed_cpus)
            if unavailable:
                raise ValueError(
                    f"Worker slot {slot.index} requests CPUs outside this process affinity: "
                    + ", ".join(map(str, unavailable))
                )
    gpu_devices = [slot.device for slot in slots if slot.device != "cpu"]
    if gpu_devices:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            raise FileNotFoundError("GPU workers requested but nvidia-smi was not found")
        completed = subprocess.run(
            [nvidia_smi, "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        index_to_uuid: dict[str, str] = {}
        for line in completed.stdout.splitlines():
            columns = [column.strip() for column in line.split(",", 1)]
            if len(columns) == 2 and columns[0].isdigit() and _GPU_ID.fullmatch(columns[1]):
                index_to_uuid[str(int(columns[0]))] = columns[1]
        uuid_to_index = {uuid.lower(): index for index, uuid in index_to_uuid.items()}
        missing = sorted(
            {
                device
                for device in gpu_devices
                if (
                    device not in index_to_uuid
                    if device.isdigit()
                    else device.lower() not in uuid_to_index
                )
            }
        )
        if missing:
            raise ValueError("Requested GPUs are unavailable: " + ", ".join(missing))
        selected_indices = {
            device if device.isdigit() else uuid_to_index[device.lower()]
            for device in gpu_devices
        }
        if "0" in selected_indices and not args.allow_gpu_zero:
            raise ValueError(
                "GPU 0 is protected because the current E01 trainer uses it; this also "
                "applies when GPU 0 is selected by UUID. Pass --allow-gpu-zero to acknowledge."
            )


def _binding_prefix(args: argparse.Namespace, slot: WorkerSlot) -> list[str]:
    if slot.numa_node is None and slot.cpu_set is None:
        return []
    command = [args.numactl]
    if slot.cpu_set is not None:
        command.append(f"--physcpubind={slot.cpu_set}")
    elif slot.numa_node is not None:
        # Exact CPU binding already implies a CPU placement; do not combine
        # numactl's two mutually competing CPU-placement policies.
        command.append(f"--cpunodebind={slot.numa_node}")
    if slot.numa_node is not None:
        command.append(f"--membind={slot.numa_node}")
    return command


def trainer_command(
    args: argparse.Namespace,
    plan: JobPlan,
    slot: WorkerSlot,
) -> list[str]:
    command = _binding_prefix(args, slot)
    command.extend(
        (
            args.python,
            str(TRAINER),
            "--scenario",
            plan.selector,
            "--rounds",
            str(args.rounds),
            "--run-id",
            plan.run_name,
            "--result-dir",
            str(plan.result_dir),
            "--model-dir",
            str(plan.model_dir),
            "--seed",
            str(args.seed),
            "--blue-policy",
            args.blue_policy,
            "--device",
            slot.trainer_device,
            "--rollout-episodes",
            str(args.rollout_episodes),
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--progress-every",
            str(args.progress_every),
            "--render-mode",
            "none",
            "--disable-log-color",
        )
    )
    if plan.resume is not None:
        command.extend(("--resume", str(plan.resume.path)))
    if args.verbose_workers:
        command.append("--verbose")
    command.extend(args.trainer_args)
    return command


def worker_environment(
    args: argparse.Namespace,
    plan: JobPlan,
    slot: WorkerSlot,
    launcher_root: Path,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "COMPETITION_REPO_ROOT": str(args.repository_root),
            "OMP_NUM_THREADS": str(args.threads_per_worker),
            "MKL_NUM_THREADS": str(args.threads_per_worker),
            "OPENBLAS_NUM_THREADS": str(args.threads_per_worker),
            "TORCHINDUCTOR_CACHE_DIR": str(
                launcher_root / "worker_cache" / plan.output_label / "inductor"
            ),
            "TMPDIR": str(launcher_root / "worker_cache" / plan.output_label / "tmp"),
            "CUDA_VISIBLE_DEVICES": "" if slot.device == "cpu" else slot.device,
        }
    )
    return environment


def _atomic_json(path: Path, value: Any) -> None:
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


def _resume_document(resume: ResumeCheckpoint | None) -> dict[str, Any] | None:
    return asdict(resume) | {"path": str(resume.path)} if resume is not None else None


def batch_plan_document(
    args: argparse.Namespace,
    plans: Sequence[JobPlan],
    slots: Sequence[WorkerSlot],
    launcher_root: Path,
    *,
    created_at: str,
) -> dict[str, Any]:
    return {
        "schema": LAUNCHER_SCHEMA,
        "schema_version": LAUNCHER_SCHEMA_VERSION,
        "batch_id": args.batch_id,
        "suite": args.suite,
        "created_at": created_at,
        "rounds_per_scenario": args.rounds,
        "max_parallel": args.max_parallel,
        "repository_root": str(args.repository_root),
        "python": args.python,
        "launcher_run_dir": str(launcher_root),
        "worker_slots": [asdict(slot) for slot in slots],
        "jobs": [
            {
                "scenario": plan.scenario,
                "suite": plan.suite,
                "job_key": plan.job_key,
                "scenario_selector": plan.selector,
                "scenario_path": str(plan.scenario_file) if plan.scenario_file else None,
                "run_name": plan.run_name,
                "initialization": plan.initialization,
                "result_dir": str(plan.result_dir),
                "model_dir": str(plan.model_dir),
                "resume": _resume_document(plan.resume),
            }
            for plan in plans
        ],
    }


def _empty_states(plans: Sequence[JobPlan]) -> dict[str, dict[str, Any]]:
    return {
        plan.job_key: {
            "scenario": plan.scenario,
            "suite": plan.suite,
            "scenario_selector": plan.selector,
            "status": "pending",
            "initialization": plan.initialization,
            "resume": str(plan.resume.path) if plan.resume else None,
            "result_dir": str(plan.result_dir),
            "model_dir": str(plan.model_dir),
            "slot": None,
            "host_device": None,
            "trainer_device": None,
            "numa_node": None,
            "cpu_set": None,
            "pid": None,
            "started_at": None,
            "finished_at": None,
            "elapsed_seconds": None,
            "return_code": None,
            "error": None,
        }
        for plan in plans
    }


def _status_document(
    args: argparse.Namespace,
    states: Mapping[str, Mapping[str, Any]],
    *,
    created_at: str,
    interrupted: bool,
) -> dict[str, Any]:
    values = [str(state["status"]) for state in states.values()]
    if interrupted:
        overall = "interrupted"
    elif any(value in {"pending", "starting", "running", "stopping"} for value in values):
        overall = "running"
    elif values and all(value == "succeeded" for value in values):
        overall = "succeeded"
    else:
        overall = "failed"
    return {
        "schema": LAUNCHER_SCHEMA,
        "schema_version": LAUNCHER_SCHEMA_VERSION,
        "batch_id": args.batch_id,
        "suite": args.suite,
        "status": overall,
        "created_at": created_at,
        "updated_at": datetime.now().astimezone().isoformat(),
        "interrupted": interrupted,
        "counts": {value: values.count(value) for value in sorted(set(values))},
        "jobs": dict(states),
    }


def _read_rounds(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except (OSError, csv.Error, UnicodeError):
        return []


def _validate_checkpoint_manifest(plan: JobPlan) -> list[str]:
    """Verify that successful trainer checkpoints still match its signed-off manifest."""

    path = plan.model_dir / "checkpoint_validation.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["checkpoint_validation.json is invalid"]
    if not isinstance(document, dict):
        return ["checkpoint_validation.json is invalid"]
    errors: list[str] = []
    try:
        schema_version = int(document.get("schema_version", -1))
    except (TypeError, ValueError, OverflowError):
        schema_version = -1
    if (
        document.get("algorithm") != JOINT_ALGORITHM
        or schema_version != JOINT_CHECKPOINT_SCHEMA_VERSION
    ):
        errors.append("checkpoint validation algorithm/schema is incompatible")
    records = document.get("checkpoints")
    if not isinstance(records, dict):
        return errors + ["checkpoint validation records are missing"]
    expectations = {
        "best": (plan.model_dir / "best.pt", "best_behavior", False),
        "latest": (plan.model_dir / "latest.pt", "latest", True),
    }
    for name, (checkpoint, expected_role, expected_resume_safe) in expectations.items():
        record = records.get(name)
        if not isinstance(record, dict):
            errors.append(f"checkpoint validation record is missing: {name}")
            continue
        try:
            recorded_size = int(record.get("size_bytes", -1))
        except (TypeError, ValueError, OverflowError):
            recorded_size = -1
        actual_size = checkpoint.stat().st_size if checkpoint.is_file() else -1
        recorded_digest = str(record.get("sha256", "")).lower()
        actual_digest = _sha256_file(checkpoint) if actual_size > 0 else ""
        if record.get("file") != checkpoint.name:
            errors.append(f"checkpoint validation filename mismatch: {name}")
        if recorded_size <= 0 or recorded_size != actual_size:
            errors.append(f"checkpoint validation size mismatch: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", recorded_digest) or recorded_digest != actual_digest:
            errors.append(f"checkpoint validation hash mismatch: {name}")
        if record.get("checkpoint_role") != expected_role:
            errors.append(f"checkpoint validation role mismatch: {name}")
        if record.get("resume_safe") is not expected_resume_safe:
            errors.append(f"checkpoint validation resume flag mismatch: {name}")
    return errors


def validate_job_outputs(plan: JobPlan, expected_rounds: int) -> list[str]:
    errors: list[str] = []
    required = (
        plan.result_dir / "run_config.json",
        plan.result_dir / "status.json",
        plan.result_dir / "rounds.csv",
        plan.model_dir / "latest.pt",
        plan.model_dir / "best.pt",
        plan.model_dir / "model_metadata.json",
        plan.model_dir / "checkpoint_validation.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        errors.append("missing required outputs: " + ", ".join(missing))
    markers = [
        str(path)
        for path in (plan.result_dir / "failure.json", plan.result_dir / "interruption.json")
        if path.exists()
    ]
    if markers:
        errors.append("training left failure/interruption markers: " + ", ".join(markers))
    if (plan.model_dir / "checkpoint_validation.json").is_file():
        errors.extend(_validate_checkpoint_manifest(plan))
    rounds = _read_rounds(plan.result_dir / "rounds.csv")
    try:
        indices = [int(row["round"]) for row in rounds]
    except (KeyError, TypeError, ValueError):
        indices = []
    if indices != list(range(1, expected_rounds + 1)):
        errors.append(f"rounds.csv does not contain exactly rounds 1..{expected_rounds}")
    try:
        scores = [float(row["score"]) for row in rounds]
    except (KeyError, TypeError, ValueError):
        scores = []
    if len(scores) != expected_rounds or any(
        not math.isfinite(score) or not 0.0 <= score <= 100.0 for score in scores
    ):
        errors.append("rounds.csv contains an invalid official score")
    try:
        run_config = json.loads((plan.result_dir / "run_config.json").read_text(encoding="utf-8"))
        path_text = str(run_config.get("scenario_path", "")) if isinstance(run_config, dict) else ""
        recorded_path = Path(path_text).expanduser().resolve() if path_text else None
        if (
            not isinstance(run_config, dict)
            or str(run_config.get("scenario", "")).upper() != plan.scenario
            or plan.scenario_file is None
            or recorded_path != plan.scenario_file.expanduser().resolve()
            or int(run_config.get("rounds", -1)) != expected_rounds
        ):
            errors.append("run_config.json scenario path/ID or round count does not match the job")
    except (OSError, ValueError, TypeError, OverflowError, json.JSONDecodeError):
        if (plan.result_dir / "run_config.json").is_file():
            errors.append("run_config.json is invalid")
    try:
        status = json.loads((plan.result_dir / "status.json").read_text(encoding="utf-8"))
        if (
            not isinstance(status, dict)
            or status.get("status") != "complete"
            or int(status.get("completed_rounds", -1)) != expected_rounds
        ):
            errors.append("trainer status.json is not complete for the requested rounds")
    except (OSError, ValueError, TypeError, OverflowError, json.JSONDecodeError):
        if (plan.result_dir / "status.json").is_file():
            errors.append("status.json is invalid")
    try:
        metadata = json.loads((plan.model_dir / "model_metadata.json").read_text(encoding="utf-8"))
        if (
            not isinstance(metadata, dict)
            or metadata.get("algorithm") != JOINT_ALGORITHM
            or int(metadata.get("schema_version", -1)) != JOINT_CHECKPOINT_SCHEMA_VERSION
            or int(metadata.get("latest_round", -1)) != expected_rounds
            or metadata.get("status") != "complete"
        ):
            errors.append("model_metadata.json is not a complete compatible joint-PPO run")
    except (OSError, ValueError, TypeError, OverflowError, json.JSONDecodeError):
        if (plan.model_dir / "model_metadata.json").is_file():
            errors.append("model_metadata.json is invalid")
    return errors


def _summary_rows(
    plans: Sequence[JobPlan], states: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plan in plans:
        rounds = _read_rounds(plan.result_dir / "rounds.csv")
        scores: list[float] = []
        for row in rounds:
            try:
                score = float(row["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(score) and 0.0 <= score <= 100.0:
                scores.append(score)
        state = states[plan.job_key]
        rows.append(
            {
                "scenario": plan.scenario,
                "suite": plan.suite,
                "scenario_selector": plan.selector,
                "status": state["status"],
                "host_device": state["host_device"],
                "rounds_completed": len(rounds),
                "best_score": max(scores) if scores else None,
                "final_score": scores[-1] if scores else None,
                "elapsed_seconds": state["elapsed_seconds"],
                "result_dir": str(plan.result_dir),
                "model_dir": str(plan.model_dir),
                "resume": str(plan.resume.path) if plan.resume else None,
            }
        )
    return rows


def _write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_json(path / "batch_summary.json", list(rows))
    columns = (
        "scenario",
        "suite",
        "scenario_selector",
        "status",
        "host_device",
        "rounds_completed",
        "best_score",
        "final_score",
        "elapsed_seconds",
        "result_dir",
        "model_dir",
        "resume",
    )
    temporary = path / ".batch_summary.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path / "batch_summary.csv")


def _worker_status_details(plan: JobPlan) -> dict[str, Any]:
    try:
        status = json.loads((plan.result_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(status, dict):
        return {}
    return {
        key: status[key]
        for key in (
            "completed_rounds",
            "active_round",
            "update_count",
            "recovery_checkpoint",
            "recovery_mode",
            "policy_state_consistent",
        )
        if key in status
    }


def _send_group_signal(running: RunningJob, signum: int) -> None:
    if running.process.poll() is not None:
        return
    try:
        os.killpg(running.process.pid, signum)
    except ProcessLookupError:
        return


def run_batch(
    args: argparse.Namespace,
    plans: Sequence[JobPlan],
    slots: Sequence[WorkerSlot],
    launcher_root: Path,
) -> int:
    created_at = datetime.now().astimezone().isoformat()
    launcher_root.mkdir(parents=True, exist_ok=False)
    (launcher_root / "logs").mkdir()
    for plan in plans:
        (launcher_root / "worker_cache" / plan.output_label / "tmp").mkdir(parents=True)
        (launcher_root / "worker_cache" / plan.output_label / "inductor").mkdir()
    _atomic_json(
        launcher_root / "batch_plan.json",
        batch_plan_document(args, plans, slots, launcher_root, created_at=created_at),
    )

    states = _empty_states(plans)
    pending = deque(plans)
    free_slots = deque(slots)
    active: dict[str, RunningJob] = {}
    requested_signal: int | None = None
    abort_reason: str | None = None

    def persist() -> None:
        _atomic_json(
            launcher_root / "batch_status.json",
            _status_document(
                args,
                states,
                created_at=created_at,
                interrupted=requested_signal is not None,
            ),
        )
        _write_summary(launcher_root, _summary_rows(plans, states))

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal requested_signal
        requested_signal = int(signum)

    handled = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled.append(signal.SIGHUP)
    previous_handlers = {signum: signal.getsignal(signum) for signum in handled}
    for signum in handled:
        signal.signal(signum, request_stop)

    persist()
    try:
        while pending or active:
            stopping = requested_signal is not None or abort_reason is not None
            if stopping:
                while pending:
                    plan = pending.popleft()
                    states[plan.job_key].update(
                        {
                            "status": "cancelled",
                            "finished_at": datetime.now().astimezone().isoformat(),
                            "error": abort_reason or f"launcher received signal {requested_signal}",
                        }
                    )
                now = time.monotonic()
                for running in active.values():
                    if running.stop_deadline is None:
                        _send_group_signal(running, signal.SIGTERM)
                        running.stop_deadline = now + args.stop_timeout
                        states[running.plan.job_key]["status"] = "stopping"
                    elif now >= running.stop_deadline and not running.kill_sent:
                        _send_group_signal(running, signal.SIGKILL)
                        running.kill_sent = True

            while (
                requested_signal is None
                and abort_reason is None
                and pending
                and free_slots
                and len(active) < args.max_parallel
            ):
                plan = pending.popleft()
                slot = free_slots.popleft()
                state = states[plan.job_key]
                state.update(
                    {
                        "status": "starting",
                        "slot": slot.index,
                        "host_device": slot.device,
                        "trainer_device": slot.trainer_device,
                        "numa_node": slot.numa_node,
                        "cpu_set": slot.cpu_set,
                        "started_at": datetime.now().astimezone().isoformat(),
                    }
                )
                log_stream = None
                try:
                    if plan.resume is not None and _sha256_file(plan.resume.path) != plan.resume.sha256:
                        raise ValueError(f"Resume checkpoint changed before launch: {plan.resume.path}")
                    command = trainer_command(args, plan, slot)
                    environment = worker_environment(args, plan, slot, launcher_root)
                    log_stream = (launcher_root / "logs" / f"{plan.output_label}.log").open(
                        "x", encoding="utf-8", buffering=1
                    )
                    log_stream.write(f"COMMAND {shlex.join(command)}\n")
                    log_stream.write(
                        "PLACEMENT "
                        f"slot={slot.index} host_device={slot.device} "
                        f"numa_node={slot.numa_node} cpu_set={slot.cpu_set}\n"
                    )
                    process = subprocess.Popen(
                        command,
                        cwd=str(PERSONAL_ROOT.parent),
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=log_stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        close_fds=True,
                    )
                    active[plan.job_key] = RunningJob(
                        plan=plan,
                        slot=slot,
                        process=process,
                        log_stream=log_stream,
                        started_monotonic=time.monotonic(),
                    )
                    state.update({"status": "running", "pid": process.pid})
                    print(
                        f"START {plan.job_key} pid={process.pid} slot={slot.index} "
                        f"device={slot.device}",
                        flush=True,
                    )
                except Exception as error:
                    if log_stream is not None:
                        log_stream.close()
                    free_slots.append(slot)
                    state.update(
                        {
                            "status": "failed",
                            "finished_at": datetime.now().astimezone().isoformat(),
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    if args.fail_fast:
                        abort_reason = f"fail-fast after {plan.job_key} launch failure"
                persist()

            completed: list[tuple[str, int]] = []
            for job_key, running in active.items():
                return_code = running.process.poll()
                if return_code is not None:
                    completed.append((job_key, int(return_code)))
            for job_key, return_code in completed:
                running = active.pop(job_key)
                running.log_stream.close()
                free_slots.append(running.slot)
                elapsed = time.monotonic() - running.started_monotonic
                errors = validate_job_outputs(running.plan, args.rounds) if return_code == 0 else []
                if return_code == 0 and not errors:
                    status = "succeeded"
                    error_text = None
                elif return_code == 130 or requested_signal is not None:
                    status = "interrupted"
                    error_text = abort_reason or f"worker exited after signal (code {return_code})"
                else:
                    status = "failed"
                    error_text = "; ".join(errors) if errors else f"worker exited with code {return_code}"
                states[job_key].update(
                    {
                        "status": status,
                        "finished_at": datetime.now().astimezone().isoformat(),
                        "elapsed_seconds": round(elapsed, 3),
                        "return_code": return_code,
                        "error": error_text,
                        **_worker_status_details(running.plan),
                    }
                )
                print(
                    f"DONE {job_key} status={status} code={return_code} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                if status == "failed" and args.fail_fast and abort_reason is None:
                    abort_reason = f"fail-fast after {job_key} failure"
                persist()

            if pending or active:
                time.sleep(args.poll_interval)
    except BaseException as error:
        abort_reason = f"launcher error: {type(error).__name__}: {error}"
        for running in active.values():
            _send_group_signal(running, signal.SIGTERM)
        deadline = time.monotonic() + min(args.stop_timeout, 30.0)
        for running in active.values():
            remaining = max(0.0, deadline - time.monotonic())
            try:
                running.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                _send_group_signal(running, signal.SIGKILL)
                running.process.wait()
            running.log_stream.close()
            states[running.plan.job_key].update(
                {
                    "status": "failed",
                    "finished_at": datetime.now().astimezone().isoformat(),
                    "return_code": running.process.poll(),
                    "error": abort_reason,
                    **_worker_status_details(running.plan),
                }
            )
        while pending:
            plan = pending.popleft()
            states[plan.job_key].update({"status": "cancelled", "error": abort_reason})
        raise
    finally:
        for signum, old_handler in previous_handlers.items():
            signal.signal(signum, old_handler)
        persist()

    if requested_signal is not None:
        return 130
    return 0 if all(state["status"] == "succeeded" for state in states.values()) else 1


def dry_run_document(
    args: argparse.Namespace,
    plans: Sequence[JobPlan],
    slots: Sequence[WorkerSlot],
    launcher_root: Path,
) -> dict[str, Any]:
    jobs = []
    for index, plan in enumerate(plans):
        slot = slots[index % len(slots)]
        environment = worker_environment(args, plan, slot, launcher_root, base={})
        jobs.append(
            {
                "scenario": plan.scenario,
                "suite": plan.suite,
                "job_key": plan.job_key,
                "scenario_selector": plan.selector,
                "scenario_path": str(plan.scenario_file) if plan.scenario_file else None,
                "slot_preview": asdict(slot),
                "initialization": plan.initialization,
                "result_dir": str(plan.result_dir),
                "model_dir": str(plan.model_dir),
                "resume": _resume_document(plan.resume),
                "environment": environment,
                "command": trainer_command(args, plan, slot),
            }
        )
    return {
        "dry_run": True,
        "schema": LAUNCHER_SCHEMA,
        "schema_version": LAUNCHER_SCHEMA_VERSION,
        "batch_id": args.batch_id,
        "suite": args.suite,
        "rounds_per_scenario": args.rounds,
        "max_parallel": args.max_parallel,
        "launcher_run_dir": str(launcher_root),
        "worker_slots": [asdict(slot) for slot in slots],
        "jobs": jobs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    invocation_cwd = Path.cwd()
    args = parse_args(argv)
    if args.resume_batch is not None:
        resumes = resolve_resume_batch(
            args.resume_batch,
            args.scenarios,
            invocation_cwd,
            suite=args.suite,
            repository_root=args.repository_root,
        )
    else:
        resumes = resolve_explicit_resumes(
            args.resume_from,
            args.scenarios,
            invocation_cwd,
            suite=args.suite,
        )
    timestamp = make_timestamp()
    args.batch_id = make_batch_id(args.batch_id, timestamp=timestamp)
    plans = build_job_plans(
        args.scenarios,
        timestamp,
        resumes,
        suite=args.suite,
        repository_root=args.repository_root,
    )
    slots = build_worker_slots(args)
    launcher_root = validate_outputs_available(args.batch_id, plans)
    if args.dry_run:
        print(
            json.dumps(
                dry_run_document(args, plans, slots, launcher_root),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    preflight(args, slots)
    return run_batch(args, plans, slots, launcher_root)


if __name__ == "__main__":
    raise SystemExit(main())
