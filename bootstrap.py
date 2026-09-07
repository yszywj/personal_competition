"""Locate an external project without writing into its source checkout."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


PERSONAL_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProjectLayout:
    repository_root: Path
    core_root: Path

    @property
    def scenarios_root(self) -> Path:
        return self.repository_root / "scenarios"


def discover_project_layout(
    personal_root: Path,
    repository_override: str | Path | None = None,
) -> ProjectLayout:
    """Support sibling checkouts, the old nested layout and flat Docker images.

    An explicit override is authoritative: a typo must not silently select an
    older source tree baked into a container image.
    """
    personal_root = personal_root.resolve()
    candidates = (
        [Path(repository_override).expanduser()]
        if repository_override is not None
        else [personal_root.parent / "competition-platform-env", personal_root.parent]
    )
    for candidate in candidates:
        repository = candidate.resolve()
        source_core = repository / "core"
        core = source_core if (source_core / "envengine").is_dir() else repository
        if (
            (core / "envengine").is_dir()
            and (repository / "policies").is_dir()
            and (repository / "scenarios" / "cases").is_dir()
        ):
            return ProjectLayout(repository, core)
    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Cannot locate the competition source tree. Set COMPETITION_REPO_ROOT "
        "to a checkout containing core/envengine, policies and scenarios/cases. "
        f"Tried: {attempted}"
    )


LAYOUT = discover_project_layout(PERSONAL_ROOT, os.environ.get("COMPETITION_REPO_ROOT"))
REPOSITORY_ROOT = LAYOUT.repository_root
CORE_ROOT = LAYOUT.core_root
SCENARIOS_ROOT = LAYOUT.scenarios_root


def install_project_paths() -> None:
    """Make the existing project packages importable without installing them."""

    # Do not create __pycache__ in the upstream checkout, including when this
    # package is imported from a host-side tool rather than its CLI entry point.
    sys.dont_write_bytecode = True
    preferred = list(dict.fromkeys(str(path) for path in (
        CORE_ROOT, REPOSITORY_ROOT, PERSONAL_ROOT.parent,
    )))
    sys.path[:] = preferred + [entry for entry in sys.path if entry not in preferred]


def validate_personal_output_path(path: Path) -> Path:
    """Validate before any mkdir; symlinks cannot redirect output upstream."""
    resolved = path.expanduser().resolve()
    personal = PERSONAL_ROOT.resolve()
    if personal not in resolved.parents:
        raise ValueError(f"Output must be below personal_train: {resolved}")
    if resolved == REPOSITORY_ROOT or REPOSITORY_ROOT in resolved.parents:
        raise ValueError(f"Output must be outside the upstream checkout: {resolved}")
    return resolved


def prepare_runtime_directory(
    runtime_root: Path,
    *,
    core_root: Path = CORE_ROOT,
    native_results: Path | None = None,
) -> Path:
    """Expose relative read-only resources from a private working directory.

    Source directories are never created or edited. ``Results`` is separate
    from the resources; Docker supplies it as an ephemeral tmpfs.
    """
    runtime_root = validate_personal_output_path(runtime_root)
    core_root = core_root.resolve()
    if any(
        runtime_root == source or source in runtime_root.parents
        for source in (core_root, REPOSITORY_ROOT)
    ):
        raise ValueError("The runtime directory must be outside the upstream checkout")
    if native_results is not None:
        native_results = native_results.resolve()
        if not native_results.is_dir():
            raise FileNotFoundError(f"Native results mount is missing: {native_results}")
        if any(
            native_results == source or source in native_results.parents
            for source in (core_root, REPOSITORY_ROOT)
        ):
            raise ValueError("Native results must be outside the upstream checkout")
    runtime_root.mkdir(parents=True, exist_ok=False)
    for resource in core_root.iterdir():
        if resource.name in {"Results", "__pycache__", ".git"}:
            continue
        (runtime_root / resource.name).symlink_to(
            resource, target_is_directory=resource.is_dir()
        )
    if native_results is None:
        (runtime_root / "Results").mkdir()
    else:
        (runtime_root / "Results").symlink_to(native_results, target_is_directory=True)
    return runtime_root
