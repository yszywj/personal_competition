"""Locate the project in both the source tree and the packaged Docker image."""

from __future__ import annotations

import sys
from pathlib import Path


PERSONAL_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PERSONAL_ROOT.parent

# In the source tree the Python core lives in ``<repo>/core``.  The current
# competition image copies the contents of that directory directly to /app.
_source_core = REPOSITORY_ROOT / "core"
CORE_ROOT = _source_core if (_source_core / "envengine").is_dir() else REPOSITORY_ROOT


def install_project_paths() -> None:
    """Make the existing project packages importable without installing them."""

    for path in (REPOSITORY_ROOT, CORE_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

