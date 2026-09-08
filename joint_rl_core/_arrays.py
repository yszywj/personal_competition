"""Small helpers for arrays stored in immutable public records."""

from __future__ import annotations

from typing import Any

import numpy as np


def immutable_array(value: Any, *, dtype: np.dtype[Any] | type) -> np.ndarray:
    """Return a C-contiguous array backed by immutable bytes.

    A normal NumPy owner array can be made writable again with ``setflags``.
    Building the public view from a ``bytes`` object prevents callers from
    mutating records that are used as rollout snapshots.
    """

    contiguous = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
