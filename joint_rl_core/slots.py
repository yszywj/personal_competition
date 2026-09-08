"""Stable fixed-capacity slot assignment for opaque external game IDs."""

from __future__ import annotations

from typing import Iterable, TypeAlias

import numpy as np


ExternalId: TypeAlias = int | str


def _stable_key(value: ExternalId) -> tuple[int, int | str]:
    return (0, value) if isinstance(value, int) else (1, value)


class StableSlotRegistry:
    """Map opaque IDs to stable neural-network slots for one episode.

    New IDs in one frame are sorted before allocation, so dictionary or sensor
    iteration order cannot change the assignment.  Allocated slots are never
    reused until ``reset``.  Overflow IDs are retained to avoid counting the
    same dropped object repeatedly.
    """

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("slot capacity must be a positive integer")
        self.capacity = int(capacity)
        self.reset()

    def reset(self) -> None:
        self._slot_by_external: dict[ExternalId, int] = {}
        self._external_by_slot: list[ExternalId | None] = [None] * self.capacity
        self._overflow_ids: set[ExternalId] = set()

    def __len__(self) -> int:
        return len(self._slot_by_external)

    @property
    def overflow_count(self) -> int:
        return len(self._overflow_ids)

    @property
    def valid_mask(self) -> np.ndarray:
        mask = np.asarray(
            [external_id is not None for external_id in self._external_by_slot],
            dtype=np.bool_,
        )
        mask.setflags(write=False)
        return mask

    @property
    def external_by_slot(self) -> tuple[ExternalId | None, ...]:
        return tuple(self._external_by_slot)

    def slot_for(self, external_id: ExternalId) -> int | None:
        self._validate_id(external_id)
        return self._slot_by_external.get(external_id)

    def external_id_for(self, slot: int) -> ExternalId | None:
        if slot < 0 or slot >= self.capacity:
            raise IndexError("slot is out of range")
        return self._external_by_slot[slot]

    def register_batch(self, external_ids: Iterable[ExternalId]) -> dict[ExternalId, int]:
        """Register unseen IDs deterministically and return assigned entries."""

        unique: set[ExternalId] = set()
        for external_id in external_ids:
            self._validate_id(external_id)
            unique.add(external_id)
        unseen = sorted(
            (
                external_id
                for external_id in unique
                if external_id not in self._slot_by_external
                and external_id not in self._overflow_ids
            ),
            key=_stable_key,
        )
        assigned: dict[ExternalId, int] = {}
        for external_id in unseen:
            if len(self._slot_by_external) >= self.capacity:
                self._overflow_ids.add(external_id)
                continue
            slot = len(self._slot_by_external)
            self._slot_by_external[external_id] = slot
            self._external_by_slot[slot] = external_id
            assigned[external_id] = slot
        return assigned

    @staticmethod
    def _validate_id(external_id: ExternalId) -> None:
        if isinstance(external_id, bool) or not isinstance(external_id, (int, str)):
            raise TypeError("external IDs must be int or str, excluding bool")
