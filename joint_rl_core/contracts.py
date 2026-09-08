"""Simulator-neutral contracts for a joint reinforcement-learning controller.

The classes in this module deliberately describe generic game entities,
objectives and a shared sensor resource.  They do not import the competition
simulator or translate decisions into simulator commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence


NO_OBJECTIVE = -1


@dataclass(frozen=True)
class JointSpaceSpec:
    """Shared fixed dimensions used by state, masks, encoding and rollout."""

    unit_count: int
    objective_count: int

    def __post_init__(self) -> None:
        for name in ("unit_count", "objective_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class UnitPhase(IntEnum):
    """Lifecycle of one controllable game entity."""

    STAGED = 0
    PENDING = 1
    ACTIVE = 2
    TERMINAL = 3


class BinaryChoice(IntEnum):
    NO = 0
    YES = 1


class Movement(IntEnum):
    """A single three-way movement-control axis."""

    NEGATIVE = 0
    NEUTRAL = 1
    POSITIVE = 2


@dataclass(frozen=True)
class UnitAction:
    """One entity's conditional hybrid action.

    ``placement`` is normalized to ``[-1, 1]`` on each axis.  It is active
    only when a staged entity chooses ``activate=YES``.  ``objective_slot`` is
    active on activation or when an active entity chooses ``retarget=YES``.
    """

    slot: int
    activate: BinaryChoice = BinaryChoice.NO
    placement: tuple[float, float] = (0.0, 0.0)
    objective_slot: int = NO_OBJECTIVE
    retarget: BinaryChoice = BinaryChoice.NO
    movement: Movement = Movement.NEUTRAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "placement", tuple(self.placement))

    @classmethod
    def noop(cls, slot: int) -> "UnitAction":
        return cls(slot=slot)


@dataclass(frozen=True)
class SharedSensorAction:
    """Requests a shared sensor service for zero or more active entities."""

    requester_slots: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requester_slots",
            tuple(int(slot) for slot in self.requester_slots),
        )

    @classmethod
    def from_sequence(cls, requester_slots: Sequence[int]) -> "SharedSensorAction":
        return cls(tuple(int(slot) for slot in requester_slots))


@dataclass(frozen=True)
class JointAction:
    """One simultaneous action for the complete controlled team."""

    units: tuple[UnitAction, ...]
    shared_sensor: SharedSensorAction = SharedSensorAction()

    def __post_init__(self) -> None:
        object.__setattr__(self, "units", tuple(self.units))

    @classmethod
    def from_sequence(
        cls,
        units: Sequence[UnitAction],
        shared_sensor: SharedSensorAction | None = None,
    ) -> "JointAction":
        return cls(tuple(units), shared_sensor or SharedSensorAction())


@dataclass(frozen=True)
class UnitControlState:
    """Controller-owned history that is absent from raw observations."""

    slot: int
    phase: UnitPhase = UnitPhase.STAGED
    activation_requested_step: int | None = None
    pending_objective_slot: int = NO_OBJECTIVE
    activated_step: int | None = None
    current_objective_slot: int = NO_OBJECTIVE
    last_objective_change_step: int | None = None
    last_movement: Movement = Movement.NEUTRAL


@dataclass(frozen=True)
class SharedSensorConfig:
    capacity: int
    max_requests_per_step: int = 1
    cooldown_steps: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.capacity, bool)
            or not isinstance(self.capacity, int)
            or self.capacity < 0
        ):
            raise ValueError("shared sensor capacity must be a non-negative integer")
        if (
            isinstance(self.max_requests_per_step, bool)
            or not isinstance(self.max_requests_per_step, int)
            or self.max_requests_per_step <= 0
        ):
            raise ValueError("max_requests_per_step must be a positive integer")
        if (
            isinstance(self.cooldown_steps, bool)
            or not isinstance(self.cooldown_steps, int)
            or self.cooldown_steps < 0
        ):
            raise ValueError("cooldown_steps must be a non-negative integer")


@dataclass(frozen=True)
class SharedSensorState:
    config: SharedSensorConfig
    remaining: int
    last_request_step: int | None = None
    pending: tuple["PendingSharedSensorRequest", ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "pending", tuple(self.pending))
        if isinstance(self.remaining, bool) or not isinstance(self.remaining, int):
            raise ValueError("shared sensor remaining budget must be an integer")
        if self.remaining < 0 or self.remaining > self.config.capacity:
            raise ValueError("shared sensor remaining budget is out of range")
        if self.last_request_step is not None and (
            isinstance(self.last_request_step, bool)
            or not isinstance(self.last_request_step, int)
            or self.last_request_step < 0
        ):
            raise ValueError("last_request_step must be a non-negative integer")
        tokens = {(item.requester_slot, item.request_step) for item in self.pending}
        if len(tokens) != len(self.pending):
            raise ValueError("shared sensor state contains duplicate pending requests")
        if len(self.pending) > self.remaining:
            raise ValueError("pending shared sensor requests exceed remaining budget")

    @classmethod
    def initial(cls, config: SharedSensorConfig) -> "SharedSensorState":
        return cls(config=config, remaining=config.capacity)

    @property
    def available(self) -> int:
        return self.remaining - len(self.pending)

    def is_ready(self, step: int) -> bool:
        if self.available <= 0 or self.pending:
            return False
        if self.last_request_step is None:
            return True
        return step - self.last_request_step > self.config.cooldown_steps


@dataclass(frozen=True)
class BranchActivity:
    """Policy terms that participate in one entity's sampled action."""

    activation: bool = False
    placement: bool = False
    objective: bool = False
    retarget: bool = False
    movement: bool = False

    def names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in ("activation", "placement", "objective", "retarget", "movement")
            if bool(getattr(self, name))
        )


@dataclass(frozen=True)
class ActivationIntent:
    slot: int
    placement: tuple[float, float]
    objective_slot: int
    request_step: int


@dataclass(frozen=True)
class ActivationReceipt:
    slot: int
    request_step: int
    accepted: bool


@dataclass(frozen=True)
class RetargetIntent:
    slot: int
    objective_slot: int


@dataclass(frozen=True)
class MovementIntent:
    slot: int
    movement: Movement


@dataclass(frozen=True)
class SharedSensorIntent:
    requester_slot: int
    request_step: int


@dataclass(frozen=True)
class PendingSharedSensorRequest:
    requester_slot: int
    request_step: int


@dataclass(frozen=True)
class SharedSensorReceipt:
    requester_slot: int
    request_step: int
    accepted: bool


@dataclass(frozen=True)
class ResolvedIntents:
    """Validated, simulator-neutral effects for one decision step."""

    activations: tuple[ActivationIntent, ...] = ()
    retargets: tuple[RetargetIntent, ...] = ()
    movements: tuple[MovementIntent, ...] = ()
    shared_sensor: tuple[SharedSensorIntent, ...] = ()
