"""Conditional-policy trajectory records independent of a learning library."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ._arrays import immutable_array
from .contracts import JointAction, JointSpaceSpec, UnitControlState
from .masking import (
    JointActionMask,
    UnitActionMask,
    canonicalize_joint_action,
    expected_log_prob_terms,
    validate_action_against_mask,
)


@dataclass(frozen=True)
class SharedSensorPolicyTrace:
    """Autoregressive STOP/slot samples for the shared-resource head.

    Token 0 means STOP and token ``slot + 1`` selects one entity.  A mask and
    log probability are retained for every draw, which is sufficient to
    reproduce sampling when more than one request per step is allowed.
    """

    tokens: tuple[int, ...]
    masks: tuple[np.ndarray, ...]
    log_probs: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", tuple(int(value) for value in self.tokens))
        object.__setattr__(
            self,
            "log_probs",
            tuple(float(value) for value in self.log_probs),
        )
        copied: list[np.ndarray] = []
        for raw_mask in self.masks:
            copied.append(immutable_array(raw_mask, dtype=np.bool_))
        object.__setattr__(self, "masks", tuple(copied))

    @property
    def total_log_prob(self) -> float:
        return float(sum(self.log_probs))

    def validate(self, action: JointAction, mask: JointActionMask) -> None:
        if not self.tokens or not (
            len(self.tokens) == len(self.masks) == len(self.log_probs)
        ):
            raise ValueError("shared sensor trace lengths are inconsistent")
        if not bool(np.isfinite(np.asarray(self.log_probs, dtype=float)).all()):
            raise ValueError("shared sensor trace contains a non-finite log probability")

        available = mask.shared_sensor_eligible.copy()
        selected: list[int] = []
        stopped = False
        for index, (token, stored_mask) in enumerate(zip(self.tokens, self.masks)):
            expected_mask = np.concatenate(
                (np.asarray((True,), dtype=np.bool_), available)
            )
            if stored_mask.shape != expected_mask.shape or not np.array_equal(
                stored_mask, expected_mask
            ):
                raise ValueError("shared sensor trace does not preserve its sampling mask")
            if token < 0 or token >= expected_mask.size or not expected_mask[token]:
                raise ValueError("shared sensor trace selected a masked token")
            if token == 0:
                if index != len(self.tokens) - 1:
                    raise ValueError("shared sensor STOP token must end the trace")
                stopped = True
                continue
            slot = token - 1
            selected.append(slot)
            available[slot] = False
            if len(selected) > mask.shared_sensor_max_requests:
                raise ValueError("shared sensor trace exceeds the saved request limit")
            if (
                len(selected) == mask.shared_sensor_max_requests
                and index != len(self.tokens) - 1
            ):
                raise ValueError("shared sensor trace continued after reaching its limit")

        if tuple(selected) != action.shared_sensor.requester_slots:
            raise ValueError("shared sensor trace and joint action select different slots")
        if len(selected) < mask.shared_sensor_max_requests and not stopped:
            raise ValueError("shared sensor trace ended without STOP")


@dataclass(frozen=True)
class JointPolicyTrace:
    """Policy statistics for per-entity and team-level advantages."""

    log_prob_by_term: Mapping[str, float]
    values_by_unit: tuple[float, ...]
    team_value: float
    shared_sensor: SharedSensorPolicyTrace | None = None
    # Optional branch-specific baselines.  ``None`` keeps legacy checkpoints
    # and callers valid while allowing planning, movement and sensor returns to
    # use independent critics.
    plan_values_by_unit: tuple[float, ...] | None = None
    motion_values_by_unit: tuple[float, ...] | None = None
    sensor_value: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "log_prob_by_term",
            MappingProxyType(
                {str(name): float(value) for name, value in self.log_prob_by_term.items()}
            ),
        )
        object.__setattr__(
            self,
            "values_by_unit",
            tuple(float(value) for value in self.values_by_unit),
        )
        object.__setattr__(self, "team_value", float(self.team_value))
        for name in ("plan_values_by_unit", "motion_values_by_unit"):
            raw_values = getattr(self, name)
            if raw_values is not None:
                object.__setattr__(
                    self,
                    name,
                    tuple(float(value) for value in raw_values),
                )
        if self.sensor_value is not None:
            object.__setattr__(self, "sensor_value", float(self.sensor_value))

    @property
    def total_log_prob(self) -> float:
        return float(sum(self.log_prob_by_term.values()))

    def validate(
        self,
        states: Sequence[UnitControlState],
        action: JointAction,
        mask: JointActionMask,
    ) -> None:
        sensor_active = mask.shared_sensor_max_requests > 0
        expected = expected_log_prob_terms(
            states,
            action,
            shared_sensor_active=sensor_active,
        )
        actual = frozenset(self.log_prob_by_term)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"conditional log-probability terms differ: missing={missing}, extra={extra}"
            )
        if len(self.values_by_unit) != len(states):
            raise ValueError("policy trace must contain one value per unit")
        branch_values: tuple[float, ...] = ()
        for name in ("plan_values_by_unit", "motion_values_by_unit"):
            raw_values = getattr(self, name)
            if raw_values is None:
                continue
            if len(raw_values) != len(states):
                raise ValueError(f"{name} must contain one value per unit")
            branch_values += raw_values
        sensor_values = (
            () if self.sensor_value is None else (float(self.sensor_value),)
        )
        values = np.asarray(
            tuple(self.log_prob_by_term.values())
            + self.values_by_unit
            + (float(self.team_value),)
            + branch_values
            + sensor_values,
            dtype=float,
        )
        if not bool(np.isfinite(values).all()):
            raise ValueError("policy trace contains a non-finite value")

        if sensor_active:
            if self.shared_sensor is None:
                raise ValueError("active shared sensor head is missing its sampling trace")
            self.shared_sensor.validate(action, mask)
            if not math.isclose(
                self.shared_sensor.total_log_prob,
                self.log_prob_by_term["shared_sensor"],
                rel_tol=1e-7,
                abs_tol=1e-7,
            ):
                raise ValueError("shared sensor aggregate log probability is inconsistent")
        elif self.shared_sensor is not None:
            raise ValueError("inactive shared sensor head must not have a sampling trace")


@dataclass(frozen=True)
class JointTransition:
    observations: tuple[np.ndarray, ...]
    states: tuple[UnitControlState, ...]
    mask: JointActionMask
    action: JointAction
    trace: JointPolicyTrace
    rewards: tuple[float, ...]
    team_reward: float
    next_observations: tuple[np.ndarray, ...]
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    team_terminated: bool = False
    team_truncated: bool = False
    # Explicit reward channels are optional for source compatibility.  New
    # branch-aware trainers should populate all three instead of inferring
    # planning or sensor credit from ``rewards``.
    plan_rewards: tuple[float, ...] | None = None
    motion_rewards: tuple[float, ...] | None = None
    sensor_reward: float | None = None


class JointTrajectoryBuffer:
    """Append-only, immutable snapshots of complete joint game steps."""

    def __init__(self, space: JointSpaceSpec, observation_dim: int) -> None:
        if (
            isinstance(observation_dim, bool)
            or not isinstance(observation_dim, int)
            or observation_dim <= 0
        ):
            raise ValueError("observation_dim must be a positive integer")
        self.space = space
        self.observation_dim = int(observation_dim)
        self._items: list[JointTransition] = []

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> tuple[JointTransition, ...]:
        return tuple(self._items)

    def clear(self) -> None:
        self._items.clear()

    def extend(self, other: "JointTrajectoryBuffer") -> None:
        """Append validated immutable copies of another compatible buffer."""

        self._validate_compatible_buffer(other, operation="extend")
        for transition in other.items:
            self.append(transition)

    def extend_snapshots(self, other: "JointTrajectoryBuffer") -> None:
        """Append another buffer's already immutable transition snapshots.

        Unlike :meth:`extend`, this method deliberately shares snapshot
        objects.  It is intended for merging collector-owned buffers after
        :meth:`append` has already performed defensive copies and validation.
        Every nested array in such a snapshot is backed by immutable bytes and
        all other records are frozen, so clearing the source buffer cannot
        mutate the destination.
        """

        self._validate_compatible_buffer(other, operation="extend snapshots from")
        self._items.extend(other._items)

    def with_episode_plan_rewards(
        self,
        plan_rewards: Sequence[float],
    ) -> "JointTrajectoryBuffer":
        """Return shallow snapshot replacements with final episode credit.

        Planning credit is known only after an episode finishes.  Replacing
        the frozen transition records is sufficient here: the observations,
        masks, actions and traces in this buffer are already immutable,
        validated snapshots and therefore do not need to be copied again.
        """

        if len(plan_rewards) != self.space.unit_count:
            raise ValueError("plan_rewards must contain one reward per unit")
        validated_plan_rewards = tuple(float(value) for value in plan_rewards)
        if not bool(
            np.isfinite(np.asarray(validated_plan_rewards, dtype=float)).all()
        ):
            raise ValueError("plan_rewards contain a non-finite value")

        finalized = JointTrajectoryBuffer(self.space, self.observation_dim)
        finalized._items = [
            replace(
                transition,
                plan_rewards=validated_plan_rewards,
                motion_rewards=transition.rewards,
                sensor_reward=(
                    transition.team_reward
                    if transition.sensor_reward is None
                    else transition.sensor_reward
                ),
            )
            for transition in self._items
        ]
        return finalized

    def _validate_compatible_buffer(
        self,
        other: "JointTrajectoryBuffer",
        *,
        operation: str,
    ) -> None:
        if not isinstance(other, JointTrajectoryBuffer):
            raise TypeError(f"{operation} expects a JointTrajectoryBuffer")
        if other is self:
            raise ValueError(f"a trajectory buffer cannot {operation} itself")
        if other.space != self.space:
            raise ValueError("trajectory buffers use different joint spaces")
        if other.observation_dim != self.observation_dim:
            raise ValueError(
                "trajectory buffers use different observation dimensions"
            )

    def append(self, transition: JointTransition) -> None:
        """Validate and defensively copy a transition from an arbitrary caller."""

        self._append(transition, reuse_frozen_records=False)

    def append_collected(self, transition: JointTransition) -> None:
        """Append collector data while reusing intrinsically frozen records.

        Observations and scalar sequences still receive defensive snapshots.
        Masks and policy traces already make their mappings and nested arrays
        immutable during construction, while actions and control states are
        frozen dataclasses, so copying those records again adds no isolation.
        """

        self._append(transition, reuse_frozen_records=True)

    def _append(
        self,
        transition: JointTransition,
        *,
        reuse_frozen_records: bool,
    ) -> None:
        expected_count = self.space.unit_count
        fields = (
            transition.observations,
            transition.states,
            transition.action.units,
            transition.rewards,
            transition.next_observations,
            transition.terminated,
            transition.truncated,
        )
        if any(len(field) != expected_count for field in fields):
            raise ValueError("joint transition must contain one item per unit")
        expected_slots = list(range(expected_count))
        if [state.slot for state in transition.states] != expected_slots:
            raise ValueError("transition states must be ordered by configured unit slots")
        if [action.slot for action in transition.action.units] != expected_slots:
            raise ValueError("transition actions must be ordered by configured unit slots")
        if transition.mask.space != self.space:
            raise ValueError("transition mask uses a different joint space")
        validate_action_against_mask(
            transition.states,
            transition.action,
            transition.mask,
        )
        transition.trace.validate(
            transition.states,
            transition.action,
            transition.mask,
        )
        observations = self._copy_observations(transition.observations)
        next_observations = self._copy_observations(transition.next_observations)
        states = (
            tuple(transition.states)
            if reuse_frozen_records
            else tuple(replace(state) for state in transition.states)
        )
        mask = (
            transition.mask
            if reuse_frozen_records
            else self._copy_mask(transition.mask)
        )
        action = canonicalize_joint_action(states, transition.action)
        trace = (
            transition.trace
            if reuse_frozen_records
            else self._copy_trace(transition.trace)
        )
        rewards = tuple(float(value) for value in transition.rewards)
        branch_rewards: dict[str, tuple[float, ...] | None] = {}
        for name in ("plan_rewards", "motion_rewards"):
            raw_values = getattr(transition, name)
            if raw_values is None:
                branch_rewards[name] = None
                continue
            if len(raw_values) != expected_count:
                raise ValueError(f"{name} must contain one reward per unit")
            branch_rewards[name] = tuple(float(value) for value in raw_values)
        sensor_rewards = (
            ()
            if transition.sensor_reward is None
            else (float(transition.sensor_reward),)
        )
        reward_values = np.asarray(
            rewards
            + (float(transition.team_reward),)
            + (branch_rewards["plan_rewards"] or ())
            + (branch_rewards["motion_rewards"] or ())
            + sensor_rewards,
            dtype=float,
        )
        if not bool(np.isfinite(reward_values).all()):
            raise ValueError("transition rewards contain a non-finite value")
        terminated = tuple(bool(value) for value in transition.terminated)
        truncated = tuple(bool(value) for value in transition.truncated)
        if any(left and right for left, right in zip(terminated, truncated)):
            raise ValueError("a unit cannot be both terminated and truncated")
        if transition.team_terminated and transition.team_truncated:
            raise ValueError("the team cannot be both terminated and truncated")

        self._items.append(
            JointTransition(
                observations=observations,
                states=states,
                mask=mask,
                action=action,
                trace=trace,
                rewards=rewards,
                team_reward=float(transition.team_reward),
                next_observations=next_observations,
                terminated=terminated,
                truncated=truncated,
                team_terminated=bool(transition.team_terminated),
                team_truncated=bool(transition.team_truncated),
                plan_rewards=branch_rewards["plan_rewards"],
                motion_rewards=branch_rewards["motion_rewards"],
                sensor_reward=(
                    None
                    if transition.sensor_reward is None
                    else float(transition.sensor_reward)
                ),
            )
        )

    def _copy_observations(
        self, observations: Sequence[np.ndarray]
    ) -> tuple[np.ndarray, ...]:
        copied: list[np.ndarray] = []
        for observation in observations:
            array = np.asarray(observation, dtype=np.float32)
            if array.shape != (self.observation_dim,):
                raise ValueError(
                    f"expected observation shape {(self.observation_dim,)}, got {array.shape}"
                )
            if not bool(np.isfinite(array).all()):
                raise ValueError("transition observation contains a non-finite value")
            copied.append(immutable_array(array, dtype=np.float32))
        return tuple(copied)

    @staticmethod
    def _copy_mask(mask: JointActionMask) -> JointActionMask:
        return JointActionMask(
            space=mask.space,
            by_unit={
                slot: UnitActionMask(
                    activation=value.activation,
                    objective=value.objective,
                    retarget=value.retarget,
                    movement=value.movement,
                    placement_possible=bool(value.placement_possible),
                )
                for slot, value in mask.by_unit.items()
            },
            shared_sensor_eligible=mask.shared_sensor_eligible,
            shared_sensor_max_requests=int(mask.shared_sensor_max_requests),
        )

    @staticmethod
    def _copy_trace(trace: JointPolicyTrace) -> JointPolicyTrace:
        sensor = trace.shared_sensor
        sensor_copy = (
            SharedSensorPolicyTrace(
                tokens=sensor.tokens,
                masks=sensor.masks,
                log_probs=sensor.log_probs,
            )
            if sensor is not None
            else None
        )
        return JointPolicyTrace(
            log_prob_by_term=dict(trace.log_prob_by_term),
            values_by_unit=trace.values_by_unit,
            team_value=trace.team_value,
            shared_sensor=sensor_copy,
            plan_values_by_unit=trace.plan_values_by_unit,
            motion_values_by_unit=trace.motion_values_by_unit,
            sensor_value=trace.sensor_value,
        )
