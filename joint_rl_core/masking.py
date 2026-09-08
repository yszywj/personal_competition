"""Conditional masks and validation for joint hybrid actions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ._arrays import immutable_array
from .contracts import (
    BinaryChoice,
    BranchActivity,
    JointAction,
    JointSpaceSpec,
    Movement,
    NO_OBJECTIVE,
    SharedSensorState,
    UnitAction,
    UnitControlState,
    UnitPhase,
)


class ActionValidationError(ValueError):
    """Raised when a joint action violates its current conditional mask."""


@dataclass(frozen=True)
class UnitActionMask:
    activation: np.ndarray
    objective: np.ndarray
    retarget: np.ndarray
    movement: np.ndarray
    placement_possible: bool

    def __post_init__(self) -> None:
        for name in ("activation", "objective", "retarget", "movement"):
            value = immutable_array(getattr(self, name), dtype=np.bool_)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class JointActionMask:
    space: JointSpaceSpec
    by_unit: Mapping[int, UnitActionMask]
    shared_sensor_eligible: np.ndarray
    shared_sensor_max_requests: int

    def __post_init__(self) -> None:
        if set(self.by_unit) != set(range(self.space.unit_count)):
            raise ValueError("joint mask must contain exactly one mask per unit slot")
        for slot, unit_mask in self.by_unit.items():
            expected_shapes = {
                "activation": (2,),
                "objective": (self.space.objective_count,),
                "retarget": (2,),
                "movement": (3,),
            }
            for name, expected_shape in expected_shapes.items():
                if getattr(unit_mask, name).shape != expected_shape:
                    raise ValueError(f"unit {slot} {name} mask has the wrong shape")
            if bool(unit_mask.placement_possible) != bool(unit_mask.activation[1]):
                raise ValueError("placement availability must match activation availability")
        object.__setattr__(self, "by_unit", MappingProxyType(dict(self.by_unit)))
        eligible = immutable_array(self.shared_sensor_eligible, dtype=np.bool_)
        if eligible.shape != (self.space.unit_count,):
            raise ValueError("shared sensor eligibility has the wrong shape")
        object.__setattr__(self, "shared_sensor_eligible", eligible)
        if self.shared_sensor_max_requests < 0:
            raise ValueError("shared_sensor_max_requests must be non-negative")
        if self.shared_sensor_max_requests > int(eligible.sum()):
            raise ValueError("shared sensor request limit exceeds eligible entities")


def build_joint_action_mask(
    space: JointSpaceSpec,
    states: Sequence[UnitControlState],
    objective_valid: Sequence[bool],
    sensor_state: SharedSensorState,
    *,
    step: int,
) -> JointActionMask:
    """Build masks from lifecycle state without inspecting simulator internals."""

    objective = np.asarray(objective_valid, dtype=np.bool_)
    if objective.shape != (space.objective_count,):
        raise ValueError(
            f"objective_valid must have shape {(space.objective_count,)}, "
            f"got {objective.shape}"
        )

    by_unit: dict[int, UnitActionMask] = {}
    slots = [state.slot for state in states]
    if len(states) != space.unit_count or sorted(slots) != list(range(space.unit_count)):
        raise ValueError("states must contain exactly the configured unit slots")

    sensor_eligible = np.zeros(len(states), dtype=np.bool_)
    resource_ready = sensor_state.is_ready(step)
    for state in states:
        staged = state.phase == UnitPhase.STAGED
        active = state.phase == UnitPhase.ACTIVE
        activation = np.asarray((True, staged and bool(objective.any())), dtype=np.bool_)
        retarget = np.asarray((True, active and bool(objective.any())), dtype=np.bool_)
        movement = np.asarray((active, True, active), dtype=np.bool_)
        by_unit[state.slot] = UnitActionMask(
            activation=activation,
            objective=objective.copy(),
            retarget=retarget,
            movement=movement,
            placement_possible=staged and bool(objective.any()),
        )
        sensor_eligible[state.slot] = active and resource_ready

    maximum = (
        min(
            sensor_state.config.max_requests_per_step,
            sensor_state.available,
            int(sensor_eligible.sum()),
        )
        if resource_ready
        else 0
    )
    return JointActionMask(
        space=space,
        by_unit=by_unit,
        shared_sensor_eligible=sensor_eligible,
        shared_sensor_max_requests=maximum,
    )


def branch_activity(state: UnitControlState, action: UnitAction) -> BranchActivity:
    """Return exactly the log-probability terms active for this decision."""

    if state.phase == UnitPhase.STAGED:
        activates = action.activate == BinaryChoice.YES
        return BranchActivity(
            activation=True,
            placement=activates,
            objective=activates,
        )
    if state.phase == UnitPhase.ACTIVE:
        retargets = action.retarget == BinaryChoice.YES
        return BranchActivity(
            objective=retargets,
            retarget=True,
            movement=True,
        )
    return BranchActivity()


def expected_log_prob_terms(
    states: Sequence[UnitControlState],
    action: JointAction,
    *,
    shared_sensor_active: bool = False,
) -> frozenset[str]:
    """Name active policy terms so PPO cannot score ignored branches."""

    state_by_slot = {state.slot: state for state in states}
    names: set[str] = set()
    for unit_action in action.units:
        activity = branch_activity(state_by_slot[unit_action.slot], unit_action)
        names.update(f"unit/{unit_action.slot}/{name}" for name in activity.names())
    if shared_sensor_active:
        names.add("shared_sensor")
    return frozenset(names)


def validate_joint_action(
    space: JointSpaceSpec,
    states: Sequence[UnitControlState],
    objective_valid: Sequence[bool],
    sensor_state: SharedSensorState,
    action: JointAction,
    *,
    step: int,
) -> JointActionMask:
    """Validate categorical choices and only the continuous branches in use."""

    mask = build_joint_action_mask(
        space, states, objective_valid, sensor_state, step=step
    )
    validate_action_against_mask(states, action, mask)
    return mask


def validate_action_against_mask(
    states: Sequence[UnitControlState],
    action: JointAction,
    mask: JointActionMask,
) -> None:
    """Validate an action against the exact mask saved at sampling time."""

    state_by_slot = {state.slot: state for state in states}
    if set(state_by_slot) != set(range(mask.space.unit_count)):
        raise ActionValidationError("states do not match the saved joint mask")
    action_by_slot = {unit.slot: unit for unit in action.units}
    if len(action_by_slot) != len(action.units):
        raise ActionValidationError("joint action contains duplicate unit slots")
    if set(action_by_slot) != set(state_by_slot):
        raise ActionValidationError("joint action must contain exactly one action per unit")

    for slot, unit_action in action_by_slot.items():
        state = state_by_slot[slot]
        unit_mask = mask.by_unit[slot]
        try:
            activate_index = int(BinaryChoice(unit_action.activate))
            retarget_index = int(BinaryChoice(unit_action.retarget))
            movement_index = int(Movement(unit_action.movement))
        except ValueError as error:
            raise ActionValidationError(
                f"unit {slot} has an invalid categorical action"
            ) from error
        if not unit_mask.activation[activate_index]:
            raise ActionValidationError(f"unit {slot} activation choice is masked")
        if not unit_mask.retarget[retarget_index]:
            raise ActionValidationError(f"unit {slot} retarget choice is masked")
        if not unit_mask.movement[movement_index]:
            raise ActionValidationError(f"unit {slot} movement choice is masked")

        activity = branch_activity(state, unit_action)
        if activity.placement:
            if len(unit_action.placement) != 2 or not all(
                math.isfinite(float(value)) and -1.0 <= float(value) <= 1.0
                for value in unit_action.placement
            ):
                raise ActionValidationError(
                    f"unit {slot} active placement must contain two finite values in [-1, 1]"
                )
        if activity.objective:
            objective_slot = int(unit_action.objective_slot)
            if (
                objective_slot < 0
                or objective_slot >= len(unit_mask.objective)
                or not unit_mask.objective[objective_slot]
            ):
                raise ActionValidationError(f"unit {slot} selected an invalid objective slot")

    requesters = action.shared_sensor.requester_slots
    if len(set(requesters)) != len(requesters):
        raise ActionValidationError("shared sensor request contains duplicate unit slots")
    if len(requesters) > mask.shared_sensor_max_requests:
        raise ActionValidationError(
            "shared sensor request exceeds the per-step or remaining budget"
        )
    for slot in requesters:
        if slot < 0 or slot >= mask.shared_sensor_eligible.size:
            raise ActionValidationError(f"shared sensor requester slot {slot} is out of range")
        if not mask.shared_sensor_eligible[slot]:
            raise ActionValidationError(f"unit {slot} is not eligible for the shared sensor")


def canonicalize_unit_action(
    state: UnitControlState, action: UnitAction
) -> UnitAction:
    """Replace inactive branch values with stable no-op sentinels."""

    if state.phase == UnitPhase.STAGED:
        if action.activate == BinaryChoice.YES:
            return UnitAction(
                slot=state.slot,
                activate=BinaryChoice.YES,
                placement=(float(action.placement[0]), float(action.placement[1])),
                objective_slot=int(action.objective_slot),
            )
        return UnitAction.noop(state.slot)
    if state.phase == UnitPhase.ACTIVE:
        return UnitAction(
            slot=state.slot,
            objective_slot=(
                int(action.objective_slot)
                if action.retarget == BinaryChoice.YES
                else NO_OBJECTIVE
            ),
            retarget=BinaryChoice(action.retarget),
            movement=Movement(action.movement),
        )
    return UnitAction.noop(state.slot)


def canonicalize_joint_action(
    states: Sequence[UnitControlState], action: JointAction
) -> JointAction:
    state_by_slot = {state.slot: state for state in states}
    units = tuple(
        canonicalize_unit_action(state_by_slot[item.slot], item)
        for item in sorted(action.units, key=lambda value: value.slot)
    )
    return JointAction(units=units, shared_sensor=action.shared_sensor)
