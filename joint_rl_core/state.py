"""Lifecycle and shared-resource state transitions for a joint controller."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .contracts import (
    ActivationIntent,
    ActivationReceipt,
    BinaryChoice,
    JointAction,
    JointSpaceSpec,
    MovementIntent,
    NO_OBJECTIVE,
    PendingSharedSensorRequest,
    ResolvedIntents,
    RetargetIntent,
    SharedSensorConfig,
    SharedSensorIntent,
    SharedSensorReceipt,
    SharedSensorState,
    UnitControlState,
    UnitPhase,
)
from .masking import canonicalize_joint_action, validate_joint_action


class JointControlTracker:
    """Own the history needed by a policy but absent from raw observations."""

    def __init__(self, space: JointSpaceSpec, sensor_config: SharedSensorConfig) -> None:
        self.space = space
        self._initial_states = tuple(
            UnitControlState(slot=slot) for slot in range(space.unit_count)
        )
        self._sensor_config = sensor_config
        self.reset()

    def reset(self) -> None:
        self._states = self._initial_states
        self._sensor = SharedSensorState.initial(self._sensor_config)
        self._last_applied_step: int | None = None
        self._latest_event_step: int | None = None

    @property
    def states(self) -> tuple[UnitControlState, ...]:
        return self._states

    @property
    def sensor_state(self) -> SharedSensorState:
        return self._sensor

    @property
    def latest_event_step(self) -> int | None:
        return self._latest_event_step

    def _validate_event_step(self, step: int) -> None:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")
        if self._latest_event_step is not None and step < self._latest_event_step:
            raise ValueError("controller events must use non-decreasing steps")

    def _record_event_step(self, step: int) -> None:
        self._latest_event_step = step

    def mark_terminal(self, slots: Sequence[int], *, step: int) -> None:
        self._validate_event_step(step)
        terminal_slots = {int(slot) for slot in slots}
        if any(slot < 0 or slot >= len(self._states) for slot in terminal_slots):
            raise IndexError("terminal unit slot is out of range")
        self._states = tuple(
            replace(
                state,
                phase=UnitPhase.TERMINAL,
                activation_requested_step=None,
                pending_objective_slot=NO_OBJECTIVE,
            )
            if state.slot in terminal_slots
            else state
            for state in self._states
        )
        # Shared-resource requests remain pending even when their requester
        # terminates.  The external service may already have accepted one, so
        # only an explicit receipt or timeout may release its reservation.
        self._record_event_step(step)

    def confirm_activations(
        self,
        receipts: Sequence[ActivationReceipt],
        *,
        step: int,
    ) -> None:
        """Apply explicit execution receipts for pending activation intents.

        An integration may call this in the same outer game step when execution
        is synchronous.  Keeping the acknowledgement explicit prevents a
        dropped external command from silently changing controller history.
        """

        self._validate_event_step(step)
        by_token = {(int(item.slot), int(item.request_step)): item for item in receipts}
        if len(by_token) != len(receipts):
            raise ValueError("activation receipts contain duplicate request tokens")
        state_by_slot = {state.slot: state for state in self._states}
        for slot, request_step in by_token:
            state = state_by_slot.get(slot)
            if state is None:
                raise IndexError("activation receipt contains an out-of-range slot")
            if (
                state.phase != UnitPhase.PENDING
                or state.activation_requested_step != request_step
            ):
                raise ValueError(f"unit {slot} receipt does not match its pending request")
            if step < request_step:
                raise ValueError("activation receipt predates its request")

        updated: list[UnitControlState] = []
        for state in self._states:
            request_step = state.activation_requested_step
            receipt = (
                by_token.get((state.slot, request_step))
                if request_step is not None
                else None
            )
            if receipt is not None and receipt.accepted:
                updated.append(
                    replace(
                        state,
                        phase=UnitPhase.ACTIVE,
                        activated_step=step,
                        current_objective_slot=state.pending_objective_slot,
                        last_objective_change_step=step,
                        activation_requested_step=None,
                        pending_objective_slot=NO_OBJECTIVE,
                    )
                )
            elif receipt is not None:
                updated.append(
                    replace(
                        state,
                        phase=UnitPhase.STAGED,
                        activation_requested_step=None,
                        pending_objective_slot=NO_OBJECTIVE,
                    )
                )
            else:
                updated.append(state)
        self._states = tuple(updated)
        self._record_event_step(step)

    def expire_pending_activations(
        self, *, step: int, timeout_steps: int
    ) -> tuple[int, ...]:
        """Reject pending activation requests whose explicit timeout elapsed."""

        if timeout_steps <= 0:
            raise ValueError("timeout_steps must be positive")
        expired = tuple(
            state.slot
            for state in self._states
            if state.phase == UnitPhase.PENDING
            and state.activation_requested_step is not None
            and step - state.activation_requested_step >= timeout_steps
        )
        receipts = tuple(
            ActivationReceipt(
                slot=slot,
                request_step=int(self._states[slot].activation_requested_step),
                accepted=False,
            )
            for slot in expired
        )
        self.confirm_activations(receipts, step=step)
        return expired

    def confirm_shared_sensor(
        self,
        receipts: Sequence[SharedSensorReceipt],
        *,
        step: int,
    ) -> None:
        """Commit shared-resource usage only for matching accepted receipts."""

        self._validate_event_step(step)
        by_token = {
            (int(item.requester_slot), int(item.request_step)): item
            for item in receipts
        }
        if len(by_token) != len(receipts):
            raise ValueError("shared sensor receipts contain duplicate request tokens")
        pending_by_token = {
            (item.requester_slot, item.request_step): item
            for item in self._sensor.pending
        }
        for token in by_token:
            if token not in pending_by_token:
                raise ValueError("shared sensor receipt does not match a pending request")
            if step < token[1]:
                raise ValueError("shared sensor receipt predates its request")
        accepted_count = sum(bool(item.accepted) for item in by_token.values())
        remaining_pending = tuple(
            item
            for token, item in pending_by_token.items()
            if token not in by_token
        )
        self._sensor = replace(
            self._sensor,
            remaining=self._sensor.remaining - accepted_count,
            last_request_step=(
                step if accepted_count else self._sensor.last_request_step
            ),
            pending=remaining_pending,
        )
        self._record_event_step(step)

    def expire_pending_shared_sensor(
        self, *, step: int, timeout_steps: int
    ) -> tuple[int, ...]:
        if timeout_steps <= 0:
            raise ValueError("timeout_steps must be positive")
        expired = tuple(
            item
            for item in self._sensor.pending
            if step - item.request_step >= timeout_steps
        )
        self.confirm_shared_sensor(
            tuple(
                SharedSensorReceipt(
                    requester_slot=item.requester_slot,
                    request_step=item.request_step,
                    accepted=False,
                )
                for item in expired
            ),
            step=step,
        )
        return tuple(item.requester_slot for item in expired)

    def apply(
        self,
        action: JointAction,
        objective_valid: Sequence[bool],
        *,
        step: int,
    ) -> ResolvedIntents:
        """Validate one action, update history, and emit neutral intents."""

        self._validate_event_step(step)
        if self._last_applied_step is not None and step <= self._last_applied_step:
            raise ValueError("joint actions must use strictly increasing steps")
        if len(objective_valid) != self.space.objective_count:
            raise ValueError(
                f"expected {self.space.objective_count} objective slots, "
                f"got {len(objective_valid)}"
            )
        validate_joint_action(
            self.space,
            self._states,
            objective_valid,
            self._sensor,
            action,
            step=step,
        )
        canonical = canonicalize_joint_action(self._states, action)
        by_slot = {item.slot: item for item in canonical.units}

        activations: list[ActivationIntent] = []
        retargets: list[RetargetIntent] = []
        movements: list[MovementIntent] = []
        updated: list[UnitControlState] = []
        for state in self._states:
            unit_action = by_slot[state.slot]
            if state.phase == UnitPhase.STAGED and unit_action.activate == BinaryChoice.YES:
                activations.append(
                    ActivationIntent(
                        slot=state.slot,
                        placement=unit_action.placement,
                        objective_slot=unit_action.objective_slot,
                        request_step=step,
                    )
                )
                updated.append(
                    replace(
                        state,
                        phase=UnitPhase.PENDING,
                        activation_requested_step=step,
                        pending_objective_slot=unit_action.objective_slot,
                    )
                )
            elif state.phase == UnitPhase.ACTIVE:
                current_objective = state.current_objective_slot
                last_change = state.last_objective_change_step
                if unit_action.retarget == BinaryChoice.YES:
                    current_objective = unit_action.objective_slot
                    last_change = step
                    retargets.append(
                        RetargetIntent(state.slot, unit_action.objective_slot)
                    )
                movements.append(MovementIntent(state.slot, unit_action.movement))
                updated.append(
                    replace(
                        state,
                        current_objective_slot=current_objective,
                        last_objective_change_step=last_change,
                        last_movement=unit_action.movement,
                    )
                )
            else:
                updated.append(state)

        sensor_intents = tuple(
            SharedSensorIntent(requester_slot=int(slot), request_step=step)
            for slot in canonical.shared_sensor.requester_slots
        )
        if sensor_intents:
            self._sensor = replace(
                self._sensor,
                pending=tuple(
                    PendingSharedSensorRequest(
                        requester_slot=item.requester_slot,
                        request_step=item.request_step,
                    )
                    for item in sensor_intents
                ),
            )
        self._states = tuple(updated)
        self._last_applied_step = step
        self._record_event_step(step)
        return ResolvedIntents(
            activations=tuple(activations),
            retargets=tuple(retargets),
            movements=tuple(movements),
            shared_sensor=sensor_intents,
        )
