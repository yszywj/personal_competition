"""Compact per-episode diagnostics for the joint hybrid policy.

The trainer already retains every sampled action in its PPO trajectory.  This
module deliberately keeps a much smaller, JSON-safe summary instead of copying
that trajectory into result files.  It counts only active conditional branches:
sentinel values carried by pending or terminal units never become policy data.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, Sequence

from .joint_rl_core import (
    BinaryChoice,
    JointAction,
    JointActionMask,
    JointPolicyTrace,
    JointSpaceSpec,
    Movement,
    UnitAction,
    UnitControlState,
    UnitPhase,
    validate_action_against_mask,
)


_MOVEMENT_LABELS = ("negative", "neutral", "positive")
_PLACEMENT_GRID_SIZE = 5
_PLACEMENT_EDGE_THRESHOLD = 0.98


class _Outcome(Protocol):
    accepted_activations: Sequence[int]
    accepted_sensor_requests: Sequence[int]


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _slot_sequence(values: Sequence[int], count: int, *, name: str) -> tuple[int, ...]:
    converted = tuple(int(value) for value in values)
    if len(converted) != count:
        raise ValueError(f"{name} must contain exactly {count} entries")
    return converted


class _PlacementAccumulator:
    """Online placement moments and a bounded two-dimensional histogram."""

    def __init__(self) -> None:
        self.count = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x_squared = 0.0
        self.sum_y_squared = 0.0
        self.min_x: float | None = None
        self.max_x: float | None = None
        self.min_y: float | None = None
        self.max_y: float | None = None
        self.x_edge_count = 0
        self.y_edge_count = 0
        self.any_edge_count = 0
        self.grid = [0] * (_PLACEMENT_GRID_SIZE * _PLACEMENT_GRID_SIZE)

    def add(self, x: float, y: float) -> None:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("placement diagnostics require finite coordinates")
        if not (-1.0 <= x <= 1.0 and -1.0 <= y <= 1.0):
            raise ValueError("placement diagnostics require normalized coordinates")
        self.count += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_x_squared += x * x
        self.sum_y_squared += y * y
        self.min_x = x if self.min_x is None else min(self.min_x, x)
        self.max_x = x if self.max_x is None else max(self.max_x, x)
        self.min_y = y if self.min_y is None else min(self.min_y, y)
        self.max_y = y if self.max_y is None else max(self.max_y, y)
        x_edge = abs(x) >= _PLACEMENT_EDGE_THRESHOLD
        y_edge = abs(y) >= _PLACEMENT_EDGE_THRESHOLD
        self.x_edge_count += int(x_edge)
        self.y_edge_count += int(y_edge)
        self.any_edge_count += int(x_edge or y_edge)

        x_bin = min(
            _PLACEMENT_GRID_SIZE - 1,
            int((x + 1.0) * 0.5 * _PLACEMENT_GRID_SIZE),
        )
        y_bin = min(
            _PLACEMENT_GRID_SIZE - 1,
            int((y + 1.0) * 0.5 * _PLACEMENT_GRID_SIZE),
        )
        self.grid[y_bin * _PLACEMENT_GRID_SIZE + x_bin] += 1

    @staticmethod
    def _axis(
        count: int,
        total: float,
        total_squared: float,
        minimum: float | None,
        maximum: float | None,
    ) -> dict[str, float | None]:
        if count == 0:
            return {"mean": None, "std": None, "min": None, "max": None}
        mean = total / count
        variance = max(0.0, total_squared / count - mean * mean)
        return {
            "mean": float(mean),
            "std": float(math.sqrt(variance)),
            "min": float(minimum),
            "max": float(maximum),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": int(self.count),
            "x": self._axis(
                self.count,
                self.sum_x,
                self.sum_x_squared,
                self.min_x,
                self.max_x,
            ),
            "y": self._axis(
                self.count,
                self.sum_y,
                self.sum_y_squared,
                self.min_y,
                self.max_y,
            ),
            "edge_threshold": _PLACEMENT_EDGE_THRESHOLD,
            "x_edge_count": int(self.x_edge_count),
            "y_edge_count": int(self.y_edge_count),
            "any_edge_count": int(self.any_edge_count),
            "any_edge_fraction": _rate(self.any_edge_count, self.count),
            "grid_size": _PLACEMENT_GRID_SIZE,
            "grid_row_major_yx": [int(value) for value in self.grid],
        }


class EpisodeActionDiagnostics:
    """Accumulate one compact empirical action distribution for an episode.

    ``step`` is the zero-based decision step used to build the supplied mask.
    Exact placement records are retained only for accepted activations, so the
    output remains bounded by the configured unit count even if a backend keeps
    rejecting and retrying launch requests.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        space: JointSpaceSpec,
        objective_ids: Sequence[int | None],
        unit_ids: Sequence[int],
        unit_types: Sequence[int],
    ) -> None:
        if not isinstance(space, JointSpaceSpec):
            raise TypeError("space must be a JointSpaceSpec")
        self.space = space
        if len(objective_ids) != space.objective_count:
            raise ValueError(
                "objective_ids must contain exactly one entry per objective slot"
            )
        self.objective_ids = tuple(
            None if value is None else int(value) for value in objective_ids
        )
        self.unit_ids = _slot_sequence(
            unit_ids, space.unit_count, name="unit_ids"
        )
        self.unit_types = _slot_sequence(
            unit_types, space.unit_count, name="unit_types"
        )
        if len(set(self.unit_ids)) != len(self.unit_ids):
            raise ValueError("unit_ids must be unique")

        self._type_keys = tuple(str(value) for value in self.unit_types)
        unique_type_keys = tuple(dict.fromkeys(self._type_keys))
        objective_count = space.objective_count
        unit_count = space.unit_count

        self._steps_observed = 0
        self._last_step: int | None = None
        self._finalized = False

        self._activation_decisions = 0
        self._activation_wait = 0
        self._activation_requested = 0
        self._activation_accepted = 0
        self._activation_rejected = 0
        self._activation_wait_by_unit = [0] * unit_count
        self._activation_requested_by_unit = [0] * unit_count
        self._activation_accepted_by_unit = [0] * unit_count
        self._activation_rejected_by_unit = [0] * unit_count
        self._activation_accepted_step_by_unit: list[int | None] = [None] * unit_count
        self._activation_by_type = {
            key: {
                "decisions": 0,
                "wait": 0,
                "requested": 0,
                "accepted": 0,
                "rejected": 0,
            }
            for key in unique_type_keys
        }

        self._target_activation = [0] * objective_count
        self._target_activation_accepted = [0] * objective_count
        self._target_retarget = [0] * objective_count
        self._target_effective = [0] * objective_count
        self._target_unassigned_effective = 0
        self._retarget_decisions = 0
        self._retarget_selected = 0
        self._retarget_kept = 0
        self._target_by_type = {
            key: {
                "activation_selected_by_slot": [0] * objective_count,
                "retarget_selected_by_slot": [0] * objective_count,
                "effective_by_slot": [0] * objective_count,
            }
            for key in unique_type_keys
        }

        self._placement_requested = _PlacementAccumulator()
        self._placement_accepted = _PlacementAccumulator()
        self._placement_requested_by_type = {
            key: _PlacementAccumulator() for key in unique_type_keys
        }
        self._placement_accepted_by_type = {
            key: _PlacementAccumulator() for key in unique_type_keys
        }
        self._accepted_placements: list[dict[str, Any]] = []

        self._movement_selected = [0, 0, 0]
        self._movement_activation = [0, 0, 0]
        self._movement_active = [0, 0, 0]
        self._movement_by_type = {key: [0, 0, 0] for key in unique_type_keys}
        self._movement_by_objective = [
            [0, 0, 0] for _ in range(objective_count)
        ]
        self._movement_switch_opportunities = 0
        self._movement_switches = 0

        self._sensor_head_active_steps = 0
        self._sensor_stop_selections = 0
        self._sensor_requested = 0
        self._sensor_accepted = 0
        self._sensor_rejected = 0
        self._sensor_requested_by_unit = [0] * unit_count
        self._sensor_accepted_by_unit = [0] * unit_count
        self._sensor_rejected_by_unit = [0] * unit_count
        self._sensor_by_type = {
            key: {"requested": 0, "accepted": 0, "rejected": 0}
            for key in unique_type_keys
        }

    def observe(
        self,
        states: Sequence[UnitControlState],
        mask: JointActionMask,
        action: JointAction,
        trace: JointPolicyTrace,
        outcome: _Outcome,
        step: int,
    ) -> None:
        """Observe one sampled action and its synchronous execution receipts."""

        if self._finalized:
            raise RuntimeError("cannot observe after diagnostics were finalized")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")
        if self._last_step is not None and step <= self._last_step:
            raise ValueError("diagnostic steps must be strictly increasing")
        ordered_states = tuple(states)
        if (
            len(ordered_states) != self.space.unit_count
            or [state.slot for state in ordered_states]
            != list(range(self.space.unit_count))
        ):
            raise ValueError("states must be ordered by configured unit slot")
        if mask.space != self.space:
            raise ValueError("mask uses a different joint space")
        validate_action_against_mask(ordered_states, action, mask)
        trace.validate(ordered_states, action, mask)
        action_by_slot = {item.slot: item for item in action.units}

        requested_activation_slots = {
            state.slot
            for state in ordered_states
            if state.phase == UnitPhase.STAGED
            and action_by_slot[state.slot].activate == BinaryChoice.YES
        }
        requested_sensor_slots = set(action.shared_sensor.requester_slots)
        accepted_activation_slots = self._receipt_slots(
            outcome.accepted_activations,
            requested_activation_slots,
            name="accepted activations",
        )
        accepted_sensor_slots = self._receipt_slots(
            outcome.accepted_sensor_requests,
            requested_sensor_slots,
            name="accepted sensor requests",
        )

        for state in ordered_states:
            slot = state.slot
            unit_action = action_by_slot[slot]
            type_key = self._type_keys[slot]
            if state.phase == UnitPhase.STAGED:
                self._observe_staged(
                    state,
                    unit_action,
                    type_key,
                    accepted=slot in accepted_activation_slots,
                    step=step,
                )
            elif state.phase == UnitPhase.ACTIVE:
                self._observe_active(state, unit_action, type_key)

        self._observe_sensor(
            mask,
            action,
            trace,
            accepted_sensor_slots,
        )
        self._steps_observed += 1
        self._last_step = step

    def _receipt_slots(
        self,
        raw_slots: Sequence[int],
        requested_slots: set[int],
        *,
        name: str,
    ) -> set[int]:
        slots = tuple(int(slot) for slot in raw_slots)
        if len(set(slots)) != len(slots):
            raise ValueError(f"{name} contain duplicate slots")
        if any(slot < 0 or slot >= self.space.unit_count for slot in slots):
            raise ValueError(f"{name} contain an out-of-range slot")
        accepted = set(slots)
        if not accepted.issubset(requested_slots):
            raise ValueError(f"{name} contain a slot that was not requested")
        return accepted

    def _observe_staged(
        self,
        state: UnitControlState,
        unit_action: UnitAction,
        type_key: str,
        *,
        accepted: bool,
        step: int,
    ) -> None:
        slot = state.slot
        by_type = self._activation_by_type[type_key]
        self._activation_decisions += 1
        by_type["decisions"] += 1
        if unit_action.activate != BinaryChoice.YES:
            self._activation_wait += 1
            self._activation_wait_by_unit[slot] += 1
            by_type["wait"] += 1
            return

        objective_slot = int(unit_action.objective_slot)
        movement = int(Movement(unit_action.movement))
        x, y = (float(value) for value in unit_action.placement)
        self._activation_requested += 1
        self._activation_requested_by_unit[slot] += 1
        by_type["requested"] += 1
        self._target_activation[objective_slot] += 1
        self._target_by_type[type_key]["activation_selected_by_slot"][objective_slot] += 1
        self._record_effective_target(type_key, objective_slot, movement)
        self._record_movement(type_key, movement, activation=True)
        self._placement_requested.add(x, y)
        self._placement_requested_by_type[type_key].add(x, y)

        if accepted:
            self._activation_accepted += 1
            self._activation_accepted_by_unit[slot] += 1
            by_type["accepted"] += 1
            self._target_activation_accepted[objective_slot] += 1
            self._placement_accepted.add(x, y)
            self._placement_accepted_by_type[type_key].add(x, y)
            if self._activation_accepted_step_by_unit[slot] is not None:
                raise ValueError(f"unit slot {slot} has multiple accepted activations")
            self._activation_accepted_step_by_unit[slot] = int(step)
            self._accepted_placements.append(
                {
                    "step": int(step),
                    "unit_slot": int(slot),
                    "unit_id": int(self.unit_ids[slot]),
                    "unit_type": int(self.unit_types[slot]),
                    "x": x,
                    "y": y,
                    "objective_slot": objective_slot,
                    "objective_id": self.objective_ids[objective_slot],
                }
            )
        else:
            self._activation_rejected += 1
            self._activation_rejected_by_unit[slot] += 1
            by_type["rejected"] += 1

    def _observe_active(
        self,
        state: UnitControlState,
        unit_action: UnitAction,
        type_key: str,
    ) -> None:
        self._retarget_decisions += 1
        movement = int(Movement(unit_action.movement))
        if unit_action.retarget == BinaryChoice.YES:
            objective_slot = int(unit_action.objective_slot)
            self._retarget_selected += 1
            self._target_retarget[objective_slot] += 1
            self._target_by_type[type_key]["retarget_selected_by_slot"][
                objective_slot
            ] += 1
        else:
            self._retarget_kept += 1
            objective_slot = int(state.current_objective_slot)

        self._record_effective_target(type_key, objective_slot, movement)
        self._record_movement(type_key, movement, activation=False)
        self._movement_switch_opportunities += 1
        self._movement_switches += int(movement != int(state.last_movement))

    def _record_effective_target(
        self, type_key: str, objective_slot: int, movement: int
    ) -> None:
        if 0 <= objective_slot < self.space.objective_count:
            self._target_effective[objective_slot] += 1
            self._target_by_type[type_key]["effective_by_slot"][objective_slot] += 1
            self._movement_by_objective[objective_slot][movement] += 1
        else:
            self._target_unassigned_effective += 1

    def _record_movement(
        self, type_key: str, movement: int, *, activation: bool
    ) -> None:
        self._movement_selected[movement] += 1
        if activation:
            self._movement_activation[movement] += 1
        else:
            self._movement_active[movement] += 1
        self._movement_by_type[type_key][movement] += 1

    def _observe_sensor(
        self,
        mask: JointActionMask,
        action: JointAction,
        trace: JointPolicyTrace,
        accepted_slots: set[int],
    ) -> None:
        if mask.shared_sensor_max_requests <= 0:
            return
        self._sensor_head_active_steps += 1
        sensor_trace = trace.shared_sensor
        if sensor_trace is None:
            raise ValueError("active sensor branch is missing its policy trace")
        self._sensor_stop_selections += int(0 in sensor_trace.tokens)
        for slot in action.shared_sensor.requester_slots:
            slot = int(slot)
            type_key = self._type_keys[slot]
            accepted = slot in accepted_slots
            self._sensor_requested += 1
            self._sensor_requested_by_unit[slot] += 1
            self._sensor_by_type[type_key]["requested"] += 1
            if accepted:
                self._sensor_accepted += 1
                self._sensor_accepted_by_unit[slot] += 1
                self._sensor_by_type[type_key]["accepted"] += 1
            else:
                self._sensor_rejected += 1
                self._sensor_rejected_by_unit[slot] += 1
                self._sensor_by_type[type_key]["rejected"] += 1

    def finalize(self) -> dict[str, Any]:
        """Return an idempotent tree containing only JSON-native values."""

        self._finalized = True
        target_selected = [
            self._target_activation[slot] + self._target_retarget[slot]
            for slot in range(self.space.objective_count)
        ]
        target_selected_total = sum(target_selected)
        if target_selected_total:
            dominant_slot = max(
                range(self.space.objective_count),
                key=lambda slot: target_selected[slot],
            )
            dominant_fraction = _rate(
                target_selected[dominant_slot], target_selected_total
            )
        else:
            dominant_slot = -1
            dominant_fraction = 0.0

        movement_total = sum(self._movement_selected)
        movement_fractions = [
            _rate(value, movement_total) for value in self._movement_selected
        ]
        max_sensor_requests = max(self._sensor_requested_by_unit, default=0)
        unique_sensor_requesters = sum(
            value > 0 for value in self._sensor_requested_by_unit
        )

        placement_requested = self._placement_requested.as_dict()
        placement_accepted = self._placement_accepted.as_dict()
        result: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "steps_observed": int(self._steps_observed),
            "activation": {
                "decisions": int(self._activation_decisions),
                "wait": int(self._activation_wait),
                "requested": int(self._activation_requested),
                "accepted": int(self._activation_accepted),
                "rejected": int(self._activation_rejected),
                "yes_rate": _rate(
                    self._activation_requested, self._activation_decisions
                ),
                "acceptance_rate": _rate(
                    self._activation_accepted, self._activation_requested
                ),
                "wait_by_unit": [int(value) for value in self._activation_wait_by_unit],
                "requested_by_unit": [
                    int(value) for value in self._activation_requested_by_unit
                ],
                "accepted_by_unit": [
                    int(value) for value in self._activation_accepted_by_unit
                ],
                "rejected_by_unit": [
                    int(value) for value in self._activation_rejected_by_unit
                ],
                "accepted_step_by_unit": list(self._activation_accepted_step_by_unit),
                "by_unit_type": {
                    key: {name: int(value) for name, value in counts.items()}
                    for key, counts in self._activation_by_type.items()
                },
            },
            "target": {
                "objective_ids": list(self.objective_ids),
                "activation_selected_by_slot": [
                    int(value) for value in self._target_activation
                ],
                "activation_accepted_by_slot": [
                    int(value) for value in self._target_activation_accepted
                ],
                "retarget_selected_by_slot": [
                    int(value) for value in self._target_retarget
                ],
                "selected_by_slot": [int(value) for value in target_selected],
                "effective_by_slot": [
                    int(value) for value in self._target_effective
                ],
                "unassigned_effective": int(self._target_unassigned_effective),
                "retarget_decisions": int(self._retarget_decisions),
                "retarget_selected": int(self._retarget_selected),
                "retarget_kept": int(self._retarget_kept),
                "retarget_yes_rate": _rate(
                    self._retarget_selected, self._retarget_decisions
                ),
                "dominant_slot": int(dominant_slot),
                "dominant_objective_id": (
                    self.objective_ids[dominant_slot] if dominant_slot >= 0 else None
                ),
                "dominant_fraction": dominant_fraction,
                "by_unit_type": {
                    key: {
                        name: [int(value) for value in counts]
                        for name, counts in distributions.items()
                    }
                    for key, distributions in self._target_by_type.items()
                },
            },
            "placement": {
                "coordinates": "normalized [-1, 1]",
                "requested": placement_requested,
                "accepted": placement_accepted,
                "accepted_records": [dict(record) for record in self._accepted_placements],
                "requested_by_unit_type": {
                    key: values.as_dict()
                    for key, values in self._placement_requested_by_type.items()
                },
                "accepted_by_unit_type": {
                    key: values.as_dict()
                    for key, values in self._placement_accepted_by_type.items()
                },
            },
            "movement": {
                "labels": list(_MOVEMENT_LABELS),
                "selected": [int(value) for value in self._movement_selected],
                "fractions": movement_fractions,
                "on_activation": [
                    int(value) for value in self._movement_activation
                ],
                "while_active": [int(value) for value in self._movement_active],
                "switch_opportunities": int(self._movement_switch_opportunities),
                "switches": int(self._movement_switches),
                "switch_rate": _rate(
                    self._movement_switches,
                    self._movement_switch_opportunities,
                ),
                "by_unit_type": {
                    key: [int(value) for value in counts]
                    for key, counts in self._movement_by_type.items()
                },
                "by_objective_slot": [
                    [int(value) for value in counts]
                    for counts in self._movement_by_objective
                ],
            },
            "sensor": {
                "head_active_steps": int(self._sensor_head_active_steps),
                "stop_selections": int(self._sensor_stop_selections),
                "stop_rate": _rate(
                    self._sensor_stop_selections,
                    self._sensor_head_active_steps,
                ),
                "requested": int(self._sensor_requested),
                "accepted": int(self._sensor_accepted),
                "rejected": int(self._sensor_rejected),
                "acceptance_rate": _rate(
                    self._sensor_accepted, self._sensor_requested
                ),
                "unique_requesters": int(unique_sensor_requesters),
                "max_requests_by_one_unit": int(max_sensor_requests),
                "requested_by_unit": [
                    int(value) for value in self._sensor_requested_by_unit
                ],
                "accepted_by_unit": [
                    int(value) for value in self._sensor_accepted_by_unit
                ],
                "rejected_by_unit": [
                    int(value) for value in self._sensor_rejected_by_unit
                ],
                "by_unit_type": {
                    key: {name: int(value) for name, value in counts.items()}
                    for key, counts in self._sensor_by_type.items()
                },
            },
        }
        result["csv_scalars"] = {
            "activation_yes_rate": result["activation"]["yes_rate"],
            "activation_rejected": result["activation"]["rejected"],
            "retarget_yes_rate": result["target"]["retarget_yes_rate"],
            "dominant_objective_slot": result["target"]["dominant_slot"],
            "dominant_objective_fraction": result["target"]["dominant_fraction"],
            "placement_edge_fraction": placement_requested["any_edge_fraction"],
            "movement_negative_fraction": movement_fractions[0],
            "movement_neutral_fraction": movement_fractions[1],
            "movement_positive_fraction": movement_fractions[2],
            "movement_switch_rate": result["movement"]["switch_rate"],
            "sensor_stop_rate": result["sensor"]["stop_rate"],
            "sensor_rejected": result["sensor"]["rejected"],
        }
        return result


__all__ = ["EpisodeActionDiagnostics"]
