"""Pure reward-credit helpers for the branch-aware joint policy.

This module deliberately has no simulator dependency.  Objective health is an
episode-outcome input used only to construct a training target; callers must
not append it to actor observations.  Stable objective and unit slots are used
instead of external entity IDs so the functions also work with hidden targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


def _non_negative_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _slot(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class ObjectiveDamageRecord:
    """End-of-episode score inputs for one objective slot.

    ``final_health`` may be below zero because some simulators report
    over-damage.  The resulting damage fraction is clipped to ``[0, 1]`` in
    the same way as the competition score.
    """

    objective_slot: int
    weight: float
    initial_health: float
    final_health: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "objective_slot", _slot("objective_slot", self.objective_slot)
        )
        weight = float(self.weight)
        initial_health = float(self.initial_health)
        final_health = float(self.final_health)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("objective weight must be finite and positive")
        if not math.isfinite(initial_health) or initial_health <= 0.0:
            raise ValueError("initial health must be finite and positive")
        if not math.isfinite(final_health):
            raise ValueError("final health must be finite")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "initial_health", initial_health)
        object.__setattr__(self, "final_health", final_health)

    @property
    def damage_fraction(self) -> float:
        return max(
            0.0,
            min(1.0, (self.initial_health - self.final_health) / self.initial_health),
        )


@dataclass(frozen=True, slots=True)
class ObjectiveContribution:
    """One objective's normalized contribution to the official 0--1 score."""

    objective_slot: int
    damage_fraction: float
    normalized_contribution: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "objective_slot", _slot("objective_slot", self.objective_slot)
        )
        damage = float(self.damage_fraction)
        contribution = float(self.normalized_contribution)
        if not math.isfinite(damage) or not 0.0 <= damage <= 1.0:
            raise ValueError("damage_fraction must be finite and in [0, 1]")
        if not math.isfinite(contribution) or not 0.0 <= contribution <= 1.0:
            raise ValueError(
                "normalized_contribution must be finite and in [0, 1]"
            )
        object.__setattr__(self, "damage_fraction", damage)
        object.__setattr__(self, "normalized_contribution", contribution)


@dataclass(frozen=True, slots=True)
class ObjectiveParticipation:
    """Controller-owned responsibility evidence for one unit/objective pair.

    ``effective_responsibility`` should be derived from legal own-unit history,
    for example time spent inside an effective engagement radius.  A caller
    can provide assignment duration as ``fallback_responsibility``.  Fallback
    is considered only when an objective has no participant meeting the
    effective threshold, preventing weak evidence from diluting real
    participants' credit.
    """

    unit_slot: int
    objective_slot: int
    effective_responsibility: float = 0.0
    fallback_responsibility: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_slot", _slot("unit_slot", self.unit_slot))
        object.__setattr__(
            self, "objective_slot", _slot("objective_slot", self.objective_slot)
        )
        object.__setattr__(
            self,
            "effective_responsibility",
            _non_negative_finite(
                "effective_responsibility", self.effective_responsibility
            ),
        )
        object.__setattr__(
            self,
            "fallback_responsibility",
            _non_negative_finite(
                "fallback_responsibility", self.fallback_responsibility
            ),
        )


@dataclass(frozen=True, slots=True)
class LocalCreditAllocation:
    """Auditable result of conservative target-level credit allocation."""

    credit_by_unit: tuple[float, ...]
    contribution_by_objective: tuple[tuple[int, float], ...]
    allocated_by_objective: tuple[tuple[int, float], ...]
    unallocated_by_objective: tuple[tuple[int, float], ...]
    fallback_objective_slots: tuple[int, ...]

    @property
    def total_contribution(self) -> float:
        return float(sum(value for _, value in self.contribution_by_objective))

    @property
    def total_allocated(self) -> float:
        return float(sum(self.credit_by_unit))

    @property
    def total_unallocated(self) -> float:
        return float(sum(value for _, value in self.unallocated_by_objective))


def compute_objective_damage_contributions(
    records: Iterable[ObjectiveDamageRecord],
) -> tuple[ObjectiveContribution, ...]:
    """Reproduce each objective's weighted share of the official score.

    The denominator includes every supplied objective, including undamaged
    ones.  Consequently the sum of ``normalized_contribution`` values equals
    the official weighted-damage score divided by 100.
    """

    items = tuple(records)
    if not items:
        raise ValueError("at least one objective damage record is required")
    if any(not isinstance(item, ObjectiveDamageRecord) for item in items):
        raise TypeError("records must contain ObjectiveDamageRecord values")
    slots = [item.objective_slot for item in items]
    if len(set(slots)) != len(slots):
        raise ValueError("objective damage records contain duplicate slots")
    total_weight = float(sum(item.weight for item in items))
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("total objective weight must be finite and positive")
    return tuple(
        ObjectiveContribution(
            objective_slot=item.objective_slot,
            damage_fraction=item.damage_fraction,
            normalized_contribution=(
                item.weight * item.damage_fraction / total_weight
            ),
        )
        for item in items
    )


def allocate_local_objective_credit(
    contributions: Iterable[ObjectiveContribution],
    participations: Iterable[ObjectiveParticipation],
    *,
    unit_count: int,
    minimum_effective_responsibility: float = 0.0,
) -> LocalCreditAllocation:
    """Allocate each target contribution without increasing its total mass.

    Effective responsibility is aggregated per unit and then filtered by
    ``minimum_effective_responsibility``.  When none remains, positive
    fallback responsibility is used.  If neither exists, the target's credit
    remains explicitly unallocated instead of being broadcast to unrelated
    units.
    """

    if isinstance(unit_count, bool) or not isinstance(unit_count, int) or unit_count <= 0:
        raise ValueError("unit_count must be a positive integer")
    threshold = _non_negative_finite(
        "minimum_effective_responsibility", minimum_effective_responsibility
    )
    objective_items = tuple(contributions)
    if any(not isinstance(item, ObjectiveContribution) for item in objective_items):
        raise TypeError("contributions must contain ObjectiveContribution values")
    objective_slots = [item.objective_slot for item in objective_items]
    if len(set(objective_slots)) != len(objective_slots):
        raise ValueError("objective contributions contain duplicate slots")
    known_objectives = set(objective_slots)

    effective_by_objective: dict[int, dict[int, float]] = {
        slot: {} for slot in objective_slots
    }
    fallback_by_objective: dict[int, dict[int, float]] = {
        slot: {} for slot in objective_slots
    }
    for item in participations:
        if not isinstance(item, ObjectiveParticipation):
            raise TypeError(
                "participations must contain ObjectiveParticipation values"
            )
        if item.unit_slot >= unit_count:
            raise ValueError("participation unit_slot is outside configured units")
        if item.objective_slot not in known_objectives:
            raise ValueError("participation references an unknown objective slot")
        effective = effective_by_objective[item.objective_slot]
        fallback = fallback_by_objective[item.objective_slot]
        effective[item.unit_slot] = (
            effective.get(item.unit_slot, 0.0) + item.effective_responsibility
        )
        fallback[item.unit_slot] = (
            fallback.get(item.unit_slot, 0.0) + item.fallback_responsibility
        )

    credit_by_unit = [0.0] * unit_count
    allocated_pairs: list[tuple[int, float]] = []
    unallocated_pairs: list[tuple[int, float]] = []
    fallback_slots: list[int] = []
    contribution_pairs: list[tuple[int, float]] = []
    for contribution in objective_items:
        slot = contribution.objective_slot
        amount = contribution.normalized_contribution
        contribution_pairs.append((slot, amount))
        if amount == 0.0:
            allocated_pairs.append((slot, 0.0))
            unallocated_pairs.append((slot, 0.0))
            continue

        effective = {
            unit_slot: value
            for unit_slot, value in effective_by_objective[slot].items()
            if value > 0.0 and value >= threshold
        }
        selected = effective
        if not selected:
            selected = {
                unit_slot: value
                for unit_slot, value in fallback_by_objective[slot].items()
                if value > 0.0
            }
            if selected:
                fallback_slots.append(slot)
        if not selected:
            allocated_pairs.append((slot, 0.0))
            unallocated_pairs.append((slot, amount))
            continue

        responsibility_sum = float(sum(selected.values()))
        ordered = sorted(selected.items())
        allocated = 0.0
        for index, (unit_slot, responsibility) in enumerate(ordered):
            # Assign the floating-point remainder to the final participant so
            # even a large unit set remains conservative to machine precision.
            share = (
                amount - allocated
                if index == len(ordered) - 1
                else amount * responsibility / responsibility_sum
            )
            credit_by_unit[unit_slot] += share
            allocated += share
        allocated_pairs.append((slot, allocated))
        unallocated_pairs.append((slot, max(0.0, amount - allocated)))

    result = LocalCreditAllocation(
        credit_by_unit=tuple(float(value) for value in credit_by_unit),
        contribution_by_objective=tuple(contribution_pairs),
        allocated_by_objective=tuple(allocated_pairs),
        unallocated_by_objective=tuple(unallocated_pairs),
        fallback_objective_slots=tuple(fallback_slots),
    )
    if not math.isclose(
        result.total_allocated + result.total_unallocated,
        result.total_contribution,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ArithmeticError("objective credit allocation violated conservation")
    return result


def mix_planning_rewards(
    *,
    team_score_fraction: float,
    local_credit_by_unit: Sequence[float],
    eligible_units: Sequence[bool],
    team_weight: float = 0.7,
    local_weight: float = 0.3,
) -> tuple[float, ...]:
    """Combine team outcome and conservative local credit for plan actions.

    The team outcome is broadcast only to units that made an accepted planning
    decision.  This mixed per-action learning signal is intentionally not a
    conserved global quantity; ``local_credit_by_unit`` is the conserved and
    auditable component.
    """

    team_score = float(team_score_fraction)
    if not math.isfinite(team_score) or not 0.0 <= team_score <= 1.0:
        raise ValueError("team_score_fraction must be finite and in [0, 1]")
    if len(local_credit_by_unit) != len(eligible_units) or not eligible_units:
        raise ValueError("local credits and eligibility must have equal non-zero length")
    team_mix = _non_negative_finite("team_weight", team_weight)
    local_mix = _non_negative_finite("local_weight", local_weight)
    if not math.isclose(team_mix + local_mix, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("planning reward weights must sum to 1")

    result: list[float] = []
    for local_raw, eligible_raw in zip(local_credit_by_unit, eligible_units):
        local = _non_negative_finite("local planning credit", local_raw)
        eligible = bool(eligible_raw)
        if local > 0.0 and not eligible:
            raise ValueError("an ineligible unit cannot receive local planning credit")
        result.append(team_mix * team_score + local_mix * local if eligible else 0.0)
    return tuple(result)


def objective_information_potential(
    *,
    known_objective_slots: Iterable[int],
    objective_weights: Mapping[int, float],
    scale: float = 0.015,
) -> float:
    """Return a bounded potential for legally discovered objective value."""

    potential_scale = _non_negative_finite("scale", scale)
    if not objective_weights:
        raise ValueError("objective_weights must not be empty")
    normalized_weights: dict[int, float] = {}
    for raw_slot, raw_weight in objective_weights.items():
        slot = _slot("objective weight slot", raw_slot)
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("objective weights must be finite and positive")
        normalized_weights[slot] = weight
    known = {_slot("known objective slot", value) for value in known_objective_slots}
    unknown = known - set(normalized_weights)
    if unknown:
        raise ValueError(f"known objective slots lack weights: {sorted(unknown)}")
    total_weight = float(sum(normalized_weights.values()))
    known_weight = float(sum(normalized_weights[slot] for slot in known))
    return potential_scale * known_weight / total_weight


def interceptor_track_information_potential(
    *,
    threat_age_steps: Iterable[float],
    max_track_age_steps: int,
    threat_normalizer: int,
    scale: float = 0.015,
) -> float:
    """Return a bounded potential for fresh, legally observed interceptor tracks.

    The caller supplies one age for each unique interceptor visible in the
    controller observation.  A fresh track contributes one unit and then
    decays linearly to zero at ``max_track_age_steps``.  ``threat_normalizer``
    is normally the scenario's initial interceptor count, so discovering every
    threat reaches ``scale`` without exposing hidden health or position data to
    the actor.
    """

    potential_scale = _non_negative_finite("scale", scale)
    if (
        isinstance(max_track_age_steps, bool)
        or not isinstance(max_track_age_steps, int)
        or max_track_age_steps <= 0
    ):
        raise ValueError("max_track_age_steps must be a positive integer")
    if (
        isinstance(threat_normalizer, bool)
        or not isinstance(threat_normalizer, int)
        or threat_normalizer <= 0
    ):
        raise ValueError("threat_normalizer must be a positive integer")

    freshness = 0.0
    for raw_age in threat_age_steps:
        age = float(raw_age)
        if not math.isfinite(age) or age < 0.0:
            raise ValueError("threat ages must be finite and non-negative")
        freshness += max(0.0, 1.0 - age / max_track_age_steps)
    return potential_scale * min(1.0, freshness / threat_normalizer)


def potential_difference(*, previous: float, current: float, gamma: float) -> float:
    """Compute policy-invariant potential shaping ``gamma*current-previous``."""

    previous_value = float(previous)
    current_value = float(current)
    discount = float(gamma)
    if not math.isfinite(previous_value) or not math.isfinite(current_value):
        raise ValueError("potential values must be finite")
    if not math.isfinite(discount) or not 0.0 <= discount <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    return discount * current_value - previous_value
