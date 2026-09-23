"""Pure plan validator: PASS or REJECT, never a repair.

The validator checks structural bindings (ids, step ranges, legality of the
information referenced, contradictions, satellite budget) and nothing else.
A legal-but-foolish plan (e.g. firing everything at empty ocean, or pairing
weapons with targets they cannot damage) must PASS -- effectiveness is the
model's problem, not the validator's.  The plan object is treated as read-only
in the strongest sense: validators never assign to it, and callers can verify
``plan_before == plan_after`` at any time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .plan_schema import (
    AtStepTrigger,
    BattlePlan,
    GlobalRule,
    LaunchedStepsAgoTrigger,
    NewDetectionTrigger,
    PlatformPlan,
    RetargetAction,
    SatelliteRequestAction,
    TargetRef,
)


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    errors: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "errors": list(self.errors),
        }


def _target_ref_error(
    ref: TargetRef,
    path: str,
    *,
    known_entity_ids: Iterable[int],
    allow_event_entity: bool,
) -> str | None:
    if ref.mode == "entity":
        if ref.entity_id not in set(known_entity_ids):
            return (
                f"{path}: references entity {ref.entity_id} which the model "
                "does not legally know at planning time"
            )
        return None
    if ref.mode == "coordinate":
        if ref.lon is None or ref.lat is None:
            return f"{path}: coordinate reference is missing lon/lat"
        if not (math.isfinite(ref.lon) and math.isfinite(ref.lat)):
            return f"{path}: coordinate values must be finite"
        if not (-180.0 <= ref.lon <= 180.0 and -90.0 <= ref.lat <= 90.0):
            return f"{path}: coordinate values are outside lon/lat ranges"
        return None
    if ref.mode == "event_entity":
        if not allow_event_entity:
            return (
                f"{path}: event_entity is only legal inside an action of a "
                "new_detection rule"
            )
        return None
    return f"{path}: unknown target mode {ref.mode!r}"


def validate_plan(
    plan: BattlePlan,
    *,
    controlled_platform_ids: Iterable[int],
    known_entity_ids: Iterable[int],
    max_steps: int,
    satellite_max_use_count: int,
) -> ValidationReport:
    """Run every legality check and return a report. The plan is not mutated."""

    errors: list[str] = []
    controlled = set(int(value) for value in controlled_platform_ids)
    known = set(int(value) for value in known_entity_ids)
    step_upper = int(max_steps)

    seen_platform_ids: set[int] = set()
    launch_step_by_platform: dict[int, int | None] = {}

    # ---------------- per-platform checks ----------------
    for platform in plan.platforms:
        pid = platform.platform_id
        label = f"platform {pid}"
        if pid not in controlled:
            errors.append(
                f"{label}: not a controllable red platform of type "
                "21000/21001/21002"
            )
            continue
        if pid in seen_platform_ids:
            errors.append(f"{label}: duplicate platform plan")
            continue
        seen_platform_ids.add(pid)

        if platform.launch.mode == "at_step":
            if platform.launch.step is None or not (
                0 <= platform.launch.step < step_upper
            ):
                errors.append(
                    f"{label}: launch step must be within [0, {step_upper - 1}]"
                )
                launch_step_by_platform[pid] = None
            else:
                launch_step_by_platform[pid] = platform.launch.step
            if platform.initial_target is None:
                errors.append(
                    f"{label}: launch at_step requires initial_target"
                )
            else:
                problem = _target_ref_error(
                    platform.initial_target,
                    f"{label}.initial_target",
                    known_entity_ids=known,
                    allow_event_entity=False,
                )
                if problem:
                    errors.append(problem)
        else:  # never
            launch_step_by_platform[pid] = None
            if platform.initial_target is not None:
                errors.append(
                    f"{label}: launch=never must not carry an initial_target"
                )
            if platform.retarget_orders:
                errors.append(
                    f"{label}: retarget orders are contradictory for a "
                    "platform that never launches"
                )

        retarget_steps: set[int] = set()
        for order in platform.retarget_orders:
            path = f"{label}.retarget@{order.step}"
            if not 0 <= order.step < step_upper:
                errors.append(
                    f"{path}: step must be within [0, {step_upper - 1}]"
                )
            if order.step in retarget_steps:
                errors.append(
                    f"{path}: another retarget for the same platform is "
                    "already scheduled at this step"
                )
            retarget_steps.add(order.step)
            launch_at = launch_step_by_platform.get(pid)
            if (
                launch_at is not None
                and order.step < launch_at
            ):
                errors.append(
                    f"{path}: retarget before the platform's launch step "
                    f"{launch_at} cannot execute"
                )
            problem = _target_ref_error(
                order.target,
                path,
                known_entity_ids=known,
                allow_event_entity=False,
            )
            if problem:
                errors.append(problem)

        satellite_steps: set[int] = set()
        for request in platform.satellite_steps:
            if not 0 <= request.step < step_upper:
                errors.append(
                    f"{label}.satellite@{request.step}: step must be within "
                    f"[0, {step_upper - 1}]"
                )
            if request.step in satellite_steps:
                errors.append(
                    f"{label}.satellite@{request.step}: duplicate satellite "
                    "request step for this platform"
                )
            satellite_steps.add(request.step)

    # ---------------- global rules ----------------
    rule_ids: set[str] = set()
    scheduled_satellite_steps: dict[int, int] = {}  # step -> count
    theoretical_satellite_uses = 0

    for rule in plan.global_rules:
        label = f"rule {rule.rule_id!r}"
        if rule.rule_id in rule_ids:
            errors.append(f"{label}: duplicate rule_id")
            continue
        rule_ids.add(rule.rule_id)
        if not rule.actions:
            errors.append(f"{label}: has no actions")

        trigger = rule.trigger
        event_bound = isinstance(trigger, NewDetectionTrigger)
        if isinstance(trigger, AtStepTrigger):
            if not 0 <= trigger.step < step_upper:
                errors.append(
                    f"{label}.trigger: step must be within "
                    f"[0, {step_upper - 1}]"
                )
        elif isinstance(trigger, LaunchedStepsAgoTrigger):
            if trigger.platform_id not in controlled:
                errors.append(
                    f"{label}.trigger: platform {trigger.platform_id} is not "
                    "controllable"
                )
            else:
                launch_at = launch_step_by_platform.get(trigger.platform_id)
                if launch_at is None:
                    errors.append(
                        f"{label}.trigger: platform {trigger.platform_id} "
                        "never launches, so a launch-relative trigger cannot "
                        "fire"
                    )
                elif launch_at + trigger.steps >= step_upper:
                    errors.append(
                        f"{label}.trigger: fires at step "
                        f"{launch_at + trigger.steps} beyond the episode end"
                    )

        for index, action in enumerate(rule.actions):
            path = f"{label}.actions[{index}]"
            if action.platform_id not in controlled:
                errors.append(
                    f"{path}: platform {action.platform_id} is not controllable"
                )
                continue
            if isinstance(action, RetargetAction):
                problem = _target_ref_error(
                    action.target,
                    f"{path}.target",
                    known_entity_ids=known,
                    allow_event_entity=event_bound,
                )
                if problem:
                    errors.append(problem)
                launch_at = launch_step_by_platform.get(action.platform_id)
                if launch_at is None:
                    # Platform never launches: a retarget can never execute.
                    errors.append(
                        f"{path}: platform {action.platform_id} never "
                        "launches, so this retarget cannot execute"
                    )
                elif (
                    isinstance(trigger, AtStepTrigger)
                    and trigger.step < launch_at
                ):
                    errors.append(
                        f"{path}: retarget at step {trigger.step} precedes "
                        f"the platform's launch step {launch_at}"
                    )
            elif isinstance(action, SatelliteRequestAction):
                theoretical_satellite_uses += 1
                if isinstance(trigger, AtStepTrigger):
                    step = trigger.step
                    count = scheduled_satellite_steps.get(step, 0) + 1
                    scheduled_satellite_steps[step] = count
                    if count > 1:
                        errors.append(
                            f"{path}: more than one satellite request is "
                            f"scheduled at step {step}; the team executes at "
                            "most one per step"
                        )

    # Platform-scheduled satellite steps join the same per-step budget.
    for platform in plan.platforms:
        for request in platform.satellite_steps:
            theoretical_satellite_uses += 1
            count = scheduled_satellite_steps.get(request.step, 0) + 1
            scheduled_satellite_steps[request.step] = count
            if count > 1:
                errors.append(
                    f"platform {platform.platform_id}.satellite@"
                    f"{request.step}: more than one satellite request is "
                    f"scheduled at step {request.step}; the team executes at "
                    "most one per step"
                )

    if theoretical_satellite_uses > int(satellite_max_use_count):
        errors.append(
            f"plan schedules {theoretical_satellite_uses} satellite uses in "
            f"total, exceeding the team budget of {satellite_max_use_count}"
        )

    if not plan.platforms:
        errors.append("plan contains no platform entries")

    return ValidationReport(passed=not errors, errors=tuple(errors))
