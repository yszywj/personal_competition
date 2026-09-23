"""Strict V0 plan schema for the LLM one-shot planner.

The schema is intentionally expressive: per-platform launch/retarget/satellite
schedules plus event-triggered global rules with three target-reference modes
(``entity`` / ``coordinate`` / ``event_entity``).  ``motion`` exists in the
schema but V0 accepts only ``"straight"``; any other value is a structural
error and the plan is INVALID.  Nothing here is ever auto-corrected.

This module is pure Python (no simulator imports) so it can be unit-tested
without the native runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

PLAN_VERSION = "v0"
SUPPORTED_MOTION_MODES = ("straight",)
TARGET_REF_MODES = ("entity", "coordinate", "event_entity")
LAUNCH_MODES = ("at_step", "never")
TRIGGER_TYPES = ("at_step", "new_detection", "launched_steps_ago")
RULE_ACTION_TYPES = ("retarget", "satellite_request")
# Entity types that can legally appear in a red platform's detectInfo and are
# meaningful as detection-event triggers.
EVENT_ENTITY_TYPES = (9400, 9500, 9600, 24000)


class PlanParseError(ValueError):
    """Collected structural problems found while parsing one raw plan."""


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a Mapping or an attribute-style object."""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclass(frozen=True)
class TargetRef:
    """A reference to an attackable point.

    ``entity``      -- a legally known blue entity id (9400/9600 from the init
                       catalogue, or any id that later appears in red
                       detectInfo).
    ``coordinate``  -- a direct lon/lat chosen by the model.
    ``event_entity``-- only legal inside an action of a ``new_detection``
                       rule; binds to the entity that fired that event.
    """

    mode: str
    entity_id: int | None = None
    lon: float | None = None
    lat: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"mode": self.mode}
        if self.entity_id is not None:
            result["entity_id"] = int(self.entity_id)
        if self.lon is not None:
            result["lon"] = float(self.lon)
        if self.lat is not None:
            result["lat"] = float(self.lat)
        return result


@dataclass(frozen=True)
class LaunchSpec:
    mode: str  # "at_step" | "never"
    step: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"mode": self.mode}
        if self.step is not None:
            result["step"] = int(self.step)
        return result


@dataclass(frozen=True)
class RetargetOrder:
    step: int
    target: TargetRef

    def to_dict(self) -> dict[str, Any]:
        return {"step": int(self.step), "target": self.target.to_dict()}


@dataclass(frozen=True)
class SatelliteStep:
    step: int

    def to_dict(self) -> dict[str, Any]:
        return {"step": int(self.step)}


@dataclass(frozen=True)
class PlatformPlan:
    platform_id: int
    launch: LaunchSpec
    initial_target: TargetRef | None
    retarget_orders: tuple[RetargetOrder, ...] = ()
    satellite_steps: tuple[SatelliteStep, ...] = ()
    motion: str = "straight"

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform_id": int(self.platform_id),
            "launch": self.launch.to_dict(),
            "initial_target": (
                self.initial_target.to_dict() if self.initial_target else None
            ),
            "retarget_orders": [item.to_dict() for item in self.retarget_orders],
            "satellite_steps": [item.to_dict() for item in self.satellite_steps],
            "motion": self.motion,
        }


@dataclass(frozen=True)
class AtStepTrigger:
    step: int

    @property
    def type(self) -> str:
        return "at_step"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "at_step", "step": int(self.step)}


@dataclass(frozen=True)
class NewDetectionTrigger:
    entity_type: int
    occurrence: int

    @property
    def type(self) -> str:
        return "new_detection"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "new_detection",
            "entity_type": int(self.entity_type),
            "occurrence": int(self.occurrence),
        }


@dataclass(frozen=True)
class LaunchedStepsAgoTrigger:
    platform_id: int
    steps: int

    @property
    def type(self) -> str:
        return "launched_steps_ago"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "launched_steps_ago",
            "platform_id": int(self.platform_id),
            "steps": int(self.steps),
        }


Trigger = AtStepTrigger | NewDetectionTrigger | LaunchedStepsAgoTrigger


@dataclass(frozen=True)
class RetargetAction:
    platform_id: int
    target: TargetRef

    @property
    def type(self) -> str:
        return "retarget"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "retarget",
            "platform_id": int(self.platform_id),
            "target": self.target.to_dict(),
        }


@dataclass(frozen=True)
class SatelliteRequestAction:
    platform_id: int

    @property
    def type(self) -> str:
        return "satellite_request"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "satellite_request",
            "platform_id": int(self.platform_id),
        }


RuleAction = RetargetAction | SatelliteRequestAction


@dataclass(frozen=True)
class GlobalRule:
    rule_id: str
    trigger: Trigger
    actions: tuple[RuleAction, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "trigger": self.trigger.to_dict(),
            "actions": [item.to_dict() for item in self.actions],
        }


@dataclass(frozen=True)
class BattlePlan:
    plan_version: str
    platforms: tuple[PlatformPlan, ...]
    global_rules: tuple[GlobalRule, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_version": self.plan_version,
            "platforms": [item.to_dict() for item in self.platforms],
            "global_rules": [item.to_dict() for item in self.global_rules],
        }


# ---------------------------------------------------------------------------
# Parsing helpers.  Structural checks only; semantic validation (ids, ranges,
# contradictions) lives in validator.py.
# ---------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_mapping(value: Any, path: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{path}: expected an object, got {type(value).__name__}")
        return {}
    return value


def _require_int(
    container: Mapping[str, Any],
    name: str,
    path: str,
    errors: list[str],
    *,
    minimum: int | None = None,
) -> int | None:
    if name not in container:
        errors.append(f"{path}.{name}: required integer is missing")
        return None
    value = container[name]
    if not _is_int(value):
        errors.append(f"{path}.{name}: expected an integer, got {value!r}")
        return None
    if minimum is not None and value < minimum:
        errors.append(f"{path}.{name}: expected >= {minimum}, got {value}")
        return None
    return int(value)


def _require_number(
    container: Mapping[str, Any],
    name: str,
    path: str,
    errors: list[str],
) -> float | None:
    if name not in container:
        errors.append(f"{path}.{name}: required number is missing")
        return None
    value = container[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{path}.{name}: expected a number, got {value!r}")
        return None
    number = float(value)
    if not math.isfinite(number):
        errors.append(f"{path}.{name}: value must be finite, got {value!r}")
        return None
    return number


def _parse_target_ref(value: Any, path: str, errors: list[str]) -> TargetRef | None:
    data = _require_mapping(value, path, errors)
    if not data:
        return None
    mode = data.get("mode")
    if mode not in TARGET_REF_MODES:
        errors.append(
            f"{path}.mode: must be one of {list(TARGET_REF_MODES)}, got {mode!r}"
        )
        return None
    if mode == "entity":
        entity_id = _require_int(data, "entity_id", path, errors)
        if entity_id is None:
            return None
        return TargetRef(mode="entity", entity_id=entity_id)
    if mode == "coordinate":
        lon = _require_number(data, "lon", path, errors)
        lat = _require_number(data, "lat", path, errors)
        if lon is None or lat is None:
            return None
        return TargetRef(mode="coordinate", lon=lon, lat=lat)
    # event_entity carries no parameters of its own; it binds to the rule.
    return TargetRef(mode="event_entity")


def _parse_launch(value: Any, path: str, errors: list[str]) -> LaunchSpec | None:
    data = _require_mapping(value, path, errors)
    if not data:
        return None
    mode = data.get("mode")
    if mode not in LAUNCH_MODES:
        errors.append(
            f"{path}.mode: must be one of {list(LAUNCH_MODES)}, got {mode!r}"
        )
        return None
    if mode == "never":
        return LaunchSpec(mode="never")
    step = _require_int(data, "step", path, errors, minimum=0)
    if step is None:
        return None
    return LaunchSpec(mode="at_step", step=step)


def _parse_trigger(value: Any, path: str, errors: list[str]) -> Trigger | None:
    data = _require_mapping(value, path, errors)
    if not data:
        return None
    kind = data.get("type")
    if kind not in TRIGGER_TYPES:
        errors.append(
            f"{path}.type: must be one of {list(TRIGGER_TYPES)}, got {kind!r}"
        )
        return None
    if kind == "at_step":
        step = _require_int(data, "step", path, errors, minimum=0)
        return AtStepTrigger(step=step) if step is not None else None
    if kind == "new_detection":
        entity_type = _require_int(data, "entity_type", path, errors)
        occurrence = _require_int(data, "occurrence", path, errors, minimum=1)
        if entity_type is None or occurrence is None:
            return None
        if entity_type not in EVENT_ENTITY_TYPES:
            errors.append(
                f"{path}.entity_type: must be one of {list(EVENT_ENTITY_TYPES)}, "
                f"got {entity_type}"
            )
            return None
        return NewDetectionTrigger(
            entity_type=entity_type, occurrence=occurrence
        )
    steps = _require_int(data, "steps", path, errors, minimum=0)
    platform_id = _require_int(data, "platform_id", path, errors)
    if steps is None or platform_id is None:
        return None
    return LaunchedStepsAgoTrigger(platform_id=platform_id, steps=steps)


def _parse_rule_action(value: Any, path: str, errors: list[str]) -> RuleAction | None:
    data = _require_mapping(value, path, errors)
    if not data:
        return None
    kind = data.get("type")
    if kind not in RULE_ACTION_TYPES:
        errors.append(
            f"{path}.type: must be one of {list(RULE_ACTION_TYPES)}, got {kind!r}"
        )
        return None
    platform_id = _require_int(data, "platform_id", path, errors)
    if platform_id is None:
        return None
    if kind == "satellite_request":
        return SatelliteRequestAction(platform_id=platform_id)
    target = _parse_target_ref(data.get("target"), f"{path}.target", errors)
    if target is None:
        return None
    return RetargetAction(platform_id=platform_id, target=target)


def parse_plan(raw: Any) -> tuple[BattlePlan | None, list[str]]:
    """Strictly parse a raw plan payload.

    Returns ``(plan, errors)``.  ``plan`` is ``None`` when any structural error
    exists.  The input is never mutated.
    """

    errors: list[str] = []
    data = _require_mapping(raw, "plan", errors)
    if not data:
        return None, errors

    version = data.get("plan_version")
    if version != PLAN_VERSION:
        errors.append(f"plan.plan_version: must be {PLAN_VERSION!r}, got {version!r}")

    raw_platforms = data.get("platforms")
    if not isinstance(raw_platforms, list) or not raw_platforms:
        errors.append("plan.platforms: must be a non-empty list of platform plans")
        raw_platforms = []
    platforms: list[PlatformPlan] = []
    for index, item in enumerate(raw_platforms):
        path = f"plan.platforms[{index}]"
        entry = _require_mapping(item, path, errors)
        if not entry:
            continue
        platform_id = _require_int(entry, "platform_id", path, errors)
        launch = _parse_launch(entry.get("launch"), f"{path}.launch", errors)
        motion = entry.get("motion", "straight")
        if motion not in SUPPORTED_MOTION_MODES:
            errors.append(
                f"{path}.motion: V0 supports only {list(SUPPORTED_MOTION_MODES)}, "
                f"got {motion!r}"
            )
            motion = "straight"
        initial_target = None
        if entry.get("initial_target") is not None:
            initial_target = _parse_target_ref(
                entry.get("initial_target"), f"{path}.initial_target", errors
            )
        retarget_orders: list[RetargetOrder] = []
        raw_orders = entry.get("retarget_orders", [])
        if raw_orders is None:
            raw_orders = []
        if not isinstance(raw_orders, list):
            errors.append(f"{path}.retarget_orders: must be a list")
        else:
            for order_index, order in enumerate(raw_orders):
                order_path = f"{path}.retarget_orders[{order_index}]"
                order_data = _require_mapping(order, order_path, errors)
                if not order_data:
                    continue
                step = _require_int(order_data, "step", order_path, errors, minimum=0)
                target = _parse_target_ref(
                    order_data.get("target"), f"{order_path}.target", errors
                )
                if step is not None and target is not None:
                    retarget_orders.append(RetargetOrder(step=step, target=target))
        satellite_steps: list[SatelliteStep] = []
        raw_satellites = entry.get("satellite_steps", [])
        if raw_satellites is None:
            raw_satellites = []
        if not isinstance(raw_satellites, list):
            errors.append(f"{path}.satellite_steps: must be a list")
        else:
            for sat_index, sat in enumerate(raw_satellites):
                sat_path = f"{path}.satellite_steps[{sat_index}]"
                sat_data = _require_mapping(sat, sat_path, errors)
                if not sat_data:
                    continue
                step = _require_int(sat_data, "step", sat_path, errors, minimum=0)
                if step is not None:
                    satellite_steps.append(SatelliteStep(step=step))
        if platform_id is not None and launch is not None:
            platforms.append(
                PlatformPlan(
                    platform_id=platform_id,
                    launch=launch,
                    initial_target=initial_target,
                    retarget_orders=tuple(retarget_orders),
                    satellite_steps=tuple(satellite_steps),
                    motion=motion,
                )
            )

    raw_rules = data.get("global_rules", [])
    if raw_rules is None:
        raw_rules = []
    if not isinstance(raw_rules, list):
        errors.append("plan.global_rules: must be a list")
        raw_rules = []
    rules: list[GlobalRule] = []
    for rule_index, item in enumerate(raw_rules):
        path = f"plan.global_rules[{rule_index}]"
        entry = _require_mapping(item, path, errors)
        if not entry:
            continue
        rule_id = entry.get("rule_id")
        if not isinstance(rule_id, str) or not rule_id:
            errors.append(f"{path}.rule_id: must be a non-empty string")
            continue
        trigger = _parse_trigger(entry.get("trigger"), f"{path}.trigger", errors)
        raw_actions = entry.get("actions", [])
        if raw_actions is None:
            raw_actions = []
        if not isinstance(raw_actions, list) or not raw_actions:
            errors.append(f"{path}.actions: must be a non-empty list")
            raw_actions = []
        actions: list[RuleAction] = []
        for action_index, action in enumerate(raw_actions):
            parsed_action = _parse_rule_action(
                action, f"{path}.actions[{action_index}]", errors
            )
            if parsed_action is not None:
                actions.append(parsed_action)
        if trigger is not None:
            rules.append(
                GlobalRule(
                    rule_id=rule_id,
                    trigger=trigger,
                    actions=tuple(actions),
                )
            )

    if errors:
        return None, errors
    return (
        BattlePlan(
            plan_version=PLAN_VERSION,
            platforms=tuple(platforms),
            global_rules=tuple(rules),
        ),
        [],
    )
