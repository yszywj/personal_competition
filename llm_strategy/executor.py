"""Literal executor for an accepted battle plan.

The executor only:
    * tracks which planned triggers have fired (step schedules, legal
      detection events, launch-relative timers),
    * resolves the plan's target references deterministically
      (entity -> latest legally known coordinates, coordinate -> direct,
      event_entity -> the entity that fired the triggering detection event),
    * emits the corresponding engine action rows.

It never re-decides anything: no target substitution, no re-planning, no
fallback, no lateral-acceleration commands (V0 motion is fixed "straight").
Commands whose platform is dead or whose target cannot be resolved are
recorded as execution failures and skipped.  Commands that are mutually
ambiguous at one step (several retargets for one platform, or several team
satellite requests) are recorded as ``runtime_plan_conflict`` failures and
none of the conflicting commands is executed -- the executor has no
authority to pick a winner.  The accepted plan object is treated as
immutable read-only data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

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
from .state_builder import EVENT_ENTITY_TYPES, _field, _finite

LAUNCH_ROW = 1
RETARGET_ROW = 2
SATELLITE_ROW = 3

# V0 opening-catalogue entity types (mirrors commander/state_builder policy):
# even if a future upstream _get_init_ship_observation() returned more entity
# types, the executor's target-registry seed only ever admits these.
OPENING_CATALOGUE_TYPES = frozenset({9400, 9600})


@dataclass(frozen=True)
class TrackPoint:
    lon: float
    lat: float
    time: float


class TargetRegistry:
    """Latest legally known coordinates per blue entity id.

    Seeded with the public opening catalogue (9400/9600 only, enforced here)
    and refreshed from red platforms' own ``detectInfo`` only.
    """

    def __init__(self, init_ship_observation: Mapping[str, Any]):
        self._points: dict[int, TrackPoint] = {}
        entities = _field(init_ship_observation, "entities") or {}
        items = entities.items() if isinstance(entities, Mapping) else []
        for key, entity in items:
            if entity is None:
                continue
            try:
                entity_id = int(key)
                entity_type = int(_field(entity, "type"))
            except (TypeError, ValueError):
                continue
            if entity_type not in OPENING_CATALOGUE_TYPES:
                continue
            position = _field(entity, "position") or {}
            self._points[entity_id] = TrackPoint(
                lon=_finite(_field(position, "lon")),
                lat=_finite(_field(position, "lat")),
                time=0.0,
            )

    def update_from_detections(self, detections: Sequence[Any]) -> None:
        for detection in detections:
            try:
                detected_id = int(
                    _field(detection, "entity_id", _field(detection, "id"))
                )
            except (TypeError, ValueError):
                continue
            timestamp = _finite(_field(detection, "time"), default=-1.0)
            lla = _field(detection, "lla")
            if lla is None:
                continue
            point = TrackPoint(
                lon=_finite(_field(lla, "x")), lat=_finite(_field(lla, "y")), time=timestamp
            )
            if not (math.isfinite(point.lon) and math.isfinite(point.lat)):
                continue
            existing = self._points.get(detected_id)
            if existing is None or timestamp >= existing.time:
                self._points[detected_id] = point

    def resolve(self, entity_id: int) -> TrackPoint | None:
        return self._points.get(int(entity_id))


@dataclass(frozen=True)
class PlannedCommand:
    rule_id: str
    trigger: str
    planned_action: dict[str, Any]
    kind: str  # launch | retarget | satellite_request
    platform_id: int
    target: TargetRef | None = None
    bound_entity_id: int | None = None


TraceCallback = Callable[[Mapping[str, Any]], None]


class PlanExecutor:
    """Deterministic interpreter of one accepted plan for one episode."""

    def __init__(
        self,
        plan: BattlePlan,
        *,
        init_ship_observation: Mapping[str, Any],
        controlled_platform_ids: Sequence[int],
        on_trace: TraceCallback | None = None,
    ) -> None:
        self.plan = plan
        self.controlled_platform_ids = frozenset(
            int(value) for value in controlled_platform_ids
        )
        self.registry = TargetRegistry(init_ship_observation)
        self._on_trace = on_trace
        self._platform_plan_by_id = {
            platform.platform_id: platform for platform in plan.platforms
        }
        # Detection-event bookkeeping: ordered first-seen ids per entity type.
        self._seen_by_type: dict[int, list[int]] = {}
        # rule_id -> bound entity id for fired new_detection rules.
        self._event_bindings: dict[str, int] = {}
        self._fired_rules: set[str] = set()
        self._current_step = -1
        self._emitted_launch_step: dict[int, int] = {}
        self._pending_by_platform: dict[int, list[list[float]]] = {}
        self.launched_platforms: set[int] = set()
        self.executed_commands = 0
        self.failed_commands = 0
        self.runtime_conflicts = 0

    # ------------------------------------------------------------------
    # Per-step bookkeeping
    # ------------------------------------------------------------------

    def _extract_detections(
        self, observations: Sequence[Mapping[str, Any]]
    ) -> list[Any]:
        collected: list[Any] = []
        for observation in observations:
            self_info = _field(observation, "self")
            if self_info is None:
                continue
            detect_info = _field(self_info, "detectInfo") or {}
            values = (
                detect_info.values()
                if isinstance(detect_info, Mapping)
                else list(detect_info)
            )
            for detection in values:
                collected.append(detection)
        return collected

    def _update_detection_events(
        self, detections: Sequence[Any]
    ) -> dict[int, list[tuple[int, TrackPoint]]]:
        """Register newly seen entities; return this step's new events."""

        new_entities: dict[int, list[tuple[int, TrackPoint]]] = {}
        fresh_candidates: dict[int, dict[int, TrackPoint]] = {}
        for detection in detections:
            try:
                entity_type = int(_field(detection, "entity_type", -1))
                detected_id = int(
                    _field(detection, "entity_id", _field(detection, "id"))
                )
            except (TypeError, ValueError):
                continue
            if entity_type not in EVENT_ENTITY_TYPES:
                continue
            if detected_id in self._seen_by_type.setdefault(entity_type, []):
                continue
            lla = _field(detection, "lla")
            if lla is None:
                continue
            point = TrackPoint(
                lon=_finite(_field(lla, "x")),
                lat=_finite(_field(lla, "y")),
                time=_finite(_field(detection, "time"), default=0.0),
            )
            if not (math.isfinite(point.lon) and math.isfinite(point.lat)):
                continue
            fresh_candidates.setdefault(entity_type, {})[detected_id] = point

        for entity_type, candidates in fresh_candidates.items():
            seen = self._seen_by_type.setdefault(entity_type, [])
            # Deterministic occurrence ordering within one step: ascending id.
            for detected_id in sorted(candidates):
                seen.append(detected_id)
                new_entities.setdefault(entity_type, []).append(
                    (detected_id, candidates[detected_id])
                )
        return new_entities

    def _rule_fires(
        self,
        rule: GlobalRule,
        step: int,
        new_entities: Mapping[int, Sequence[tuple[int, TrackPoint]]],
    ) -> tuple[bool, str, int | None]:
        """Return (fired, description, bound_entity_id_or_None)."""

        trigger = rule.trigger
        if rule.rule_id in self._fired_rules:
            return False, "", None
        if isinstance(trigger, AtStepTrigger):
            if step == trigger.step:
                return True, self._trigger_text(rule), None
            return False, "", None
        if isinstance(trigger, NewDetectionTrigger):
            seen = self._seen_by_type.get(trigger.entity_type, [])
            if len(seen) >= trigger.occurrence:
                entity_id = seen[trigger.occurrence - 1]
                return True, self._trigger_text(rule), entity_id
            return False, "", None
        if isinstance(trigger, LaunchedStepsAgoTrigger):
            launch_step = self._emitted_launch_step.get(trigger.platform_id)
            if launch_step is not None and step == launch_step + trigger.steps:
                return True, self._trigger_text(rule), None
            return False, "", None
        return False, "", None

    @staticmethod
    def _trigger_text(rule: GlobalRule) -> str:
        trigger = rule.trigger
        if isinstance(trigger, AtStepTrigger):
            return f"at_step(step={trigger.step})"
        if isinstance(trigger, NewDetectionTrigger):
            return (
                f"new_detection(entity_type={trigger.entity_type}, "
                f"occurrence={trigger.occurrence})"
            )
        if isinstance(trigger, LaunchedStepsAgoTrigger):
            return (
                f"launched_steps_ago(platform_id={trigger.platform_id}, "
                f"steps={trigger.steps})"
            )
        return "unknown_trigger"

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def _resolve_target(
        self, ref: TargetRef, bound_entity_id: int | None
    ) -> tuple[tuple[float, float] | None, str | None]:
        if ref.mode == "coordinate":
            if ref.lon is None or ref.lat is None:
                return None, "coordinate_reference_incomplete"
            if not (math.isfinite(ref.lon) and math.isfinite(ref.lat)):
                return None, "coordinate_not_finite"
            return (float(ref.lon), float(ref.lat)), None
        if ref.mode == "entity":
            point = self.registry.resolve(int(ref.entity_id))
            if point is None:
                return None, f"target_entity_{ref.entity_id}_unknown"
            return (point.lon, point.lat), None
        if ref.mode == "event_entity":
            if bound_entity_id is None:
                return None, "event_entity_not_bound"
            point = self.registry.resolve(bound_entity_id)
            if point is None:
                return None, f"event_entity_{bound_entity_id}_unknown"
            return (point.lon, point.lat), None
        return None, "unknown_target_mode"

    def _emit(self, command: PlannedCommand, alive: set[int]) -> None:
        """Resolve one planned command; emit a row or record a failure."""

        if command.platform_id not in alive:
            self._trace(
                command,
                engine_action=None,
                success=False,
                failure_reason="platform_not_alive",
            )
            return
        if command.kind == "satellite_request":
            row = [float(SATELLITE_ROW), float(command.platform_id), 0.0, 0.0]
            self._finish(command, row)
            return
        assert command.target is not None
        coordinates, failure = self._resolve_target(
            command.target, command.bound_entity_id
        )
        if coordinates is None:
            self._trace(
                command,
                engine_action=None,
                success=False,
                failure_reason=failure or "target_resolution_failed",
            )
            return
        if command.kind == "launch":
            if command.platform_id in self.launched_platforms:
                self._trace(
                    command,
                    engine_action=None,
                    success=False,
                    failure_reason="platform_already_launched",
                )
                return
            row = [
                float(LAUNCH_ROW),
                float(command.platform_id),
                float(coordinates[0]),
                float(coordinates[1]),
            ]
            self._finish(command, row)
            self.launched_platforms.add(command.platform_id)
            self._emitted_launch_step[command.platform_id] = self._current_step
            return
        # retarget
        if command.platform_id not in self.launched_platforms:
            self._trace(
                command,
                engine_action=None,
                success=False,
                failure_reason="platform_not_launched",
            )
            return
        row = [
            float(RETARGET_ROW),
            float(command.platform_id),
            float(coordinates[0]),
            float(coordinates[1]),
        ]
        self._finish(command, row)

    def _finish(self, command: PlannedCommand, row: list[float]) -> None:
        self._pending_by_platform.setdefault(command.platform_id, []).append(row)
        self.executed_commands += 1
        self._trace(command, engine_action=row, success=True, failure_reason=None)

    def _trace(
        self,
        command: PlannedCommand,
        *,
        engine_action: list[float] | None,
        success: bool,
        failure_reason: str | None,
        conflicting_rule_ids: list[str] | None = None,
    ) -> None:
        if not success:
            self.failed_commands += 1
        if self._on_trace is None:
            return
        record = {
            "step": int(self._current_step),
            "platform_id": int(command.platform_id),
            "plan_rule_id": command.rule_id,
            "trigger": command.trigger,
            "planned_action": command.planned_action,
            "engine_action": engine_action,
            "success": bool(success),
            "failure_reason": failure_reason,
            "conflicting_rule_ids": conflicting_rule_ids,
        }
        self._on_trace(record)

    # ------------------------------------------------------------------
    # Runtime conflict detection
    # ------------------------------------------------------------------

    @staticmethod
    def _partition_conflicts(
        candidates: Sequence[PlannedCommand],
    ) -> tuple[list[PlannedCommand], list[tuple[list[PlannedCommand], list[str]]]]:
        """Split this step's candidates into (executable, conflict groups).

        A conflict is an unambiguous plan ambiguity the executor has no
        authority to resolve.  Deliberately minimal definitions only:

        * two or more *different* retarget candidates for the same platform
          in the same step (the engine would apply them in some order and
          only the last would matter -- a hidden selection);
        * two or more satellite_request candidates in the same step (the
          team executes at most one request per step).

        Everything else -- different platforms, retarget plus satellite on
        one platform, launch plus retarget -- executes in deterministic
        order and is NOT a conflict.
        """

        executable: list[PlannedCommand] = []
        conflict_groups: list[tuple[list[PlannedCommand], list[str]]] = []

        retargets_by_platform: dict[int, list[PlannedCommand]] = {}
        satellites: list[PlannedCommand] = []
        for command in candidates:
            if command.kind == "retarget":
                retargets_by_platform.setdefault(command.platform_id, []).append(
                    command
                )
            elif command.kind == "satellite_request":
                satellites.append(command)
            else:
                executable.append(command)

        for platform_id, group in sorted(retargets_by_platform.items()):
            if len(group) > 1:
                # Two or more retarget commands for one platform in one step
                # (even to the identical target) cannot be honoured as
                # written: sending both would let the engine's command order
                # silently keep only the last one.  That hidden selection is
                # exactly what the executor may not perform.
                conflict_groups.append((list(group), [c.rule_id for c in group]))
            else:
                executable.append(group[0])

        if len(satellites) > 1:
            conflict_groups.append((list(satellites), [c.rule_id for c in satellites]))
        else:
            executable.extend(satellites)

        return executable, conflict_groups

    def _record_conflicts(
        self, groups: Sequence[tuple[list[PlannedCommand], list[str]]]
    ) -> None:
        """Trace every command of a conflict group; execute none of them."""

        for group, rule_ids in groups:
            self.runtime_conflicts += 1
            ordered_ids = sorted(rule_ids)
            for command in group:
                self._trace(
                    command,
                    engine_action=None,
                    success=False,
                    failure_reason="runtime_plan_conflict",
                    conflicting_rule_ids=ordered_ids,
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def begin_step(
        self, step: int, observations: Sequence[Mapping[str, Any]]
    ) -> None:
        if step <= self._current_step:
            raise ValueError(
                "executor steps must strictly increase "
                f"(got {step} after {self._current_step})"
            )
        self._current_step = int(step)
        self._pending_by_platform = {}

        detections = self._extract_detections(observations)
        self.registry.update_from_detections(detections)
        new_entities = self._update_detection_events(detections)

        alive = {
            int(_field(observation, "entity_id"))
            for observation in observations
            if _field(observation, "self") is not None
        }

        candidates: list[PlannedCommand] = []

        # 1. Platform-scheduled commands (launch, retargets, satellite).
        for platform_id in sorted(self._platform_plan_by_id):
            platform = self._platform_plan_by_id[platform_id]
            if (
                platform.launch.mode == "at_step"
                and platform.launch.step == step
                and platform_id not in self.launched_platforms
            ):
                candidates.append(
                    PlannedCommand(
                        rule_id=f"platform:{platform_id}:launch",
                        trigger=f"launch_schedule(step={platform.launch.step})",
                        planned_action={
                            "type": "launch",
                            "platform_id": platform_id,
                            "step": int(platform.launch.step),
                            "target": (
                                platform.initial_target.to_dict()
                                if platform.initial_target
                                else None
                            ),
                        },
                        kind="launch",
                        platform_id=platform_id,
                        target=platform.initial_target,
                    )
                )
            for order in platform.retarget_orders:
                if order.step != step:
                    continue
                candidates.append(
                    PlannedCommand(
                        rule_id=f"platform:{platform_id}:retarget@{order.step}",
                        trigger=f"retarget_schedule(step={order.step})",
                        planned_action={
                            "type": "retarget",
                            "platform_id": platform_id,
                            "step": int(order.step),
                            "target": order.target.to_dict(),
                        },
                        kind="retarget",
                        platform_id=platform_id,
                        target=order.target,
                    )
                )
            for request in platform.satellite_steps:
                if request.step != step:
                    continue
                candidates.append(
                    PlannedCommand(
                        rule_id=f"platform:{platform_id}:satellite@{request.step}",
                        trigger=f"satellite_schedule(step={request.step})",
                        planned_action={
                            "type": "satellite_request",
                            "platform_id": platform_id,
                            "step": int(request.step),
                        },
                        kind="satellite_request",
                        platform_id=platform_id,
                    )
                )

        # 2. Global rules whose trigger fired at this step. Detection events
        #    bind event_entity to the entity that fired the occurrence.
        for rule in self.plan.global_rules:
            fired, description, bound = self._rule_fires(rule, step, new_entities)
            if not fired:
                continue
            self._fired_rules.add(rule.rule_id)
            if isinstance(rule.trigger, NewDetectionTrigger) and bound is not None:
                self._event_bindings[rule.rule_id] = int(bound)
            for index, action in enumerate(rule.actions):
                if isinstance(action, RetargetAction):
                    candidates.append(
                        PlannedCommand(
                            rule_id=f"{rule.rule_id}:actions[{index}]",
                            trigger=description,
                            planned_action=action.to_dict(),
                            kind="retarget",
                            platform_id=action.platform_id,
                            target=action.target,
                            bound_entity_id=bound,
                        )
                    )
                elif isinstance(action, SatelliteRequestAction):
                    candidates.append(
                        PlannedCommand(
                            rule_id=f"{rule.rule_id}:actions[{index}]",
                            trigger=description,
                            planned_action=action.to_dict(),
                            kind="satellite_request",
                            platform_id=action.platform_id,
                        )
                    )

        # 3. Conflict gate: ambiguous command groups are traced as failures
        #    and never executed; the episode continues with the rest.
        executable, conflict_groups = self._partition_conflicts(candidates)
        self._record_conflicts(conflict_groups)
        for command in executable:
            self._emit(command, alive)

    def actions_for(self, platform_id: int, step: int) -> list[list[float]]:
        """Rows for one platform at the current step (straight motion only)."""

        if step != self._current_step:
            raise ValueError(
                f"actions_for step {step} does not match executor step "
                f"{self._current_step}"
            )
        return [
            list(row)
            for row in self._pending_by_platform.get(int(platform_id), ())
        ]
