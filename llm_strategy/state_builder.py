"""Battle-state construction from *legal* red-side information only.

Allowed inputs (anything else is a bug):
    1. ``TrainingEnv._get_init_ship_observation()`` -- the opening catalogue.
       Current engine code puts only entity types 9400/9600 in it; 9500 ships
       are therefore unknown until a red platform detects them.
    2. The isolated per-platform observations handed to
       ``commander.begin_step(...)`` by ``TrainingEnv`` (each contains the
       platform's own ``self`` record, ``detectInfo`` and
       ``is_using_satellite``).
    3. Public environment configuration (max steps, sim step, map bounds,
       satellite budget/duration) and public weapon/scoring rules.

This module must never import ``envengine`` or read
``TrainingEnv._get_observation()``, scenario.json, case_info.json or
manifest.json.  It only arranges facts: no target prioritisation, no grouping,
no allocation, no waves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .plan_schema import EVENT_ENTITY_TYPES

RED_PLATFORM_TYPES = {21000, 21001, 21002}
OBJECTIVE_TYPES = {9400, 9500, 9600}
# V0 opening-catalogue admission: whatever the upstream environment returns,
# only these entity types may enter the LLM's opening target catalogue.
OPENING_CATALOGUE_TYPES = frozenset({9400, 9600})
TYPE_NAMES = {
    21000: "high_speed_strike",
    21001: "medium_speed_strike",
    21002: "low_cost_strike",
    9400: "hive_target",
    9500: "dark_nest_ship",
    9600: "sentinel_nest_site",
    24000: "interceptor_swarm",
    44000: "vision_tower",
    9202: "satellite",
}

# Public weapon facts mirroring SimulatorFactory.process_hit tables.
BASE_HIT_RATES = {
    21000: {9400: 0.8, 9600: 0.6, 9500: 0.0},
    21001: {9400: 0.8, 9600: 0.6, 9500: 0.0},
    21002: {9400: 0.05, 9600: 0.05, 9500: 0.8},
}
BASE_DAMAGE = {
    21000: {9400: 20, 9600: 20, 9500: 20},
    21001: {9400: 5, 9600: 5, 9500: 5},
    21002: {9400: 0, 9600: 0, 9500: 1},
}
OBJECTIVE_VALUE_WEIGHTS = {9400: 5.0, 9600: 2.0, 9500: 1.0}
MISSILE_COMMUNICATION_RANGE_KM = {21000: 50.0, 21001: 20.0, 21002: 10.0}


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a Mapping or an attribute-style object (DetectInfo
    values may be dataclasses or dicts)."""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


@dataclass(frozen=True)
class EnvironmentRules:
    """Public, model-facing configuration (no hidden blue state)."""

    max_steps: int
    sim_step_ms: int
    map_lon_min: float
    map_lon_max: float
    map_lat_min: float
    map_lat_max: float
    satellite_max_use_count: int
    satellite_active_minutes: float
    hit_increase_time_interval_minutes: float
    hit_increase_min_angle_deg: float
    hit_increase_max_fraction: float
    hit_decrease_time_interval_minutes: float
    hit_decrease_max_angle_deg: float
    hit_decrease_max_fraction: float

    def map_rules(self) -> dict[str, Any]:
        return {
            "lon_range": [self.map_lon_min, self.map_lon_max],
            "lat_range": [self.map_lat_min, self.map_lat_max],
            "note": (
                "Red platforms are already deployed; deployment positions are "
                "listed per platform below and cannot be changed."
            ),
        }

    def satellite_rules(self) -> dict[str, Any]:
        return {
            "team_max_total_uses": int(self.satellite_max_use_count),
            "active_window_minutes_per_use": float(self.satellite_active_minutes),
            "team_shared": True,
            "effect": (
                "While the window is active the red satellite detects all alive "
                "and visible 24000 interceptor entities and shares those "
                "tracks with the whole red team; high-speed platforms (21000) "
                "get hit rate 1.0 against 9400/9600 during the window."
            ),
            "command": (
                "A satellite request is issued by any one red platform via the "
                "satellite_request action; each accepted request consumes one "
                "team use and opens/re-opens the shared window."
            ),
        }

    def timing_rules(self) -> dict[str, Any]:
        return {
            "max_steps": int(self.max_steps),
            "sim_step_ms": int(self.sim_step_ms),
            "valid_plan_steps": [0, int(self.max_steps) - 1],
        }


def weapon_rules() -> dict[str, Any]:
    return {
        "base_hit_rate": {
            str(platform): {str(target): rate for target, rate in targets.items()}
            for platform, targets in BASE_HIT_RATES.items()
        },
        "base_damage_points": {
            str(platform): {str(target): damage for target, damage in targets.items()}
            for platform, targets in BASE_DAMAGE.items()
        },
        "satellite_hit_bonus": (
            "While the satellite window is active, 21000 hit rate against "
            "9400 and 9600 becomes 1.0."
        ),
        "multi_hit_modifier": (
            "For 21000/21001 attacking 9400 only: if another 21000/21001 hit "
            "the same target within the configured interval and the impact "
            "angle difference exceeds the configured minimum angle, the hit "
            "rate is multiplied by (1 + up to the configured max increase), "
            "decaying linearly with the time gap; similarly, near-simultaneous "
            "hits arriving at nearly the same angle multiply the hit rate by "
            "(1 - up to the configured max decrease)."
        ),
        "multi_hit_parameters": {
            "increase_interval_minutes": "see satellite_and_hit_rules in the "
            "battle state",
            "decrease_interval_minutes": "see satellite_and_hit_rules in the "
            "battle state",
        },
        "communication_ranges_km": dict(MISSILE_COMMUNICATION_RANGE_KM),
        "detection_sharing": (
            "Red platforms within mutual communication range share and fuse "
            "their detections per step; the satellite shares its detections "
            "with the whole team while active."
        ),
    }


def scoring_rules() -> dict[str, Any]:
    return {
        "objective_value_weights": {
            str(entity_type): weight
            for entity_type, weight in OBJECTIVE_VALUE_WEIGHTS.items()
        },
        "formula": (
            "final score (0..100) = 100 * sum(objective weight * damage "
            "fraction of that objective's initial health) / sum(objective "
            "weights); damage fraction is clipped to [0, 1] per objective."
        ),
        "time_independent": True,
    }


def _platform_record(observation: Mapping[str, Any]) -> dict[str, Any] | None:
    """One red-platform fact record from an isolated observation."""

    entity_id = _field(observation, "entity_id")
    self_info = _field(observation, "self")
    if self_info is None:
        return None
    entity_type = _field(self_info, "type")
    try:
        entity_id = int(entity_id)
        entity_type = int(entity_type)
    except (TypeError, ValueError):
        return None
    if entity_type not in RED_PLATFORM_TYPES:
        return None
    position = _field(self_info, "position") or {}
    return {
        "entity_id": entity_id,
        "type": entity_type,
        "type_name": TYPE_NAMES.get(entity_type, "unknown"),
        "position": {
            "lon": _finite(_field(position, "lon")),
            "lat": _finite(_field(position, "lat")),
            "alt": _finite(_field(position, "alt")),
        },
        "health": _finite(_field(self_info, "health")),
        "stage": _field(self_info, "stage"),
        "is_visible": bool(_field(self_info, "isVisible", False)),
    }


def _iter_detections(detect_info: Any) -> list[Any]:
    if not detect_info:
        return []
    if isinstance(detect_info, Mapping):
        return list(detect_info.values())
    return []


def _collect_detected_tracks(
    platform_observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Union of tracks across red platforms, freshest timestamp per entity."""

    freshest: dict[int, dict[str, Any]] = {}
    for observation in platform_observations:
        self_info = _field(observation, "self")
        if self_info is None:
            continue
        for detection in _iter_detections(_field(self_info, "detectInfo")):
            try:
                detected_id = int(
                    _field(detection, "entity_id", _field(detection, "id"))
                )
            except (TypeError, ValueError):
                continue
            timestamp = _finite(_field(detection, "time"), default=-1.0)
            lla = _field(detection, "lla")
            vel = _field(detection, "vel_ecf")
            record = {
                "entity_id": detected_id,
                "entity_type": _field(detection, "entity_type"),
                "detect_time": timestamp,
                "lon": _finite(_field(lla, "x")),
                "lat": _finite(_field(lla, "y")),
                "velocity_ecf": {
                    "x": _finite(_field(vel, "x")),
                    "y": _finite(_field(vel, "y")),
                    "z": _finite(_field(vel, "z")),
                },
            }
            existing = freshest.get(detected_id)
            if existing is None or timestamp >= existing["detect_time"]:
                freshest[detected_id] = record
    # Deterministic presentation only: sorting by id is not prioritisation.
    return [freshest[key] for key in sorted(freshest)]


def _known_targets_from_init(init_ship_observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Opening catalogue facts, hard-filtered to 9400/9600.

    The filter is enforced locally on purpose: even if a future upstream
    ``_get_init_ship_observation()`` were to return other entity types
    (e.g. 9500 ships), the V0 opening catalogue must never widen the LLM's
    information rights.  9500 stays reachable only through legal detectInfo
    event rules.
    """

    entities = _field(init_ship_observation, "entities") or {}
    records: list[dict[str, Any]] = []
    items = entities.items() if isinstance(entities, Mapping) else []
    for key, entity in items:
        if entity is None:
            continue
        try:
            entity_id = int(_field(entity, "id", key))
            entity_type = int(_field(entity, "type"))
        except (TypeError, ValueError):
            continue
        if entity_type not in OPENING_CATALOGUE_TYPES:
            continue
        position = _field(entity, "position") or {}
        records.append(
            {
                "entity_id": entity_id,
                "type": entity_type,
                "type_name": TYPE_NAMES.get(entity_type, "unknown"),
                "position": {
                    "lon": _finite(_field(position, "lon")),
                    "lat": _finite(_field(position, "lat")),
                    "alt": _finite(_field(position, "alt")),
                },
                "initial_health": _finite(_field(entity, "health")),
                "initially_public": True,
            }
        )
    return sorted(records, key=lambda item: item["entity_id"])


def build_battle_state(
    *,
    step: int,
    init_ship_observation: Mapping[str, Any],
    platform_observations: Sequence[Mapping[str, Any]],
    rules: EnvironmentRules,
) -> dict[str, Any]:
    """Assemble the factual one-shot planning input for the LLM.

    ``platform_observations`` are the isolated observations received by
    ``commander.begin_step`` at the current step (after deployment).  Nothing
    outside the documented legal inputs is read.
    """

    platforms: list[dict[str, Any]] = []
    is_using_satellite: bool | None = None
    for observation in platform_observations:
        if is_using_satellite is None:
            flag = _field(observation, "is_using_satellite")
            if isinstance(flag, bool):
                is_using_satellite = flag
        record = _platform_record(observation)
        if record is not None:
            platforms.append(record)
    platforms.sort(key=lambda item: (item["type"], item["entity_id"]))

    known_targets = _known_targets_from_init(init_ship_observation)
    detected_tracks = [
        track
        for track in _collect_detected_tracks(platform_observations)
        if int(track["entity_type"] or -1) in EVENT_ENTITY_TYPES
        or int(track["entity_type"] or -1) in OBJECTIVE_TYPES
        or int(track["entity_type"] or -1) in RED_PLATFORM_TYPES
    ]

    return {
        "step": int(step),
        "max_steps": int(rules.max_steps),
        "red_platforms": platforms,
        "known_targets": known_targets,
        "detected_tracks": detected_tracks,
        "satellite_status": {
            "is_using_satellite": (
                bool(is_using_satellite) if is_using_satellite is not None else False
            ),
            "team_uses_remaining_at_planning_time": int(
                rules.satellite_max_use_count
            ),
        },
        "satellite_rules": rules.satellite_rules(),
        "map_rules": rules.map_rules(),
        "timing_rules": rules.timing_rules(),
        "weapon_rules": weapon_rules(),
        "scoring_rules": scoring_rules(),
        "hit_timing_parameters": {
            "increase_interval_minutes": float(
                rules.hit_increase_time_interval_minutes
            ),
            "increase_min_angle_deg": float(rules.hit_increase_min_angle_deg),
            "increase_max_fraction": float(rules.hit_increase_max_fraction),
            "decrease_interval_minutes": float(
                rules.hit_decrease_time_interval_minutes
            ),
            "decrease_max_angle_deg": float(rules.hit_decrease_max_angle_deg),
            "decrease_max_fraction": float(rules.hit_decrease_max_fraction),
        },
    }
