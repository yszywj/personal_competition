"""Deterministic fixtures: isolated observations, opening catalogues, plans.

``nameChn`` values are deliberately misleading (e.g. a 21000 platform named
"9500_暗巢_x") to prove nothing keys off entity names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from personal_train.llm_strategy.state_builder import EnvironmentRules


@dataclass
class FakeVector:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class FakeDetectInfo:
    """Attribute-style detect record, mirroring the engine's DetectInfo."""

    detect_from: int = 0
    time: int = 0
    entity_id: int = 0
    entity_type: int = -1
    nameChn: str = ""
    lla: FakeVector = field(default_factory=FakeVector)
    pos_ecf: FakeVector = field(default_factory=FakeVector)
    vel_ecf: FakeVector = field(default_factory=FakeVector)


def make_detection(
    *,
    entity_id: int,
    entity_type: int,
    lon: float,
    lat: float,
    time: int = 0,
) -> FakeDetectInfo:
    return FakeDetectInfo(
        entity_id=entity_id,
        entity_type=entity_type,
        time=time,
        lla=FakeVector(x=lon, y=lat, z=0.0),
    )


def make_detection_dict(
    *,
    entity_id: int,
    entity_type: int,
    lon: float,
    lat: float,
    time: int = 0,
) -> dict[str, Any]:
    return {
        "detect_from": 0,
        "time": time,
        "entity_id": entity_id,
        "entity_type": entity_type,
        "nameChn": f"误导名_{entity_id}",
        "lla": {"x": lon, "y": lat, "z": 0.0},
        "pos_ecf": {"x": 0.0, "y": 0.0, "z": 0.0},
        "vel_ecf": {"x": 0.0, "y": 0.0, "z": 0.0},
    }


def make_isolated_obs(
    *,
    entity_id: int,
    entity_type: int,
    step: int = 0,
    lon: float = 116.0,
    lat: float = 22.0,
    health: float = 10.0,
    stage: int = 0,
    visible: bool = True,
    detections: Sequence[Any] = (),
    is_using_satellite: bool = False,
    name: str | None = None,
) -> dict[str, Any]:
    detect_info: dict[int, Any] = {}
    for detection in detections:
        if isinstance(detection, Mapping):
            detect_info[int(detection["entity_id"])] = detection
        else:
            detect_info[int(detection.entity_id)] = detection
    return {
        "step": int(step),
        "entity_id": int(entity_id),
        "agent_id": int(entity_id),
        "self": {
            "nameChn": name or f"误导名_{entity_type}_{entity_id}",
            "position": {"lon": float(lon), "lat": float(lat), "alt": 0.0},
            "pos_ecf": {"x": 0.0, "y": 0.0, "z": 0.0},
            "stage": int(stage),
            "health": float(health),
            "isVisible": bool(visible),
            "type": int(entity_type),
            "side": 0,
            "detectInfo": detect_info,
            "commRangeInfo": [],
        },
        "is_using_satellite": bool(is_using_satellite),
    }


def make_init_obs(entities: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Opening catalogue in the exact shape of _get_init_ship_observation()."""

    built: dict[int, Any] = {}
    for entity in entities:
        entity_id = int(entity["id"])
        built[entity_id] = {
            "nameChn": entity.get("name", f"误导名_{entity_id}"),
            "position": {
                "lon": float(entity["lon"]),
                "lat": float(entity["lat"]),
                "alt": 0.0,
            },
            "health": float(entity["health"]),
            "isVisible": True,
            "type": int(entity["type"]),
            "side": 1,
            "detectInfo": {},
            "commRangeInfo": [],
        }
    return {"step": 0, "entities": built}


def default_rules(max_steps: int = 1200) -> EnvironmentRules:
    return EnvironmentRules(
        max_steps=max_steps,
        sim_step_ms=1000,
        map_lon_min=115.8,
        map_lon_max=128.2,
        map_lat_min=18.4,
        map_lat_max=28.2,
        satellite_max_use_count=100,
        satellite_active_minutes=3.0,
        hit_increase_time_interval_minutes=2.0,
        hit_increase_min_angle_deg=30.0,
        hit_increase_max_fraction=0.2,
        hit_decrease_time_interval_minutes=2.0,
        hit_decrease_max_angle_deg=10.0,
        hit_decrease_max_fraction=0.4,
    )


class CountingClient:
    """In-memory model that returns a fixed payload and counts calls."""

    model = "counting-fake"

    def __init__(self, payload: Any):
        self.payload = payload
        self.call_count = 0
        self.prompts: list[tuple[str, str]] = []

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        self.prompts.append((system_prompt, user_prompt))
        if isinstance(self.payload, str):
            return self.payload
        return json.dumps(self.payload, ensure_ascii=False)
