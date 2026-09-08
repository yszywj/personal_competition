"""Fixed-width observation encoding for generic joint-control games."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contracts import (
    JointSpaceSpec,
    Movement,
    NO_OBJECTIVE,
    SharedSensorState,
    UnitControlState,
    UnitPhase,
)


@dataclass(frozen=True)
class MapBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    altitude_min: float = 0.0
    altitude_max: float = 1.0

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(float(value))
            for value in (
                self.x_min,
                self.x_max,
                self.y_min,
                self.y_max,
                self.altitude_min,
                self.altitude_max,
            )
        ):
            raise ValueError("map bounds must be finite")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("map bounds must have positive width and height")
        if self.altitude_max <= self.altitude_min:
            raise ValueError("altitude bounds must have positive range")


@dataclass(frozen=True)
class UnitFrame:
    slot: int
    type_index: int | None
    position: tuple[float, float, float]
    velocity_xy: tuple[float, float] = (0.0, 0.0)
    position_known: bool = True
    velocity_known: bool = False
    health_fraction: float = 1.0
    visible: bool = True


@dataclass(frozen=True)
class ObjectiveFrame:
    slot: int
    valid: bool
    known: bool
    position: tuple[float, float] = (0.0, 0.0)
    velocity_xy: tuple[float, float] = (0.0, 0.0)
    velocity_known: bool = False
    age_steps: int = 0
    type_index: int | None = None


@dataclass(frozen=True)
class ObservationEncoderConfig:
    max_steps: int
    space: JointSpaceSpec
    bounds: MapBounds
    max_speed: float
    max_track_age_steps: int
    unit_type_count: int
    objective_type_count: int

    def __post_init__(self) -> None:
        for name in (
            "max_steps",
            "max_track_age_steps",
            "unit_type_count",
            "objective_type_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(float(self.max_speed)) or self.max_speed <= 0.0:
            raise ValueError("max_speed must be positive")


def _clip(value: float, low: float, high: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return float(np.clip(value, low, high))


def _finite(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else 0.0


def _one_hot(index: int | None, size: int) -> list[float]:
    values = [0.0] * size
    if index is not None and 0 <= int(index) < size:
        values[int(index)] = 1.0
    return values


class JointObservationEncoder:
    """Encode only supplied observations plus explicit controller history."""

    GLOBAL_FEATURES = (
        "time",
        "time_remaining",
        "sensor_available",
        "sensor_pending",
        "sensor_ready",
    )

    def __init__(self, config: ObservationEncoderConfig) -> None:
        self.config = config

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        config = self.config
        names = list(self.GLOBAL_FEATURES)
        names.extend(f"phase_{phase.name.lower()}" for phase in UnitPhase)
        names.extend(
            (
                "self_position_known",
                "self_x",
                "self_y",
                "self_altitude",
                "self_velocity_known",
                "self_vx",
                "self_vy",
            )
        )
        names.extend(("health", "visible", "activation_age", "objective_change_age"))
        names.extend(f"last_movement_{movement.name.lower()}" for movement in Movement)
        names.extend(f"unit_type_{index}" for index in range(config.unit_type_count))
        names.append("current_objective_none")
        names.extend(
            f"current_objective_{index}"
            for index in range(config.space.objective_count)
        )
        for slot in range(config.space.objective_count):
            prefix = f"objective_{slot}"
            names.extend(
                (
                    f"{prefix}_valid",
                    f"{prefix}_known",
                    f"{prefix}_rel_x",
                    f"{prefix}_rel_y",
                    f"{prefix}_distance",
                    f"{prefix}_bearing_sin",
                    f"{prefix}_bearing_cos",
                    f"{prefix}_velocity_known",
                    f"{prefix}_vx",
                    f"{prefix}_vy",
                    f"{prefix}_age",
                )
            )
            names.extend(
                f"{prefix}_type_{index}"
                for index in range(config.objective_type_count)
            )
        return tuple(names)

    def encode(
        self,
        unit: UnitFrame,
        control: UnitControlState,
        objectives: Sequence[ObjectiveFrame],
        sensor_state: SharedSensorState,
        *,
        step: int,
    ) -> np.ndarray:
        config = self.config
        if unit.slot != control.slot:
            raise ValueError("unit frame and controller state refer to different slots")
        if unit.slot < 0 or unit.slot >= config.space.unit_count:
            raise ValueError("unit slot is out of range")
        if step < 0:
            raise ValueError("step must be non-negative")
        objective_by_slot = {item.slot: item for item in objectives}
        if len(objective_by_slot) != len(objectives):
            raise ValueError("objective frames contain duplicate slots")
        if any(
            slot < 0 or slot >= config.space.objective_count
            for slot in objective_by_slot
        ):
            raise ValueError("objective slot is out of range")
        if (
            control.current_objective_slot != NO_OBJECTIVE
            and not 0
            <= control.current_objective_slot
            < config.space.objective_count
        ):
            raise ValueError("controller state has an out-of-range objective slot")

        time_fraction = _clip(step / config.max_steps, 0.0, 1.0)
        sensor_available_fraction = (
            sensor_state.available / sensor_state.config.capacity
            if sensor_state.config.capacity > 0
            else 0.0
        )
        sensor_pending_fraction = (
            len(sensor_state.pending) / sensor_state.config.capacity
            if sensor_state.config.capacity > 0
            else 0.0
        )
        values: list[float] = [
            time_fraction,
            1.0 - time_fraction,
            _clip(sensor_available_fraction, 0.0, 1.0),
            _clip(sensor_pending_fraction, 0.0, 1.0),
            float(sensor_state.is_ready(step)),
        ]
        values.extend(float(control.phase == phase) for phase in UnitPhase)

        raw_x, raw_y, raw_altitude = unit.position
        position_known = bool(unit.position_known) and all(
            math.isfinite(float(value)) for value in unit.position
        )
        velocity_known = bool(unit.velocity_known) and all(
            math.isfinite(float(value)) for value in unit.velocity_xy
        )
        x, y, altitude = _finite(raw_x), _finite(raw_y), _finite(raw_altitude)
        vx, vy = _finite(unit.velocity_xy[0]), _finite(unit.velocity_xy[1])
        x_span = config.bounds.x_max - config.bounds.x_min
        y_span = config.bounds.y_max - config.bounds.y_min
        altitude_span = config.bounds.altitude_max - config.bounds.altitude_min
        normalized_x = 2.0 * (float(x) - config.bounds.x_min) / x_span - 1.0
        normalized_y = 2.0 * (float(y) - config.bounds.y_min) / y_span - 1.0
        normalized_altitude = (
            2.0 * (float(altitude) - config.bounds.altitude_min) / altitude_span - 1.0
        )
        values.extend(
            (
                float(position_known),
                _clip(normalized_x, -1.0, 1.0) if position_known else 0.0,
                _clip(normalized_y, -1.0, 1.0) if position_known else 0.0,
                _clip(normalized_altitude, -1.0, 1.0) if position_known else 0.0,
                float(velocity_known),
                _clip(vx / config.max_speed, -1.0, 1.0) if velocity_known else 0.0,
                _clip(vy / config.max_speed, -1.0, 1.0) if velocity_known else 0.0,
                _clip(unit.health_fraction, 0.0, 1.0),
                float(unit.visible),
                self._age(step, control.activated_step),
                self._age(step, control.last_objective_change_step),
            )
        )
        values.extend(float(control.last_movement == movement) for movement in Movement)
        values.extend(_one_hot(unit.type_index, config.unit_type_count))
        current_index = (
            0
            if control.current_objective_slot == NO_OBJECTIVE
            else control.current_objective_slot + 1
        )
        values.extend(_one_hot(current_index, config.space.objective_count + 1))

        diagonal = math.hypot(x_span, y_span)
        for slot in range(config.space.objective_count):
            objective = objective_by_slot.get(slot)
            if objective is None:
                values.extend([0.0] * (11 + config.objective_type_count))
                continue
            valid = bool(objective.valid)
            objective_position_finite = all(
                math.isfinite(float(value)) for value in objective.position
            )
            known = (
                valid
                and bool(objective.known)
                and position_known
                and objective_position_finite
            )
            if known:
                dx = float(objective.position[0]) - float(x)
                dy = float(objective.position[1]) - float(y)
                distance = math.hypot(dx, dy)
                bearing = math.atan2(dy, dx) if distance > 0.0 else 0.0
                objective_velocity_known = bool(objective.velocity_known) and all(
                    math.isfinite(float(value)) for value in objective.velocity_xy
                )
                values.extend(
                    (
                        float(valid),
                        1.0,
                        _clip(dx / x_span, -1.0, 1.0),
                        _clip(dy / y_span, -1.0, 1.0),
                        _clip(distance / diagonal, 0.0, 1.0),
                        math.sin(bearing),
                        math.cos(bearing),
                        float(objective_velocity_known),
                        _clip(objective.velocity_xy[0] / config.max_speed, -1.0, 1.0)
                        if objective_velocity_known
                        else 0.0,
                        _clip(objective.velocity_xy[1] / config.max_speed, -1.0, 1.0)
                        if objective_velocity_known
                        else 0.0,
                        _clip(
                            objective.age_steps / config.max_track_age_steps,
                            0.0,
                            1.0,
                        ),
                    )
                )
                values.extend(
                    _one_hot(objective.type_index, config.objective_type_count)
                )
            else:
                values.extend((float(valid), 0.0))
                values.extend([0.0] * (9 + config.objective_type_count))

        encoded = np.asarray(values, dtype=np.float32)
        if encoded.shape != (self.dimension,):
            raise AssertionError(
                f"encoder produced {encoded.shape}, expected {(self.dimension,)}"
            )
        if not bool(np.isfinite(encoded).all()):
            raise ValueError("encoded observation contains a non-finite value")
        return encoded

    def _age(self, step: int, event_step: int | None) -> float:
        if event_step is None:
            return 0.0
        return _clip((step - event_step) / self.config.max_steps, 0.0, 1.0)
