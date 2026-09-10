"""Fixed-width observation encoding for generic joint-control games."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

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
    detected_threat_count: int = 0
    nearest_threat_known: bool = False
    nearest_threat_position: tuple[float, float] = (0.0, 0.0)
    nearest_threat_velocity_xy: tuple[float, float] = (0.0, 0.0)
    nearest_threat_velocity_known: bool = False
    nearest_threat_age_steps: int = 0
    target_progress_reference: float = 0.0
    target_progress_fraction: float = 0.0

    def __post_init__(self) -> None:
        ranges = {
            "target_progress_reference": (0.0, 1.0),
            "target_progress_fraction": (-1.0, 1.0),
        }
        for name, (low, high) in ranges.items():
            raw_value = getattr(self, name)
            if isinstance(raw_value, bool):
                raise ValueError(f"{name} must be a finite value in [{low}, {high}]")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{name} must be a finite value in [{low}, {high}]"
                ) from error
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be a finite value in [{low}, {high}]")
            object.__setattr__(self, name, value)


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
    assigned_total: float = 0.0
    assigned_high: float = 0.0
    assigned_medium: float = 0.0
    assigned_low: float = 0.0

    def __post_init__(self) -> None:
        # These values are controller-owned load fractions.  Keeping their
        # contract here prevents a bad denominator or an accidental raw count
        # from silently changing the observation scale.
        for name in (
            "assigned_total",
            "assigned_high",
            "assigned_medium",
            "assigned_low",
        ):
            raw_value = getattr(self, name)
            if isinstance(raw_value, bool):
                raise ValueError(f"{name} must be a finite fraction in [0, 1]")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{name} must be a finite fraction in [0, 1]"
                ) from error
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite fraction in [0, 1]")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ObservationEncoderConfig:
    max_steps: int
    space: JointSpaceSpec
    bounds: MapBounds
    max_speed: float
    max_track_age_steps: int
    unit_type_count: int
    objective_type_count: int
    detected_threat_count_normalizer: float = 32.0

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
        if (
            not math.isfinite(float(self.detected_threat_count_normalizer))
            or self.detected_threat_count_normalizer <= 0.0
        ):
            raise ValueError("detected_threat_count_normalizer must be positive")


def _clip(value: float, low: float, high: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        return 0.0
    if numeric < low:
        return float(low)
    if numeric > high:
        return float(high)
    return numeric


def _finite(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else 0.0


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
        # The encoder is called once per controlled unit and simulator step.
        # These values depend only on the frozen configuration, so constructing
        # them in the hot path needlessly dominates small scalar encodes.
        self._feature_names = self._build_feature_names()
        bounds = config.bounds
        self._x_span = float(bounds.x_max - bounds.x_min)
        self._y_span = float(bounds.y_max - bounds.y_min)
        self._altitude_span = float(bounds.altitude_max - bounds.altitude_min)
        self._map_diagonal = math.hypot(self._x_span, self._y_span)
        self._map_center_x = 0.5 * float(bounds.x_min + bounds.x_max)
        self._map_center_y = 0.5 * float(bounds.y_min + bounds.y_max)
        self._empty_objective_features = (0.0,) * (
            15 + config.objective_type_count
        )
        self._unit_type_vectors = self._one_hot_vectors(config.unit_type_count)
        self._objective_type_vectors = self._one_hot_vectors(
            config.objective_type_count
        )
        self._current_objective_vectors = self._one_hot_vectors(
            config.space.objective_count + 1
        )

    @property
    def dimension(self) -> int:
        return len(self._feature_names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    def _build_feature_names(self) -> tuple[str, ...]:
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
        names.extend(
            (
                "health",
                "visible",
                "activation_age",
                "objective_change_age",
                "target_progress_reference",
                "target_progress_fraction",
            )
        )
        names.extend(f"last_movement_{movement.name.lower()}" for movement in Movement)
        names.extend(f"unit_type_{index}" for index in range(config.unit_type_count))
        names.extend(
            (
                "detected_threat_count",
                "nearest_threat_known",
                "nearest_threat_rel_x",
                "nearest_threat_rel_y",
                "nearest_threat_distance",
                "nearest_threat_bearing_sin",
                "nearest_threat_bearing_cos",
                "nearest_threat_velocity_known",
                "nearest_threat_vx",
                "nearest_threat_vy",
                "nearest_threat_age",
            )
        )
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
            names.extend(
                (
                    f"{prefix}_assigned_total",
                    f"{prefix}_assigned_high",
                    f"{prefix}_assigned_medium",
                    f"{prefix}_assigned_low",
                )
            )
        return tuple(names)

    @staticmethod
    def _one_hot_vectors(size: int) -> tuple[tuple[float, ...], ...]:
        zero = (0.0,) * size
        vectors = [zero]
        for index in range(size):
            values = [0.0] * size
            values[index] = 1.0
            vectors.append(tuple(values))
        return tuple(vectors)

    @staticmethod
    def _cached_one_hot(
        index: int | None,
        vectors: tuple[tuple[float, ...], ...],
    ) -> tuple[float, ...]:
        if index is None:
            return vectors[0]
        numeric = int(index)
        if numeric < 0 or numeric >= len(vectors) - 1:
            return vectors[0]
        return vectors[numeric + 1]

    def _index_objectives(
        self,
        objectives: Sequence[ObjectiveFrame],
    ) -> Mapping[int, ObjectiveFrame]:
        objective_by_slot = {item.slot: item for item in objectives}
        if len(objective_by_slot) != len(objectives):
            raise ValueError("objective frames contain duplicate slots")
        if any(
            slot < 0 or slot >= self.config.space.objective_count
            for slot in objective_by_slot
        ):
            raise ValueError("objective slot is out of range")
        return objective_by_slot

    def _shared_features(
        self,
        sensor_state: SharedSensorState,
        *,
        step: int,
        sensor_ready_override: bool | None = None,
    ) -> tuple[float, ...]:
        config = self.config
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
        sensor_ready = (
            sensor_state.is_ready(step)
            if sensor_ready_override is None
            else bool(sensor_ready_override)
        )
        return (
            time_fraction,
            1.0 - time_fraction,
            _clip(sensor_available_fraction, 0.0, 1.0),
            _clip(sensor_pending_fraction, 0.0, 1.0),
            float(sensor_ready),
        )

    def encode_many(
        self,
        units: Sequence[UnitFrame],
        controls: Sequence[UnitControlState],
        objectives: Sequence[ObjectiveFrame],
        sensor_state: SharedSensorState,
        *,
        step: int,
        sensor_ready_override: bool | None = None,
    ) -> tuple[np.ndarray, ...]:
        """Encode a team while indexing shared objective frames only once."""

        if len(units) != len(controls):
            raise ValueError("units and controller states must have equal lengths")
        if step < 0:
            raise ValueError("step must be non-negative")
        objective_by_slot = self._index_objectives(objectives)
        shared_features = self._shared_features(
            sensor_state,
            step=step,
            sensor_ready_override=sensor_ready_override,
        )
        return tuple(
            self._encode_one(
                unit,
                control,
                objective_by_slot,
                sensor_state,
                step=step,
                shared_features=shared_features,
            )
            for unit, control in zip(units, controls)
        )

    def encode(
        self,
        unit: UnitFrame,
        control: UnitControlState,
        objectives: Sequence[ObjectiveFrame],
        sensor_state: SharedSensorState,
        *,
        step: int,
        sensor_ready_override: bool | None = None,
    ) -> np.ndarray:
        return self._encode_one(
            unit,
            control,
            self._index_objectives(objectives),
            sensor_state,
            step=step,
            sensor_ready_override=sensor_ready_override,
        )

    def _encode_one(
        self,
        unit: UnitFrame,
        control: UnitControlState,
        objective_by_slot: Mapping[int, ObjectiveFrame],
        sensor_state: SharedSensorState,
        *,
        step: int,
        shared_features: tuple[float, ...] | None = None,
        sensor_ready_override: bool | None = None,
    ) -> np.ndarray:
        config = self.config
        if unit.slot != control.slot:
            raise ValueError("unit frame and controller state refer to different slots")
        if unit.slot < 0 or unit.slot >= config.space.unit_count:
            raise ValueError("unit slot is out of range")
        if step < 0:
            raise ValueError("step must be non-negative")
        if (
            control.current_objective_slot != NO_OBJECTIVE
            and not 0
            <= control.current_objective_slot
            < config.space.objective_count
        ):
            raise ValueError("controller state has an out-of-range objective slot")

        values = list(
            self._shared_features(
                sensor_state,
                step=step,
                sensor_ready_override=sensor_ready_override,
            )
            if shared_features is None
            else shared_features
        )
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
        x_span = self._x_span
        y_span = self._y_span
        altitude_span = self._altitude_span
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
                unit.target_progress_reference,
                unit.target_progress_fraction,
            )
        )
        values.extend(float(control.last_movement == movement) for movement in Movement)
        values.extend(self._cached_one_hot(unit.type_index, self._unit_type_vectors))

        diagonal = self._map_diagonal
        # A staged entity has no meaningful own position yet, but its placement
        # head still needs the public objective geometry.  In that case use the
        # map centre as an explicit reference.  ``self_position_known`` remains
        # zero, so this cannot be confused with an observed entity position.
        geometry_x = x if position_known else self._map_center_x
        geometry_y = y if position_known else self._map_center_y
        threat_position_finite = all(
            math.isfinite(float(value)) for value in unit.nearest_threat_position
        )
        threat_known = (
            bool(unit.nearest_threat_known)
            and threat_position_finite
        )
        values.append(
            _clip(
                unit.detected_threat_count
                / config.detected_threat_count_normalizer,
                0.0,
                1.0,
            )
        )
        if threat_known:
            threat_dx = float(unit.nearest_threat_position[0]) - float(geometry_x)
            threat_dy = float(unit.nearest_threat_position[1]) - float(geometry_y)
            threat_distance = math.hypot(threat_dx, threat_dy)
            threat_bearing = (
                math.atan2(threat_dy, threat_dx) if threat_distance > 0.0 else 0.0
            )
            threat_velocity_known = bool(
                unit.nearest_threat_velocity_known
            ) and all(
                math.isfinite(float(value))
                for value in unit.nearest_threat_velocity_xy
            )
            values.extend(
                (
                    1.0,
                    _clip(threat_dx / x_span, -1.0, 1.0),
                    _clip(threat_dy / y_span, -1.0, 1.0),
                    _clip(threat_distance / diagonal, 0.0, 1.0),
                    math.sin(threat_bearing),
                    math.cos(threat_bearing),
                    float(threat_velocity_known),
                    _clip(
                        unit.nearest_threat_velocity_xy[0] / config.max_speed,
                        -1.0,
                        1.0,
                    )
                    if threat_velocity_known
                    else 0.0,
                    _clip(
                        unit.nearest_threat_velocity_xy[1] / config.max_speed,
                        -1.0,
                        1.0,
                    )
                    if threat_velocity_known
                    else 0.0,
                    _clip(
                        unit.nearest_threat_age_steps
                        / config.max_track_age_steps,
                        0.0,
                        1.0,
                    ),
                )
            )
        else:
            values.extend([0.0] * 10)
        current_index = (
            0
            if control.current_objective_slot == NO_OBJECTIVE
            else control.current_objective_slot + 1
        )
        values.extend(
            self._cached_one_hot(current_index, self._current_objective_vectors)
        )

        for slot in range(config.space.objective_count):
            objective = objective_by_slot.get(slot)
            if objective is None:
                values.extend(self._empty_objective_features)
                continue
            valid = bool(objective.valid)
            objective_position_finite = all(
                math.isfinite(float(value)) for value in objective.position
            )
            known = (
                valid
                and bool(objective.known)
                and objective_position_finite
            )
            if known:
                dx = float(objective.position[0]) - float(geometry_x)
                dy = float(objective.position[1]) - float(geometry_y)
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
                    self._cached_one_hot(
                        objective.type_index,
                        self._objective_type_vectors,
                    )
                )
            else:
                values.extend((float(valid), 0.0))
                values.extend([0.0] * (9 + config.objective_type_count))
            values.extend(
                (
                    objective.assigned_total,
                    objective.assigned_high,
                    objective.assigned_medium,
                    objective.assigned_low,
                )
            )

        encoded = np.asarray(values, dtype=np.float32)
        if encoded.shape != (len(self._feature_names),):
            raise AssertionError(
                "encoder produced "
                f"{encoded.shape}, expected {(len(self._feature_names),)}"
            )
        if not bool(np.isfinite(encoded).all()):
            raise ValueError("encoded observation contains a non-finite value")
        return encoded

    def _age(self, step: int, event_step: int | None) -> float:
        if event_step is None:
            return 0.0
        return _clip((step - event_step) / self.config.max_steps, 0.0, 1.0)
