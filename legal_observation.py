"""Competition-compatible corrections for the existing 90-D R9 observation.

The public Actor schema is not expanded or reordered.  Missing motion fields
are reconstructed from consecutive *isolated own-platform observations*, and
the corrected values are fed through the project's original encoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .bootstrap import install_project_paths

install_project_paths()

from policies.red.learning import ObservationEncoder  # noqa: E402


@dataclass(frozen=True)
class _VectorProxy:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass(frozen=True)
class _TrackProxy:
    detect_from: int
    time: int
    entity_id: int
    entity_type: int
    nameChn: str
    lla: _VectorProxy
    pos_ecf: _VectorProxy
    vel_ecf: _VectorProxy


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _vector(value: Any) -> _VectorProxy:
    return _VectorProxy(
        float(_field(value, "x", 0.0)),
        float(_field(value, "y", 0.0)),
        float(_field(value, "z", 0.0)),
    )


class CorrectedR9ObservationEncoder(ObservationEncoder):
    """Preserve the original 90-D contract while correcting broken features.

    Feature indices and normalization remain those of ``ObservationEncoder``.
    In particular, target slots stay at the competition value of five and the
    R9 context remains the last five values.
    """

    INTERCEPTOR_TYPE = 24000

    def __init__(
        self,
        init_observation: Mapping[str, Any],
        *,
        max_steps: int,
        agent_id: int,
        team_size: int,
        initial_sim_time_ms: float,
        sim_step_ms: float,
    ) -> None:
        super().__init__(
            init_observation,
            max_steps=max_steps,
            agent_id=agent_id,
            team_size=team_size,
            hierarchical_task_context=True,
        )
        self.initial_sim_time_ms = float(initial_sim_time_ms)
        self.sim_step_ms = max(float(sim_step_ms), 1.0)
        self._last_step: int | None = None
        self._last_ecf: np.ndarray | None = None
        self._last_heading = 0.0
        self._velocity = {"speed": 0.0, "up": 0.0, "heading": 0.0}
        self._track_snapshots: dict[int, tuple[float, _TrackProxy]] = {}

    def reset_history(self) -> None:
        self._last_step = None
        self._last_ecf = None
        self._last_heading = 0.0
        self._velocity = {"speed": 0.0, "up": 0.0, "heading": 0.0}
        self._track_snapshots.clear()

    def observe_frame(self, observation: Mapping[str, Any]) -> None:
        """Update legal motion estimates once for each new environment step."""

        own = observation.get("self")
        if not isinstance(own, Mapping):
            return
        ecf = own.get("pos_ecf")
        position = own.get("position")
        if not isinstance(ecf, Mapping) or not isinstance(position, Mapping):
            return
        step = int(observation.get("step", 0))
        current_ecf = np.asarray(
            [float(ecf.get("x", 0.0)), float(ecf.get("y", 0.0)), float(ecf.get("z", 0.0))],
            dtype=np.float64,
        )

        # record_step(s') and the next get_action(s') both encode the same
        # frame.  Reusing the cache prevents the second call from creating a
        # false zero velocity.
        if self._last_step == step:
            return

        if self._last_step is not None and self._last_ecf is not None and step > self._last_step:
            dt_seconds = (step - self._last_step) * self.sim_step_ms / 1000.0
            velocity_ecf = (current_ecf - self._last_ecf) / dt_seconds
            lon = math.radians(float(position.get("lon", 0.0)))
            lat = math.radians(float(position.get("lat", 0.0)))
            vx, vy, vz = velocity_ecf

            east = -math.sin(lon) * vx + math.cos(lon) * vy
            north = (
                -math.sin(lat) * math.cos(lon) * vx
                - math.sin(lat) * math.sin(lon) * vy
                + math.cos(lat) * vz
            )
            up = (
                math.cos(lat) * math.cos(lon) * vx
                + math.cos(lat) * math.sin(lon) * vy
                + math.sin(lat) * vz
            )
            horizontal_speed = math.hypot(east, north)
            if horizontal_speed > 1e-6:
                self._last_heading = math.atan2(east, north)
            self._velocity = {
                "speed": float(np.linalg.norm(velocity_ecf)),
                "up": float(up),
                "heading": float(self._last_heading),
            }

        self._last_step = step
        self._last_ecf = current_ecf

    def encode(
        self,
        observation: Mapping[str, Any],
        *,
        launched: bool,
        launch_step: int,
        satellite_used: bool,
        maneuver_state: int,
        current_target_index: int | None = None,
        target_switch_elapsed: int = 0,
        task_context: Sequence[float] | None = None,
    ) -> np.ndarray:
        self.observe_frame(observation)
        prepared = self._prepare_observation(observation)
        encoded = super().encode(
            prepared,
            launched=launched,
            launch_step=launch_step,
            satellite_used=satellite_used,
            maneuver_state=maneuver_state,
            current_target_index=current_target_index,
            target_switch_elapsed=target_switch_elapsed,
            task_context=task_context,
        )

        # The base encoder's planar bearing is close enough locally but becomes
        # wrong near longitude wrapping.  Replace only the four bearing values
        # with a spherical initial bearing and a wrapped relative bearing.
        self._replace_target_bearings(encoded, prepared)

        # Count interceptors by the stable entity type, not the display-name
        # prefix "标6".  Feature position is unchanged.
        detect_info = prepared["self"].get("detectInfo") or {}
        interceptor_count = sum(
            int(_field(track, "entity_type", -1)) == self.INTERCEPTOR_TYPE
            for track in detect_info.values()
        )
        detection_start = self.SELF_FEATURES + self.target_slots * self.TARGET_FEATURES
        encoded[detection_start + 1] = np.clip(interceptor_count / 200.0, 0.0, 1.0)
        return encoded

    def target_index(self, target_id: int | None) -> int | None:
        if target_id is None:
            return None
        return next(
            (
                index
                for index, target in enumerate(self.targets)
                if int(target.get("entity_id", -1)) == int(target_id)
            ),
            None,
        )

    def _prepare_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        own = dict(observation["self"])
        own["velocity"] = dict(self._velocity)
        current_step = int(observation.get("step", 0))
        raw_tracks = own.get("detectInfo") or {}
        own["detectInfo"] = {
            raw_id: self._track_proxy(raw_id, track, current_step)
            for raw_id, track in raw_tracks.items()
        }
        prepared = dict(observation)
        prepared["self"] = own
        return prepared

    def _track_proxy(self, raw_id: Any, track: Any, current_step: int) -> _TrackProxy:
        raw_time = float(_field(track, "time", self.initial_sim_time_ms))
        entity_id = int(_field(track, "entity_id", raw_id))
        cached = self._track_snapshots.get(entity_id)
        if cached is not None and cached[0] == raw_time:
            # Some simulator tracks hold mutable Vector3d references.  A stale
            # timestamp must therefore retain the first legally observed
            # coordinates rather than drift with an unreported target update.
            return cached[1]
        # Native DetectInfo timestamps are absolute logic time in milliseconds.
        # Already-normalized small values are accepted as steps to keep tests
        # and alternate frontends compatible.
        if raw_time >= self.initial_sim_time_ms - self.sim_step_ms:
            track_step = int(round((raw_time - self.initial_sim_time_ms) / self.sim_step_ms))
        elif 0.0 <= raw_time <= self.max_steps:
            track_step = int(round(raw_time))
        else:
            track_step = current_step
        track_step = max(0, track_step)
        entity_type = int(_field(track, "entity_type", -1))
        proxy = _TrackProxy(
            detect_from=int(_field(track, "detect_from", 0)),
            time=track_step,
            entity_id=entity_id,
            entity_type=entity_type,
            nameChn=str(_field(track, "nameChn", "")),
            lla=_vector(_field(track, "lla", None)),
            pos_ecf=_vector(_field(track, "pos_ecf", None)),
            vel_ecf=_vector(_field(track, "vel_ecf", None)),
        )
        self._track_snapshots[entity_id] = (raw_time, proxy)
        return proxy

    def _replace_target_bearings(
        self,
        encoded: np.ndarray,
        observation: Mapping[str, Any],
    ) -> None:
        own = observation["self"]
        position = own["position"]
        lon1 = math.radians(float(position["lon"]))
        lat1 = math.radians(float(position["lat"]))
        heading = float(self._velocity["heading"])
        detections = {
            int(_field(track, "entity_id", raw_id)): track
            for raw_id, track in (own.get("detectInfo") or {}).items()
        }

        for index, target in enumerate(self.targets[: self.target_slots]):
            target_id = int(target["entity_id"])
            detection = detections.get(target_id)
            if detection is None:
                target_position = target["position"]
                lon2 = math.radians(float(target_position["lon"]))
                lat2 = math.radians(float(target_position["lat"]))
            else:
                lla = _field(detection, "lla", None)
                lon2 = math.radians(float(_field(lla, "x", target["position"]["lon"])))
                lat2 = math.radians(float(_field(lla, "y", target["position"]["lat"])))
            delta_lon = math.atan2(math.sin(lon2 - lon1), math.cos(lon2 - lon1))
            x = math.sin(delta_lon) * math.cos(lat2)
            y = (
                math.cos(lat1) * math.sin(lat2)
                - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
            )
            bearing = math.atan2(x, y)
            relative = math.atan2(math.sin(bearing - heading), math.cos(bearing - heading))
            base = self.SELF_FEATURES + index * self.TARGET_FEATURES
            encoded[base + 3] = math.sin(bearing)
            encoded[base + 4] = math.cos(bearing)
            encoded[base + 10] = math.sin(relative)
            encoded[base + 11] = math.cos(relative)
