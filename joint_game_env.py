"""Simulator bridge for the joint high-level and movement policy.

The upstream checkout is imported read-only.  Deployment is represented to the
policy as one activation decision; this bridge applies the chosen position and
submits launch plus the first movement command in the same engine step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .bootstrap import install_project_paths
from .joint_rl_core import (
    ActivationReceipt,
    JointAction,
    JointActionMask,
    JointControlTracker,
    JointObservationEncoder,
    JointSpaceSpec,
    MapBounds,
    Movement,
    ObjectiveFrame,
    ObservationEncoderConfig,
    SharedSensorConfig,
    SharedSensorReceipt,
    UnitControlState,
    UnitFrame,
    UnitPhase,
    build_joint_action_mask,
    validate_action_against_mask,
)
from .joint_reward_credit import (
    interceptor_track_information_potential,
    objective_information_potential,
    potential_difference,
)

install_project_paths()

from envengine import TrainingEnv  # noqa: E402
from envengine.agent_manager.actions.aircraft_action import (  # noqa: E402
    ChangeTargetAction,
    MissileLaunchAction,
    SetDesiredAccZ,
)
from envengine.agent_manager.actions.aircraft_action.use_satellite import (  # noqa: E402
    UseSatelliteAction,
)
from envengine.common import Vector3d  # noqa: E402
from envengine.environment.command_adapter import CommandAdapter  # noqa: E402
from envengine.sdk.writer import write_ai_action, write_state  # noqa: E402
from scenarios.cases import RewardPolicy, RewardTracker  # noqa: E402


RED_MISSILE_TYPES = (21000, 21001, 21002)
UNIT_TYPE_INDEX = {21000: 0, 21001: 1, 21002: 2}
OBJECTIVE_TYPE_INDEX = {9400: 0, 9600: 1, 9500: 2}


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _entity(
    observation: Mapping[str, Any], entity_id: int
) -> Mapping[str, Any] | None:
    entities = observation.get("entities") or {}
    value = entities.get(entity_id, entities.get(str(entity_id)))
    return value if isinstance(value, Mapping) else None


def _alive(value: Mapping[str, Any] | None) -> bool:
    return bool(
        value is not None
        and float(value.get("health", 0.0)) > 0.0
        and bool(value.get("isVisible", True))
    )


def _position(value: Mapping[str, Any] | None) -> tuple[float, float, float]:
    raw = (value or {}).get("position") or {}
    return (
        float(raw.get("lon", 0.0)),
        float(raw.get("lat", 0.0)),
        float(raw.get("alt", 0.0)),
    )


def _vector(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        if isinstance(value, Mapping):
            result = (
                float(value.get("x", 0.0)),
                float(value.get("y", 0.0)),
                float(value.get("z", 0.0)),
            )
        else:
            result = (float(value.x), float(value.y), float(value.z))
    except (AttributeError, TypeError, ValueError):
        return None
    return result if all(math.isfinite(component) for component in result) else None


def _ecf_velocity_to_enu_xy(
    velocity: tuple[float, float, float],
    position_lla: tuple[float, float, float],
) -> tuple[float, float]:
    """Convert an ECF velocity into local east/north components."""

    lon = math.radians(float(position_lla[0]))
    lat = math.radians(float(position_lla[1]))
    vx, vy, vz = velocity
    east = -math.sin(lon) * vx + math.cos(lon) * vy
    north = (
        -math.sin(lat) * math.cos(lon) * vx
        - math.sin(lat) * math.sin(lon) * vy
        + math.cos(lat) * vz
    )
    return float(east), float(north)


def _bbox(coordinates: Sequence[Sequence[Sequence[float]]]) -> tuple[float, float, float, float]:
    if not coordinates or not coordinates[0]:
        raise ValueError("deployment polygon is empty")
    points = coordinates[0]
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    low_x, high_x = min(xs), max(xs)
    low_y, high_y = min(ys), max(ys)
    if high_x <= low_x or high_y <= low_y:
        raise ValueError("deployment polygon has no area")
    supplied = {(float(point[0]), float(point[1])) for point in points}
    rectangle = {
        (low_x, low_y),
        (low_x, high_y),
        (high_x, low_y),
        (high_x, high_y),
    }
    if supplied != rectangle:
        raise ValueError(
            "only axis-aligned rectangular deployment polygons are supported"
        )
    return low_x, high_x, low_y, high_y


def denormalize_placement(
    placement: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Map a policy value in [-1, 1]^2 to a deployment bounding box."""

    px = float(np.clip(placement[0], -1.0, 1.0))
    py = float(np.clip(placement[1], -1.0, 1.0))
    x_min, x_max, y_min, y_max = bounds
    return (
        x_min + 0.5 * (px + 1.0) * (x_max - x_min),
        y_min + 0.5 * (py + 1.0) * (y_max - y_min),
    )


@dataclass(frozen=True)
class JointGameConfig:
    """Stable game-facing dimensions and reward scales."""

    objective_slots: int = 18
    max_track_age_steps: int = 300
    max_speed_mps: float = 3_000.0
    sensor_capacity: int | None = None
    sensor_max_requests_per_step: int = 1
    sensor_cooldown_steps: int = 0
    official_reward_scale: float = 1.0
    progress_potential_scale: float = 0.05
    planning_team_weight: float = 0.7
    planning_local_weight: float = 0.3
    sensor_information_potential_scale: float = 0.015
    gamma: float = 0.995
    terminate_on_all_objectives_destroyed: bool = True
    strict_weapon_target_compatibility: bool = True
    allow_low_altitude_search_fallback: bool = True
    allow_low_altitude_search_replanning: bool = False
    retarget_min_dwell_steps: int = 0
    retarget_decision_interval_steps: int = 1
    motion_decision_interval_steps: int = 1
    post_launch_motion_only: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.objective_slots, bool)
            or not isinstance(self.objective_slots, int)
            or self.objective_slots <= 0
        ):
            raise ValueError("objective_slots must be a positive integer")
        if (
            isinstance(self.max_track_age_steps, bool)
            or not isinstance(self.max_track_age_steps, int)
            or self.max_track_age_steps <= 0
            or not math.isfinite(float(self.max_speed_mps))
            or self.max_speed_mps <= 0.0
        ):
            raise ValueError("track age and speed scales must be positive")
        if self.sensor_capacity is not None and (
            isinstance(self.sensor_capacity, bool)
            or not isinstance(self.sensor_capacity, int)
            or self.sensor_capacity < 0
        ):
            raise ValueError("sensor_capacity must be a non-negative integer")
        if (
            isinstance(self.sensor_max_requests_per_step, bool)
            or not isinstance(self.sensor_max_requests_per_step, int)
            or self.sensor_max_requests_per_step <= 0
        ):
            raise ValueError("sensor_max_requests_per_step must be a positive integer")
        if (
            isinstance(self.sensor_cooldown_steps, bool)
            or not isinstance(self.sensor_cooldown_steps, int)
            or self.sensor_cooldown_steps < 0
        ):
            raise ValueError("sensor_cooldown_steps must be a non-negative integer")
        if not math.isfinite(float(self.gamma)) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not all(
            math.isfinite(float(value)) and value >= 0.0
            for value in (
                self.official_reward_scale,
                self.progress_potential_scale,
                self.planning_team_weight,
                self.planning_local_weight,
                self.sensor_information_potential_scale,
            )
        ):
            raise ValueError("reward scales must be non-negative")
        if not math.isclose(
            float(self.planning_team_weight)
            + float(self.planning_local_weight),
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("planning reward weights must sum to 1")
        if not isinstance(self.terminate_on_all_objectives_destroyed, bool):
            raise ValueError("terminate_on_all_objectives_destroyed must be boolean")
        if not isinstance(self.strict_weapon_target_compatibility, bool):
            raise ValueError("strict_weapon_target_compatibility must be boolean")
        if not isinstance(self.allow_low_altitude_search_fallback, bool):
            raise ValueError("allow_low_altitude_search_fallback must be boolean")
        if not isinstance(self.allow_low_altitude_search_replanning, bool):
            raise ValueError("allow_low_altitude_search_replanning must be boolean")
        if (
            isinstance(self.retarget_min_dwell_steps, bool)
            or not isinstance(self.retarget_min_dwell_steps, int)
            or self.retarget_min_dwell_steps < 0
        ):
            raise ValueError("retarget_min_dwell_steps must be a non-negative integer")
        if (
            isinstance(self.retarget_decision_interval_steps, bool)
            or not isinstance(self.retarget_decision_interval_steps, int)
            or self.retarget_decision_interval_steps <= 0
        ):
            raise ValueError(
                "retarget_decision_interval_steps must be a positive integer"
            )

        if (
            isinstance(self.motion_decision_interval_steps, bool)
            or not isinstance(self.motion_decision_interval_steps, int)
            or self.motion_decision_interval_steps <= 0
        ):
            raise ValueError("motion_decision_interval_steps must be a positive integer")
        if not isinstance(self.post_launch_motion_only, bool):
            raise ValueError("post_launch_motion_only must be boolean")


@dataclass(frozen=True)
class JointGameStep:
    observations: tuple[np.ndarray, ...]
    states: tuple[UnitControlState, ...]
    rewards: tuple[float, ...]
    team_reward: float
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    team_terminated: bool
    team_truncated: bool
    done: bool
    score: float
    raw_observation: Mapping[str, Any]
    submitted_commands: int
    accepted_activations: tuple[int, ...]
    accepted_sensor_requests: tuple[int, ...]
    sensor_reward: float | None = None
    # Confirmed post-action states before a team-wide terminal marker is
    # applied.  Credit collection uses these so the successful final step is
    # not discarded when all surviving units become TERMINAL for bootstrapping.
    assignment_states: tuple[UnitControlState, ...] | None = None


@dataclass
class _ObjectiveTrack:
    slot: int
    entity_id: int | None
    known: bool = False
    position: tuple[float, float] = (0.0, 0.0)
    velocity_xy: tuple[float, float] = (0.0, 0.0)
    velocity_known: bool = False
    type_index: int | None = None
    last_seen_step: int = 0
    static_public: bool = False
    source_timestamp: int = -1


class JointGameEnv:
    """One centralized game controller backed by the unmodified TrainingEnv."""

    def __init__(
        self,
        profile: Any,
        *,
        reward_policy: RewardPolicy,
        config: JointGameConfig | None = None,
        render_mode: str | None = None,
        render_fps: int = 10,
        max_steps_override: int | None = None,
    ) -> None:
        self.config = config or JointGameConfig()
        if len(reward_policy.objective_ids) > self.config.objective_slots:
            raise ValueError(
                f"scenario needs {len(reward_policy.objective_ids)} objective slots, "
                f"configured {self.config.objective_slots}"
            )
        if max_steps_override is not None and (
            isinstance(max_steps_override, bool)
            or not isinstance(max_steps_override, int)
            or max_steps_override <= 0
        ):
            raise ValueError("max_steps_override must be a positive integer")
        if (
            isinstance(render_fps, bool)
            or not isinstance(render_fps, int)
            or render_fps <= 0
        ):
            raise ValueError("render_fps must be a positive integer")
        self.reward_policy = reward_policy
        self._environment = TrainingEnv(profile, render_mode=render_mode)
        simulator_max_steps = int(self._environment.max_steps)
        if simulator_max_steps != int(reward_policy.max_steps):
            raise ValueError(
                "case_info max_steps differs from the simulator horizon: "
                f"{reward_policy.max_steps} != {simulator_max_steps}"
            )
        if (
            max_steps_override is not None
            and max_steps_override > int(reward_policy.max_steps)
        ):
            raise ValueError(
                "max_steps_override cannot exceed the official scenario horizon"
            )
        if self._environment.renderer is not None:
            self._environment.renderer.fps = int(render_fps)
        self.engine = self._environment.engine
        self.max_steps = min(
            int(reward_policy.max_steps),
            int(max_steps_override) if max_steps_override is not None else int(reward_policy.max_steps),
        )
        self.is_debug_horizon = self.max_steps < int(reward_policy.max_steps)
        self.current_step = 0

        red = sorted(
            (
                simulator
                for simulator in self.engine.simulator_factory.get_all_simulators()
                if int(simulator.entity_ext.entity.entityType) in RED_MISSILE_TYPES
                and int(getattr(simulator.entity_ext.entity, "sideId", 0)) == 0
            ),
            key=lambda simulator: (
                UNIT_TYPE_INDEX[int(simulator.entity_ext.entity.entityType)],
                int(simulator.entity_ext.entity.id),
            ),
        )
        if not red:
            raise RuntimeError("scenario contains no controllable red missiles")
        self.unit_ids = tuple(int(item.entity_ext.entity.id) for item in red)
        self.unit_slot_by_id = {
            entity_id: slot for slot, entity_id in enumerate(self.unit_ids)
        }
        self.unit_types = tuple(
            int(item.entity_ext.entity.entityType) for item in red
        )
        self._initial_unit_health = tuple(
            max(float(item.entity_ext.entity.survivePoints), 1e-6) for item in red
        )

        objective_ids: list[int | None] = [
            int(value) for value in reward_policy.objective_ids
        ]
        objective_ids.extend(
            [None] * (self.config.objective_slots - len(objective_ids))
        )
        self.objective_ids = tuple(objective_ids)
        self.objective_slot_by_id = {
            entity_id: slot
            for slot, entity_id in enumerate(self.objective_ids)
            if entity_id is not None
        }
        objective_weights = dict(reward_policy.objective_weights) or {
            entity_id: 1.0 for entity_id in reward_policy.objective_ids
        }
        self._objective_weight_by_slot = {
            slot: float(objective_weights[entity_id])
            for slot, entity_id in enumerate(self.objective_ids)
            if entity_id is not None
        }
        self.space = JointSpaceSpec(
            unit_count=len(self.unit_ids),
            objective_count=self.config.objective_slots,
        )

        profile_capacity = max(
            0,
            int(getattr(profile.imagineProfile, "satelliteMaxUseCount", 100)),
        )
        factory = self.engine.simulator_factory
        global_sensor_api = (
            hasattr(factory, "red_sat_use_count")
            and hasattr(factory, "red_sat_max_use_count")
            and callable(getattr(factory, "is_using_satellite", None))
        )
        if global_sensor_api:
            self.sensor_backend = "team_global"
            self.sensor_backend_capacity_per_unit: int | None = None
            self.sensor_backend_capacity_team = max(
                0, int(factory.red_sat_max_use_count)
            )
            self.sensor_backend_active_minutes: float | None = float(
                getattr(factory, "sat_use_minutes", 0.0)
            )
            self.sensor_information_source = "fresh_interceptor_tracks"
        else:
            self.sensor_backend = "per_unit"
            self.sensor_backend_capacity_per_unit = profile_capacity
            self.sensor_backend_capacity_team = (
                self.sensor_backend_capacity_per_unit * len(self.unit_ids)
            )
            self.sensor_backend_active_minutes = None
            self.sensor_information_source = "known_objective_value"
        backend_team_capacity = self.sensor_backend_capacity_team
        if (
            self.config.sensor_capacity is not None
            and self.config.sensor_capacity > backend_team_capacity
        ):
            raise ValueError(
                "sensor_capacity exceeds the simulator backend's team capacity "
                f"({backend_team_capacity})"
            )
        sensor_capacity = (
            backend_team_capacity
            if self.config.sensor_capacity is None
            else int(self.config.sensor_capacity)
        )
        self.tracker = JointControlTracker(
            self.space,
            SharedSensorConfig(
                capacity=sensor_capacity,
                max_requests_per_step=self.config.sensor_max_requests_per_step,
                cooldown_steps=self.config.sensor_cooldown_steps,
            ),
        )

        self._sensor_threat_normalizer = max(
            1,
            sum(
                int(
                    int(getattr(item.entity_ext.entity, "sideId", 0)) != 0
                    and float(item.entity_ext.entity.survivePoints) > 0.0
                )
                for item in factory.get_simulators_by_type(24000)
            ),
        )

        map_area = profile.imagineProfile.mapArea
        self.encoder = JointObservationEncoder(
            ObservationEncoderConfig(
                max_steps=max(1, self.max_steps),
                space=self.space,
                bounds=MapBounds(
                    x_min=float(map_area.lonMin),
                    x_max=float(map_area.lonMax),
                    y_min=float(map_area.latMin),
                    y_max=float(map_area.latMax),
                    altitude_min=0.0,
                    altitude_max=40_000.0,
                ),
                max_speed=self.config.max_speed_mps,
                max_track_age_steps=self.config.max_track_age_steps,
                unit_type_count=len(UNIT_TYPE_INDEX),
                objective_type_count=len(OBJECTIVE_TYPE_INDEX),
                detected_threat_count_normalizer=(
                    float(self._sensor_threat_normalizer)
                    if self.sensor_backend == "team_global"
                    else 32.0
                ),
            )
        )
        self._map_diagonal_km = self._distance_km(
            (float(map_area.lonMin), float(map_area.latMin)),
            (float(map_area.lonMax), float(map_area.latMax)),
        )
        if not math.isfinite(self._map_diagonal_km) or self._map_diagonal_km <= 0.0:
            raise ValueError("map bounds must have a positive geodesic diagonal")

        red_area = profile.imagineProfile.redArea
        normal_bounds = _bbox(red_area.coordinates)
        hm_coordinates = red_area.coordinatesHM or red_area.coordinates
        hm_bounds = _bbox(hm_coordinates)
        self._deployment_bounds = {
            21000: hm_bounds,
            21001: hm_bounds,
            21002: normal_bounds,
        }
        self._public_objectives = self._environment._get_init_ship_observation()
        self._tracks: list[_ObjectiveTrack] = []
        self._last_raw_observation: Mapping[str, Any] | None = None
        self._last_unit_ecf: dict[int, tuple[float, float, float]] = {}
        self._progress: dict[int, tuple[int, float, float]] = {}
        self._reward_tracker: RewardTracker | None = None
        self._score = 0.0
        self._launch_count = 0
        self._sensor_request_count = 0
        self._sensor_information_potential = 0.0
        self._cached_action_mask_step = -1
        self._cached_action_mask: JointActionMask | None = None

    @property
    def observation_dim(self) -> int:
        return self.encoder.dimension

    @property
    def states(self) -> tuple[UnitControlState, ...]:
        return self.tracker.states

    @property
    def score(self) -> float:
        return float(self._score)

    @property
    def launch_count(self) -> int:
        return self._launch_count

    @property
    def sensor_request_count(self) -> int:
        return self._sensor_request_count

    def _global_sensor_status(self) -> tuple[int, int, bool]:
        """Return synchronized team-global sensor usage information."""

        if getattr(self, "sensor_backend", "per_unit") != "team_global":
            raise RuntimeError("global sensor status requested from a per-unit backend")
        factory = self.engine.simulator_factory
        # Upstream synchronizes this clock immediately before command handling,
        # then advances Engine.sim_time at the end of the step.  Synchronize on
        # reads as well so a three-minute window does not stay masked one extra
        # simulator step at its boundary.
        factory.sim_time = self.engine.sim_time
        used = int(factory.red_sat_use_count)
        maximum = max(0, int(factory.red_sat_max_use_count))
        return used, maximum, bool(factory.is_using_satellite())

    def _sensor_ready_override(self) -> bool | None:
        """Expose backend readiness without changing the observation schema."""

        if getattr(self, "sensor_backend", "per_unit") != "team_global":
            return None
        used, maximum, active = self._global_sensor_status()
        return bool(
            self.tracker.sensor_state.is_ready(self.current_step)
            and used < maximum
            and not active
        )

    def reset(self) -> tuple[np.ndarray, ...]:
        # Step zero is revisited on every episode, so a previous episode's
        # immutable mask must not survive the simulator reset.
        self._cached_action_mask_step = -1
        self._cached_action_mask = None
        raw = self._environment.reset()
        # The upstream Engine resets its clock but older simulator reset paths
        # leave private clocks at zero.  Commands may arrive before update().
        initial_time = float(self.engine.profile.imagineProfile.simTime)
        for simulator in self.engine.simulator_factory.get_all_simulators():
            simulator.sim_time = initial_time
        if getattr(self, "sensor_backend", "per_unit") == "team_global":
            self.engine.simulator_factory.sim_time = self.engine.sim_time
        self.current_step = 0
        self.tracker.reset()
        self._tracks = [
            _ObjectiveTrack(slot=slot, entity_id=entity_id)
            for slot, entity_id in enumerate(self.objective_ids)
        ]
        public_entities = self._public_objectives.get("entities") or {}
        for track in self._tracks:
            if track.entity_id is None:
                continue
            value = public_entities.get(
                track.entity_id, public_entities.get(str(track.entity_id))
            )
            if not isinstance(value, Mapping):
                continue
            position = _position(value)
            track.known = True
            track.position = (position[0], position[1])
            track.velocity_xy = (0.0, 0.0)
            track.velocity_known = True
            track.type_index = OBJECTIVE_TYPE_INDEX.get(int(value.get("type", -1)))
            track.last_seen_step = 0
            track.static_public = True
        self._update_tracks(raw)
        self._last_raw_observation = raw
        self._last_unit_ecf = self._ecf_snapshot(raw)
        self._progress.clear()
        self._reward_tracker = RewardTracker(self.reward_policy)
        self._reward_tracker.check_completion(0, raw)
        self._score = float(self._reward_tracker.finish(raw).score)
        self._launch_count = 0
        self._sensor_request_count = 0
        self._sensor_information_potential = self._sensor_information_value(raw)
        return self._encode(raw)

    def action_mask(self) -> JointActionMask:
        if (
            self._cached_action_mask is not None
            and self._cached_action_mask_step == self.current_step
        ):
            return self._cached_action_mask
        base = build_joint_action_mask(
            self.space,
            self.tracker.states,
            self._objective_validity(),
            self.tracker.sensor_state,
            step=self.current_step,
            allow_staged_sensor=(
                getattr(self, "sensor_backend", "per_unit") == "team_global"
            ),
            objective_valid_by_unit=self._objective_validity_by_unit(),
            routine_retarget_allowed_by_unit=(
                self._routine_retarget_allowed_by_unit()
            ),
            retarget_min_dwell_steps=self.config.retarget_min_dwell_steps,
            retarget_decision_interval_steps=(
                self.config.retarget_decision_interval_steps
            ),
            motion_decision_interval_steps=(
                self.config.motion_decision_interval_steps
            ),
            post_launch_motion_only=self.config.post_launch_motion_only,
        )
        eligible = np.array(base.shared_sensor_eligible, dtype=np.bool_, copy=True)
        backend_remaining = int(eligible.sum())
        backend_step_limit = int(base.shared_sensor_max_requests)
        if getattr(self, "sensor_backend", "per_unit") == "team_global":
            used, maximum, active = self._global_sensor_status()
            backend_remaining = max(0, maximum - used)
            # A second request in the same step only spends another global use
            # and resets the same team-wide window, so never submit more than one.
            backend_step_limit = 0 if active else 1
            if active or backend_remaining <= 0:
                eligible[:] = False
        else:
            for slot in np.flatnonzero(eligible):
                simulator = self.engine.get_simulator_by_id(self.unit_ids[int(slot)])
                if simulator is None:
                    eligible[slot] = False
                    continue
                used = int(getattr(simulator, "RED_SAT_USE_COUNT", 0))
                maximum = int(
                    getattr(
                        simulator,
                        "RED_SAT_MAX_USE_COUNT",
                        self.sensor_backend_capacity_per_unit,
                    )
                )
                is_active = getattr(simulator, "is_using_satellite", None)
                if (
                    used >= maximum
                    or not callable(is_active)
                    or bool(is_active())
                ):
                    eligible[slot] = False
        maximum_requests = min(
            int(base.shared_sensor_max_requests),
            backend_step_limit,
            backend_remaining,
            int(eligible.sum()),
        )
        result = JointActionMask(
            space=base.space,
            by_unit=base.by_unit,
            shared_sensor_eligible=eligible,
            shared_sensor_max_requests=maximum_requests,
        )
        self._cached_action_mask_step = self.current_step
        self._cached_action_mask = result
        return result

    def step(self, action: JointAction) -> JointGameStep:
        if self._last_raw_observation is None or self._reward_tracker is None:
            raise RuntimeError("reset() must be called before step()")
        if self.current_step >= self.max_steps:
            raise RuntimeError("episode has already reached its horizon")

        old_states = self.tracker.states
        validate_action_against_mask(old_states, action, self.action_mask())
        # Tracker state and simulator-side sensor counters can change below.
        # Invalidate immediately so an exception cannot expose a stale mask.
        self._cached_action_mask_step = -1
        self._cached_action_mask = None
        intents = self.tracker.apply(
            action,
            self._objective_validity(),
            step=self.current_step,
            allow_staged_sensor=(
                getattr(self, "sensor_backend", "per_unit") == "team_global"
            ),
        )
        command_dicts: list[dict[str, Any]] = []
        activation_submitted: dict[int, int] = {}
        for intent in intents.activations:
            entity_id = self.unit_ids[intent.slot]
            target = self._target_for_slot(intent.objective_slot)
            lon, lat = denormalize_placement(
                intent.placement,
                self._deployment_bounds[self.unit_types[intent.slot]],
            )
            altitude = 10_000.0 if self.unit_types[intent.slot] == 21002 else 0.0
            if self.engine.get_simulator_by_id(entity_id) is None:
                raise RuntimeError(f"missing simulator for entity {entity_id}")
            self.engine.simulator_factory.modify_simulator_position(
                entity_id,
                {"x": lon, "y": lat, "z": altitude},
            )
            command_dicts.append(
                MissileLaunchAction(
                    executor_id=entity_id,
                    target=Vector3d(target.position[0], target.position[1], 0.0),
                ).to_dict()
            )
            activation_submitted[intent.slot] = intent.request_step

        for intent in intents.retargets:
            target = self._target_for_slot(intent.objective_slot)
            command_dicts.append(
                ChangeTargetAction(
                    executor_id=self.unit_ids[intent.slot],
                    target=Vector3d(target.position[0], target.position[1], 0.0),
                ).to_dict()
            )

        acceleration = {
            Movement.NEGATIVE: -1.0,
            Movement.NEUTRAL: 0.0,
            Movement.POSITIVE: 1.0,
        }
        for intent in intents.movements:
            command_dicts.append(
                SetDesiredAccZ(
                    executor_id=self.unit_ids[intent.slot],
                    acc_z=acceleration[intent.movement],
                ).to_dict()
            )

        sensor_before: dict[int, int] = {}
        sensor_global_before: int | None = None
        if (
            getattr(self, "sensor_backend", "per_unit") == "team_global"
            and intents.shared_sensor
        ):
            sensor_global_before = self._global_sensor_status()[0]
        for intent in intents.shared_sensor:
            entity_id = self.unit_ids[intent.requester_slot]
            if getattr(self, "sensor_backend", "per_unit") == "per_unit":
                simulator = self.engine.get_simulator_by_id(entity_id)
                sensor_before[intent.requester_slot] = int(
                    getattr(simulator, "RED_SAT_USE_COUNT", 0)
                )
            command_dicts.append(
                UseSatelliteAction(executor_id=entity_id).to_dict()
            )

        for command in command_dicts:
            write_ai_action(command, str(self._environment.current_round) + ".json")
        converted = CommandAdapter.common_adapter(command_dicts)
        if len(converted) != len(command_dicts):
            raise RuntimeError(
                "the simulator command adapter rejected one or more joint actions"
            )
        self.engine.step(converted)
        self.current_step += 1
        self._environment.current_step = self.current_step
        raw = self._environment._get_observation()
        write_state(raw, str(self._environment.current_round) + ".json")

        activation_receipts: list[ActivationReceipt] = []
        accepted_activation_slots: list[int] = []
        requested_slots = {
            int(item.slot): int(item.request_step) for item in intents.activations
        }
        for slot, request_step in requested_slots.items():
            accepted = (
                slot in activation_submitted and self._simulator_is_launched(slot)
            )
            activation_receipts.append(
                ActivationReceipt(
                    slot=slot,
                    request_step=request_step,
                    accepted=accepted,
                )
            )
            if accepted:
                accepted_activation_slots.append(slot)
        if activation_receipts:
            self.tracker.confirm_activations(
                activation_receipts,
                step=self.current_step,
            )
            self._launch_count += len(accepted_activation_slots)

        sensor_receipts: list[SharedSensorReceipt] = []
        accepted_sensor_slots: list[int] = []
        global_accept_count = 0
        if sensor_global_before is not None:
            sensor_global_after = self._global_sensor_status()[0]
            global_accept_count = min(
                len(intents.shared_sensor),
                max(0, sensor_global_after - sensor_global_before),
            )
        for request_index, intent in enumerate(intents.shared_sensor):
            if getattr(self, "sensor_backend", "per_unit") == "team_global":
                # Factory processes commands in order.  Attribute a positive
                # global counter delta to that same prefix of submitted intents.
                accepted = request_index < global_accept_count
            else:
                simulator = self.engine.get_simulator_by_id(
                    self.unit_ids[intent.requester_slot]
                )
                after = int(getattr(simulator, "RED_SAT_USE_COUNT", 0))
                accepted = after > sensor_before[intent.requester_slot]
            sensor_receipts.append(
                SharedSensorReceipt(
                    requester_slot=intent.requester_slot,
                    request_step=intent.request_step,
                    accepted=accepted,
                )
            )
            if accepted:
                accepted_sensor_slots.append(intent.requester_slot)
        if sensor_receipts:
            self.tracker.confirm_shared_sensor(
                sensor_receipts,
                step=self.current_step,
            )
            self._sensor_request_count += len(accepted_sensor_slots)

        terminal_slots = [
            slot
            for slot, entity_id in enumerate(self.unit_ids)
            if self.tracker.states[slot].phase != UnitPhase.TERMINAL
            and not _alive(_entity(raw, entity_id))
        ]
        if terminal_slots:
            self.tracker.mark_terminal(terminal_slots, step=self.current_step)

        self._update_tracks(raw)
        self._reward_tracker.check_completion(self.current_step, raw)
        score_now = float(self._reward_tracker.finish(raw).score)
        official_delta = max(0.0, score_now - self._score) / 100.0
        team_reward = self.config.official_reward_scale * official_delta
        self._score = score_now

        score_completed = bool(self._reward_tracker.finish(raw).completed)
        debug_limit = bool(self.is_debug_horizon and self.current_step >= self.max_steps)
        official_limit = bool(
            not self.is_debug_horizon
            and self.current_step >= int(self.reward_policy.max_steps)
        )
        natural_done = bool(self._environment.get_is_done())
        team_terminated = bool(
            official_limit
            or natural_done
            or (
                self.config.terminate_on_all_objectives_destroyed
                and score_completed
            )
        )
        team_truncated = bool(debug_limit and not team_terminated)
        done = team_terminated or team_truncated

        # Sensor shaping uses only information present in controller-visible
        # observations.  New global-satellite environments expose fresh
        # interceptor tracks; legacy per-unit environments exposed objectives.
        # A true terminal closes the potential at zero while a deliberately
        # shortened debug horizon keeps it for value bootstrapping.
        current_information_potential = self._sensor_information_value(raw)
        sensor_reward = self._advance_sensor_information_reward(
            team_reward=team_reward,
            current_potential=current_information_potential,
            true_terminal=team_terminated,
        )

        rewards = self._unit_rewards(
            old_states,
            raw,
            team_reward=team_reward,
            force_terminal=team_terminated,
        )
        assignment_states = self.tracker.states
        if team_terminated:
            remaining_slots = [
                slot
                for slot, state in enumerate(self.tracker.states)
                if state.phase != UnitPhase.TERMINAL
            ]
            if remaining_slots:
                self.tracker.mark_terminal(remaining_slots, step=self.current_step)
        current_states = self.tracker.states
        terminated = tuple(
            bool(
                team_terminated
                or (
                    old_states[slot].phase != UnitPhase.TERMINAL
                    and current_states[slot].phase == UnitPhase.TERMINAL
                )
            )
            for slot in range(self.space.unit_count)
        )
        truncated = tuple(
            bool(team_truncated and not terminated[slot])
            for slot in range(self.space.unit_count)
        )
        observations = self._encode(raw)
        self._last_raw_observation = raw
        self._last_unit_ecf = self._ecf_snapshot(raw)
        if self._environment.render_mode == "human" and self._environment.renderer:
            self._environment.renderer.update_data(raw["entities"])
        return JointGameStep(
            observations=observations,
            states=current_states,
            rewards=rewards,
            team_reward=float(team_reward),
            terminated=terminated,
            truncated=truncated,
            team_terminated=team_terminated,
            team_truncated=team_truncated,
            done=done,
            score=score_now,
            raw_observation=raw,
            submitted_commands=len(converted),
            accepted_activations=tuple(accepted_activation_slots),
            accepted_sensor_requests=tuple(accepted_sensor_slots),
            sensor_reward=float(sensor_reward),
            assignment_states=assignment_states,
        )

    def _objective_information_potential(self) -> float:
        """Value of legally known objective tracks for sensor credit."""

        known_slots = (
            track.slot
            for track in self._tracks
            if track.entity_id is not None and track.known
        )
        return objective_information_potential(
            known_objective_slots=known_slots,
            objective_weights=self._objective_weight_by_slot,
            scale=self.config.sensor_information_potential_scale,
        )

    def _detection_age_steps(self, detection: Any) -> float | None:
        """Convert an upstream millisecond detection timestamp to step age."""

        try:
            timestamp = float(_field(detection, "time", self.engine.sim_time))
            sim_time = float(self.engine.sim_time)
            sim_step = max(
                float(self.engine.profile.imagineProfile.simStep), 1.0
            )
        except (AttributeError, TypeError, ValueError):
            return None
        if not math.isfinite(timestamp) or not math.isfinite(sim_time):
            return None
        return max(0.0, (sim_time - timestamp) / sim_step)

    def _fresh_interceptor_track_ages(
        self, observation: Mapping[str, Any]
    ) -> dict[int, float]:
        """Union fresh, positioned interceptor detections across the red team."""

        freshest: dict[int, float] = {}
        for entity_id in self.unit_ids:
            own = _entity(observation, entity_id)
            for key, detection in ((own or {}).get("detectInfo") or {}).items():
                try:
                    if int(_field(detection, "entity_type", -1)) != 24000:
                        continue
                    detected_id = int(_field(detection, "entity_id", key))
                except (TypeError, ValueError):
                    continue
                if _vector(_field(detection, "lla")) is None:
                    continue
                age = self._detection_age_steps(detection)
                if age is None or age > self.config.max_track_age_steps:
                    continue
                freshest[detected_id] = min(age, freshest.get(detected_id, age))
        return freshest

    def _interceptor_information_potential(
        self, observation: Mapping[str, Any]
    ) -> float:
        return interceptor_track_information_potential(
            threat_age_steps=self._fresh_interceptor_track_ages(observation).values(),
            max_track_age_steps=self.config.max_track_age_steps,
            threat_normalizer=self._sensor_threat_normalizer,
            scale=self.config.sensor_information_potential_scale,
        )

    def _sensor_information_value(self, observation: Mapping[str, Any]) -> float:
        if getattr(self, "sensor_backend", "per_unit") == "team_global":
            return self._interceptor_information_potential(observation)
        return self._objective_information_potential()

    def _advance_sensor_information_reward(
        self,
        *,
        team_reward: float,
        current_potential: float,
        true_terminal: bool,
    ) -> float:
        """Apply sensor PBRS with correct terminal versus truncation semantics."""

        next_potential = 0.0 if true_terminal else float(current_potential)
        shaped = float(team_reward) + potential_difference(
            previous=self._sensor_information_potential,
            current=next_potential,
            gamma=self.config.gamma,
        )
        self._sensor_information_potential = next_potential
        return float(shaped)

    def _objective_validity(self) -> tuple[bool, ...]:
        if not self.config.strict_weapon_target_compatibility:
            # Preserve the historical never-expire target table for exact
            # continuation of legacy checkpoints.
            return tuple(
                bool(track.entity_id is not None and track.known)
                for track in self._tracks
            )
        return tuple(
            bool(
                track.entity_id is not None
                and track.known
                and (
                    track.static_public
                    or self.current_step - track.last_seen_step
                    <= self.config.max_track_age_steps
                )
            )
            for track in self._tracks
        )

    def _objective_validity_by_unit(self) -> tuple[tuple[bool, ...], ...]:
        """Return type-aware target masks without deadlocking ship discovery.

        H/M weapons never damage ships, so they remain restricted to land
        objectives.  L weapons damage only ships, but an L weapon must already
        be airborne to discover a hidden ship with its local sensor.  Until the
        first ship is legally known, public land objectives may therefore act
        as navigation/search anchors.  As soon as any ship is known, those
        anchors disappear and every L weapon can retarget immediately because
        its former anchor is no longer valid for that unit.
        """

        objective_valid = self._objective_validity()
        if not self.config.strict_weapon_target_compatibility:
            return tuple(objective_valid for _ in self.unit_types)

        compatible_types = {
            21000: frozenset((0, 1)),
            21001: frozenset((0, 1)),
            21002: frozenset((2,)),
        }
        ship_known = any(
            is_valid and track.type_index == 2
            for is_valid, track in zip(
                objective_valid,
                self._tracks,
                strict=True,
            )
        )
        return tuple(
            tuple(
                bool(
                    is_valid
                    and (
                        track.type_index
                        in compatible_types.get(unit_type, frozenset())
                        or (
                            unit_type == 21002
                            and self.config.allow_low_altitude_search_fallback
                            and not ship_known
                            and track.type_index in (0, 1)
                        )
                    )
                )
                for is_valid, track in zip(
                    objective_valid,
                    self._tracks,
                    strict=True,
                )
            )
            for unit_type in self.unit_types
        )

    def _routine_retarget_allowed_by_unit(self) -> tuple[bool, ...]:
        """Lock an L missile's land search waypoint until its role changes.

        A land objective is only a navigation/search anchor for an L missile,
        not a damage-compatible mission target. Reconsidering peer land
        anchors at every ordinary retarget pulse caused periodic target churn.
        The current anchor therefore stays locked while it remains legal.

        This gate only suppresses routine retarget pulses. When a fresh ship
        becomes known, the per-unit objective mask makes the land anchor
        invalid, and the generic invalid-target bypass still unlocks an
        immediate mission retarget. No extra controller state is required.
        """

        if (
            not self.config.strict_weapon_target_compatibility
            or not self.config.allow_low_altitude_search_fallback
            or self.config.allow_low_altitude_search_replanning
        ):
            return tuple(True for _ in self.unit_types)

        allowed: list[bool] = []
        for state, unit_type in zip(
            self.tracker.states,
            self.unit_types,
            strict=True,
        ):
            objective_slot = int(state.current_objective_slot)
            current_is_search_anchor = bool(
                state.phase == UnitPhase.ACTIVE
                and unit_type == 21002
                and 0 <= objective_slot < len(self._tracks)
                and self._tracks[objective_slot].type_index in (0, 1)
            )
            allowed.append(not current_is_search_anchor)
        return tuple(allowed)

    def assignment_is_damage_compatible(
        self,
        unit_slot: int,
        objective_slot: int,
    ) -> bool:
        """Whether an assignment can directly damage that objective type.

        Temporary L-to-land search anchors are intentionally excluded.  They
        may help discover a hidden ship, but must not share local credit for
        damage actually caused by an H/M weapon.
        """

        if not 0 <= unit_slot < len(self.unit_types):
            raise IndexError("unit slot is out of range")
        if not 0 <= objective_slot < len(self._tracks):
            raise IndexError("objective slot is out of range")
        if not self.config.strict_weapon_target_compatibility:
            # Exact legacy continuation: permissive action masks historically
            # also treated every assignment as local-credit participation.
            return True
        compatible_types = {
            21000: frozenset((0, 1)),
            21001: frozenset((0, 1)),
            21002: frozenset((2,)),
        }
        return bool(
            self._tracks[objective_slot].type_index
            in compatible_types.get(self.unit_types[unit_slot], frozenset())
        )

    def _target_for_slot(self, slot: int) -> _ObjectiveTrack:
        if slot < 0 or slot >= len(self._tracks):
            raise IndexError("objective slot is out of range")
        track = self._tracks[slot]
        if track.entity_id is None or not track.known:
            raise ValueError(f"objective slot {slot} is not currently known")
        return track

    def _simulator_is_launched(self, slot: int) -> bool:
        simulator = self.engine.get_simulator_by_id(self.unit_ids[slot])
        if simulator is None:
            return False
        value = getattr(simulator, "launch", None)
        if value is None:
            return False
        return bool(float(value) >= 0.0) if self.unit_types[slot] == 21002 else bool(float(value) != 0.0)

    def _update_tracks(self, observation: Mapping[str, Any]) -> None:
        for entity_id in self.unit_ids:
            own = _entity(observation, entity_id)
            if own is None:
                continue
            for key, detection in (own.get("detectInfo") or {}).items():
                try:
                    detected_id = int(_field(detection, "entity_id", key))
                except (TypeError, ValueError):
                    continue
                slot = self.objective_slot_by_id.get(detected_id)
                if slot is None:
                    continue
                lla = _vector(_field(detection, "lla"))
                if lla is None:
                    continue
                track = self._tracks[slot]
                timestamp = int(_field(detection, "time", self.current_step))
                if track.known and timestamp <= track.source_timestamp:
                    continue
                track.known = True
                track.position = (lla[0], lla[1])
                velocity = _vector(_field(detection, "vel_ecf"))
                track.velocity_known = velocity is not None
                track.velocity_xy = (
                    _ecf_velocity_to_enu_xy(velocity, lla)
                    if velocity is not None
                    else (0.0, 0.0)
                )
                entity_type = int(_field(detection, "entity_type", -1))
                track.type_index = OBJECTIVE_TYPE_INDEX.get(entity_type)
                track.last_seen_step = self.current_step
                track.source_timestamp = timestamp

    def _ecf_snapshot(
        self, observation: Mapping[str, Any]
    ) -> dict[int, tuple[float, float, float]]:
        result: dict[int, tuple[float, float, float]] = {}
        for entity_id in self.unit_ids:
            value = _entity(observation, entity_id)
            vector = _vector((value or {}).get("pos_ecf"))
            if vector is not None and all(math.isfinite(item) for item in vector):
                result[entity_id] = vector
        return result

    def _unit_frames(
        self, observation: Mapping[str, Any]
    ) -> tuple[UnitFrame, ...]:
        seconds = max(float(self.engine.profile.imagineProfile.simStep) / 1000.0, 1e-6)
        frames: list[UnitFrame] = []
        for slot, entity_id in enumerate(self.unit_ids):
            value = _entity(observation, entity_id)
            state = self.tracker.states[slot]
            known_position = bool(value is not None and state.phase != UnitPhase.STAGED)
            current_ecf = _vector((value or {}).get("pos_ecf"))
            previous_ecf = self._last_unit_ecf.get(entity_id)
            velocity_known = bool(
                current_ecf is not None
                and previous_ecf is not None
                and state.phase == UnitPhase.ACTIVE
                and state.activated_step is not None
                and self.current_step > state.activated_step
            )
            own_position = _position(value)
            velocity_ecf = (
                tuple(
                    (current_ecf[index] - previous_ecf[index]) / seconds
                    for index in range(3)
                )
                if velocity_known and current_ecf is not None and previous_ecf is not None
                else None
            )
            velocity = (
                _ecf_velocity_to_enu_xy(velocity_ecf, own_position)
                if velocity_ecf is not None
                else (0.0, 0.0)
            )
            threats: list[
                tuple[float, Any, tuple[float, float, float], int]
            ] = []
            for key, detection in ((value or {}).get("detectInfo") or {}).items():
                try:
                    if int(_field(detection, "entity_type", -1)) != 24000:
                        continue
                except (TypeError, ValueError):
                    continue
                threat_lla = _vector(_field(detection, "lla"))
                if threat_lla is None:
                    continue
                raw_age = self._detection_age_steps(detection)
                threat_age = 0 if raw_age is None else max(0, int(raw_age))
                if (
                    getattr(self, "sensor_backend", "per_unit") == "team_global"
                    and (
                        raw_age is None
                        or raw_age > self.config.max_track_age_steps
                    )
                ):
                    continue
                distance = self._distance_km(
                    (own_position[0], own_position[1]),
                    (threat_lla[0], threat_lla[1]),
                )
                threats.append((distance, detection, threat_lla, threat_age))
            nearest = min(threats, key=lambda item: item[0]) if threats else None
            nearest_velocity = (
                _vector(_field(nearest[1], "vel_ecf")) if nearest is not None else None
            )
            threat_age = nearest[3] if nearest is not None else 0
            progress_reference, progress_fraction = self._target_progress_features(
                slot,
                own_position,
            )
            frames.append(
                UnitFrame(
                    slot=slot,
                    type_index=UNIT_TYPE_INDEX[self.unit_types[slot]],
                    position=_position(value),
                    velocity_xy=velocity,
                    position_known=known_position,
                    velocity_known=velocity_known,
                    health_fraction=(
                        float((value or {}).get("health", 0.0))
                        / self._initial_unit_health[slot]
                    ),
                    visible=bool((value or {}).get("isVisible", False)),
                    detected_threat_count=len(threats),
                    nearest_threat_known=nearest is not None,
                    nearest_threat_position=(
                        (nearest[2][0], nearest[2][1])
                        if nearest is not None
                        else (0.0, 0.0)
                    ),
                    nearest_threat_velocity_xy=(
                        _ecf_velocity_to_enu_xy(nearest_velocity, nearest[2])
                        if nearest_velocity is not None
                        else (0.0, 0.0)
                    ),
                    nearest_threat_velocity_known=nearest_velocity is not None,
                    nearest_threat_age_steps=threat_age,
                    target_progress_reference=progress_reference,
                    target_progress_fraction=progress_fraction,
                )
            )
        return tuple(frames)

    def _target_progress_features(
        self,
        slot: int,
        own_position: tuple[float, float, float],
    ) -> tuple[float, float]:
        """Expose the distance-potential state needed by the value function."""

        state = self.tracker.states[slot]
        progress = self._progress.get(slot)
        if (
            state.phase != UnitPhase.ACTIVE
            or progress is None
            or progress[0] != state.current_objective_slot
        ):
            return 0.0, 0.0
        objective_slot, reference, _ = progress
        if not 0 <= objective_slot < len(self._tracks):
            raise RuntimeError("progress state has an invalid objective slot")
        target = self._tracks[objective_slot]
        if not target.known:
            return 0.0, 0.0
        values = (*own_position[:2], *target.position, reference, self._map_diagonal_km)
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError("progress state contains a non-finite value")
        if reference <= 0.0 or self._map_diagonal_km <= 0.0:
            raise RuntimeError("progress distance scales must be positive")
        distance = self._distance_km(
            (own_position[0], own_position[1]),
            target.position,
        )
        return (
            float(np.clip(reference / self._map_diagonal_km, 0.0, 1.0)),
            float(np.clip((reference - distance) / reference, -1.0, 1.0)),
        )

    def _objective_frames(self) -> tuple[ObjectiveFrame, ...]:
        loads = self._objective_loads()
        validity = self._objective_validity()
        return tuple(
            ObjectiveFrame(
                slot=track.slot,
                valid=validity[track.slot],
                known=track.known,
                position=track.position,
                velocity_xy=track.velocity_xy,
                velocity_known=track.velocity_known,
                age_steps=(
                    0
                    if track.static_public
                    else max(0, self.current_step - track.last_seen_step)
                ),
                type_index=track.type_index,
                assigned_total=loads[track.slot][0],
                assigned_high=loads[track.slot][1],
                assigned_medium=loads[track.slot][2],
                assigned_low=loads[track.slot][3],
            )
            for track in self._tracks
        )

    def _objective_loads(self) -> tuple[tuple[float, float, float, float], ...]:
        """Return controller-owned target load fractions.

        The total channel is normalized by the complete red team.  The H/M/L
        channels are each normalized by that missile type's team total.  Only
        currently ACTIVE units contribute, so activation receipts, retargets,
        and terminal transitions are reflected without a second mutable load
        tracker.  No enemy health or other hidden simulator state is read.
        """

        states = self.tracker.states
        if len(states) != len(self.unit_types):
            raise RuntimeError("controller states and red unit types are misaligned")
        objective_count = int(self.space.objective_count)
        counts = np.zeros((objective_count, 4), dtype=np.int64)
        type_totals = {
            entity_type: self.unit_types.count(entity_type)
            for entity_type in RED_MISSILE_TYPES
        }
        for expected_slot, (state, entity_type) in enumerate(
            zip(states, self.unit_types, strict=True)
        ):
            if state.slot != expected_slot:
                raise RuntimeError("controller states are not in stable slot order")
            if entity_type not in UNIT_TYPE_INDEX:
                raise RuntimeError(f"unsupported red unit type {entity_type}")
            if state.phase != UnitPhase.ACTIVE:
                continue
            objective_slot = int(state.current_objective_slot)
            if not 0 <= objective_slot < objective_count:
                raise RuntimeError(
                    f"active unit {state.slot} has an invalid objective slot"
                )
            counts[objective_slot, 0] += 1
            counts[objective_slot, 1 + UNIT_TYPE_INDEX[entity_type]] += 1

        total_units = len(states)
        denominators = (
            total_units,
            type_totals[21000],
            type_totals[21001],
            type_totals[21002],
        )
        result = tuple(
            tuple(
                0.0 if denominator == 0 else float(counts[slot, index] / denominator)
                for index, denominator in enumerate(denominators)
            )
            for slot in range(objective_count)
        )
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for load in result
            for value in load
        ):
            raise RuntimeError("objective load normalization produced an invalid value")
        return result

    def _encode(
        self, observation: Mapping[str, Any]
    ) -> tuple[np.ndarray, ...]:
        units = self._unit_frames(observation)
        objectives = self._objective_frames()
        return self.encoder.encode_many(
            units,
            self.tracker.states,
            objectives,
            self.tracker.sensor_state,
            step=self.current_step,
            sensor_ready_override=self._sensor_ready_override(),
        )

    def _unit_rewards(
        self,
        old_states: Sequence[UnitControlState],
        observation: Mapping[str, Any],
        *,
        team_reward: float,
        force_terminal: bool,
    ) -> tuple[float, ...]:
        rewards = [
            float(team_reward) if state.phase != UnitPhase.TERMINAL else 0.0
            for state in old_states
        ]
        for slot, old_state in enumerate(old_states):
            if old_state.phase == UnitPhase.TERMINAL:
                continue
            current = self.tracker.states[slot]
            progress = self._progress.get(slot)
            if force_terminal or current.phase == UnitPhase.TERMINAL:
                if progress is not None:
                    rewards[slot] -= progress[2]
                    self._progress.pop(slot, None)
                continue
            if current.phase != UnitPhase.ACTIVE:
                if progress is not None:
                    rewards[slot] -= progress[2]
                    self._progress.pop(slot, None)
                continue
            objective_slot = current.current_objective_slot
            target = (
                self._tracks[objective_slot]
                if 0 <= objective_slot < len(self._tracks)
                else None
            )
            own = _entity(observation, self.unit_ids[slot])
            if target is None or not target.known or own is None:
                if progress is not None:
                    rewards[slot] -= progress[2]
                    self._progress.pop(slot, None)
                continue
            own_position = _position(own)
            distance = self._distance_km(
                (own_position[0], own_position[1]),
                target.position,
            )
            if progress is None or progress[0] != objective_slot:
                if progress is not None:
                    rewards[slot] -= progress[2]
                self._progress[slot] = (
                    objective_slot,
                    max(distance, 25.0),
                    0.0,
                )
                continue
            _, reference, previous_potential = progress
            fraction = float(np.clip((reference - distance) / reference, -1.0, 1.0))
            potential = self.config.progress_potential_scale * fraction
            rewards[slot] += (
                self.config.gamma * potential - previous_potential
            )
            self._progress[slot] = (objective_slot, reference, potential)
        return tuple(float(value) for value in rewards)

    @staticmethod
    def _distance_km(
        first: tuple[float, float], second: tuple[float, float]
    ) -> float:
        lon1, lat1, lon2, lat2 = map(
            math.radians, (first[0], first[1], second[0], second[1])
        )
        dlon, dlat = lon2 - lon1, lat2 - lat1
        value = (
            math.sin(dlat / 2.0) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
        )
        return 6371.0 * 2.0 * math.atan2(
            math.sqrt(max(0.0, value)),
            math.sqrt(max(0.0, 1.0 - value)),
        )

    def close(self) -> None:
        self._environment.close()
