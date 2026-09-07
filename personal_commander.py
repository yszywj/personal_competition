"""A legal R9 Commander extension for dynamically detected 9500 targets."""

from __future__ import annotations

from typing import Any, Mapping

from .bootstrap import install_project_paths

install_project_paths()

from policies.red.baselines import TargetPrior  # noqa: E402
from policies.red.commander import RedBaselineCommander  # noqa: E402
from policies.red.contracts import Position  # noqa: E402


class DynamicDetectedTargetTrackFusion:
    """Maintain initial priors plus 9500 ships seen in legal detectInfo.

    No scenario/root catalogue is read here.  A ship's ID and coordinates enter
    the planner only after at least one red platform receives that track.
    DetectInfo has no health field, so a discovered target remains as a last
    known track rather than being silently declared dead.
    """

    DISCOVERABLE_DYNAMIC_TYPE = 9500
    DYNAMIC_TARGET_VALUE = 3.0

    def __init__(self, targets: tuple[TargetPrior, ...]) -> None:
        self._targets = {int(item.entity_id): item for item in targets}
        self._track_times: dict[int, float | None] = {}

    @property
    def targets(self) -> tuple[TargetPrior, ...]:
        return tuple(sorted(self._targets.values(), key=lambda item: item.entity_id))

    def ingest(self, observation: Mapping[str, Any]) -> bool:
        changed = False
        tracks = observation.get("self", {}).get("detectInfo") or {}
        for raw_id, track in tracks.items():
            entity_id = int(self._field(track, "entity_id", raw_id))
            current = self._targets.get(entity_id)
            default_type = current.entity_type if current is not None else -1
            entity_type = int(self._field(track, "entity_type", default_type))
            if current is None and entity_type != self.DISCOVERABLE_DYNAMIC_TYPE:
                continue
            if current is not None and entity_type != current.entity_type:
                continue
            lla = self._field(track, "lla", None)
            if lla is None:
                continue
            raw_time = self._field(track, "time", None)
            track_time = float(raw_time) if raw_time is not None else None
            if entity_id in self._track_times:
                previous_time = self._track_times[entity_id]
                # Native DetectInfo coordinates can reference mutable simulator
                # vectors.  Without a newer report timestamp, a changing lla
                # is not new legal information and must not move the track.
                if (
                    track_time is None
                    or (previous_time is not None and track_time <= previous_time)
                ):
                    continue
            position = Position(
                lon=float(self._field(lla, "x", current.position.lon if current else 0.0)),
                lat=float(self._field(lla, "y", current.position.lat if current else 0.0)),
                alt=float(self._field(lla, "z", current.position.alt if current else 0.0)),
            )
            replacement = TargetPrior(
                entity_id=entity_id,
                entity_type=entity_type,
                position=position,
                value=(current.value if current is not None else self.DYNAMIC_TARGET_VALUE),
                alive=(current.alive if current is not None else True),
            )
            if replacement != current:
                self._targets[entity_id] = replacement
                changed = True
            self._track_times[entity_id] = track_time
        return changed

    @staticmethod
    def _field(value: Any, name: str, default: Any) -> Any:
        return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


class PersonalR9Commander(RedBaselineCommander):
    """Original R9 policy plus organizer-approved dynamic ship discovery."""

    def __init__(self, targets: tuple[TargetPrior, ...], *, seed: int = 1) -> None:
        super().__init__(
            targets,
            policy_name="r9_hierarchical_learning",
            seed=seed,
        )
        self.track_fusion = DynamicDetectedTargetTrackFusion(self.initial_targets)
        self._step_launch_fraction = 0.0

    def begin_step(self, observations: tuple[dict, ...]) -> None:
        """Plan once from the joint legal snapshot before querying any Agent.

        The base Commander otherwise plans lazily inside the first
        ``action_for`` call.  Eager planning makes the target context stored in
        PPO's next state identical to the context used for the next action and
        prevents Python Agent iteration order from changing feature 90.
        """

        super().begin_step(observations)
        if observations:
            step = int(observations[0].get("step", 0))
            if self._should_plan(step):
                self._plan(step)
        self._step_launch_fraction = len(self.launched_ids) / max(
            1, len(self.expected_platform_ids)
        )

    def learning_task_context(
        self, platform_id: int
    ) -> tuple[float, float, float, float, float]:
        context = super().learning_task_context(platform_id)
        return (*context[:4], self._step_launch_fraction)

    def reset(self) -> None:
        super().reset()
        # Never carry a detected ship from one episode into the next one.
        self.targets = self.initial_targets
        self.track_fusion = DynamicDetectedTargetTrackFusion(self.initial_targets)
        self._step_launch_fraction = 0.0
