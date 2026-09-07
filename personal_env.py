"""Training-only environment wrapper and reward shaping for R9 PPO."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .bootstrap import install_project_paths
from .personal_agent import PersonalR9PPOAttackAgent, ResetAwareAgentManager

install_project_paths()

from envengine import TrainingEnv  # noqa: E402
from envengine.common import SimmerTriggerType  # noqa: E402
from scenarios.cases import RewardPolicy, RewardTracker  # noqa: E402


RED_MISSILE_TYPES = (21000, 21001, 21002)
INTERCEPTOR_TYPE = 24000


@dataclass(frozen=True)
class R9RewardConfig:
    """Score-aligned reward with small, bounded shaping terms."""

    # Each missile's target-progress potential is bounded to +/- this value.
    progress_potential_scale: float = 0.10
    progress_reference_min_km: float = 25.0
    # Keep potential shaping policy-invariant for the trainer's discount.
    progress_discount: float = 0.999
    # Sparse damage shaping.  Across all objectives its theoretical maximum is
    # the sum of official objective weights, not hundreds of points.
    damage_scale: float = 1.0
    # Every target-health change receives its exact 0--100 official-score
    # increment under the current weighted-damage judge.
    official_score_scale: float = 100.0
    # No per-flight cost: otherwise dying early avoids future cost.  Direction
    # changes retain only a very small regularizer.
    flight_step_cost: float = 0.0
    action_switch_cost: float = 0.0002
    # Death and surviving an unsuccessful time limit use the same type cost.
    high_loss_cost: float = 1.00
    medium_loss_cost: float = 0.35
    low_loss_cost: float = 0.10

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class PersonalR9TrainingEnv(TrainingEnv):
    """Original environment lifecycle with local reset and reward corrections."""

    # Nominal expected damage mirrors the simulator's public damage/rate table.
    # It is used only to split simultaneous health loss among source missiles;
    # the rewarded amount always comes from the observed health delta.
    _DAMAGE_POTENTIAL = {
        (21000, 9400): 16.0,
        (21000, 9600): 12.0,
        (21001, 9400): 4.0,
        (21001, 9600): 3.0,
        (21002, 9500): 0.8,
    }

    def __init__(
        self,
        profile,
        *,
        reward_policy: RewardPolicy,
        reward_config: R9RewardConfig | None = None,
        render_mode: str | None = None,
        render_fps: int = 10,
    ) -> None:
        if render_fps <= 0:
            raise ValueError("render_fps must be positive")
        super().__init__(profile, render_mode=render_mode)
        if self.renderer is not None:
            # The core renderer redraws the complete scene every frame.  A
            # lower display-only FPS greatly reduces VNC/X11 contention while
            # leaving every simulation step and policy decision untouched.
            self.renderer.fps = int(render_fps)
        # The base manager is empty here, so replacement cannot discard agents.
        self.agent_manager = ResetAwareAgentManager()
        self.reward_policy = reward_policy
        self.reward_config = reward_config or R9RewardConfig()
        self.learning_team_size = 0
        # Competition case_info is the authority for the episode horizon.
        self.max_steps = int(reward_policy.max_steps)
        self._reward_tracker: RewardTracker | None = None
        self._reward_previous_observation: Mapping[str, Any] | None = None
        # entity -> (target_id, reference_distance_km, previous_potential)
        self._progress_state: dict[int, tuple[int, float, float]] = {}
        self._relation_offsets: dict[int, int] = {}
        self._successful_sources: set[int] = set()
        self._settled_failures: set[int] = set()
        self._previous_official_score = 0.0
        self._raw_official_score_delta = 0.0
        self._episode_reward_components: defaultdict[str, float] = defaultdict(float)
        self._episode_agent_returns: defaultdict[int, float] = defaultdict(float)
        self._initial_health_by_objective = self._load_initial_objective_health()
        # The core writer currently does not persist destroy events.  Capture
        # their causal source locally so future summaries can distinguish a
        # normal missile terminal from a genuine interceptor kill.
        self._red_destroy_causes: dict[int, dict[str, int | float]] = {}
        self._install_destroy_event_capture()

    def reset(self) -> dict:
        # TrainingEnv.reset() calls our ResetAwareAgentManager.reset_all(), so a
        # shared Commander is reset exactly once even on an unpatched checkout.
        self._red_destroy_causes.clear()
        observation = super().reset()
        self._restore_simulator_clocks()
        self._reset_reward_state(observation)
        return observation

    def _restore_simulator_clocks(self) -> None:
        """Restore every simulator clock to the scenario's initial logic time.

        The current upstream factory resets the engine clock but leaves each
        simulator's private clock at zero.  Commands can be dispatched before
        the first simulator update, so the per-simulator clocks must already
        match the scenario clock when ``reset()`` returns.
        """

        initial_time = float(self.engine.profile.imagineProfile.simTime)
        for simulator in self.engine.simulator_factory.get_all_simulators():
            simulator.sim_time = initial_time

    def _install_destroy_event_capture(self) -> None:
        for simulator in self.engine.simulator_factory.get_all_simulators():
            original_sender = simulator._send_events

            def capture(event, sender=original_sender):
                self._capture_destroy_event(event)
                sender(event)

            simulator._send_events = capture

    def _capture_destroy_event(self, event: Mapping[str, Any]) -> None:
        try:
            if int(event.get("type", -1)) != int(SimmerTriggerType.DESTROY):
                return
            payload = event.get("content", {}).get("event", {})
            victim = payload.get("entity", {})
            source = payload.get("destroySrc", {})
            victim_id = int(victim.get("id", -1))
            victim_type = int(victim.get("type", -1))
            if victim_id < 0 or victim_type not in RED_MISSILE_TYPES:
                return
            # ISimulator may emit another destroy event if overlapping damage
            # commands arrive after health already reached zero.  The first
            # event is the actual terminal cause under engine command order.
            self._red_destroy_causes.setdefault(
                victim_id,
                {
                    "victim_type": victim_type,
                    "source_id": int(source.get("id", -1)),
                    "source_type": int(source.get("type", -1)),
                    "logic_time_ms": float(event.get("logicTime", payload.get("time", 0.0))),
                },
            )
        except (AttributeError, TypeError, ValueError):
            # Reporting must never alter the competition simulation path.
            return

    def interception_metrics(self) -> dict[str, Any]:
        """Return causally attributed interception statistics for this round."""

        interceptors = self.engine.simulator_factory.get_simulators_by_type(
            INTERCEPTOR_TYPE
        )
        launched_ids = {
            int(simulator.entity_ext.entity.id)
            for simulator in interceptors
            if int(getattr(simulator, "launched", -1)) != -1
        }
        alive_ids = {
            int(simulator.entity_ext.entity.id)
            for simulator in interceptors
            if float(simulator.entity_ext.entity.survivePoints) > 0.0
        }
        kills_by_interceptor: Counter[int] = Counter()
        intercepted_by_type: Counter[int] = Counter()
        intercepted_ids: list[int] = []
        for victim_id, cause in self._red_destroy_causes.items():
            if int(cause["source_type"]) != INTERCEPTOR_TYPE:
                continue
            source_id = int(cause["source_id"])
            kills_by_interceptor[source_id] += 1
            intercepted_by_type[int(cause["victim_type"])] += 1
            intercepted_ids.append(int(victim_id))

        successful_ids = set(kills_by_interceptor)
        histogram = Counter(kills_by_interceptor.values())
        return {
            "red_missiles_intercepted": len(intercepted_ids),
            "red_missiles_intercepted_by_type": {
                "high": int(intercepted_by_type.get(21000, 0)),
                "medium": int(intercepted_by_type.get(21001, 0)),
                "low": int(intercepted_by_type.get(21002, 0)),
            },
            "interceptors_total": len(interceptors),
            "interceptors_launched": len(launched_ids),
            "interceptors_alive_at_end": len(alive_ids),
            "successful_interceptors": len(successful_ids),
            "launched_without_a_kill": len(launched_ids - successful_ids),
            "multi_kill_interceptors": sum(
                count for kills, count in histogram.items() if kills > 1
            ),
            "max_kills_by_one_interceptor": max(
                kills_by_interceptor.values(), default=0
            ),
            "kills_per_interceptor_histogram": {
                str(kills): int(count)
                for kills, count in sorted(histogram.items())
            },
            "intercepted_red_ids": sorted(intercepted_ids),
            "successful_interceptor_ids": sorted(successful_ids),
        }

    def prepare_round_after_deploy(self) -> None:
        """Anchor shaping state after random deployment and before step zero."""

        observation = self._get_observation()
        self._reward_previous_observation = observation
        self._anchor_official_score(observation)
        self._progress_state.clear()
        self._relation_offsets = {
            int(target_id): len(relations)
            for target_id, relations in self.engine.simulator_factory.target_hit_relation.items()
        }
        self._last_observation = observation.copy()

    def reward_metrics(self) -> dict[str, Any]:
        returns = list(self._episode_agent_returns.values())
        return {
            "components": dict(sorted(self._episode_reward_components.items())),
            "agent_return_sum": float(sum(returns)),
            "agent_return_mean": float(sum(returns) / max(1, len(returns))),
            "agents_with_transitions": len(returns),
            "raw_official_score_delta": float(self._raw_official_score_delta),
        }

    def _reset_reward_state(self, observation: Mapping[str, Any]) -> None:
        self._reward_tracker = RewardTracker(self.reward_policy)
        self._reward_previous_observation = observation
        self._anchor_official_score(observation)
        self._progress_state.clear()
        # Anchor the native hit-relation append-only lists at the new episode.
        # Current simulator versions clear them during Engine.reset(), but
        # taking an explicit cursor also keeps this wrapper safe with older
        # builds that retained diagnostic history across rounds.
        self._relation_offsets = {
            int(target_id): len(relations)
            for target_id, relations in self.engine.simulator_factory.target_hit_relation.items()
        }
        self._successful_sources.clear()
        self._settled_failures.clear()
        self._raw_official_score_delta = 0.0
        self._episode_reward_components.clear()
        self._episode_agent_returns.clear()

    def _compute_reward(self) -> dict[int, float]:
        current = self._get_observation()
        previous = self._reward_previous_observation or current
        agents = self.agent_manager.get_all_agents()
        rewards = {int(agent.agent_id): 0.0 for agent in agents}
        by_entity = {int(agent.entity_id): agent for agent in agents}
        acting = [
            agent
            for agent in agents
            if isinstance(agent, PersonalR9PPOAttackAgent)
            and getattr(agent, "_learning_transition", None) is not None
        ]
        acting_ids = {int(agent.entity_id) for agent in acting}

        for agent in acting:
            if self.reward_config.flight_step_cost:
                self._credit(
                    rewards,
                    agent,
                    "flight_cost",
                    -self.reward_config.flight_step_cost,
                )
            if agent.switched_on_last_action:
                self._credit(
                    rewards,
                    agent,
                    "action_switch_cost",
                    -self.reward_config.action_switch_cost,
                )
            self._credit_progress(rewards, agent, current)

        damage_sources = self._credit_actual_damage(
            rewards, by_entity, acting_ids, previous, current
        )
        self._successful_sources.update(damage_sources)
        objectives_completed = self._objectives_completed(current)
        self._credit_losses(
            rewards,
            by_entity,
            acting_ids,
            previous,
            current,
            damage_sources,
            mission_completed=objectives_completed,
        )
        self._credit_official_score_delta(rewards, acting, current)

        if self.current_step >= self.max_steps and not objectives_completed:
            self._credit_timeout_losses(rewards, acting, current)

        # The action and progress reward above belong to the target assignment
        # from s.  Only now ingest s' and replan, before Agent.record_step encodes
        # the next observation used by PPO.
        self._synchronize_commanders(current)
        for agent in acting:
            self._episode_agent_returns[int(agent.agent_id)] += rewards[int(agent.agent_id)]
        self._reward_previous_observation = current
        return rewards

    def _synchronize_commanders(self, observation: Mapping[str, Any]) -> None:
        isolated = []
        commanders: dict[int, object] = {}
        for agent in self.agent_manager.get_all_agents():
            commander = getattr(agent, "commander", None)
            if commander is None:
                continue
            commanders[id(commander)] = commander
            agent_observation = self.agent_manager.extract_observation_for_agent(
                observation, agent.agent_id
            )
            if agent_observation:
                isolated.append(agent_observation)
        snapshot = tuple(isolated)
        for commander in commanders.values():
            begin_step = getattr(commander, "begin_step", None)
            if callable(begin_step):
                begin_step(snapshot)

    def _credit_progress(
        self,
        rewards: dict[int, float],
        agent: PersonalR9PPOAttackAgent,
        observation: Mapping[str, Any],
    ) -> None:
        own = self._entity(observation, agent.entity_id)
        entity_id = int(agent.entity_id)
        state = self._progress_state.get(entity_id)
        terminal = (
            own is None
            or self._health(own) <= 0
            or not bool(own.get("isVisible", True))
            or self.current_step >= self.max_steps
        )
        if terminal:
            if state is not None:
                self._credit(
                    rewards,
                    agent,
                    "target_progress",
                    -state[2],
                )
            self._progress_state.pop(entity_id, None)
            return
        target_id = agent.commander.target_id_for(agent.entity_id)
        target = next(
            (
                item
                for item in agent.commander.targets
                if int(item.entity_id) == int(target_id)
            ),
            None,
        ) if target_id is not None else None
        if target is None or own is None:
            if state is not None:
                self._credit(
                    rewards,
                    agent,
                    "target_progress",
                    -state[2],
                )
            self._progress_state.pop(entity_id, None)
            return
        target_position = {
            "lon": float(target.position.lon),
            "lat": float(target.position.lat),
        }
        distance = self._distance_km(own.get("position", {}), target_position)
        if state is None or state[0] != int(target_id):
            if state is not None:
                # Changing the task changes the shaping potential.  Close the
                # previous task at Phi=0 before anchoring the new one.
                self._credit(
                    rewards,
                    agent,
                    "target_progress",
                    -state[2],
                )
            reference = max(
                distance,
                self.reward_config.progress_reference_min_km,
            )
            self._progress_state[entity_id] = (int(target_id), reference, 0.0)
            return

        _, reference, previous_potential = state
        progress_fraction = max(-1.0, min(1.0, (reference - distance) / reference))
        potential = self.reward_config.progress_potential_scale * progress_fraction
        self._progress_state[entity_id] = (int(target_id), reference, potential)
        # Discount-correct potential differences cannot make a profitable
        # out-and-back loop; Phi itself stays within the configured cap.
        self._credit(
            rewards,
            agent,
            "target_progress",
            self.reward_config.progress_discount * potential - previous_potential,
        )

    def _credit_actual_damage(
        self,
        rewards: dict[int, float],
        by_entity: Mapping[int, object],
        acting_ids: set[int],
        previous: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> set[int]:
        successful_sources: set[int] = set()
        relation_map = self.engine.simulator_factory.target_hit_relation
        objective_weights = dict(self.reward_policy.objective_weights)
        for objective_id in self.reward_policy.objective_ids:
            before = self._health(self._entity(previous, objective_id))
            after = self._health(self._entity(current, objective_id))
            health_drop = max(0.0, before - after)
            relations = relation_map.get(int(objective_id), [])
            offset = min(self._relation_offsets.get(int(objective_id), 0), len(relations))
            new_relations = relations[offset:]
            self._relation_offsets[int(objective_id)] = len(relations)
            if health_drop <= 0.0 or not new_relations:
                continue

            target = self._entity(current, objective_id) or self._entity(previous, objective_id)
            target_type = int((target or {}).get("type", -1))
            all_weighted_sources: list[tuple[int, float]] = []
            for relation in new_relations:
                source_id = int(relation.get("id", -1))
                source_type = int(relation.get("entity_type", -1))
                if (
                    source_id not in by_entity
                    or source_type not in RED_MISSILE_TYPES
                ):
                    continue
                potential = self._DAMAGE_POTENTIAL.get((source_type, target_type), 0.0)
                if potential > 0.0:
                    all_weighted_sources.append((source_id, potential))
            if not all_weighted_sources:
                continue

            # Keep every legitimate source in the denominator.  Normally all
            # sources are acting, but this prevents a filtered-out source from
            # having its damage reassigned to another Agent in edge cases.
            total_potential = sum(value for _, value in all_weighted_sources)
            initial = self._initial_objective_health(objective_id, previous, current)
            objective_weight = float(objective_weights.get(int(objective_id), 1.0))
            total_reward = (
                self.reward_config.damage_scale
                * objective_weight
                * min(1.0, health_drop / max(initial, 1e-6))
            )
            # A source can appear more than once; credit every damage command in
            # proportion to its nominal contribution.
            for source_id, potential in all_weighted_sources:
                if source_id not in acting_ids:
                    continue
                agent = by_entity[source_id]
                value = total_reward * potential / total_potential
                self._credit(rewards, agent, "actual_damage", value)
                successful_sources.add(source_id)
        return successful_sources

    def _credit_losses(
        self,
        rewards: dict[int, float],
        by_entity: Mapping[int, object],
        acting_ids: set[int],
        previous: Mapping[str, Any],
        current: Mapping[str, Any],
        successful_sources: set[int],
        *,
        mission_completed: bool = False,
    ) -> None:
        for entity_id, agent in by_entity.items():
            if entity_id not in acting_ids or entity_id in self._settled_failures:
                continue
            if not isinstance(agent, PersonalR9PPOAttackAgent):
                continue
            before = self._entity(previous, entity_id)
            after = self._entity(current, entity_id)
            if self._health(before) <= 0 or self._health(after) > 0:
                continue
            self._settled_failures.add(entity_id)
            if (
                mission_completed
                or entity_id in successful_sources
                or entity_id in self._successful_sources
            ):
                # A platform that produced objective damage, or died after the
                # whole mission was complete, is not an unsuccessful loss.
                continue
            entity_type = int((after or before or {}).get("type", -1))
            self._credit(
                rewards,
                agent,
                "unsuccessful_loss",
                -self._loss_cost(entity_type),
            )

    def _credit_timeout_losses(
        self,
        rewards: dict[int, float],
        acting: list[PersonalR9PPOAttackAgent],
        current: Mapping[str, Any],
    ) -> None:
        """Treat an unsuccessful horizon exactly like another failed terminal."""

        for agent in acting:
            entity_id = int(agent.entity_id)
            if entity_id in self._settled_failures:
                continue
            entity = self._entity(current, entity_id)
            if self._health(entity) <= 0:
                continue
            self._settled_failures.add(entity_id)
            if entity_id in self._successful_sources:
                continue
            self._credit(
                rewards,
                agent,
                "timeout_loss",
                -self._loss_cost(int((entity or {}).get("type", -1))),
            )

    def _loss_cost(self, entity_type: int) -> float:
        return float(
            {
                21000: self.reward_config.high_loss_cost,
                21001: self.reward_config.medium_loss_cost,
                21002: self.reward_config.low_loss_cost,
            }.get(int(entity_type), 0.0)
        )

    def _anchor_official_score(self, observation: Mapping[str, Any]) -> None:
        """Record the judge score at an episode boundary without rewarding it."""

        assert self._reward_tracker is not None
        self._reward_tracker.check_completion(self.current_step, observation)
        self._previous_official_score = float(
            self._reward_tracker.finish(observation).score
        )

    def _credit_official_score_delta(
        self,
        rewards: dict[int, float],
        acting: list[PersonalR9PPOAttackAgent],
        current: Mapping[str, Any],
    ) -> None:
        """Broadcast the exact incremental score from the current pku judge.

        The judge now scores weighted fractional health loss and has no time
        term, so score changes occur on every damaging hit rather than only on
        first destruction.  Computing the difference through RewardTracker
        keeps this trainer synchronized with the read-only upstream formula.
        """

        assert self._reward_tracker is not None
        self._reward_tracker.check_completion(self.current_step, current)
        breakdown = self._reward_tracker.finish(current)
        current_score = float(breakdown.score)
        official_delta = current_score - self._previous_official_score
        self._previous_official_score = current_score
        if math.isclose(official_delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
            return
        self._raw_official_score_delta += official_delta
        # RewardTracker currently exposes a 0--100 score.  Keep the configured
        # scale explicit so a deliberate training-only rescale remains possible
        # without reimplementing the upstream weighting formula.
        team_bonus = official_delta * self.reward_config.official_score_scale / 100.0
        if not acting:
            return
        # Keep per-transition reward scale independent of how many missiles
        # happen to survive until this event.  Shares belonging to already-lost
        # platforms are not reassigned to a handful of survivors.
        denominator = max(1, int(self.learning_team_size or len(acting)))
        per_agent = team_bonus / denominator
        for agent in acting:
            self._credit(rewards, agent, "official_score_delta", per_agent)

    def _objectives_completed(self, observation: Mapping[str, Any]) -> bool:
        for entity_id in self.reward_policy.objective_ids:
            entity = self._entity(observation, entity_id)
            if entity is None or self._health(entity) > 0.0:
                return False
        return True

    def _credit(
        self,
        rewards: dict[int, float],
        agent,
        component: str,
        value: float,
    ) -> None:
        agent_id = int(agent.agent_id)
        value = float(value)
        rewards[agent_id] = rewards.get(agent_id, 0.0) + value
        self._episode_reward_components[component] += value

    def _initial_objective_health(
        self,
        entity_id: int,
        previous: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> float:
        return max(
            float(self._initial_health_by_objective.get(int(entity_id), 0.0)),
            self._health(self._entity(previous, entity_id)),
            self._health(self._entity(current, entity_id)),
            1.0,
        )

    def _load_initial_objective_health(self) -> dict[int, float]:
        try:
            import json

            info_path = self.reward_policy.scenario_path.with_name("case_info.json")
            roots = json.loads(info_path.read_text(encoding="utf-8")).get("roots", [])
            return {
                int(item["id"]): float(item["hp"])
                for item in roots
                if int(item.get("id", -1)) in self.reward_policy.objective_ids
            }
        except (OSError, ValueError, KeyError, TypeError):
            return {}

    @staticmethod
    def _entity(
        observation: Mapping[str, Any], entity_id: int
    ) -> Mapping[str, Any] | None:
        entities = observation.get("entities", {})
        return entities.get(int(entity_id), entities.get(str(entity_id)))

    @staticmethod
    def _health(entity: Mapping[str, Any] | None) -> float:
        return float(entity.get("health", 0.0)) if entity else 0.0

    @staticmethod
    def _distance_km(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
        lon1 = math.radians(float(first.get("lon", 0.0)))
        lat1 = math.radians(float(first.get("lat", 0.0)))
        lon2 = math.radians(float(second.get("lon", 0.0)))
        lat2 = math.radians(float(second.get("lat", 0.0)))
        delta_lon = lon2 - lon1
        delta_lat = lat2 - lat1
        haversine = (
            math.sin(delta_lat / 2.0) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2.0) ** 2
        )
        return 6371.0 * 2.0 * math.atan2(
            math.sqrt(haversine), math.sqrt(max(0.0, 1.0 - haversine))
        )
