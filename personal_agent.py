"""R9 missile Agent fixes kept outside the competition source tree."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .bootstrap import install_project_paths
from .legal_observation import CorrectedR9ObservationEncoder

install_project_paths()

from envengine.agent_manager import AgentManager  # noqa: E402
from policies.red.learning import (  # noqa: E402
    HighLevelAction,
    LearningActionAdapter,
    PolicyTransition,
    SharedPolicy,
)
from user_agents import AttackMissileAgent  # noqa: E402
from user_agents.base_agent import BaseAgent  # noqa: E402


class ResetAwareAgentManager(AgentManager):
    """Reset every shared Commander exactly once per environment reset.

    This local manager makes the trainer independent of whether the checked-out
    version of ``AgentManager`` already contains the Commander reset fix.
    """

    def reset_all(self) -> None:
        commanders: dict[int, object] = {}
        for agent in self.get_all_agents():
            agent.reset()
            commander = getattr(agent, "commander", None)
            if commander is not None:
                commanders[id(commander)] = commander
        for commander in commanders.values():
            reset = getattr(commander, "reset", None)
            if callable(reset):
                reset()


class PersonalR9PPOAttackAgent(AttackMissileAgent):
    """Use the original Agent protocol with corrected state bookkeeping."""

    _MANEUVER_FEATURE_BY_ACTION = {
        int(HighLevelAction.MANEUVER_LEFT): -2,
        int(HighLevelAction.STOP_MANEUVER): 0,
        int(HighLevelAction.MANEUVER_RIGHT): 2,
    }

    def __init__(
        self,
        agent_id: int,
        entity_id: int,
        init_observation: dict,
        commander,
        learning_policy: SharedPolicy,
        *,
        learning_max_steps: int,
        team_size: int,
        initial_sim_time_ms: float,
        sim_step_ms: float,
    ) -> None:
        super().__init__(
            agent_id,
            entity_id,
            init_observation,
            commander=commander,
            motion_policy="ppo",
            learning_policy=learning_policy,
            learning_max_steps=learning_max_steps,
            hierarchical_learning=True,
        )
        encoder = CorrectedR9ObservationEncoder(
            init_observation,
            max_steps=learning_max_steps,
            agent_id=agent_id,
            team_size=team_size,
            initial_sim_time_ms=initial_sim_time_ms,
            sim_step_ms=sim_step_ms,
        )
        self.learning_encoder = encoder
        # Keep target ordering identical in the observation and adapter.  This
        # does not alter the five-slot interface.
        self.learning_adapter = LearningActionAdapter(encoder.targets)
        self.action_counts: Counter[int] = Counter()
        self.action_switch_count = 0
        self.last_selected_action: int | None = None
        self.switched_on_last_action = False

    @property
    def corrected_encoder(self) -> CorrectedR9ObservationEncoder:
        assert isinstance(self.learning_encoder, CorrectedR9ObservationEncoder)
        return self.learning_encoder

    def set_observation(self, observation: dict) -> None:
        super().set_observation(observation)
        self.corrected_encoder.observe_frame(observation)

    def _target_index(self) -> int | None:
        return self.corrected_encoder.target_index(
            self.commander.target_id_for(self.entity_id)
        )

    def _task_context(self):
        return self.commander.learning_task_context(self.entity_id)

    def _satellite_used_feature(self, observation: dict) -> bool:
        """Return the satellite state defined by the detected backend contract."""

        if "is_using_satellite" in observation:
            # In the new backend this is a shared active-window state, rather
            # than a record of whether this particular missile requested it.
            return bool(observation["is_using_satellite"])
        # Old isolated observations have no team-global field.
        return bool(self.sat_used)

    def _set_acc_z_learning(self, actions: list[list[float]], observation: dict) -> None:
        if self.launch_step < 0:
            self.switched_on_last_action = False
            return
        assert self.learning_policy is not None
        assert self.learning_adapter is not None

        satellite_used = self._satellite_used_feature(observation)
        action_mask = self.learning_adapter.build_action_mask(
            launched=True,
            satellite_used=satellite_used,
        )
        encoded = self.corrected_encoder.encode(
            observation,
            launched=True,
            launch_step=self.launch_step,
            satellite_used=satellite_used,
            # s_t contains the command that governed the flight into s_t.
            maneuver_state=self.set_acc_z_z,
            current_target_index=self._target_index(),
            task_context=self._task_context(),
        )
        # The local PPO can bind behaviour-policy statistics to an explicit
        # Agent ID.  Keep the generic SharedPolicy fallback for smoke policies
        # and other competition-compatible implementations.
        select_for_agent = getattr(self.learning_policy, "select_action_for_agent", None)
        if callable(select_for_agent):
            selected_action = int(
                select_for_agent(self.agent_id, encoded, action_mask)
            )
        else:
            selected_action = int(self.learning_policy.select_action(encoded, action_mask))
        if bool(getattr(self.learning_policy, "training", False)):
            self._learning_transition = (encoded, selected_action, action_mask.copy())

        self.switched_on_last_action = (
            self.last_selected_action is not None
            and selected_action != self.last_selected_action
        )
        if self.switched_on_last_action:
            self.action_switch_count += 1
        self.last_selected_action = selected_action
        self.action_counts[selected_action] += 1

        # ObservationEncoder divides maneuver_state by two.  Store twice the
        # actual command so the unchanged feature spans exactly -1/0/+1.
        self.set_acc_z_z = self._MANEUVER_FEATURE_BY_ACTION[selected_action]
        engine_actions = self.learning_adapter.to_engine_action(
            selected_action, self.entity_id
        )
        actions.extend(engine_actions.tolist())

    def record_step(self, observation: dict, action, reward: float, info: dict = None):
        # Calling AttackMissileAgent.record_step would execute the original
        # transition path as well, so call its simple history base directly.
        BaseAgent.record_step(self, observation, action, reward, info)
        if self._learning_transition is None or self.learning_policy is None:
            return

        encoded, selected_action, action_mask = self._learning_transition
        self._learning_transition = None
        if not bool(getattr(self.learning_policy, "training", False)):
            return

        next_observation = (
            self.corrected_encoder.encode(
                observation,
                launched=True,
                launch_step=self.launch_step,
                satellite_used=self._satellite_used_feature(observation),
                maneuver_state=self.set_acc_z_z,
                # Fix: s' uses the same current target slot as s.
                current_target_index=self._target_index(),
                task_context=self._task_context(),
            )
            if observation.get("self")
            else np.zeros_like(encoded)
        )
        global_state, next_global_state = self.learning_policy.get_global_state_context()
        self.learning_policy.observe(
            PolicyTransition(
                agent_id=self.agent_id,
                observation=encoded,
                action=selected_action,
                action_mask=action_mask,
                reward=float(reward),
                next_observation=next_observation,
                done=bool((info or {}).get("done", False) or not observation.get("self")),
                global_state=global_state,
                next_global_state=next_global_state,
            )
        )

    def reset(self):
        super().reset()
        # Upstream AttackMissileAgent.reset() currently leaves the most recent
        # isolated frame cached.  Without clearing it, the first decision of a
        # new episode can consume the terminal observation from the preceding
        # round.
        self.latest_observation = None
        self.corrected_encoder.reset_history()
        self.action_counts.clear()
        self.action_switch_count = 0
        self.last_selected_action = None
        self.switched_on_last_action = False
