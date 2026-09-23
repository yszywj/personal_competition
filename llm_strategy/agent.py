"""Red missile agent that only relays accepted-plan commands.

Unlike ``AttackMissileAgent``, this agent carries no policy of its own: no
motion heuristics, no launch-time satellite binding, no learning adapter.
Each step it forwards its isolated observation ownership to the shared
``LLMPlanCommander`` and returns whatever the plan schedules for this
platform at this step.  Because the executor never emits lateral
acceleration, every missile flies straight (V0 condition).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..bootstrap import install_project_paths

install_project_paths()

from user_agents.base_agent import BaseAgent, AgentType  # noqa: E402


class LLMPlanAgent(BaseAgent):
    """Agent shell around the one-shot LLM plan executor."""

    def __init__(
        self,
        agent_id: int,
        entity_id: int,
        init_observation: dict,
        *,
        commander,
    ) -> None:
        super().__init__(agent_id, entity_id, AgentType.AIRCRAFT, init_observation)
        if commander is None:
            raise ValueError("LLMPlanAgent requires a shared LLMPlanCommander")
        self.commander = commander
        self.latest_observation: dict | None = None

    def set_observation(self, observation: dict) -> None:
        self.latest_observation = observation

    def get_action(self) -> np.ndarray:
        observation = self.latest_observation
        if not observation:
            # The platform is dead or not visible: the plan's commands for it
            # are recorded as execution failures by the executor.
            return np.zeros((0, 4), dtype=np.float64)
        step = int(observation.get("step", 0))
        rows = self.commander.actions_for(self.entity_id, step)
        if not rows:
            return np.zeros((0, 4), dtype=np.float64)
        return np.array(rows, dtype=np.float64)

    def reset(self) -> None:
        super().reset()
        self.latest_observation = None

    def record_step(
        self, observation: dict, action: Any, reward: float, info: dict | None = None
    ) -> None:
        # History recording is unnecessary for a one-shot planned policy; the
        # authoritative record is the executor's audit trace.
        super().record_step(observation, action, reward, info)
