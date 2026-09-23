"""One-shot LLM plan commander.

The first ``begin_step`` (step 0, after deployment) builds the legal battle
state, calls the GLM API exactly once, parses the plan with the strict
schema, validates it, and stores the accepted plan.  Every later step only
feeds legal observations to the executor.  If the plan is rejected, the
episode is marked INVALID_PLAN and no actions are ever issued; there is no
retry, no repair and no fallback policy.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .audit import AuditWriter
from .executor import PlanExecutor
from .glm_client import GLMClient, MockGLMClient, extract_json_payload
from .plan_schema import BattlePlan, parse_plan
from .state_builder import EnvironmentRules, build_battle_state
from .validator import validate_plan

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system_prompt.md"


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


class InvalidPlanError(RuntimeError):
    """Raised when the model output fails parsing or validation."""


class LLMPlanCommander:
    """Shared commander owning the single LLM call and the executor."""

    def __init__(
        self,
        *,
        init_ship_observation: Mapping[str, Any],
        client: GLMClient | MockGLMClient,
        rules: EnvironmentRules,
        audit: AuditWriter,
    ) -> None:
        self.init_ship_observation = init_ship_observation
        self.client = client
        self.rules = rules
        self.audit = audit
        self._platform_ids: list[int] = []
        self._executor: PlanExecutor | None = None
        self._accepted_plan: BattlePlan | None = None
        self._accepted_plan_sha256: str | None = None
        self._raw_plan_payload: Mapping[str, Any] | None = None
        self._planned = False
        self.plan_rejected = False
        self.rejection_reason: str | None = None
        self._last_step = -1
        self.api_call_count = 0

    # ------------------------------------------------------------------
    # Registration / lifecycle
    # ------------------------------------------------------------------

    @property
    def planned(self) -> bool:
        return self._planned

    @property
    def accepted_plan(self) -> BattlePlan | None:
        return self._accepted_plan

    @property
    def accepted_plan_sha256(self) -> str | None:
        return self._accepted_plan_sha256

    @property
    def launched_platform_count(self) -> int:
        return len(self._executor.launched_platforms) if self._executor else 0

    def register_platform(self, entity_id: int) -> None:
        self._platform_ids.append(int(entity_id))

    def reset(self) -> None:
        """Full per-episode reset: no plan, executor or trace survives."""

        self.audit.new_round()
        self._executor = None
        self._accepted_plan = None
        self._accepted_plan_sha256 = None
        self._raw_plan_payload = None
        self._planned = False
        self.plan_rejected = False
        self.rejection_reason = None
        self._last_step = -1
        self.api_call_count = 0

    # ------------------------------------------------------------------
    # One-shot planning
    # ------------------------------------------------------------------

    def _known_entity_ids(self) -> set[int]:
        entities = self.init_ship_observation.get("entities") or {}
        known: set[int] = set()
        for key, entity in entities.items():
            try:
                known.add(int(key))
            except (TypeError, ValueError):
                continue
        return known

    def _plan_once(
        self, observations: Sequence[Mapping[str, Any]], step: int
    ) -> None:
        state = build_battle_state(
            step=step,
            init_ship_observation=self.init_ship_observation,
            platform_observations=observations,
            rules=self.rules,
        )
        self.audit.write_state_input(state)
        system_prompt = load_system_prompt()
        user_prompt = json.dumps(state, ensure_ascii=False)

        started = time.perf_counter()
        content = self.client.chat(system_prompt, user_prompt)
        latency = time.perf_counter() - started
        self.api_call_count += 1
        is_mock = isinstance(self.client, MockGLMClient)
        self.audit.write_raw_response(
            content=content,
            mock=is_mock,
            latency_s=latency,
            prompt_chars=len(system_prompt) + len(user_prompt),
        )

        payload, error = extract_json_payload(content)
        if error is not None or not isinstance(payload, Mapping):
            reason = error or "model response is not a JSON object"
            self._reject([f"parse: {reason}"], raw_content_saved=True)
            return

        # The payload is treated as immutable from here on.  Parsing and
        # validation operate on copies/read-only views; this dict is written
        # verbatim to the audit artefacts.
        plan, parse_errors = parse_plan(payload)
        if plan is None:
            self._reject([f"parse: {item}" for item in parse_errors])
            return

        report = validate_plan(
            plan,
            controlled_platform_ids=self._platform_ids,
            known_entity_ids=self._known_entity_ids(),
            max_steps=self.rules.max_steps,
            satellite_max_use_count=self.rules.satellite_max_use_count,
        )
        self.audit.write_validation_report(report.to_dict())
        if not report.passed:
            self._reject([f"validation: {item}" for item in report.errors])
            return

        self._accepted_plan = plan
        self._raw_plan_payload = payload
        self._accepted_plan_sha256 = self.audit.write_accepted_plan(payload)
        self._executor = PlanExecutor(
            plan,
            init_ship_observation=self.init_ship_observation,
            controlled_platform_ids=self._platform_ids,
            on_trace=self.audit.append_trace,
        )
        self._planned = True

    def _reject(self, reasons: Sequence[str], raw_content_saved: bool = False) -> None:
        self.plan_rejected = True
        self.rejection_reason = "INVALID_PLAN: " + "; ".join(reasons)
        self.audit.write_validation_report(
            {"passed": False, "errors": list(reasons)}
        )
        self._planned = True  # Planning was attempted exactly once; it failed.

    # ------------------------------------------------------------------
    # Per-step interface used by TrainingEnv / the agents
    # ------------------------------------------------------------------

    def begin_step(self, observations: Sequence[Mapping[str, Any]] | tuple) -> None:
        observations = tuple(observations)
        step = (
            int(observations[0].get("step", self._last_step + 1))
            if observations
            else self._last_step + 1
        )
        if not self._planned:
            self._plan_once(observations, step)
            # The executor for this episode starts at the planning step.
            if self._executor is not None:
                self._executor.begin_step(step, observations)
            self._last_step = step
            return
        if self._executor is None:
            self._last_step = step
            return  # Rejected plan: no commands will ever be issued.
        self._executor.begin_step(step, observations)
        self._last_step = step

    def actions_for(self, platform_id: int, step: int) -> list[list[float]]:
        if self._executor is None:
            return []
        return self._executor.actions_for(platform_id, step)

    def write_round_metrics(self, *, extra: Mapping[str, Any] | None = None) -> None:
        metrics: dict[str, Any] = {
            "api_calls": int(self.api_call_count),
            "planned": bool(self._planned),
            "plan_rejected": bool(self.plan_rejected),
            "rejection_reason": self.rejection_reason,
            "accepted_plan_sha256": self._accepted_plan_sha256,
            "launched_platforms": self.launched_platform_count,
        }
        if self._executor is not None:
            metrics["executed_commands"] = self._executor.executed_commands
            metrics["failed_commands"] = self._executor.failed_commands
        else:
            metrics["executed_commands"] = 0
            metrics["failed_commands"] = 0
        if extra:
            metrics.update(dict(extra))
        self.audit.write_metrics(metrics)
