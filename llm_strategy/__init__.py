"""LLM one-shot global battle planner for the read-only competition simulator.

Layout note: the upstream checkout ``competition_envs`` is read-only.  This
package therefore lives under ``personal_train`` and is driven by the
standalone entry point ``personal_train/run_llm_plan.py`` instead of patching
``run.py``/``core/main.py``.  Nothing in this package writes to the upstream
checkout and no R0--R9/PPO/MAPPO code is imported.

Roles (strict separation):
    StateBuilder -> facts only, from legal red-side information.
    GLMClient    -> exactly one call per episode (or a mock file).
    Validator    -> PASS / REJECT only, never mutates the plan.
    Executor     -> literal interpretation of the accepted plan, never decides.
"""

from __future__ import annotations


__all__ = [
    "agent",
    "audit",
    "commander",
    "executor",
    "glm_client",
    "plan_schema",
    "state_builder",
    "validator",
]
