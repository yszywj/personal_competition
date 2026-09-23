"""E01 end-to-end smoke test through run_llm_plan.py with a mock plan.

Requires the native simulator runtime; run with the project's glibc-2.38
Python from the workspace root:

    cd /home/amax/ry/competition
    /home/amax/ry/competition/glibc-2.38/python3.11-glibc238 \
        -m unittest personal_train.llm_strategy.tests.test_smoke_e01
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from personal_train.bootstrap import install_project_paths

install_project_paths()

from envengine import Profile, TrainingEnv  # noqa: E402

from personal_train.run_llm_plan import environment_rules, load_profile, resolve_scenario  # noqa: E402

STEP_LIMIT = 40
ROUNDS = 2


def _build_mock_plan(scenario_path: Path) -> dict:
    """Derive a valid, feature-covering plan from the scenario itself.

    Test-harness side only: this plays the role of the LLM's output file. It
    references nothing outside what the model would legally know (opening
    catalogue entity ids and red platform ids).
    """

    profile = load_profile(scenario_path)
    environment = TrainingEnv(profile, render_mode=None)
    try:
        init_obs = environment._get_init_ship_observation()
        known_ids = sorted(int(key) for key in init_obs["entities"])
        by_type: dict[int, list[int]] = {21000: [], 21001: [], 21002: []}
        for simulator in environment.engine.simulator_factory.get_all_simulators():
            entity = simulator.entity_ext.entity
            entity_type = int(entity.entityType)
            if entity_type in by_type and int(getattr(entity, "sideId", 0)) == 0:
                by_type[entity_type].append(int(entity.id))
        for ids in by_type.values():
            ids.sort()
    finally:
        environment.close()

    assert known_ids, "opening catalogue must not be empty"

    platforms = []
    for index, entity_id in enumerate(by_type[21000]):
        platforms.append(
            {
                "platform_id": entity_id,
                "launch": {"mode": "at_step", "step": 0},
                "initial_target": {
                    "mode": "entity",
                    "entity_id": known_ids[index % len(known_ids)],
                },
                "retarget_orders": [],
                "satellite_steps": [{"step": 3}] if index == 0 else [],
                "motion": "straight",
            }
        )
    for index, entity_id in enumerate(by_type[21001]):
        platforms.append(
            {
                "platform_id": entity_id,
                "launch": {"mode": "at_step", "step": 5},
                "initial_target": {
                    "mode": "entity",
                    "entity_id": known_ids[index % len(known_ids)],
                },
                "retarget_orders": [],
                "satellite_steps": [],
                "motion": "straight",
            }
        )
    low_ids = by_type[21002]
    for index, entity_id in enumerate(low_ids[:10]):
        platforms.append(
            {
                "platform_id": entity_id,
                "launch": {"mode": "at_step", "step": 10},
                "initial_target": {
                    "mode": "entity",
                    "entity_id": known_ids[index % len(known_ids)],
                },
                "retarget_orders": [],
                "satellite_steps": [],
                "motion": "straight",
            }
        )
    for entity_id in low_ids[10:]:
        platforms.append(
            {
                "platform_id": entity_id,
                "launch": {"mode": "never"},
                "satellite_steps": [],
                "motion": "straight",
            }
        )

    medium_ids = by_type[21001]
    return {
        "plan_version": "v0",
        "platforms": platforms,
        "global_rules": [
            {
                "rule_id": "ship_watch",
                "trigger": {
                    "type": "new_detection",
                    "entity_type": 9500,
                    "occurrence": 1,
                },
                "actions": [
                    {
                        "type": "retarget",
                        "platform_id": low_ids[0],
                        "target": {"mode": "event_entity"},
                    }
                ],
            },
            {
                "rule_id": "mid_satellite",
                "trigger": {"type": "at_step", "step": 20},
                "actions": [
                    {"type": "satellite_request", "platform_id": medium_ids[0]}
                ],
            },
        ],
    }


class E01MockPlanSmokeTest(unittest.TestCase):
    """Full pipeline: DeployAgent -> state -> mock GLM -> plan -> execution."""

    def test_mock_plan_completes_e01_and_traces_back_to_the_plan(self):
        from personal_train.run_llm_plan import main as run_main

        scenario_path = resolve_scenario("easy/E01")
        plan_payload = _build_mock_plan(scenario_path)
        default_root = (
            Path(__file__).resolve().parents[2] / "llm_strategy" / "results"
        )
        with tempfile.TemporaryDirectory() as tmp:
            plan_file = Path(tmp) / "mock_plan.json"
            plan_file.write_text(
                json.dumps(plan_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            exit_code = run_main(
                [
                    "--scenario", str(scenario_path),
                    "--rounds", str(ROUNDS),
                    "--seed", "7",
                    "--run-id", "smoke_e01_auto",
                    "--blue-policy", "b0_fixed_ratio_random",
                    "--mock-plan", str(plan_file),
                    "--render-mode", "none",
                    "--max-steps", str(STEP_LIMIT),
                ]
            )
        self.assertEqual(exit_code, 0)

        run_roots = sorted(default_root.glob("smoke_e01_auto_*"))
        self.assertTrue(run_roots, "no smoke run directory was created")
        run_root = run_roots[-1]

        # Official scoring artefacts come from the untouched upstream path.
        self.assertTrue((run_root / "summary_round_1.json").is_file())
        self.assertTrue((run_root / "summary_round_2.json").is_file())
        summary = json.loads(
            (run_root / "summary_round_1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["policies"]["red"], "llm_plan")
        self.assertEqual(summary["policies"]["red_motion"], "straight")
        # RunSummary puts the RewardTracker breakdown dict under "score".
        self.assertIsInstance(summary["score"]["score"], (int, float))
        self.assertGreaterEqual(summary["score"]["total_objective_weight"], 0)
        self.assertGreaterEqual(summary["red"]["launched"], 1)

        status = json.loads(
            (run_root / "llm_run_status.json").read_text(encoding="utf-8")
        )
        self.assertEqual(status["status"], "complete")
        self.assertFalse(status["invalid_plan_encountered"])
        self.assertFalse(status["api_call_violation"])

        rounds = sorted(run_root.glob("round_*"))
        self.assertEqual(len(rounds), ROUNDS)
        total_high_launches = 0
        for round_dir in rounds:
            # Exactly one mock model call per episode, fresh artefacts.
            metrics = json.loads(
                (round_dir / "llm_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["api_calls"], 1)
            self.assertFalse(metrics["plan_rejected"])

            # Accepted plan integrity: sha256 matches the stored plan bytes.
            plan_bytes = (round_dir / "accepted_plan.json").read_bytes()
            digest = hashlib.sha256(plan_bytes).hexdigest()
            stored = (round_dir / "plan_sha256.txt").read_text(encoding="utf-8").strip()
            self.assertEqual(digest, stored)

            # The planning state saw only legal information: no 9500 leak.
            state_text = (round_dir / "state_input.json").read_text(encoding="utf-8")
            self.assertNotIn('"entity_type": 9500', state_text)
            self.assertNotIn('"type": 9500', state_text)

            traces = [
                json.loads(line)
                for line in (round_dir / "execution_trace.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(traces)
            for record in traces:
                self.assertTrue(record["plan_rule_id"])
                self.assertIn(record["engine_action"][0], (1.0, 2.0, 3.0))
                self.assertTrue(record["success"])
            launches = [r for r in traces if r["engine_action"][0] == 1.0]
            satellites = [r for r in traces if r["engine_action"][0] == 3.0]
            step0 = [r for r in launches if r["step"] == 0]
            step5 = [r for r in launches if r["step"] == 5]
            step10 = [r for r in launches if r["step"] == 10]
            self.assertEqual(len(step0), 30)  # every 21000 launched at step 0
            self.assertEqual(len(step5), 44)  # every 21001 launched at step 5
            self.assertEqual(len(step10), 10)  # first ten 21002 at step 10
            satellite_steps = {r["step"] for r in satellites}
            self.assertIn(3, satellite_steps)  # non-launch-step request
            self.assertIn(20, satellite_steps)  # rule-triggered request
            total_high_launches = len(step0)

        self.assertEqual(total_high_launches, 30)

        # Round 2 did not inherit round 1's trace content.
        first_steps = [
            json.loads(line)["step"]
            for line in (rounds[0] / "execution_trace.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        second_steps = [
            json.loads(line)["step"]
            for line in (rounds[1] / "execution_trace.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(first_steps[0], 0)
        self.assertEqual(second_steps[0], 0)


if __name__ == "__main__":
    unittest.main()
