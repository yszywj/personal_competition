"""Commander tests: one API call per episode, clean resets, INVALID_PLAN."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from personal_train.llm_strategy.audit import AuditWriter
from personal_train.llm_strategy.commander import LLMPlanCommander
from personal_train.llm_strategy.glm_client import MockGLMClient
from personal_train.llm_strategy.tests.fakes import (
    CountingClient,
    default_rules,
    make_init_obs,
    make_isolated_obs,
)

INIT_CATALOGUE = make_init_obs(
    [
        {"id": 51, "type": 9400, "lon": 123.25, "lat": 24.75, "health": 32.0},
        {"id": 52, "type": 9400, "lon": 123.6, "lat": 23.0, "health": 32.0},
    ]
)


def valid_payload():
    return {
        "plan_version": "v0",
        "platforms": [
            {
                "platform_id": 10,
                "launch": {"mode": "at_step", "step": 0},
                "initial_target": {"mode": "entity", "entity_id": 51},
                "retarget_orders": [],
                "satellite_steps": [],
                "motion": "straight",
            },
            {
                "platform_id": 11,
                "launch": {"mode": "never"},
                "satellite_steps": [],
                "motion": "straight",
            },
        ],
        "global_rules": [],
    }


def observations(step):
    return [
        make_isolated_obs(entity_id=10, entity_type=21000, step=step),
        make_isolated_obs(entity_id=11, entity_type=21001, step=step),
    ]


def make_commander(client, root: Path) -> LLMPlanCommander:
    audit = AuditWriter(root)
    commander = LLMPlanCommander(
        init_ship_observation=INIT_CATALOGUE,
        client=client,
        rules=default_rules(),
        audit=audit,
    )
    commander.register_platform(10)
    commander.register_platform(11)
    return commander


class OneCallPerEpisodeTests(unittest.TestCase):
    def test_exactly_one_api_call_across_many_steps(self):
        client = CountingClient(valid_payload())
        with tempfile.TemporaryDirectory() as tmp:
            commander = make_commander(client, Path(tmp))
            commander.reset()
            for step in range(0, 30):
                commander.begin_step(observations(step))
                commander.actions_for(10, step)
        self.assertEqual(client.call_count, 1)
        self.assertEqual(commander.api_call_count, 1)

    def test_no_second_call_after_reset(self):
        client = CountingClient(valid_payload())
        with tempfile.TemporaryDirectory() as tmp:
            commander = make_commander(client, Path(tmp))
            commander.reset()
            for step in range(0, 5):
                commander.begin_step(observations(step))
            self.assertEqual(client.call_count, 1)
            commander.reset()
            for step in range(0, 5):
                commander.begin_step(observations(step))
        self.assertEqual(client.call_count, 2)  # one per episode, never more
        self.assertEqual(commander.api_call_count, 1)  # counter reset per round


class ResetIsolationTests(unittest.TestCase):
    def test_reset_clears_plan_and_executor_state(self):
        client = CountingClient(valid_payload())
        with tempfile.TemporaryDirectory() as tmp:
            commander = make_commander(client, Path(tmp))
            commander.reset()
            for step in range(0, 3):
                commander.begin_step(observations(step))
            self.assertTrue(commander.planned)
            self.assertEqual(commander.launched_platform_count, 1)
            first_sha = commander.accepted_plan_sha256
            commander.reset()
            self.assertFalse(commander.planned)
            self.assertIsNone(commander.accepted_plan)
            self.assertIsNone(commander.accepted_plan_sha256)
            self.assertFalse(commander.plan_rejected)
            self.assertEqual(commander.launched_platform_count, 0)
            self.assertEqual(commander.actions_for(10, 0), [])
            # New episode plans and launches again from scratch.
            for step in range(0, 3):
                commander.begin_step(observations(step))
            self.assertTrue(commander.planned)
            self.assertEqual(commander.launched_platform_count, 1)
            self.assertEqual(commander.accepted_plan_sha256, first_sha)

    def test_round_two_writes_a_fresh_audit_directory(self):
        client = CountingClient(valid_payload())
        with tempfile.TemporaryDirectory() as tmp:
            commander = make_commander(client, Path(tmp))
            commander.reset()
            commander.begin_step(observations(0))
            commander.write_round_metrics()
            commander.reset()
            commander.begin_step(observations(0))
            commander.write_round_metrics()
            audit_root = commander.audit.root
            rounds = sorted(item.name for item in audit_root.iterdir())
            self.assertEqual(rounds, ["round_001", "round_002"])
            for name in rounds:
                round_dir = audit_root / name
                for artefact in (
                    "state_input.json",
                    "glm_raw_response.json",
                    "validation_report.json",
                    "accepted_plan.json",
                    "plan_sha256.txt",
                    "llm_metrics.json",
                ):
                    self.assertTrue(
                        (round_dir / artefact).is_file(),
                        f"{name}/{artefact} missing",
                    )


class InvalidPlanTests(unittest.TestCase):
    def test_invalid_plan_rejects_without_retry(self):
        client = CountingClient({"plan_version": "v0", "platforms": "nonsense"})
        with tempfile.TemporaryDirectory() as tmp:
            commander = make_commander(client, Path(tmp))
            commander.reset()
            for step in range(0, 5):
                commander.begin_step(observations(step))
        self.assertTrue(commander.plan_rejected)
        self.assertIn("INVALID_PLAN", commander.rejection_reason)
        self.assertEqual(client.call_count, 1)  # no second attempt
        self.assertEqual(commander.actions_for(10, 0), [])

    def test_unlegal_entity_reference_rejects(self):
        payload = valid_payload()
        payload["platforms"][0]["initial_target"] = {
            "mode": "entity",
            "entity_id": 168,  # hidden ship id the model cannot know
        }
        client = CountingClient(payload)
        with tempfile.TemporaryDirectory() as tmp:
            commander = make_commander(client, Path(tmp))
            commander.reset()
            commander.begin_step(observations(0))
            self.assertTrue(commander.plan_rejected)
            report = commander.audit.round_dir / "validation_report.json"
            self.assertTrue(report.is_file())


class MockClientPipelineTests(unittest.TestCase):
    def test_mock_plan_flows_through_the_same_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_file = Path(tmp) / "mock_plan.json"
            import json

            plan_file.write_text(json.dumps(valid_payload()), encoding="utf-8")
            client = MockGLMClient(plan_file)
            commander = make_commander(client, Path(tmp) / "audit")
            commander.reset()
            commander.begin_step(observations(0))
            self.assertTrue(commander.planned)
            self.assertFalse(commander.plan_rejected)
            self.assertEqual(client.call_count, 1)
            rows = commander.actions_for(10, 0)
            self.assertEqual(rows, [[1.0, 10.0, 123.25, 24.75]])

    def test_mock_client_missing_file_raises_clear_error(self):
        from personal_train.llm_strategy.glm_client import LLMClientError

        with self.assertRaises(LLMClientError):
            MockGLMClient("/nonexistent/plan.json")

    def test_real_client_without_api_key_raises_clear_error(self):
        import os

        from personal_train.llm_strategy.glm_client import GLMClient, LLMClientError

        previous = os.environ.pop("GLM_API_KEY", None)
        try:
            with self.assertRaises(LLMClientError) as ctx:
                GLMClient.from_env()
            self.assertIn("GLM_API_KEY", str(ctx.exception))
        finally:
            if previous is not None:
                os.environ["GLM_API_KEY"] = previous


if __name__ == "__main__":
    unittest.main()
