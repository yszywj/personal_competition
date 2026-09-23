"""Plan schema parsing tests."""

from __future__ import annotations

import copy
import unittest

from personal_train.llm_strategy.plan_schema import parse_plan
from personal_train.llm_strategy.tests.fakes import CountingClient  # noqa: F401


def valid_plan_payload():
    return {
        "plan_version": "v0",
        "platforms": [
            {
                "platform_id": 10,
                "launch": {"mode": "at_step", "step": 0},
                "initial_target": {"mode": "entity", "entity_id": 51},
                "retarget_orders": [
                    {
                        "step": 300,
                        "target": {"mode": "coordinate", "lon": 120.5, "lat": 22.1},
                    }
                ],
                "satellite_steps": [{"step": 10}],
                "motion": "straight",
            },
            {
                "platform_id": 12,
                "launch": {"mode": "never"},
                "satellite_steps": [{"step": 5}],
                "motion": "straight",
            },
        ],
        "global_rules": [
            {
                "rule_id": "redirect_first_ship",
                "trigger": {
                    "type": "new_detection",
                    "entity_type": 9500,
                    "occurrence": 1,
                },
                "actions": [
                    {
                        "type": "retarget",
                        "platform_id": 10,
                        "target": {"mode": "event_entity"},
                    }
                ],
            }
        ],
    }


class ParsePlanTests(unittest.TestCase):
    def test_valid_plan_parses_without_errors(self):
        plan, errors = parse_plan(valid_plan_payload())
        self.assertEqual(errors, [])
        self.assertIsNotNone(plan)
        self.assertEqual(plan.plan_version, "v0")
        self.assertEqual(len(plan.platforms), 2)
        self.assertEqual(plan.platforms[0].launch.step, 0)
        self.assertEqual(plan.platforms[0].initial_target.entity_id, 51)
        self.assertEqual(plan.platforms[1].launch.mode, "never")
        self.assertEqual(len(plan.global_rules), 1)
        self.assertEqual(plan.global_rules[0].rule_id, "redirect_first_ship")

    def test_non_straight_motion_is_a_structural_error(self):
        payload = valid_plan_payload()
        payload["platforms"][0]["motion"] = "weave_evasion"
        plan, errors = parse_plan(payload)
        self.assertIsNone(plan)
        self.assertTrue(any("motion" in item for item in errors))

    def test_unknown_target_mode_is_rejected(self):
        payload = valid_plan_payload()
        payload["platforms"][0]["initial_target"] = {
            "mode": "nearest_enemy",
        }
        plan, errors = parse_plan(payload)
        self.assertIsNone(plan)
        self.assertTrue(any("mode" in item for item in errors))

    def test_unknown_trigger_type_is_rejected(self):
        payload = valid_plan_payload()
        payload["global_rules"][0]["trigger"] = {"type": "on_target_destroyed"}
        plan, errors = parse_plan(payload)
        self.assertIsNone(plan)
        self.assertTrue(any("trigger" in item for item in errors))

    def test_wrong_plan_version_is_rejected(self):
        payload = valid_plan_payload()
        payload["plan_version"] = "v1"
        plan, errors = parse_plan(payload)
        self.assertTrue(any("plan_version" in item for item in errors))

    def test_parsing_never_mutates_the_raw_payload(self):
        payload = valid_plan_payload()
        snapshot = copy.deepcopy(payload)
        parse_plan(payload)
        self.assertEqual(payload, snapshot)

    def test_missing_platforms_is_rejected(self):
        payload = {"plan_version": "v0", "platforms": []}
        plan, errors = parse_plan(payload)
        self.assertIsNone(plan)
        self.assertTrue(any("platforms" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
