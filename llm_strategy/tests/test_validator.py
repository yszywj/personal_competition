"""Validator tests: pure PASS/REJECT, no plan mutation, no quality judging."""

from __future__ import annotations

import copy
import unittest

from personal_train.llm_strategy.plan_schema import parse_plan
from personal_train.llm_strategy.validator import validate_plan

CONTROLLED = [10, 11, 12, 13, 14]
KNOWN = {51, 52, 53, 54, 106}


def validate(payload, *, satellite_cap=100, max_steps=1200):
    plan, errors = parse_plan(payload)
    if plan is None:
        raise AssertionError(f"payload failed to parse: {errors}")
    return validate_plan(
        plan,
        controlled_platform_ids=CONTROLLED,
        known_entity_ids=KNOWN,
        max_steps=max_steps,
        satellite_max_use_count=satellite_cap,
    )


def base_payload():
    return {
        "plan_version": "v0",
        "platforms": [
            {
                "platform_id": 10,
                "launch": {"mode": "at_step", "step": 4},
                "initial_target": {"mode": "entity", "entity_id": 51},
                "retarget_orders": [
                    {"step": 400, "target": {"mode": "entity", "entity_id": 52}}
                ],
                "satellite_steps": [],
                "motion": "straight",
            }
        ],
        "global_rules": [],
    }


class ValidatorPassTests(unittest.TestCase):
    def test_reasonable_plan_passes(self):
        report = validate(base_payload())
        self.assertTrue(report.passed, report.errors)

    def test_legally_terrible_plan_still_passes(self):
        # Everything launches at step 0 at one ocean coordinate; the L
        # platform (which cannot damage land targets) attacks a 9400 entity;
        # one platform never launches. All legal -> must PASS.
        payload = {
            "plan_version": "v0",
            "platforms": [
                {
                    "platform_id": pid,
                    "launch": {"mode": "at_step", "step": 0},
                    "initial_target": {
                        "mode": "coordinate",
                        "lon": 126.0,
                        "lat": 27.5,
                    },
                    "retarget_orders": [],
                    "satellite_steps": [],
                    "motion": "straight",
                }
                for pid in (10, 11, 12)
            ]
            + [
                {
                    "platform_id": 13,
                    "launch": {"mode": "at_step", "step": 1},
                    "initial_target": {"mode": "entity", "entity_id": 51},
                    "retarget_orders": [],
                    "satellite_steps": [],
                    "motion": "straight",
                },
                {
                    "platform_id": 14,
                    "launch": {"mode": "never"},
                    "satellite_steps": [],
                    "motion": "straight",
                },
            ],
            "global_rules": [],
        }
        report = validate(payload)
        self.assertTrue(report.passed, report.errors)

    def test_event_entity_inside_new_detection_rule_passes(self):
        payload = base_payload()
        payload["global_rules"] = [
            {
                "rule_id": "ship1",
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
        ]
        report = validate(payload)
        self.assertTrue(report.passed, report.errors)


class ValidatorRejectTests(unittest.TestCase):
    def test_unknown_platform_id(self):
        payload = base_payload()
        payload["platforms"][0]["platform_id"] = 999
        report = validate(payload)
        self.assertFalse(report.passed)
        self.assertTrue(any("not a controllable" in item for item in report.errors))

    def test_duplicate_platform_plan(self):
        payload = base_payload()
        entry = copy.deepcopy(payload["platforms"][0])
        payload["platforms"].append(entry)
        report = validate(payload)
        self.assertFalse(report.passed)
        self.assertTrue(any("duplicate" in item for item in report.errors))

    def test_step_out_of_range(self):
        payload = base_payload()
        payload["platforms"][0]["launch"]["step"] = 1200
        report = validate(payload)
        self.assertFalse(report.passed)
        self.assertTrue(any("launch step" in item for item in report.errors))

    def test_unknown_entity_reference(self):
        payload = base_payload()
        payload["platforms"][0]["initial_target"]["entity_id"] = 168  # hidden ship
        report = validate(payload)
        self.assertFalse(report.passed)
        self.assertTrue(any("does not legally know" in item for item in report.errors))

    def test_event_entity_outside_event_rule(self):
        payload = base_payload()
        payload["platforms"][0]["retarget_orders"] = [
            {"step": 500, "target": {"mode": "event_entity"}}
        ]
        report = validate(payload)
        self.assertFalse(report.passed)
        self.assertTrue(
            any("event_entity" in item for item in report.errors)
        )

    def test_launch_never_with_initial_target(self):
        payload = base_payload()
        payload["platforms"][0]["launch"] = {"mode": "never"}
        report = validate(payload)
        self.assertFalse(report.passed)
        self.assertTrue(any("never" in item for item in report.errors))

    def test_retarget_before_launch(self):
        payload = base_payload()
        payload["platforms"][0]["retarget_orders"] = [
            {"step": 2, "target": {"mode": "entity", "entity_id": 52}}
        ]
        report = validate(payload)
        self.assertFalse(report.passed)
        self.assertTrue(any("before" in item for item in report.errors))

    def test_two_retargets_same_step(self):
        payload = base_payload()
        payload["platforms"][0]["retarget_orders"] = [
            {"step": 100, "target": {"mode": "entity", "entity_id": 52}},
            {"step": 100, "target": {"mode": "entity", "entity_id": 53}},
        ]
        report = validate(payload)
        self.assertFalse(report.passed)
        self.assertTrue(any("already scheduled" in item for item in report.errors))

    def test_satellite_budget_exceeded(self):
        payload = {
            "plan_version": "v0",
            "platforms": [
                {
                    "platform_id": 10,
                    "launch": {"mode": "at_step", "step": 0},
                    "initial_target": {"mode": "entity", "entity_id": 51},
                    "retarget_orders": [],
                    "satellite_steps": [{"step": step} for step in range(1, 6)],
                    "motion": "straight",
                }
            ],
            "global_rules": [],
        }
        report = validate(payload, satellite_cap=4)
        self.assertFalse(report.passed)
        self.assertTrue(any("exceeding the team budget" in item for item in report.errors))

    def test_two_satellite_requests_same_step(self):
        payload = base_payload()
        payload["platforms"][0]["satellite_steps"] = [{"step": 50}]
        payload["platforms"].append(
            {
                "platform_id": 11,
                "launch": {"mode": "at_step", "step": 0},
                "initial_target": {"mode": "entity", "entity_id": 52},
                "retarget_orders": [],
                "satellite_steps": [{"step": 50}],
                "motion": "straight",
            }
        )
        report = validate(payload)
        self.assertFalse(report.passed)
        self.assertTrue(any("at most one per step" in item for item in report.errors))

    def test_launched_steps_ago_on_never_platform(self):
        payload = base_payload()
        payload["global_rules"] = [
            {
                "rule_id": "bad_timer",
                "trigger": {
                    "type": "launched_steps_ago",
                    "platform_id": 14,
                    "steps": 10,
                },
                "actions": [
                    {
                        "type": "satellite_request",
                        "platform_id": 14,
                    }
                ],
            }
        ]
        payload["platforms"].append(
            {
                "platform_id": 14,
                "launch": {"mode": "never"},
                "satellite_steps": [],
                "motion": "straight",
            }
        )
        report = validate(payload)
        self.assertFalse(report.passed)
        self.assertTrue(any("never launches" in item for item in report.errors))

    def test_duplicate_rule_id(self):
        payload = base_payload()
        payload["global_rules"] = [
            {
                "rule_id": "same",
                "trigger": {"type": "at_step", "step": 100},
                "actions": [{"type": "satellite_request", "platform_id": 10}],
            },
            {
                "rule_id": "same",
                "trigger": {"type": "at_step", "step": 200},
                "actions": [{"type": "satellite_request", "platform_id": 11}],
            },
        ]
        payload["platforms"].append(
            {
                "platform_id": 11,
                "launch": {"mode": "at_step", "step": 0},
                "initial_target": {"mode": "entity", "entity_id": 52},
                "retarget_orders": [],
                "satellite_steps": [],
                "motion": "straight",
            }
        )
        report = validate(payload)
        self.assertFalse(report.passed)
        self.assertTrue(any("duplicate rule_id" in item for item in report.errors))

    def test_non_finite_coordinate(self):
        payload = base_payload()
        payload["platforms"][0]["initial_target"] = {
            "mode": "coordinate",
            "lon": float("inf"),
            "lat": 22.0,
        }
        plan, errors = parse_plan(payload)
        self.assertIsNone(plan)  # structural parser rejects non-finite numbers


class ValidatorPurityTests(unittest.TestCase):
    def test_validator_never_modifies_the_plan(self):
        payload = base_payload()
        snapshot = copy.deepcopy(payload)
        plan, parse_errors = parse_plan(payload)
        self.assertEqual(parse_errors, [])
        plan_snapshot = copy.deepcopy(plan.to_dict())
        validate_plan(
            plan,
            controlled_platform_ids=CONTROLLED,
            known_entity_ids=KNOWN,
            max_steps=1200,
            satellite_max_use_count=100,
        )
        self.assertEqual(payload, snapshot)  # raw payload untouched
        self.assertEqual(plan.to_dict(), plan_snapshot)  # parsed plan untouched

    def test_rejected_report_still_leaves_plan_unchanged(self):
        payload = base_payload()
        payload["platforms"][0]["platform_id"] = 4242  # will be rejected
        plan, _ = parse_plan(payload)
        snapshot = copy.deepcopy(plan.to_dict())
        report = validate_plan(
            plan,
            controlled_platform_ids=CONTROLLED,
            known_entity_ids=KNOWN,
            max_steps=1200,
            satellite_max_use_count=100,
        )
        self.assertFalse(report.passed)
        self.assertEqual(plan.to_dict(), snapshot)
        self.assertTrue(report.errors)


if __name__ == "__main__":
    unittest.main()
