"""Executor tests: literal execution, event binding, no substitution."""

from __future__ import annotations

import copy
import unittest

from personal_train.llm_strategy.executor import PlanExecutor
from personal_train.llm_strategy.plan_schema import parse_plan
from personal_train.llm_strategy.tests.fakes import (
    make_detection,
    make_init_obs,
    make_isolated_obs,
)

INIT_CATALOGUE = make_init_obs(
    [
        {"id": 51, "type": 9400, "lon": 123.25, "lat": 24.75, "health": 32.0},
        {"id": 52, "type": 9400, "lon": 123.6, "lat": 23.0, "health": 32.0},
    ]
)
CONTROLLED = [10, 11, 12]


def executor_for(payload, trace):
    plan, errors = parse_plan(payload)
    if plan is None:
        raise AssertionError(f"fixture plan failed to parse: {errors}")
    return PlanExecutor(
        plan,
        init_ship_observation=INIT_CATALOGUE,
        controlled_platform_ids=CONTROLLED,
        on_trace=trace,
    )


def obs_at(step, *, alive=(10, 11, 12), detections=()):
    return [
        make_isolated_obs(
            entity_id=pid, entity_type=21000, step=step, detections=detections
        )
        for pid in alive
    ]


class LaunchAndTargetResolutionTests(unittest.TestCase):
    def test_entity_target_resolves_to_registry_coordinates(self):
        trace = []
        executor = executor_for(
            {
                "plan_version": "v0",
                "platforms": [
                    {
                        "platform_id": 10,
                        "launch": {"mode": "at_step", "step": 0},
                        "initial_target": {"mode": "entity", "entity_id": 51},
                        "retarget_orders": [],
                        "satellite_steps": [],
                        "motion": "straight",
                    }
                ],
                "global_rules": [],
            },
            trace.append,
        )
        executor.begin_step(0, obs_at(0))
        rows = executor.actions_for(10, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], 1.0)
        self.assertEqual(rows[0][1], 10.0)
        self.assertAlmostEqual(rows[0][2], 123.25)
        self.assertAlmostEqual(rows[0][3], 24.75)

    def test_coordinate_target_is_used_verbatim(self):
        trace = []
        executor = executor_for(
            {
                "plan_version": "v0",
                "platforms": [
                    {
                        "platform_id": 10,
                        "launch": {"mode": "at_step", "step": 3},
                        "initial_target": {
                            "mode": "coordinate",
                            "lon": 120.0,
                            "lat": 21.0,
                        },
                        "retarget_orders": [],
                        "satellite_steps": [],
                        "motion": "straight",
                    }
                ],
                "global_rules": [],
            },
            trace.append,
        )
        executor.begin_step(0, obs_at(0))
        self.assertEqual(executor.actions_for(10, 0), [])
        executor.begin_step(3, obs_at(3))
        rows = executor.actions_for(10, 3)
        self.assertEqual(rows, [[1.0, 10.0, 120.0, 21.0]])

    def test_entity_track_updates_move_the_aim_point(self):
        trace = []
        executor = executor_for(
            {
                "plan_version": "v0",
                "platforms": [
                    {
                        "platform_id": 10,
                        "launch": {"mode": "at_step", "step": 0},
                        "initial_target": {"mode": "entity", "entity_id": 51},
                        "retarget_orders": [],
                        "satellite_steps": [],
                        "motion": "straight",
                    }
                ],
                "global_rules": [],
            },
            trace.append,
        )
        executor.begin_step(0, obs_at(0))
        executor.begin_step(5, obs_at(5, detections=[
            make_detection(entity_id=51, entity_type=9400, lon=124.0, lat=25.0, time=5)
        ]))
        executor.begin_step(
            6,
            obs_at(6),
        )
        # Launch already happened at step 0 with the catalogue coordinates.
        self.assertEqual(executor.actions_for(10, 6), [])
        # A later retarget to the same entity uses the freshest track.
        trace2 = []
        executor2 = executor_for(
            {
                "plan_version": "v0",
                "platforms": [
                    {
                        "platform_id": 10,
                        "launch": {"mode": "at_step", "step": 0},
                        "initial_target": {"mode": "entity", "entity_id": 51},
                        "retarget_orders": [
                            {"step": 6, "target": {"mode": "entity", "entity_id": 51}}
                        ],
                        "satellite_steps": [],
                        "motion": "straight",
                    }
                ],
                "global_rules": [],
            },
            trace2.append,
        )
        executor2.begin_step(0, obs_at(0))
        executor2.begin_step(5, obs_at(5, detections=[
            make_detection(entity_id=51, entity_type=9400, lon=124.0, lat=25.0, time=5)
        ]))
        executor2.begin_step(6, obs_at(6))
        rows = executor2.actions_for(10, 6)
        self.assertEqual(rows, [[2.0, 10.0, 124.0, 25.0]])


class EventBindingTests(unittest.TestCase):
    def payload_with_ship_rule(self, occurrence=1):
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
                }
            ],
            "global_rules": [
                {
                    "rule_id": "ship_strike",
                    "trigger": {
                        "type": "new_detection",
                        "entity_type": 9500,
                        "occurrence": occurrence,
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

    def test_event_entity_uses_the_triggering_detection_coordinates(self):
        trace = []
        executor = executor_for(self.payload_with_ship_rule(), trace.append)
        executor.begin_step(0, obs_at(0))
        self.assertEqual(executor.actions_for(10, 0), [[1.0, 10.0, 123.25, 24.75]])
        executor.begin_step(4, obs_at(4, detections=[
            make_detection(entity_id=168, entity_type=9500, lon=119.3, lat=25.35, time=4)
        ]))
        rows = executor.actions_for(10, 4)
        self.assertEqual(rows, [[2.0, 10.0, 119.3, 25.35]])
        record = [item for item in trace if item["plan_rule_id"].startswith("ship_strike")]
        self.assertEqual(len(record), 1)
        self.assertTrue(record[0]["success"])

    def test_occurrence_orders_entities_by_id_within_a_step(self):
        trace = []
        executor = executor_for(self.payload_with_ship_rule(occurrence=2), trace.append)
        executor.begin_step(0, obs_at(0))
        # Two new ships detected in the same step; occurrence 2 -> id 200.
        executor.begin_step(3, obs_at(3, detections=[
            make_detection(entity_id=200, entity_type=9500, lon=1.0, lat=2.0, time=3),
            make_detection(entity_id=199, entity_type=9500, lon=3.0, lat=4.0, time=3),
        ]))
        rows = executor.actions_for(10, 3)
        self.assertEqual(rows, [[2.0, 10.0, 1.0, 2.0]])

    def test_rule_fires_at_most_once(self):
        trace = []
        executor = executor_for(self.payload_with_ship_rule(), trace.append)
        executor.begin_step(0, obs_at(0))
        executor.begin_step(4, obs_at(4, detections=[
            make_detection(entity_id=168, entity_type=9500, lon=119.3, lat=25.35, time=4)
        ]))
        executor.begin_step(5, obs_at(5, detections=[
            make_detection(entity_id=168, entity_type=9500, lon=119.4, lat=25.4, time=5)
        ]))
        fired = [item for item in trace if item["plan_rule_id"].startswith("ship_strike")]
        self.assertEqual(len(fired), 1)

    def test_launched_steps_ago_trigger_fires_on_schedule(self):
        trace = []
        payload = {
            "plan_version": "v0",
            "platforms": [
                {
                    "platform_id": 10,
                    "launch": {"mode": "at_step", "step": 2},
                    "initial_target": {"mode": "entity", "entity_id": 51},
                    "retarget_orders": [],
                    "satellite_steps": [],
                    "motion": "straight",
                }
            ],
            "global_rules": [
                {
                    "rule_id": "after_flight",
                    "trigger": {
                        "type": "launched_steps_ago",
                        "platform_id": 10,
                        "steps": 3,
                    },
                    "actions": [
                        {
                            "type": "retarget",
                            "platform_id": 10,
                            "target": {"mode": "entity", "entity_id": 52},
                        }
                    ],
                }
            ],
        }
        executor = executor_for(payload, trace.append)
        for step in range(0, 6):
            executor.begin_step(step, obs_at(step))
        rows = executor.actions_for(10, 5)
        self.assertEqual(rows, [[2.0, 10.0, 123.6, 23.0]])


class SatelliteAndStraightTests(unittest.TestCase):
    def test_satellite_request_executes_on_a_non_launch_step(self):
        trace = []
        payload = {
            "plan_version": "v0",
            "platforms": [
                {
                    "platform_id": 10,
                    "launch": {"mode": "at_step", "step": 0},
                    "initial_target": {"mode": "entity", "entity_id": 51},
                    "retarget_orders": [],
                    "satellite_steps": [{"step": 7}],
                    "motion": "straight",
                }
            ],
            "global_rules": [],
        }
        executor = executor_for(payload, trace.append)
        step_rows: dict[int, list[list[float]]] = {}
        for step in range(0, 9):
            executor.begin_step(step, obs_at(step))
            rows = executor.actions_for(10, step)
            step_rows[step] = [list(row) for row in rows]
            for row in rows:
                self.assertNotEqual(row[0], 0.0)  # never a lateral command
        self.assertIn([3.0, 10.0, 0.0, 0.0], step_rows[7])

    def test_straight_motion_never_emits_acceleration_commands(self):
        trace = []
        payload = {
            "plan_version": "v0",
            "platforms": [
                {
                    "platform_id": pid,
                    "launch": {"mode": "at_step", "step": pid - 10},
                    "initial_target": {"mode": "entity", "entity_id": 51},
                    "retarget_orders": [
                        {"step": 5, "target": {"mode": "coordinate", "lon": 1.0, "lat": 2.0}}
                    ],
                    "satellite_steps": [{"step": 6}],
                    "motion": "straight",
                }
                for pid in (10, 11, 12)
            ],
            "global_rules": [],
        }
        executor = executor_for(payload, trace.append)
        for step in range(0, 12):
            executor.begin_step(step, obs_at(step))
            for pid in (10, 11, 12):
                for row in executor.actions_for(pid, step):
                    self.assertNotEqual(row[0], 0.0)
        self.assertTrue(
            all(
                item["engine_action"] is None
                or item["engine_action"][0] != 0.0
                for item in trace
            )
        )


class FailureHandlingTests(unittest.TestCase):
    def test_dead_platform_fails_without_substitution(self):
        trace = []
        payload = {
            "plan_version": "v0",
            "platforms": [
                {
                    "platform_id": 10,
                    "launch": {"mode": "at_step", "step": 0},
                    "initial_target": {"mode": "entity", "entity_id": 51},
                    "retarget_orders": [
                        {"step": 5, "target": {"mode": "entity", "entity_id": 52}}
                    ],
                    "satellite_steps": [],
                    "motion": "straight",
                },
                {
                    "platform_id": 11,
                    "launch": {"mode": "at_step", "step": 5},
                    "initial_target": {"mode": "entity", "entity_id": 52},
                    "retarget_orders": [],
                    "satellite_steps": [],
                    "motion": "straight",
                },
            ],
            "global_rules": [],
        }
        executor = executor_for(payload, trace.append)
        executor.begin_step(0, obs_at(0))
        # Platform 10 dies before step 5; platform 11 is alive.
        executor.begin_step(5, obs_at(5, alive=(11,)))
        failures = [
            item
            for item in trace
            if item["step"] == 5 and item["platform_id"] == 10
        ]
        self.assertEqual(len(failures), 1)
        self.assertFalse(failures[0]["success"])
        self.assertEqual(failures[0]["failure_reason"], "platform_not_alive")
        self.assertIsNone(failures[0]["engine_action"])
        # The surviving platform still executes its own plan.
        rows = executor.actions_for(11, 5)
        self.assertEqual(rows, [[1.0, 11.0, 123.6, 23.0]])
        # No other platform was commandeered for platform 10's command.
        self.assertEqual(executor.actions_for(10, 5), [])

    def test_unknown_entity_target_fails_without_fabrication(self):
        trace = []
        payload = {
            "plan_version": "v0",
            "platforms": [
                {
                    "platform_id": 10,
                    "launch": {"mode": "at_step", "step": 0},
                    "initial_target": {"mode": "entity", "entity_id": 51},
                    "retarget_orders": [
                        # 999 was never legally known: the validator rejects
                        # this at plan time, but the executor must also stay
                        # literal if handed such a plan.
                        {"step": 4, "target": {"mode": "entity", "entity_id": 999}}
                    ],
                    "satellite_steps": [],
                    "motion": "straight",
                }
            ],
            "global_rules": [],
        }
        executor = executor_for(payload, trace.append)
        executor.begin_step(0, obs_at(0))
        executor.begin_step(4, obs_at(4))
        failures = [item for item in trace if item["step"] == 4]
        self.assertEqual(len(failures), 1)
        self.assertFalse(failures[0]["success"])
        self.assertIn("999", failures[0]["failure_reason"])
        self.assertEqual(executor.actions_for(10, 4), [])

    def test_retarget_on_unlaunched_platform_fails(self):
        trace = []
        payload = {
            "plan_version": "v0",
            "platforms": [
                {
                    "platform_id": 10,
                    "launch": {"mode": "at_step", "step": 10},
                    "initial_target": {"mode": "entity", "entity_id": 51},
                    "retarget_orders": [],
                    "satellite_steps": [],
                    "motion": "straight",
                }
            ],
            "global_rules": [
                {
                    "rule_id": "early",
                    "trigger": {"type": "at_step", "step": 3},
                    "actions": [
                        {
                            "type": "retarget",
                            "platform_id": 10,
                            "target": {"mode": "entity", "entity_id": 52},
                        }
                    ],
                }
            ],
        }
        executor = executor_for(payload, trace.append)
        executor.begin_step(3, obs_at(3))
        record = trace[-1]
        self.assertFalse(record["success"])
        self.assertEqual(record["failure_reason"], "platform_not_launched")


class RuntimeConflictTests(unittest.TestCase):
    @staticmethod
    def _payload(actions):
        return {
            "plan_version": "v0",
            "platforms": [
                {
                    "platform_id": platform_id,
                    "launch": {"mode": "at_step", "step": 0},
                    "initial_target": {"mode": "entity", "entity_id": 51},
                    "retarget_orders": [],
                    "satellite_steps": [],
                    "motion": "straight",
                }
                for platform_id in (10, 11)
            ],
            "global_rules": [
                {
                    "rule_id": rule_id,
                    "trigger": {
                        "type": "new_detection",
                        "entity_type": entity_type,
                        "occurrence": 1,
                    },
                    "actions": [action],
                }
                for rule_id, entity_type, action in actions
            ],
        }

    def _run(self, actions):
        trace = []
        executor = executor_for(self._payload(actions), trace.append)
        executor.begin_step(0, obs_at(0))
        detections = [
            make_detection(
                entity_id=168, entity_type=9500, lon=119.3, lat=25.35, time=4
            ),
            make_detection(
                entity_id=240, entity_type=24000, lon=120.1, lat=24.2, time=4
            ),
        ]
        executor.begin_step(4, obs_at(4, detections=detections))
        return executor, trace

    def test_dynamic_retargets_for_same_platform_conflict(self):
        executor, trace = self._run(
            [
                ("ship_rule", 9500, {"type": "retarget", "platform_id": 10,
                    "target": {"mode": "event_entity"}}),
                ("air_rule", 24000, {"type": "retarget", "platform_id": 10,
                    "target": {"mode": "event_entity"}}),
            ]
        )
        self.assertEqual(executor.actions_for(10, 4), [])
        conflicts = [item for item in trace if item["step"] == 4]
        self.assertEqual(len(conflicts), 2)
        self.assertTrue(all(not item["success"] for item in conflicts))
        self.assertTrue(all(
            item["failure_reason"] == "runtime_plan_conflict"
            for item in conflicts
        ))
        expected = ["air_rule:actions[0]", "ship_rule:actions[0]"]
        self.assertTrue(all(item["conflicting_rule_ids"] == expected for item in conflicts))
        self.assertEqual(executor.runtime_conflicts, 1)

    def test_dynamic_satellite_requests_conflict(self):
        executor, trace = self._run(
            [
                ("ship_satellite", 9500,
                 {"type": "satellite_request", "platform_id": 10}),
                ("air_satellite", 24000,
                 {"type": "satellite_request", "platform_id": 11}),
            ]
        )
        self.assertEqual(executor.actions_for(10, 4), [])
        self.assertEqual(executor.actions_for(11, 4), [])
        conflicts = [item for item in trace if item["step"] == 4]
        self.assertEqual(len(conflicts), 2)
        self.assertTrue(all(
            item["failure_reason"] == "runtime_plan_conflict"
            for item in conflicts
        ))

    def test_dynamic_retargets_for_different_platforms_both_execute(self):
        executor, trace = self._run(
            [
                ("ship_rule", 9500, {"type": "retarget", "platform_id": 10,
                    "target": {"mode": "event_entity"}}),
                ("air_rule", 24000, {"type": "retarget", "platform_id": 11,
                    "target": {"mode": "event_entity"}}),
            ]
        )
        self.assertEqual(executor.actions_for(10, 4), [[2.0, 10.0, 119.3, 25.35]])
        self.assertEqual(executor.actions_for(11, 4), [[2.0, 11.0, 120.1, 24.2]])
        self.assertTrue(all(item["success"] for item in trace if item["step"] == 4))
        self.assertEqual(executor.runtime_conflicts, 0)

    def test_retarget_and_satellite_both_execute(self):
        executor, trace = self._run(
            [
                ("ship_rule", 9500, {"type": "retarget", "platform_id": 10,
                    "target": {"mode": "event_entity"}}),
                ("air_satellite", 24000,
                 {"type": "satellite_request", "platform_id": 10}),
            ]
        )
        self.assertEqual(
            executor.actions_for(10, 4),
            [[2.0, 10.0, 119.3, 25.35], [3.0, 10.0, 0.0, 0.0]],
        )
        self.assertTrue(all(item["success"] for item in trace if item["step"] == 4))
        self.assertEqual(executor.runtime_conflicts, 0)


class PlanImmutabilityTests(unittest.TestCase):
    def test_executor_never_modifies_the_accepted_plan(self):
        payload = {
            "plan_version": "v0",
            "platforms": [
                {
                    "platform_id": 10,
                    "launch": {"mode": "at_step", "step": 0},
                    "initial_target": {"mode": "entity", "entity_id": 51},
                    "retarget_orders": [
                        {"step": 3, "target": {"mode": "coordinate", "lon": 1.0, "lat": 2.0}}
                    ],
                    "satellite_steps": [{"step": 4}],
                    "motion": "straight",
                }
            ],
            "global_rules": [
                {
                    "rule_id": "ship",
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
        trace = []
        executor = executor_for(payload, trace.append)
        before = copy.deepcopy(executor.plan.to_dict())
        for step in range(0, 8):
            detections = (
                [make_detection(entity_id=168, entity_type=9500, lon=5.0, lat=6.0, time=step)]
                if step == 6
                else ()
            )
            executor.begin_step(step, obs_at(step, detections=detections))
            executor.actions_for(10, step)
        self.assertEqual(executor.plan.to_dict(), before)
        self.assertGreater(executor.executed_commands, 0)


if __name__ == "__main__":
    unittest.main()
