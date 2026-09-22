"""Small contract tests for the competition simulator bridge.

The full native smoke test is intentionally run through ``train_joint_ppo.py``;
these tests keep ordinary unit-test runs fast and avoid creating simulator files.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import personal_train.joint_game_env as joint_game_env_module

from personal_train.joint_game_env import (
    JointGameConfig,
    JointGameEnv,
    _ObjectiveTrack,
    _bbox,
    _ecf_velocity_to_enu_xy,
    _vector,
    denormalize_placement,
)
from personal_train.joint_rl_core import (
    ActivationReceipt,
    BinaryChoice,
    JointAction,
    JointControlTracker,
    JointSpaceSpec,
    SharedSensorAction,
    SharedSensorConfig,
    UnitAction,
)


class JointGameGeometryTests(unittest.TestCase):
    def test_deployment_action_covers_the_complete_rectangle(self):
        bounds = _bbox(
            (((116.0, 21.5), (116.0, 24.8), (117.0, 24.8), (117.0, 21.5)),)
        )
        self.assertEqual(bounds, (116.0, 117.0, 21.5, 24.8))
        self.assertEqual(denormalize_placement((-1.0, -1.0), bounds), (116.0, 21.5))
        self.assertEqual(denormalize_placement((1.0, 1.0), bounds), (117.0, 24.8))
        self.assertEqual(denormalize_placement((0.0, 0.0), bounds), (116.5, 23.15))
        # The adapter clamps small numerical excursions from tanh sampling.
        self.assertEqual(denormalize_placement((2.0, -2.0), bounds), (117.0, 21.5))
        with self.assertRaisesRegex(ValueError, "rectangular"):
            _bbox((((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),))

    def test_ecf_velocity_is_rotated_to_local_east_north(self):
        east, north = _ecf_velocity_to_enu_xy((0.0, 10.0, 0.0), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(east, 10.0)
        self.assertAlmostEqual(north, 0.0)
        east, north = _ecf_velocity_to_enu_xy((0.0, 0.0, 12.0), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(east, 0.0)
        self.assertAlmostEqual(north, 12.0)
        self.assertTrue(math.isfinite(east) and math.isfinite(north))

    def test_vector_accepts_upstream_mapping_and_object_forms(self):
        class VectorLike:
            x = 4
            y = 5.5
            z = -6

        self.assertEqual(_vector({"x": 1, "y": 2.5, "z": -3}), (1.0, 2.5, -3.0))
        self.assertEqual(_vector(VectorLike()), (4.0, 5.5, -6.0))
        self.assertIsNone(_vector(None))
        self.assertIsNone(_vector(object()))
        self.assertIsNone(_vector({"x": float("nan"), "y": 0.0, "z": 0.0}))
        self.assertIsNone(_vector({"x": 0.0, "y": float("inf"), "z": 0.0}))

    def test_team_track_union_requires_a_newer_source_timestamp(self):
        environment = object.__new__(JointGameEnv)
        environment.unit_ids = (10, 11)
        environment.objective_slot_by_id = {950: 0}
        environment._tracks = [_ObjectiveTrack(slot=0, entity_id=950)]
        environment.current_step = 5
        environment._update_tracks(
            {
                "entities": {
                    10: {
                        "detectInfo": {
                            950: {
                                "entity_id": 950,
                                "entity_type": 9500,
                                "time": 100,
                                "lla": {"x": 120.0, "y": 20.0, "z": 0.0},
                                "vel_ecf": {"x": 1.0, "y": 2.0, "z": 3.0},
                            }
                        }
                    }
                }
            }
        )
        track = environment._tracks[0]
        self.assertTrue(track.known)
        self.assertEqual(track.position, (120.0, 20.0))
        self.assertEqual(track.source_timestamp, 100)
        self.assertEqual(track.last_seen_step, 5)

        environment.current_step = 6
        environment._update_tracks(
            {
                "entities": {
                    11: {
                        "detectInfo": {
                            950: {
                                "entity_id": 950,
                                "entity_type": 9500,
                                "time": 100,
                                "lla": {"x": 130.0, "y": 30.0, "z": 0.0},
                            }
                        }
                    }
                }
            }
        )
        self.assertEqual(track.position, (120.0, 20.0))
        self.assertEqual(track.last_seen_step, 5)

        environment.current_step = 7
        environment._update_tracks(
            {
                "entities": {
                    11: {
                        "detectInfo": {
                            950: {
                                "entity_id": 950,
                                "entity_type": 9500,
                                "time": 101,
                                "lla": {"x": 130.0, "y": 30.0, "z": 0.0},
                            }
                        }
                    }
                }
            }
        )
        self.assertEqual(track.position, (130.0, 30.0))
        self.assertEqual(track.source_timestamp, 101)
        self.assertEqual(track.last_seen_step, 7)

    def test_objective_loads_follow_active_assignment_retarget_and_terminal_state(self):
        environment = object.__new__(JointGameEnv)
        environment.config = JointGameConfig(
            objective_slots=2,
            strict_weapon_target_compatibility=False,
        )
        environment.space = JointSpaceSpec(unit_count=5, objective_count=2)
        environment.unit_types = (21000, 21000, 21001, 21002, 21002)
        environment.tracker = JointControlTracker(
            environment.space,
            sensor_config=SharedSensorConfig(capacity=0),
        )
        environment.current_step = 0
        environment._tracks = [
            _ObjectiveTrack(slot=0, entity_id=950, known=True, position=(1.0, 0.0)),
            _ObjectiveTrack(slot=1, entity_id=951, known=True, position=(2.0, 0.0)),
        ]
        environment._progress = {}
        environment._map_diagonal_km = 200.0
        activation = JointAction.from_sequence(
            (
                UnitAction(slot=0, activate=BinaryChoice.YES, objective_slot=0),
                UnitAction.noop(1),
                UnitAction(slot=2, activate=BinaryChoice.YES, objective_slot=0),
                UnitAction(slot=3, activate=BinaryChoice.YES, objective_slot=0),
                UnitAction(slot=4, activate=BinaryChoice.YES, objective_slot=1),
            )
        )
        environment.tracker.apply(activation, (True, True), step=0)
        environment.tracker.confirm_activations(
            tuple(
                ActivationReceipt(slot=slot, request_step=0, accepted=True)
                for slot in (0, 2, 3, 4)
            ),
            step=0,
        )

        self.assertEqual(
            environment._objective_loads(),
            (
                (3 / 5, 1 / 2, 1.0, 1 / 2),
                (1 / 5, 0.0, 0.0, 1 / 2),
            ),
        )
        frames = environment._objective_frames()
        self.assertEqual(frames[0].assigned_total, 3 / 5)
        self.assertEqual(frames[0].assigned_high, 1 / 2)
        self.assertEqual(frames[0].assigned_medium, 1.0)
        self.assertEqual(frames[0].assigned_low, 1 / 2)
        environment._progress[0] = (0, 100.0, 0.0)
        reference, fraction = environment._target_progress_features(
            0, (0.5, 0.0, 0.0)
        )
        self.assertEqual(reference, 0.5)
        self.assertAlmostEqual(
            fraction,
            (
                100.0
                - environment._distance_km((0.5, 0.0), (1.0, 0.0))
            )
            / 100.0,
        )
        self.assertEqual(
            environment._target_progress_features(1, (0.0, 0.0, 0.0)),
            (0.0, 0.0),
        )

        environment.tracker.apply(
            JointAction.from_sequence(
                (
                    UnitAction(
                        slot=0,
                        retarget=BinaryChoice.YES,
                        objective_slot=1,
                    ),
                    UnitAction.noop(1),
                    UnitAction.noop(2),
                    UnitAction.noop(3),
                    UnitAction.noop(4),
                )
            ),
            (True, True),
            step=1,
        )
        self.assertEqual(
            environment._objective_loads(),
            (
                (2 / 5, 0.0, 1.0, 1 / 2),
                (2 / 5, 1 / 2, 0.0, 1 / 2),
            ),
        )
        self.assertEqual(
            environment._target_progress_features(0, (0.5, 0.0, 0.0)),
            (0.0, 0.0),
        )

        environment.tracker.mark_terminal((2,), step=2)
        self.assertEqual(
            environment._objective_loads(),
            (
                (1 / 5, 0.0, 0.0, 1 / 2),
                (2 / 5, 1 / 2, 0.0, 1 / 2),
            ),
        )
        environment._progress[2] = (0, 100.0, 0.0)
        self.assertEqual(
            environment._target_progress_features(2, (0.0, 0.0, 0.0)),
            (0.0, 0.0),
        )

    def test_invalid_environment_hyperparameters_fail_early(self):
        for kwargs in (
            {"objective_slots": 0},
            {"max_track_age_steps": 0},
            {"max_speed_mps": 0.0},
            {"sensor_capacity": -1},
            {"sensor_max_requests_per_step": 0},
            {"sensor_cooldown_steps": -1},
            {"gamma": 1.1},
            {"gamma": float("nan")},
            {"official_reward_scale": -1.0},
            {"planning_team_weight": -0.1, "planning_local_weight": 1.1},
            {"planning_team_weight": 0.6, "planning_local_weight": 0.3},
            {"sensor_information_potential_scale": -0.01},
            {"objective_slots": 2.5},
            {"sensor_capacity": True},
            {"terminate_on_all_objectives_destroyed": 1},
            {"strict_weapon_target_compatibility": 1},
            {"allow_low_altitude_search_fallback": 1},
            {"allow_low_altitude_search_replanning": 1},
            {"retarget_min_dwell_steps": -1},
            {"retarget_min_dwell_steps": True},
            {"retarget_decision_interval_steps": 0},
            {"retarget_decision_interval_steps": True},
            {"motion_decision_interval_steps": 0},
            {"motion_decision_interval_steps": True},
            {"post_launch_motion_only": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                JointGameConfig(**kwargs)

    def test_weapon_target_compatibility_masks_ineffective_pairings(self):
        environment = object.__new__(JointGameEnv)
        environment.config = JointGameConfig(
            objective_slots=3,
            retarget_min_dwell_steps=10,
        )
        environment.space = JointSpaceSpec(unit_count=3, objective_count=3)
        environment.unit_types = (21000, 21001, 21002)
        environment.tracker = JointControlTracker(
            environment.space,
            sensor_config=SharedSensorConfig(capacity=0),
        )
        environment.sensor_backend = "team_global"
        environment.engine = SimpleNamespace(
            simulator_factory=SimpleNamespace(
                red_sat_use_count=0,
                red_sat_max_use_count=0,
                sim_time=0,
                is_using_satellite=lambda: False,
            ),
            sim_time=0,
        )
        environment.current_step = 0
        environment._tracks = [
            _ObjectiveTrack(
                slot=0,
                entity_id=40,
                known=True,
                type_index=0,
                static_public=True,
            ),
            _ObjectiveTrack(
                slot=1,
                entity_id=41,
                known=True,
                type_index=1,
                static_public=True,
            ),
            _ObjectiveTrack(slot=2, entity_id=42, known=True, type_index=2),
        ]
        environment._cached_action_mask = None
        environment._cached_action_mask_step = -1

        mask = environment.action_mask()
        self.assertEqual(mask.by_unit[0].objective.tolist(), [True, True, False])
        self.assertEqual(mask.by_unit[1].objective.tolist(), [True, True, False])
        self.assertEqual(mask.by_unit[2].objective.tolist(), [False, False, True])
        self.assertEqual(mask.by_unit[0].movement.tolist(), [True, True, True])
        environment.config = JointGameConfig(
            objective_slots=3,
            retarget_min_dwell_steps=10,
            motion_decision_interval_steps=1,
            post_launch_motion_only=True,
        )
        environment._cached_action_mask = None
        launch_only = environment.action_mask()
        self.assertEqual(
            launch_only.by_unit[0].movement.tolist(), [False, True, False]
        )
        environment.config = JointGameConfig(
            objective_slots=3,
            retarget_min_dwell_steps=10,
        )
        environment._cached_action_mask = None
        self.assertTrue(environment.assignment_is_damage_compatible(0, 0))
        self.assertFalse(environment.assignment_is_damage_compatible(0, 2))
        self.assertFalse(environment.assignment_is_damage_compatible(2, 0))
        self.assertTrue(environment.assignment_is_damage_compatible(2, 2))

        environment.current_step = 301
        environment._cached_action_mask = None
        stale_ship = environment.action_mask()
        self.assertEqual(
            stale_ship.by_unit[2].objective.tolist(),
            [True, True, False],
        )

        # Before a ship is legally known, land objectives are explicit search
        # anchors so L missiles can become airborne and discover it locally.
        environment._tracks[2].known = False
        environment.current_step = 302
        environment._cached_action_mask = None
        no_ship = environment.action_mask()
        self.assertEqual(no_ship.by_unit[2].activation.tolist(), [True, True])
        self.assertEqual(no_ship.by_unit[2].objective.tolist(), [True, True, False])

        environment.config = JointGameConfig(
            objective_slots=3,
            allow_low_altitude_search_fallback=False,
        )
        environment.current_step = 303
        environment._cached_action_mask = None
        strict_wait = environment.action_mask()
        self.assertEqual(strict_wait.by_unit[2].activation.tolist(), [True, False])
        self.assertEqual(
            strict_wait.by_unit[2].objective.tolist(),
            [False, False, False],
        )

        # Re-enable search fallback and launch L against a land search anchor.
        # A throttled movement branch is neutral at launch and between pulses.
        environment.config = JointGameConfig(
            objective_slots=3,
            retarget_min_dwell_steps=10,
            motion_decision_interval_steps=10,
        )
        environment.current_step = 304
        environment._cached_action_mask = None
        activation = JointAction.from_sequence(
            (
                UnitAction.noop(0),
                UnitAction.noop(1),
                UnitAction(
                    slot=2,
                    activate=BinaryChoice.YES,
                    objective_slot=0,
                ),
            )
        )
        environment.tracker.apply(
            activation,
            environment._objective_validity(),
            step=304,
        )
        environment.tracker.confirm_activations(
            (ActivationReceipt(slot=2, request_step=304, accepted=True),),
            step=304,
        )
        environment._cached_action_mask = None
        launch_step = environment.action_mask()
        self.assertEqual(
            launch_step.by_unit[2].movement.tolist(),
            [False, True, False],
        )

        environment.current_step = 313
        environment._cached_action_mask = None
        before_pulse = environment.action_mask()
        self.assertEqual(
            before_pulse.by_unit[2].movement.tolist(),
            [False, True, False],
        )

        # Routine land-to-land replanning is locked even after dwell elapses.
        environment.current_step = 314
        environment._cached_action_mask = None
        locked_search = environment.action_mask()
        self.assertEqual(
            locked_search.by_unit[2].objective.tolist(),
            [False, True, False],
        )
        self.assertEqual(
            locked_search.by_unit[2].retarget.tolist(),
            [True, False],
        )
        self.assertEqual(
            locked_search.by_unit[2].movement.tolist(),
            [True, True, True],
        )

        # The compatibility escape hatch restores routine replanning.
        environment.config = JointGameConfig(
            objective_slots=3,
            allow_low_altitude_search_replanning=True,
            retarget_min_dwell_steps=10,
            motion_decision_interval_steps=10,
        )
        environment.current_step = 315
        environment._cached_action_mask = None
        replanning = environment.action_mask()
        self.assertEqual(replanning.by_unit[2].retarget.tolist(), [True, True])
        self.assertEqual(
            replanning.by_unit[2].movement.tolist(),
            [False, True, False],
        )

        # A newly known ship invalidates the search anchor and bypasses both
        # the routine lock and dwell, so mission retargeting opens immediately.
        environment.config = JointGameConfig(
            objective_slots=3,
            retarget_min_dwell_steps=100,
            motion_decision_interval_steps=10,
        )
        environment._tracks[2].known = True
        environment._tracks[2].last_seen_step = 315
        environment._cached_action_mask = None
        fresh_ship = environment.action_mask()
        self.assertEqual(
            fresh_ship.by_unit[2].objective.tolist(),
            [False, False, True],
        )
        self.assertEqual(fresh_ship.by_unit[2].retarget.tolist(), [True, True])
        self.assertEqual(
            fresh_ship.by_unit[2].movement.tolist(),
            [False, True, False],
        )

    def test_compatibility_filter_can_be_disabled_explicitly(self):
        environment = object.__new__(JointGameEnv)
        environment.config = JointGameConfig(
            objective_slots=2,
            strict_weapon_target_compatibility=False,
        )
        environment.unit_types = (21002,)
        environment._tracks = [
            _ObjectiveTrack(slot=0, entity_id=40, known=True, type_index=0),
            _ObjectiveTrack(slot=1, entity_id=41, known=True, type_index=1),
        ]
        self.assertEqual(
            environment._objective_validity_by_unit(),
            ((True, True),),
        )
        self.assertTrue(environment.assignment_is_damage_compatible(0, 0))

    def test_sensor_information_reward_closes_only_on_true_terminal(self):
        environment = object.__new__(JointGameEnv)
        environment.config = JointGameConfig(gamma=0.9)
        environment._sensor_information_potential = 0.01

        truncated_reward = environment._advance_sensor_information_reward(
            team_reward=0.2,
            current_potential=0.015,
            true_terminal=False,
        )
        self.assertAlmostEqual(truncated_reward, 0.2 + 0.9 * 0.015 - 0.01)
        self.assertAlmostEqual(environment._sensor_information_potential, 0.015)

        terminal_reward = environment._advance_sensor_information_reward(
            team_reward=0.3,
            current_potential=0.015,
            true_terminal=True,
        )
        self.assertAlmostEqual(terminal_reward, 0.3 - 0.015)
        self.assertEqual(environment._sensor_information_potential, 0.0)

    def test_information_potential_uses_known_objective_weight(self):
        environment = object.__new__(JointGameEnv)
        environment.config = JointGameConfig(
            sensor_information_potential_scale=0.018
        )
        environment._objective_weight_by_slot = {0: 5.0, 1: 1.0}
        environment._tracks = [
            _ObjectiveTrack(slot=0, entity_id=51, known=False),
            _ObjectiveTrack(slot=1, entity_id=52, known=True),
        ]
        self.assertAlmostEqual(environment._objective_information_potential(), 0.003)

    def test_team_global_sensor_mask_uses_factory_window_and_capacity(self):
        class FakeFactory:
            red_sat_use_count = 0
            red_sat_max_use_count = 100
            sim_time = 0
            end_time = 0

            def is_using_satellite(self) -> bool:
                return self.sim_time < self.end_time

        environment = object.__new__(JointGameEnv)
        environment.config = JointGameConfig(
            objective_slots=1,
            strict_weapon_target_compatibility=False,
        )
        environment.space = JointSpaceSpec(unit_count=2, objective_count=1)
        environment.unit_types = (21000, 21000)
        environment.tracker = JointControlTracker(
            environment.space,
            sensor_config=SharedSensorConfig(
                capacity=100,
                max_requests_per_step=3,
            ),
        )
        activation = JointAction.from_sequence(
            (
                UnitAction(slot=0, activate=BinaryChoice.YES, objective_slot=0),
                UnitAction(slot=1, activate=BinaryChoice.YES, objective_slot=0),
            )
        )
        environment.tracker.apply(activation, (True,), step=0)
        environment.tracker.confirm_activations(
            (
                ActivationReceipt(slot=0, request_step=0, accepted=True),
                ActivationReceipt(slot=1, request_step=0, accepted=True),
            ),
            step=0,
        )
        factory = FakeFactory()
        environment.engine = SimpleNamespace(
            simulator_factory=factory,
            sim_time=1_000,
        )
        environment.sensor_backend = "team_global"
        environment.current_step = 1
        environment._tracks = [
            _ObjectiveTrack(slot=0, entity_id=51, known=True)
        ]
        environment._cached_action_mask = None
        environment._cached_action_mask_step = -1

        available = environment.action_mask()
        self.assertEqual(available.shared_sensor_eligible.tolist(), [True, True])
        self.assertEqual(available.shared_sensor_max_requests, 1)
        self.assertEqual(factory.sim_time, 1_000)

        factory.end_time = 181_000
        environment.current_step = 2
        environment.engine.sim_time = 2_000
        environment._cached_action_mask = None
        active = environment.action_mask()
        self.assertEqual(active.shared_sensor_eligible.tolist(), [False, False])
        self.assertEqual(active.shared_sensor_max_requests, 0)
        self.assertFalse(environment._sensor_ready_override())

        environment.current_step = 181
        environment.engine.sim_time = 181_000
        environment._cached_action_mask = None
        expired = environment.action_mask()
        self.assertEqual(expired.shared_sensor_eligible.tolist(), [True, True])
        self.assertEqual(expired.shared_sensor_max_requests, 1)

        factory.red_sat_use_count = 100
        environment.current_step = 182
        environment._cached_action_mask = None
        exhausted = environment.action_mask()
        self.assertEqual(exhausted.shared_sensor_eligible.tolist(), [False, False])

    def test_only_team_global_sensor_is_available_while_units_are_staged(self):
        class FakeFactory:
            red_sat_use_count = 0
            red_sat_max_use_count = 100
            sim_time = 0

            @staticmethod
            def is_using_satellite() -> bool:
                return False

        environment = object.__new__(JointGameEnv)
        environment.config = JointGameConfig(
            objective_slots=1,
            strict_weapon_target_compatibility=False,
        )
        environment.space = JointSpaceSpec(unit_count=2, objective_count=1)
        environment.unit_types = (21000, 21000)
        environment.tracker = JointControlTracker(
            environment.space,
            sensor_config=SharedSensorConfig(capacity=100),
        )
        environment.tracker.reset()
        environment.engine = SimpleNamespace(
            simulator_factory=FakeFactory(),
            sim_time=0,
        )
        environment.current_step = 0
        environment._tracks = [_ObjectiveTrack(slot=0, entity_id=51, known=True)]
        environment._cached_action_mask = None
        environment._cached_action_mask_step = -1

        environment.sensor_backend = "team_global"
        global_mask = environment.action_mask()
        self.assertEqual(global_mask.shared_sensor_eligible.tolist(), [True, True])
        self.assertEqual(global_mask.shared_sensor_max_requests, 1)

        environment.sensor_backend = "per_unit"
        environment.sensor_backend_capacity_per_unit = 100
        environment._cached_action_mask = None
        per_unit_mask = environment.action_mask()
        self.assertEqual(per_unit_mask.shared_sensor_eligible.tolist(), [False, False])
        self.assertEqual(per_unit_mask.shared_sensor_max_requests, 0)

    def test_team_global_step_accepts_activation_and_sensor_from_staged_unit(self):
        class FakeFactory:
            def __init__(self) -> None:
                self.red_sat_use_count = 0
                self.red_sat_max_use_count = 100
                self.sim_time = 0
                self.end_time = 0
                self.modified_positions: list[tuple[int, dict[str, float]]] = []

            def is_using_satellite(self) -> bool:
                return self.sim_time < self.end_time

            def modify_simulator_position(
                self,
                entity_id: int,
                position: dict[str, float],
            ) -> None:
                self.modified_positions.append((entity_id, position))

        class FakeEngine:
            def __init__(self) -> None:
                self.simulator_factory = FakeFactory()
                self.sim_time = 0
                self.simulator = SimpleNamespace(launch=0)

            def get_simulator_by_id(self, entity_id: int):
                return self.simulator if entity_id == 10 else None

            def step(self, commands) -> None:
                self.simulator.launch = 1
                self.simulator_factory.red_sat_use_count += 1
                self.simulator_factory.end_time = 180_000
                self.sim_time = 1_000

        class FakeRewardTracker:
            @staticmethod
            def check_completion(step: int, observation) -> None:
                return None

            @staticmethod
            def finish(observation):
                return SimpleNamespace(score=0.0, completed=False)

        raw = {
            "entities": {
                10: {
                    "health": 1.0,
                    "isVisible": True,
                    "position": {"lon": 0.5, "lat": 0.5, "alt": 0.0},
                }
            }
        }
        environment = object.__new__(JointGameEnv)
        environment.config = JointGameConfig(objective_slots=1)
        environment.reward_policy = SimpleNamespace(max_steps=10)
        environment.max_steps = 10
        environment.is_debug_horizon = False
        environment.current_step = 0
        environment.space = JointSpaceSpec(unit_count=1, objective_count=1)
        environment.unit_ids = (10,)
        environment.unit_types = (21000,)
        environment.tracker = JointControlTracker(
            environment.space,
            sensor_config=SharedSensorConfig(capacity=100),
        )
        environment.sensor_backend = "team_global"
        environment.sensor_backend_capacity_per_unit = None
        environment.engine = FakeEngine()
        environment._environment = SimpleNamespace(
            current_round=1,
            current_step=0,
            _get_observation=lambda: raw,
            get_is_done=lambda: False,
            render_mode=None,
            renderer=None,
        )
        environment._tracks = [
            _ObjectiveTrack(
                slot=0,
                entity_id=51,
                known=True,
                position=(1.0, 1.0),
                type_index=0,
            )
        ]
        environment._deployment_bounds = {21000: (0.0, 1.0, 0.0, 1.0)}
        environment._last_raw_observation = raw
        environment._last_unit_ecf = {}
        environment._reward_tracker = FakeRewardTracker()
        environment._score = 0.0
        environment._launch_count = 0
        environment._sensor_request_count = 0
        environment._sensor_information_potential = 0.0
        environment._cached_action_mask = None
        environment._cached_action_mask_step = -1
        environment._update_tracks = lambda observation: None
        environment._sensor_information_value = lambda observation: 0.0
        environment._unit_rewards = lambda *args, **kwargs: (0.0,)
        environment._encode = lambda observation: (
            np.zeros(1, dtype=np.float32),
        )
        environment._ecf_snapshot = lambda observation: {}

        action = JointAction.from_sequence(
            (
                UnitAction(
                    slot=0,
                    activate=BinaryChoice.YES,
                    placement=(0.0, 0.0),
                    objective_slot=0,
                ),
            ),
            SharedSensorAction.from_sequence((0,)),
        )
        with (
            patch.object(joint_game_env_module, "write_ai_action"),
            patch.object(joint_game_env_module, "write_state"),
            patch.object(
                joint_game_env_module.CommandAdapter,
                "common_adapter",
                side_effect=lambda commands: list(commands),
            ),
        ):
            result = environment.step(action)

        self.assertEqual(result.accepted_activations, (0,))
        self.assertEqual(result.accepted_sensor_requests, (0,))
        self.assertEqual(result.submitted_commands, 3)
        self.assertEqual(environment.tracker.sensor_state.remaining, 99)
        self.assertEqual(environment.sensor_request_count, 1)
        self.assertEqual(environment.launch_count, 1)
        self.assertEqual(environment.engine.simulator_factory.red_sat_use_count, 1)

    def test_global_sensor_information_uses_unique_fresh_interceptors(self):
        environment = object.__new__(JointGameEnv)
        environment.unit_ids = (10, 11)
        environment.config = JointGameConfig(
            max_track_age_steps=100,
            sensor_information_potential_scale=0.02,
        )
        environment._sensor_threat_normalizer = 2
        environment.engine = SimpleNamespace(
            sim_time=100_000,
            profile=SimpleNamespace(
                imagineProfile=SimpleNamespace(simStep=1_000)
            ),
        )
        observation = {
            "entities": {
                10: {
                    "detectInfo": {
                        2401: {
                            "entity_id": 2401,
                            "entity_type": 24000,
                            "time": 99_000,
                            "lla": {"x": 120.0, "y": 22.0, "z": 1_000.0},
                        },
                        2402: {
                            "entity_id": 2402,
                            "entity_type": 24000,
                            "time": -1_000,
                            "lla": {"x": 121.0, "y": 23.0, "z": 1_000.0},
                        },
                    }
                },
                11: {
                    "detectInfo": {
                        2401: {
                            "entity_id": 2401,
                            "entity_type": 24000,
                            "time": 98_000,
                            "lla": {"x": 120.0, "y": 22.0, "z": 1_000.0},
                        }
                    }
                },
            }
        }
        ages = environment._fresh_interceptor_track_ages(observation)
        self.assertEqual(ages, {2401: 1.0})
        self.assertAlmostEqual(
            environment._interceptor_information_potential(observation),
            0.02 * 0.99 / 2,
        )


if __name__ == "__main__":
    unittest.main()
