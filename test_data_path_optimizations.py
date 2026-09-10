"""Regression tests for allocation-saving joint rollout data paths."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from personal_train.joint_game_env import JointGameEnv
from personal_train.joint_rl_core import (
    JointAction,
    JointObservationEncoder,
    JointPolicyTrace,
    JointSpaceSpec,
    JointTrajectoryBuffer,
    JointTransition,
    MapBounds,
    ObjectiveFrame,
    ObservationEncoderConfig,
    SharedSensorAction,
    SharedSensorConfig,
    SharedSensorState,
    UnitAction,
    UnitControlState,
    UnitFrame,
    build_joint_action_mask,
)


class ObservationBatchTests(unittest.TestCase):
    def test_encode_many_is_bitwise_equal_to_individual_encoding(self) -> None:
        space = JointSpaceSpec(unit_count=2, objective_count=2)
        encoder = JointObservationEncoder(
            ObservationEncoderConfig(
                max_steps=100,
                space=space,
                bounds=MapBounds(0.0, 10.0, -5.0, 5.0, 0.0, 20.0),
                max_speed=4.0,
                max_track_age_steps=20,
                unit_type_count=3,
                objective_type_count=2,
            )
        )
        units = (
            UnitFrame(
                slot=0,
                type_index=1,
                position=(2.0, -1.0, 5.0),
                velocity_xy=(1.0, -2.0),
                velocity_known=True,
            ),
            UnitFrame(
                slot=1,
                type_index=2,
                position=(8.0, 3.0, 15.0),
                nearest_threat_known=True,
                nearest_threat_position=(6.0, 2.0),
            ),
        )
        states = (UnitControlState(slot=0), UnitControlState(slot=1))
        objectives = (
            ObjectiveFrame(slot=0, valid=True, known=True, position=(4.0, 1.0)),
            ObjectiveFrame(slot=1, valid=False, known=False),
        )
        sensor = SharedSensorState.initial(SharedSensorConfig(capacity=2))

        expected = tuple(
            encoder.encode(unit, state, objectives, sensor, step=7)
            for unit, state in zip(units, states)
        )
        actual = encoder.encode_many(units, states, objectives, sensor, step=7)

        self.assertIs(encoder.feature_names, encoder.feature_names)
        self.assertEqual(encoder.dimension, len(encoder.feature_names))
        for individual, batched in zip(expected, actual):
            np.testing.assert_array_equal(batched, individual)


class TrajectorySnapshotTests(unittest.TestCase):
    @staticmethod
    def _buffer() -> JointTrajectoryBuffer:
        space = JointSpaceSpec(unit_count=1, objective_count=1)
        state = UnitControlState(slot=0)
        sensor = SharedSensorState.initial(SharedSensorConfig(capacity=0))
        mask = build_joint_action_mask(
            space,
            (state,),
            objective_valid=(True,),
            sensor_state=sensor,
            step=0,
        )
        action = JointAction(
            units=(UnitAction.noop(0),),
            shared_sensor=SharedSensorAction(),
        )
        buffer = JointTrajectoryBuffer(space, observation_dim=2)
        buffer.append(
            JointTransition(
                observations=(np.asarray((1.0, 2.0), dtype=np.float32),),
                states=(state,),
                mask=mask,
                action=action,
                trace=JointPolicyTrace(
                    log_prob_by_term={"unit/0/activation": -0.25},
                    values_by_unit=(0.5,),
                    team_value=0.75,
                ),
                rewards=(0.2,),
                team_reward=0.3,
                next_observations=(np.asarray((2.0, 3.0), dtype=np.float32),),
                terminated=(False,),
                truncated=(True,),
                team_truncated=True,
                sensor_reward=0.4,
            )
        )
        return buffer

    def test_finalization_and_merge_share_only_immutable_snapshots(self) -> None:
        source = self._buffer()
        original = source.items[0]

        finalized = source.with_episode_plan_rewards((0.8,))
        replacement = finalized.items[0]
        self.assertIsNot(replacement, original)
        self.assertIs(replacement.observations[0], original.observations[0])
        self.assertIs(replacement.mask, original.mask)
        self.assertIs(replacement.trace, original.trace)
        self.assertEqual(replacement.plan_rewards, (0.8,))
        self.assertEqual(replacement.motion_rewards, (0.2,))
        self.assertEqual(replacement.sensor_reward, 0.4)

        merged = JointTrajectoryBuffer(source.space, source.observation_dim)
        merged.extend_snapshots(finalized)
        self.assertIs(merged.items[0], replacement)
        source.clear()
        finalized.clear()
        self.assertEqual(len(merged), 1)
        with self.assertRaises(ValueError):
            merged.items[0].observations[0].setflags(write=True)

    def test_collector_append_reuses_frozen_records_but_copies_observations(self) -> None:
        original_buffer = self._buffer()
        input_transition = original_buffer.items[0]
        collected = JointTrajectoryBuffer(
            original_buffer.space,
            original_buffer.observation_dim,
        )

        collected.append_collected(input_transition)

        snapshot = collected.items[0]
        self.assertIs(snapshot.mask, input_transition.mask)
        self.assertIs(snapshot.trace, input_transition.trace)
        self.assertIsNot(snapshot.observations[0], input_transition.observations[0])
        np.testing.assert_array_equal(
            snapshot.observations[0],
            input_transition.observations[0],
        )
        with self.assertRaises(ValueError):
            snapshot.observations[0].setflags(write=True)


class ActionMaskCacheTests(unittest.TestCase):
    def test_same_step_reuses_the_exact_immutable_mask(self) -> None:
        space = JointSpaceSpec(unit_count=1, objective_count=1)
        states = (UnitControlState(slot=0),)
        sensor = SharedSensorState.initial(SharedSensorConfig(capacity=0))
        base = build_joint_action_mask(
            space,
            states,
            objective_valid=(True,),
            sensor_state=sensor,
            step=0,
        )
        environment = object.__new__(JointGameEnv)
        environment.space = space
        environment.tracker = SimpleNamespace(states=states, sensor_state=sensor)
        environment.current_step = 0
        environment.unit_ids = (10,)
        environment.sensor_backend_capacity_per_unit = 0
        environment.engine = SimpleNamespace(
            get_simulator_by_id=lambda _entity_id: None
        )
        environment._objective_validity = lambda: (True,)
        environment._cached_action_mask_step = -1
        environment._cached_action_mask = None

        with patch(
            "personal_train.joint_game_env.build_joint_action_mask",
            return_value=base,
        ) as builder:
            first = environment.action_mask()
            second = environment.action_mask()

        self.assertIs(second, first)
        self.assertEqual(builder.call_count, 1)


if __name__ == "__main__":
    unittest.main()
