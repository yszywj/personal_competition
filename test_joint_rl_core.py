"""Tests for the simulator-neutral joint RL core.

These tests use only in-memory game entities.  They do not import or start the
competition simulator and do not require Torch or Gymnasium.
"""

from __future__ import annotations

import unittest

import numpy as np

from personal_train.joint_rl_core import (
    ActionValidationError,
    ActivationReceipt,
    BinaryChoice,
    JointAction,
    JointControlTracker,
    JointObservationEncoder,
    JointPolicyTrace,
    JointSpaceSpec,
    JointTrajectoryBuffer,
    JointTransition,
    MapBounds,
    Movement,
    ObjectiveFrame,
    ObservationEncoderConfig,
    SharedSensorAction,
    SharedSensorConfig,
    SharedSensorPolicyTrace,
    SharedSensorReceipt,
    StableSlotRegistry,
    UnitAction,
    UnitFrame,
    UnitPhase,
    build_joint_action_mask,
    expected_log_prob_terms,
    validate_joint_action,
)


def _joint_action(*units: UnitAction, sensor_slots=()) -> JointAction:
    return JointAction.from_sequence(
        units,
        SharedSensorAction.from_sequence(sensor_slots),
    )


class LifecycleAndMaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.space = JointSpaceSpec(unit_count=3, objective_count=3)
        self.tracker = JointControlTracker(
            self.space,
            sensor_config=SharedSensorConfig(
                capacity=2,
                max_requests_per_step=1,
                cooldown_steps=1,
            ),
        )
        self.objectives = (True, False, True)

    def test_activation_is_one_conditional_action_and_wait_is_trainable(self):
        states = self.tracker.states
        mask = build_joint_action_mask(
            self.space,
            states,
            self.objectives,
            self.tracker.sensor_state,
            step=0,
        )
        self.assertEqual(mask.by_unit[0].activation.tolist(), [True, True])
        self.assertEqual(mask.by_unit[0].movement.tolist(), [False, True, False])
        self.assertFalse(mask.shared_sensor_eligible.any())

        action = _joint_action(
            UnitAction(
                slot=0,
                activate=BinaryChoice.YES,
                placement=(-0.25, 0.75),
                objective_slot=2,
            ),
            UnitAction.noop(1),
            UnitAction.noop(2),
        )
        terms = expected_log_prob_terms(states, action)
        self.assertEqual(
            terms,
            {
                "unit/0/activation",
                "unit/0/placement",
                "unit/0/objective",
                "unit/1/activation",
                "unit/2/activation",
            },
        )

        intents = self.tracker.apply(action, self.objectives, step=0)
        self.assertEqual(len(intents.activations), 1)
        self.assertEqual(intents.activations[0].placement, (-0.25, 0.75))
        self.assertEqual(self.tracker.states[0].phase, UnitPhase.PENDING)

        # A pending entity has no policy branches until execution is confirmed.
        pending_mask = build_joint_action_mask(
            self.space,
            self.tracker.states,
            self.objectives,
            self.tracker.sensor_state,
            step=0,
        )
        self.assertEqual(pending_mask.by_unit[0].activation.tolist(), [True, False])
        self.assertEqual(pending_mask.by_unit[0].movement.tolist(), [False, True, False])

        self.tracker.confirm_activations(
            (ActivationReceipt(slot=0, request_step=0, accepted=True),),
            step=0,
        )
        state = self.tracker.states[0]
        self.assertEqual(state.phase, UnitPhase.ACTIVE)
        self.assertEqual(state.current_objective_slot, 2)
        self.assertEqual(state.activated_step, 0)

    def test_rejected_activation_returns_to_staged(self):
        action = _joint_action(
            UnitAction(slot=0, activate=BinaryChoice.YES, objective_slot=0),
            UnitAction.noop(1),
            UnitAction.noop(2),
        )
        self.tracker.apply(action, self.objectives, step=2)
        self.tracker.confirm_activations(
            (ActivationReceipt(slot=0, request_step=2, accepted=False),),
            step=2,
        )
        self.assertEqual(self.tracker.states[0].phase, UnitPhase.STAGED)

    def test_no_objective_masks_activation_and_late_receipt_cannot_confirm_retry(self):
        no_objectives = (False, False, False)
        mask = build_joint_action_mask(
            self.space,
            self.tracker.states,
            no_objectives,
            self.tracker.sensor_state,
            step=0,
        )
        self.assertEqual(mask.by_unit[0].activation.tolist(), [True, False])
        invalid = _joint_action(
            UnitAction(slot=0, activate=BinaryChoice.YES, objective_slot=0),
            UnitAction.noop(1),
            UnitAction.noop(2),
        )
        with self.assertRaises(ActionValidationError):
            self.tracker.apply(invalid, no_objectives, step=0)

        first = _joint_action(
            UnitAction(slot=0, activate=BinaryChoice.YES, objective_slot=0),
            UnitAction.noop(1),
            UnitAction.noop(2),
        )
        self.tracker.apply(first, self.objectives, step=0)
        self.tracker.confirm_activations(
            (ActivationReceipt(slot=0, request_step=0, accepted=False),),
            step=0,
        )
        self.tracker.apply(first, self.objectives, step=1)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.tracker.confirm_activations(
                (ActivationReceipt(slot=0, request_step=0, accepted=True),),
                step=1,
            )
        self.assertEqual(self.tracker.states[0].activation_requested_step, 1)
        self.assertEqual(
            self.tracker.expire_pending_activations(step=3, timeout_steps=2),
            (0,),
        )
        self.assertEqual(self.tracker.states[0].phase, UnitPhase.STAGED)

    def test_active_entity_can_move_reassign_and_request_shared_sensor(self):
        activation = _joint_action(
            UnitAction(slot=0, activate=BinaryChoice.YES, objective_slot=0),
            UnitAction.noop(1),
            UnitAction.noop(2),
        )
        self.tracker.apply(activation, self.objectives, step=0)
        self.tracker.confirm_activations(
            (ActivationReceipt(slot=0, request_step=0, accepted=True),),
            step=0,
        )
        before = self.tracker.states
        action = _joint_action(
            UnitAction(
                slot=0,
                retarget=BinaryChoice.YES,
                objective_slot=2,
                movement=Movement.POSITIVE,
            ),
            UnitAction.noop(1),
            UnitAction.noop(2),
            sensor_slots=(0,),
        )
        mask = validate_joint_action(
            self.space,
            before,
            self.objectives,
            self.tracker.sensor_state,
            action,
            step=1,
        )
        terms = expected_log_prob_terms(
            before,
            action,
            shared_sensor_active=bool(mask.shared_sensor_eligible.any()),
        )
        self.assertIn("shared_sensor", terms)
        self.assertIn("unit/0/retarget", terms)
        self.assertIn("unit/0/movement", terms)
        self.assertIn("unit/0/objective", terms)

        intents = self.tracker.apply(action, self.objectives, step=1)
        self.assertEqual(intents.retargets[0].objective_slot, 2)
        self.assertEqual(intents.movements[0].movement, Movement.POSITIVE)
        self.assertEqual(intents.shared_sensor[0].requester_slot, 0)
        self.assertEqual(self.tracker.sensor_state.remaining, 2)
        self.assertEqual(len(self.tracker.sensor_state.pending), 1)
        self.tracker.confirm_shared_sensor(
            (
                SharedSensorReceipt(
                    requester_slot=0,
                    request_step=1,
                    accepted=True,
                ),
            ),
            step=1,
        )
        self.assertEqual(self.tracker.sensor_state.remaining, 1)

        # One full cooldown step must elapse after a successful request.
        cooldown_mask = build_joint_action_mask(
            self.space,
            self.tracker.states,
            self.objectives,
            self.tracker.sensor_state,
            step=2,
        )
        ready_mask = build_joint_action_mask(
            self.space,
            self.tracker.states,
            self.objectives,
            self.tracker.sensor_state,
            step=3,
        )
        self.assertFalse(cooldown_mask.shared_sensor_eligible[0])
        self.assertTrue(ready_mask.shared_sensor_eligible[0])

    def test_invalid_objective_duplicate_requests_and_unconfirmed_control_fail(self):
        invalid_objective = _joint_action(
            UnitAction(slot=0, activate=BinaryChoice.YES, objective_slot=1),
            UnitAction.noop(1),
            UnitAction.noop(2),
        )
        with self.assertRaises(ActionValidationError):
            self.tracker.apply(invalid_objective, self.objectives, step=0)

        valid = _joint_action(
            UnitAction(slot=0, activate=BinaryChoice.YES, objective_slot=0),
            UnitAction.noop(1),
            UnitAction.noop(2),
        )
        self.tracker.apply(valid, self.objectives, step=0)
        pending_control = _joint_action(
            UnitAction(slot=0, movement=Movement.POSITIVE),
            UnitAction.noop(1),
            UnitAction.noop(2),
        )
        with self.assertRaises(ActionValidationError):
            self.tracker.apply(pending_control, self.objectives, step=1)

        self.tracker.confirm_activations(
            (ActivationReceipt(slot=0, request_step=0, accepted=True),),
            step=1,
        )
        duplicate_sensor = _joint_action(
            UnitAction.noop(0),
            UnitAction.noop(1),
            UnitAction.noop(2),
            sensor_slots=(0, 0),
        )
        with self.assertRaises(ActionValidationError):
            self.tracker.apply(duplicate_sensor, self.objectives, step=2)

    def test_duplicate_step_fails_and_reset_clears_history_and_budget(self):
        wait = _joint_action(
            UnitAction.noop(0),
            UnitAction.noop(1),
            UnitAction.noop(2),
        )
        self.tracker.apply(wait, self.objectives, step=0)
        with self.assertRaises(ValueError):
            self.tracker.apply(wait, self.objectives, step=0)
        self.tracker.reset()
        self.tracker.apply(wait, self.objectives, step=0)
        self.assertEqual(self.tracker.sensor_state.remaining, 2)
        self.assertTrue(all(state.phase == UnitPhase.STAGED for state in self.tracker.states))

    def test_rejected_shared_sensor_request_does_not_consume_budget_or_cooldown(self):
        activation = _joint_action(
            UnitAction(slot=0, activate=BinaryChoice.YES, objective_slot=0),
            UnitAction.noop(1),
            UnitAction.noop(2),
        )
        self.tracker.apply(activation, self.objectives, step=0)
        self.tracker.confirm_activations(
            (ActivationReceipt(slot=0, request_step=0, accepted=True),),
            step=0,
        )
        request = _joint_action(
            UnitAction.noop(0),
            UnitAction.noop(1),
            UnitAction.noop(2),
            sensor_slots=(0,),
        )
        self.tracker.apply(request, self.objectives, step=1)
        self.assertEqual(self.tracker.sensor_state.remaining, 2)
        self.assertEqual(self.tracker.sensor_state.available, 1)
        self.assertFalse(self.tracker.sensor_state.is_ready(2))
        self.tracker.confirm_shared_sensor(
            (
                SharedSensorReceipt(
                    requester_slot=0,
                    request_step=1,
                    accepted=False,
                ),
            ),
            step=2,
        )
        self.assertEqual(self.tracker.sensor_state.remaining, 2)
        self.assertEqual(self.tracker.sensor_state.available, 2)
        self.assertTrue(self.tracker.sensor_state.is_ready(2))

    def test_terminal_requester_keeps_resource_reserved_until_receipt(self):
        activation = _joint_action(
            UnitAction(slot=0, activate=BinaryChoice.YES, objective_slot=0),
            UnitAction.noop(1),
            UnitAction.noop(2),
        )
        self.tracker.apply(activation, self.objectives, step=0)
        self.tracker.confirm_activations(
            (ActivationReceipt(slot=0, request_step=0, accepted=True),),
            step=0,
        )
        request = _joint_action(
            UnitAction.noop(0),
            UnitAction.noop(1),
            UnitAction.noop(2),
            sensor_slots=(0,),
        )
        self.tracker.apply(request, self.objectives, step=1)
        self.tracker.mark_terminal((0,), step=2)
        self.assertEqual(self.tracker.states[0].phase, UnitPhase.TERMINAL)
        self.assertEqual(len(self.tracker.sensor_state.pending), 1)
        self.tracker.confirm_shared_sensor(
            (
                SharedSensorReceipt(
                    requester_slot=0,
                    request_step=1,
                    accepted=True,
                ),
            ),
            step=3,
        )
        self.assertEqual(self.tracker.sensor_state.remaining, 1)
        self.assertEqual(self.tracker.sensor_state.pending, ())

    def test_receipts_and_terminal_events_advance_one_monotonic_timeline(self):
        activation = _joint_action(
            UnitAction(slot=0, activate=BinaryChoice.YES, objective_slot=0),
            UnitAction.noop(1),
            UnitAction.noop(2),
        )
        self.tracker.apply(activation, self.objectives, step=0)
        self.tracker.confirm_activations(
            (ActivationReceipt(slot=0, request_step=0, accepted=True),),
            step=10,
        )
        self.assertEqual(self.tracker.latest_event_step, 10)
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            self.tracker.apply(
                _joint_action(
                    UnitAction.noop(0),
                    UnitAction.noop(1),
                    UnitAction.noop(2),
                ),
                self.objectives,
                step=1,
            )
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            self.tracker.mark_terminal((0,), step=9)


class ObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.space = JointSpaceSpec(unit_count=1, objective_count=3)
        self.encoder = JointObservationEncoder(
            ObservationEncoderConfig(
                max_steps=100,
                space=self.space,
                bounds=MapBounds(-10.0, 10.0, -20.0, 20.0, 0.0, 100.0),
                max_speed=10.0,
                max_track_age_steps=20,
                unit_type_count=2,
                objective_type_count=3,
            )
        )
        self.tracker = JointControlTracker(
            self.space,
            sensor_config=SharedSensorConfig(capacity=4),
        )

    def test_fixed_width_encoding_has_validity_bits_and_no_external_ids(self):
        unit = UnitFrame(
            slot=0,
            type_index=1,
            position=(0.0, 0.0, 50.0),
            velocity_xy=(5.0, -5.0),
            velocity_known=True,
            health_fraction=0.8,
        )
        objectives = (
            ObjectiveFrame(
                slot=0,
                valid=True,
                known=True,
                position=(10.0, 0.0),
                velocity_xy=(2.0, 0.0),
                velocity_known=True,
                age_steps=5,
                type_index=2,
            ),
            ObjectiveFrame(slot=1, valid=True, known=False),
        )
        encoded = self.encoder.encode(
            unit,
            self.tracker.states[0],
            objectives,
            self.tracker.sensor_state,
            step=25,
        )
        values = dict(zip(self.encoder.feature_names, encoded.tolist()))
        self.assertEqual(encoded.shape, (self.encoder.dimension,))
        self.assertTrue(np.isfinite(encoded).all())
        self.assertAlmostEqual(values["time"], 0.25)
        self.assertAlmostEqual(values["sensor_available"], 1.0)
        self.assertAlmostEqual(values["sensor_pending"], 0.0)
        self.assertEqual(values["phase_staged"], 1.0)
        self.assertEqual(values["self_position_known"], 1.0)
        self.assertEqual(values["self_velocity_known"], 1.0)
        self.assertEqual(values["objective_0_known"], 1.0)
        self.assertEqual(values["objective_0_velocity_known"], 1.0)
        self.assertEqual(values["objective_1_valid"], 1.0)
        self.assertEqual(values["objective_1_known"], 0.0)
        self.assertEqual(values["objective_1_rel_x"], 0.0)
        self.assertFalse(any("entity_id" in name for name in self.encoder.feature_names))

    def test_non_finite_input_is_neutralized_and_shape_remains_stable(self):
        unit = UnitFrame(
            slot=0,
            type_index=None,
            position=(float("nan"), 0.0, 0.0),
            velocity_xy=(float("inf"), 0.0),
            velocity_known=True,
        )
        encoded = self.encoder.encode(
            unit,
            self.tracker.states[0],
            (),
            self.tracker.sensor_state,
            step=0,
        )
        self.assertEqual(encoded.shape, (self.encoder.dimension,))
        self.assertTrue(np.isfinite(encoded).all())
        values = dict(zip(self.encoder.feature_names, encoded.tolist()))
        self.assertEqual(values["self_position_known"], 0.0)
        self.assertEqual(values["self_velocity_known"], 0.0)

    def test_space_and_integer_configuration_are_validated(self):
        with self.assertRaises(ValueError):
            JointSpaceSpec(unit_count=1, objective_count=2.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ObservationEncoderConfig(
                max_steps=2.5,  # type: ignore[arg-type]
                space=self.space,
                bounds=MapBounds(0.0, 1.0, 0.0, 1.0),
                max_speed=1.0,
                max_track_age_steps=1,
                unit_type_count=1,
                objective_type_count=1,
            )


class SlotRegistryTests(unittest.TestCase):
    def test_batch_order_does_not_change_slots_and_overflow_is_counted_once(self):
        first = StableSlotRegistry(capacity=3)
        second = StableSlotRegistry(capacity=3)
        first.register_batch(("z", "a", "m"))
        second.register_batch(("m", "z", "a"))
        self.assertEqual(first.external_by_slot, second.external_by_slot)
        self.assertEqual(first.external_by_slot, ("a", "m", "z"))

        first.register_batch(("overflow", "overflow"))
        first.register_batch(("overflow",))
        self.assertEqual(first.overflow_count, 1)
        self.assertEqual(first.slot_for("overflow"), None)
        self.assertEqual(first.valid_mask.tolist(), [True, True, True])

    def test_slots_are_opaque_and_reset_is_the_only_reuse_boundary(self):
        registry = StableSlotRegistry(capacity=2)
        registry.register_batch((101, 202))
        self.assertEqual(registry.slot_for(101), 0)
        self.assertEqual(registry.external_id_for(1), 202)
        registry.register_batch((303,))
        self.assertEqual(registry.slot_for(303), None)
        registry.reset()
        registry.register_batch((303,))
        self.assertEqual(registry.slot_for(303), 0)

    def test_bool_and_mutable_ids_are_rejected(self):
        registry = StableSlotRegistry(capacity=1)
        for invalid in (True, []):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                registry.register_batch((invalid,))


class TrajectoryTests(unittest.TestCase):
    def test_buffer_requires_exact_conditional_terms_and_copies_arrays(self):
        space = JointSpaceSpec(unit_count=1, objective_count=1)
        tracker = JointControlTracker(
            space,
            sensor_config=SharedSensorConfig(capacity=0),
        )
        action = _joint_action(UnitAction.noop(0))
        mask = build_joint_action_mask(
            space,
            tracker.states,
            (True,),
            tracker.sensor_state,
            step=0,
        )
        observation = np.asarray((1.0, 2.0), dtype=np.float32)
        transition = JointTransition(
            observations=(observation,),
            states=tracker.states,
            mask=mask,
            action=action,
            trace=JointPolicyTrace(
                log_prob_by_term={"unit/0/activation": -0.2},
                values_by_unit=(0.5,),
                team_value=0.25,
            ),
            rewards=(1.0,),
            team_reward=1.0,
            next_observations=(observation + 1.0,),
            terminated=(False,),
            truncated=(False,),
        )
        buffer = JointTrajectoryBuffer(space=space, observation_dim=2)
        buffer.append(transition)
        observation[0] = 99.0
        self.assertEqual(float(buffer.items[0].observations[0][0]), 1.0)
        self.assertAlmostEqual(buffer.items[0].trace.total_log_prob, -0.2)
        self.assertIsNot(buffer.items[0].mask, mask)
        self.assertIsNot(buffer.items[0].trace, transition.trace)
        with self.assertRaises(ValueError):
            buffer.items[0].observations[0].setflags(write=True)
        with self.assertRaises(ValueError):
            buffer.items[0].mask.by_unit[0].activation.setflags(write=True)

        invalid = JointTransition(
            observations=(np.zeros(2, dtype=np.float32),),
            states=tracker.states,
            mask=mask,
            action=action,
            trace=JointPolicyTrace(
                log_prob_by_term={},
                values_by_unit=(0.0,),
                team_value=0.0,
            ),
            rewards=(0.0,),
            team_reward=0.0,
            next_observations=(np.zeros(2, dtype=np.float32),),
            terminated=(False,),
            truncated=(False,),
        )
        with self.assertRaises(ValueError):
            buffer.append(invalid)

    def test_buffer_rejects_ambiguous_unit_order(self):
        space = JointSpaceSpec(unit_count=2, objective_count=1)
        tracker = JointControlTracker(
            space,
            sensor_config=SharedSensorConfig(capacity=0),
        )
        mask = build_joint_action_mask(
            space,
            tracker.states,
            (True,),
            tracker.sensor_state,
            step=0,
        )
        reversed_action = _joint_action(UnitAction.noop(1), UnitAction.noop(0))
        transition = JointTransition(
            observations=(np.zeros(2), np.ones(2)),
            states=tuple(reversed(tracker.states)),
            mask=mask,
            action=reversed_action,
            trace=JointPolicyTrace(
                log_prob_by_term={
                    "unit/0/activation": 0.0,
                    "unit/1/activation": 0.0,
                },
                values_by_unit=(0.0, 0.0),
                team_value=0.0,
            ),
            rewards=(0.0, 0.0),
            team_reward=0.0,
            next_observations=(np.zeros(2), np.ones(2)),
            terminated=(False, False),
            truncated=(False, False),
        )
        with self.assertRaisesRegex(ValueError, "ordered"):
            JointTrajectoryBuffer(space, observation_dim=2).append(transition)

    def test_multi_request_trace_retains_each_without_replacement_mask(self):
        space = JointSpaceSpec(unit_count=2, objective_count=1)
        tracker = JointControlTracker(
            space,
            SharedSensorConfig(capacity=2, max_requests_per_step=2),
        )
        activate = _joint_action(
            UnitAction(slot=0, activate=BinaryChoice.YES, objective_slot=0),
            UnitAction(slot=1, activate=BinaryChoice.YES, objective_slot=0),
        )
        tracker.apply(activate, (True,), step=0)
        tracker.confirm_activations(
            (
                ActivationReceipt(slot=0, request_step=0, accepted=True),
                ActivationReceipt(slot=1, request_step=0, accepted=True),
            ),
            step=0,
        )
        action = _joint_action(
            UnitAction.noop(0),
            UnitAction.noop(1),
            sensor_slots=(1,),
        )
        mask = build_joint_action_mask(
            space,
            tracker.states,
            (True,),
            tracker.sensor_state,
            step=1,
        )
        sensor_trace = SharedSensorPolicyTrace(
            tokens=(2, 0),
            masks=(
                np.asarray((True, True, True)),
                np.asarray((True, True, False)),
            ),
            log_probs=(-1.0, -0.5),
        )
        transition = JointTransition(
            observations=(np.zeros(2), np.ones(2)),
            states=tracker.states,
            mask=mask,
            action=action,
            trace=JointPolicyTrace(
                log_prob_by_term={
                    "unit/0/retarget": 0.0,
                    "unit/0/movement": -0.1,
                    "unit/1/retarget": 0.0,
                    "unit/1/movement": -0.1,
                    "shared_sensor": -1.5,
                },
                values_by_unit=(0.2, 0.3),
                team_value=0.4,
                shared_sensor=sensor_trace,
            ),
            rewards=(0.0, 0.0),
            team_reward=0.0,
            next_observations=(np.ones(2), np.ones(2)),
            terminated=(False, False),
            truncated=(False, False),
        )
        buffer = JointTrajectoryBuffer(space, observation_dim=2)
        buffer.append(transition)
        self.assertEqual(buffer.items[0].trace.shared_sensor.tokens, (2, 0))
        self.assertFalse(buffer.items[0].trace.shared_sensor.masks[1][2])

        bad_action = JointAction(
            units=[UnitAction.noop(0), UnitAction.noop(0)],  # type: ignore[arg-type]
        )
        bad_transition = JointTransition(
            observations=(np.zeros(2), np.zeros(2)),
            states=tracker.states,
            mask=mask,
            action=bad_action,
            trace=transition.trace,
            rewards=(0.0, 0.0),
            team_reward=0.0,
            next_observations=(np.zeros(2), np.zeros(2)),
            terminated=(False, False),
            truncated=(False, False),
        )
        with self.assertRaisesRegex(ValueError, "configured unit slots"):
            buffer.append(bad_transition)


if __name__ == "__main__":
    unittest.main()
