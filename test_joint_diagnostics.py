"""Focused tests for compact joint-policy action diagnostics."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import numpy as np

from personal_train.joint_diagnostics import EpisodeActionDiagnostics
from personal_train.joint_rl_core import (
    BinaryChoice,
    JointAction,
    JointPolicyTrace,
    JointSpaceSpec,
    Movement,
    SharedSensorAction,
    SharedSensorConfig,
    SharedSensorPolicyTrace,
    SharedSensorState,
    UnitAction,
    UnitControlState,
    UnitPhase,
    build_joint_action_mask,
    expected_log_prob_terms,
)


def _trace(states, action, mask) -> JointPolicyTrace:
    sensor = None
    sensor_log_prob = 0.0
    if mask.shared_sensor_max_requests > 0:
        available = np.asarray(mask.shared_sensor_eligible, dtype=np.bool_).copy()
        tokens: list[int] = []
        masks: list[np.ndarray] = []
        log_probs: list[float] = []
        for slot in action.shared_sensor.requester_slots:
            masks.append(np.concatenate((np.asarray((True,), dtype=np.bool_), available)))
            tokens.append(int(slot) + 1)
            log_probs.append(-0.2)
            available[int(slot)] = False
        if len(tokens) < mask.shared_sensor_max_requests:
            masks.append(np.concatenate((np.asarray((True,), dtype=np.bool_), available)))
            tokens.append(0)
            log_probs.append(-0.1)
        sensor = SharedSensorPolicyTrace(
            tokens=tuple(tokens), masks=tuple(masks), log_probs=tuple(log_probs)
        )
        sensor_log_prob = sensor.total_log_prob

    names = expected_log_prob_terms(
        states,
        action,
        shared_sensor_active=mask.shared_sensor_max_requests > 0,
    )
    log_probs_by_term = {name: -0.3 for name in names}
    if sensor is not None:
        log_probs_by_term["shared_sensor"] = sensor_log_prob
    return JointPolicyTrace(
        log_prob_by_term=log_probs_by_term,
        values_by_unit=tuple(0.0 for _ in states),
        team_value=0.0,
        shared_sensor=sensor,
    )


class EpisodeActionDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.space = JointSpaceSpec(unit_count=4, objective_count=3)
        self.diagnostics = EpisodeActionDiagnostics(
            self.space,
            objective_ids=(51, 52, 54),
            unit_ids=(100, 101, 102, 103),
            unit_types=(21000, 21002, 21000, 21001),
        )

    def test_conditional_actions_receipts_and_json_output(self) -> None:
        first_states = (
            UnitControlState(slot=0, phase=UnitPhase.STAGED),
            UnitControlState(
                slot=1,
                phase=UnitPhase.ACTIVE,
                current_objective_slot=0,
                last_movement=Movement.NEGATIVE,
            ),
            UnitControlState(slot=2, phase=UnitPhase.PENDING),
            UnitControlState(slot=3, phase=UnitPhase.TERMINAL),
        )
        first_mask = build_joint_action_mask(
            self.space,
            first_states,
            objective_valid=(True, True, True),
            sensor_state=SharedSensorState.initial(
                SharedSensorConfig(capacity=2, max_requests_per_step=1)
            ),
            step=0,
        )
        first_action = JointAction.from_sequence(
            (
                UnitAction(
                    slot=0,
                    activate=BinaryChoice.YES,
                    placement=(0.99, -0.5),
                    objective_slot=2,
                    movement=Movement.POSITIVE,
                ),
                UnitAction(
                    slot=1,
                    retarget=BinaryChoice.YES,
                    objective_slot=1,
                    movement=Movement.NEGATIVE,
                ),
                UnitAction.noop(2),
                UnitAction.noop(3),
            ),
            SharedSensorAction.from_sequence((1,)),
        )
        self.diagnostics.observe(
            first_states,
            first_mask,
            first_action,
            _trace(first_states, first_action, first_mask),
            SimpleNamespace(
                accepted_activations=(0,), accepted_sensor_requests=(1,)
            ),
            step=0,
        )

        second_states = (
            UnitControlState(
                slot=0,
                phase=UnitPhase.ACTIVE,
                current_objective_slot=2,
                last_movement=Movement.POSITIVE,
            ),
            UnitControlState(
                slot=1,
                phase=UnitPhase.ACTIVE,
                current_objective_slot=1,
                last_movement=Movement.NEGATIVE,
            ),
            UnitControlState(slot=2, phase=UnitPhase.PENDING),
            UnitControlState(slot=3, phase=UnitPhase.TERMINAL),
        )
        second_mask = build_joint_action_mask(
            self.space,
            second_states,
            objective_valid=(True, True, True),
            sensor_state=SharedSensorState.initial(
                SharedSensorConfig(capacity=2, max_requests_per_step=1)
            ),
            step=1,
        )
        second_action = JointAction.from_sequence(
            (
                UnitAction(slot=0, movement=Movement.POSITIVE),
                UnitAction(slot=1, movement=Movement.NEUTRAL),
                UnitAction.noop(2),
                UnitAction.noop(3),
            )
        )
        self.diagnostics.observe(
            second_states,
            second_mask,
            second_action,
            _trace(second_states, second_action, second_mask),
            SimpleNamespace(accepted_activations=(), accepted_sensor_requests=()),
            step=1,
        )

        result = self.diagnostics.finalize()
        self.assertEqual(result["steps_observed"], 2)
        self.assertEqual(
            result["activation"],
            {
                **result["activation"],
                "decisions": 1,
                "wait": 0,
                "requested": 1,
                "accepted": 1,
                "rejected": 0,
            },
        )
        self.assertEqual(result["target"]["activation_selected_by_slot"], [0, 0, 1])
        self.assertEqual(result["target"]["retarget_selected_by_slot"], [0, 1, 0])
        self.assertEqual(result["target"]["effective_by_slot"], [0, 2, 2])
        self.assertEqual(result["movement"]["selected"], [1, 1, 2])
        self.assertEqual(result["movement"]["on_activation"], [0, 0, 1])
        self.assertEqual(result["movement"]["while_active"], [1, 1, 1])
        self.assertEqual(result["movement"]["switches"], 1)
        self.assertEqual(result["movement"]["switch_opportunities"], 3)
        self.assertEqual(result["sensor"]["head_active_steps"], 2)
        self.assertEqual(result["sensor"]["stop_selections"], 1)
        self.assertEqual(result["sensor"]["requested"], 1)
        self.assertEqual(result["sensor"]["accepted"], 1)
        self.assertEqual(result["sensor"]["by_unit_type"]["21002"]["accepted"], 1)
        self.assertEqual(result["placement"]["requested"]["count"], 1)
        self.assertEqual(result["placement"]["requested"]["any_edge_fraction"], 1.0)
        self.assertEqual(len(result["placement"]["accepted_records"]), 1)
        self.assertEqual(
            result["placement"]["accepted_records"][0]["objective_id"], 54
        )
        # Strict encoding proves all NumPy values and enums were normalized.
        json.dumps(result, allow_nan=False)

    def test_wait_rejection_and_inactive_sentinels_are_counted_correctly(self) -> None:
        states = tuple(
            UnitControlState(slot=slot, phase=phase)
            for slot, phase in enumerate(
                (UnitPhase.STAGED, UnitPhase.STAGED, UnitPhase.PENDING, UnitPhase.TERMINAL)
            )
        )
        mask = build_joint_action_mask(
            self.space,
            states,
            objective_valid=(True, True, True),
            sensor_state=SharedSensorState.initial(SharedSensorConfig(capacity=0)),
            step=0,
        )
        action = JointAction.from_sequence(
            (
                UnitAction(
                    slot=0,
                    activate=BinaryChoice.YES,
                    placement=(-0.25, 0.25),
                    objective_slot=0,
                    movement=Movement.NEGATIVE,
                ),
                UnitAction.noop(1),
                UnitAction.noop(2),
                UnitAction.noop(3),
            )
        )
        self.diagnostics.observe(
            states,
            mask,
            action,
            _trace(states, action, mask),
            SimpleNamespace(accepted_activations=(), accepted_sensor_requests=()),
            step=0,
        )
        result = self.diagnostics.finalize()
        self.assertEqual(result["activation"]["decisions"], 2)
        self.assertEqual(result["activation"]["wait"], 1)
        self.assertEqual(result["activation"]["requested"], 1)
        self.assertEqual(result["activation"]["rejected"], 1)
        self.assertEqual(result["movement"]["selected"], [1, 0, 0])
        self.assertEqual(result["placement"]["requested"]["count"], 1)
        self.assertEqual(result["placement"]["accepted"]["count"], 0)
        self.assertEqual(result["placement"]["accepted_records"], [])
        self.assertEqual(result["sensor"]["head_active_steps"], 0)
        self.assertEqual(result["sensor"]["stop_selections"], 0)

    def test_multi_request_stop_and_receipt_validation(self) -> None:
        space = JointSpaceSpec(unit_count=3, objective_count=1)
        diagnostics = EpisodeActionDiagnostics(
            space,
            objective_ids=(51,),
            unit_ids=(10, 11, 12),
            unit_types=(21000, 21000, 21002),
        )
        states = tuple(
            UnitControlState(
                slot=slot,
                phase=UnitPhase.ACTIVE,
                current_objective_slot=0,
            )
            for slot in range(3)
        )
        mask = build_joint_action_mask(
            space,
            states,
            objective_valid=(True,),
            sensor_state=SharedSensorState.initial(
                SharedSensorConfig(capacity=4, max_requests_per_step=2)
            ),
            step=0,
        )
        action = JointAction.from_sequence(
            tuple(UnitAction(slot=slot) for slot in range(3)),
            SharedSensorAction.from_sequence((2,)),
        )
        diagnostics.observe(
            states,
            mask,
            action,
            _trace(states, action, mask),
            SimpleNamespace(accepted_activations=(), accepted_sensor_requests=()),
            step=0,
        )
        result = diagnostics.finalize()
        self.assertEqual(result["sensor"]["stop_selections"], 1)
        self.assertEqual(result["sensor"]["requested"], 1)
        self.assertEqual(result["sensor"]["rejected"], 1)
        self.assertEqual(result["sensor"]["by_unit_type"]["21002"]["rejected"], 1)

        bad = EpisodeActionDiagnostics(
            space,
            objective_ids=(51,),
            unit_ids=(10, 11, 12),
            unit_types=(21000, 21000, 21002),
        )
        with self.assertRaisesRegex(ValueError, "not requested"):
            bad.observe(
                states,
                mask,
                action,
                _trace(states, action, mask),
                SimpleNamespace(
                    accepted_activations=(), accepted_sensor_requests=(1,)
                ),
                step=0,
            )

    def test_contract_lengths_and_lifecycle_are_guarded(self) -> None:
        with self.assertRaisesRegex(ValueError, "objective_ids"):
            EpisodeActionDiagnostics(
                self.space,
                objective_ids=(51,),
                unit_ids=(100, 101, 102, 103),
                unit_types=(21000, 21002, 21000, 21001),
            )
        result = self.diagnostics.finalize()
        self.assertEqual(result["target"]["dominant_slot"], -1)
        self.assertEqual(result["movement"]["fractions"], [0.0, 0.0, 0.0])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
