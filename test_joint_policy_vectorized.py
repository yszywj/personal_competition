"""CPU equivalence checks for the tensorized joint-policy hot paths."""

from __future__ import annotations

import math
import os
import time
import unittest

import numpy as np
import torch

from personal_train.joint_policy import JointPPOConfig, JointPPOPolicy
from personal_train.joint_rl_core import (
    JointSpaceSpec,
    JointTrajectoryBuffer,
    JointTransition,
    SharedSensorConfig,
    SharedSensorState,
    UnitControlState,
    UnitPhase,
    build_joint_action_mask,
)


class VectorizedJointPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.space = JointSpaceSpec(unit_count=8, objective_count=4)
        self.states = (
            UnitControlState(slot=0, phase=UnitPhase.STAGED),
            UnitControlState(slot=1, phase=UnitPhase.STAGED),
            UnitControlState(slot=2, phase=UnitPhase.STAGED),
            UnitControlState(
                slot=3,
                phase=UnitPhase.ACTIVE,
                activated_step=0,
                current_objective_slot=0,
            ),
            UnitControlState(
                slot=4,
                phase=UnitPhase.ACTIVE,
                activated_step=0,
                current_objective_slot=1,
            ),
            UnitControlState(
                slot=5,
                phase=UnitPhase.ACTIVE,
                activated_step=0,
                current_objective_slot=2,
            ),
            UnitControlState(slot=6, phase=UnitPhase.PENDING),
            UnitControlState(slot=7, phase=UnitPhase.TERMINAL),
        )
        self.mask = build_joint_action_mask(
            self.space,
            self.states,
            (True, True, True, True),
            SharedSensorState.initial(
                SharedSensorConfig(capacity=2, max_requests_per_step=2)
            ),
            step=1,
        )
        rng = np.random.default_rng(123)
        self.observations = tuple(
            row for row in rng.normal(size=(self.space.unit_count, 6)).astype(np.float32)
        )
        self.policy = JointPPOPolicy(
            self.space,
            JointPPOConfig(
                observation_dim=6,
                hidden_dim=32,
                update_epochs=1,
                minibatch_size=8,
                target_kl=1.0,
                seed=19,
                device="cpu",
            ),
        )

    def _set_choice_biases(self, *, choose_yes: bool) -> None:
        sign = 7.0 if choose_yes else -7.0
        with torch.no_grad():
            self.policy.network.activation_head.bias.copy_(
                torch.tensor((-sign, sign))
            )
            self.policy.network.retarget_head.bias.copy_(torch.tensor((-sign, sign)))
            self.policy.network.objective_head.bias.copy_(
                torch.tensor((-1.5, -0.5, 0.5, 1.5))
            )
            self.policy.network.initial_movement_head.bias.copy_(
                torch.tensor((-1.0, 0.0, 1.0))
            )
            self.policy.network.movement_head.bias.copy_(
                torch.tensor((1.0, 0.0, -1.0))
            )
            self.policy.network.sensor_stop_head.bias.fill_(-sign)
            self.policy.network.sensor_slot_head[-1].bias.fill_(sign)

    def _make_transitions(self) -> tuple[JointTransition, ...]:
        buffer = JointTrajectoryBuffer(self.space, observation_dim=6)
        for index, choose_yes in enumerate((True, False, True, False, True, False)):
            self._set_choice_biases(choose_yes=choose_yes)
            observations = tuple(
                row + np.float32(index * 0.01) for row in self.observations
            )
            action, trace = self.policy.sample(
                observations,
                self.states,
                self.mask,
                deterministic=True,
            )
            buffer.append(
                JointTransition(
                    observations=observations,
                    states=self.states,
                    mask=self.mask,
                    action=action,
                    trace=trace,
                    rewards=tuple(float(index) * 0.01 for _ in self.states),
                    team_reward=float(index) * 0.02,
                    next_observations=observations,
                    terminated=tuple(True for _ in self.states),
                    truncated=tuple(False for _ in self.states),
                    team_terminated=True,
                    plan_rewards=tuple(0.1 + index * 0.01 for _ in self.states),
                    motion_rewards=tuple(0.2 + index * 0.01 for _ in self.states),
                    sensor_reward=0.3 + index * 0.01,
                )
            )
        # Evaluate every saved action under one common, non-saturated policy.
        with torch.no_grad():
            self.policy.network.activation_head.bias.copy_(torch.tensor((0.2, -0.1)))
            self.policy.network.retarget_head.bias.copy_(torch.tensor((-0.3, 0.1)))
            self.policy.network.sensor_stop_head.bias.fill_(0.25)
            self.policy.network.sensor_slot_head[-1].bias.fill_(-0.15)
        return buffer.items

    def _scalar_statistics(self, output, transitions):
        plan_log_probs = []
        plan_entropies = []
        motion_log_probs = []
        motion_entropies = []
        sensor_log_probs = []
        sensor_entropies = []
        planning_names = ("activation", "placement", "objective", "retarget")
        for index, transition in enumerate(transitions):
            evaluation = self.policy._evaluate_output(
                output.item(index),
                transition.states,
                transition.mask,
                transition.action,
                transition.trace.shared_sensor,
            )
            for slot in range(self.space.unit_count):
                names = tuple(
                    f"unit/{slot}/{name}"
                    for name in planning_names
                    if f"unit/{slot}/{name}" in evaluation.log_prob_by_term
                )
                if names:
                    plan_log_probs.append(
                        torch.stack(
                            tuple(evaluation.log_prob_by_term[name] for name in names)
                        ).sum()
                    )
                    plan_entropies.append(
                        torch.stack(
                            tuple(evaluation.entropy_by_term[name] for name in names)
                        ).sum()
                    )
                movement_name = f"unit/{slot}/movement"
                if movement_name in evaluation.log_prob_by_term:
                    motion_log_probs.append(evaluation.log_prob_by_term[movement_name])
                    motion_entropies.append(evaluation.entropy_by_term[movement_name])
            if "shared_sensor" in evaluation.log_prob_by_term:
                sensor_log_probs.append(evaluation.log_prob_by_term["shared_sensor"])
                sensor_entropies.append(evaluation.entropy_by_term["shared_sensor"])
        return tuple(
            torch.stack(values)
            for values in (
                plan_log_probs,
                plan_entropies,
                motion_log_probs,
                motion_entropies,
                sensor_log_probs,
                sensor_entropies,
            )
        )

    @staticmethod
    def _objective(statistics) -> torch.Tensor:
        return (
            statistics[0].sum()
            + 0.017 * statistics[1].sum()
            + 0.7 * statistics[2].sum()
            + 0.013 * statistics[3].sum()
            + 1.3 * statistics[4].sum()
            + 0.011 * statistics[5].sum()
        )

    def _vector_statistics(self, output, packed):
        evaluation = self.policy._evaluate_packed_output(output, packed)
        return (
            evaluation.plan_log_probs[packed.plan_active],
            evaluation.plan_entropies[packed.plan_active],
            evaluation.motion_log_probs[packed.movement_active],
            evaluation.motion_entropies[packed.movement_active],
            evaluation.sensor_log_probs[packed.sensor_active],
            evaluation.sensor_entropies[packed.sensor_active],
        )

    def test_packed_evaluator_matches_scalar_values_and_gradients(self) -> None:
        transitions = self._make_transitions()
        packed, plan_active, motion_active, sensor_active, term_count = (
            self.policy._pack_policy_rollout(transitions)
        )
        observations = torch.as_tensor(
            np.asarray([item.observations for item in transitions], dtype=np.float32)
        )

        output = self.policy.network(observations)
        scalar = self._scalar_statistics(output, transitions)
        vector = self._vector_statistics(output, packed)
        for scalar_values, vector_values in zip(scalar, vector):
            torch.testing.assert_close(
                vector_values,
                scalar_values,
                rtol=2e-6,
                atol=2e-6,
            )
        self.assertEqual(int(packed.plan_active.sum()), int(plan_active.sum()))
        self.assertTrue(
            torch.equal(packed.motion_active, torch.from_numpy(motion_active))
        )
        self.assertTrue(
            torch.equal(packed.motion_value_active, packed.movement_active)
        )
        self.assertEqual(int(packed.sensor_active.sum()), int(sensor_active.sum()))
        self.assertEqual(term_count, sum(len(item.trace.log_prob_by_term) for item in transitions))

        parameters = tuple(self.policy.network.parameters())
        scalar_output = self.policy.network(observations)
        scalar_objective = self._objective(
            self._scalar_statistics(scalar_output, transitions)
        )
        scalar_gradients = torch.autograd.grad(
            scalar_objective, parameters, allow_unused=True
        )
        vector_output = self.policy.network(observations)
        vector_objective = self._objective(
            self._vector_statistics(vector_output, packed)
        )
        vector_gradients = torch.autograd.grad(
            vector_objective, parameters, allow_unused=True
        )
        for scalar_gradient, vector_gradient in zip(
            scalar_gradients, vector_gradients
        ):
            self.assertEqual(scalar_gradient is None, vector_gradient is None)
            if scalar_gradient is not None:
                torch.testing.assert_close(
                    vector_gradient,
                    scalar_gradient,
                    rtol=3e-5,
                    atol=3e-5,
                )

    def test_packed_rollout_is_host_resident_and_moves_only_selected_rows(self) -> None:
        transitions = self._make_transitions()
        packed, *_ = self.policy._pack_policy_rollout(transitions)
        for value in packed.__dict__.values():
            self.assertEqual(value.device.type, "cpu")

        indices = torch.tensor((4, 1), dtype=torch.long)
        selected = packed.select(indices, device=self.policy.device)
        for name, value in packed.__dict__.items():
            selected_value = getattr(selected, name)
            self.assertEqual(selected_value.shape[0], 2)
            self.assertEqual(selected_value.device, self.policy.device)
            torch.testing.assert_close(
                selected_value.cpu(), value.index_select(0, indices)
            )

    @unittest.skipUnless(
        os.environ.get("JOINT_POLICY_BENCHMARK") == "1",
        "set JOINT_POLICY_BENCHMARK=1 to run the CPU microbenchmark",
    )
    def test_vectorized_evaluator_cpu_benchmark(self) -> None:
        base = self._make_transitions()
        transition_count = 128
        transitions = (base * math.ceil(transition_count / len(base)))[
            :transition_count
        ]
        packed, *_ = self.policy._pack_policy_rollout(transitions)
        observations = torch.as_tensor(
            np.asarray([item.observations for item in transitions], dtype=np.float32)
        )
        with torch.no_grad():
            output = self.policy.network(observations)
            self._vector_statistics(output, packed)
            scalar_start = time.perf_counter()
            scalar = self._scalar_statistics(output, transitions)
            scalar_seconds = time.perf_counter() - scalar_start
            vector_start = time.perf_counter()
            vector = self._vector_statistics(output, packed)
            vector_seconds = time.perf_counter() - vector_start
        for scalar_values, vector_values in zip(scalar, vector):
            torch.testing.assert_close(vector_values, scalar_values, rtol=2e-6, atol=2e-6)
        speedup = scalar_seconds / max(vector_seconds, 1e-12)
        print(
            f"scalar={scalar_seconds:.6f}s vectorized={vector_seconds:.6f}s "
            f"speedup={speedup:.2f}x"
        )
        self.assertLess(vector_seconds, scalar_seconds)

    @unittest.skipUnless(
        os.environ.get("JOINT_POLICY_CUDA_MEMORY_TEST") == "1"
        and torch.cuda.is_available(),
        "set JOINT_POLICY_CUDA_MEMORY_TEST=1 with a free visible GPU",
    )
    def test_cuda_update_memory_is_bounded_by_minibatch(self) -> None:
        """An 8x longer rollout must not materially raise accelerator memory."""

        space = JointSpaceSpec(unit_count=286, objective_count=18)
        states = tuple(
            UnitControlState(slot=slot, phase=UnitPhase.STAGED)
            for slot in range(space.unit_count)
        )
        mask = build_joint_action_mask(
            space,
            states,
            tuple(True for _ in range(space.objective_count)),
            SharedSensorState.initial(
                SharedSensorConfig(capacity=1, max_requests_per_step=1)
            ),
            step=0,
        )
        observations = tuple(
            row
            for row in np.random.default_rng(91)
            .normal(size=(space.unit_count, 382))
            .astype(np.float32)
        )

        allocated_peaks = []
        reserved_peaks = []
        for transition_count in (256, 2048):
            policy = JointPPOPolicy(
                space,
                JointPPOConfig(
                    observation_dim=382,
                    hidden_dim=256,
                    update_epochs=1,
                    minibatch_size=128,
                    value_inference_batch_size=128,
                    target_kl=0.0,
                    seed=31,
                    device="cuda:0",
                ),
            )
            action, trace = policy.sample(
                observations, states, mask, deterministic=True
            )
            transition = JointTransition(
                observations=observations,
                states=states,
                mask=mask,
                action=action,
                trace=trace,
                rewards=tuple(0.0 for _ in states),
                team_reward=0.0,
                next_observations=observations,
                terminated=tuple(True for _ in states),
                truncated=tuple(False for _ in states),
                team_terminated=True,
                plan_rewards=tuple(0.0 for _ in states),
                motion_rewards=tuple(0.0 for _ in states),
                sensor_reward=0.0,
            )
            buffer = JointTrajectoryBuffer(space, observation_dim=382)
            # Reusing an already immutable synthetic snapshot keeps this test's
            # host-memory footprint small; update treats the tuple as read-only.
            buffer._items = [transition] * transition_count

            packed, *_ = policy._pack_policy_rollout(buffer.items)
            self.assertTrue(
                all(value.device.type == "cpu" for value in packed.__dict__.values())
            )
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            policy.update(buffer, clear_buffer=False)
            torch.cuda.synchronize()
            allocated_peaks.append(torch.cuda.max_memory_allocated())
            reserved_peaks.append(torch.cuda.max_memory_reserved())
            del packed, buffer, transition, trace, action, policy
            torch.cuda.empty_cache()

        print(
            "cuda rollout staging peaks: "
            f"allocated T=256 {allocated_peaks[0] / 2**20:.1f} MiB, "
            f"T=2048 {allocated_peaks[1] / 2**20:.1f} MiB; "
            f"reserved T=256 {reserved_peaks[0] / 2**20:.1f} MiB, "
            f"T=2048 {reserved_peaks[1] / 2**20:.1f} MiB"
        )
        self.assertLessEqual(
            allocated_peaks[1], allocated_peaks[0] + 96 * 2**20
        )
        self.assertLessEqual(
            reserved_peaks[1], reserved_peaks[0] + 128 * 2**20
        )


if __name__ == "__main__":
    unittest.main()
