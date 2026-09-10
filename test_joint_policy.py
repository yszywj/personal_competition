"""Focused CPU tests for the masked joint actor and PPO implementation."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from personal_train.joint_policy import JointPPOConfig, JointPPOPolicy
from personal_train.joint_rl_core import (
    BinaryChoice,
    JointAction,
    JointSpaceSpec,
    JointTrajectoryBuffer,
    JointTransition,
    Movement,
    SharedSensorAction,
    SharedSensorConfig,
    SharedSensorState,
    UnitAction,
    UnitControlState,
    UnitPhase,
    branch_activity,
    build_joint_action_mask,
    expected_log_prob_terms,
)


class JointPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.space = JointSpaceSpec(unit_count=3, objective_count=2)
        self.states = (
            UnitControlState(slot=0, phase=UnitPhase.STAGED),
            UnitControlState(
                slot=1,
                phase=UnitPhase.ACTIVE,
                activated_step=0,
                current_objective_slot=0,
            ),
            UnitControlState(slot=2, phase=UnitPhase.TERMINAL),
        )
        self.mask = build_joint_action_mask(
            self.space,
            self.states,
            (True, True),
            SharedSensorState.initial(
                SharedSensorConfig(capacity=1, max_requests_per_step=1)
            ),
            step=1,
        )
        self.observations = tuple(
            np.asarray((slot, 0.25, -0.5, 1.0), dtype=np.float32)
            for slot in range(self.space.unit_count)
        )
        self.policy = JointPPOPolicy(
            self.space,
            JointPPOConfig(
                observation_dim=4,
                hidden_dim=32,
                update_epochs=2,
                minibatch_size=2,
                target_kl=1.0,
                seed=17,
                device="cpu",
            ),
        )

    def _force_conditional_yes_actions(self) -> None:
        with torch.no_grad():
            for parameter in self.policy.network.parameters():
                parameter.zero_()
            self.policy.network.activation_head.bias.copy_(torch.tensor((-3.0, 3.0)))
            self.policy.network.retarget_head.bias.copy_(torch.tensor((-3.0, 3.0)))
            self.policy.network.objective_head.bias.copy_(torch.tensor((-2.0, 2.0)))
            self.policy.network.movement_head.bias.copy_(torch.tensor((-2.0, -1.0, 2.0)))
            self.policy.network.initial_movement_head.bias.copy_(
                torch.tensor((-2.0, -1.0, 2.0))
            )
            self.policy.network.placement_mean_head.bias.copy_(
                torch.tensor((0.25, -0.5))
            )
            self.policy.network.placement_log_std_head.bias.fill_(-0.7)
            self.policy.network.sensor_stop_head.bias.fill_(-3.0)
            self.policy.network.sensor_slot_head[-1].bias.fill_(3.0)

    def test_sample_and_evaluate_preserve_only_active_terms(self) -> None:
        self._force_conditional_yes_actions()
        action, trace = self.policy.sample(
            self.observations, self.states, self.mask, deterministic=True
        )

        self.assertEqual(action.units[0].activate, BinaryChoice.YES)
        self.assertEqual(action.units[0].objective_slot, 1)
        self.assertEqual(action.units[1].retarget, BinaryChoice.YES)
        self.assertEqual(action.units[1].objective_slot, 1)
        self.assertEqual(action.units[1].movement, Movement.POSITIVE)
        self.assertEqual(action.units[2], UnitAction.noop(2))
        self.assertEqual(action.shared_sensor.requester_slots, (1,))
        self.assertEqual(
            frozenset(trace.log_prob_by_term),
            expected_log_prob_terms(
                self.states, action, shared_sensor_active=True
            ),
        )
        # The updated core treats launch-time movement as conditional on YES.
        if branch_activity(self.states[0], action.units[0]).movement:
            self.assertEqual(action.units[0].movement, Movement.POSITIVE)
            self.assertIn("unit/0/movement", trace.log_prob_by_term)
        self.assertNotIn("unit/2/activation", trace.log_prob_by_term)
        trace.validate(self.states, action, self.mask)

        evaluation = self.policy.evaluate(
            self.observations,
            self.states,
            self.mask,
            action,
            shared_sensor_trace=trace.shared_sensor,
        )
        self.assertEqual(
            frozenset(evaluation.log_prob_by_term),
            frozenset(trace.log_prob_by_term),
        )
        for name, behavior_log_prob in trace.log_prob_by_term.items():
            self.assertAlmostEqual(
                float(evaluation.log_prob_by_term[name].detach().item()),
                behavior_log_prob,
                places=5,
                msg=name,
            )
        self.assertEqual(tuple(evaluation.values_by_unit.shape), (3,))
        self.assertEqual(tuple(evaluation.team_value.shape), ())
        self.assertEqual(tuple(evaluation.plan_values_by_unit.shape), (3,))
        self.assertEqual(tuple(evaluation.motion_values_by_unit.shape), (3,))
        self.assertEqual(tuple(evaluation.sensor_value.shape), ())

    def test_placement_and_launch_movement_are_conditioned_on_target(self) -> None:
        no_sensor_mask = build_joint_action_mask(
            self.space,
            self.states,
            (True, True),
            SharedSensorState.initial(SharedSensorConfig(capacity=0)),
            step=1,
        )
        common_units = (
            UnitAction(
                slot=1,
                retarget=BinaryChoice.NO,
                movement=Movement.NEUTRAL,
            ),
            UnitAction.noop(2),
        )
        actions = tuple(
            JointAction(
                units=(
                    UnitAction(
                        slot=0,
                        activate=BinaryChoice.YES,
                        placement=(0.1, -0.2),
                        objective_slot=objective_slot,
                        movement=Movement.NEUTRAL,
                    ),
                    *common_units,
                )
            )
            for objective_slot in (0, 1)
        )
        evaluations = tuple(
            self.policy.evaluate(
                self.observations,
                self.states,
                no_sensor_mask,
                action,
            )
            for action in actions
        )
        placement_log_probs = tuple(
            float(item.log_prob_by_term["unit/0/placement"].detach().item())
            for item in evaluations
        )
        movement_log_probs = tuple(
            float(item.log_prob_by_term["unit/0/movement"].detach().item())
            for item in evaluations
        )
        self.assertNotAlmostEqual(*placement_log_probs, places=6)
        self.assertNotAlmostEqual(*movement_log_probs, places=6)

    def test_inactive_branch_parameters_cannot_change_scored_log_prob(self) -> None:
        no_sensor_mask = build_joint_action_mask(
            self.space,
            self.states,
            (True, True),
            SharedSensorState.initial(SharedSensorConfig(capacity=0)),
            step=1,
        )
        action = JointAction(
            units=(
                UnitAction.noop(0),
                UnitAction(
                    slot=1,
                    retarget=BinaryChoice.NO,
                    movement=Movement.POSITIVE,
                ),
                UnitAction.noop(2),
            ),
            shared_sensor=SharedSensorAction(),
        )
        before = self.policy.evaluate(
            self.observations, self.states, no_sensor_mask, action
        )
        self.assertEqual(
            frozenset(before.log_prob_by_term),
            {
                "unit/0/activation",
                "unit/1/retarget",
                "unit/1/movement",
            },
        )
        with torch.no_grad():
            self.policy.network.objective_head.weight.add_(100.0)
            self.policy.network.objective_head.bias.add_(100.0)
            self.policy.network.placement_mean_head.weight.sub_(50.0)
            self.policy.network.placement_log_std_head.bias.add_(10.0)
        after = self.policy.evaluate(
            self.observations, self.states, no_sensor_mask, action
        )
        for name in before.log_prob_by_term:
            self.assertEqual(
                float(before.log_prob_by_term[name].detach().item()),
                float(after.log_prob_by_term[name].detach().item()),
                name,
            )

    def test_tanh_placement_log_prob_is_finite_near_legal_bounds(self) -> None:
        no_sensor_mask = build_joint_action_mask(
            self.space,
            self.states,
            (True, True),
            SharedSensorState.initial(SharedSensorConfig(capacity=0)),
            step=1,
        )
        launch_movement = (
            Movement.POSITIVE
            if no_sensor_mask.by_unit[0].movement[Movement.POSITIVE]
            else Movement.NEUTRAL
        )
        action = JointAction(
            units=(
                UnitAction(
                    slot=0,
                    activate=BinaryChoice.YES,
                    placement=(0.999999, -0.999999),
                    objective_slot=0,
                    movement=launch_movement,
                ),
                UnitAction(slot=1, movement=Movement.NEUTRAL),
                UnitAction.noop(2),
            )
        )
        evaluation = self.policy.evaluate(
            self.observations, self.states, no_sensor_mask, action
        )
        self.assertTrue(
            torch.isfinite(evaluation.log_prob_by_term["unit/0/placement"]).item()
        )

    def test_shared_sensor_samples_without_replacement_up_to_capacity(self) -> None:
        space = JointSpaceSpec(unit_count=2, objective_count=2)
        states = tuple(
            UnitControlState(
                slot=slot,
                phase=UnitPhase.ACTIVE,
                activated_step=0,
                current_objective_slot=0,
            )
            for slot in range(2)
        )
        mask = build_joint_action_mask(
            space,
            states,
            (True, True),
            SharedSensorState.initial(
                SharedSensorConfig(capacity=2, max_requests_per_step=2)
            ),
            step=1,
        )
        policy = JointPPOPolicy(
            space,
            JointPPOConfig(observation_dim=4, hidden_dim=16, device="cpu"),
        )
        with torch.no_grad():
            for parameter in policy.network.parameters():
                parameter.zero_()
            policy.network.sensor_stop_head.bias.fill_(-2.0)
            policy.network.sensor_slot_head[-1].bias.fill_(2.0)
        observations = tuple(np.zeros(4, dtype=np.float32) for _ in states)
        action, trace = policy.sample(
            observations, states, mask, deterministic=True
        )
        self.assertEqual(action.shared_sensor.requester_slots, (0, 1))
        self.assertEqual(trace.shared_sensor.tokens, (1, 2))
        self.assertTrue(trace.shared_sensor.masks[0].tolist() == [True, True, True])
        self.assertTrue(trace.shared_sensor.masks[1].tolist() == [True, False, True])
        evaluation = policy.evaluate(
            observations,
            states,
            mask,
            action,
            shared_sensor_trace=trace.shared_sensor,
        )
        self.assertAlmostEqual(
            float(evaluation.log_prob_by_term["shared_sensor"].detach().item()),
            trace.log_prob_by_term["shared_sensor"],
            places=6,
        )

    def test_finish_rollout_updates_once_and_counts_complete_episodes(self) -> None:
        buffer = JointTrajectoryBuffer(self.space, observation_dim=4)
        for step in range(5):
            observations = tuple(
                value + np.float32(step * 0.01) for value in self.observations
            )
            next_observations = tuple(
                value + np.float32((step + 1) * 0.01)
                for value in self.observations
            )
            action, trace = self.policy.sample(observations, self.states, self.mask)
            # Four complete episodes: lengths 1, 1, 2 and 1 transitions.
            final = step in {0, 1, 3, 4}
            branch_rewards = (0.1 * step, 1.0 - 0.1 * step, 1000.0)
            sensor_reward = 0.2 * step
            buffer.append(
                JointTransition(
                    observations=observations,
                    states=self.states,
                    mask=self.mask,
                    action=action,
                    trace=trace,
                    rewards=branch_rewards,
                    team_reward=sensor_reward,
                    next_observations=next_observations,
                    terminated=(final, final, True),
                    truncated=(False, False, False),
                    team_terminated=final,
                    plan_rewards=branch_rewards,
                    motion_rewards=branch_rewards,
                    sensor_reward=sensor_reward,
                )
            )
        parameter_before = {
            name: value.detach().clone()
            for name, value in self.policy.network.named_parameters()
        }
        metrics = self.policy.finish_rollout(buffer, episode_count=4)
        self.assertEqual(len(buffer), 0)
        self.assertEqual(self.policy.update_count, 1)
        self.assertEqual(self.policy.episode_count, 4)
        self.assertEqual(metrics["joint_steps"], 5.0)
        self.assertEqual(metrics["plan_decisions"], 10.0)
        self.assertGreaterEqual(metrics["motion_decisions"], 5.0)
        self.assertEqual(metrics["sensor_decisions"], 5.0)
        self.assertEqual(
            metrics["actor_decisions"],
            metrics["plan_decisions"]
            + metrics["motion_decisions"]
            + metrics["sensor_decisions"],
        )
        self.assertEqual(
            metrics["unit_value_samples"], metrics["motion_value_samples"]
        )
        self.assertGreater(metrics["active_log_prob_terms"], 0.0)
        self.assertGreater(metrics["minibatch_updates"], 0.0)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertTrue(
            any(
                not torch.equal(parameter_before[name], value.detach())
                for name, value in self.policy.network.named_parameters()
            )
        )

        empty = JointTrajectoryBuffer(self.space, observation_dim=4)
        for invalid in (True, 0, -1, 1.5):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "positive integer"
            ):
                self.policy.finish_rollout(empty, episode_count=invalid)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.policy.finish_rollout(empty, episode_count=1)

        incomplete = JointTrajectoryBuffer(self.space, observation_dim=4)
        action, trace = self.policy.sample(self.observations, self.states, self.mask)
        incomplete.append(
            JointTransition(
                observations=self.observations,
                states=self.states,
                mask=self.mask,
                action=action,
                trace=trace,
                rewards=(0.0, 0.0, 0.0),
                team_reward=0.0,
                next_observations=self.observations,
                terminated=(False, False, True),
                truncated=(False, False, False),
            )
        )
        with self.assertRaisesRegex(ValueError, "complete team episode boundary"):
            self.policy.finish_rollout(incomplete, episode_count=1)

    def test_explicit_branch_rewards_produce_separate_advantages(self) -> None:
        self._force_conditional_yes_actions()
        action, trace = self.policy.sample(
            self.observations, self.states, self.mask, deterministic=True
        )
        buffer = JointTrajectoryBuffer(self.space, observation_dim=4)
        buffer.append(
            JointTransition(
                observations=self.observations,
                states=self.states,
                mask=self.mask,
                action=action,
                trace=trace,
                rewards=(99.0, 99.0, 99.0),
                team_reward=99.0,
                next_observations=self.observations,
                terminated=(True, True, True),
                truncated=(False, False, False),
                team_terminated=True,
                plan_rewards=(0.8, 0.4, 0.0),
                motion_rewards=(0.1, 0.2, 0.0),
                sensor_reward=0.3,
            )
        )

        metrics = self.policy.finish_episode(buffer)

        self.assertAlmostEqual(metrics["plan_advantage_mean"], 0.6, places=6)
        self.assertAlmostEqual(metrics["motion_advantage_mean"], 0.15, places=6)
        self.assertAlmostEqual(metrics["sensor_advantage_mean"], 0.3, places=6)
        self.assertEqual(metrics["plan_decisions"], 2.0)
        self.assertEqual(metrics["motion_decisions"], 2.0)
        self.assertEqual(metrics["sensor_decisions"], 1.0)

    def test_plan_weights_balance_units_with_different_decision_counts(self) -> None:
        class Boundary:
            def __init__(self, done: bool) -> None:
                self.team_terminated = done
                self.team_truncated = False

        transitions = (Boundary(False), Boundary(False), Boundary(True))
        active = np.asarray(
            ((True, True), (True, False), (True, False)), dtype=np.bool_
        )

        weights = self.policy._episode_balanced_plan_weights(transitions, active)

        self.assertAlmostEqual(float(weights[:, 0].sum()), 2.0)
        self.assertAlmostEqual(float(weights[:, 1].sum()), 2.0)
        self.assertAlmostEqual(float(weights[active].mean()), 1.0)

    def test_update_rejects_missing_branch_reward_contract(self) -> None:
        action, trace = self.policy.sample(self.observations, self.states, self.mask)
        buffer = JointTrajectoryBuffer(self.space, observation_dim=4)
        buffer.append(
            JointTransition(
                observations=self.observations,
                states=self.states,
                mask=self.mask,
                action=action,
                trace=trace,
                rewards=(0.0, 0.0, 0.0),
                team_reward=0.0,
                next_observations=self.observations,
                terminated=(True, True, True),
                truncated=(False, False, False),
                team_terminated=True,
            )
        )

        with self.assertRaisesRegex(ValueError, "explicit plan/motion/sensor"):
            self.policy.finish_episode(buffer)

    def test_finish_episode_rejects_empty_buffer_without_counting(self) -> None:
        buffer = JointTrajectoryBuffer(self.space, observation_dim=4)
        self.policy.episode_count = 7

        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.policy.finish_episode(buffer, clear_buffer=False)

        self.assertEqual(self.policy.episode_count, 7)

    def test_checkpoint_round_trip_restores_config_optimizer_and_rng(self) -> None:
        self.policy.update_count = 4
        self.policy.transition_count = 123
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "joint-policy.pt"
            self.policy.save(path)
            expected_action, expected_trace = self.policy.sample(
                self.observations, self.states, self.mask
            )
            restored = JointPPOPolicy.from_checkpoint(path, device="cpu")
            actual_action, actual_trace = restored.sample(
                self.observations, self.states, self.mask
            )

        self.assertEqual(restored.space, self.space)
        self.assertEqual(restored.config.observation_dim, 4)
        self.assertEqual(restored.update_count, 4)
        self.assertEqual(restored.transition_count, 123)
        self.assertEqual(actual_action, expected_action)
        self.assertEqual(
            dict(actual_trace.log_prob_by_term), dict(expected_trace.log_prob_by_term)
        )
        self.assertEqual(
            actual_trace.shared_sensor.tokens, expected_trace.shared_sensor.tokens
        )

    def test_checkpoint_load_rejects_mixed_training_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "joint-policy.pt"
            self.policy.save(path)
            incompatible = JointPPOPolicy(
                self.space,
                JointPPOConfig(
                    observation_dim=4,
                    hidden_dim=32,
                    learning_rate=2e-4,
                    update_epochs=2,
                    minibatch_size=2,
                    target_kl=1.0,
                    seed=17,
                    device="cpu",
                ),
            )
            with self.assertRaisesRegex(ValueError, "learning_rate"):
                incompatible.load(path)

    def test_load_weights_accepts_new_training_config_and_preserves_state(self) -> None:
        self._force_conditional_yes_actions()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "joint-policy.pt"
            self.policy.update_count = 4
            self.policy.transition_count = 123
            self.policy.episode_count = 5
            self.policy.save(path)
            expected_weights = {
                name: value.detach().clone()
                for name, value in self.policy.network.state_dict().items()
            }

            target = JointPPOPolicy(
                self.space,
                JointPPOConfig(
                    observation_dim=4,
                    hidden_dim=32,
                    learning_rate=1e-4,
                    learning_rate_final=1e-5,
                    target_kl=0.01,
                    update_epochs=1,
                    minibatch_size=4,
                    seed=99,
                    device="cpu",
                ),
            )
            target.optimizer.zero_grad(set_to_none=True)
            sum(parameter.square().sum() for parameter in target.network.parameters()).backward()
            target.optimizer.step()
            target.update_count = 9
            target.transition_count = 456
            target.episode_count = 12
            target.last_metrics = {"preserved": 3.5}
            config_before = copy.deepcopy(target.config.__dict__)
            optimizer_before = copy.deepcopy(target.optimizer.state_dict())
            numpy_rng_before = copy.deepcopy(target._rng.bit_generator.state)
            torch_rng_before = torch.get_rng_state().clone()

            target.load_weights(path)

        self.assertEqual(target.config.__dict__, config_before)
        self.assertEqual(target.update_count, 9)
        self.assertEqual(target.transition_count, 456)
        self.assertEqual(target.episode_count, 12)
        self.assertEqual(target.last_metrics, {"preserved": 3.5})
        optimizer_after = target.optimizer.state_dict()
        self.assertEqual(optimizer_after["param_groups"], optimizer_before["param_groups"])
        self.assertEqual(optimizer_after["state"].keys(), optimizer_before["state"].keys())
        for parameter_id, saved_state in optimizer_before["state"].items():
            actual_state = optimizer_after["state"][parameter_id]
            self.assertEqual(actual_state.keys(), saved_state.keys())
            for name, expected in saved_state.items():
                actual = actual_state[name]
                if isinstance(expected, torch.Tensor):
                    self.assertTrue(torch.equal(actual, expected), name)
                else:
                    self.assertEqual(actual, expected)
        self.assertEqual(target._rng.bit_generator.state, numpy_rng_before)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng_before))
        for name, expected in expected_weights.items():
            self.assertTrue(torch.equal(target.network.state_dict()[name], expected), name)

    def test_load_weights_rejects_incompatible_network_structure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "joint-policy.pt"
            self.policy.save(path)
            incompatible = JointPPOPolicy(
                self.space,
                JointPPOConfig(
                    observation_dim=4,
                    hidden_dim=16,
                    device="cpu",
                ),
            )

            with self.assertRaisesRegex(ValueError, "network structure"):
                incompatible.load_weights(path)

    def test_bad_shapes_and_missing_sensor_trace_fail_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "observations"):
            self.policy.sample(self.observations[:2], self.states, self.mask)
        action, _ = self.policy.sample(
            self.observations, self.states, self.mask, deterministic=True
        )
        with self.assertRaisesRegex(ValueError, "requires its saved trace"):
            self.policy.evaluate(
                self.observations, self.states, self.mask, action
            )


if __name__ == "__main__":
    unittest.main()
