"""Focused tests for fixed-seed joint PPO evaluation."""

from __future__ import annotations

import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from personal_train.eval_joint_ppo import (
    RESULTS_ROOT,
    _aggregate,
    _checkpoint_metadata,
    _contract_differences,
    _distribution_summary,
    _evaluate_episode,
    _parse_seeds,
    _resolve_result_directory,
    _seed_summary,
    _validate_environment_contract,
    parse_args,
)
from personal_train.joint_game_env import JointGameStep
from personal_train.joint_policy import JointPPOConfig, JointPPOPolicy
from personal_train.joint_rl_core import (
    BinaryChoice,
    JointAction,
    JointPolicyTrace,
    JointSpaceSpec,
    Movement,
    SharedSensorConfig,
    SharedSensorState,
    UnitAction,
    UnitControlState,
    UnitPhase,
    build_joint_action_mask,
)
from personal_train.train_joint_ppo import _save_policy, _training_contract


class JointEvaluationArgumentTests(unittest.TestCase):
    def test_defaults_are_fixed_deterministic_seed_set(self) -> None:
        args = parse_args(["--checkpoint", "model.pt", "--scenario", "E01"])
        self.assertEqual(args.seeds, tuple(range(1, 11)))
        self.assertEqual(args.episodes_per_seed, 1)
        self.assertTrue(args.deterministic)
        self.assertEqual(args.cvar_alpha, 0.2)

    def test_seed_formats_and_stochastic_mode(self) -> None:
        args = parse_args(
            [
                "--checkpoint",
                "model.pt",
                "--scenario",
                "easy/E01",
                "--seeds",
                "7,11",
                "19",
                "--episodes-per-seed",
                "3",
                "--stochastic",
            ]
        )
        self.assertEqual(args.seeds, (7, 11, 19))
        self.assertEqual(args.episodes_per_seed, 3)
        self.assertFalse(args.deterministic)

    def test_invalid_arguments_are_rejected(self) -> None:
        invalid = (
            ["--checkpoint", "x", "--scenario", "E01", "--seeds", "1,1"],
            ["--checkpoint", "x", "--scenario", "E01", "--seeds", "-1"],
            ["--checkpoint", "x", "--scenario", "E01", "--seeds", "1,"],
            ["--checkpoint", "x", "--scenario", "E01", "--episodes-per-seed", "0"],
            ["--checkpoint", "x", "--scenario", "E01", "--cvar-alpha", "0"],
            ["--checkpoint", "x", "--scenario", "E01", "--blue-policy", "  "],
        )
        for argv in invalid:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(argv)

    def test_result_directory_is_always_below_results(self) -> None:
        run_name, path = _resolve_result_directory(
            scenario_id="E01",
            checkpoint=Path("best.pt"),
            run_id="fixed_eval",
            timestamp="20260101_010203_000004",
        )
        self.assertEqual(run_name, "fixed_eval_20260101_010203_000004")
        self.assertEqual(path.parent, RESULTS_ROOT.resolve())


class JointEvaluationStatisticsTests(unittest.TestCase):
    def test_distribution_summary_uses_population_std_and_lower_tail_cvar(self) -> None:
        summary = _distribution_summary((10.0, 20.0, 30.0, 100.0), cvar_alpha=0.5)
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["mean"], 40.0)
        self.assertAlmostEqual(summary["std"], float(np.std([10, 20, 30, 100])))
        self.assertEqual(summary["median"], 25.0)
        self.assertEqual(summary["min"], 10.0)
        self.assertEqual(summary["max"], 100.0)
        self.assertEqual(summary["cvar_lower"], 15.0)
        self.assertEqual(summary["cvar_tail_count"], 2)

    def test_cvar_rounds_tail_count_up_and_rejects_nonfinite_values(self) -> None:
        summary = _distribution_summary((1.0, 2.0, 9.0), cvar_alpha=0.5)
        self.assertEqual(summary["cvar_tail_count"], 2)
        self.assertEqual(summary["cvar_lower"], 1.5)
        with self.assertRaises(ValueError):
            _distribution_summary((1.0, math.nan), cvar_alpha=0.2)

    def test_seed_and_global_aggregates_keep_both_risk_views(self) -> None:
        records = [
            {
                "seed": 1,
                "score": 10.0,
                "steps": 4,
                "team_return": 0.1,
                "planning_reward_sum": 0.4,
                "planning_reward_mean": 0.2,
                "planning_eligible_units": 2,
                "planning_local_credit_allocated": 0.1,
                "planning_local_credit_unallocated": 0.0,
                "motion_return": 1.0,
                "sensor_return": 0.05,
                "unit_reward_sum": 1.0,
                "accepted_activations": 2,
                "accepted_sensor_requests": 3,
            },
            {
                "seed": 1,
                "score": 20.0,
                "steps": 6,
                "team_return": 0.2,
                "planning_reward_sum": 0.8,
                "planning_reward_mean": 0.4,
                "planning_eligible_units": 4,
                "planning_local_credit_allocated": 0.2,
                "planning_local_credit_unallocated": 0.0,
                "motion_return": 2.0,
                "sensor_return": 0.1,
                "unit_reward_sum": 2.0,
                "accepted_activations": 4,
                "accepted_sensor_requests": 5,
            },
            {
                "seed": 2,
                "score": 40.0,
                "steps": 8,
                "team_return": 0.4,
                "planning_reward_sum": 1.2,
                "planning_reward_mean": 0.6,
                "planning_eligible_units": 6,
                "planning_local_credit_allocated": 0.4,
                "planning_local_credit_unallocated": 0.0,
                "motion_return": 4.0,
                "sensor_return": 0.2,
                "unit_reward_sum": 4.0,
                "accepted_activations": 6,
                "accepted_sensor_requests": 7,
            },
        ]
        first = _seed_summary(1, records[:2], cvar_alpha=0.5)
        second = _seed_summary(2, records[2:], cvar_alpha=0.5)
        aggregate = _aggregate(records, (first, second), cvar_alpha=0.5)
        self.assertEqual(first["score_mean"], 15.0)
        self.assertEqual(first["steps_mean"], 5.0)
        self.assertEqual(first["motion_return_mean"], 1.5)
        self.assertAlmostEqual(first["sensor_return_mean"], 0.075)
        self.assertAlmostEqual(first["planning_reward_mean"], 0.3)
        self.assertAlmostEqual(aggregate["episode_score"]["mean"], 70.0 / 3.0)
        self.assertEqual(aggregate["seed_mean_score"]["mean"], 27.5)
        self.assertAlmostEqual(aggregate["motion_return"]["mean"], 7.0 / 3.0)
        self.assertAlmostEqual(aggregate["sensor_return"]["mean"], 0.35 / 3.0)


class JointEvaluationContractTests(unittest.TestCase):
    def test_contract_comparison_reports_nested_leaf_paths(self) -> None:
        saved = {"version": 2, "sensor": {"capacity": 100}, "ids": (1, 2)}
        current = {"version": 2, "sensor": {"capacity": 50}, "ids": (1, 3)}
        self.assertEqual(
            _contract_differences(saved, current),
            ("ids[1]", "sensor.capacity"),
        )
        with self.assertRaisesRegex(ValueError, "ids\\[1\\].*sensor.capacity"):
            _validate_environment_contract(saved, current)
        _validate_environment_contract(saved, dict(saved))

    def test_checkpoint_metadata_accepts_best_and_requires_env_contract(self) -> None:
        torch.set_num_threads(1)
        space = JointSpaceSpec(unit_count=1, objective_count=1)
        policy = JointPPOPolicy(
            space,
            JointPPOConfig(observation_dim=3, hidden_dim=8, device="cpu"),
        )
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "best.pt"
            _save_policy(
                policy,
                path,
                environment_contract={"version": 2, "scenario_id": "TEST"},
                trainer_contract=_training_contract(
                    seed=4, blue_policy="test_blue", rollout_episodes=4
                ),
                checkpoint_role="best_behavior",
                resume_safe=False,
            )
            metadata = _checkpoint_metadata(path)
            self.assertEqual(metadata["checkpoint_role"], "best_behavior")
            self.assertEqual(metadata["trainer_contract"]["blue_policy"], "test_blue")

            raw_checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            raw_checkpoint.pop("joint_env_contract")
            invalid = Path(raw) / "invalid.pt"
            torch.save(raw_checkpoint, invalid)
            with self.assertRaisesRegex(ValueError, "environment contract"):
                _checkpoint_metadata(invalid)


class JointEvaluationEpisodeTests(unittest.TestCase):
    def test_episode_uses_requested_mode_and_does_not_update_policy(self) -> None:
        torch.set_num_threads(1)
        space = JointSpaceSpec(unit_count=1, objective_count=1)
        states = (UnitControlState(slot=0, phase=UnitPhase.TERMINAL),)
        sensor_state = SharedSensorState.initial(SharedSensorConfig(capacity=0))
        mask = build_joint_action_mask(
            space,
            states,
            objective_valid=(True,),
            sensor_state=sensor_state,
            step=0,
        )
        observations = (np.asarray((0.0, 0.5, -0.5), dtype=np.float32),)
        inner = JointPPOPolicy(
            space,
            JointPPOConfig(
                observation_dim=3,
                hidden_dim=8,
                update_epochs=1,
                minibatch_size=1,
                seed=3,
                device="cpu",
            ),
        )
        inner.set_training(False)

        class RecordingPolicy:
            def __init__(self) -> None:
                self.deterministic_values: list[bool] = []

            def sample(self, observations, states, mask, *, deterministic):
                self.deterministic_values.append(deterministic)
                return inner.sample(
                    observations, states, mask, deterministic=deterministic
                )

        class FakeEnvironment:
            observation_dim = 3
            max_steps = 1
            current_step = 0
            is_debug_horizon = True
            objective_ids = (51,)
            unit_ids = (2,)
            unit_types = (21000,)
            reward_policy = SimpleNamespace(
                objective_ids=(51,),
                objective_weights=((51, 1.0),),
                objective_initial_health=((51, 100.0),),
            )
            config = SimpleNamespace(
                planning_team_weight=0.7,
                planning_local_weight=0.3,
            )

            def __init__(self) -> None:
                self.space = space
                self.states = states

            def reset(self):
                self.current_step = 0
                return observations

            def action_mask(self):
                return mask

            def step(self, _action):
                self.current_step = 1
                return JointGameStep(
                    observations=observations,
                    states=states,
                    rewards=(0.0,),
                    team_reward=0.25,
                    terminated=(False,),
                    truncated=(True,),
                    team_terminated=False,
                    team_truncated=True,
                    done=True,
                    score=12.5,
                    raw_observation={"entities": {51: {"health": 87.5}}},
                    submitted_commands=0,
                    accepted_activations=(),
                    accepted_sensor_requests=(),
                    sensor_reward=None,
                )

        policy = RecordingPolicy()
        before = inner.update_count
        record = _evaluate_episode(
            policy,  # type: ignore[arg-type]
            FakeEnvironment(),  # type: ignore[arg-type]
            deterministic=False,
            seed=11,
            episode=2,
            global_episode=5,
            progress_every=10,
        )
        self.assertEqual(policy.deterministic_values, [False])
        self.assertEqual(inner.update_count, before)
        self.assertEqual(record["score"], 12.5)
        self.assertEqual(record["mode"], "stochastic")
        self.assertEqual(record["termination_reason"], "debug_horizon")
        self.assertEqual(record["motion_return"], 0.0)
        self.assertEqual(record["sensor_return"], 0.25)
        self.assertEqual(record["planning_reward_sum"], 0.0)
        self.assertEqual(record["planning_eligible_units"], 0)
        self.assertEqual(record["planning_local_credit_allocated"], 0.0)
        self.assertEqual(record["planning_local_credit_unallocated"], 0.125)
        self.assertIn("planning_credit", record)
        self.assertIn("action_diagnostics", record)

    def test_planning_credit_tracks_accepted_post_step_assignment(self) -> None:
        space = JointSpaceSpec(unit_count=1, objective_count=1)
        staged = (UnitControlState(slot=0, phase=UnitPhase.STAGED),)
        active = (
            UnitControlState(
                slot=0,
                phase=UnitPhase.ACTIVE,
                activated_step=0,
                current_objective_slot=0,
            ),
        )
        terminal = (UnitControlState(slot=0, phase=UnitPhase.TERMINAL),)
        sensor_state = SharedSensorState.initial(SharedSensorConfig(capacity=0))
        mask = build_joint_action_mask(
            space,
            staged,
            objective_valid=(True,),
            sensor_state=sensor_state,
            step=0,
        )
        observations = (np.asarray((0.0, 0.0, 0.0), dtype=np.float32),)
        action = JointAction.from_sequence(
            (
                UnitAction(
                    slot=0,
                    activate=BinaryChoice.YES,
                    placement=(0.0, 0.0),
                    objective_slot=0,
                    movement=Movement.NEUTRAL,
                ),
            )
        )
        trace = JointPolicyTrace(
            log_prob_by_term={
                "unit/0/activation": 0.0,
                "unit/0/placement": 0.0,
                "unit/0/objective": 0.0,
                "unit/0/movement": 0.0,
            },
            values_by_unit=(0.0,),
            team_value=0.0,
        )

        class FixedPolicy:
            def sample(self, _observations, _states, _mask, *, deterministic):
                self.deterministic = deterministic
                return action, trace

        class FakeEnvironment:
            observation_dim = 3
            max_steps = 1
            current_step = 0
            is_debug_horizon = True
            objective_ids = (51,)
            unit_ids = (2,)
            unit_types = (21000,)
            reward_policy = SimpleNamespace(
                objective_ids=(51,),
                objective_weights=((51, 1.0),),
                objective_initial_health=((51, 100.0),),
            )
            config = SimpleNamespace(
                planning_team_weight=0.7,
                planning_local_weight=0.3,
            )

            def __init__(self) -> None:
                self.space = space
                self.states = staged

            def reset(self):
                self.current_step = 0
                self.states = staged
                return observations

            def action_mask(self):
                return mask

            def step(self, _action):
                self.current_step = 1
                self.states = terminal
                return JointGameStep(
                    observations=observations,
                    states=terminal,
                    rewards=(0.75,),
                    team_reward=0.5,
                    terminated=(True,),
                    truncated=(False,),
                    team_terminated=True,
                    team_truncated=False,
                    done=True,
                    score=50.0,
                    raw_observation={"entities": {51: {"health": 50.0}}},
                    submitted_commands=1,
                    accepted_activations=(0,),
                    accepted_sensor_requests=(),
                    sensor_reward=0.125,
                    assignment_states=active,
                )

        record = _evaluate_episode(
            FixedPolicy(),  # type: ignore[arg-type]
            FakeEnvironment(),  # type: ignore[arg-type]
            deterministic=True,
            seed=17,
            episode=1,
            global_episode=1,
            progress_every=10,
        )
        self.assertEqual(record["planning_eligible_units"], 1)
        self.assertEqual(record["planning_reward_sum"], 0.5)
        self.assertEqual(record["planning_reward_mean"], 0.5)
        self.assertEqual(record["planning_local_credit_allocated"], 0.5)
        self.assertEqual(record["planning_local_credit_unallocated"], 0.0)
        self.assertEqual(record["motion_return"], 0.75)
        self.assertEqual(record["sensor_return"], 0.125)
        self.assertEqual(
            record["planning_credit"]["assignment_duration"],
            [{"unit_slot": 0, "objective_slot": 0, "objective_id": 51, "steps": 1}],
        )


if __name__ == "__main__":
    unittest.main()
