"""Focused tests for the single-process joint PPO training entry point."""

from __future__ import annotations

import csv
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from personal_train.joint_game_env import JointGameStep
from personal_train.joint_policy import JointPPOConfig, JointPPOPolicy
from personal_train.joint_rl_core import (
    JointSpaceSpec,
    SharedSensorConfig,
    SharedSensorState,
    UnitControlState,
    UnitPhase,
    build_joint_action_mask,
)
from personal_train.train_joint_ppo import (
    PERSONAL_ROOT,
    _append_round_records,
    _append_update_records,
    _checkpoint_config,
    _checkpoint_contract,
    _checkpoint_record,
    _collect_episode,
    _debug_max_steps_from_checkpoint_contract,
    _episode_planning_credit,
    _environment_contract,
    _fresh_policy_config,
    _game_config,
    _game_config_from_checkpoint_contract,
    _recovery_checkpoint,
    _resolve_run_directories,
    _resume_training_contract,
    _save_policy,
    _training_contract,
    _warm_start_contract_differences,
    parse_args,
    resolve_scenario,
)


class JointTrainerArgumentTests(unittest.TestCase):
    def test_defaults_build_consistent_policy_and_game_configs(self) -> None:
        args = parse_args(["--device", "cpu"])
        policy = _fresh_policy_config(args, observation_dim=37)
        game = _game_config(args, gamma=policy.gamma)

        self.assertEqual(policy.observation_dim, 37)
        self.assertEqual(policy.device, "cpu")
        self.assertEqual(game.objective_slots, 18)
        self.assertEqual(game.gamma, policy.gamma)
        self.assertEqual(game.sensor_capacity, 100)
        self.assertEqual(game.planning_team_weight, 0.7)
        self.assertEqual(game.planning_local_weight, 0.3)
        self.assertEqual(game.sensor_information_potential_scale, 0.015)
        self.assertEqual(args.rollout_episodes, 4)
        self.assertEqual(policy.learning_rate, 1e-4)
        self.assertEqual(policy.learning_rate_final, 1e-5)
        self.assertEqual(policy.learning_rate_decay_updates, 100)
        self.assertEqual(policy.target_kl, 0.01)
        self.assertEqual(policy.kl_guard_mode, "rollout_branch")
        self.assertEqual(policy.min_actor_decisions, 128)
        self.assertEqual(policy.training_phase, "joint")
        self.assertEqual(policy.motion_behavior_mode, "curriculum")
        self.assertEqual(policy.motion_curriculum_updates, 1)
        self.assertEqual(policy.motion_actor_start_update, 0)
        self.assertEqual(policy.sensor_actor_start_update, 0)
        self.assertEqual(policy.kl_hard_multiplier, 3.0)
        self.assertEqual(policy.minibatch_size, 128)
        self.assertTrue(game.strict_weapon_target_compatibility)
        self.assertTrue(game.allow_low_altitude_search_fallback)
        self.assertFalse(game.allow_low_altitude_search_replanning)
        self.assertEqual(game.retarget_min_dwell_steps, 60)
        self.assertEqual(game.retarget_decision_interval_steps, 60)
        self.assertEqual(game.motion_decision_interval_steps, 20)
        self.assertFalse(game.post_launch_motion_only)
        self.assertFalse(policy.post_launch_motion_only)
        self.assertTrue(game.terminate_on_all_objectives_destroyed)

        final20_game = _game_config(
            args,
            gamma=policy.gamma,
            required_objective_slots=48,
        )
        self.assertEqual(final20_game.objective_slots, 48)

    def test_all_joint_game_options_are_mapped(self) -> None:
        args = parse_args(
            [
                "--objective-slots",
                "21",
                "--max-track-age-steps",
                "44",
                "--max-speed-mps",
                "1234.5",
                "--sensor-capacity",
                "7",
                "--sensor-max-requests-per-step",
                "2",
                "--sensor-cooldown-steps",
                "3",
                "--training-phase",
                "motion_only",
                "--motion-behavior-mode",
                "learned",
                "--motion-curriculum-updates",
                "7",
                "--retarget-min-dwell-steps",
                "17",
                "--retarget-decision-interval-steps",
                "19",
                "--motion-decision-interval-steps",
                "23",
                "--post-launch-motion-only",
                "--allow-low-altitude-search-replanning",
                "--disable-weapon-target-compatibility",
                "--disable-low-altitude-search-fallback",
                "--official-reward-scale",
                "0.75",
                "--progress-potential-scale",
                "0.125",
                "--planning-team-weight",
                "0.6",
                "--planning-local-weight",
                "0.4",
                "--sensor-information-potential-scale",
                "0.02",
                "--gamma",
                "0.9",
                "--no-early-stop-on-completion",
            ]
        )
        config = _game_config(args, gamma=args.gamma)
        policy = _fresh_policy_config(args, observation_dim=37)
        self.assertEqual(config.objective_slots, 21)
        self.assertEqual(config.max_track_age_steps, 44)
        self.assertEqual(config.max_speed_mps, 1234.5)
        self.assertEqual(config.sensor_capacity, 7)
        self.assertEqual(config.sensor_max_requests_per_step, 2)
        self.assertEqual(config.sensor_cooldown_steps, 3)
        self.assertEqual(config.retarget_min_dwell_steps, 17)
        self.assertEqual(config.retarget_decision_interval_steps, 19)
        self.assertEqual(config.motion_decision_interval_steps, 23)
        self.assertTrue(config.post_launch_motion_only)
        self.assertFalse(config.strict_weapon_target_compatibility)
        self.assertFalse(config.allow_low_altitude_search_fallback)
        self.assertTrue(config.allow_low_altitude_search_replanning)
        self.assertEqual(policy.training_phase, "motion_only")
        self.assertTrue(policy.post_launch_motion_only)
        self.assertEqual(policy.motion_behavior_mode, "learned")
        self.assertEqual(policy.motion_curriculum_updates, 7)
        self.assertEqual(config.official_reward_scale, 0.75)
        self.assertEqual(config.progress_potential_scale, 0.125)
        self.assertEqual(config.planning_team_weight, 0.6)
        self.assertEqual(config.planning_local_weight, 0.4)
        self.assertEqual(config.sensor_information_potential_scale, 0.02)
        self.assertEqual(config.gamma, 0.9)
        self.assertFalse(config.terminate_on_all_objectives_destroyed)

    def test_invalid_limits_and_half_managed_outputs_are_rejected(self) -> None:
        for argv in (
            ["--rounds", "0"],
            ["--rollout-episodes", "0"],
            ["--debug-max-steps", "0"],
            ["--sensor-capacity", "-1"],
            ["--min-actor-decisions", "-1"],
            ["--motion-actor-start-update", "-1"],
            ["--sensor-actor-start-update", "-1"],
            ["--motion-curriculum-updates", "0"],
            ["--retarget-min-dwell-steps", "-1"],
            ["--retarget-decision-interval-steps", "0"],
            ["--motion-decision-interval-steps", "0"],
            ["--training-phase", "planner_sensor"],
            ["--kl-hard-multiplier", "1.5"],
            ["--kl-guard-mode", "unknown"],
            ["--gamma", "1.1"],
            ["--planning-team-weight", "0.8", "--planning-local-weight", "0.3"],
            ["--sensor-information-potential-scale", "-0.1"],
            ["--blue-policy", "  "],
            ["--placement-log-std-min", "1", "--placement-log-std-max", "1"],
            ["--result-dir", "results/only"],
            ["--resume", "one.pt", "--init-from", "two.pt"],
            ["--devi", "cpu"],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(argv)

    def test_scenario_resolver_accepts_an_explicit_json(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "scenario.json"
            path.write_text("{}", encoding="utf-8")
            self.assertEqual(resolve_scenario(str(path), Path(raw)), path.resolve())

    def test_output_paths_are_scoped_and_non_nested(self) -> None:
        _, result, model = _resolve_run_directories(
            scenario_id="E01",
            run_id="unit",
            timestamp="20260101_000000_000000",
            result_dir=None,
            model_dir=None,
            invocation_cwd=PERSONAL_ROOT,
        )
        self.assertEqual(result.parent, PERSONAL_ROOT / "results")
        self.assertEqual(model.parent, PERSONAL_ROOT / "models")
        self.assertNotEqual(result, model)

        with self.assertRaises(ValueError):
            _resolve_run_directories(
                scenario_id="E01",
                run_id=None,
                timestamp="x",
                result_dir=PERSONAL_ROOT / "scratch" / "run",
                model_dir=PERSONAL_ROOT / "scratch" / "run" / "models",
                invocation_cwd=PERSONAL_ROOT,
            )
        with self.assertRaises(ValueError):
            _resolve_run_directories(
                scenario_id="E01",
                run_id=None,
                timestamp="x",
                result_dir=Path("/tmp/joint-result"),
                model_dir=PERSONAL_ROOT / "models" / "joint-model",
                invocation_cwd=PERSONAL_ROOT,
            )


class JointTrainerPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def test_recovery_checkpoint_prefers_new_latest_then_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            model = root / "model"
            model.mkdir()
            resume = root / "source-latest.pt"
            init_from = root / "source-best.pt"
            resume.touch()
            init_from.touch()

            self.assertEqual(
                _recovery_checkpoint(
                    model, resume_path=resume, init_from_path=init_from
                ),
                (resume, "resume"),
            )
            latest = model / "latest.pt"
            latest.touch()
            self.assertEqual(
                _recovery_checkpoint(
                    model, resume_path=resume, init_from_path=init_from
                ),
                (latest, "resume"),
            )
            latest.unlink()
            resume.unlink()
            best = model / "best.pt"
            best.touch()
            self.assertEqual(
                _recovery_checkpoint(
                    model, resume_path=None, init_from_path=init_from
                ),
                (best, "init_from"),
            )

    def test_round_csv_has_one_header_and_jsonl_keeps_full_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = {
                "round": 1,
                "score": 2.5,
                "steps": 3,
                "policy_loss": 0.1,
                "custom_metric": 11.0,
            }
            second = {
                "round": 2,
                "score": 4.5,
                "steps": 5,
                "policy_loss": 0.2,
                "custom_metric": 12.0,
            }
            _append_round_records(root, first)
            _append_round_records(root, second)

            with (root / "rounds.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["round"] for row in rows], ["1", "2"])
            self.assertNotIn("custom_metric", rows[0])
            json_rows = [
                json.loads(line)
                for line in (root / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["custom_metric"] for row in json_rows], [11.0, 12.0])

    def test_update_records_are_separate_from_episode_records(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _append_round_records(root, {"round": 1, "score": 2.0})
            _append_update_records(
                root,
                {
                    "update_count": 1,
                    "round_start": 1,
                    "round_end": 4,
                    "episodes": 4,
                    "transitions": 123,
                    "policy_loss": 0.25,
                },
            )

            episode_rows = (root / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
            update_rows = (root / "updates.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(episode_rows), 1)
            self.assertEqual(len(update_rows), 1)
            self.assertEqual(json.loads(update_rows[0])["episodes"], 4)
            with (root / "updates.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["round_end"], "4")

    def test_policy_checkpoint_is_validated_and_describes_config(self) -> None:
        space = JointSpaceSpec(unit_count=2, objective_count=3)
        policy = JointPPOPolicy(
            space,
            JointPPOConfig(
                observation_dim=5,
                hidden_dim=16,
                update_epochs=1,
                minibatch_size=1,
                device="cpu",
            ),
        )
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "policy.pt"
            contract = {
                "version": 1,
                "scenario_id": "TEST",
                "feature_names": ("a", "b"),
                "max_steps": 4,
            }
            trainer_contract = _training_contract(
                seed=7, blue_policy="test_blue", rollout_episodes=4
            )
            _save_policy(
                policy,
                path,
                environment_contract=contract,
                trainer_contract=trainer_contract,
                checkpoint_role="latest",
                resume_safe=True,
            )
            config = _checkpoint_config(path)
            record = _checkpoint_record(path, expected_contract=contract)

            self.assertEqual(config["observation_dim"], 5)
            self.assertEqual(record["file"], "policy.pt")
            self.assertGreater(record["size_bytes"], 0)
            self.assertEqual(len(record["sha256"]), 64)
            self.assertEqual(record["checkpoint_role"], "latest")
            self.assertTrue(record["resume_safe"])
            self.assertEqual(_checkpoint_contract(path), contract)
            self.assertEqual(_resume_training_contract(path), trainer_contract)

    def test_resume_rejects_checkpoint_not_marked_as_safe_boundary(self) -> None:
        space = JointSpaceSpec(unit_count=1, objective_count=1)
        policy = JointPPOPolicy(
            space,
            JointPPOConfig(observation_dim=2, hidden_dim=8, device="cpu"),
        )
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "best.pt"
            _save_policy(
                policy,
                path,
                environment_contract={"version": 2},
                trainer_contract=_training_contract(
                    seed=1, blue_policy="test_blue", rollout_episodes=4
                ),
                checkpoint_role="best_behavior",
                resume_safe=False,
            )
            with self.assertRaisesRegex(ValueError, "--init-from"):
                _resume_training_contract(path)

    def test_debug_horizon_is_restored_from_checkpoint_contract(self) -> None:
        self.assertEqual(
            _debug_max_steps_from_checkpoint_contract(
                {"debug_horizon": True, "max_steps": 17}
            ),
            17,
        )
        self.assertIsNone(
            _debug_max_steps_from_checkpoint_contract(
                {"debug_horizon": False, "max_steps": 1200}
            )
        )
        with self.assertRaisesRegex(ValueError, "horizon"):
            _debug_max_steps_from_checkpoint_contract(
                {"debug_horizon": True, "max_steps": 0}
            )

    def test_environment_contract_covers_observation_scaling(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            scenario = Path(raw) / "scenario.json"
            scenario.write_text("{}", encoding="utf-8")
            environment = SimpleNamespace(
                tracker=SimpleNamespace(
                    sensor_state=SimpleNamespace(
                        config=SimpleNamespace(
                            capacity=100,
                            max_requests_per_step=1,
                            cooldown_steps=0,
                        )
                    )
                ),
                objective_ids=(51, None),
                unit_ids=(2,),
                unit_types=(21000,),
                space=JointSpaceSpec(unit_count=1, objective_count=2),
                observation_dim=7,
                encoder=SimpleNamespace(
                    feature_names=("a", "b"),
                    config=SimpleNamespace(
                        detected_threat_count_normalizer=148.0,
                    ),
                ),
                max_steps=4,
                is_debug_horizon=True,
                sensor_backend="per_unit",
                sensor_backend_capacity_per_unit=100,
                sensor_backend_capacity_team=100,
                sensor_backend_active_minutes=None,
                sensor_information_source="known_objective_value",
                reward_policy=SimpleNamespace(
                    objective_weights=((51, 5.0),),
                    objective_initial_health=((51, 100.0),),
                ),
                config=SimpleNamespace(
                    max_track_age_steps=300,
                    max_speed_mps=3000.0,
                    official_reward_scale=1.0,
                    progress_potential_scale=0.05,
                    planning_team_weight=0.7,
                    planning_local_weight=0.3,
                    sensor_information_potential_scale=0.015,
                    gamma=0.995,
                    terminate_on_all_objectives_destroyed=True,
                    strict_weapon_target_compatibility=False,
                    allow_low_altitude_search_fallback=True,
                    allow_low_altitude_search_replanning=True,
                    retarget_min_dwell_steps=0,
                    retarget_decision_interval_steps=1,
                    motion_decision_interval_steps=1,
                ),
            )
            contract = _environment_contract(
                environment, scenario_id="TEST", scenario_path=scenario
            )

        self.assertEqual(contract["version"], 3)
        self.assertEqual(
            contract["observation"],
            {"max_track_age_steps": 300, "max_speed_mps": 3000.0},
        )
        self.assertNotIn("control", contract)
        self.assertEqual(
            contract["official_scoring"],
            {
                "objective_weights": ((51, 5.0),),
                "objective_initial_health": ((51, 100.0),),
            },
        )
        restored = _game_config_from_checkpoint_contract(contract)
        self.assertEqual(restored.objective_slots, 2)
        self.assertEqual(restored.sensor_capacity, 100)
        self.assertEqual(restored.max_track_age_steps, 300)
        self.assertEqual(restored.planning_team_weight, 0.7)
        self.assertEqual(restored.planning_local_weight, 0.3)
        self.assertEqual(restored.sensor_information_potential_scale, 0.015)
        self.assertFalse(restored.strict_weapon_target_compatibility)
        self.assertTrue(restored.allow_low_altitude_search_replanning)
        self.assertEqual(restored.motion_decision_interval_steps, 1)
        self.assertFalse(restored.post_launch_motion_only)
        self.assertEqual(restored.retarget_min_dwell_steps, 0)
        self.assertEqual(restored.retarget_decision_interval_steps, 1)

        controlled_environment = SimpleNamespace(
            **{
                **environment.__dict__,
                "config": SimpleNamespace(
                    **{
                        **environment.config.__dict__,
                        "strict_weapon_target_compatibility": True,
                        "allow_low_altitude_search_fallback": True,
                        "allow_low_altitude_search_replanning": False,
                        "retarget_min_dwell_steps": 60,
                        "retarget_decision_interval_steps": 60,
                        "motion_decision_interval_steps": 20,
                    }
                ),
            }
        )
        with tempfile.TemporaryDirectory() as controlled_raw:
            controlled_scenario = Path(controlled_raw) / "scenario.json"
            controlled_scenario.write_text("{}", encoding="utf-8")
            controlled_contract = _environment_contract(
                controlled_environment,
                scenario_id="TEST",
                scenario_path=controlled_scenario,
            )
        self.assertEqual(
            controlled_contract["control"],
            {
                "strict_weapon_target_compatibility": True,
                "allow_low_altitude_search_fallback": True,
                "allow_low_altitude_search_replanning": False,
                "retarget_min_dwell_steps": 60,
                "retarget_decision_interval_steps": 60,
                "motion_decision_interval_steps": 20,
            },
        )
        controlled_restored = _game_config_from_checkpoint_contract(
            controlled_contract
        )
        self.assertTrue(controlled_restored.strict_weapon_target_compatibility)
        self.assertFalse(controlled_restored.allow_low_altitude_search_replanning)
        self.assertEqual(controlled_restored.motion_decision_interval_steps, 20)
        self.assertFalse(controlled_restored.post_launch_motion_only)
        self.assertEqual(controlled_restored.retarget_min_dwell_steps, 60)
        self.assertEqual(controlled_restored.retarget_decision_interval_steps, 60)

        post_launch_environment = SimpleNamespace(
            **{
                **environment.__dict__,
                "config": SimpleNamespace(
                    **{
                        **environment.config.__dict__,
                        "post_launch_motion_only": True,
                    }
                ),
            }
        )
        with tempfile.TemporaryDirectory() as post_launch_raw:
            post_launch_scenario = Path(post_launch_raw) / "scenario.json"
            post_launch_scenario.write_text("{}", encoding="utf-8")
            post_launch_contract = _environment_contract(
                post_launch_environment,
                scenario_id="TEST",
                scenario_path=post_launch_scenario,
            )
        self.assertTrue(post_launch_contract["control"]["post_launch_motion_only"])
        self.assertTrue(
            _game_config_from_checkpoint_contract(
                post_launch_contract
            ).post_launch_motion_only
        )
        self.assertEqual(
            _warm_start_contract_differences(contract, post_launch_contract),
            (),
        )

        tuned = dict(contract)
        tuned["sensor"] = {**contract["sensor"], "coordinated_team_capacity": 7}
        tuned["reward"] = {**contract["reward"], "progress_potential_scale": 0.0}
        self.assertEqual(_warm_start_contract_differences(contract, tuned), ())
        incompatible = {**tuned, "observation_dim": 8}
        self.assertEqual(
            _warm_start_contract_differences(contract, incompatible),
            ("observation_dim",),
        )

        global_environment = SimpleNamespace(
            **{
                **environment.__dict__,
                "sensor_backend": "team_global",
                "sensor_backend_capacity_per_unit": None,
                "sensor_backend_capacity_team": 100,
                "sensor_backend_active_minutes": 3.0,
                "sensor_information_source": "fresh_interceptor_tracks",
            }
        )
        with tempfile.TemporaryDirectory() as global_raw:
            global_scenario = Path(global_raw) / "scenario.json"
            global_scenario.write_text("{}", encoding="utf-8")
            global_contract = _environment_contract(
                global_environment,
                scenario_id="TEST",
                scenario_path=global_scenario,
            )
        self.assertEqual(global_contract["version"], 4)
        self.assertEqual(
            global_contract["sensor"],
            {
                "backend": "team_global",
                "backend_capacity_team": 100,
                "backend_active_minutes": 3.0,
                "coordinated_team_capacity": 100,
                "max_requests_per_step": 1,
                "effective_max_requests_per_step": 1,
                "cooldown_steps": 0,
            },
        )
        self.assertEqual(
            global_contract["reward"]["sensor_information_source"],
            "fresh_interceptor_tracks",
        )
        self.assertEqual(
            global_contract["observation"]["detected_threat_count_normalizer"],
            148.0,
        )
        self.assertIn(
            "version",
            _warm_start_contract_differences(contract, global_contract),
        )


class JointTrainerCollectionTests(unittest.TestCase):
    def test_one_step_truncation_is_a_complete_training_transition(self) -> None:
        torch.set_num_threads(1)
        space = JointSpaceSpec(unit_count=1, objective_count=1)
        states = (UnitControlState(slot=0, phase=UnitPhase.STAGED),)
        assignment_states = (
            UnitControlState(
                slot=0,
                phase=UnitPhase.ACTIVE,
                activated_step=0,
                current_objective_slot=0,
            ),
        )
        sensor_state = SharedSensorState.initial(SharedSensorConfig(capacity=0))
        mask = build_joint_action_mask(
            space,
            states,
            objective_valid=(True,),
            sensor_state=sensor_state,
            step=0,
        )
        observations = (np.asarray((0.0, 0.5, -0.5, 1.0), dtype=np.float32),)
        policy = JointPPOPolicy(
            space,
            JointPPOConfig(
                observation_dim=4,
                hidden_dim=16,
                update_epochs=1,
                minibatch_size=1,
                seed=4,
                device="cpu",
            ),
        )
        with torch.no_grad():
            policy.network.activation_head.weight.zero_()
            policy.network.activation_head.bias.copy_(
                torch.tensor((-100.0, 100.0))
            )

        class FakeEnvironment:
            observation_dim = 4
            max_steps = 1
            current_step = 0
            launch_count = 0
            sensor_request_count = 0
            is_debug_horizon = True
            objective_ids = (51,)
            unit_ids = (2,)
            unit_types = (21000,)

            def __init__(self) -> None:
                self.space = space
                self.states = states
                self.config = SimpleNamespace(
                    planning_team_weight=0.7,
                    planning_local_weight=0.3,
                )
                self.reward_policy = SimpleNamespace(
                    objective_ids=(51,),
                    objective_weights=((51, 1.0),),
                    objective_initial_health=((51, 1.0),),
                )

            def reset(self):
                self.current_step = 0
                return observations

            def action_mask(self):
                return mask

            def assignment_is_damage_compatible(self, _unit_slot, _objective_slot):
                return True

            def step(self, _action):
                self.current_step = 1
                return JointGameStep(
                    observations=observations,
                    states=states,
                    rewards=(0.25,),
                    team_reward=0.5,
                    terminated=(False,),
                    truncated=(True,),
                    team_terminated=False,
                    team_truncated=True,
                    done=True,
                    score=7.0,
                    raw_observation={"entities": {51: {"health": 0.93}}},
                    submitted_commands=0,
                    accepted_activations=(0,),
                    accepted_sensor_requests=(),
                    sensor_reward=0.75,
                    assignment_states=assignment_states,
                )

        episode = _collect_episode(
            policy,
            FakeEnvironment(),  # type: ignore[arg-type]
            round_index=1,
            progress_every=10,
        )
        self.assertEqual(len(episode.buffer), 1)
        self.assertEqual(episode.steps, 1)
        self.assertEqual(episode.score, 7.0)
        self.assertEqual(episode.team_return, 0.5)
        self.assertEqual(episode.unit_reward_sum, 0.25)
        self.assertEqual(episode.motion_return, 0.25)
        self.assertEqual(episode.sensor_return, 0.75)
        self.assertAlmostEqual(episode.planning_rewards[0], 0.07)
        self.assertEqual(
            episode.planning_credit["assignment_duration"],
            [{"unit_slot": 0, "objective_slot": 0, "objective_id": 51, "steps": 1}],
        )
        self.assertEqual(episode.termination_reason, "debug_horizon")
        transition = episode.buffer.items[0]
        self.assertEqual(transition.truncated, (True,))
        self.assertTrue(transition.team_truncated)
        self.assertAlmostEqual(transition.plan_rewards[0], 0.07)
        self.assertEqual(transition.motion_rewards, (0.25,))
        self.assertEqual(transition.sensor_reward, 0.75)

    def test_planning_credit_is_conserved_and_excludes_never_launched_units(self):
        environment = SimpleNamespace(
            space=JointSpaceSpec(unit_count=3, objective_count=2),
            objective_ids=(51, 52),
            reward_policy=SimpleNamespace(
                objective_ids=(51, 52),
                objective_weights=((51, 5.0), (52, 1.0)),
                objective_initial_health=((51, 100.0), (52, 100.0)),
            ),
            config=SimpleNamespace(
                planning_team_weight=0.7,
                planning_local_weight=0.3,
            ),
        )
        result = _episode_planning_credit(
            environment,  # type: ignore[arg-type]
            final_observation={
                "entities": {
                    51: {"health": 50.0},
                    52: {"health": 0.0},
                }
            },
            final_score=100.0 * 7.0 / 12.0,
            eligible_units=(True, True, False),
            assignment_duration=np.asarray(
                ((3, 0), (1, 2), (0, 0)), dtype=np.int64
            ),
        )

        diagnostics = result.diagnostics
        self.assertAlmostEqual(
            diagnostics["total_objective_contribution"], 7.0 / 12.0
        )
        self.assertAlmostEqual(
            diagnostics["total_local_credit_allocated"], 7.0 / 12.0
        )
        self.assertEqual(diagnostics["total_local_credit_unallocated"], 0.0)
        self.assertAlmostEqual(diagnostics["local_credit_by_unit"][0], 5.0 / 16.0)
        self.assertAlmostEqual(diagnostics["local_credit_by_unit"][1], 13.0 / 48.0)
        self.assertEqual(diagnostics["local_credit_learning_scale"], 2.0)
        self.assertEqual(diagnostics["local_credit_learning_clipped_units"], 0)
        self.assertAlmostEqual(
            diagnostics["learning_local_credit_by_unit"][0], 2.0 * 5.0 / 16.0
        )
        self.assertTrue(
            all(
                0.0 <= value <= 1.0
                for value in diagnostics["learning_local_credit_by_unit"]
            )
        )
        self.assertEqual(result.rewards_by_unit[2], 0.0)
        self.assertAlmostEqual(
            result.rewards_by_unit[0],
            0.7 * 7.0 / 12.0 + 0.3 * 2.0 * 5.0 / 16.0,
        )

    def test_learning_local_credit_is_bounded_after_unit_count_scaling(self):
        environment = SimpleNamespace(
            space=JointSpaceSpec(unit_count=4, objective_count=1),
            objective_ids=(51,),
            reward_policy=SimpleNamespace(
                objective_ids=(51,),
                objective_weights=((51, 1.0),),
                objective_initial_health=((51, 100.0),),
            ),
            config=SimpleNamespace(
                planning_team_weight=0.7,
                planning_local_weight=0.3,
            ),
        )
        result = _episode_planning_credit(
            environment,  # type: ignore[arg-type]
            final_observation={"entities": {51: {"health": 0.0}}},
            final_score=100.0,
            eligible_units=(True, True, True, True),
            assignment_duration=np.asarray(((5,), (0,), (0,), (0,)), dtype=np.int64),
        )

        diagnostics = result.diagnostics
        self.assertEqual(diagnostics["local_credit_by_unit"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(
            diagnostics["learning_local_credit_by_unit"],
            [1.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(diagnostics["local_credit_learning_clipped_units"], 1)
        self.assertEqual(result.rewards_by_unit, (1.0, 0.7, 0.7, 0.7))


if __name__ == "__main__":
    unittest.main()
