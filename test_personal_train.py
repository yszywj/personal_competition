"""Focused regression tests for the training-side fixes."""

from __future__ import annotations

import csv
import json
import inspect
import math
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from personal_train.bootstrap import PERSONAL_ROOT
from personal_train.legal_observation import CorrectedR9ObservationEncoder
from personal_train.personal_agent import PersonalR9PPOAttackAgent, ResetAwareAgentManager
from personal_train.personal_commander import (
    DynamicDetectedTargetTrackFusion,
    PersonalR9Commander,
)
from personal_train.personal_env import PersonalR9TrainingEnv, R9RewardConfig
from personal_train.ppo_policy import PPOConfig, PPOSharedPolicy
from personal_train.reporting import TrainingReporter
from personal_train.train_r9_ppo import (
    CHECKPOINT_SCHEMA,
    _policy_config,
    _resolve_run_directories,
    _run_directories,
    _resume_loads_optimizer,
    _resume_mode,
    _save_policy,
    parse_args,
)
from policies.red import initial_targets_from_observation
from policies.red.learning.red_policy import PolicyTransition
from scenarios.cases import RewardPolicy, RewardTracker


INITIAL_TIME = 1_783_391_450_000


class OutputDirectoryTests(unittest.TestCase):
    def test_default_run_uses_one_timestamped_directory_level(self):
        timestamp = "20260905_144652_664856"
        run_name, result_dir, model_dir = _run_directories(
            "E01", None, timestamp
        )

        self.assertEqual(run_name, "e01_r9_ppo_20260905_144652_664856")
        self.assertEqual(result_dir.parent, PERSONAL_ROOT / "results")
        self.assertEqual(model_dir.parent, PERSONAL_ROOT / "models")
        self.assertEqual(result_dir.name, model_dir.name)
        self.assertNotIn("gae", run_name)
        self.assertNotIn("v2", run_name)

    def test_custom_run_label_is_preserved_without_extra_nesting(self):
        timestamp = "20260905_150000_000001"
        run_name, result_dir, model_dir = _run_directories(
            "E01", "ablation", timestamp
        )

        self.assertEqual(run_name, "ablation_20260905_150000_000001")
        self.assertEqual(result_dir.parts[-2:], ("results", run_name))
        self.assertEqual(model_dir.parts[-2:], ("models", run_name))

    def test_batch_launcher_can_select_exact_nonexistent_leaf_directories(self):
        with tempfile.TemporaryDirectory(dir=PERSONAL_ROOT) as directory:
            root = Path(directory)
            run_name, result_dir, model_dir = _resolve_run_directories(
                scenario_id="M02",
                run_id="m02",
                timestamp="ignored",
                result_dir=root / "results" / "batch" / "M02",
                model_dir=root / "models" / "batch" / "M02",
                invocation_cwd=root,
            )

            self.assertEqual(run_name, "m02")
            self.assertEqual(result_dir, root / "results" / "batch" / "M02")
            self.assertEqual(model_dir, root / "models" / "batch" / "M02")

    def test_exact_output_arguments_must_be_paired(self):
        with self.assertRaises(SystemExit):
            parse_args(["--result-dir", "results/only"])

    def test_exact_output_directories_cannot_be_nested(self):
        with tempfile.TemporaryDirectory(dir=PERSONAL_ROOT) as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                _resolve_run_directories(
                    scenario_id="E01",
                    run_id=None,
                    timestamp="ignored",
                    result_dir=root / "run" / "results",
                    model_dir=root / "run",
                    invocation_cwd=root,
                )


def _initial_targets():
    return {
        "step": 0,
        "entities": {
            51: {
                "nameChn": "目标_51",
                "position": {"lon": 0.0, "lat": 1.0, "alt": 0.0},
                "health": 32.0,
                "type": 9400,
                "side": 1,
                "detectInfo": {},
                "commRangeInfo": [],
            }
        },
    }


def _observation(step: int, ecf, *, tracks=None):
    return {
        "step": step,
        "entity_id": 2,
        "agent_id": 1,
        "self": {
            "nameChn": "高性能飞行器_2",
            "position": {"lon": 0.0, "lat": 0.0, "alt": 0.0},
            "pos_ecf": {"x": ecf[0], "y": ecf[1], "z": ecf[2]},
            "health": 1.0,
            "isVisible": True,
            "type": 21000,
            "side": 0,
            "detectInfo": tracks or {},
            "commRangeInfo": [],
        },
    }


class ObservationCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.encoder = CorrectedR9ObservationEncoder(
            _initial_targets(),
            max_steps=100,
            agent_id=1,
            team_size=164,
            initial_sim_time_ms=INITIAL_TIME,
            sim_step_ms=1000,
        )

    def test_motion_bearing_track_age_and_interceptor_count_keep_90d_contract(self):
        self.encoder.observe_frame(_observation(4, (6_371_000.0, 0.0, 0.0)))
        tracks = {
            51: {
                "time": INITIAL_TIME + 2_000,
                "entity_id": 51,
                "entity_type": 9400,
                "nameChn": "目标_51",
                "lla": {"x": 0.0, "y": 1.0, "z": 0.0},
            },
            999: {
                "time": INITIAL_TIME + 5_000,
                "entity_id": 999,
                "entity_type": 24000,
                # Deliberately not prefixed with 标6.
                "nameChn": "拦截弹_999",
                "lla": {"x": 0.1, "y": 0.1, "z": 1000.0},
            },
        }
        current = _observation(5, (6_371_000.0, 100.0, 100.0), tracks=tracks)
        encoded = self.encoder.encode(
            current,
            launched=True,
            launch_step=0,
            satellite_used=False,
            maneuver_state=2,
            current_target_index=0,
            task_context=(1.0, 0.0, 0.0, 0.2, 0.3),
        )

        self.assertEqual(encoded.shape, (90,))
        self.assertAlmostEqual(float(encoded[13]), math.sqrt(20_000.0) / 1500.0, places=5)
        self.assertAlmostEqual(float(encoded[14]), 0.0, places=5)
        self.assertAlmostEqual(float(encoded[15]), math.sqrt(0.5), places=5)
        self.assertAlmostEqual(float(encoded[16]), math.sqrt(0.5), places=5)
        self.assertAlmostEqual(float(encoded[27]), 0.03, places=5)
        self.assertAlmostEqual(float(encoded[30]), 1.0, places=5)
        self.assertAlmostEqual(float(encoded[31]), -math.sqrt(0.5), places=5)
        self.assertAlmostEqual(float(encoded[32]), math.sqrt(0.5), places=5)
        self.assertAlmostEqual(float(encoded[82]), 1.0 / 200.0, places=6)
        self.assertAlmostEqual(float(encoded[11]), 1.0, places=6)
        np.testing.assert_allclose(encoded[-5:], (1.0, 0.0, 0.0, 0.2, 0.3))

        # Encoding the same s' again must retain, not zero, its velocity.
        repeated = self.encoder.encode(
            current,
            launched=True,
            launch_step=0,
            satellite_used=False,
            maneuver_state=2,
            current_target_index=0,
            task_context=(1.0, 0.0, 0.0, 0.2, 0.3),
        )
        self.assertAlmostEqual(float(repeated[13]), float(encoded[13]), places=7)

    def test_stale_timestamp_keeps_the_first_legal_track_snapshot(self):
        track = {
            "time": INITIAL_TIME + 1_000,
            "entity_id": 51,
            "entity_type": 9400,
            "nameChn": "目标_51",
            "lla": {"x": 1.0, "y": 2.0, "z": 3.0},
        }
        first = self.encoder._prepare_observation(
            _observation(1, (6_371_000.0, 0.0, 0.0), tracks={51: track})
        )["self"]["detectInfo"][51]

        # The native SDK can mutate a Vector3d in place without advancing the
        # report timestamp.  That must not reveal an unreported new position.
        track["lla"]["x"] = 9.0
        second = self.encoder._prepare_observation(
            _observation(2, (6_371_000.0, 0.0, 0.0), tracks={51: track})
        )["self"]["detectInfo"][51]
        self.assertEqual(first.lla.x, 1.0)
        self.assertEqual(second.lla.x, 1.0)


class _DummyPolicy:
    training = True

    def __init__(self):
        self.transitions = []

    def select_action(self, observation, action_mask):
        return 2

    def observe(self, transition):
        self.transitions.append(transition)

    def get_global_state_context(self):
        return None, None


class _DummyCommander:
    targets = ()

    def target_id_for(self, entity_id):
        return 51

    def learning_task_context(self, entity_id):
        return (1.0, 0.0, 0.0, 0.0, 0.1)


class AgentTransitionTests(unittest.TestCase):
    def test_maneuver_state_and_next_target_index_are_present(self):
        policy = _DummyPolicy()
        agent = PersonalR9PPOAttackAgent(
            1,
            2,
            _initial_targets(),
            _DummyCommander(),
            policy,
            learning_max_steps=100,
            team_size=164,
            initial_sim_time_ms=INITIAL_TIME,
            sim_step_ms=1000,
        )
        agent.launch_step = 0
        state = _observation(0, (6_371_000.0, 0.0, 0.0))
        agent.set_observation(state)
        actions = []
        agent._set_acc_z_learning(actions, state)
        self.assertEqual(agent.set_acc_z_z, 2)
        self.assertEqual(actions[0][2], 1.0)

        next_state = _observation(1, (6_371_000.0, 100.0, 0.0))
        agent.record_step(next_state, None, 0.25, {"done": False})
        self.assertEqual(len(policy.transitions), 1)
        transition = policy.transitions[0]
        self.assertAlmostEqual(float(transition.next_observation[11]), 1.0)
        self.assertAlmostEqual(float(transition.next_observation[30]), 1.0)


class ResetTests(unittest.TestCase):
    def test_shared_commander_resets_once(self):
        class Commander:
            def __init__(self):
                self.calls = 0

            def reset(self):
                self.calls += 1

        class Agent:
            def __init__(self, agent_id, entity_id, commander):
                self.agent_id = agent_id
                self.entity_id = entity_id
                self.commander = commander
                self.reset_calls = 0

            def get_entity_id(self):
                return self.entity_id

            def reset(self):
                self.reset_calls += 1

        commander = Commander()
        first, second = Agent(1, 11, commander), Agent(2, 12, commander)
        manager = ResetAwareAgentManager()
        manager.register_agent(first)
        manager.register_agent(second)
        manager.reset_all()
        self.assertEqual(first.reset_calls, 1)
        self.assertEqual(second.reset_calls, 1)
        self.assertEqual(commander.calls, 1)


class RewardTests(unittest.TestCase):
    def test_real_health_drop_is_split_by_nominal_damage_and_success_waives_loss(self):
        environment = PersonalR9TrainingEnv.__new__(PersonalR9TrainingEnv)
        environment.reward_config = R9RewardConfig()
        environment.reward_policy = SimpleNamespace(
            objective_ids=(51,),
            objective_weights=((51, 3.0),),
        )
        environment._initial_health_by_objective = {51: 32.0}
        environment._episode_reward_components = defaultdict(float)
        environment._relation_offsets = {51: 0}
        environment._successful_sources = set()
        environment._settled_failures = set()
        environment.engine = SimpleNamespace(
            simulator_factory=SimpleNamespace(
                target_hit_relation={
                    51: [
                        {"id": 2, "entity_type": 21000},
                        {"id": 3, "entity_type": 21001},
                    ]
                }
            )
        )

        high = PersonalR9PPOAttackAgent.__new__(PersonalR9PPOAttackAgent)
        high.agent_id, high.entity_id = 1, 2
        medium = PersonalR9PPOAttackAgent.__new__(PersonalR9PPOAttackAgent)
        medium.agent_id, medium.entity_id = 2, 3
        previous = {
            "entities": {
                2: {"health": 1.0, "type": 21000},
                3: {"health": 1.0, "type": 21001},
                51: {"health": 32.0, "type": 9400},
            }
        }
        current = {
            "entities": {
                2: {"health": 0.0, "type": 21000},
                3: {"health": 1.0, "type": 21001},
                51: {"health": 12.0, "type": 9400},
            }
        }
        rewards = {1: 0.0, 2: 0.0}
        agents = {2: high, 3: medium}
        successful = environment._credit_actual_damage(
            rewards,
            agents,
            {2, 3},
            previous,
            current,
        )
        environment._credit_losses(
            rewards,
            agents,
            {2, 3},
            previous,
            current,
            successful,
        )

        # Sparse shaping total: 1 * weight 3 * (20/32) = 1.875;
        # simultaneous sources are still split H:M = 16:4.
        self.assertAlmostEqual(rewards[1], 1.5)
        self.assertAlmostEqual(rewards[2], 0.375)
        self.assertNotIn("unsuccessful_loss", environment._episode_reward_components)

    def test_first_destruction_team_reward_matches_official_score_increment(self):
        policy = RewardPolicy(
            scenario_id="E01",
            objective_ids=(51, 52),
            max_steps=1200,
            scenario_path=Path("unused.json"),
            objective_weights=((51, 3.0), (52, 10.0)),
        )
        environment = PersonalR9TrainingEnv.__new__(PersonalR9TrainingEnv)
        environment.reward_config = R9RewardConfig()
        environment.reward_policy = policy
        environment._reward_tracker = RewardTracker(policy)
        environment._seen_destroyed = set()
        environment._episode_reward_components = defaultdict(float)
        environment._raw_official_score_delta = 0.0
        environment.learning_team_size = 4
        environment.current_step = 600
        environment.max_steps = 1200
        agents = [SimpleNamespace(agent_id=1), SimpleNamespace(agent_id=2)]
        rewards = {1: 0.0, 2: 0.0}
        current = {
            "entities": {
                51: {"health": 0.0},
                52: {"health": 1.0},
            }
        }

        environment._credit_new_destructions(rewards, agents, current)

        expected = 100.0 * (3.0 / 13.0) * (0.8 + 0.2 * 0.5)
        self.assertAlmostEqual(rewards[1], expected / 4.0)
        self.assertAlmostEqual(rewards[2], expected / 4.0)
        self.assertAlmostEqual(
            environment._episode_reward_components["official_score_delta"],
            expected / 2.0,
        )
        self.assertAlmostEqual(environment._raw_official_score_delta, expected)
        # A repeated observation cannot pay the same destruction twice.
        environment._credit_new_destructions(rewards, agents, current)
        self.assertAlmostEqual(rewards[1], expected / 4.0)
        self.assertAlmostEqual(environment._raw_official_score_delta, expected)

    def test_progress_is_bounded_potential_and_target_switch_reanchors(self):
        class Commander:
            def __init__(self):
                self.current = 51
                self.targets = (
                    SimpleNamespace(
                        entity_id=51,
                        position=SimpleNamespace(lon=0.0, lat=0.0),
                    ),
                    SimpleNamespace(
                        entity_id=52,
                        position=SimpleNamespace(lon=3.0, lat=0.0),
                    ),
                )

            def target_id_for(self, _entity_id):
                return self.current

        environment = PersonalR9TrainingEnv.__new__(PersonalR9TrainingEnv)
        environment.reward_config = R9RewardConfig()
        environment._progress_state = {}
        environment._episode_reward_components = defaultdict(float)
        environment.current_step = 0
        environment.max_steps = 1200
        commander = Commander()
        agent = PersonalR9PPOAttackAgent.__new__(PersonalR9PPOAttackAgent)
        agent.agent_id, agent.entity_id, agent.commander = 1, 2, commander
        rewards = {1: 0.0}

        def frame(lon):
            return {
                "entities": {
                    2: {
                        "health": 1.0,
                        "position": {"lon": lon, "lat": 0.0},
                    }
                }
            }

        environment._credit_progress(rewards, agent, frame(-1.0))
        environment._credit_progress(rewards, agent, frame(-0.5))
        self.assertGreater(rewards[1], 0.0)
        self.assertLessEqual(rewards[1], environment.reward_config.progress_potential_scale)
        environment._credit_progress(rewards, agent, frame(-1.0))
        # Discount-correct potential shaping makes a delayed out-and-back
        # slightly negative rather than harvestable.
        self.assertLessEqual(rewards[1], 0.0)

        environment._credit_progress(rewards, agent, frame(-0.5))
        commander.current = 52
        before_switch = rewards[1]
        environment._credit_progress(rewards, agent, frame(-1.0))
        self.assertLess(rewards[1], before_switch)

        environment.current_step = 1
        environment._credit_progress(rewards, agent, frame(0.0))
        before_terminal = rewards[1]
        environment.current_step = environment.max_steps
        environment._credit_progress(rewards, agent, frame(1.0))
        self.assertLess(rewards[1], before_terminal)
        self.assertNotIn(agent.entity_id, environment._progress_state)

    def test_early_death_and_unsuccessful_timeout_use_same_type_cost(self):
        environment = PersonalR9TrainingEnv.__new__(PersonalR9TrainingEnv)
        environment.reward_config = R9RewardConfig()
        environment._episode_reward_components = defaultdict(float)
        environment._successful_sources = set()
        environment._settled_failures = set()

        dead = PersonalR9PPOAttackAgent.__new__(PersonalR9PPOAttackAgent)
        dead.agent_id, dead.entity_id = 1, 2
        survivor = PersonalR9PPOAttackAgent.__new__(PersonalR9PPOAttackAgent)
        survivor.agent_id, survivor.entity_id = 2, 3
        rewards = {1: 0.0, 2: 0.0}
        previous = {"entities": {2: {"health": 1.0, "type": 21001}}}
        current = {
            "entities": {
                2: {"health": 0.0, "type": 21001},
                3: {"health": 1.0, "type": 21001},
            }
        }

        environment._credit_losses(
            rewards,
            {2: dead},
            {2},
            previous,
            current,
            set(),
            mission_completed=False,
        )
        environment._credit_timeout_losses(rewards, [survivor], current)
        self.assertAlmostEqual(rewards[1], -environment.reward_config.medium_loss_cost)
        self.assertAlmostEqual(rewards[2], -environment.reward_config.medium_loss_cost)


class DynamicShipDiscoveryTests(unittest.TestCase):
    @staticmethod
    def _ship_observation():
        return {
            "step": 0,
            "entity_id": 900,
            "self": {
                "type": 21002,
                "position": {"lon": 0.0, "lat": 0.0, "alt": 10_000.0},
                "health": 1.0,
                "isVisible": True,
                "detectInfo": {
                    168: {
                        "time": INITIAL_TIME,
                        "entity_id": 168,
                        "entity_type": 9500,
                        "lla": {"x": 0.1, "y": 0.0, "z": 0.0},
                    },
                    999: {
                        "entity_id": 999,
                        "entity_type": 24000,
                        "lla": {"x": 0.2, "y": 0.0, "z": 1000.0},
                    },
                },
            },
        }

    def test_only_detected_9500_is_added_and_r9_assigns_low_platform(self):
        initial = initial_targets_from_observation(_initial_targets())
        fusion = DynamicDetectedTargetTrackFusion(initial)
        ship_observation = self._ship_observation()
        self.assertTrue(fusion.ingest(ship_observation))
        self.assertEqual([target.entity_id for target in fusion.targets], [51, 168])
        ship = fusion.targets[1]
        self.assertEqual(ship.entity_type, 9500)
        self.assertEqual(ship.value, 3.0)
        ship_observation["self"]["detectInfo"][168]["lla"]["x"] = 9.0
        self.assertFalse(fusion.ingest(ship_observation))
        self.assertAlmostEqual(fusion.targets[1].position.lon, 0.1)

        commander = PersonalR9Commander(initial, seed=1)
        commander.register_platform(900)
        commander.begin_step((self._ship_observation(),))
        # Planning is completed from the joint snapshot, before any per-Agent
        # action query, and the launch-fraction context is frozen for the step.
        self.assertEqual(commander.target_id_for(900), 168)
        context_before_action = commander.learning_task_context(900)
        launch = commander.action_for(900, 0)
        self.assertIsNotNone(launch)
        self.assertEqual(commander.target_id_for(900), 168)
        self.assertAlmostEqual(launch[2], 0.1)
        self.assertEqual(
            commander.learning_task_context(900),
            context_before_action,
        )

        commander.reset()
        self.assertEqual([target.entity_id for target in commander.targets], [51])


class LocalPPOTests(unittest.TestCase):
    def test_trainer_uses_the_personal_train_ppo_copy(self):
        self.assertEqual(PPOSharedPolicy.__module__, "personal_train.ppo_policy")
        self.assertEqual(Path(inspect.getfile(PPOSharedPolicy)).parent, PERSONAL_ROOT)

    def test_trainer_defaults_select_stable_episode_gae(self):
        args = parse_args([])
        config = _policy_config(args)
        self.assertEqual((config.observation_dim, config.action_dim), (90, 3))
        self.assertEqual(config.update_mode, "episode")
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.learning_rate_final, 2e-5)
        self.assertEqual(config.gamma, 0.999)
        self.assertEqual(config.gae_lambda, 0.995)
        self.assertEqual(config.update_epochs, 2)
        self.assertEqual(config.minibatch_size, 4096)
        self.assertLessEqual(config.entropy_final_coef, config.entropy_coef)
        self.assertEqual(args.render_fps, 10)

    def test_gae_keeps_interleaved_agent_trajectories_separate(self):
        advantages, returns = PPOSharedPolicy._compute_gae(
            np.asarray([1, 2, 1, 2]),
            np.asarray([1.0, 10.0, 2.0, 20.0]),
            np.asarray([False, False, True, True]),
            np.zeros(4),
            np.zeros(4),
            gamma=1.0,
            gae_lambda=1.0,
        )
        np.testing.assert_allclose(advantages, (3.0, 30.0, 2.0, 20.0))
        np.testing.assert_allclose(returns, advantages)

    def test_gae_bootstraps_a_cut_rollout_but_not_a_terminal(self):
        cut_advantages, _ = PPOSharedPolicy._compute_gae(
            np.asarray([1]),
            np.asarray([1.0]),
            np.asarray([False]),
            np.asarray([0.0]),
            np.asarray([5.0]),
            gamma=1.0,
            gae_lambda=1.0,
        )
        terminal_advantages, _ = PPOSharedPolicy._compute_gae(
            np.asarray([1]),
            np.asarray([1.0]),
            np.asarray([True]),
            np.asarray([0.0]),
            np.asarray([5.0]),
            gamma=1.0,
            gae_lambda=1.0,
        )
        self.assertAlmostEqual(float(cut_advantages[0]), 6.0)
        self.assertAlmostEqual(float(terminal_advantages[0]), 1.0)

    def test_rollout_threshold_updates_only_after_the_joint_step(self):
        policy = PPOSharedPolicy(
            PPOConfig(
                observation_dim=2,
                action_dim=3,
                hidden_dim=8,
                update_mode="rollout",
                rollout_size=1,
                minibatch_size=2,
                update_epochs=1,
                value_inference_batch_size=2,
                target_kl=0.0,
                device="cpu",
            )
        )
        observation = np.asarray([0.0, 1.0], dtype=np.float32)
        mask = np.ones(3, dtype=np.bool_)
        selected = {
            agent_id: policy.select_action_for_agent(agent_id, observation, mask)
            for agent_id in (1, 2)
        }
        for agent_id in (1, 2):
            policy.observe(
                PolicyTransition(
                    agent_id=agent_id,
                    observation=observation,
                    action=selected[agent_id],
                    action_mask=mask,
                    reward=float(agent_id),
                    next_observation=observation,
                    done=True,
                )
            )
            self.assertEqual(policy.update_count, 0)

        policy.finish_environment_step()
        self.assertEqual(policy.update_count, 1)
        self.assertEqual(policy.last_metrics["samples"], 2.0)
        self.assertTrue(all(math.isfinite(value) for value in policy.last_metrics.values()))

    def test_default_episode_mode_freezes_policy_until_finish_episode(self):
        policy = PPOSharedPolicy(
            PPOConfig(
                observation_dim=2,
                action_dim=3,
                hidden_dim=8,
                update_mode="episode",
                rollout_size=1,
                minibatch_size=2,
                update_epochs=1,
                value_inference_batch_size=2,
                target_kl=0.0,
                device="cpu",
            )
        )
        observation = np.asarray([0.25, -0.25], dtype=np.float32)
        mask = np.ones(3, dtype=np.bool_)
        for step in range(2):
            action = policy.select_action_for_agent(1, observation, mask)
            policy.observe(
                PolicyTransition(
                    agent_id=1,
                    observation=observation,
                    action=action,
                    action_mask=mask,
                    reward=1.0,
                    next_observation=observation,
                    done=step == 1,
                )
            )
            policy.finish_environment_step()
            self.assertEqual(policy.update_count, 0)

        policy.finish_episode()
        self.assertEqual(policy.update_count, 1)
        self.assertEqual(policy.episode_count, 1)
        self.assertFalse(policy._buffer)


class InterceptionMetricsTests(unittest.TestCase):
    @staticmethod
    def _destroy_event(victim_id, victim_type, source_id, source_type, time_ms):
        return {
            "type": 40000,
            "logicTime": time_ms,
            "content": {
                "event": {
                    "entity": {"id": victim_id, "type": victim_type},
                    "destroySrc": {"id": source_id, "type": source_type},
                }
            },
        }

    def test_interceptor_kills_are_causal_deduplicated_and_multi_kill_counted(self):
        environment = PersonalR9TrainingEnv.__new__(PersonalR9TrainingEnv)
        environment._red_destroy_causes = {}
        for event in (
            self._destroy_event(1, 21000, 501, 24000, 1000),
            # Overlapping damage after the first terminal must not reattribute.
            self._destroy_event(1, 21000, 502, 24000, 1000),
            self._destroy_event(2, 21002, 501, 24000, 1000),
            # Normal self-terminal is not an interception.
            self._destroy_event(3, 21001, 3, 21001, 2000),
        ):
            environment._capture_destroy_event(event)

        def interceptor(entity_id, launched, health):
            entity = SimpleNamespace(id=entity_id, survivePoints=health)
            return SimpleNamespace(
                launched=launched,
                entity_ext=SimpleNamespace(entity=entity),
            )

        interceptors = (
            interceptor(501, 1, 0.0),
            interceptor(502, 1, 0.0),
            interceptor(503, -1, 1.0),
        )
        factory = SimpleNamespace(
            get_simulators_by_type=lambda entity_type: (
                list(interceptors) if entity_type == 24000 else []
            )
        )
        environment.engine = SimpleNamespace(simulator_factory=factory)

        metrics = environment.interception_metrics()
        self.assertEqual(metrics["red_missiles_intercepted"], 2)
        self.assertEqual(
            metrics["red_missiles_intercepted_by_type"],
            {"high": 1, "medium": 0, "low": 1},
        )
        self.assertEqual(metrics["interceptors_launched"], 2)
        self.assertEqual(metrics["successful_interceptors"], 1)
        self.assertEqual(metrics["multi_kill_interceptors"], 1)
        self.assertEqual(metrics["max_kills_by_one_interceptor"], 2)
        self.assertEqual(metrics["kills_per_interceptor_histogram"], {"2": 1})


class _ReporterPolicy:
    last_metrics = {
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 1.0,
        "samples": 32,
        "approx_kl": 0.01,
        "clip_fraction": 0.02,
        "explained_variance": 0.3,
        "learning_rate": 1e-4,
        "entropy_coef": 0.002,
        "epochs_ran": 2,
        "early_stopped": 0,
    }
    update_count = 2
    transition_count = 100


class ReportingTests(unittest.TestCase):
    def test_mandatory_text_json_csv_and_svg_outputs(self):
        with tempfile.TemporaryDirectory(dir=PERSONAL_ROOT) as directory:
            output = Path(directory)
            reporter = TrainingReporter(output)
            # The two promised charts exist even if a job is interrupted
            # before its first complete round.
            self.assertTrue((output / "round_scores.svg").is_file())
            self.assertTrue((output / "training_dashboard.svg").is_file())
            reporter.write_run_config({"scenario": "E01"})
            reporter.record_round(
                round_index=1,
                summary={
                    "steps_executed": 100,
                    "score": {"score": 42.0, "K": 0.5, "T": 0.1, "completed": False},
                    "red": {"launched": 20, "alive": 150, "lost": 14},
                },
                reward_metrics={"agent_return_sum": 8.0, "agent_return_mean": 0.4, "components": {}},
                policy=_ReporterPolicy(),
                elapsed_seconds=1.5,
                action_counts={0: 10, 1: 20, 2: 30},
                action_switches=4,
            )
            for name in (
                "round_scores.txt",
                "round_scores.csv",
                "round_scores.svg",
                "training_dashboard.svg",
                "round_metrics.jsonl",
                "summary_round_0001.json",
                "latest_summary.json",
                "ppo_metrics.csv",
                "run_config.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            self.assertIn("score=42.000000", (output / "round_scores.txt").read_text(encoding="utf-8"))
            parsed = json.loads((output / "summary_round_0001.json").read_text(encoding="utf-8"))
            self.assertEqual(parsed["official_summary"]["score"]["score"], 42.0)
            with (output / "ppo_metrics.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                ppo_row = next(csv.DictReader(stream))
            self.assertAlmostEqual(float(ppo_row["approx_kl"]), 0.01)
            self.assertAlmostEqual(float(ppo_row["clip_fraction"]), 0.02)
            self.assertAlmostEqual(float(ppo_row["explained_variance"]), 0.3)
            self.assertAlmostEqual(float(ppo_row["learning_rate"]), 1e-4)


class CheckpointTests(unittest.TestCase):
    @staticmethod
    def _config():
        return PPOConfig(
            observation_dim=2,
            action_dim=3,
            hidden_dim=8,
            rollout_size=8,
            minibatch_size=1,
            update_epochs=1,
            value_inference_batch_size=1,
            target_kl=0.0,
            device="cpu",
        )

    @staticmethod
    def _add_terminal_transition(policy):
        observation = np.asarray([0.1, -0.2], dtype=np.float32)
        mask = np.ones(3, dtype=np.bool_)
        action = policy.select_action_for_agent(1, observation, mask)
        policy.observe(
            PolicyTransition(
                agent_id=1,
                observation=observation,
                action=action,
                action_mask=mask,
                reward=1.0,
                next_observation=observation,
                done=True,
            )
        )

    def test_saved_local_ppo_has_current_schema_marker(self):
        with tempfile.TemporaryDirectory(dir=PERSONAL_ROOT) as directory:
            path = Path(directory) / "policy.pt"
            policy = PPOSharedPolicy(
                PPOConfig(
                    observation_dim=90,
                    action_dim=3,
                    hidden_dim=8,
                    rollout_size=8,
                    minibatch_size=4,
                    update_epochs=1,
                    device="cpu",
                )
            )
            _save_policy(policy, path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["personal_train_schema"], CHECKPOINT_SCHEMA)
            self.assertEqual(checkpoint["algorithm"], PPOSharedPolicy.ALGORITHM)
            self.assertTrue(checkpoint["safe_training_boundary"])

    def test_safe_current_checkpoint_restores_network_optimizer_and_counters(self):
        with tempfile.TemporaryDirectory(dir=PERSONAL_ROOT) as directory:
            path = Path(directory) / "latest.pt"
            original = PPOSharedPolicy(self._config())
            self._add_terminal_transition(original)
            original.finish_episode()
            _save_policy(original, path)
            self.assertTrue(_resume_loads_optimizer(path))

            restored = PPOSharedPolicy(self._config())
            restored.load(str(path), load_optimizer=True)
            for expected, actual in zip(
                original.network.parameters(), restored.network.parameters()
            ):
                torch.testing.assert_close(expected, actual)
            self.assertEqual(restored.update_count, original.update_count)
            self.assertEqual(restored.transition_count, original.transition_count)
            self.assertEqual(restored.episode_count, original.episode_count)
            self.assertEqual(
                restored.optimizer.state_dict()["state"].keys(),
                original.optimizer.state_dict()["state"].keys(),
            )

    def test_unsafe_best_checkpoint_is_weights_only(self):
        with tempfile.TemporaryDirectory(dir=PERSONAL_ROOT) as directory:
            path = Path(directory) / "best.pt"
            original = PPOSharedPolicy(self._config())
            self._add_terminal_transition(original)
            _save_policy(original, path)
            self.assertFalse(_resume_loads_optimizer(path))

            restored = PPOSharedPolicy(self._config())
            restored.load(str(path), load_optimizer=False)
            for expected, actual in zip(
                original.network.parameters(), restored.network.parameters()
            ):
                torch.testing.assert_close(expected, actual)
            self.assertEqual(restored.update_count, 0)
            self.assertEqual(restored.transition_count, 0)

    def test_safe_best_checkpoint_is_still_weights_only(self):
        with tempfile.TemporaryDirectory(dir=PERSONAL_ROOT) as directory:
            path = Path(directory) / "best.pt"
            original = PPOSharedPolicy(self._config())
            _save_policy(original, path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertTrue(checkpoint["safe_training_boundary"])
            self.assertEqual(_resume_mode(path), "current_weights_only")
            self.assertFalse(_resume_loads_optimizer(path))

    def test_current_schema_with_wrong_algorithm_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=PERSONAL_ROOT) as directory:
            path = Path(directory) / "wrong_algorithm.pt"
            policy = PPOSharedPolicy(
                PPOConfig(
                    observation_dim=90,
                    action_dim=3,
                    hidden_dim=8,
                    rollout_size=8,
                    minibatch_size=4,
                    update_epochs=1,
                    device="cpu",
                )
            )
            _save_policy(policy, path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint["algorithm"] = "different_algorithm"
            torch.save(checkpoint, path)

            args = parse_args(["--resume", str(path), "--device", "cpu"])
            with self.assertRaises(ValueError):
                _policy_config(args, path)


if __name__ == "__main__":
    unittest.main()
