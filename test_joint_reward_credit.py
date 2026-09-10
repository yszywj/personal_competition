"""Tests for branch reward storage and conservative planning credit."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np

from personal_train.joint_reward_credit import (
    ObjectiveContribution,
    ObjectiveDamageRecord,
    ObjectiveParticipation,
    allocate_local_objective_credit,
    compute_objective_damage_contributions,
    interceptor_track_information_potential,
    mix_planning_rewards,
    objective_information_potential,
    potential_difference,
)
from personal_train.joint_rl_core import (
    JointAction,
    JointPolicyTrace,
    JointSpaceSpec,
    JointTrajectoryBuffer,
    JointTransition,
    SharedSensorConfig,
    SharedSensorState,
    UnitAction,
    UnitControlState,
    build_joint_action_mask,
)


class PlanningCreditTests(unittest.TestCase):
    def test_weighted_damage_contributions_match_official_fraction(self) -> None:
        contributions = compute_objective_damage_contributions(
            (
                ObjectiveDamageRecord(0, weight=5.0, initial_health=100.0, final_health=50.0),
                ObjectiveDamageRecord(1, weight=2.0, initial_health=20.0, final_health=0.0),
                ObjectiveDamageRecord(2, weight=1.0, initial_health=10.0, final_health=20.0),
            )
        )

        self.assertAlmostEqual(contributions[0].normalized_contribution, 2.5 / 8.0)
        self.assertAlmostEqual(contributions[1].normalized_contribution, 2.0 / 8.0)
        self.assertEqual(contributions[2].normalized_contribution, 0.0)
        self.assertAlmostEqual(
            sum(item.normalized_contribution for item in contributions), 4.5 / 8.0
        )

    def test_local_allocation_is_conservative_and_prefers_effective_evidence(self) -> None:
        contributions = (
            ObjectiveContribution(0, damage_fraction=0.6, normalized_contribution=0.3),
            ObjectiveContribution(1, damage_fraction=1.0, normalized_contribution=0.2),
        )
        allocation = allocate_local_objective_credit(
            contributions,
            (
                ObjectiveParticipation(0, 0, effective_responsibility=1.0, fallback_responsibility=20.0),
                ObjectiveParticipation(1, 0, effective_responsibility=3.0, fallback_responsibility=1.0),
                ObjectiveParticipation(1, 1, fallback_responsibility=2.0),
                ObjectiveParticipation(2, 1, fallback_responsibility=1.0),
            ),
            unit_count=3,
        )

        self.assertAlmostEqual(allocation.credit_by_unit[0], 0.075)
        self.assertAlmostEqual(allocation.credit_by_unit[1], 0.225 + 2.0 / 3.0 * 0.2)
        self.assertAlmostEqual(allocation.credit_by_unit[2], 1.0 / 3.0 * 0.2)
        self.assertEqual(allocation.fallback_objective_slots, (1,))
        self.assertAlmostEqual(allocation.total_allocated, 0.5)
        self.assertAlmostEqual(allocation.total_unallocated, 0.0)
        self.assertAlmostEqual(
            allocation.total_allocated + allocation.total_unallocated,
            allocation.total_contribution,
        )

    def test_threshold_uses_fallback_and_no_participant_stays_unallocated(self) -> None:
        contributions = (
            ObjectiveContribution(0, damage_fraction=1.0, normalized_contribution=0.4),
            ObjectiveContribution(1, damage_fraction=1.0, normalized_contribution=0.1),
        )
        allocation = allocate_local_objective_credit(
            contributions,
            (
                ObjectiveParticipation(
                    0,
                    0,
                    effective_responsibility=0.25,
                    fallback_responsibility=2.0,
                ),
                ObjectiveParticipation(1, 0, fallback_responsibility=1.0),
            ),
            unit_count=2,
            minimum_effective_responsibility=0.5,
        )

        self.assertEqual(allocation.fallback_objective_slots, (0,))
        self.assertAlmostEqual(allocation.credit_by_unit[0], 0.4 * 2.0 / 3.0)
        self.assertAlmostEqual(allocation.credit_by_unit[1], 0.4 / 3.0)
        self.assertAlmostEqual(allocation.total_allocated, 0.4)
        self.assertAlmostEqual(allocation.total_unallocated, 0.1)
        self.assertEqual(dict(allocation.unallocated_by_objective)[1], 0.1)

    def test_duplicate_records_and_invalid_responsibility_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compute_objective_damage_contributions(
                (
                    ObjectiveDamageRecord(0, 1.0, 1.0, 0.0),
                    ObjectiveDamageRecord(0, 1.0, 1.0, 0.0),
                )
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ObjectiveParticipation(0, 0, effective_responsibility=-1.0)
        with self.assertRaisesRegex(ValueError, "unknown objective"):
            allocate_local_objective_credit(
                (ObjectiveContribution(0, 1.0, 1.0),),
                (ObjectiveParticipation(0, 1, fallback_responsibility=1.0),),
                unit_count=1,
            )

    def test_team_and_local_plan_mix_only_rewards_eligible_units(self) -> None:
        rewards = mix_planning_rewards(
            team_score_fraction=0.5,
            local_credit_by_unit=(0.2, 0.0, 0.1),
            eligible_units=(True, False, True),
        )
        self.assertAlmostEqual(rewards[0], 0.7 * 0.5 + 0.3 * 0.2)
        self.assertEqual(rewards[1], 0.0)
        self.assertAlmostEqual(rewards[2], 0.7 * 0.5 + 0.3 * 0.1)
        with self.assertRaisesRegex(ValueError, "ineligible"):
            mix_planning_rewards(
                team_score_fraction=0.5,
                local_credit_by_unit=(0.1,),
                eligible_units=(False,),
            )

    def test_sensor_information_uses_potential_difference(self) -> None:
        previous = objective_information_potential(
            known_objective_slots=(0,),
            objective_weights={0: 5.0, 1: 2.0, 2: 1.0},
        )
        current = objective_information_potential(
            known_objective_slots=(0, 2),
            objective_weights={0: 5.0, 1: 2.0, 2: 1.0},
        )
        self.assertAlmostEqual(previous, 0.015 * 5.0 / 8.0)
        self.assertAlmostEqual(current, 0.015 * 6.0 / 8.0)
        self.assertAlmostEqual(
            potential_difference(previous=previous, current=current, gamma=0.995),
            0.995 * current - previous,
        )

    def test_interceptor_information_counts_unique_fresh_track_mass(self) -> None:
        potential = interceptor_track_information_potential(
            threat_age_steps=(0, 50, 100, 150),
            max_track_age_steps=100,
            threat_normalizer=2,
            scale=0.02,
        )
        # Freshness mass is 1 + .5; the expired tracks contribute zero.
        self.assertAlmostEqual(potential, 0.015)
        saturated = interceptor_track_information_potential(
            threat_age_steps=(0, 0, 0),
            max_track_age_steps=100,
            threat_normalizer=2,
            scale=0.02,
        )
        self.assertEqual(saturated, 0.02)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            interceptor_track_information_potential(
                threat_age_steps=(-1,),
                max_track_age_steps=100,
                threat_normalizer=2,
            )


class BranchTrajectoryTests(unittest.TestCase):
    @staticmethod
    def _transition() -> tuple[JointSpaceSpec, JointTransition]:
        space = JointSpaceSpec(unit_count=1, objective_count=1)
        states = (UnitControlState(slot=0),)
        mask = build_joint_action_mask(
            space,
            states,
            (True,),
            SharedSensorState.initial(SharedSensorConfig(capacity=0)),
            step=0,
        )
        observation = np.asarray((0.0, 1.0), dtype=np.float32)
        transition = JointTransition(
            observations=(observation,),
            states=states,
            mask=mask,
            action=JointAction.from_sequence((UnitAction.noop(0),)),
            trace=JointPolicyTrace(
                log_prob_by_term={"unit/0/activation": -0.25},
                values_by_unit=(0.1,),
                team_value=0.2,
                plan_values_by_unit=(0.3,),
                motion_values_by_unit=(0.4,),
                sensor_value=0.5,
            ),
            rewards=(0.6,),
            team_reward=0.7,
            next_observations=(observation + 1.0,),
            terminated=(False,),
            truncated=(False,),
            plan_rewards=(0.8,),
            motion_rewards=(0.9,),
            sensor_reward=1.0,
        )
        return space, transition

    def test_buffer_preserves_explicit_branch_rewards_and_values(self) -> None:
        space, transition = self._transition()
        buffer = JointTrajectoryBuffer(space, observation_dim=2)
        buffer.append(transition)
        stored = buffer.items[0]
        self.assertEqual(stored.plan_rewards, (0.8,))
        self.assertEqual(stored.motion_rewards, (0.9,))
        self.assertEqual(stored.sensor_reward, 1.0)
        self.assertEqual(stored.trace.plan_values_by_unit, (0.3,))
        self.assertEqual(stored.trace.motion_values_by_unit, (0.4,))
        self.assertEqual(stored.trace.sensor_value, 0.5)
        self.assertIsNot(stored.trace, transition.trace)

    def test_buffer_rejects_bad_branch_shape_and_non_finite_values(self) -> None:
        space, transition = self._transition()
        buffer = JointTrajectoryBuffer(space, observation_dim=2)
        with self.assertRaisesRegex(ValueError, "one reward per unit"):
            buffer.append(replace(transition, plan_rewards=(1.0, 2.0)))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            buffer.append(replace(transition, sensor_reward=math.nan))
        bad_trace = replace(transition.trace, plan_values_by_unit=(0.0, 1.0))
        with self.assertRaisesRegex(ValueError, "one value per unit"):
            buffer.append(replace(transition, trace=bad_trace))


if __name__ == "__main__":
    unittest.main()
