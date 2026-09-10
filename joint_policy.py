"""Masked parameter-shared actor-critic and PPO for joint game control.

The simulator adapter owns lifecycle state, masks and command translation.  This
module only consumes the immutable contracts from :mod:`joint_rl_core`.  Every
PPO probability ratio is formed for one entity (or for the shared-sensor
decision) so a large team never produces one product of hundreds of action
probabilities.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F

from .joint_rl_core import (
    BinaryChoice,
    JointAction,
    JointActionMask,
    JointPolicyTrace,
    JointSpaceSpec,
    JointTrajectoryBuffer,
    JointTransition,
    Movement,
    SharedSensorAction,
    SharedSensorPolicyTrace,
    UnitAction,
    UnitControlState,
    UnitPhase,
    branch_activity,
    expected_log_prob_terms,
    validate_action_against_mask,
)


@dataclass
class JointPPOConfig:
    """Hyperparameters for one single-process, single-device joint policy."""

    observation_dim: int
    hidden_dim: int = 256
    learning_rate: float = 1e-4
    learning_rate_final: float = 1e-5
    learning_rate_decay_updates: int = 100
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    value_clip_ratio: float = 0.20
    value_coef: float = 0.5
    sensor_policy_coef: float = 1.0
    entropy_coef: float = 0.002
    entropy_final_coef: float = 0.0002
    entropy_decay_updates: int = 500
    target_kl: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 128
    value_inference_batch_size: int = 128
    # JointGameEnv already folds the official team score into unit rewards.
    # Keep this at zero unless a different adapter supplies purely local terms.
    unit_team_reward_weight: float = 0.0
    placement_log_std_min: float = -5.0
    placement_log_std_max: float = 1.0
    seed: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class JointNetworkOutput:
    """Batched outputs; the leading dimensions are ``[batch, unit]``."""

    actor_features: Tensor
    activation_logits: Tensor
    objective_logits: Tensor
    retarget_logits: Tensor
    movement_logits: Tensor
    sensor_logits: Tensor
    plan_values_by_unit: Tensor
    motion_values_by_unit: Tensor
    sensor_value: Tensor

    @property
    def values_by_unit(self) -> Tensor:
        """Compatibility alias for callers that previously exposed one critic."""

        return self.motion_values_by_unit

    @property
    def team_value(self) -> Tensor:
        """Compatibility alias for the shared-resource critic."""

        return self.sensor_value

    def item(self, index: int) -> "JointNetworkOutput":
        """Select one joint observation while preserving all unit axes."""

        return JointNetworkOutput(
            actor_features=self.actor_features[index],
            activation_logits=self.activation_logits[index],
            objective_logits=self.objective_logits[index],
            retarget_logits=self.retarget_logits[index],
            movement_logits=self.movement_logits[index],
            sensor_logits=self.sensor_logits[index],
            plan_values_by_unit=self.plan_values_by_unit[index],
            motion_values_by_unit=self.motion_values_by_unit[index],
            sensor_value=self.sensor_value[index],
        )


@dataclass(frozen=True)
class JointPolicyEvaluation:
    """Differentiable statistics for one saved joint action."""

    log_prob_by_term: Mapping[str, Tensor]
    entropy_by_term: Mapping[str, Tensor]
    plan_values_by_unit: Tensor
    motion_values_by_unit: Tensor
    sensor_value: Tensor

    @property
    def values_by_unit(self) -> Tensor:
        return self.motion_values_by_unit

    @property
    def team_value(self) -> Tensor:
        return self.sensor_value

    @property
    def total_log_prob(self) -> Tensor:
        values = tuple(self.log_prob_by_term.values())
        if values:
            return torch.stack(values).sum()
        return self.sensor_value * 0.0


@dataclass(frozen=True)
class _PackedPolicyRollout:
    """Host-tensor saved actions and masks for repeated PPO minibatch use.

    The rollout buffer intentionally stores simulator-neutral Python objects.
    Rebuilding distributions from those objects inside every PPO epoch is very
    expensive, though, so :meth:`JointPPOPolicy.update` converts them once to
    dense CPU tensors.  Only the selected minibatch is moved to the policy
    device.  This bounds accelerator memory by ``minibatch_size`` rather than
    the complete rollout length.  All tensors have a leading transition
    dimension; unit branches additionally have a second unit dimension.
    """

    activation_actions: Tensor
    objective_actions: Tensor
    retarget_actions: Tensor
    movement_actions: Tensor
    placement_actions: Tensor
    activation_masks: Tensor
    objective_masks: Tensor
    retarget_masks: Tensor
    movement_masks: Tensor
    activation_active: Tensor
    placement_active: Tensor
    objective_active: Tensor
    retarget_active: Tensor
    movement_active: Tensor
    plan_active: Tensor
    sensor_active: Tensor
    sensor_draw_tokens: Tensor
    sensor_draw_masks: Tensor
    sensor_draw_active: Tensor
    old_plan_log_probs: Tensor
    old_motion_log_probs: Tensor
    old_sensor_log_probs: Tensor

    def select(
        self,
        indices: Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> "_PackedPolicyRollout":
        """Select a PPO minibatch and optionally transfer only that batch."""

        source_device = self.activation_actions.device
        source_indices = indices.to(device=source_device, dtype=torch.long)

        def selected(value: Tensor) -> Tensor:
            result = value.index_select(0, source_indices)
            return result.to(device=device) if device is not None else result

        return _PackedPolicyRollout(
            **{
                name: selected(value)
                for name, value in self.__dict__.items()
            }
        )


@dataclass(frozen=True)
class _PackedPolicyEvaluation:
    """Aggregated per-decision statistics for a tensorized minibatch."""

    plan_log_probs: Tensor
    plan_entropies: Tensor
    motion_log_probs: Tensor
    motion_entropies: Tensor
    sensor_log_probs: Tensor
    sensor_entropies: Tensor


class JointActorCritic(nn.Module):
    """Shared entity actor with a critic conditioned on the complete team."""

    def __init__(
        self,
        observation_dim: int,
        objective_count: int,
        hidden_dim: int,
        *,
        placement_log_std_min: float,
        placement_log_std_max: float,
    ) -> None:
        super().__init__()
        self.placement_log_std_min = float(placement_log_std_min)
        self.placement_log_std_max = float(placement_log_std_max)
        self.entity_encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # Every entity action sees a pooled summary of the complete controlled
        # team.  This lets activation, target allocation and movement react to
        # how many peers are staged/active/terminal and where their assignments
        # currently point, while retaining a factorized per-entity PPO ratio.
        self.joint_actor_encoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.activation_head = nn.Linear(hidden_dim, 2)
        self.objective_head = nn.Linear(hidden_dim, objective_count)
        # Planning is autoregressive: after selecting a target, placement and
        # launch-time movement are conditioned on that discrete target.  A
        # parallel placement/target head cannot represent this dependency.
        self.objective_embedding = nn.Embedding(objective_count, hidden_dim)
        self.target_condition_encoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.placement_mean_head = nn.Linear(hidden_dim, 2)
        self.placement_log_std_head = nn.Linear(hidden_dim, 2)
        self.initial_movement_head = nn.Linear(hidden_dim, 3)
        self.retarget_head = nn.Linear(hidden_dim, 2)
        self.movement_head = nn.Linear(hidden_dim, 3)

        # The same slot scorer is applied to every entity.  Eligibility and
        # sampling-without-replacement are supplied by JointActionMask.
        self.sensor_slot_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.sensor_stop_head = nn.Linear(hidden_dim, 1)

        # Planning and high-frequency motion use separate value baselines.  A
        # single critic cannot fit an undiscounted episode-level planning
        # credit and dense step-level flight return at the same time.
        self.plan_critic = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.motion_critic = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.sensor_critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self._initialize_action_priors()

    def _initialize_action_priors(self) -> None:
        """Avoid destructive half-team actions from zero-information logits."""

        with torch.no_grad():
            self.activation_head.bias.zero_()
            self.activation_head.bias[BinaryChoice.YES] = -4.0
            self.retarget_head.bias.zero_()
            self.retarget_head.bias[BinaryChoice.YES] = -3.0
            self.movement_head.bias.zero_()
            self.movement_head.bias[Movement.NEUTRAL] = 1.0
            self.initial_movement_head.bias.zero_()
            self.initial_movement_head.bias[Movement.NEUTRAL] = 1.0
            self.placement_log_std_head.bias.fill_(-0.5)
            # STOP competes against every eligible slot, so it needs a larger
            # prior than a binary head.  At equal slot logits this gives about
            # 5% request probability for 164 eligible entities and 9% for 304.
            self.sensor_stop_head.bias.fill_(8.0)
            self.sensor_slot_head[-1].bias.zero_()

    def condition_on_objective(
        self,
        actor_features: Tensor,
        objective_slots: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return target-conditioned placement and first-movement parameters."""

        slots = objective_slots.to(device=actor_features.device, dtype=torch.long)
        target_features = self.objective_embedding(slots)
        if target_features.shape != actor_features.shape:
            raise ValueError("objective slots and actor features have incompatible shapes")
        conditioned = self.target_condition_encoder(
            torch.cat((actor_features, target_features), dim=-1)
        )
        log_std = self.placement_log_std_head(conditioned).clamp(
            self.placement_log_std_min,
            self.placement_log_std_max,
        )
        return (
            self.placement_mean_head(conditioned),
            log_std,
            self.initial_movement_head(conditioned),
        )

    def forward(self, observations: Tensor) -> JointNetworkOutput:
        if observations.ndim != 3:
            raise ValueError("joint observations must have shape [batch, unit, feature]")
        features = self.entity_encoder(observations)
        context = features.mean(dim=1)
        expanded_context = context.unsqueeze(1).expand(-1, features.shape[1], -1)
        entity_and_team = torch.cat((features, expanded_context), dim=-1)
        actor_features = self.joint_actor_encoder(entity_and_team)
        slot_logits = self.sensor_slot_head(entity_and_team).squeeze(-1)
        stop_logit = self.sensor_stop_head(context)
        return JointNetworkOutput(
            actor_features=actor_features,
            activation_logits=self.activation_head(actor_features),
            objective_logits=self.objective_head(actor_features),
            retarget_logits=self.retarget_head(actor_features),
            movement_logits=self.movement_head(actor_features),
            sensor_logits=torch.cat((stop_logit, slot_logits), dim=-1),
            plan_values_by_unit=self.plan_critic(entity_and_team).squeeze(-1),
            motion_values_by_unit=self.motion_critic(entity_and_team).squeeze(-1),
            sensor_value=self.sensor_critic(context).squeeze(-1),
        )


def _resolve_device(requested: str) -> torch.device:
    value = str(requested or "auto").strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("joint PPO device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"requested {value}, but CUDA is unavailable to PyTorch")
        index = device.index if device.index is not None else 0
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"requested {value}, but only {torch.cuda.device_count()} CUDA device(s) are visible"
            )
        return torch.device(f"cuda:{index}")
    return device


def _masked_categorical(logits: Tensor, raw_mask: np.ndarray | Sequence[bool]) -> Categorical:
    # Core masks are intentionally immutable NumPy arrays.  Copy before giving
    # their storage to Torch so Torch never warns about a non-writable tensor.
    mask = torch.as_tensor(
        np.array(raw_mask, dtype=np.bool_, copy=True),
        dtype=torch.bool,
        device=logits.device,
    )
    if mask.shape != logits.shape:
        raise ValueError(f"categorical mask shape {tuple(mask.shape)} != {tuple(logits.shape)}")
    if not bool(mask.any().item()):
        raise ValueError("categorical branch has no legal action")
    return Categorical(logits=logits.masked_fill(~mask, torch.finfo(logits.dtype).min))


def _batched_masked_categorical(logits: Tensor, mask: Tensor) -> Categorical:
    """Build one categorical distribution for an arbitrary batch shape."""

    if mask.dtype != torch.bool:
        raise TypeError("batched categorical masks must be boolean tensors")
    if mask.shape != logits.shape:
        raise ValueError(
            f"categorical mask shape {tuple(mask.shape)} != {tuple(logits.shape)}"
        )
    empty = ~mask.any(dim=-1, keepdim=True)
    fallback = torch.zeros_like(mask)
    fallback[..., 0] = True
    safe_mask = mask | (empty & fallback)
    return Categorical(
        logits=logits.masked_fill(~safe_mask, torch.finfo(logits.dtype).min)
    )


def _batched_masked_categorical_statistics(
    logits: Tensor,
    mask: Tensor,
    selected: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return batched selected log probabilities and categorical entropies.

    Some dense rows represent inactive conditional branches whose saved mask
    can contain no legal objective.  Those rows are ignored by an activity
    mask later, but ``Categorical`` still needs a well-defined distribution;
    token zero is therefore enabled only for such empty rows.  Active rows are
    unchanged and were already validated when appended to the rollout buffer.
    """

    if selected.shape != logits.shape[:-1]:
        raise ValueError("batched categorical actions have the wrong shape")
    distribution = _batched_masked_categorical(logits, mask)
    return distribution.log_prob(selected), distribution.entropy()


def _squashed_normal_log_prob(mean: Tensor, log_std: Tensor, action: Tensor) -> Tensor:
    """Log density of tanh(Normal), evaluated without storing pre-tanh samples."""

    epsilon = torch.finfo(action.dtype).eps
    bounded = action.clamp(-1.0 + epsilon, 1.0 - epsilon)
    pre_tanh = torch.atanh(bounded)
    base = Normal(mean, log_std.exp()).log_prob(pre_tanh)
    # Stable form of log(1 - tanh(z)^2), as used by SAC implementations.
    log_jacobian = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
    return (base - log_jacobian).sum(dim=-1)


class JointPPOPolicy:
    """Conditional hybrid joint policy and clipped PPO optimizer.

    ``sample`` returns the exact core action/trace pair expected by
    :class:`JointTrajectoryBuffer`.  ``update`` consumes a complete buffer and
    optimizes planning, motion and shared-sensor decisions with independent
    returns and critics.  Planning uses direct episode-level credit, while
    motion and sensor control use step-level GAE.
    """

    ALGORITHM = "personal_joint_masked_ppo"
    CHECKPOINT_SCHEMA_VERSION = 3

    def __init__(self, space: JointSpaceSpec, config: JointPPOConfig) -> None:
        self.space = space
        self.config = config
        self._validate_config()
        self.device = _resolve_device(config.device)
        self.config.device = str(self.device)

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(config.seed)
        self._rng = np.random.default_rng(config.seed)
        self.network = JointActorCritic(
            observation_dim=config.observation_dim,
            objective_count=space.objective_count,
            hidden_dim=config.hidden_dim,
            placement_log_std_min=config.placement_log_std_min,
            placement_log_std_max=config.placement_log_std_max,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=config.learning_rate
        )
        self.training = True
        self.update_count = 0
        self.transition_count = 0
        self.episode_count = 0
        self.last_metrics: dict[str, float] = {}

    def _validate_config(self) -> None:
        config = self.config
        for name in (
            "observation_dim",
            "hidden_dim",
            "learning_rate_decay_updates",
            "entropy_decay_updates",
            "update_epochs",
            "minibatch_size",
            "value_inference_batch_size",
        ):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (config.learning_rate, config.learning_rate_final)
        ):
            raise ValueError("learning rates must be positive")
        if config.learning_rate_final > config.learning_rate:
            raise ValueError("final learning rate cannot exceed initial learning rate")
        if not 0.0 <= config.gamma <= 1.0 or not 0.0 <= config.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if (
            not math.isfinite(config.clip_ratio)
            or not math.isfinite(config.value_clip_ratio)
            or config.clip_ratio <= 0.0
            or config.value_clip_ratio < 0.0
        ):
            raise ValueError("PPO clipping ratios are invalid")
        for name in (
            "value_coef",
            "sensor_policy_coef",
            "entropy_coef",
            "entropy_final_coef",
            "target_kl",
            "unit_team_reward_weight",
        ):
            if not math.isfinite(float(getattr(config, name))) or getattr(config, name) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if config.entropy_final_coef > config.entropy_coef:
            raise ValueError("final entropy coefficient cannot exceed its initial value")
        if not math.isfinite(config.max_grad_norm) or config.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if not (
            math.isfinite(config.placement_log_std_min)
            and math.isfinite(config.placement_log_std_max)
            and config.placement_log_std_min < config.placement_log_std_max
        ):
            raise ValueError("placement log-std bounds are invalid")
        if isinstance(config.seed, bool) or not isinstance(config.seed, int):
            raise ValueError("seed must be an integer")

    def _observation_tensor(self, observations: Sequence[np.ndarray]) -> Tensor:
        if len(observations) != self.space.unit_count:
            raise ValueError(
                f"expected {self.space.unit_count} unit observations, got {len(observations)}"
            )
        array = np.asarray(observations, dtype=np.float32)
        expected = (self.space.unit_count, self.config.observation_dim)
        if array.shape != expected:
            raise ValueError(f"expected observations shape {expected}, got {array.shape}")
        if not bool(np.isfinite(array).all()):
            raise ValueError("joint observations contain NaN or infinity")
        return torch.as_tensor(array, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _validate_inputs(
        self,
        states: Sequence[UnitControlState],
        mask: JointActionMask,
    ) -> tuple[UnitControlState, ...]:
        ordered = tuple(states)
        if len(ordered) != self.space.unit_count or tuple(
            state.slot for state in ordered
        ) != tuple(range(self.space.unit_count)):
            raise ValueError("states must be ordered by every configured unit slot")
        if mask.space != self.space:
            raise ValueError("joint action mask uses a different space")
        return ordered

    @staticmethod
    def _draw(distribution: Categorical, deterministic: bool) -> Tensor:
        return (
            torch.argmax(distribution.logits, dim=-1)
            if deterministic
            else distribution.sample()
        )

    def sample(
        self,
        observations: Sequence[np.ndarray],
        states: Sequence[UnitControlState],
        mask: JointActionMask,
        *,
        deterministic: bool | None = None,
    ) -> tuple[JointAction, JointPolicyTrace]:
        """Sample one simultaneous action and the exact PPO behavior trace."""

        ordered_states = self._validate_inputs(states, mask)
        deterministic = (not self.training) if deterministic is None else bool(deterministic)
        with torch.no_grad():
            output = self.network(self._observation_tensor(observations)).item(0)
            units, log_probs = self._sample_units_batched(
                output,
                ordered_states,
                mask,
                deterministic=deterministic,
            )
            values = torch.cat(
                (
                    output.plan_values_by_unit,
                    output.motion_values_by_unit,
                    output.sensor_value.reshape(1),
                )
            ).cpu().numpy()
            unit_count = self.space.unit_count
            plan_values = tuple(float(value) for value in values[:unit_count])
            motion_values = tuple(
                float(value) for value in values[unit_count : 2 * unit_count]
            )
            sensor_value = float(values[-1])

            sensor_action, sensor_trace, sensor_log_prob = self._sample_sensor(
                output.sensor_logits,
                mask,
                deterministic=deterministic,
            )
            if sensor_trace is not None:
                log_probs["shared_sensor"] = sensor_log_prob

            action = JointAction(units=tuple(units), shared_sensor=sensor_action)
            validate_action_against_mask(ordered_states, action, mask)
            trace = JointPolicyTrace(
                log_prob_by_term=log_probs,
                values_by_unit=motion_values,
                team_value=sensor_value,
                plan_values_by_unit=plan_values,
                motion_values_by_unit=motion_values,
                sensor_value=sensor_value,
                shared_sensor=sensor_trace,
            )
            trace.validate(ordered_states, action, mask)
            return action, trace

    # A descriptive alias for adapters that use the older policy naming style.
    sample_action = sample

    def _sample_units_batched(
        self,
        output: JointNetworkOutput,
        states: tuple[UnitControlState, ...],
        mask: JointActionMask,
        *,
        deterministic: bool,
    ) -> tuple[list[UnitAction], dict[str, float]]:
        """Sample every unit branch in a handful of batched device operations.

        Sampling is still factorized per unit and preserves every conditional
        branch and action mask.  The only intentional reproducibility change
        from the former scalar loop is RNG consumption order: stochastic draws
        are grouped by branch rather than interleaved unit by unit.
        """

        unit_count = self.space.unit_count
        device = output.actor_features.device
        categorical_masks = {
            name: torch.as_tensor(
                np.stack(
                    [getattr(mask.by_unit[slot], name) for slot in range(unit_count)]
                ),
                dtype=torch.bool,
                device=device,
            )
            for name in ("activation", "objective", "retarget", "movement")
        }
        staged = torch.as_tensor(
            [state.phase == UnitPhase.STAGED for state in states],
            dtype=torch.bool,
            device=device,
        )
        active = torch.as_tensor(
            [state.phase == UnitPhase.ACTIVE for state in states],
            dtype=torch.bool,
            device=device,
        )
        staged_slots = torch.nonzero(staged, as_tuple=False).squeeze(-1)
        active_slots = torch.nonzero(active, as_tuple=False).squeeze(-1)

        activation_actions = torch.zeros(unit_count, dtype=torch.long, device=device)
        retarget_actions = torch.zeros(unit_count, dtype=torch.long, device=device)
        objective_actions = torch.zeros(unit_count, dtype=torch.long, device=device)
        movement_actions = torch.full(
            (unit_count,), int(Movement.NEUTRAL), dtype=torch.long, device=device
        )
        placement_actions = output.actor_features.new_zeros((unit_count, 2))
        activation_log_probs = output.sensor_value.new_zeros(unit_count)
        placement_log_probs = output.sensor_value.new_zeros(unit_count)
        objective_log_probs = output.sensor_value.new_zeros(unit_count)
        retarget_log_probs = output.sensor_value.new_zeros(unit_count)
        movement_log_probs = output.sensor_value.new_zeros(unit_count)

        if staged_slots.numel():
            distribution = _batched_masked_categorical(
                output.activation_logits.index_select(0, staged_slots),
                categorical_masks["activation"].index_select(0, staged_slots),
            )
            selected = self._draw(distribution, deterministic)
            activation_actions.index_copy_(0, staged_slots, selected)
            activation_log_probs.index_copy_(
                0, staged_slots, distribution.log_prob(selected)
            )

        if active_slots.numel():
            distribution = _batched_masked_categorical(
                output.retarget_logits.index_select(0, active_slots),
                categorical_masks["retarget"].index_select(0, active_slots),
            )
            selected = self._draw(distribution, deterministic)
            retarget_actions.index_copy_(0, active_slots, selected)
            retarget_log_probs.index_copy_(
                0, active_slots, distribution.log_prob(selected)
            )

        activation_yes = staged & (activation_actions == int(BinaryChoice.YES))
        retarget_yes = active & (retarget_actions == int(BinaryChoice.YES))
        objective_active = activation_yes | retarget_yes
        objective_slots = torch.nonzero(objective_active, as_tuple=False).squeeze(-1)
        conditioned_movement_logits = output.movement_logits
        conditioned_means = output.actor_features.new_zeros((unit_count, 2))
        conditioned_log_stds = output.actor_features.new_zeros((unit_count, 2))
        if objective_slots.numel():
            distribution = _batched_masked_categorical(
                output.objective_logits.index_select(0, objective_slots),
                categorical_masks["objective"].index_select(0, objective_slots),
            )
            selected = self._draw(distribution, deterministic)
            objective_actions.index_copy_(0, objective_slots, selected)
            objective_log_probs.index_copy_(
                0, objective_slots, distribution.log_prob(selected)
            )
            mean, log_std, initial_movement_logits = self.network.condition_on_objective(
                output.actor_features.index_select(0, objective_slots), selected
            )
            conditioned_means.index_copy_(0, objective_slots, mean)
            conditioned_log_stds.index_copy_(0, objective_slots, log_std)
            conditioned_movement_logits = conditioned_movement_logits.index_copy(
                0, objective_slots, initial_movement_logits
            )

        placement_slots = torch.nonzero(activation_yes, as_tuple=False).squeeze(-1)
        if placement_slots.numel():
            mean = conditioned_means.index_select(0, placement_slots)
            log_std = conditioned_log_stds.index_select(0, placement_slots)
            normal = Normal(mean, log_std.exp())
            pre_tanh = mean if deterministic else normal.sample()
            selected = torch.tanh(pre_tanh)
            placement_actions.index_copy_(0, placement_slots, selected)
            placement_log_probs.index_copy_(
                0,
                placement_slots,
                _squashed_normal_log_prob(mean, log_std, selected),
            )

        movement_active = active | activation_yes
        movement_slots = torch.nonzero(movement_active, as_tuple=False).squeeze(-1)
        if movement_slots.numel():
            distribution = _batched_masked_categorical(
                conditioned_movement_logits.index_select(0, movement_slots),
                categorical_masks["movement"].index_select(0, movement_slots),
            )
            selected = self._draw(distribution, deterministic)
            movement_actions.index_copy_(0, movement_slots, selected)
            movement_log_probs.index_copy_(
                0, movement_slots, distribution.log_prob(selected)
            )

        # One transfer replaces hundreds of per-unit ``item()`` synchronizations.
        packed = torch.cat(
            (
                activation_actions[:, None].to(output.actor_features.dtype),
                retarget_actions[:, None].to(output.actor_features.dtype),
                objective_actions[:, None].to(output.actor_features.dtype),
                movement_actions[:, None].to(output.actor_features.dtype),
                placement_actions,
                activation_log_probs[:, None],
                placement_log_probs[:, None],
                objective_log_probs[:, None],
                retarget_log_probs[:, None],
                movement_log_probs[:, None],
            ),
            dim=-1,
        ).cpu().numpy()

        units: list[UnitAction] = []
        log_probs: dict[str, float] = {}
        for slot, state in enumerate(states):
            row = packed[slot]
            if state.phase == UnitPhase.STAGED:
                activation = BinaryChoice(int(row[0]))
                log_probs[f"unit/{slot}/activation"] = float(row[6])
                if activation == BinaryChoice.YES:
                    movement = Movement(int(row[3]))
                    action = UnitAction(
                        slot=slot,
                        activate=activation,
                        placement=(float(row[4]), float(row[5])),
                        objective_slot=int(row[2]),
                        movement=movement,
                    )
                    log_probs[f"unit/{slot}/placement"] = float(row[7])
                    log_probs[f"unit/{slot}/objective"] = float(row[8])
                    if branch_activity(state, action).movement:
                        log_probs[f"unit/{slot}/movement"] = float(row[10])
                    units.append(action)
                else:
                    units.append(UnitAction.noop(slot))
            elif state.phase == UnitPhase.ACTIVE:
                retarget = BinaryChoice(int(row[1]))
                objective_slot = int(row[2]) if retarget == BinaryChoice.YES else -1
                log_probs[f"unit/{slot}/retarget"] = float(row[9])
                if retarget == BinaryChoice.YES:
                    log_probs[f"unit/{slot}/objective"] = float(row[8])
                log_probs[f"unit/{slot}/movement"] = float(row[10])
                units.append(
                    UnitAction(
                        slot=slot,
                        objective_slot=objective_slot,
                        retarget=retarget,
                        movement=Movement(int(row[3])),
                    )
                )
            else:
                units.append(UnitAction.noop(slot))
        return units, log_probs

    def _sample_sensor(
        self,
        logits: Tensor,
        mask: JointActionMask,
        *,
        deterministic: bool,
    ) -> tuple[SharedSensorAction, SharedSensorPolicyTrace | None, float]:
        maximum = mask.shared_sensor_max_requests
        if maximum <= 0:
            return SharedSensorAction(), None, 0.0
        available = np.asarray(mask.shared_sensor_eligible, dtype=np.bool_).copy()
        selected: list[int] = []
        tokens: list[int] = []
        masks: list[np.ndarray] = []
        log_probs: list[float] = []
        while True:
            draw_mask = np.concatenate((np.asarray((True,), dtype=np.bool_), available))
            distribution = _masked_categorical(logits, draw_mask)
            token_tensor = self._draw(distribution, deterministic)
            token = int(token_tensor.item())
            tokens.append(token)
            masks.append(draw_mask.copy())
            log_probs.append(float(distribution.log_prob(token_tensor).item()))
            if token == 0:
                break
            slot = token - 1
            selected.append(slot)
            available[slot] = False
            if len(selected) >= maximum:
                break
        return (
            SharedSensorAction.from_sequence(selected),
            SharedSensorPolicyTrace(
                tokens=tuple(tokens), masks=tuple(masks), log_probs=tuple(log_probs)
            ),
            float(sum(log_probs)),
        )

    def evaluate(
        self,
        observations: Sequence[np.ndarray],
        states: Sequence[UnitControlState],
        mask: JointActionMask,
        action: JointAction,
        *,
        shared_sensor_trace: SharedSensorPolicyTrace | None = None,
    ) -> JointPolicyEvaluation:
        """Recompute differentiable statistics for a previously sampled action."""

        ordered_states = self._validate_inputs(states, mask)
        output = self.network(self._observation_tensor(observations)).item(0)
        return self._evaluate_output(
            output,
            ordered_states,
            mask,
            action,
            shared_sensor_trace,
        )

    evaluate_action = evaluate

    def _evaluate_output(
        self,
        output: JointNetworkOutput,
        states: tuple[UnitControlState, ...],
        mask: JointActionMask,
        action: JointAction,
        shared_sensor_trace: SharedSensorPolicyTrace | None,
    ) -> JointPolicyEvaluation:
        validate_action_against_mask(states, action, mask)
        actions = {item.slot: item for item in action.units}
        log_probs: dict[str, Tensor] = {}
        entropies: dict[str, Tensor] = {}
        for slot, state in enumerate(states):
            item = actions[slot]
            activity = branch_activity(state, item)
            unit_mask = mask.by_unit[slot]
            if activity.activation:
                distribution = _masked_categorical(
                    output.activation_logits[slot], unit_mask.activation
                )
                selected = torch.as_tensor(
                    int(item.activate), dtype=torch.long, device=self.device
                )
                name = f"unit/{slot}/activation"
                log_probs[name] = distribution.log_prob(selected)
                entropies[name] = distribution.entropy()
            if activity.placement:
                objective_tensor = torch.as_tensor(
                    int(item.objective_slot), dtype=torch.long, device=self.device
                )
                mean, log_std, _ = self.network.condition_on_objective(
                    output.actor_features[slot], objective_tensor
                )
                selected = torch.as_tensor(
                    item.placement, dtype=torch.float32, device=self.device
                )
                name = f"unit/{slot}/placement"
                log_probs[name] = _squashed_normal_log_prob(mean, log_std, selected)
                # The base-Normal entropy is a deterministic exploration proxy;
                # the action log density above includes the exact tanh Jacobian.
                entropies[name] = Normal(mean, log_std.exp()).entropy().sum()
            if activity.objective:
                distribution = _masked_categorical(
                    output.objective_logits[slot], unit_mask.objective
                )
                selected = torch.as_tensor(
                    int(item.objective_slot), dtype=torch.long, device=self.device
                )
                name = f"unit/{slot}/objective"
                log_probs[name] = distribution.log_prob(selected)
                entropies[name] = distribution.entropy()
            if activity.retarget:
                distribution = _masked_categorical(
                    output.retarget_logits[slot], unit_mask.retarget
                )
                selected = torch.as_tensor(
                    int(item.retarget), dtype=torch.long, device=self.device
                )
                name = f"unit/{slot}/retarget"
                log_probs[name] = distribution.log_prob(selected)
                entropies[name] = distribution.entropy()
            if activity.movement:
                if (
                    state.phase == UnitPhase.STAGED
                    and item.activate == BinaryChoice.YES
                ) or (
                    state.phase == UnitPhase.ACTIVE
                    and item.retarget == BinaryChoice.YES
                ):
                    objective_tensor = torch.as_tensor(
                        int(item.objective_slot), dtype=torch.long, device=self.device
                    )
                    _, _, movement_logits = self.network.condition_on_objective(
                        output.actor_features[slot], objective_tensor
                    )
                else:
                    movement_logits = output.movement_logits[slot]
                distribution = _masked_categorical(
                    movement_logits, unit_mask.movement
                )
                selected = torch.as_tensor(
                    int(item.movement), dtype=torch.long, device=self.device
                )
                name = f"unit/{slot}/movement"
                log_probs[name] = distribution.log_prob(selected)
                entropies[name] = distribution.entropy()

        sensor_active = mask.shared_sensor_max_requests > 0
        if sensor_active:
            if shared_sensor_trace is None:
                raise ValueError("active shared sensor branch requires its saved trace")
            shared_sensor_trace.validate(action, mask)
            sensor_log_probs: list[Tensor] = []
            sensor_entropies: list[Tensor] = []
            for token, draw_mask in zip(
                shared_sensor_trace.tokens, shared_sensor_trace.masks
            ):
                distribution = _masked_categorical(output.sensor_logits, draw_mask)
                selected = torch.as_tensor(token, dtype=torch.long, device=self.device)
                sensor_log_probs.append(distribution.log_prob(selected))
                sensor_entropies.append(distribution.entropy())
            log_probs["shared_sensor"] = torch.stack(sensor_log_probs).sum()
            entropies["shared_sensor"] = torch.stack(sensor_entropies).sum()
        elif shared_sensor_trace is not None:
            raise ValueError("inactive shared sensor branch cannot have a saved trace")

        expected = expected_log_prob_terms(
            states, action, shared_sensor_active=sensor_active
        )
        if frozenset(log_probs) != expected:
            raise AssertionError(
                f"policy produced conditional terms {sorted(log_probs)}, expected {sorted(expected)}"
            )
        return JointPolicyEvaluation(
            log_prob_by_term=log_probs,
            entropy_by_term=entropies,
            plan_values_by_unit=output.plan_values_by_unit,
            motion_values_by_unit=output.motion_values_by_unit,
            sensor_value=output.sensor_value,
        )

    def _pack_policy_rollout(
        self,
        transitions: Sequence[JointTransition],
    ) -> tuple[_PackedPolicyRollout, np.ndarray, np.ndarray, np.ndarray, int]:
        """Convert validated rollout actions/traces to reusable device tensors."""

        transition_count = len(transitions)
        unit_count = self.space.unit_count
        objective_count = self.space.objective_count
        action_shape = (transition_count, unit_count)

        activation_actions = np.zeros(action_shape, dtype=np.int64)
        objective_actions = np.zeros(action_shape, dtype=np.int64)
        retarget_actions = np.zeros(action_shape, dtype=np.int64)
        movement_actions = np.full(
            action_shape, int(Movement.NEUTRAL), dtype=np.int64
        )
        placement_actions = np.zeros((*action_shape, 2), dtype=np.float32)
        activation_masks = np.zeros((*action_shape, 2), dtype=np.bool_)
        objective_masks = np.zeros(
            (*action_shape, objective_count), dtype=np.bool_
        )
        retarget_masks = np.zeros((*action_shape, 2), dtype=np.bool_)
        movement_masks = np.zeros((*action_shape, 3), dtype=np.bool_)
        activation_active = np.zeros(action_shape, dtype=np.bool_)
        placement_active = np.zeros(action_shape, dtype=np.bool_)
        objective_active = np.zeros(action_shape, dtype=np.bool_)
        retarget_active = np.zeros(action_shape, dtype=np.bool_)
        movement_active = np.zeros(action_shape, dtype=np.bool_)
        plan_active = np.zeros(action_shape, dtype=np.bool_)
        sensor_active = np.zeros(transition_count, dtype=np.bool_)
        old_plan_log_probs = np.zeros(action_shape, dtype=np.float32)
        old_motion_log_probs = np.zeros(action_shape, dtype=np.float32)
        old_sensor_log_probs = np.zeros(transition_count, dtype=np.float32)

        max_sensor_draws = max(
            (
                len(item.trace.shared_sensor.tokens)
                if item.trace.shared_sensor is not None
                else 0
            )
            for item in transitions
        )
        sensor_draw_tokens = np.zeros(
            (transition_count, max_sensor_draws), dtype=np.int64
        )
        sensor_draw_masks = np.zeros(
            (transition_count, max_sensor_draws, unit_count + 1), dtype=np.bool_
        )
        sensor_draw_active = np.zeros(
            (transition_count, max_sensor_draws), dtype=np.bool_
        )
        planning_names = ("activation", "placement", "objective", "retarget")
        active_term_count = 0

        for step_index, item in enumerate(transitions):
            terms = item.trace.log_prob_by_term
            active_term_count += len(terms)
            sensor_active[step_index] = "shared_sensor" in terms
            if sensor_active[step_index]:
                sensor_trace = item.trace.shared_sensor
                if sensor_trace is None:
                    raise ValueError("active shared sensor branch requires its saved trace")
                draw_count = len(sensor_trace.tokens)
                sensor_draw_tokens[step_index, :draw_count] = sensor_trace.tokens
                sensor_draw_masks[step_index, :draw_count] = np.asarray(
                    sensor_trace.masks, dtype=np.bool_
                )
                sensor_draw_active[step_index, :draw_count] = True
                old_sensor_log_probs[step_index] = float(terms["shared_sensor"])

            for slot, (state, action) in enumerate(zip(item.states, item.action.units)):
                unit_mask = item.mask.by_unit[slot]
                activation_masks[step_index, slot] = unit_mask.activation
                objective_masks[step_index, slot] = unit_mask.objective
                retarget_masks[step_index, slot] = unit_mask.retarget
                movement_masks[step_index, slot] = unit_mask.movement
                activation_actions[step_index, slot] = int(action.activate)
                retarget_actions[step_index, slot] = int(action.retarget)
                movement_actions[step_index, slot] = int(action.movement)
                placement_actions[step_index, slot] = action.placement

                activity = branch_activity(state, action)
                activation_active[step_index, slot] = activity.activation
                placement_active[step_index, slot] = activity.placement
                objective_active[step_index, slot] = activity.objective
                retarget_active[step_index, slot] = activity.retarget
                movement_active[step_index, slot] = activity.movement
                plan_active[step_index, slot] = any(
                    bool(getattr(activity, name)) for name in planning_names
                )
                if activity.objective:
                    objective_actions[step_index, slot] = int(action.objective_slot)

                prefix = f"unit/{slot}/"
                active_plan_names = tuple(
                    prefix + name
                    for name in planning_names
                    if bool(getattr(activity, name))
                )
                if active_plan_names:
                    old_plan_log_probs[step_index, slot] = float(
                        sum(terms[name] for name in active_plan_names)
                    )
                if activity.movement:
                    old_motion_log_probs[step_index, slot] = float(
                        terms[prefix + "movement"]
                    )

        def on_host(values: np.ndarray) -> Tensor:
            # Keep rollout-sized tensors out of accelerator memory.  The
            # selected rows are transferred in ``update`` after shuffling.
            return torch.from_numpy(values)

        packed = _PackedPolicyRollout(
            activation_actions=on_host(activation_actions),
            objective_actions=on_host(objective_actions),
            retarget_actions=on_host(retarget_actions),
            movement_actions=on_host(movement_actions),
            placement_actions=on_host(placement_actions),
            activation_masks=on_host(activation_masks),
            objective_masks=on_host(objective_masks),
            retarget_masks=on_host(retarget_masks),
            movement_masks=on_host(movement_masks),
            activation_active=on_host(activation_active),
            placement_active=on_host(placement_active),
            objective_active=on_host(objective_active),
            retarget_active=on_host(retarget_active),
            movement_active=on_host(movement_active),
            plan_active=on_host(plan_active),
            sensor_active=on_host(sensor_active),
            sensor_draw_tokens=on_host(sensor_draw_tokens),
            sensor_draw_masks=on_host(sensor_draw_masks),
            sensor_draw_active=on_host(sensor_draw_active),
            old_plan_log_probs=on_host(old_plan_log_probs),
            old_motion_log_probs=on_host(old_motion_log_probs),
            old_sensor_log_probs=on_host(old_sensor_log_probs),
        )
        return (
            packed,
            plan_active,
            movement_active,
            sensor_active,
            active_term_count,
        )

    def _evaluate_packed_output(
        self,
        output: JointNetworkOutput,
        packed: _PackedPolicyRollout,
    ) -> _PackedPolicyEvaluation:
        """Evaluate a complete minibatch without Python transition/unit loops."""

        batch_size, unit_count, hidden_dim = output.actor_features.shape
        flat_count = batch_size * unit_count
        unit_shape = (batch_size, unit_count)

        def categorical_branch(
            logits: Tensor,
            masks: Tensor,
            actions: Tensor,
            active: Tensor,
        ) -> tuple[Tensor, Tensor]:
            indices = torch.nonzero(active.reshape(-1), as_tuple=False).squeeze(-1)
            flat_log_probs = logits.new_zeros(flat_count)
            flat_entropies = logits.new_zeros(flat_count)
            if indices.numel():
                branch_log_probs, branch_entropies = (
                    _batched_masked_categorical_statistics(
                        logits.reshape(flat_count, logits.shape[-1]).index_select(
                            0, indices
                        ),
                        masks.reshape(flat_count, masks.shape[-1]).index_select(
                            0, indices
                        ),
                        actions.reshape(flat_count).index_select(0, indices),
                    )
                )
                flat_log_probs = flat_log_probs.index_copy(
                    0, indices, branch_log_probs
                )
                flat_entropies = flat_entropies.index_copy(
                    0, indices, branch_entropies
                )
            return flat_log_probs.view(unit_shape), flat_entropies.view(unit_shape)

        activation_log_probs, activation_entropies = categorical_branch(
            output.activation_logits,
            packed.activation_masks,
            packed.activation_actions,
            packed.activation_active,
        )
        objective_log_probs, objective_entropies = categorical_branch(
            output.objective_logits,
            packed.objective_masks,
            packed.objective_actions,
            packed.objective_active,
        )
        retarget_log_probs, retarget_entropies = categorical_branch(
            output.retarget_logits,
            packed.retarget_masks,
            packed.retarget_actions,
            packed.retarget_active,
        )

        conditioned_indices = torch.nonzero(
            packed.objective_active.reshape(-1), as_tuple=False
        ).squeeze(-1)
        flat_actor_features = output.actor_features.reshape(flat_count, hidden_dim)
        flat_objectives = packed.objective_actions.reshape(flat_count)
        flat_placement_actions = packed.placement_actions.reshape(flat_count, 2)
        flat_movement_masks = packed.movement_masks.reshape(flat_count, 3)
        flat_movement_actions = packed.movement_actions.reshape(flat_count)
        placement_log_probs = output.actor_features.new_zeros(flat_count)
        placement_entropies = output.actor_features.new_zeros(flat_count)
        conditioned_movement_log_probs = output.actor_features.new_zeros(flat_count)
        conditioned_movement_entropies = output.actor_features.new_zeros(flat_count)
        if conditioned_indices.numel():
            mean, log_std, conditioned_movement_logits = (
                self.network.condition_on_objective(
                    flat_actor_features.index_select(0, conditioned_indices),
                    flat_objectives.index_select(0, conditioned_indices),
                )
            )
            conditioned_log_probs, conditioned_entropies = (
                _batched_masked_categorical_statistics(
                    conditioned_movement_logits,
                    flat_movement_masks.index_select(0, conditioned_indices),
                    flat_movement_actions.index_select(0, conditioned_indices),
                )
            )
            conditioned_movement_log_probs = (
                conditioned_movement_log_probs.index_copy(
                    0, conditioned_indices, conditioned_log_probs
                )
            )
            conditioned_movement_entropies = (
                conditioned_movement_entropies.index_copy(
                    0, conditioned_indices, conditioned_entropies
                )
            )

            placement_in_conditioned = packed.placement_active.reshape(
                -1
            ).index_select(0, conditioned_indices)
            placement_positions = torch.nonzero(
                placement_in_conditioned, as_tuple=False
            ).squeeze(-1)
            if placement_positions.numel():
                placement_indices = conditioned_indices.index_select(
                    0, placement_positions
                )
                placement_mean = mean.index_select(0, placement_positions)
                placement_log_std = log_std.index_select(0, placement_positions)
                selected_placements = flat_placement_actions.index_select(
                    0, placement_indices
                )
                active_placement_log_probs = _squashed_normal_log_prob(
                    placement_mean,
                    placement_log_std,
                    selected_placements,
                )
                active_placement_entropies = Normal(
                    placement_mean, placement_log_std.exp()
                ).entropy().sum(dim=-1)
                placement_log_probs = placement_log_probs.index_copy(
                    0, placement_indices, active_placement_log_probs
                )
                placement_entropies = placement_entropies.index_copy(
                    0, placement_indices, active_placement_entropies
                )

        base_movement_active = packed.movement_active & ~packed.objective_active
        base_movement_log_probs, base_movement_entropies = categorical_branch(
            output.movement_logits,
            packed.movement_masks,
            packed.movement_actions,
            base_movement_active,
        )
        placement_log_probs = placement_log_probs.view(unit_shape)
        placement_entropies = placement_entropies.view(unit_shape)
        movement_log_probs = base_movement_log_probs + (
            conditioned_movement_log_probs.view(unit_shape)
        )
        movement_entropies = base_movement_entropies + (
            conditioned_movement_entropies.view(unit_shape)
        )
        motion_log_probs = movement_log_probs
        motion_entropies = movement_entropies

        plan_log_probs = torch.stack(
            (
                activation_log_probs,
                placement_log_probs,
                objective_log_probs,
                retarget_log_probs,
            ),
            dim=-1,
        ).sum(dim=-1)
        plan_entropies = torch.stack(
            (
                activation_entropies,
                placement_entropies,
                objective_entropies,
                retarget_entropies,
            ),
            dim=-1,
        ).sum(dim=-1)

        draw_count = packed.sensor_draw_tokens.shape[1]
        if draw_count:
            sensor_logits = output.sensor_logits.unsqueeze(1).expand(
                -1, draw_count, -1
            )
            sensor_draw_log_probs, sensor_draw_entropies = (
                _batched_masked_categorical_statistics(
                    sensor_logits,
                    packed.sensor_draw_masks,
                    packed.sensor_draw_tokens,
                )
            )
            zero_by_draw = sensor_draw_log_probs * 0.0
            sensor_log_probs = torch.where(
                packed.sensor_draw_active,
                sensor_draw_log_probs,
                zero_by_draw,
            ).sum(dim=1)
            sensor_entropies = torch.where(
                packed.sensor_draw_active,
                sensor_draw_entropies,
                zero_by_draw,
            ).sum(dim=1)
        else:
            sensor_log_probs = output.sensor_value * 0.0
            sensor_entropies = output.sensor_value * 0.0

        return _PackedPolicyEvaluation(
            plan_log_probs=plan_log_probs,
            plan_entropies=plan_entropies,
            motion_log_probs=motion_log_probs,
            motion_entropies=motion_entropies,
            sensor_log_probs=sensor_log_probs,
            sensor_entropies=sensor_entropies,
        )

    def predict_values(
        self, observations: Sequence[np.ndarray]
    ) -> tuple[tuple[float, ...], float]:
        """Compatibility view returning motion and sensor values."""

        with torch.no_grad():
            output = self.network(self._observation_tensor(observations)).item(0)
        return (
            tuple(float(value) for value in output.values_by_unit.tolist()),
            float(output.team_value.item()),
        )

    def predict_branch_values(
        self, observations: Sequence[np.ndarray]
    ) -> tuple[tuple[float, ...], tuple[float, ...], float]:
        """Return planning, motion and shared-sensor critic predictions."""

        with torch.no_grad():
            output = self.network(self._observation_tensor(observations)).item(0)
        return (
            tuple(float(value) for value in output.plan_values_by_unit.tolist()),
            tuple(float(value) for value in output.motion_values_by_unit.tolist()),
            float(output.sensor_value.item()),
        )

    @staticmethod
    def _linear_schedule(start: float, end: float, index: int, duration: int) -> float:
        fraction = min(max(index / max(duration, 1), 0.0), 1.0)
        return float(start + fraction * (end - start))

    @staticmethod
    def _compute_gae(
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """GAE with terminal bootstrap rules and truncation sequence boundaries."""

        if not (
            rewards.shape
            == terminated.shape
            == truncated.shape
            == values.shape
            == next_values.shape
        ):
            raise ValueError("GAE arrays must have identical shapes")
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = np.zeros(rewards.shape[1:], dtype=np.float32)
        for index in range(rewards.shape[0] - 1, -1, -1):
            bootstrap = (~terminated[index]).astype(np.float32)
            continuation = (~(terminated[index] | truncated[index])).astype(np.float32)
            delta = (
                rewards[index]
                + gamma * next_values[index] * bootstrap
                - values[index]
            )
            gae = delta + gamma * gae_lambda * continuation * gae
            advantages[index] = gae
        return advantages, advantages + values

    def _predict_batched_values(
        self, observations: Sequence[np.ndarray] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        plan_values: list[np.ndarray] = []
        motion_values: list[np.ndarray] = []
        sensor_values: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(observations), self.config.value_inference_batch_size):
                batch_array = np.asarray(
                    observations[
                        start : start + self.config.value_inference_batch_size
                    ],
                    dtype=np.float32,
                )
                batch = torch.as_tensor(
                    batch_array,
                    dtype=torch.float32,
                    device=self.device,
                )
                output = self.network(batch)
                plan_values.append(output.plan_values_by_unit.detach().cpu().numpy())
                motion_values.append(output.motion_values_by_unit.detach().cpu().numpy())
                sensor_values.append(output.sensor_value.detach().cpu().numpy())
        return (
            np.concatenate(plan_values).astype(np.float32, copy=False),
            np.concatenate(motion_values).astype(np.float32, copy=False),
            np.concatenate(sensor_values).astype(np.float32, copy=False),
        )

    @staticmethod
    def _normalize_active(
        values: np.ndarray,
        active: np.ndarray,
        *,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        result = np.zeros_like(values, dtype=np.float32)
        selected = values[active]
        if selected.size == 0:
            return result
        if weights is None:
            mean = float(selected.mean())
            std = float(selected.std())
        else:
            if weights.shape != values.shape:
                raise ValueError("advantage weights must match the value shape")
            selected_weights = weights[active].astype(np.float64, copy=False)
            weight_sum = float(selected_weights.sum())
            if weight_sum <= 0.0 or not math.isfinite(weight_sum):
                raise ValueError("active advantage weights must have positive finite mass")
            mean = float(np.sum(selected * selected_weights) / weight_sum)
            variance = float(
                np.sum(np.square(selected - mean) * selected_weights) / weight_sum
            )
            std = math.sqrt(max(variance, 0.0))
        result[active] = (
            (selected - mean) / (std + 1e-8)
            if selected.size > 1 and std > 1e-8
            else selected - mean
        )
        return result

    @staticmethod
    def _episode_balanced_plan_weights(
        transitions: Sequence[object], active: np.ndarray
    ) -> np.ndarray:
        """Give each unit one total planning weight per episode.

        STAGED ``NO`` and ACTIVE retarget decisions can occur on every step.
        Without this correction a long-lived unit would dominate the planning
        loss merely because it generated more decisions.  The final rescaling
        keeps the mean active-sample weight at one for stable optimizer scale.
        """

        if active.ndim != 2 or active.shape[0] != len(transitions):
            raise ValueError("planning activity has the wrong rollout shape")
        weights = np.zeros(active.shape, dtype=np.float32)
        episode_start = 0
        for index, item in enumerate(transitions):
            boundary = bool(
                getattr(item, "team_terminated") or getattr(item, "team_truncated")
            )
            if not boundary and index + 1 != len(transitions):
                continue
            episode_slice = slice(episode_start, index + 1)
            counts = active[episode_slice].sum(axis=0)
            for slot, count in enumerate(counts.tolist()):
                if count:
                    slot_active = active[episode_slice, slot]
                    weights[episode_slice, slot][slot_active] = 1.0 / float(count)
            episode_start = index + 1
        total = float(weights.sum())
        active_count = int(active.sum())
        if active_count and total > 0.0:
            weights *= float(active_count) / total
        return weights

    def update(
        self,
        buffer: JointTrajectoryBuffer,
        *,
        clear_buffer: bool = True,
    ) -> dict[str, float]:
        """Run PPO over a rollout; by default consume the supplied buffer."""

        if not isinstance(buffer, JointTrajectoryBuffer):
            raise TypeError("update expects a JointTrajectoryBuffer")
        if buffer.space != self.space or buffer.observation_dim != self.config.observation_dim:
            raise ValueError("trajectory buffer and joint policy dimensions differ")
        transitions = buffer.items
        if not transitions:
            return self.last_metrics
        missing_branch_data = [
            index
            for index, item in enumerate(transitions)
            if item.plan_rewards is None
            or item.motion_rewards is None
            or item.sensor_reward is None
            or item.trace.plan_values_by_unit is None
            or item.trace.motion_values_by_unit is None
            or item.trace.sensor_value is None
        ]
        if missing_branch_data:
            preview = ", ".join(str(value) for value in missing_branch_data[:8])
            raise ValueError(
                "schema 3 PPO updates require explicit plan/motion/sensor rewards "
                f"and values; missing transition indices: {preview}"
            )

        # Observation matrices dominate large scenarios (for example H02 is
        # [8759, 286, 382], or 3.57 GiB).  Keep the simulator-owned arrays on
        # the host and stack only the current inference/update minibatch.
        next_observations = tuple(item.next_observations for item in transitions)
        old_plan_values = np.asarray(
            [item.trace.plan_values_by_unit for item in transitions],
            dtype=np.float32,
        )
        old_motion_values = np.asarray(
            [item.trace.motion_values_by_unit for item in transitions],
            dtype=np.float32,
        )
        old_sensor_values = np.asarray(
            [item.trace.sensor_value for item in transitions],
            dtype=np.float32,
        )
        plan_returns = np.asarray(
            [item.plan_rewards for item in transitions],
            dtype=np.float32,
        )
        motion_rewards = np.asarray(
            [item.motion_rewards for item in transitions],
            dtype=np.float32,
        )
        sensor_rewards = np.asarray(
            [item.sensor_reward for item in transitions],
            dtype=np.float32,
        )
        team_rewards = np.asarray(
            [item.team_reward for item in transitions], dtype=np.float32
        )
        motion_rewards += self.config.unit_team_reward_weight * team_rewards[:, None]
        unit_terminated = np.asarray(
            [item.terminated for item in transitions], dtype=np.bool_
        )
        unit_truncated = np.asarray(
            [item.truncated for item in transitions], dtype=np.bool_
        )
        team_terminated = np.asarray(
            [item.team_terminated for item in transitions], dtype=np.bool_
        )
        team_truncated = np.asarray(
            [item.team_truncated for item in transitions], dtype=np.bool_
        )
        _, next_motion_values, next_sensor_values = self._predict_batched_values(
            next_observations
        )
        del next_observations
        # Planning credit is already the complete, undiscounted episode result
        # assigned to each unit by the rollout collector.  Applying step GAE to
        # it would reintroduce the long-horizon decay this branch removes.
        plan_advantages = plan_returns - old_plan_values
        motion_advantages, motion_returns = self._compute_gae(
            motion_rewards,
            unit_terminated,
            unit_truncated,
            old_motion_values,
            next_motion_values,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        sensor_advantages, sensor_returns = self._compute_gae(
            sensor_rewards[:, None],
            team_terminated[:, None],
            team_truncated[:, None],
            old_sensor_values[:, None],
            next_sensor_values[:, None],
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        sensor_advantages = sensor_advantages[:, 0]
        sensor_returns = sensor_returns[:, 0]

        (
            packed_rollout,
            plan_active,
            motion_active,
            sensor_active,
            active_term_count,
        ) = self._pack_policy_rollout(transitions)
        plan_sample_weights = self._episode_balanced_plan_weights(
            transitions, plan_active
        )
        normalized_plan_advantages = self._normalize_active(
            plan_advantages,
            plan_active,
            weights=plan_sample_weights,
        )
        normalized_motion_advantages = self._normalize_active(
            motion_advantages, motion_active
        )
        normalized_sensor_advantages = self._normalize_active(
            sensor_advantages, sensor_active
        )

        # Rollout-sized training targets also remain on the host.  At most one
        # minibatch of each tensor is resident on the accelerator at a time.
        def on_host(values: np.ndarray) -> Tensor:
            return torch.from_numpy(values)

        old_plan_values_tensor = on_host(old_plan_values)
        plan_returns_tensor = on_host(plan_returns)
        old_motion_values_tensor = on_host(old_motion_values)
        motion_returns_tensor = on_host(motion_returns)
        old_sensor_values_tensor = on_host(old_sensor_values)
        sensor_returns_tensor = on_host(sensor_returns)
        normalized_plan_advantages_tensor = on_host(normalized_plan_advantages)
        normalized_motion_advantages_tensor = on_host(
            normalized_motion_advantages
        )
        normalized_sensor_advantages_tensor = on_host(
            normalized_sensor_advantages
        )
        plan_sample_weights_tensor = on_host(plan_sample_weights)

        learning_rate = self._linear_schedule(
            self.config.learning_rate,
            self.config.learning_rate_final,
            self.update_count,
            self.config.learning_rate_decay_updates,
        )
        entropy_coef = self._linear_schedule(
            self.config.entropy_coef,
            self.config.entropy_final_coef,
            self.update_count,
            self.config.entropy_decay_updates,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

        totals = {
            "policy_loss": 0.0,
            "plan_policy_loss": 0.0,
            "motion_policy_loss": 0.0,
            "sensor_policy_loss": 0.0,
            "value_loss": 0.0,
            "plan_value_loss": 0.0,
            "motion_value_loss": 0.0,
            "sensor_value_loss": 0.0,
            "entropy": 0.0,
            "plan_entropy": 0.0,
            "motion_entropy": 0.0,
            "sensor_entropy": 0.0,
            "approx_kl": 0.0,
            "plan_approx_kl": 0.0,
            "motion_approx_kl": 0.0,
            "sensor_approx_kl": 0.0,
            "clip_fraction": 0.0,
            "plan_clip_fraction": 0.0,
            "motion_clip_fraction": 0.0,
            "sensor_clip_fraction": 0.0,
            "grad_norm": 0.0,
        }
        minibatch_updates = 0
        epochs_ran = 0
        early_stopped = False
        early_stop_kl = 0.0
        max_observed_kl = 0.0
        max_observed_plan_kl = 0.0
        max_observed_motion_kl = 0.0
        max_observed_sensor_kl = 0.0
        plan_decisions = int(plan_active.sum())
        motion_decisions = int(motion_active.sum())
        sensor_decisions = int(sensor_active.sum())
        actor_decisions = plan_decisions + motion_decisions + sensor_decisions
        total_plan_weight = float(plan_sample_weights.sum())
        for epoch in range(self.config.update_epochs):
            epochs_ran = epoch + 1
            permutation = self._rng.permutation(len(transitions))
            for start in range(0, len(transitions), self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                index_tensor = torch.as_tensor(
                    indices, dtype=torch.long
                )
                observation_batch = torch.as_tensor(
                    np.asarray(
                        [transitions[int(index)].observations for index in indices],
                        dtype=np.float32,
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                output_batch = self.network(observation_batch)
                packed_batch = packed_rollout.select(
                    index_tensor, device=self.device
                )

                def device_batch(values: Tensor) -> Tensor:
                    return values.index_select(0, index_tensor).to(self.device)

                evaluation = self._evaluate_packed_output(
                    output_batch, packed_batch
                )
                plan_mask = packed_batch.plan_active
                motion_mask = packed_batch.movement_active
                sensor_mask = packed_batch.sensor_active
                new_plan_log_probs = evaluation.plan_log_probs[plan_mask]
                old_plan_log_probs = packed_batch.old_plan_log_probs[plan_mask]
                plan_advantage_batch = device_batch(
                    normalized_plan_advantages_tensor
                )[plan_mask]
                plan_sample_weight_batch = device_batch(
                    plan_sample_weights_tensor
                )
                plan_weight_batch = plan_sample_weight_batch[plan_mask]
                plan_entropies = evaluation.plan_entropies[plan_mask]
                new_motion_log_probs = evaluation.motion_log_probs[motion_mask]
                old_motion_log_probs = packed_batch.old_motion_log_probs[motion_mask]
                motion_advantage_batch = device_batch(
                    normalized_motion_advantages_tensor
                )[motion_mask]
                motion_entropies = evaluation.motion_entropies[motion_mask]
                new_sensor_log_probs = evaluation.sensor_log_probs[sensor_mask]
                old_sensor_log_probs = packed_batch.old_sensor_log_probs[sensor_mask]
                sensor_advantage_batch = device_batch(
                    normalized_sensor_advantages_tensor
                )[sensor_mask]
                sensor_entropies = evaluation.sensor_entropies[sensor_mask]

                zero = output_batch.sensor_value.sum() * 0.0
                # Use a rollout-global denominator scaled by transition batch
                # size.  Renormalizing by each minibatch's observed weight
                # would undo the per-unit/per-episode correction whenever a
                # batch happens to contain mostly long-lived units.
                plan_loss_normalizer = (
                    total_plan_weight * float(len(indices)) / float(len(transitions))
                    if total_plan_weight > 0.0
                    else None
                )
                plan_loss, plan_kl, plan_clip = self._ppo_actor_loss(
                    new_plan_log_probs,
                    old_plan_log_probs,
                    plan_advantage_batch,
                    zero,
                    weights=plan_weight_batch,
                    loss_normalizer=plan_loss_normalizer,
                )
                motion_loss, motion_kl, motion_clip = self._ppo_actor_loss(
                    new_motion_log_probs,
                    old_motion_log_probs,
                    motion_advantage_batch,
                    zero,
                )
                sensor_loss, sensor_kl, sensor_clip = self._ppo_actor_loss(
                    new_sensor_log_probs,
                    old_sensor_log_probs,
                    sensor_advantage_batch,
                    zero,
                )
                policy_loss = (
                    plan_loss
                    + motion_loss
                    + self.config.sensor_policy_coef * sensor_loss
                )
                kl_values = [
                    value
                    for value, present in (
                        (plan_kl, new_plan_log_probs.numel() > 0),
                        (motion_kl, new_motion_log_probs.numel() > 0),
                        (sensor_kl, new_sensor_log_probs.numel() > 0),
                    )
                    if present
                ]
                approximate_kl = torch.stack(kl_values).max() if kl_values else zero
                clip_values = [
                    value
                    for value, present in (
                        (plan_clip, new_plan_log_probs.numel() > 0),
                        (motion_clip, new_motion_log_probs.numel() > 0),
                        (sensor_clip, new_sensor_log_probs.numel() > 0),
                    )
                    if present
                ]
                clip_fraction = (
                    torch.stack(clip_values).mean() if clip_values else zero
                )
                observed_kl = float(approximate_kl.detach().item())
                max_observed_kl = max(max_observed_kl, observed_kl)
                if new_plan_log_probs.numel():
                    max_observed_plan_kl = max(
                        max_observed_plan_kl, float(plan_kl.detach().item())
                    )
                if new_motion_log_probs.numel():
                    max_observed_motion_kl = max(
                        max_observed_motion_kl, float(motion_kl.detach().item())
                    )
                if new_sensor_log_probs.numel():
                    max_observed_sensor_kl = max(
                        max_observed_sensor_kl, float(sensor_kl.detach().item())
                    )
                if (
                    self.config.target_kl > 0.0
                    and observed_kl > 1.5 * self.config.target_kl
                ):
                    early_stopped = True
                    early_stop_kl = observed_kl
                    break

                old_plan_batch = device_batch(old_plan_values_tensor)
                plan_return_batch = device_batch(plan_returns_tensor)
                old_motion_batch = device_batch(old_motion_values_tensor)
                motion_return_batch = device_batch(motion_returns_tensor)
                old_sensor_batch = device_batch(old_sensor_values_tensor)
                sensor_return_batch = device_batch(sensor_returns_tensor)
                plan_value_loss = self._clipped_value_loss(
                    output_batch.plan_values_by_unit,
                    old_plan_batch,
                    plan_return_batch,
                    mask=plan_mask,
                    weights=plan_sample_weight_batch,
                    loss_normalizer=plan_loss_normalizer,
                )
                motion_value_loss = self._clipped_value_loss(
                    output_batch.motion_values_by_unit,
                    old_motion_batch,
                    motion_return_batch,
                    mask=motion_mask,
                )
                sensor_value_loss = self._clipped_value_loss(
                    output_batch.sensor_value,
                    old_sensor_batch,
                    sensor_return_batch,
                )
                active_value_losses = [sensor_value_loss]
                if bool(plan_active[indices].any()):
                    active_value_losses.append(plan_value_loss)
                if bool(motion_active[indices].any()):
                    active_value_losses.append(motion_value_loss)
                value_loss = torch.stack(active_value_losses).mean()
                if plan_entropies.numel():
                    entropy_weights = plan_weight_batch
                    entropy_normalizer = float(
                        plan_loss_normalizer
                        if plan_loss_normalizer is not None
                        else max(float(entropy_weights.sum().item()), 1e-8)
                    )
                    plan_entropy = (
                        plan_entropies * entropy_weights
                    ).sum() / entropy_normalizer
                else:
                    plan_entropy = zero
                motion_entropy = (
                    motion_entropies.mean()
                    if motion_entropies.numel()
                    else zero
                )
                sensor_entropy = (
                    sensor_entropies.mean()
                    if sensor_entropies.numel()
                    else zero
                )
                active_entropies = [
                    value
                    for value, present in (
                        (plan_entropy, plan_entropies.numel() > 0),
                        (motion_entropy, motion_entropies.numel() > 0),
                        (sensor_entropy, sensor_entropies.numel() > 0),
                    )
                    if present
                ]
                entropy = torch.stack(active_entropies).mean() if active_entropies else zero
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - entropy_coef * entropy
                )
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("joint PPO loss became NaN or infinite")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )
                if not bool(torch.isfinite(grad_norm).item()):
                    raise FloatingPointError("joint PPO gradient became NaN or infinite")
                self.optimizer.step()

                totals["policy_loss"] += float(policy_loss.detach().item())
                totals["plan_policy_loss"] += float(plan_loss.detach().item())
                totals["motion_policy_loss"] += float(motion_loss.detach().item())
                totals["sensor_policy_loss"] += float(sensor_loss.detach().item())
                totals["value_loss"] += float(value_loss.detach().item())
                totals["plan_value_loss"] += float(plan_value_loss.detach().item())
                totals["motion_value_loss"] += float(motion_value_loss.detach().item())
                totals["sensor_value_loss"] += float(sensor_value_loss.detach().item())
                totals["entropy"] += float(entropy.detach().item())
                totals["plan_entropy"] += float(plan_entropy.detach().item())
                totals["motion_entropy"] += float(motion_entropy.detach().item())
                totals["sensor_entropy"] += float(sensor_entropy.detach().item())
                totals["approx_kl"] += float(approximate_kl.detach().item())
                totals["plan_approx_kl"] += float(plan_kl.detach().item())
                totals["motion_approx_kl"] += float(motion_kl.detach().item())
                totals["sensor_approx_kl"] += float(sensor_kl.detach().item())
                totals["clip_fraction"] += float(clip_fraction.detach().item())
                totals["plan_clip_fraction"] += float(plan_clip.detach().item())
                totals["motion_clip_fraction"] += float(motion_clip.detach().item())
                totals["sensor_clip_fraction"] += float(sensor_clip.detach().item())
                totals["grad_norm"] += float(grad_norm.detach().item())
                minibatch_updates += 1

                # Python keeps loop locals alive until their next assignment.
                # Drop the completed graph/output explicitly so the following
                # forward does not overlap it with a stale minibatch.
                del loss, policy_loss, value_loss, entropy, grad_norm
                del plan_loss, motion_loss, sensor_loss
                del plan_value_loss, motion_value_loss, sensor_value_loss
                del plan_entropy, motion_entropy, sensor_entropy
                del evaluation, output_batch, packed_batch, observation_batch
                del new_plan_log_probs, new_motion_log_probs, new_sensor_log_probs
                del old_plan_log_probs, old_motion_log_probs, old_sensor_log_probs
                del plan_entropies, motion_entropies, sensor_entropies
                del plan_advantage_batch, motion_advantage_batch
                del sensor_advantage_batch, plan_sample_weight_batch
                del plan_weight_batch, plan_mask, motion_mask, sensor_mask
                del old_plan_batch, old_motion_batch, old_sensor_batch
                del plan_return_batch, motion_return_batch, sensor_return_batch
                del plan_kl, motion_kl, sensor_kl, approximate_kl
                del plan_clip, motion_clip, sensor_clip, clip_fraction
                del active_value_losses, active_entropies, kl_values, clip_values, zero
            if early_stopped:
                break

        if clear_buffer:
            buffer.clear()
        self.update_count += 1
        self.transition_count += len(transitions)
        divisor = max(minibatch_updates, 1)
        self.last_metrics = {key: value / divisor for key, value in totals.items()}
        weighted_plan_advantage_mean = (
            float(
                np.average(
                    plan_advantages[plan_active],
                    weights=plan_sample_weights[plan_active],
                )
            )
            if bool(plan_active.any())
            else 0.0
        )
        self.last_metrics.update(
            {
                "joint_steps": float(len(transitions)),
                "actor_decisions": float(actor_decisions),
                "plan_decisions": float(plan_decisions),
                "motion_decisions": float(motion_decisions),
                "sensor_decisions": float(sensor_decisions),
                "plan_value_samples": float(plan_active.sum()),
                "motion_value_samples": float(motion_active.sum()),
                "sensor_value_samples": float(len(transitions)),
                "active_log_prob_terms": float(active_term_count),
                "epochs_ran": float(epochs_ran),
                "minibatch_updates": float(minibatch_updates),
                "early_stopped": float(early_stopped),
                "early_stop_kl": float(early_stop_kl),
                "max_approx_kl": float(max_observed_kl),
                "max_plan_approx_kl": float(max_observed_plan_kl),
                "max_motion_approx_kl": float(max_observed_motion_kl),
                "max_sensor_approx_kl": float(max_observed_sensor_kl),
                "learning_rate": learning_rate,
                "entropy_coef": entropy_coef,
                "plan_advantage_mean": weighted_plan_advantage_mean,
                "motion_advantage_mean": float(
                    motion_advantages[motion_active].mean()
                )
                if bool(motion_active.any())
                else 0.0,
                "sensor_advantage_mean": float(
                    sensor_advantages[sensor_active].mean()
                )
                if bool(sensor_active.any())
                else 0.0,
            }
        )
        # Keep these aliases for existing result readers while new dashboards
        # migrate to the explicit branch names above.
        self.last_metrics["unit_policy_loss"] = (
            self.last_metrics["plan_policy_loss"]
            + self.last_metrics["motion_policy_loss"]
        )
        self.last_metrics["unit_value_samples"] = self.last_metrics[
            "motion_value_samples"
        ]
        self.last_metrics["unit_advantage_mean"] = self.last_metrics[
            "motion_advantage_mean"
        ]
        self.last_metrics["team_advantage_mean"] = self.last_metrics[
            "sensor_advantage_mean"
        ]
        return self.last_metrics

    def _ppo_actor_loss(
        self,
        new_log_probs: Tensor | Sequence[Tensor],
        old_log_probs: Tensor | Sequence[float],
        advantages: Tensor | Sequence[float],
        zero: Tensor,
        *,
        weights: Tensor | Sequence[float] | None = None,
        loss_normalizer: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if isinstance(new_log_probs, Tensor):
            new = new_log_probs
        else:
            if not new_log_probs:
                return zero, zero, zero
            new = torch.stack(tuple(new_log_probs))
        if new.numel() == 0:
            return zero, zero, zero
        old = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantage = torch.as_tensor(
            advantages, dtype=torch.float32, device=self.device
        )
        if not (new.shape == old.shape == advantage.shape):
            raise ValueError("PPO actor samples have incompatible shapes")
        log_ratio = new - old
        ratio = torch.exp(log_ratio)
        unclipped = ratio * advantage
        clipped = torch.clamp(
            ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
        ) * advantage
        if weights is None:
            sample_weights = torch.ones_like(ratio)
        else:
            sample_weights = torch.as_tensor(
                weights, dtype=torch.float32, device=self.device
            )
            if sample_weights.shape != ratio.shape:
                raise ValueError("PPO actor weights have the wrong shape")
            if not bool(torch.isfinite(sample_weights).all().item()) or bool(
                (sample_weights < 0.0).any().item()
            ):
                raise ValueError("PPO actor weights must be finite and non-negative")
        weight_sum = sample_weights.sum().clamp_min(torch.finfo(ratio.dtype).eps)
        if loss_normalizer is None:
            loss_denominator = weight_sum
        else:
            if not math.isfinite(loss_normalizer) or loss_normalizer <= 0.0:
                raise ValueError("PPO loss normalizer must be positive and finite")
            loss_denominator = torch.as_tensor(
                loss_normalizer, dtype=ratio.dtype, device=self.device
            )
        loss = -(
            torch.minimum(unclipped, clipped) * sample_weights
        ).sum() / loss_denominator
        with torch.no_grad():
            approximate_kl = (
                ((ratio - 1.0) - log_ratio) * sample_weights
            ).sum() / weight_sum
            clip_fraction = (
                (
                    (torch.abs(ratio - 1.0) > self.config.clip_ratio).float()
                    * sample_weights
                ).sum()
                / weight_sum
            )
        return loss, approximate_kl, clip_fraction

    def _clipped_value_loss(
        self,
        predicted: Tensor,
        old: Tensor,
        target: Tensor,
        *,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
        loss_normalizer: float | None = None,
    ) -> Tensor:
        losses = F.smooth_l1_loss(predicted, target, reduction="none")
        if self.config.value_clip_ratio > 0.0:
            clipped = old + torch.clamp(
                predicted - old,
                -self.config.value_clip_ratio,
                self.config.value_clip_ratio,
            )
            clipped_losses = F.smooth_l1_loss(clipped, target, reduction="none")
            losses = torch.maximum(losses, clipped_losses)
        if mask is not None:
            if mask.shape != losses.shape:
                raise ValueError("value-loss mask has the wrong shape")
            if not bool(mask.any().item()):
                return predicted.sum() * 0.0
            losses = losses[mask]
            if weights is not None:
                if weights.shape != mask.shape:
                    raise ValueError("value-loss weights have the wrong shape")
                weights = weights[mask]
        if weights is None:
            return losses.mean()
        if weights.shape != losses.shape:
            raise ValueError("value-loss weights have the wrong shape")
        if not bool(torch.isfinite(weights).all().item()) or bool(
            (weights < 0.0).any().item()
        ):
            raise ValueError("value-loss weights must be finite and non-negative")
        if loss_normalizer is None:
            denominator = weights.sum().clamp_min(torch.finfo(losses.dtype).eps)
        else:
            if not math.isfinite(loss_normalizer) or loss_normalizer <= 0.0:
                raise ValueError("value-loss normalizer must be positive and finite")
            denominator = torch.as_tensor(
                loss_normalizer, dtype=losses.dtype, device=losses.device
            )
        return (losses * weights).sum() / denominator

    def finish_rollout(
        self,
        buffer: JointTrajectoryBuffer,
        *,
        episode_count: int,
        clear_buffer: bool = True,
    ) -> dict[str, float]:
        """Update once from a rollout containing one or more complete episodes."""

        if (
            isinstance(episode_count, bool)
            or not isinstance(episode_count, int)
            or episode_count <= 0
        ):
            raise ValueError("episode_count must be a positive integer")
        if not isinstance(buffer, JointTrajectoryBuffer):
            raise TypeError("finish_rollout expects a JointTrajectoryBuffer")
        transitions = buffer.items
        if not transitions:
            raise ValueError("rollout buffer must be non-empty")
        completed_episodes = sum(
            bool(item.team_terminated or item.team_truncated) for item in transitions
        )
        if not (
            transitions[-1].team_terminated or transitions[-1].team_truncated
        ):
            raise ValueError("rollout must end at a complete team episode boundary")
        if completed_episodes != episode_count:
            raise ValueError(
                "episode_count does not match complete team boundaries in rollout: "
                f"expected {episode_count}, found {completed_episodes}"
            )
        metrics = self.update(buffer, clear_buffer=clear_buffer)
        self.episode_count += episode_count
        return metrics

    def finish_episode(
        self, buffer: JointTrajectoryBuffer, *, clear_buffer: bool = True
    ) -> dict[str, float]:
        return self.finish_rollout(
            buffer,
            episode_count=1,
            clear_buffer=clear_buffer,
        )

    def set_training(self, training: bool) -> None:
        self.training = bool(training)
        self.network.train(self.training)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically save schema, config, weights, optimizer and RNG state."""

        checkpoint: dict[str, object] = {
            "algorithm": self.ALGORITHM,
            "schema_version": self.CHECKPOINT_SCHEMA_VERSION,
            "space": asdict(self.space),
            "config": asdict(self.config),
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "transition_count": self.transition_count,
            "episode_count": self.episode_count,
            "last_metrics": self.last_metrics,
            "numpy_rng_state": self._rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
        }
        if self.device.type == "cuda":
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state(self.device)
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        try:
            torch.save(checkpoint, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def _load_checkpoint(
        self,
        checkpoint: object,
        *,
        load_optimizer: bool,
    ) -> None:
        if not isinstance(checkpoint, dict):
            raise ValueError("joint PPO checkpoint is not a dictionary")
        if checkpoint.get("algorithm") != self.ALGORITHM:
            raise ValueError("checkpoint uses a different algorithm")
        if int(checkpoint.get("schema_version", -1)) != self.CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported joint PPO checkpoint schema")
        if checkpoint.get("space") != asdict(self.space):
            raise ValueError("checkpoint joint space differs from this policy")
        saved_config = checkpoint.get("config")
        if not isinstance(saved_config, dict):
            raise ValueError("checkpoint config is missing")
        current_config = asdict(self.config)
        differing_config = sorted(
            name
            for name in set(saved_config) | set(current_config)
            if name != "device" and saved_config.get(name) != current_config.get(name)
        )
        if differing_config:
            raise ValueError(
                "checkpoint policy config differs from this policy: "
                + ", ".join(differing_config)
            )
        self._load_network_weights(checkpoint)
        if load_optimizer:
            optimizer_state = checkpoint.get("optimizer")
            if not isinstance(optimizer_state, dict):
                raise ValueError("checkpoint optimizer state is missing")
            self.optimizer.load_state_dict(optimizer_state)
            self.update_count = int(checkpoint.get("update_count", 0))
            self.transition_count = int(checkpoint.get("transition_count", 0))
            self.episode_count = int(checkpoint.get("episode_count", 0))
            self.last_metrics = dict(checkpoint.get("last_metrics", {}) or {})
            numpy_state = checkpoint.get("numpy_rng_state")
            if numpy_state is not None:
                self._rng.bit_generator.state = numpy_state
            torch_state = checkpoint.get("torch_rng_state")
            if isinstance(torch_state, Tensor):
                torch.set_rng_state(torch_state.cpu())
            cuda_state = checkpoint.get("cuda_rng_state")
            if self.device.type == "cuda" and isinstance(cuda_state, Tensor):
                torch.cuda.set_rng_state(cuda_state.cpu(), self.device)

    def _load_network_weights(self, checkpoint: object) -> None:
        """Validate checkpoint identity and atomically load compatible weights."""

        if not isinstance(checkpoint, dict):
            raise ValueError("joint PPO checkpoint is not a dictionary")
        if checkpoint.get("algorithm") != self.ALGORITHM:
            raise ValueError("checkpoint uses a different algorithm")
        if int(checkpoint.get("schema_version", -1)) != self.CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported joint PPO checkpoint schema")
        if checkpoint.get("space") != asdict(self.space):
            raise ValueError("checkpoint joint space differs from this policy")
        network_state = checkpoint.get("network")
        if not isinstance(network_state, dict):
            raise ValueError("checkpoint network state is missing")

        current_state = self.network.state_dict()
        missing = sorted(set(current_state) - set(network_state))
        unexpected = sorted(set(network_state) - set(current_state))
        shape_mismatches = sorted(
            name
            for name in set(current_state) & set(network_state)
            if getattr(network_state[name], "shape", None) != current_state[name].shape
        )
        if missing or unexpected or shape_mismatches:
            details: list[str] = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if unexpected:
                details.append("unexpected=" + ",".join(unexpected))
            if shape_mismatches:
                details.append("shape=" + ",".join(shape_mismatches))
            raise ValueError(
                "checkpoint network structure differs from this policy: "
                + "; ".join(details)
            )
        self.network.load_state_dict(network_state, strict=True)

    def load(
        self,
        path: str | os.PathLike[str],
        *,
        load_optimizer: bool = True,
    ) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self._load_checkpoint(checkpoint, load_optimizer=load_optimizer)

    def load_weights(self, path: str | os.PathLike[str]) -> None:
        """Load only compatible network weights, preserving fresh training state."""

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self._load_network_weights(checkpoint)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | os.PathLike[str],
        *,
        device: str | None = None,
        load_optimizer: bool = True,
    ) -> "JointPPOPolicy":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("joint PPO checkpoint is not a dictionary")
        raw_space = checkpoint.get("space")
        raw_config = checkpoint.get("config")
        if not isinstance(raw_space, dict) or not isinstance(raw_config, dict):
            raise ValueError("joint PPO checkpoint lacks space or config metadata")
        config = JointPPOConfig(**raw_config)
        if device is not None:
            config = replace(config, device=device)
        policy = cls(JointSpaceSpec(**raw_space), config)
        policy._load_checkpoint(checkpoint, load_optimizer=load_optimizer)
        return policy


__all__ = [
    "JointActorCritic",
    "JointNetworkOutput",
    "JointPPOConfig",
    "JointPPOPolicy",
    "JointPolicyEvaluation",
]
