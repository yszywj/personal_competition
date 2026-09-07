"""Training-local PPO copied from the project baseline and upgraded with GAE.

The Actor-Critic module and public ``SharedPolicy`` contract intentionally keep
the original parameter names.  The rollout/update implementation is local to
``personal_train`` so the competition source tree remains untouched.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from policies.red.learning.device import resolve_learning_device
from policies.red.learning.red_policy import (
    ACTION_DIM,
    PolicyTransition,
    SharedPolicy,
)


@dataclass
class PPOConfig:
    """Stable PPO defaults for long, shared-policy missile trajectories."""

    observation_dim: int = 85
    action_dim: int = ACTION_DIM
    hidden_dim: int = 128
    learning_rate: float = 1e-4
    learning_rate_final: float = 2e-5
    learning_rate_decay_updates: int = 100
    gamma: float = 0.999
    gae_lambda: float = 0.995
    clip_ratio: float = 0.15
    value_clip_ratio: float = 0.20
    value_coef: float = 0.5
    entropy_coef: float = 0.002
    entropy_final_coef: float = 0.0002
    entropy_decay_updates: int = 100
    target_kl: float = 0.015
    max_grad_norm: float = 0.5
    update_epochs: int = 2
    minibatch_size: int = 4096
    # In ``rollout`` mode this is a transition threshold.  The default
    # ``episode`` mode deliberately keeps one policy frozen for a whole round.
    rollout_size: int = 65536
    update_mode: str = "episode"
    value_inference_batch_size: int = 16384
    seed: int = 0
    device: str = "auto"


class ActorCritic(nn.Module):
    """The project's original MLP and state-dict key layout."""

    def __init__(self, config: PPOConfig):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(config.observation_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(config.hidden_dim, config.action_dim)
        self.critic = nn.Linear(config.hidden_dim, 1)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(observation)
        return self.actor(features), self.critic(features).squeeze(-1)


@dataclass(frozen=True)
class _PendingAction:
    action: int
    log_prob: float
    value: float


@dataclass(frozen=True)
class _RolloutItem:
    transition: PolicyTransition
    old_log_prob: float
    old_value: float


class PPOSharedPolicy(SharedPolicy):
    """Parameter-shared PPO with per-agent GAE and atomic joint-step updates.

    ``observe`` never changes network parameters.  This is important because
    the environment submits up to 164 interleaved transitions after one joint
    simulation step.  Updating is allowed only after all of them have arrived,
    or (by default) after the complete episode.
    """

    ALGORITHM = "personal_ppo_gae_v2"

    def __init__(self, config: Optional[PPOConfig] = None):
        self.config = config or PPOConfig()
        self._validate_config()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self._rng = np.random.default_rng(self.config.seed)
        resolved_device = resolve_learning_device(
            self.config.device,
            cuda_available=torch.cuda.is_available(),
            cuda_device_count=torch.cuda.device_count(),
        )
        self.config.device = resolved_device
        self.device = torch.device(resolved_device)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)

        self.network = ActorCritic(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.config.learning_rate
        )
        self._buffer: list[_RolloutItem] = []
        self._pending_by_agent: dict[int, _PendingAction] = {}
        self._anonymous_pending: deque[_PendingAction] = deque()
        self._environment_steps_in_rollout = 0
        self.update_count = 0
        self.transition_count = 0
        self.episode_count = 0
        self.last_metrics: dict[str, float] = {}
        self.training = True

    def _validate_config(self) -> None:
        config = self.config
        if config.observation_dim <= 0 or config.action_dim <= 0 or config.hidden_dim <= 0:
            raise ValueError("PPO observation/action/hidden dimensions must be positive")
        if config.learning_rate <= 0.0 or config.learning_rate_final <= 0.0:
            raise ValueError("PPO learning rates must be positive")
        if config.learning_rate_final > config.learning_rate:
            raise ValueError("PPO final learning rate cannot exceed its initial value")
        if config.learning_rate_decay_updates <= 0 or config.entropy_decay_updates <= 0:
            raise ValueError("PPO decay update counts must be positive")
        if not 0.0 <= config.gamma <= 1.0 or not 0.0 <= config.gae_lambda <= 1.0:
            raise ValueError("PPO gamma and gae_lambda must be in [0, 1]")
        if config.clip_ratio <= 0.0 or config.value_clip_ratio < 0.0:
            raise ValueError("PPO clipping ratios are invalid")
        if config.value_coef < 0.0 or config.entropy_coef < 0.0:
            raise ValueError("PPO loss coefficients must be non-negative")
        if config.entropy_final_coef < 0.0 or config.target_kl < 0.0:
            raise ValueError("PPO entropy_final_coef/target_kl must be non-negative")
        if config.entropy_final_coef > config.entropy_coef:
            raise ValueError("PPO final entropy coefficient cannot exceed its initial value")
        if config.max_grad_norm <= 0.0 or config.update_epochs <= 0:
            raise ValueError("PPO max_grad_norm/update_epochs must be positive")
        if config.minibatch_size <= 0 or config.rollout_size <= 0:
            raise ValueError("PPO minibatch_size/rollout_size must be positive")
        if config.value_inference_batch_size <= 0:
            raise ValueError("PPO value_inference_batch_size must be positive")
        if config.update_mode not in {"episode", "rollout"}:
            raise ValueError("PPO update_mode must be 'episode' or 'rollout'")

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """Compatibility entry point for callers that cannot supply agent_id."""

        return self._select_action(None, observation, action_mask)

    def select_action_for_agent(
        self,
        agent_id: int,
        observation: np.ndarray,
        action_mask: np.ndarray,
    ) -> int:
        """Sample an action and bind its behaviour statistics to one Agent."""

        return self._select_action(int(agent_id), observation, action_mask)

    def _select_action(
        self,
        agent_id: int | None,
        observation: np.ndarray,
        action_mask: np.ndarray,
    ) -> int:
        observation_array = np.asarray(observation, dtype=np.float32)
        mask_array = np.asarray(action_mask, dtype=np.bool_)
        if observation_array.shape != (self.config.observation_dim,):
            raise ValueError(
                f"Expected observation shape {(self.config.observation_dim,)}, "
                f"got {observation_array.shape}"
            )
        if mask_array.shape != (self.config.action_dim,):
            raise ValueError(
                f"Expected action-mask shape {(self.config.action_dim,)}, got {mask_array.shape}"
            )
        if not bool(mask_array.any()):
            raise ValueError("PPO received an action mask with no legal action")
        if self.training and agent_id is not None and agent_id in self._pending_by_agent:
            raise RuntimeError(f"Agent {agent_id} sampled twice before submitting a transition")

        observation_tensor = torch.as_tensor(
            observation_array, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        mask_tensor = torch.as_tensor(
            mask_array, dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.network(observation_tensor)
            logits = logits.masked_fill(~mask_tensor, torch.finfo(logits.dtype).min)
            distribution = Categorical(logits=logits)
            action = distribution.sample() if self.training else torch.argmax(logits, dim=-1)
            log_prob = distribution.log_prob(action)

        selected = int(action.item())
        if self.training:
            pending = _PendingAction(
                action=selected,
                log_prob=float(log_prob.item()),
                value=float(value.item()),
            )
            if agent_id is None:
                self._anonymous_pending.append(pending)
            else:
                self._pending_by_agent[agent_id] = pending
        return selected

    def observe(self, transition: PolicyTransition) -> None:
        if not self.training:
            return
        agent_id = int(transition.agent_id)
        pending = self._pending_by_agent.pop(agent_id, None)
        if pending is None and self._anonymous_pending:
            pending = self._anonymous_pending.popleft()
        if pending is None:
            raise RuntimeError(
                f"PPO received transition for Agent {agent_id} without a sampled action"
            )
        if pending.action != int(transition.action):
            raise RuntimeError(
                f"PPO action mismatch for Agent {agent_id}: "
                f"sampled {pending.action}, observed {transition.action}"
            )
        self._buffer.append(
            _RolloutItem(
                transition=transition,
                old_log_prob=pending.log_prob,
                old_value=pending.value,
            )
        )
        self.transition_count += 1

    def begin_environment_step(self, full_observation) -> None:
        del full_observation
        self._assert_no_pending("at the beginning of an environment step")

    def end_environment_step(self, full_observation) -> None:
        del full_observation

    def finish_environment_step(self) -> None:
        """Commit a joint step and optionally update at this atomic boundary."""

        if not self.training:
            return
        self._assert_no_pending("after the joint environment step")
        self._environment_steps_in_rollout += 1
        if (
            self.config.update_mode == "rollout"
            and len(self._buffer) >= self.config.rollout_size
        ):
            self.update()

    def finish_episode(self) -> dict[str, float]:
        """Update once from the complete episode (the default training mode)."""

        if not self.training:
            return self.last_metrics
        self._assert_no_pending("at episode end")
        metrics = self.update() if self._buffer else self.last_metrics
        self.episode_count += 1
        return metrics

    def reset_episode(self) -> None:
        """Safety flush for environments that only expose a reset hook."""

        if not self.training:
            return
        self._assert_no_pending("during episode reset")
        if self._buffer:
            self.update()

    def set_training(self, training: bool) -> None:
        training = bool(training)
        if not training:
            self._assert_no_pending("when entering evaluation mode")
        self.training = training
        self.network.train(training)

    def _assert_no_pending(self, context: str) -> None:
        if self._pending_by_agent or self._anonymous_pending:
            named = sorted(self._pending_by_agent)[:8]
            raise RuntimeError(
                f"Unmatched PPO actions {context}: named={named}, "
                f"anonymous={len(self._anonymous_pending)}"
            )

    @staticmethod
    def _compute_gae(
        agent_ids: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        old_values: np.ndarray,
        next_values: np.ndarray,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE independently for interleaved Agent trajectories."""

        sample_count = len(rewards)
        advantages = np.zeros(sample_count, dtype=np.float64)
        grouped: defaultdict[int, list[int]] = defaultdict(list)
        for index, raw_agent_id in enumerate(agent_ids):
            grouped[int(raw_agent_id)].append(index)

        for indices in grouped.values():
            gae = 0.0
            for index in reversed(indices):
                continuation = 0.0 if bool(dones[index]) else 1.0
                delta = (
                    float(rewards[index])
                    + gamma * float(next_values[index]) * continuation
                    - float(old_values[index])
                )
                gae = delta + gamma * gae_lambda * continuation * gae
                advantages[index] = gae
        returns = advantages + old_values.astype(np.float64, copy=False)
        return advantages.astype(np.float32), returns.astype(np.float32)

    def _predict_next_values(self, next_observations: np.ndarray) -> np.ndarray:
        values: list[np.ndarray] = []
        batch_size = self.config.value_inference_batch_size
        with torch.no_grad():
            for start in range(0, len(next_observations), batch_size):
                batch = torch.as_tensor(
                    next_observations[start : start + batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                _, predicted = self.network(batch)
                values.append(predicted.detach().cpu().numpy())
        return np.concatenate(values).astype(np.float32, copy=False)

    @staticmethod
    def _linear_schedule(start: float, end: float, index: int, duration: int) -> float:
        fraction = min(max(index / max(duration, 1), 0.0), 1.0)
        return float(start + fraction * (end - start))

    def update(self) -> dict[str, float]:
        if not self._buffer:
            return self.last_metrics
        self._assert_no_pending("before PPO update")

        transitions = [item.transition for item in self._buffer]
        observations = np.stack([item.observation for item in transitions]).astype(
            np.float32, copy=False
        )
        next_observations = np.stack(
            [item.next_observation for item in transitions]
        ).astype(np.float32, copy=False)
        actions = np.asarray([item.action for item in transitions], dtype=np.int64)
        action_masks = np.stack([item.action_mask for item in transitions]).astype(
            np.bool_, copy=False
        )
        rewards = np.asarray([item.reward for item in transitions], dtype=np.float32)
        dones = np.asarray([item.done for item in transitions], dtype=np.bool_)
        agent_ids = np.asarray([item.agent_id for item in transitions], dtype=np.int64)
        old_log_probs = np.asarray(
            [item.old_log_prob for item in self._buffer], dtype=np.float32
        )
        old_values = np.asarray(
            [item.old_value for item in self._buffer], dtype=np.float32
        )

        next_values = self._predict_next_values(next_observations)
        advantages, returns = self._compute_gae(
            agent_ids,
            rewards,
            dones,
            old_values,
            next_values,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        if not np.isfinite(advantages).all() or not np.isfinite(returns).all():
            raise FloatingPointError("PPO produced non-finite GAE advantages/returns")
        advantage_mean = float(advantages.mean())
        advantage_std = float(advantages.std())
        if len(advantages) > 1 and advantage_std > 1e-8:
            advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)
        else:
            advantages = np.zeros_like(advantages)

        return_variance = float(np.var(returns))
        explained_variance = (
            1.0 - float(np.var(returns - old_values)) / return_variance
            if return_variance > 1e-8
            else 0.0
        )
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
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        minibatch_updates = 0
        epochs_ran = 0
        early_stopped = False
        sample_count = len(transitions)
        for epoch in range(self.config.update_epochs):
            epochs_ran = epoch + 1
            permutation = self._rng.permutation(sample_count)
            for start in range(0, sample_count, self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                observation_batch = torch.as_tensor(
                    observations[indices], dtype=torch.float32, device=self.device
                )
                action_batch = torch.as_tensor(
                    actions[indices], dtype=torch.long, device=self.device
                )
                mask_batch = torch.as_tensor(
                    action_masks[indices], dtype=torch.bool, device=self.device
                )
                old_log_prob_batch = torch.as_tensor(
                    old_log_probs[indices], dtype=torch.float32, device=self.device
                )
                old_value_batch = torch.as_tensor(
                    old_values[indices], dtype=torch.float32, device=self.device
                )
                advantage_batch = torch.as_tensor(
                    advantages[indices], dtype=torch.float32, device=self.device
                )
                return_batch = torch.as_tensor(
                    returns[indices], dtype=torch.float32, device=self.device
                )

                logits, values = self.network(observation_batch)
                logits = logits.masked_fill(
                    ~mask_batch, torch.finfo(logits.dtype).min
                )
                distribution = Categorical(logits=logits)
                new_log_probs = distribution.log_prob(action_batch)
                entropy = distribution.entropy().mean()
                log_ratio = new_log_probs - old_log_prob_batch
                ratio = torch.exp(log_ratio)
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.config.clip_ratio)
                        .float()
                        .mean()
                    )
                if (
                    self.config.target_kl > 0.0
                    and float(approx_kl.item()) > 1.5 * self.config.target_kl
                ):
                    # Do not take one more optimizer step after the trust-region
                    # estimate has already crossed the configured boundary.
                    early_stopped = True
                    break
                unclipped = ratio * advantage_batch
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * advantage_batch
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_losses = F.smooth_l1_loss(
                    values, return_batch, reduction="none"
                )
                if self.config.value_clip_ratio > 0.0:
                    clipped_values = old_value_batch + torch.clamp(
                        values - old_value_batch,
                        -self.config.value_clip_ratio,
                        self.config.value_clip_ratio,
                    )
                    clipped_value_losses = F.smooth_l1_loss(
                        clipped_values, return_batch, reduction="none"
                    )
                    value_loss = torch.maximum(
                        value_losses, clipped_value_losses
                    ).mean()
                else:
                    value_loss = value_losses.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - entropy_coef * entropy
                )
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("PPO loss became NaN or infinite")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                totals["policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"] += float(entropy.item())
                totals["approx_kl"] += float(approx_kl.item())
                totals["clip_fraction"] += float(clip_fraction.item())
                minibatch_updates += 1

            if early_stopped:
                break

        self._buffer.clear()
        rollout_environment_steps = self._environment_steps_in_rollout
        self._environment_steps_in_rollout = 0
        self.update_count += 1
        divisor = max(1, minibatch_updates)
        self.last_metrics = {
            key: value / divisor for key, value in totals.items()
        }
        self.last_metrics.update(
            {
                "samples": float(sample_count),
                "trajectories": float(len(set(agent_ids.tolist()))),
                "rollout_environment_steps": float(rollout_environment_steps),
                "epochs_ran": float(epochs_ran),
                "early_stopped": float(early_stopped),
                "learning_rate": learning_rate,
                "entropy_coef": entropy_coef,
                "advantage_mean": advantage_mean,
                "advantage_std": advantage_std,
                "explained_variance": explained_variance,
            }
        )
        return self.last_metrics

    def save(self, path: str) -> None:
        checkpoint = {
            "algorithm": self.ALGORITHM,
            "config": self.config.__dict__,
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "transition_count": self.transition_count,
            "episode_count": self.episode_count,
            "last_metrics": self.last_metrics,
            "numpy_rng_state": self._rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "safe_training_boundary": not self._buffer
            and not self._pending_by_agent
            and not self._anonymous_pending,
        }
        if torch.cuda.is_available():
            checkpoint["cuda_rng_states"] = torch.cuda.get_rng_state_all()
        torch.save(checkpoint, path)

    def load(self, path: str, *, load_optimizer: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or "network" not in checkpoint:
            raise ValueError(f"Invalid PPO checkpoint: {path}")
        self.network.load_state_dict(checkpoint["network"])
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if load_optimizer:
            self.update_count = int(checkpoint.get("update_count", 0))
            self.transition_count = int(checkpoint.get("transition_count", 0))
            self.episode_count = int(checkpoint.get("episode_count", 0))
            self.last_metrics = dict(checkpoint.get("last_metrics", {}) or {})
            rng_state = checkpoint.get("numpy_rng_state")
            if rng_state is not None:
                self._rng.bit_generator.state = rng_state
            torch_state = checkpoint.get("torch_rng_state")
            if torch_state is not None:
                torch.set_rng_state(torch_state.cpu())
            cuda_states = checkpoint.get("cuda_rng_states")
            if cuda_states is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
        self._buffer.clear()
        self._pending_by_agent.clear()
        self._anonymous_pending.clear()
        self._environment_steps_in_rollout = 0
