"""Independent implementation of Chen et al. (2022), Algorithm 1 and Eqs. 5-7.

DOI: 10.3389/fnbot.2022.883562. This is not the authors' source code.
The deterministic DiT-Flow actor and offline RCM data adapter are project work.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import dit_state_attention as attention
import dit_replay_actor_critic as replay
from experience_replay import transition_records as eligible_records
from train_dit_rectified_flow import _loss


TRAINING_MODULE = "frontiers2022_algorithm1_ddpg_rcm_v1"
PAPER_DOI = "10.3389/fnbot.2022.883562"


def paper_reward(target_distance_m, obstacle_distance_m=None):
    """Eqs. (5)-(7), c1=500, c2=15, a=1 m, n=35; inputs in metres.

    Omitting obstacle distance disables that term; it does NOT assume a known
    obstacle location. The RCM recordings currently have no obstacle distance.
    """
    distance = np.asarray(target_distance_m, dtype=np.float64)
    if not np.isfinite(distance).all() or (distance < 0).any():
        raise ValueError("target distances must be finite nonnegative metres")
    reward = -500.0 * 0.5 * distance ** 2
    if obstacle_distance_m is not None:
        obstacle = np.asarray(obstacle_distance_m, dtype=np.float64)
        if not np.isfinite(obstacle).all() or (obstacle < 0).any():
            raise ValueError("obstacle distances must be finite nonnegative metres")
        reward = reward - 15.0 * (1.0 / (1.0 + obstacle)) ** 35
    return reward


def transition_records(dataset, source, **options):
    """Reuse contiguous executed-action pairs, replace only the reward.

    Reward is evaluated after the action: d_T = ||OCT(s_next)|| mm / 1000.
    Source gaps, OCT validity, and explicit terminal labels retain their meaning.
    """
    records = eligible_records(dataset, source, **options)
    next_mm = np.linalg.norm(dataset.oct_navigation[records["next_row"]], axis=1)
    records["reward"] = paper_reward(next_mm / 1000.0)
    return records


class DeterministicFlowPolicy(attention.StateConditionedDiTRectifiedFlowPolicy):
    """A deterministic actor mu(s): integrate Flow from ONE fixed noise chunk.

    The same initial chunk is used for every observation, online and in target
    updates. No sample averaging, fresh policy noise or entropy term is used.
    The seed is saved in checkpoint config; the buffer adds no trained parameter.
    """

    def __init__(self, *args, fixed_noise_seed=20261118, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_noise_seed = int(fixed_noise_seed)
        noise = torch.randn(
            1, self.n_action_steps, self.action_dim,
            generator=torch.Generator(device="cpu").manual_seed(self.fixed_noise_seed),
        )
        self.register_buffer("fixed_flow_noise", noise, persistent=False)

    def deterministic_chunk(self, image, state, oct_feature, velocity,
                            previous_action, steps=None):
        steps = self.flow_integration_steps if steps is None else int(steps)
        if steps < 1:
            raise ValueError("Flow integration steps must be positive")
        tokens, condition = self._condition(image, state, oct_feature, velocity, previous_action)
        actions = self.fixed_flow_noise.to(state).expand(state.shape[0], -1, -1)
        for index in range(steps):
            tau = torch.full(
                (state.shape[0],), (index + 0.5) / steps,
                device=state.device, dtype=state.dtype,
            )
            actions = actions + self._predict_velocity_from_condition(
                actions, tau, tokens, condition,
            ) / steps
        return actions * self.action_std + self.action_mean

    @torch.no_grad()
    def sample(self, image, state, oct_feature, velocity, previous_action,
               steps=None, generator=None):
        # Keep evaluator API compatibility; generator cannot change mu(s).
        return self.deterministic_chunk(image, state, oct_feature, velocity,
                                        previous_action, steps=steps)


class SingleQ(nn.Module):
    """One Q, as in Algorithm 1. Width/SiLU are project hyperparameters."""

    def __init__(self, observation_dim, action_dim=3, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, geometry, normalized_action):
        return self.network(torch.cat([geometry, normalized_action], dim=-1)).squeeze(-1)


def policy_action(model, observation, limits):
    chunk = model.deterministic_chunk(*replay.observation_arguments(observation))
    return torch.maximum(torch.minimum(chunk[:, 0], limits), -limits)


def update_critic(model, target_actor, critic, target_critic, optimizer, transition,
                  limits, gamma, tau):
    current, following, executed, reward, terminated = transition
    with torch.no_grad():
        next_action = policy_action(target_actor, following, limits)
        next_q = target_critic(replay.geometry_features(target_actor, following),
                              replay.normalized_action(target_actor, next_action))
        # Terminal masking is an RCM engineering extension to Algorithm 1.
        target = reward + gamma * (~terminated).float() * next_q
        features = replay.geometry_features(model, current)
    prediction = critic(features, replay.normalized_action(model, executed))
    loss = F.mse_loss(prediction, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    replay.soft_update(target_critic, critic, tau)
    return float(loss.detach()), float(reward.mean()), float(target.mean())


def update_actor(model, critic, optimizer, demonstration, replay_observation,
                 limits, fm_weight, q_weight):
    """Algorithm 1: minimize -E Q(s,mu(s)), without Q-value normalization.

    fm_weight>0 adds an explicitly optional project Flow anchor, absent from the
    paper. Default zero uses only DDPG's actor objective.
    """
    critic.requires_grad_(False)
    optimizer.zero_grad(set_to_none=True)
    try:
        fm = None
        if fm_weight > 0:
            fm = _loss(model, demonstration)
            (fm_weight * fm).backward()
        action = policy_action(model, replay_observation, limits)
        with torch.no_grad():
            features = replay.geometry_features(model, replay_observation)
        q = critic(features, replay.normalized_action(model, action))
        objective = -q_weight * q.mean()
        objective.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        # Shared history's train_loss is the actor objective when FM is disabled.
        metric = fm.detach() if fm is not None else objective.detach()
        return float(metric), float(objective.detach()), float(q.detach().mean())
    finally:
        critic.requires_grad_(True)
