#!/usr/bin/env python3
"""Add transition replay, twin Q critics and expert Flow regularization to v1.

This is a Flow-compatible actor-critic extension, not an ILOSAC/SAC reproduction:
there is no tractable Gaussian policy log density or SAC entropy term here.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

import dit_state_attention as attention
from experience_replay import ACTION_LIMIT, TransitionReplayBuffer, transition_records
from train_diffusion_policy import (
    RCMDataset, configure_stream_logging, image_transform_for_dataset, seed_worker,
)
from train_dit_rectified_flow import _atomic_torch_save, _loss, _move_batch


TRAINING_MODULE = "flow_replay_actor_critic_v1"
ILOSAC_INSPIRED_MODULE = "flow_ilosac_inspired_offline_v1"
TRAINING_MODULES = (TRAINING_MODULE, ILOSAC_INSPIRED_MODULE)
STAT_NAMES = (
    "state_mean", "state_std", "oct_mean", "oct_std", "action_mean", "action_std",
    "velocity_mean", "velocity_std", "previous_action_mean", "previous_action_std",
)
OBS_NAMES = ("image", "state", "oct", "velocity", "previous_action")


def observation_arguments(batch):
    return [batch["observation." + name] for name in OBS_NAMES]


def sample_with_grad(model, batch):
    """Same midpoint Flow integration as sample(), with autograd enabled.

    The deployed policy still executes only action 0 of each predicted chunk.
    Calling model.sample() here would silently disable the actor's Q gradient.
    """
    image, state, oct_feature, velocity, previous_action = observation_arguments(batch)
    tokens, condition = model._condition(
        image, state, oct_feature, velocity, previous_action,
    )
    actions = torch.randn(
        state.shape[0], model.n_action_steps, model.action_dim,
        device=state.device, dtype=state.dtype,
    )
    steps = model.flow_integration_steps
    for index in range(steps):
        tau = torch.full(
            (state.shape[0],), (index + 0.5) / steps,
            device=state.device, dtype=state.dtype,
        )
        actions = actions + model._predict_velocity_from_condition(
            actions, tau, tokens, condition,
        ) / steps
    return actions * model.action_std + model.action_mean


def geometry_features(model, batch):
    """Critic sees the existing OCT/encoder/motion observation, without truth."""
    normalized = model._normalize_observation(*observation_arguments(batch)[1:])
    return torch.cat(normalized, dim=-1).flatten(1)


class TwinQ(nn.Module):
    def __init__(self, observation_dim, action_dim=3, hidden_dim=256):
        super().__init__()
        def network():
            return nn.Sequential(
                nn.Linear(observation_dim + action_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
            )
        self.q1, self.q2 = network(), network()

    def forward(self, geometry, normalized_action):
        inputs = torch.cat([geometry, normalized_action], dim=-1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)


class ReplayObservations:
    """Use the existing image/history loader at arbitrary valid replay rows.

    One-step replay may use the last 15 observations excluded by 16-action BC
    windows. The shared dataset arrays and images are not copied.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.view = copy.copy(dataset)
        self.view.n_action_steps = 1
        self.view.valid_indices = range(len(dataset.states))

    def get(self, row):
        sample = self.view[int(row)]
        return {"observation." + name: sample["observation." + name] for name in OBS_NAMES}


def replay_batch(buffer, sources, batch_size, device):
    records = buffer.sample(batch_size)
    current, following, executed = [], [], []
    for record in records:
        source = sources[int(record["source"])]
        current.append(source.get(record["row"]))
        following.append(source.get(record["next_row"]))
        executed.append(source.dataset.executed_actions[int(record["row"])])
    return (
        _move_batch(default_collate(current), device),
        _move_batch(default_collate(following), device),
        torch.as_tensor(np.stack(executed), device=device),
        torch.as_tensor(records["reward"].copy(), device=device),
        torch.as_tensor(records["terminated"].copy(), device=device),
    )


@torch.no_grad()
def soft_update(target, source, tau):
    for target_parameter, parameter in zip(target.parameters(), source.parameters()):
        target_parameter.lerp_(parameter, tau)


def normalized_action(model, action):
    return (action - model.action_mean) / model.action_std


def clipped_policy_action(model, batch, limits, *, gradients):
    with torch.set_grad_enabled(gradients):
        action = sample_with_grad(model, batch)[:, 0]
        # Match existing evaluation clipping before querying Q, not new limits.
        return torch.maximum(torch.minimum(action, limits), -limits)


def update_critic(model, target_actor, critic, target_critic, optimizer, transition,
                  limits, gamma, tau, td_clip=0.0):
    current, following, executed, reward, terminated = transition
    with torch.no_grad():
        next_action = clipped_policy_action(target_actor, following, limits, gradients=False)
        target_q = target_critic(
            geometry_features(target_actor, following), normalized_action(target_actor, next_action),
        )
        target = reward + gamma * (~terminated).float() * torch.minimum(*target_q)
        if td_clip:
            target = target.clamp(-td_clip, td_clip)
        features = geometry_features(model, current)
    first, second = critic(features, normalized_action(model, executed))
    loss = F.mse_loss(first, target) + F.mse_loss(second, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    soft_update(target_critic, critic, tau)
    return float(loss.detach()), float(reward.mean()), float(target.mean())


def update_actor(model, critic, optimizer, demonstration, replay_observation,
                 limits, fm_weight, q_weight, reference_actor=None,
                 demo_anchor_weight=0.0):
    # Q parameters stay fixed while its derivative wrt the Flow action is used.
    critic.requires_grad_(False)
    optimizer.zero_grad(set_to_none=True)
    try:
        fm_loss = _loss(model, demonstration)
        # Backprop separately to release the BC graph before four Flow steps.
        (fm_weight * fm_loss).backward()
        actions = clipped_policy_action(model, replay_observation, limits, gradients=True)
        with torch.no_grad():
            features = geometry_features(model, replay_observation)
        first, second = critic(features, normalized_action(model, actions))
        q = torch.minimum(first, second)
        q_objective = -q_weight * q.mean() / q.detach().abs().mean().clamp_min(1.0)
        anchor_loss = torch.zeros((), device=actions.device)
        objective = q_objective
        if reference_actor is not None and demo_anchor_weight:
            with torch.no_grad():
                reference_actions = clipped_policy_action(
                    reference_actor, demonstration, limits, gradients=False,
                )
            anchor_loss = F.mse_loss(actions, reference_actions)
            objective = objective + demo_anchor_weight * anchor_loss
        objective.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        return (
            float(fm_loss.detach()), float(q_objective.detach()),
            float(q.detach().mean()), float(anchor_loss.detach()),
        )
    finally:
        critic.requires_grad_(True)


@torch.no_grad()
def validate(model, loader, device, seed):
    model.eval()
    total, samples = 0.0, 0
    devices = [device.index or 0] if device.type == "cuda" else []
    # The same Flow noise/time draws each epoch; preserve training RNG state.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for batch in loader:
            batch = _move_batch(batch, device)
            count = len(batch["action"])
            total += float(_loss(model, batch)) * count
            samples += count
    return total / samples


def make_loader(dataset, batch_size, workers, *, shuffle, seed):
    options = dict(
        batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    if workers:
        options.update(prefetch_factor=2, persistent_workers=True)
    return DataLoader(dataset, **options)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def train(argv, *, components=None):
    """Run the shared replay loop, optionally with algorithm-specific parts.

    The original replay module remains the default. Alternate algorithms can
    replace the policy, critic, update rules, and reward adapter explicitly.
    """
    components = {} if components is None else components
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="expert-labeled BC data")
    parser.add_argument("--val-data-dir", type=Path, required=True)
    parser.add_argument("--replay-data-dir", type=Path, nargs="*", default=[], help="additional converted v2 rollouts")
    parser.add_argument("--init-from", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--replay-batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--bias-lr-scale", type=float, default=10.0)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--critic-hidden-dim", type=int, default=256)
    parser.add_argument("--critic-warmup-steps", type=int, default=250)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--fm-weight", type=float, default=1.0)
    parser.add_argument("--q-weight", type=float, default=0.1)
    parser.add_argument(
        "--variant",
        choices=("replay_v1", "ilosac_flow_v2"),
        default="replay_v1",
    )
    parser.add_argument("--td-clip", type=float, default=0.0)
    parser.add_argument("--demo-anchor-weight", type=float, default=0.0)
    parser.add_argument("--replay-capacity", type=int, default=300000)
    parser.add_argument("--distance-scale-mm", type=float, default=1.0)
    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument("--distance-weight", type=float, default=0.1)
    parser.add_argument("--effort-weight", type=float, default=0.001)
    parser.add_argument("--control-period-s", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--device", default="cuda")
    if "configure_parser" in components:
        components["configure_parser"](parser)
    args = parser.parse_args(argv)
    counts = (args.epochs, args.steps_per_epoch, args.batch_size, args.replay_batch_size,
              args.val_batch_size, args.critic_hidden_dim, args.replay_capacity)
    if min(counts) < 1 or args.num_workers < 0:
        parser.error("counts must be positive and workers nonnegative")
    if not 0 <= args.critic_warmup_steps < args.steps_per_epoch:
        parser.error("critic warmup must be nonnegative and shorter than the first epoch")
    if not (0 < args.gamma < 1 and 0 < args.tau <= 1):
        parser.error("gamma must be in (0,1) and tau in (0,1]")
    if min(args.lr, args.bias_lr_scale, args.critic_lr,
           args.q_weight, args.distance_scale_mm, args.control_period_s) <= 0:
        parser.error("learning rates, loss weights, distance scale and period must be positive")
    if args.fm_weight < 0 or (args.fm_weight == 0 and not components.get("allow_zero_fm", False)):
        parser.error("fm-weight must be positive for the original replay module")
    if min(args.progress_weight, args.distance_weight, args.effort_weight) < 0:
        parser.error("reward weights must be nonnegative")
    if args.td_clip < 0 or args.demo_anchor_weight < 0:
        parser.error("TD clip and demonstration anchor weight must be nonnegative")
    training_module = components.get(
        "training_module",
        ILOSAC_INSPIRED_MODULE
        if args.variant == "ilosac_flow_v2"
        else TRAINING_MODULE,
    )
    variant = components.get("variant_label", args.variant)
    replay_paths = list(dict.fromkeys(path.resolve() for path in [args.data_dir, *args.replay_data_dir]))
    if args.val_data_dir.resolve() in replay_paths:
        parser.error("validation data cannot enter the training replay buffer")
    if args.output_dir.exists():
        parser.error("output directory already exists")
    if "validate_args" in components:
        components["validate_args"](args, parser)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu only for a small smoke run")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    if checkpoint.get("architecture") != attention.ARCHITECTURE:
        parser.error("--init-from must be the state-attention v1 checkpoint")
    if checkpoint.get("policy_type") != "dit_rectified_flow":
        parser.error("--init-from has an incompatible policy type")
    policy_class = components.get("policy_class", attention.StateConditionedDiTRectifiedFlowPolicy)
    policy_kwargs = components.get("policy_kwargs", lambda args: {})(args)
    model = policy_class(
        **{key: config[key] for key in STAT_NAMES},
        **{key: int(config[key]) for key in (
            "n_obs_steps", "n_action_steps", "dit_dim", "dit_depth", "dit_heads", "flow_integration_steps",
        )},
        **policy_kwargs,
    ).to(device)
    # Do not let the v1 old-checkpoint fallback hide missing trained bias keys.
    if set(model.state_dict()) != set(checkpoint["model_state_dict"]):
        raise ValueError("initial v1 state dictionary has missing or unexpected parameters")
    model.load_state_dict(checkpoint["model_state_dict"])
    target_actor = copy.deepcopy(model).eval().requires_grad_(False)
    reference_actor = (
        copy.deepcopy(model).eval().requires_grad_(False)
        if args.demo_anchor_weight
        else None
    )
    geometry_dim = model.n_obs_steps * (3 + model.oct_dim + 3 + model.action_dim)
    critic_class = components.get("critic_class", TwinQ)
    critic = critic_class(geometry_dim, hidden_dim=args.critic_hidden_dim).to(device)
    target_critic = copy.deepcopy(critic).eval().requires_grad_(False)
    backbone, bias = [], []
    for name, parameter in model.named_parameters():
        (bias if ".temporal_bias." in name else backbone).append(parameter)
    actor_optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.lr},
        {"params": bias, "lr": args.lr * args.bias_lr_scale},
    ], weight_decay=components.get("actor_weight_decay", 1e-4))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    limits = torch.as_tensor(ACTION_LIMIT, device=device)

    transform = image_transform_for_dataset(args.data_dir)
    def dataset(path):
        return RCMDataset(
            path, transform=transform, n_obs_steps=model.n_obs_steps,
            n_action_steps=model.n_action_steps, include_velocity=True,
            include_previous_action=True, control_period_s=args.control_period_s,
        )
    demonstrations, validation = dataset(args.data_dir), dataset(args.val_data_dir)
    if not len(demonstrations) or not len(validation):
        raise ValueError("training and validation need complete expert action windows")
    sources = []
    buffer = TransitionReplayBuffer(args.replay_capacity, args.seed)
    source_metadata = []
    for path in replay_paths:
        data = demonstrations if path == args.data_dir.resolve() else dataset(path)
        with np.load(path / "data" / "dataset.npz", allow_pickle=False) as arrays:
            terminated = arrays["terminated"].copy() if "terminated" in arrays else None
        records = components.get("transition_records", transition_records)(
            data, len(sources), n_obs_steps=model.n_obs_steps,
            control_period_s=args.control_period_s, distance_scale_mm=args.distance_scale_mm,
            progress_weight=args.progress_weight, distance_weight=args.distance_weight,
            effort_weight=args.effort_weight, terminated=terminated,
        )
        buffer.extend(records)
        sources.append(ReplayObservations(data))
        source_metadata.append({
            "source": len(sources) - 1, "data_dir": str(path),
            "frames": len(data.states), "eligible_transitions": len(records),
            "explicit_terminated_labels": terminated is not None,
        })
    if not len(buffer):
        raise ValueError("no contiguous, OCT-valid one-step transitions for replay")
    args.output_dir.mkdir(parents=True)
    configure_stream_logging(args.output_dir)
    buffer.save(args.output_dir / "replay_index.npz")
    run_config = {
        **config, **{name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
                    if name != "replay_data_dir"},
        "replay_data_dir": [str(path) for path in args.replay_data_dir],
        "training_module": training_module, "attention_mechanism": attention.ARCHITECTURE,
        "include_velocity": True, "include_previous_action": True, "use_oct": True,
        **components.get("metadata", lambda args: {})(args),
    }
    write_json(args.output_dir / "replay_manifest.json", {
        "training_module": training_module, "variant": variant,
        "sources": source_metadata,
        "retained_transitions": len(buffer), "seen_transitions": buffer.seen,
        "capacity": buffer.capacity, "retention": "uniform_reservoir",
        "reward": "progress_weight*(d-d_next)/scale-distance_weight*d_next/scale-effort_weight*mean((executed/action_limit)^2)",
        "reward_distance_source": "observed OCT navigation norm in mm",
        "unlabeled_endings": "continuing task; bootstrap, no inferred success or failure",
        "execution_action_limit_deg_deg_mm": ACTION_LIMIT.tolist(),
        **components.get("replay_metadata", {}),
        **({"ilosac_inspired_assumptions": {
            "implemented": [
                "frozen demonstration-policy action anchor as KL proxy",
                "TD target clipping for critic scale control",
                "expert Flow matching plus replay Q guidance",
            ],
            "not_implemented": [
                "THMM Gaussian hidden-state model",
                "duration model",
                "online S_THMM membership and dynamic expansion",
                "Gaussian policy entropy temperature",
                "exact demonstration-policy KL",
            ],
            "naming": "offline ILOSAC-inspired Flow variant, not an ILOSAC reproduction",
        }} if not components else {}),
    })
    write_json(args.output_dir / "training_config.json", run_config)
    print(json.dumps({"training_module": training_module, "variant": variant,
                      "replay_size": len(buffer),
                      "init_from": str(args.init_from), "sources": source_metadata}), flush=True)
    train_loader = (make_loader(demonstrations, args.batch_size, args.num_workers, shuffle=True, seed=args.seed)
                    if args.fm_weight > 0 else None)
    val_loader = make_loader(validation, args.val_batch_size, min(args.num_workers, 4), shuffle=False, seed=args.seed)
    batches = iter(train_loader) if train_loader is not None else None
    history, best, started, global_step = [], float("inf"), time.monotonic(), 0
    for epoch in range(args.epochs):
        model.train()
        totals = dict(critic_loss=0.0, reward_mean=0.0, td_target_mean=0.0,
                      train_loss=0.0, q_objective=0.0, policy_q_mean=0.0,
                      demo_anchor_loss=0.0)
        actor_updates = 0
        for _ in tqdm(range(args.steps_per_epoch), desc=f"Replay actor-critic epoch {epoch}"):
            transition = replay_batch(buffer, sources, args.replay_batch_size, device)
            if "update_critic" in components:
                q_loss, reward, target = components["update_critic"](
                    model, target_actor, critic, target_critic, critic_optimizer,
                    transition, limits, args.gamma, args.tau,
                )
            else:
                q_loss, reward, target = update_critic(
                    model, target_actor, critic, target_critic, critic_optimizer, transition,
                    limits, args.gamma, args.tau, args.td_clip,
                )
            totals["critic_loss"] += q_loss
            totals["reward_mean"] += reward
            totals["td_target_mean"] += target
            if global_step >= args.critic_warmup_steps:
                demonstration = None
                if args.fm_weight > 0:
                    try:
                        demonstration = next(batches)
                    except StopIteration:
                        batches = iter(train_loader)
                        demonstration = next(batches)
                    demonstration = _move_batch(demonstration, device)
                anchor_weight = (
                    args.demo_anchor_weight
                    * max(
                        0.0,
                        1.0
                        - global_step
                        / max(1, args.epochs * args.steps_per_epoch),
                    )
                )
                if "update_actor" in components:
                    actor_result = components["update_actor"](
                        model, critic, actor_optimizer, demonstration,
                        transition[0], limits, args.fm_weight, args.q_weight,
                    )
                else:
                    actor_result = update_actor(
                        model, critic, actor_optimizer, demonstration,
                        transition[0], limits, args.fm_weight, args.q_weight,
                        reference_actor, anchor_weight,
                    )
                fm, q_objective, policy_q, *extra = actor_result
                anchor_loss = extra[0] if extra else 0.0
                soft_update(target_actor, model, args.tau)
                totals["train_loss"] += fm
                totals["q_objective"] += q_objective
                totals["policy_q_mean"] += policy_q
                totals["demo_anchor_loss"] += anchor_loss
                actor_updates += 1
            global_step += 1
        row = {
            "epoch": epoch, "global_step": global_step, "actor_updates": actor_updates,
            **{name: value / (actor_updates if name in ("train_loss", "q_objective", "policy_q_mean", "demo_anchor_loss")
                              else args.steps_per_epoch) for name, value in totals.items()},
            "val_loss": validate(model, val_loader, device, args.seed + 1),
            "elapsed_s": time.monotonic() - started,
        }
        history.append(row)
        write_json(args.output_dir / "training_history.json", history)
        payload = {
            "schema_version": 1, "policy_type": model.policy_type,
            "architecture": attention.ARCHITECTURE, "training_module": training_module,
            "model_state_dict": model.state_dict(), "critic_state_dict": critic.state_dict(),
            "target_actor_state_dict": target_actor.state_dict(),
            "target_critic_state_dict": target_critic.state_dict(),
            "optimizer_state_dict": actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": critic_optimizer.state_dict(),
            "epoch": epoch, "global_step": global_step, "train_loss": row["train_loss"],
            "val_loss": row["val_loss"], "config": run_config,
        }
        _atomic_torch_save(payload, args.output_dir / "last_model.pth")
        if row["val_loss"] < best:
            best = row["val_loss"]
            _atomic_torch_save(payload, args.output_dir / "best_model.pth")
        print(json.dumps(row), flush=True)
    return 0


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "eval"))
    mode = parser.parse_args(arguments[:1]).mode
    if mode == "train":
        return train(arguments[1:])
    # Require this extension's own checkpoint so evaluation cannot silently use v1.
    evaluation_parser = argparse.ArgumentParser(add_help=False)
    evaluation_parser.add_argument("--model", type=Path)
    options, _ = evaluation_parser.parse_known_args(arguments[1:])
    if options.model is not None:
        checkpoint = torch.load(options.model, map_location="cpu", weights_only=False)
        if checkpoint.get("training_module") not in TRAINING_MODULES:
            raise ValueError(
                "evaluation requires a checkpoint trained by a supported "
                "replay or ILOSAC-inspired Flow module"
            )
        del checkpoint
    return attention.main(["eval", *arguments[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
