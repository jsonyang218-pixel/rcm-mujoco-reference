#!/usr/bin/env python3
"""Train/evaluate the Frontiers 2022 Algorithm 1 adaptation on RCM data."""
from __future__ import annotations

import argparse
from functools import partial
import importlib
from pathlib import Path
import sys

import torch

import dit_rectified_flow as flow
import dit_replay_actor_critic as replay
import frontiers2022_ddpg as ddpg


def configure_parser(parser):
    parser.description = __doc__
    parser.add_argument("--fixed-noise-seed", type=int, default=20261118)
    parser.set_defaults(
        fm_weight=0.0, q_weight=1.0, bias_lr_scale=1.0,
        progress_weight=0.0, distance_weight=0.0, effort_weight=0.0,
        seed=20261117,
    )


def validate_args(args, parser):
    if (args.progress_weight, args.distance_weight, args.effort_weight) != (0.0, 0.0, 0.0):
        parser.error("legacy Replay reward weights must stay zero; this entry uses paper Eqs. (5)-(7)")
    if args.distance_scale_mm != 1.0:
        parser.error("distance-scale-mm must stay 1; paper reward converts mm to metres explicitly")


def metadata(args):
    return {
        "paper_doi": ddpg.PAPER_DOI,
        "algorithm": "Algorithm 1 DDPG; independent RCM implementation",
        "official_author_source": False,
        "training_regime": "offline stored transitions; not original online exploration",
        "actor_sampling": "fixed_noise_deterministic_flow",
        "fixed_noise_seed": args.fixed_noise_seed,
        "critic_type": "single_Q_geometry_only",
        "actor_weight_decay": 0.0,
        "flow_anchor_is_project_extension": args.fm_weight > 0,
        "demonstration_sampling": "uniform_windows" if args.fm_weight > 0 else "unused",
        "dagger_repeat": 1,
        "replay_sampling": "uniform_with_replacement",
        "train_loss_metric": "flow_matching" if args.fm_weight > 0 else "negative_weighted_policy_Q",
        "checkpoint_selection": "minimum validation Flow loss; not validation RL return",
        "reward_c1": 500.0, "reward_c2": 15.0, "reward_a_m": 1.0, "reward_n": 35,
        "obstacle_reward_enabled": False,
    }


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "eval"))
    mode = parser.parse_args(arguments[:1]).mode
    if mode == "train":
        return replay.train(arguments[1:], components={
            "training_module": ddpg.TRAINING_MODULE,
            "variant_label": "frontiers2022_algorithm1",
            "policy_class": ddpg.DeterministicFlowPolicy,
            "policy_kwargs": lambda args: {"fixed_noise_seed": args.fixed_noise_seed},
            "critic_class": ddpg.SingleQ,
            "transition_records": ddpg.transition_records,
            "update_critic": ddpg.update_critic, "update_actor": ddpg.update_actor,
            "configure_parser": configure_parser, "validate_args": validate_args,
            "metadata": metadata, "allow_zero_fm": True, "actor_weight_decay": 0.0,
            "replay_metadata": {
                "reward": "-500*0.5*(norm(OCT_next_mm)/1000)^2",
                "reward_distance_source": "observed next-state OCT navigation; mm converted to m",
                "reward_equations": "Frontiers 2022 Eqs. (5)-(7); obstacle term disabled",
                "obstacle_reward_enabled": False,
            },
        })

    evaluation_parser = argparse.ArgumentParser(add_help=False)
    evaluation_parser.add_argument("--model", type=Path)
    options, _ = evaluation_parser.parse_known_args(arguments[1:])
    if options.model is None:
        if "--help" not in arguments[1:] and "-h" not in arguments[1:]:
            parser.error("eval requires --model")
        noise_seed = 20261118
    else:
        checkpoint = torch.load(options.model, map_location="cpu", weights_only=False)
        if checkpoint.get("training_module") != ddpg.TRAINING_MODULE:
            raise ValueError("eval requires this Frontiers DDPG checkpoint; cannot substitute old Replay/v1")
        noise_seed = int(checkpoint["config"]["fixed_noise_seed"])
        del checkpoint
    # eval_policy imports the actor from this module when loading DiT weights.
    flow.DiTRectifiedFlowPolicy = partial(ddpg.DeterministicFlowPolicy, fixed_noise_seed=noise_seed)
    entry = importlib.import_module("eval_policy")
    sys.argv = [entry.__file__, *arguments[1:]]
    return entry.main()


if __name__ == "__main__":
    raise SystemExit(main())
