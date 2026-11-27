#!/usr/bin/env python3
"""State-conditioned DiT with fixed adjacent-pair K/V compression."""
from __future__ import annotations

import argparse
import importlib
import sys

import torch
from torch import nn
from torch.nn import functional as F

import dit_rectified_flow as _base
import dit_state_attention as _v1


KV_TOKENS = 8
ARCHITECTURE = "state_conditioned_temporal_attention_kv8_v1"
KV_AGGREGATION = "fixed_adjacent_mean"


class StateConditionedDiTBlockKV8(_v1.StateConditionedDiTBlock):
    """Keep Q at 16 tokens and aggregate K, V, and bias keys to 8."""

    def __init__(self, original, horizon, kv_tokens=KV_TOKENS):
        super().__init__(original, horizon)
        horizon = int(horizon)
        kv_tokens = int(kv_tokens)
        if horizon % kv_tokens:
            raise ValueError(
                "action horizon must be divisible by KV token count"
            )
        group = horizon // kv_tokens
        average = torch.zeros(kv_tokens, horizon)
        for row in range(kv_tokens):
            average[row, row * group:(row + 1) * group] = 1.0 / group
        self.kv_tokens = kv_tokens
        self.register_buffer(
            "kv_average",
            average,
            persistent=False,
        )

    def _state_attention(self, value, motion_condition):
        attention = self.self_attention
        batch, tokens, _ = value.shape
        query = attention.q(value).reshape(
            batch, tokens, attention.heads, attention.head_dim
        ).transpose(1, 2)
        key, content = attention.kv(value).reshape(
            batch, tokens, attention.heads, attention.head_dim * 2
        ).chunk(2, dim=-1)
        key = key.transpose(1, 2)
        content = content.transpose(1, 2)
        average = self.kv_average.to(dtype=key.dtype)
        key = torch.einsum("rn,bhnd->bhrd", average, key)
        content = torch.einsum("rn,bhnd->bhrd", average, content)
        bias = self.temporal_bias(motion_condition, tokens)
        bias = torch.einsum(
            "rn,bhin->bhir",
            self.kv_average.to(dtype=bias.dtype),
            bias,
        ).to(query.dtype)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            content,
            attn_mask=bias,
            dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).reshape(
            batch, tokens, attention.dim
        )
        return attention.output(attended)


class StateAttentionKV8Policy(_v1.StateConditionedDiTRectifiedFlowPolicy):
    """Reuse v1 encoders and Flow sampling with compressed action KV."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = nn.ModuleList([
            StateConditionedDiTBlockKV8(block, self.n_action_steps)
            for block in self.blocks
        ])


def _prepare_training(entry, arguments):
    _v1.ARCHITECTURE = ARCHITECTURE
    _v1._prepare_training(entry, arguments)
    original_save = entry._atomic_torch_save

    def save(payload, destination):
        payload["architecture"] = ARCHITECTURE
        payload["config"].update({
            "attention_mechanism": ARCHITECTURE,
            "attention_kv_tokens": KV_TOKENS,
            "attention_kv_aggregation": KV_AGGREGATION,
        })
        return original_save(payload, destination)

    entry._atomic_torch_save = save


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "eval"))
    mode = parser.parse_args(arguments[:1]).mode
    forwarded = arguments[1:]
    _v1.ARCHITECTURE = ARCHITECTURE
    _base.DiTRectifiedFlowPolicy = StateAttentionKV8Policy
    if mode == "train":
        entry = importlib.import_module("train_dit_rectified_flow")
        _prepare_training(entry, forwarded)
        entry.main(forwarded)
    else:
        entry = importlib.import_module("eval_policy")
        sys.argv = [entry.__file__, *forwarded]
        entry.main()


if __name__ == "__main__":
    main()
