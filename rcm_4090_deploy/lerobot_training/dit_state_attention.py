#!/usr/bin/env python3
"""State-conditioned temporal attention for the existing RCM DiT-RF policy.

Place this file beside dit_rectified_flow.py, then run:
    python dit_state_attention.py train <existing training arguments>
    python dit_state_attention.py eval <existing evaluation arguments>
"""
from __future__ import annotations

import argparse
import copy
import importlib
import os
import sys

# This PyTorch build otherwise JIT-compiles a Triton helper during the new
# attention-bias backward pass, which requires Python development headers.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
from torch import nn
from torch.nn import functional as F

try:
    from . import dit_rectified_flow as _base
except ImportError:
    import dit_rectified_flow as _base


NUM_TEMPLATES = 4
GATE_HIDDEN_DIM = 32
BACKBONE_LR_SCALE = 0.1
ARCHITECTURE = "state_conditioned_temporal_attention_v1"
_OriginalPolicy = _base.DiTRectifiedFlowPolicy


class StateConditionedTemporalBias(nn.Module):
    """B[b,h,i,j] = sum_m gate(state)[b,h,m] * template[h,m,i-j]."""

    def __init__(self, condition_dim, heads, horizon):
        super().__init__()
        self.heads = heads
        self.gate = nn.Sequential(
            nn.Linear(condition_dim, GATE_HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(GATE_HIDDEN_DIM, heads * NUM_TEMPLATES),
        )
        self.templates = nn.Parameter(
            torch.empty(heads, NUM_TEMPLATES, 2 * horizon - 1)
        )
        nn.init.normal_(self.templates, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        position = torch.arange(horizon)
        self.register_buffer(
            "relative_index",
            position[:, None] - position[None, :] + horizon - 1,
            persistent=False,
        )

    def forward(self, condition, tokens):
        weights = self.gate(condition).reshape(
            condition.shape[0], self.heads, NUM_TEMPLATES
        )
        templates = self.templates[
            :, :, self.relative_index[:tokens, :tokens]
        ]
        return torch.einsum("bhm,hmij->bhij", weights, templates)


class StateConditionedDiTBlock(_base.DiTBlock):
    """Keep the original block weights and bias only action self-attention."""

    def __init__(self, original, horizon):
        nn.Module.__init__(self)
        for name, child in original.named_children():
            self.add_module(name, child)
        attention = self.self_attention
        self.temporal_bias = StateConditionedTemporalBias(
            attention.dim, attention.heads, horizon
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
        bias = self.temporal_bias(motion_condition, tokens).to(query.dtype)
        attended = F.scaled_dot_product_attention(
            query,
            key.transpose(1, 2),
            content.transpose(1, 2),
            attn_mask=bias,
            dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).reshape(
            batch, tokens, attention.dim
        )
        return attention.output(attended)

    def forward(
        self, tokens, condition_tokens, global_condition, motion_condition
    ):
        (
            shift_self, scale_self, gate_self,
            shift_cross, scale_cross, gate_cross,
            shift_mlp, scale_mlp, gate_mlp,
        ) = self.modulation(global_condition).chunk(9, dim=-1)
        value = tokens + gate_self.unsqueeze(1) * self._state_attention(
            self._modulate(self.norm_self(tokens), shift_self, scale_self),
            motion_condition,
        )
        value = value + gate_cross.unsqueeze(1) * self.cross_attention(
            self._modulate(self.norm_cross(value), shift_cross, scale_cross),
            condition_tokens,
        )
        return value + gate_mlp.unsqueeze(1) * self.mlp(
            self._modulate(self.norm_mlp(value), shift_mlp, scale_mlp)
        )


class StateConditionedDiTRectifiedFlowPolicy(_OriginalPolicy):
    """Reuse the original observations, normalization, Flow loss and sampler."""

    def __init__(self, *args, oct_symlog_enabled=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.oct_symlog_enabled = bool(oct_symlog_enabled)
        self.blocks = nn.ModuleList([
            StateConditionedDiTBlock(block, self.n_action_steps)
            for block in self.blocks
        ])
        if not self.oct_symlog_enabled:
            return
        hidden_dim = max(64, self.dit_dim // 2)
        self.oct_symlog_geometry = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.dit_dim),
        )
        self.oct_symlog_global = nn.Sequential(
            nn.Linear(3 * self.n_obs_steps, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.dit_dim),
        )
        nn.init.zeros_(self.oct_symlog_geometry[-1].weight)
        nn.init.zeros_(self.oct_symlog_geometry[-1].bias)
        nn.init.zeros_(self.oct_symlog_global[-1].weight)
        nn.init.zeros_(self.oct_symlog_global[-1].bias)

    def normalize_oct_symlog(self, oct_feature):
        """Canonical deployable OCT transform shared by train and eval."""
        oct_float = oct_feature.float()
        valid = (oct_float[..., 3:4] > 0.5).to(torch.float32)
        oct_mm = torch.where(valid.bool(), oct_float[..., :3], 0.0)
        oct_symlog = torch.sign(oct_mm) * torch.log1p(
            torch.abs(oct_mm) / 0.001
        )
        scale = torch.log1p(
            self.oct_std[:3].float() / 0.001
        ).clamp_min(1.0)
        return (oct_symlog / scale) * valid, valid

    def _condition(self, image, state, oct_feature, velocity, previous_action):
        condition_tokens, global_condition = super()._condition(
            image, state, oct_feature, velocity, previous_action
        )
        if not self.oct_symlog_enabled:
            return condition_tokens, global_condition

        device_type = oct_feature.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            oct_symlog, valid = self.normalize_oct_symlog(oct_feature)
            geometry_residual = self.oct_symlog_geometry(oct_symlog)
            geometry_residual = geometry_residual * valid
            global_residual = self.oct_symlog_global(
                oct_symlog.flatten(start_dim=1)
            )
            global_valid = valid.squeeze(-1).any(dim=1, keepdim=True)
            global_residual = global_residual * global_valid

        geometry_count = self.n_obs_steps
        vision_tokens = condition_tokens[:, :-geometry_count]
        geometry_tokens = condition_tokens[:, -geometry_count:]
        condition_tokens = torch.cat(
            [vision_tokens, geometry_tokens + geometry_residual.to(geometry_tokens.dtype)],
            dim=1,
        )
        global_condition = global_condition + global_residual.to(
            global_condition.dtype
        )
        return condition_tokens, global_condition

    def _predict_velocity_from_condition(
        self, interpolated_actions, timestep, condition_tokens, global_condition
    ):
        # Original geometry encoder: q + OCT + velocity + previous command.
        # Keep diffusion/Flow time out of this state-conditioning path.
        motion_condition = global_condition
        timestep_embedding = self.time_mlp(
            _base._sinusoidal(timestep, self.dit_dim)
        )
        global_condition = global_condition + timestep_embedding
        tokens = self.action_input(interpolated_actions) + self.action_position
        for block in self.blocks:
            tokens = block(
                tokens, condition_tokens, global_condition, motion_condition
            )
        shift, scale = self.final_modulation(global_condition).chunk(2, dim=-1)
        tokens = (
            self.final_norm(tokens) * (1 + scale.unsqueeze(1))
            + shift.unsqueeze(1)
        )
        return self.output(tokens)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Warm-start old checkpoints by filling ONLY the newly added parameters.
        # All original missing/unexpected keys and shape mismatches stay strict.
        state_dict = copy.copy(state_dict)
        for name, value in self.state_dict().items():
            if ".temporal_bias." in name:
                state_dict.setdefault(name, value)
        if assign:
            return super().load_state_dict(state_dict, strict=strict, assign=True)
        return super().load_state_dict(state_dict, strict=strict)


def _prepare_training(entry, arguments):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--init-from", default=None)
    options, _ = parser.parse_known_args(arguments)
    if options.init_from is not None:
        checkpoint = torch.load(
            options.init_from, map_location="cpu", weights_only=False
        )
        config = checkpoint["config"]
        names = (
            "state_mean", "state_std", "oct_mean", "oct_std",
            "action_mean", "action_std", "velocity_mean", "velocity_std",
            "previous_action_mean", "previous_action_std",
        )
        statistics = {name: config[name] for name in names}
        entry._dataset_statistics = lambda dataset: statistics
        del checkpoint

    original_train_epoch = entry.train_epoch

    def train_epoch(model, loader, optimizer, scaler, device, epoch):
        if epoch == 0:
            backbone, attention = [], []
            for name, parameter in model.named_parameters():
                group = attention if ".temporal_bias." in name else backbone
                group.append(parameter)
            group = optimizer.param_groups[0]
            attention_lr = group["lr"]
            group["params"] = backbone
            group["lr"] = attention_lr * BACKBONE_LR_SCALE
            optimizer.add_param_group({"params": attention, "lr": attention_lr})
        return original_train_epoch(model, loader, optimizer, scaler, device, epoch)

    entry.train_epoch = train_epoch
    original_save = entry._atomic_torch_save

    def save(payload, destination):
        payload["architecture"] = ARCHITECTURE
        payload["config"].update({
            "attention_mechanism": ARCHITECTURE,
            "attention_num_templates": NUM_TEMPLATES,
            "attention_gate_hidden_dim": GATE_HIDDEN_DIM,
            "backbone_lr_scale": BACKBONE_LR_SCALE,
        })
        return original_save(payload, destination)

    entry._atomic_torch_save = save


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "eval"))
    mode = parser.parse_args(arguments[:1]).mode
    forwarded = arguments[1:]
    _base.DiTRectifiedFlowPolicy = StateConditionedDiTRectifiedFlowPolicy
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
