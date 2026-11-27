"""Fixed-template DiT-RF. Original condition, attention and RF velocity are reused."""
import types
import torch

NOISE_SEED = 434341


def template(model):
    return torch.randn(1, model.n_action_steps, model.action_dim,
                       device=model.action_mean.device, dtype=torch.float32,
                       generator=torch.Generator(device=model.action_mean.device).manual_seed(NOISE_SEED))


def integrate(model, args, noise=None, condition=None):
    # Exactly the original midpoint-in-time Euler RF sampler, with gradients.
    condition = model._condition(*args) if condition is None else condition
    actions = (template(model) if noise is None else noise).expand(args[1].shape[0], -1, -1)
    for index in range(4):
        tau = torch.full((actions.shape[0],), (index + .5) / 4,
                         device=actions.device, dtype=actions.dtype)
        actions = actions + .25 * model._predict_velocity_from_condition(actions, tau, *condition)
    return actions * model.action_std[None, None] + model.action_mean[None, None]


@torch.no_grad()
def fixed_sample(self, image, state, oct_feature, velocity, previous_action, steps=None, generator=None):
    assert steps in (None, 4), 'Only registered four-step RF sampling is authorized'
    return integrate(self, (image, state, oct_feature, velocity, previous_action))


def install(model, mean=None, std=None):
    if mean is not None:
        model.action_mean.copy_(torch.as_tensor(mean, device=model.action_mean.device))
        model.action_std.copy_(torch.as_tensor(std, device=model.action_std.device))
    model.sample = types.MethodType(fixed_sample, model)
    return model


def allow_visual_cache(model):
    original = model.vision.forward

    def forward(value):
        if value.ndim == 4:
            assert value.shape[1:] == (2, 49, 384) and value.dtype == torch.float32
            return value
        assert value.ndim == 5
        return original(value)
    model.vision.forward = forward
    return original
