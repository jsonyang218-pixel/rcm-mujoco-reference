"""Target-relative near DiT: original six blocks, sixteen actions, four RF steps."""
import types
import torch

NOISE_SEED = 434341
INPUT_SCALES = dict(oct_mm=[.05,.05,.05,1.], velocity_deg_deg_mm_s=[.1,.1,.1], previous_deg_deg_mm=[.005,.005,.005])

def template(model):
    return torch.randn(1, model.n_action_steps, model.action_dim,
        device=model.action_mean.device, dtype=torch.float32,
        generator=torch.Generator(device=model.action_mean.device).manual_seed(NOISE_SEED))

def integrate(model, args, noise=None, condition=None):
    condition = model._condition(*args) if condition is None else condition
    actions = (template(model) if noise is None else noise).expand(args[1].shape[0], -1, -1)
    for index in range(4):
        tau = torch.full((actions.shape[0],), (index+.5)/4, device=actions.device, dtype=actions.dtype)
        actions = actions + .25 * model._predict_velocity_from_condition(actions, tau, *condition)
    return actions * model.action_std[None,None] + model.action_mean[None,None]

@torch.no_grad()
def fixed_sample(self, image, state, oct_feature, velocity, previous_action, steps=None, generator=None):
    assert steps in (None,4)
    return integrate(self,(image,state,oct_feature,velocity,previous_action))

def constant_visual(self, image):
    # Do not run a CNN on masked pictures: the same exact constants in all modes.
    return torch.zeros(image.shape[0],2,49,self._relative_dim,device=image.device,dtype=torch.float32)

def relative_condition(self,image,state,oct_feature,velocity,previous_action):
    valid = oct_feature[...,3:4] > .5
    safe_oct = torch.cat((torch.where(valid,oct_feature[...,:3],0.), valid.to(oct_feature.dtype)),dim=-1)
    return self._relative_original_condition(image,torch.zeros_like(state),safe_oct,velocity,previous_action)

def install(model,mean=None,std=None):
    with torch.no_grad():
        if mean is not None:
            model.action_mean.copy_(torch.as_tensor(mean,device=model.action_mean.device))
            model.action_std.copy_(torch.as_tensor(std,device=model.action_std.device))
        for prefix,scale in [('state',[1.,1.,1.]),('oct',INPUT_SCALES['oct_mm']),
                             ('velocity',INPUT_SCALES['velocity_deg_deg_mm_s']),
                             ('previous_action',INPUT_SCALES['previous_deg_deg_mm'])]:
            getattr(model,prefix+'_mean').zero_()
            getattr(model,prefix+'_std').copy_(torch.tensor(scale,device=model.action_mean.device))
    if not getattr(model,'_relative_installed',False):
        model._relative_original_condition = model._condition
        model._condition = types.MethodType(relative_condition,model)
        model.vision._relative_dim = model.dit_dim
        model.vision.forward = types.MethodType(constant_visual,model.vision)
        model._relative_installed = True
    model.sample = types.MethodType(fixed_sample,model)
    return model

def allow_visual_cache(model):
    # Relative conditioning never consumes the image values or their encoding.
    return model.vision.forward
