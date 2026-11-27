"""FP32 frozen original vision + spatial softmax + deterministic next-action head."""
import torch
from torch import nn

class HeadOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual=nn.Sequential(nn.Linear(1536,64),nn.SiLU())
        self.state=nn.Sequential(nn.Linear(20,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.fusion=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,3))
    def forward(self,visual,state):return self.fusion(torch.cat([self.visual(visual),self.state(state)],-1))

class TerminalHead(nn.Module):
    def __init__(self,vision,scales):
        super().__init__();self.vision=vision;self.vision.requires_grad_(False);self.vision.eval();self.head=HeadOnly()
        for k in ('state_mean','state_std','velocity_mean','velocity_std','previous_action_mean','previous_action_std','action_mean','action_std'):
            self.register_buffer(k,torch.as_tensor(scales[k],dtype=torch.float32).clone().reshape(3))
        self.last_token_shape=None
    def train(self,mode=True):
        super().train(mode);self.vision.eval();return self
    def visual_features(self,image):
        assert image.dtype==torch.float32 and image.shape[1:]==(2,3,224,224)
        with torch.no_grad():tokens=self.vision(image)
        self.last_token_shape=list(tokens.shape)
        assert tokens.shape[1:]==(2,49,384),tokens.shape
        # Original forward creates tokens with feature.flatten(2).transpose(1,2).
        maps=tokens.transpose(-1,-2).reshape(-1,2,384,7,7)
        probability=maps.flatten(-2).softmax(-1).reshape(-1,2,384,7,7)
        grid=torch.linspace(-1.,1.,7,device=maps.device,dtype=torch.float32)
        x=(probability.sum(-2)*grid).sum(-1);y=(probability.sum(-1)*grid).sum(-1)
        return torch.stack([x,y],-1).flatten(1)
    def state_features(self,state,oct_feature,velocity,previous_action):
        assert state.shape[1:]==(2,3) and oct_feature.shape[1:]==(2,4)
        # Production velocity/previous-action may be broadcast across history; keep unique current vector.
        if velocity.ndim==3:
            assert torch.equal(velocity[:,0],velocity[:,-1]);velocity=velocity[:,-1]
        if previous_action.ndim==3:
            assert torch.equal(previous_action[:,0],previous_action[:,-1]);previous_action=previous_action[:,-1]
        assert velocity.shape[1:]==(3,) and previous_action.shape[1:]==(3,)
        normalized_oct=torch.cat([oct_feature[...,:3]/.1,oct_feature[...,3:4]],-1)
        result=torch.cat([((state-self.state_mean)/self.state_std).flatten(1),normalized_oct.flatten(1),(velocity-self.velocity_mean)/self.velocity_std,(previous_action-self.previous_action_mean)/self.previous_action_std],-1)
        assert result.shape[-1]==20 and torch.isfinite(result).all()
        return result
    def from_features(self,features,state,oct_feature,velocity,previous_action):
        assert features.dtype==torch.float32
        normalized=self.head(features,self.state_features(state,oct_feature,velocity,previous_action))
        action=normalized*self.action_std+self.action_mean
        assert torch.isfinite(action).all()
        return action
    def forward(self,image,state,oct_feature,velocity,previous_action):return self.from_features(self.visual_features(image),state,oct_feature,velocity,previous_action)
    def sample(self,image,state,oct_feature,velocity,previous_action,**compatibility):
        # Interface adapter only: no flow integration, RNG, or action chunk is evaluated.
        return self(image,state,oct_feature,velocity,previous_action).unsqueeze(1)
