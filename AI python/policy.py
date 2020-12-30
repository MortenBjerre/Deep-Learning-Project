import torch
import torch.nn as nn
from torch.distributions import Categorical
from utils import orthogonal_init
from PPO import PPO_loss, clipped_value_loss

class Policy(nn.Module):
    '''
    Policy class used to map from screen input to action distribution and
    value function. 
    '''
    def __init__(self, encoder, feature_dim, num_actions): 
        super().__init__()
        self.encoder = encoder
        self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
        self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)
        self.ppo_loss = PPO_loss()
        self.value_loss = clipped_value_loss()

    def act(self, x):
        with torch.no_grad():
            x = x.cuda().contiguous()
            dist, value = self.forward(x)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.cpu(), log_prob.cpu(), value.cpu()

    def forward(self, x):
        x = self.encoder(x)
        logits = self.policy(x)
        value = self.value(x).squeeze(1)
        dist = Categorical(logits=logits)

        return dist, value