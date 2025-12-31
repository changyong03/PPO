import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
class actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super(actor, self).__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.mean = nn.Linear(hidden_dim, act_dim)

        self.std = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        obs = self.backbone(obs)
        mean = torch.tanh(self.mean(obs))
        std = F.softplus(self.std(obs)) + 1e-3

        return mean, std

    def sample(self, obs):
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        act_raw = dist.sample()
        act = torch.tanh(act_raw)
        #Jacobian
        log_prob_raw = dist.log_prob(act_raw)

        log_prob = log_prob_raw - torch.log(1 - act.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim = -1, keepdim = True)

        return act, log_prob
    
    def evaluate(self, obs, act):
        mu, std = self.forward(obs)
        dist = Normal(mu, std)

        act = torch.clamp(act, -0.999, 0.999)
        act_raw = torch.atanh(act)

        log_prob_raw = dist.log_prob(act_raw)
        log_prob = log_prob_raw - torch.log(1 - act.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim = -1, keepdim=True)

        entropy = dist.entropy().sum(dim = -1).mean()

        return log_prob, entropy
        

class critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super(critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, obs):
        return self.net(obs)
