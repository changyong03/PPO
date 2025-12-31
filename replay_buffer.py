import torch
import numpy as np

class PPOBuffer:
    def __init__(self, device):
        self.device = device
        self.clear()

    def put(self, transition):
        state, action, reward, next_state, done, log_prob = transition
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def get_batch(self):
        s = torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device)
        a = torch.tensor(np.array(self.actions), dtype=torch.float32).view(-1, 1).to(self.device)
        r = torch.tensor(np.array(self.rewards), dtype=torch.float32).view(-1, 1).to(self.device)
        next_s = torch.tensor(np.array(self.next_states), dtype=torch.float32).to(self.device)
        d = torch.tensor(np.array(self.dones), dtype=torch.float32).view(-1, 1).to(self.device)
        old_log_p = torch.tensor(np.array(self.log_probs), dtype=torch.float32).view(-1, 1).to(self.device)
        
        return s, a, r, next_s, d, old_log_p

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []