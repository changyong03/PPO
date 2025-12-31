import torch
import wandb
import gymnasium as gym
import numpy as np

from config import Config
from network import actor, critic
from replay_buffer import PPOBuffer
from torch.optim import Adam
import torch.nn.functional as F

class Trainer:
    def __init__ (self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        wandb.init(
            project = "ppo",
            config = config.__dict__,
            reinit = True
        )

        self.env = gym.make(self.config.env_name)
        #self.env.observation_space.shape => (3. )
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        self.actor = actor(obs_dim, act_dim, self.config.hidden_dim).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr = self.config.lr)
        self.critic = critic(obs_dim, self.config.hidden_dim).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.config.lr)

        self.buffer = PPOBuffer(self.device)

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.actor.sample(state)
        
        scaled_action = action * 2.0 
        return scaled_action.cpu().numpy().flatten(), log_prob.item()
    
    def gae(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        with torch.no_grad():
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.config.gamma * next_values[t] * (1 - dones[t]) - values[t]
                gae = delta + self.config.gamma * self.config.lmbda * (1-dones[t]) * gae
                advantages.insert(0, gae)

        return torch.tensor(advantages, dtype=torch.float32).to(self.device).view(-1,1)
    
    def update(self):
        states, actions, rewards, next_states, dones, old_log_probs = self.buffer.get_batch()
        
        normalized_actions = actions / 2.0

        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
        
        advantages = self.gae(rewards, values, next_values, dones)
        returns = advantages + values

        #advantage normalization(for stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_length = states.size(0)
        indices = np.arange(dataset_length)

        for _ in range(self.config.num_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_length, self.config.batch_size):
                end = start + self.config.batch_size
                idx = indices[start:end]

                mb_states = states[idx]
                mb_actions = normalized_actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                curr_log_probs, entropy = self.actor.evaluate(mb_states, mb_actions)
                curr_values = self.critic(mb_states)

                ratio = torch.exp(curr_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * mb_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(curr_values, mb_returns)
                loss = actor_loss + critic_loss - 0.01 * entropy

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                self.actor_optim.step()
                self.critic_optim.step()

                wandb.log({
                    "loss": loss.item(),
                    "actor_loss": actor_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "entropy": entropy.item()
                })
        
        self.buffer.clear()

    def train(self):
        global_step = 0
        
        for episode in range(10000):
            state, _ = self.env.reset(seed=self.config.seed)
            score = 0
            done = False
            
            while not done:
                global_step += 1
                
                action, log_prob = self.get_action(state)
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.buffer.put((state, action, reward, next_state, done, log_prob))
                
                state = next_state
                score += reward
                
                if global_step % self.config.horizon == 0:
                    self.update()
            
            print(f"Episode: {episode} | Score: {score:.2f} | Steps: {global_step}")
            wandb.log({"score": score, "episode": episode})
            
if __name__ == "__main__":
    config = Config()
    trainer = Trainer(config)
    trainer.train()