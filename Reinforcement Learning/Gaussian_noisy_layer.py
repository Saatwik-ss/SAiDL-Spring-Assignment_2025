import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from tqdm import tqdm

# ---------------------------- NoisyLinear Definition ----------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        # Parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffer
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / self.in_features**0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())


# ---------------------------- Actor & Critic Definitions ----------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        nn.Module.__init__(self)
        self.noisy1 = NoisyLinear(state_dim, 256)
        self.noisy2 = NoisyLinear(256, 256)
        self.noisy3 = NoisyLinear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.noisy1(state))
        x = F.relu(self.noisy2(x))
        return torch.tanh(self.noisy3(x)) * self.max_action

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()
        self.noisy3.reset_noise()


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.fc4 = nn.Linear(state_dim + action_dim, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1).to(torch.device("cuda"))
        q1 = F.relu(self.fc1(state_action))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        q2 = F.relu(self.fc4(state_action))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q2,q1;


# ---------------------------- TD3 Agent ----------------------------
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, actor_lr=3e-4, critic_lr=3e-4):
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action


        self.actor = Actor(state_dim, action_dim, max_action).to(torch.device("cuda"))
        self.actor_target = Actor(state_dim, action_dim, max_action).to(torch.device("cuda"))
        self.critic = Critic(state_dim, action_dim).to(torch.device("cuda"))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target = Critic(state_dim, action_dim).to(torch.device("cuda"))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.replay_buffer = []
        self.buffer_capacity = 2000000
        self.batch_size = 128  
        self.policy_delay = 2
        self.policy_update_step = 0
        self.exploration_noise = 0.3  

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(torch.device("cuda"))
        action = self.actor(state).cpu().data.numpy().flatten()
        noise = np.random.normal(0, self.exploration_noise * self.max_action, size=action.shape)
        action = np.clip(action + noise, -self.max_action, self.max_action)
        self.exploration_noise = max(0.1, self.exploration_noise * 0.999)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.replay_buffer) >= self.buffer_capacity:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state, float(done)))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(torch.device("cuda"))
        actions = torch.FloatTensor(actions).to(torch.device("cuda"))
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(torch.device("cuda"))
        next_states = torch.FloatTensor(next_states).to(torch.device("cuda"))
        dones = torch.FloatTensor(dones).unsqueeze(1).to(torch.device("cuda"))

        next_actions = (self.actor_target(next_states)).clamp(-self.max_action, self.max_action)

        q1_target, q2_target = self.critic_target(next_states, next_actions)
        q_target = rewards + (1 - dones) * self.gamma * torch.min(q1_target, q2_target).detach()

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.policy_update_step % self.policy_delay == 0:
            actor_loss = -self.critic(states, self.actor(states))[0].mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        self.policy_update_step += 1

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


# ---------------------------- Environment Setup ----------------------------
env = gym.make(
    "Hopper-v5",
    render_mode="human",
    forward_reward_weight=10.0,
    ctrl_cost_weight=0.005,
    healthy_reward=2.0,
    terminate_when_unhealthy=False,
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
agent = TD3Agent(state_dim, action_dim, max_action)

# ---------------------------- Training Phase ----------------------------
num_episodes = 3000
save_interval = 600
for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    for t in range(1000):
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)

        agent.train()
        state = next_state
        episode_reward += reward

        if done:
            break
    print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}")

    if (episode + 1) % save_interval == 0:
        torch.save(agent.actor.state_dict(), f"td3_actor_noisy{episode+1}.pth")
        torch.save(agent.critic.state_dict(), f"td3_critic_noisy{episode+1}.pth")
        
# ---------------------------- Testing Phase ----------------------------
print("\n=== Starting Testing Phase ===")

agent.actor.load_state_dict(torch.load(r"C:\Users\saatw\td3_actor_noisy1200.pth"))
agent.actor.eval()

num_test_episodes = 10
total_rewards = []

for episode in range(num_test_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(torch.device("cuda"))
        action = agent.actor(state_tensor).cpu().data.numpy().flatten()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        episode_reward += reward

    total_rewards.append(episode_reward)
    print(f"Test Episode {episode+1}: Reward = {episode_reward:.2f}")

# Compute and display average performance
avg_reward = np.mean(total_rewards)
print(f"\nAverage Reward over {num_test_episodes} test episodes: {avg_reward:.2f}")
env.close()

import matplotlib.pyplot as plt
# ---------------------------- Reward Chart ----------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_test_episodes + 1), total_rewards, marker='o', linestyle='-', color='g', alpha=0.7)
plt.xlabel("Test Episode")
plt.ylabel("Reward")
plt.title("TD3 Test Rewards Over 10 Episodes")
plt.xticks(range(1, num_test_episodes + 1))
plt.grid()
plt.savefig("test_rewards.png")
plt.show()

# Compute and display average performance
avg_reward = np.mean(total_rewards)
print(f"\nAverage Reward over {num_test_episodes} test episodes: {avg_reward:.2f}")
env.close()
