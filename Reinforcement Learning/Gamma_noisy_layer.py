import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt





class NoisyLinearGamma(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.02):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        eps_in = self._scale_noise_gamma(self.in_features)
        eps_out = self._scale_noise_gamma(self.out_features)
        self.weight_epsilon.copy_(eps_out.unsqueeze(1) * eps_in.unsqueeze(0))
        self.bias_epsilon.copy_(eps_out)


    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)

    @staticmethod
    def _scale_noise_gamma(size, shape=2.0, scale=1.0):
        gamma_dist = torch.distributions.Gamma(torch.tensor([shape]), torch.tensor([scale]))
        x = gamma_dist.sample((size,)).squeeze()
        return x - shape * scale





class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        nn.Module.__init__(self)
        self.noisy1 = NoisyLinearGamma(state_dim, 256)
        self.noisy2 = NoisyLinearGamma(256, 256)
        self.noisy3 = NoisyLinearGamma(256, action_dim)
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
        sa = torch.cat([state, action], dim=1).to(torch.device("cuda"))
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q2, q1





class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action

        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).cuda()
        self.critic_target = Critic(state_dim, action_dim).cuda()
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.replay_buffer = []
        self.buffer_capacity = 2_000_000
        self.batch_size = 128
        self.policy_delay = 2
        self.policy_update_step = 0
        self.exploration_noise = 0.3

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).cuda()
        action = self.actor(state_t).cpu().data.numpy().flatten()
        noise = np.random.normal(0, self.exploration_noise * self.max_action, size=action.shape)
        action = np.clip(action + noise, -self.max_action, self.max_action)
        self.exploration_noise = max(0.1, self.exploration_noise * 0.999)
        return action

    def store_transition(self, s, a, r, s_next, done):
        if len(self.replay_buffer) >= self.buffer_capacity:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((s, a, r, s_next, float(done)))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).cuda()
        actions = torch.FloatTensor(actions).cuda()
        rewards = torch.FloatTensor(rewards).unsqueeze(1).cuda()
        next_states = torch.FloatTensor(next_states).cuda()
        dones = torch.FloatTensor(dones).unsqueeze(1).cuda()

        with torch.no_grad():
            next_actions = self.actor_target(next_states).clamp(-self.max_action, self.max_action)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = rewards + (1 - dones) * self.gamma * torch.min(q1_target, q2_target)

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




env = gym.make(
    "Hopper-v5",
    render_mode= "human",
    forward_reward_weight=10.0,
    ctrl_cost_weight=0.005,
    healthy_reward=2.0,
    terminate_when_unhealthy=False,
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
agent = TD3Agent(state_dim, action_dim, max_action)





num_episodes = 1200
save_interval = 600

for ep in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    for _ in range(1000):
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        episode_reward += reward
        if done: break

    print(f"Episode {ep+1}, Reward: {episode_reward:.2f}")
    if (ep + 1) % save_interval == 0:
        torch.save(agent.actor.state_dict(), f"td3_actor_gamma_{ep+1}.pth")



print("\n=== Testing Phase ===")
agent.actor.load_state_dict(torch.load(r"C:\Users\saatw\td3_actor_noisy_gamma_1200.pth"))
agent.actor.eval()

num_test_episodes = 10
total_rewards = []

for i in range(num_test_episodes):
    state, _ = env.reset()
    done = False
    ep_reward = 0
    while not done:
        s_tensor = torch.FloatTensor(state).unsqueeze(0).cuda()
        action = agent.actor(s_tensor).cpu().data.numpy().flatten()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        ep_reward += reward
    total_rewards.append(ep_reward)
    print(f"Test Episode {i+1}: Reward = {ep_reward:.2f}")
env.close()
