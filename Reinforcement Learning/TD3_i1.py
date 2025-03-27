import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        state = state.to(device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.max_action  # Deterministic action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        # Q2 architecture
        self.fc4 = nn.Linear(state_dim + action_dim, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1).to(device)
        # Q1 forward
        q1 = F.relu(self.fc1(state_action))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        # Q2 forward
        q2 = F.relu(self.fc4(state_action))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2





class TD3Agent:
# state_dim, action_dim, max_action are environment parameters
# gamma: Discount factor(Bellman equation) Default: 0.99
# tau: Interpolation factor for soft target network updates (Polyak averaging). Default: 0.005
# actor_lr: Learning rate for the actor ,default: 3e-4
# critic_lr: Learning rate for the critic ,default: 3e-4
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, actor_lr=3e-4, critic_lr=3e-4):
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
# load_state_dict: load the model's parameters from its state dictionary
        #print(self.actor.state_dict())        
# state_dict(): returns a dictionary containing the model's entire state
        #print(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
            #    for params in self.actor.parameters():
    #        print(params)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = []
        self.buffer_capacity = 2000000  
        self.batch_size = 128  
        self.policy_delay = 2
        self.policy_update_step = 0
        self.exploration_noise = 0.3  

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
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

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        next_actions = self.actor_target(next_states).clamp(-self.max_action, self.max_action)
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





env = gym.make(
    "Hopper-v5",
    render_mode="human",
    forward_reward_weight=10.0,
    ctrl_cost_weight=0.005,
    healthy_reward=2.0,
    terminate_when_unhealthy= False,
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
agent = TD3Agent(state_dim, action_dim, max_action)


# ---------------------------- Training Phase ------------------------------------ #
num_episodes = 5000
save_interval = 2500

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
        torch.save(agent.actor.state_dict(), f"td3_actor_{ep+1}.pth")
        


print("\n=== Testing ===")
agent.actor.load_state_dict(torch.load(r"C:\Users\saatw\td3_actor_5000.pth"))
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
