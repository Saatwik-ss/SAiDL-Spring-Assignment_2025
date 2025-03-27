import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F




class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 16) 
        self.fc2 =nn.Linear(16,32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return F.softmax(x)   # Action probabilities


def compute_returns(rewards, gamma=0.99):
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)

def train(env_name="CartPole-v1", lr=0.01, gamma=0.99, episodes=500):
    env = gym.make(env_name,render_mode ="human")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNet(obs_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        log_probs, rewards = [], []
        done = False

        while not done:
            action_probs = policy(state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()  # Sample action

            log_probs.append(action_dist.log_prob(action))  # Store log probability
            state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)

            state = torch.tensor(state, dtype=torch.float32)
            done = terminated or truncated

        returns = compute_returns(rewards, gamma)
        loss = -torch.sum(torch.stack(log_probs) * returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards)}")

    env.close()
train()
