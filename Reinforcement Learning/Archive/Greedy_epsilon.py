import gymnasium as gym
import numpy as np

#============================ Greedy Epsilon ================================#
env = gym.make(
    "Hopper-v4",
    render_mode="human",
    forward_reward_weight=10.0,
    ctrl_cost_weight=0.005,
    healthy_reward=2.0,
    terminate_when_unhealthy=True,
)
num_episodes = 50000
epsilon = 1.0  # Start with full exploration
epsilon_decay = 0.995  # Decay per episode
min_epsilon = 0.01  # Minimum epsilon
gamma = 0.99  # Discount factor for rewards
# Q-table initialization (Discrete simplification)
action_size = env.action_space.shape[0]  # Continuous action space
state_size = env.observation_space.shape[0]  # State dimension
q_table = {}  # Dictionary for storing Q-values
print(action_size)
print(state_size)
print(env.action_space)

print(env.observation_space)
print(env.observation_space.shape[0])
print(env.action_space.low)
print(env.action_space.high)
# Reward tracking
episode_rewards = []
for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        obs_tuple = tuple(obs)  # Convert observation to tuple for Q-table storage
        # Initialize Q-values for unseen states
        if obs_tuple not in q_table:
            q_table[obs_tuple] = np.random.uniform(-1, 1, action_size)
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Random action (exploration)
        else:
            action = q_table[obs_tuple]  #Greedy action (exploitation)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs_tuple = tuple(next_obs)
        done = terminated or truncated
        total_reward += reward
        # Update Q-values
        if next_obs_tuple not in q_table:
            q_table[next_obs_tuple]=np.random.uniform(-1, 1, action_size)
        q_table[obs_tuple]=q_table[obs_tuple] + gamma * (reward + np.max(q_table[next_obs_tuple]) - q_table[obs_tuple])
        obs = next_obs  
    episode_rewards.append(total_reward)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
env.close()
print(f"Average reward over {num_episodes} episodes: {np.mean(episode_rewards)}")
print(f"Final epsilon value: {epsilon}")
