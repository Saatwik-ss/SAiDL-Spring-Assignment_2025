import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import matplotlib.pyplot as plt
# ---------------------------- Environment Setup -------------------------------- #
train_env = gym.make(
    "Hopper-v5",
    render_mode=None,
    forward_reward_weight=10.0,
    ctrl_cost_weight=0.005,
    healthy_reward=2.0,
    terminate_when_unhealthy=True,
)
n_actions = train_env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# ---------------------------- Model Setup ---------------------------------------- #
model = TD3(
    "MlpPolicy",
    train_env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=1000000,
    learning_starts=25000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=(1, "episode"),
    gradient_steps=1,
    action_noise=action_noise,
    policy_delay=2,
    device="cuda" if torch.cuda.is_available() else "cpu",
    tensorboard_log="./td3_hopper_tensorboard/",
)
# ---------------------------- Training Phase ------------------------------------- #
total_timesteps = 500000
model.learn(total_timesteps=total_timesteps)
model.save("Pre_hopper_i10")
train_env.close()
# ---------------------------- Testing Phase -------------------------------------- #
model = TD3.load(r"C:\Users\saatw\Pre_hopper_i10.zip")
# Create testing environment (with rendering)
test_env = gym.make(
    "Hopper-v5",
    render_mode="human",
    forward_reward_weight=10.0,
    ctrl_cost_weight=0.005,
    healthy_reward=2.0,
    terminate_when_unhealthy=False,
)
num_test_episodes = 10
all_rewards = []
for episode in range(num_test_episodes):
    obs, _ = test_env.reset()
    done = False
    ep_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
        ep_reward += reward
    all_rewards.append(ep_reward)
    print(f"Test Episode {episode+1}: Reward = {ep_reward:.2f}")
avg_reward = np.mean(all_rewards)
print(f"\nAverage Reward over {num_test_episodes} test episodes: {avg_reward:.2f}")
test_env.close()
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_test_episodes + 1), all_rewards, marker='o', color='g')
plt.xlabel("Test Episode")
plt.ylabel("Reward")
plt.title("TD3 Test Rewards")
plt.grid()
plt.savefig("test_rewards.png")
plt.show()
