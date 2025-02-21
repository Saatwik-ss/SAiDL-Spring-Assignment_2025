import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

# Create the Hopper environment
env = gym.make("Hopper-v4")

# Define action noise for exploration
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Define the TD3 model
model = TD3(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    tensorboard_log="./td3_hopper_tensorboard/",
)

# Train the model for 500,000 timesteps
model.learn(total_timesteps=500000)

# Save the trained model
model.save("Pre_hopper")

# Close environment
env.close()
