import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np


env = gym.make("Hopper-v4")
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

#TD3 model
model = TD3(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    tensorboard_log="./td3_hopper_tensorboard/",
)

# Train
model.learn(total_timesteps=500000)
model.save("Pre_hopper")
env.close()

########################## Testing the model ###############################

# Load trained model
model = TD3.load(r"C:\Users\saatw\Pre_hopper.zip")

env = gym.make("Hopper-v4", render_mode="human")

# Testing
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)

env.close()
     