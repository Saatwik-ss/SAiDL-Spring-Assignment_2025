# **TD3 Implementation for Hopper-v4 (MuJoCo)**  

## **Overview**  
This repository contains an implementation of **Twin Delayed Deep Deterministic Policy Gradient (TD3)** applied to the **Hopper-v4** environment in MuJoCo. The goal of this project is to train an agent to efficiently jump rightwards while maintaining balance

Having previously worked with **Deep Double Q-Networks (DDQN)** in an **ERC project**, I wanted to extend my experience in reinforcement learning by reimplementing my prior knowledge in RL models to **TD3 model**

## **Learning stage**  
- Started with Saidl suggested resources like [Open-AI](https://spinningup.openai.com/en/latest/index.html) and **Reinforcement Learning: An Introduction by Sutton and Barto**.
- Watched some youtube videos about RL mostly from the youtube channel https://www.youtube.com/@jekyllstein
- [Gibberblot](https://gibberblot.github.io/)
- Hands-On_Machine_Learning_with_Scikit-Learn-Keras-and-TensorFlow-2nd-Edition-Aurelien-Geron

  
### **Implementing TD3 from Scratch**  
Initially had troubles whit downloading MuJoco from the site or the pip commands(as suggested in open-ai spinning up) so used the gymnasium environment to work with and make the hopper.
Rather than relying entirely on pre-built libraries, attempted to **implement the entire TD3 architecture from scratch** to gain more independence and flexibility and the ability to plug and play different features as asked in the assignment question. This approach allowed :  
Modify individual components such as the actor-critic networks, noise models, and training loops.    
Easily plug and play parts like different activation functions, optimizers, were tested and used and hyperparameters which looked best were used.  

- By implementing TD3 from the ground up, was able to experiment with custom exploration strategies, loss functions, and network structures.
- Model intially was very unstable and the rewards were very random, took some help from LLMs but the results were still similar even after many episodes and the model didn't work as expected.
- Tried searching for implementation example with but couldn't find anything substantial.
- Tried reading the gymnasium documemtation more deeply, till here i treated it just like the **ERC PROJECT** with rewards spelled out for different action spaces.
- Used the same Actor and Critic models, took help of LLMs for the TD3 agent part of the code, applied training loop and tested the model after 500 training steps(Training took extra time because i was adamant on using human render mode), the agent returned few jumps indicating that it was learning well at [this stage](initial_jump.mp4) .
---

# **Implementation Steps**  
# TD3 Implementation for Hopper-v4

The implementation has tried to include features as asked in the assignment question
- **Twin Q-Networks** (TD3's double critic for reduced overestimation bias)
- **Delayed Policy Updates** (Actor updates less frequently than Critics)
- **Prioritized Experience Replay (PER)** (Better sample efficiency).
- **n-Step Returns** (Stabilizing Q-value estimation).
- Random noise added for exploration.
- Soft Target Updates (Polyak averaging).
- [Download TD3 Actor Model](td3_actor_5000.pth)
- The hopper landed some jumps but leaned a lot more than its limit, but later stabilized after few runs
- Hopper started completing the terrain few times somewhere around 2500 episodes.
- Was able to complete the terrain 7/10 times after 5000 episodes as seen [here](Results/td3_i2_run_5000.mp4).
- This step as it went, took the most time for completion along with implementing noisy layers but helped create a framework which was resued for all later iterations of the hopper.


### TD3- working

The goal was to make the Q value algorithem learn a deterministic policy 
$\pi_\theta: \mathbb{R}^n \to \mathbb{R}^m$ 
that maximizes the expected return:

$$
J(\pi) = \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t\, r(s_t, a_t)\right], \quad \gamma = 0.99.
$$

## Critic Networks

Two critic networks, $$Q_{\phi_1}$$ and $$Q_{\phi_2}$$, are used to mitigate overestimation bias. They are trained by minimizing the Bellman loss:

$$
\min_{\phi_i} \, \mathbb{E}\left[\left(Q_{\phi_i}(s,a) - \left(r + \gamma\, \min_{j=1,2} Q_{\phi_j'}(s',\pi_{\theta'}(s'))\right)\right)^2\right].
$$

## Target Updates

The actor and critic target networks are updated using soft updates:

$$
\theta' \leftarrow \tau\, \theta + (1-\tau)\,\theta', \quad \tau = 0.005.
$$

## Exploration

Gaussian noise is added to the actions during training to encourage exploration in the continuous action space.

---
# TD3 Agent for Hopper-v4 (Stable-Baselines3)

Also tried using Stable Baseline MLP Policy model to compare the two models and from a genral eye test, the custom model was working better for similar number of episodes(5000 each) but Stable Baseline model trained at a faster rate and the model was able to hop the terrain completely after 500000 episodes as seen [here](Results/TD3_inital_run.mp4) with these [rewards](Results/Stable_Baseline_rewards_500000.png).

[Stable_Baseline_Model](Stable_Baseline_model.py)

##  Features
- Uses **TD3 with MLP policy**
- Adds **Gaussian exploration noise**
- Similar reward funct. as the scratch model
- Trains for **500,000 timesteps**
---

# Implementing noisy layers
- Used [this repo](https://github.com/thomashirtz/noisy-networks/tree/main) to get a basic understanding on the implementation of noisy networks
