# **TD3 Implementation for Hopper-v4 (MuJoCo)**  

## **Overview**  
This repository contains an implementation of **Twin Delayed Deep Deterministic Policy Gradient (TD3)** applied to the **Hopper-v4** environment in MuJoCo. The goal of this project is to train an agent to efficiently jump rightwards while maintaining balance

Having previously worked with **Deep Double Q-Networks (DDQN)** in an **ERC project**, I wanted to extend my experience in reinforcement learning by reimplementing my prior knowledge in RL models to **TD3 model**

## **Learning stage**  
- Started with Saidl suggested resources like[Open-AI](https://spinningup.openai.com/en/latest/index.html) and **Reinforcement Learning: An Introduction by Sutton and Barto**.
- Maybe not fully comfortable with the more finer details, i started with the working.

### **Implementing TD3 from Scratch**  
Initially i was having troubles whit downloading MuJoco from the site or the pip commands(as suggested in open-ai spinning up) so i decided to use the gymnasium environment to work with and make the hopper.
Rather than relying entirely on pre-built libraries, I attempted to **implement the entire TD3 architecture from scratch** to gain more independence and flexibility and the ability to plug and play different features as asked in the assignment question. This approach allowed me to:  
 **Modify individual components** such as the actor-critic networks, noise models, and training loops.    
 **Easily plug and play parts** like different activation functions, optimizers, and loss functions without constraints from pre-packaged implementations.  

By implementing TD3 from the ground up, I was able to develop a deeper understanding of reinforcement learning dynamics while maintaining the flexibility to experiment with **custom exploration strategies, loss functions, and network structures**.
However to my dissappointment, the model was very unstable and the rewards were oscillating to very high degrees very fast which in turn led to some funny movements from the agent.

---

# **Implementation Steps**  
# TD3 Implementation for Hopper-v4

The implementation has tried to include features as asked in the assignment question
- **Twin Q-Networks** (TD3's double critic for reduced overestimation bias)
- **Delayed Policy Updates** (Actor updates less frequently than Critics)
- **Prioritized Experience Replay (PER)** (Better sample efficiency)
- **n-Step Returns** (Stabilizing Q-value estimation)
- **Independent Gaussian Noise**
-  Custom reward function(achieved better results than default)
- Soft Target Updates (Polyak averaging)
- **Residual Network Architecture for Actor-Critic**
- [Download TD3 Actor Model](saved_models/td3_actor_1000.pth)
- [Download TD3 Critic Model](saved_models/td3_critic_1000.pth)
- Had troubles with implementing the .pth file for visualization and testing but it ran later.
- The hopper landed some jumps but leaned a lot more than its limit, but later stabilized after few runs
- Completed the terrain few episodes later after using the [TD3_Actor_1000](saved_models/td3_actor_1000.pth)

---
# TD3 Agent for Hopper-v4 (Stable-Baselines3)

This script trains a **TD3 agent** using **Stable-Baselines3** on the **Hopper-v4** environment.

##  Features
- Uses **TD3 with MLP policy**.
- Adds **Gaussian exploration noise** (`NormalActionNoise`).
- Similar as the scratch model.
- Trains for **500,000 timesteps**.
