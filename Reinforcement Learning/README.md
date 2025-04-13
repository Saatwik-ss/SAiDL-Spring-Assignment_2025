# **TD3 Implementation for Hopper-v4 (MuJoCo)**  

## **Overview**  
This repository contains an implementation of **Twin Delayed Deep Deterministic Policy Gradient (TD3)** applied to the **Hopper-v4** environment in MuJoCo. The goal of this project is to train an agent to efficiently jump rightwards while maintaining balance

Having previously worked with Deep Double Q-Networks (DDQN)in an ERC project, I wanted to extend my experience in reinforcement learning by reimplementing my prior knowledge in RL models to TD3 model

(Use checkpoints in .pth format to  test on their respective models without the training loop to test the hopper)

### Learning stage  
- Started with Saidl suggested resources like [Open-AI](https://spinningup.openai.com/en/latest/index.html) and **Reinforcement Learning: An Introduction by Sutton and Barto**.
- Watched some youtube videos about RL mostly from the youtube channel https://www.youtube.com/@jekyllstein
- [Gibberblot](https://gibberblot.github.io/)
- Hands-On_Machine_Learning_with_Scikit-Learn-Keras-and-TensorFlow-2nd-Edition-Aurelien-Geron
- Attatched some files which i used to explore the environment and apply some other vanilla models on the hopper in [Archive](Archive) including some torch implementations of stuff told in spinning up.

---
  
### Implementing TD3  
Attempted to implement the a basic TD3 architecture from scratch to gain more independence and flexibility and the ability to plug and play different features as asked in the assignment question.

Attempted different iterations of TD3 with different actor-critics and training-testing loops and used the final one.

different activation functions, optimizers, were tested and used and hyperparameters which looked best were used.  

- Initial iterations were a bit unstable and the rewards were random but better tweaking helped make a more stable model.
- Tried searching for implementation example with but couldn't find anything substantial.
- Read upon the papers of TD3 implementation.
- Used the same Actor and Critic models, took help of LLMs for the TD3 agent part of the code, applied training loop and tested the model after 500 training steps, the agent returned few jumps indicating that it was learning well at [this stage](initial_jump.mp4) .
---

# Implementation Steps 
## TD3 Implementation for Hopper-v4

The implementation has tried to include features as asked in the assignment question
- Twin Q-Networks (TD3's double critic for reduced overestimation bias)
- Delayed Policy Updates (Actor updates less frequently than Critics)
- Prioritized Experience Replay (PER) (Better sample efficiency).
- n-Step Returns (Stabilizing Q-value estimation).
- Random noise added for exploration.
- Soft Target Updates (Polyak averaging).
- [Download TD3 Actor Model](td3_actor_5000.pth)
- The hopper landed some jumps but leaned a lot more than its limit, but later stabilized after few runs
- Hopper started completing the terrain few times somewhere around 2500 episodes.
- Was able to complete the terrain 7/10 times after 5000 episodes as seen [here](Results/td3_i2_run_5000.mp4).
- This step as it went, took the most time for completion along with implementing noisy layers but helped create a framework which was resued for all later iterations of the hopper.
- ![td3_rewards (5)](https://github.com/user-attachments/assets/5a920120-c1e7-473b-afd9-763f1a78ef30)



## TD3- working

The goal was to make the Q value algorithem learn a deterministic policy 
$\pi_\theta: \mathbb{R}^n \to \mathbb{R}^m$ 
that maximizes the expected return:

$$
J(\pi) = \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t\. r(s_t, a_t)\right], \quad \gamma = 0.99.
$$

**Critic Loss**: computes the target Q-value using the target actor and target critics.

$$
\mathcal{L}(\phi_i) = \mathbb{E} \left[ \left( Q_{\phi_i}(s, a) - \left( r + \gamma(1 - d) \min_{j=1,2} Q_{\phi_j^{\text{target}}}(s', \pi_{\theta^{\text{target}}}(s')) \right) \right)^2 \right]
$$


**Actor Loss**: The actor is updated to maximize the Q-value from the first critic:

$$
\mathcal{L}(\theta) = -\mathbb{E} \left[ Q_{\phi_1}(s, \pi_{\theta}(s)) \right]
$$


**Target Networks**: Updated via Polyak averaging:

$$
\phi_j^{\text{target}} \gets \tau \phi_j + (1 - \tau) \phi_j^{\text{target}}
$$

$$
\theta^{\text{target}} \gets \tau \theta + (1 - \tau) \theta^{\text{target}}
$$

**Exploration Noise**: Gaussian noise is added to actions during selection, which is clipped and decays over time:

$$
a_{\text{noisy}} = \text{clip}(\pi_{\theta}(s) + \epsilon, -a_{\max}, a_{\max}), \quad \epsilon \sim \mathcal{N}(0, \sigma)
$$

---

# TD3 Agent for Hopper-v4 (Stable-Baselines3)

Also tried using Stable Baseline MLP Policy model to compare the two models and from a genral eye test, the custom model was working better for similar number of episodes(5000 each) but Stable Baseline model trained at a faster rate and the model was able to hop the terrain completely after 500000 episodes as seen [here](https://github.com/Saatwik-ss/SAiDL-Spring-Assignment_2025/blob/main/Reinforcement%20Learning/Results/TD3_initial_run.mp4) with these [rewards]([Results/Stable_Baseline_rewards_500000.png](https://github.com/Saatwik-ss/SAiDL-Spring-Assignment_2025/blob/main/Reinforcement%20Learning/Results/Stable_Baseline_rewards_500000.png)).

[Stable_Baseline_Model](Stable_Baseline_model.py)

##  Features
- Uses TD3 with MLP policy
- Adds Gaussian exploration noise
- Similar reward funct. as the scratch model
- Trains for 500,000 timesteps
---

# Implementing noisy layers
- Used [this repo](https://github.com/thomashirtz/noisy-networks/tree/main) to get a basic understanding on the implementation of noisy networks.

## NoisyNets: Parametric Noise in Neural Networks  

In RL, agents learn by trying different actions and seeing the rewards they get. A crucial part of this is exploration, where the agent tries out new things to discover better strategies, rather than just sticking to what seems good so far. Traditional methods like ε-greedy (randomly choosing an action with some probability ε) often use simple, local randomness. The authors argue that these methods might not be enough to find complex behaviours needed for efficient exploration in many situation, to tackle this, Noisy networks are introduced in this [paper](https://arxiv.org/pdf/1706.10295) by DeepMind.


NoisyNets add learnable noise directly to the weights of the neural network that the agent uses to make decisions


NoisyNets are neural networks whose weights and biases are perturbed by a parametric function of the noise. These parameters are adapted with gradient descent. Given a neural network $\( y = f_{\theta}(x) \)$ with noisy parameters $\( \theta \)$, we define:  

$$
\theta = \mu + \Sigma \odot \varepsilon
$$

where $\( \mu, \Sigma \)$ are learnable parameters, $\( \varepsilon \)$ is zero-mean noise, and $\( \odot \)$ denotes element-wise multiplication. The loss is optimized over the expectation:  

$$
\bar{L}(\zeta) = \mathbb{E} [L(\theta)]
$$

For a linear layer with input $\( x \in \mathbb{R}^p \)$, weight $\( w \in \mathbb{R}^{q \times p} \)$, and bias $\( b \in \mathbb{R}^q \)$, the noisy version is:  

$$
y = (\mu_w + \sigma_w \odot \varepsilon_w)x + \mu_b + \sigma_b \odot \varepsilon_b
$$

where $\( \mu_w, \mu_b, \sigma_w, \sigma_b \)$ are learnable, and $\( \varepsilon_w, \varepsilon_b \)$ are random noise variables.  

### Noise injection:

Code for NoisyNets remains similar to that of TD3 with the addition of a noise class, for example,

```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action
```

becomes

```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        nn.Module.__init__(self)
        self.noisy1 = NoisyLinear(state_dim, 256)
        self.noisy2 = NoisyLinear(256, 256)
        self.noisy3 = NoisyLinear(256, action_dim)
        self.max_action = max_action
```
### Methods of injection-
- Independent Gaussian noise:

The noise applied to each weight and bias is independent, thus each layer has different-independent noise applied leading to $\( p + q \)$ noise variables (for p inputs to the layer and q outputs).


- Factorized Gaussian noise:
By factorising, instead of adding independent noise to each weight, a single randomly generated oise funct. is used to generate noise per input and output neuron.
As explained in the paper, $\( \epsilon^w_{i,j} \)$ is factorized to $\( \epsilon^w_i \)$ and $\( \epsilon^w_j \)$ for input and output respectively(also for biass noise)

Each $\( \epsilon^w_{i,j} \)$ and $\( \epsilon^b_j \)$  
can then be written as:  

$$
\epsilon^w_{i,j} = f(\epsilon_i) f(\epsilon_j)
$$

$$
\epsilon^b_j = f(\epsilon_j)
$$

where \( f \) is a real-valued function. In the experiments, \( f \) is used as

$$
f(x) = \text{sgn}(x) \sqrt{|x|}
$$

- Monte Carlo method is employed to approximate the gradient of the loss function

### Independent Noise:

- Initially started with implementing Independent noise as it appeared more easier to understand and get feel of.
- Initial iterations were again bit random in terms of rewards and time taken to train and the hopper couldn't complete the jumps.
- Also started factorized noise part while doing this.
- Introduces independent Gaussian noise into actor params. Unlike traditional action noise, it perturbs weights and biases directly. trainable noise parameters 
 include sigma_weight, sigma_bias.

---

### Factorized Noise: 

- Factorized noise reduces the number of noise parameters while maintaining exploration benefits.
- Applyies independent Gaussian noise to each weight and bias, factorized noise samples two lower-dimensional noise vectors, one for input and one for output. These are then combined multiplicatively to form a full noise matrix as described above.
- Applied Guassian, Gamma and Beta noises as factorized noises, to be noted that the different noise layers differ only in the distribution of noise especialli in the (_scale_noise) functions.

  ### Gaussian noise:
![image](https://github.com/user-attachments/assets/8b2fdbbd-1647-4245-9cf2-7a1137ceacc0)



### Beta Noise
![Screenshot 2025-03-24 021612](https://github.com/user-attachments/assets/13c982c3-9f09-453c-a0db-4149c27ffe26)


### Gamma Noise:
![Screenshot 2025-03-24 021143](https://github.com/user-attachments/assets/a23a773c-bdc8-486a-89b0-26f7dce38c6f)

---
## Training rewards-
For about 2800 episodes, the training rewards of Gaussian easliy overtook the vanilla noise TD3 thus succesfully reimplementing the desired results of the paper. The style of graph used isn't as appropriate enough as the line graph but the different in rewards is clear and visible enough.



![rewards_2800](https://github.com/user-attachments/assets/0788c193-c2f3-46c5-bb9c-c1737a8946db)

## 2-b:) How would you test the adversarial robustness of your setup? Would the performance be affected by different encoder blocks (CNNs, Attention Heads) 

- Adversial robustness will be tested by using adversiarial attacks during the testing phase of the hopper. FGSM is a common method to act out such adversarial attacks.
- CNN policy was implemented by creating an episode by episode changing 3-D matrix which consisted of state, rewards and actions,this was done since CNNs are used for image processing and higher dim. data, using CNNs for a mujoco env. did not appear to be ideal for this task. Mujoco env. instead use low dimensional(1D or 2D) data but they use a lot of them for action spaces and state spaces and rewards, so combinging these low dimensional data and mapping each state to its state, action and sunbequent reward can be done and i can hoped that the CNN can map out some sort of pattern in the data and learn from it. 3D matrix like an RGB array was created for this task. However state and actions are of different sizes so to create proper order lots of mutations and paddings were done and the reward could not cross 50 hence the implementation is not attached.
- How can a transformer be applied to this task was a thoughtful question since transformers are used for NLP tasks and other feature of transformers like positional embeddings which is redundant anyways for this case and multi head attention did not appear ideal.
- A Vit and CNN architecture which captured the image of hopper in human render_mode can  be used and it might be embedded with a reward function and it can learn how to distinguish between healthy and unhealthy states can be used and might achieve good results however i couldn;t get the opportunity to try it out as well could not find a base to start with .
- RGB_render mode exists but could not find how to use it for this task hence ending this portion of the assignment for me.
---

# **BONUS** : With BYOL
