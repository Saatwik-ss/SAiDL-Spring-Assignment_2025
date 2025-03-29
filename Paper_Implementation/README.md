# Mastering the game of Go without human knowledge

This project, will try to implement a simple and small MCTS as suggested by [this paper](https://www.nature.com/articles/nature24270) by Google Deepmind.
Initially wanted to attempt Attention is all you need but changed it later.

Developed interest in the topic after seeing the documentary about [alpha go](https://www.youtube.com/watch?v=WXuK6gekU1Y&t=4568s&ab_channel=GoogleDeepMind)

## Paper review- 
Alpha Zero, by DeepMind, was focused on the idea of creating an artificial intelligence model which did not require any input from external or human data. Unlike its predecessor, AlphaGo, which learned from human expert games, AlphaGo Zero learned purely through self-play using Monte Carlo Tree Search (MCTS) and a deep neural network. It employed a single neural network for both policy and value estimation, updated through reinforcement learning with no external supervision and only knowledge about the rules of the game.

The algorithm relied solely on reinforcement learning through self-play, where a neural network predicted its own moves and game outcomes. This iterative process strengthened its tree search capabilities, leading to increasingly better move selection.

- Self-Play: AlphaGo Zero plays games against itself initially making random moves.
- **Neural Network Training: During these self-play games, a single deep neural network is trained. This network has two main objectives**:
    - Predicting move selections: The network learns to predict which move AlphaGo itself will make in a given position.
    - Predicting the winner: The network also learns to predict the outcome of the game (who will win) from any given position.
- Improving Tree Search: The trained neural network enhances the Monte Carlo Tree Search (MCTS) used by AlphaGo Zero The neural network guides the search process by evaluating positions and informing move selection.

- Iterative Improvement: The improved tree search, guided by the more accurate neural network, leads to higher quality move selection during subsequent self-play games. These stronger self-play games then provide new training data for the neural network, allowing it to further refine its predictions of moves and game outcomes.

- Becoming its Own Teacher: In essence, AlphaGo becomes its own teacher through this iterative process. Each iteration of self-play and neural network training builds upon the previous one, gradually improving the program's playing strength.


## Methodology-
uses a deep neural network $f(θ)$ with parameters $θ$. This neural network takes as input the raw board representation $s$ of the position and its history, and outputs both move probabilities and a value: $(p, v) = f_{\theta}(s)$.  
The vector of move probabilities $p$ represents the probability of selecting each move $a$, $p_a = Pr(a \mid s)$.
The neural network consists of many residual blocks of convolutional layers with batch normalization and rectifier nonlinearities.
The neural network was also aided by additional exploratio achieved by adding Dirichlet noise to the root node.


The whole working of alpha go zero can be attributed to two(or three) components, namely a single deep neural network which works as both policy head and value head along with a MCTS algorithm.

### Architecture:
The network takes the board state as input.
- It passes through multiple residual blocks and before the neural network gives out any meaningful result gets trained on thousands of random rollouts.

The output branches into:
- Policy Head.(Gives out a policy to the ai model based on calculating which node will give the maximum return for an $n$ number of future moves)
![Screenshot 2025-03-23 015700](https://github.com/user-attachments/assets/1e4c2d33-e5eb-4034-a0c1-93a912789f1d)
Through the policy network the MCTS algorithm avoids having to go through each node possible and the available moves are reduced by a great margin.
- Value Head (single scalar between -1 and 1).
![Screenshot 2025-03-23 015858](https://github.com/user-attachments/assets/959ca6b4-b530-49f4-8b86-23a2670bf6d9)
Helps the model avoid going right to the end of the tree search by reducing the number of nodes it has to travel through by introducing a confidence cutoff where if the neural network is confident enough about a particular branch, it'll get registered as a win(or loss). This helps the model avoid having to calculate each branch right to the end.

At a higher level the function of the neural network is just to calculate how good a particular state is for the AI player and how good or bad would a certain move would be.

#### **Monte Carlo Tree Search algorithm:**
![Screenshot 2025-03-23 015911](https://github.com/user-attachments/assets/96fc58d1-3467-4583-bc8f-59a2fae72508)

MCTS selects the best move based on the **Upper Confidence Bound (UCB)** or **PUCT** formula.

 **1. UCB Formula (Classic MCTS):**

 
$UCB(s, a) = \frac{W(s, a)}{N(s, a)} + c \sqrt{\frac{\ln (P)}{N(s, a)}}$
where:
- $W(s, a)$ = Total reward from action $a$.
- $N(s, a)$ = Number of times action $a$ has been taken.
- $P$ = Total number of visits to the parent node.
- $c$ = Exploration constant.

**2. PUCT Formula (Used in AlphaGo Zero)**

$U(s, a) = Q(s, a) + c P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$
where:
- $Q(s, a)$ = Estimated value of move $a$ (average past game outcomes).
- $P(s, a)$ = Prior probability of move $a$, predicted by the neural network.
- $N(s, a)$ = Visit count for action $a$.
- $sum_b N(s, b)$ = Total visits to all child nodes.
- $c$  = Exploration constant.

For each node iteratively, the node with the highest PUCT formula is chosen till it hits the final node. This formula includes scope for both exploration and exploitation.

Surprisingly this is all that was required to create the highest performing model created by humans and it took me by surprise on how easy the whole paper was to understand considering how impressive the result was.

---
# **My Implementation**
