# Mastering the game of Go without human knowledge

This project, will try to implement a simple and small MCTS as suggested by [this paper](https://www.nature.com/articles/nature24270) by Google Deepmind.
Initially wanted to attemot Attention is all you need but chnaged it later.

Developed interest in the topic after seeing the documentary about [alpha go](https://www.youtube.com/watch?v=WXuK6gekU1Y&t=4568s&ab_channel=GoogleDeepMind)

## Paper review-
Alpha Zero, by DeepMind,was focuesd on the idea of creating an artificial intelligence model which did not require any input from external or human data. Unlike its predecessor, AlphaGo, which learned from human expert games, AlphaGo Zero learned purely through self-play using Monte Carlo Tree Search (MCTS) and a deep neural network. It employed a single neural network for both policy and value estimation, updated through reinforcement learning with no external supervision and only knowledge about the rules of the game.

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
The neural network consists of many residual blocks4 of convolutional layers with batch normalization and rectifier nonlinearities
