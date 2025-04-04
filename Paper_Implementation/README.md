# Mastering the game of Go without human knowledge


This project, will try to implement a simple and small MCTS as suggested by [this paper](https://www.nature.com/articles/nature24270) by Google Deepmind.
Initially wanted to attempt 'Attention is all you need' but changed it later.

Developed interest in the topic after seeing the documentary about [Alpha Go](https://www.youtube.com/watch?v=WXuK6gekU1Y&t=4568s&ab_channel=GoogleDeepMind)

# **AlphaGo Zero**
## Paper review- 
AlphaGo Zero, by DeepMind, focused on the idea of creating an artificial intelligence model which did not require any input from external or human data. Unlike its predecessor, AlphaGo, which learned from human expert games, AlphaGo Zero learned purely through self-play using Monte Carlo Tree Search (MCTS) and a deep neural network. It employed a single neural network for both policy and value estimation, updated through reinforcement learning with no external supervision and only knowledge about the rules of the game.

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
While understanding the paper was rather easy trying to write the code was rather a challenge especially so since the implementations available were a bit beyond my understanding in a way that i could understand what each function or bit of code did but had trouble understanding how they all connected and worked together.

Initially i planned to implement the game in the traditional go environment so i also learnt the rules of them game however the game of go also not only needed the current state but also the previous states of the board to play so i dropped the idea as i felt that the game environment was of second priority. Hence i decided to work with chess. However due to low time i had to settle with connect4.

This project was extra special since i spent more time working out how i'll implement things on a notebook rather than coding which felt better.

## TicTacToe 

Since Connect4 was very similar to TicTacToe i planned to start with tic tac toe to learn and test my methods. I had experience working with pygame but i had forgotten a lot(also the pygame website is blocked on bits wifi so thats something) so i took quite some help from LLMs for game environment part of this project.

I initially created a TicTacToe environment where the initial board state consisted of a 3 **x** 3 matrix of all zeros
Moves of each player were either marked as 1 or -1 and appended to the matrix which was fed to the Convoluted Neural Network. The CNN was in fact trained on the random games the computer played with itself and let to either a +1/0/-1 outcome and hence the more games it played the better the CNN got at prediciting a good move.
The job of this neural network was to determine the probability of the opposition(Human) player to win. Hence the job of the one node tree search(if it can be called that) was to choose a move which reduced the probaility of human to win.

### Rundown
- Each player's move is marked as 1 (AI) or -1 (Human) and updated in the matrix.

- The updated board state is then fed into a Convolutional Neural Network (CNN).
    - +1 (AI win), 0 (Draw), or -1 (AI loss to human).
    - Over time, as more games are played, the CNN improves at predicting the game outcome given a board state.
    - The model gives out logits from -1 to 1 where loss function is BCEloss with logits.
      
- The CNN’s goal is to predict the probability of the human (opponent) winning from the given state.
  
- The AI uses a one-step lookahead (a "one-node tree search"), selecting the move that minimizes the opponent’s probability of winning.

The way it is different from the Alpha Go implementation is in the following 4ways-

- 1) MCTS Depth- It does not go as deep as Alpha go did it only goes for one step ahead lookup.
- 2)  The CNN has no control on the move the game simulator playes, it will be easy to implement but its not there in the current iteration.
- 3)  There is no noise implemented and unlike Alpha go which had noise both in trainig and evaluation(playing games). Which will not be implemented in later versions as well due to the simplicity of the games involved.
- 4)  CNN model- CNN will be good for this purpose since its  a 2D matrix and CNNs are good at feature extraction however Alpha Go used  ResNet architecture model.
 
## Connect4
Once TicTacToe was done it was not much difficult to implemet the same on a connect 4 model, similar structure was used with the same ideas just different game rules.

Also used a simple rule in Connect4 which gave higher probability to moves which connected the AI pieces or disconnect opponent pieces and the model trained on it played better than the vanilla no rule implementation but I later dropped it since i thought it goes against the spirit of the paper where Zero literally means Zero human input.

Trained the model to 2000 epochs for 10,000 games in each epoch and the loss came down to about 0.35(starting from 0.69) at 1200 epochs and then plateued initially and then started overfitting for the recent games, thus reaching the best performance it could on my setup.

It understood some good opening tactics and did inital defence alright since those were the moves it saw the most but as the game went on its performance dipped and many of the best moves were second in its priority so it bludered many times away by either not connecting the 4th piece after connecting 3 of each or not being able to disconnect the opponent when they have connected 3. It was mainly because those games were seen less by the model.

### Using actual MCTS in Connect4
Started by using the pre trained C4_1200 epoch model to start the training rather than starting from scratch.

To get a good understanding of MCTS, instead of creating an MCTS to train, i planned on using mcts to counter the one step lookahead feature of my current architecture.


