import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm # Import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROW_COUNT = 6
COLUMN_COUNT = 7

def winning_move(board, piece):
    # Horizontal check
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            if board[r, c] == piece and board[r, c+1] == piece and board[r, c+2] == piece and board[r, c+3] == piece:
                return True
    # Vertical check
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r, c] == piece and board[r+1, c] == piece and board[r+2, c] == piece and board[r+3, c] == piece:
                return True
    # Positive diagonal check
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            if board[r, c] == piece and board[r+1, c+1] == piece and board[r+2, c+2] == piece and board[r+3, c+3] == piece:
                return True
    # Negative diagonal check
    for r in range(3, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            if board[r, c] == piece and board[r-1, c+1] == piece and board[r-2, c+2] == piece and board[r-3, c+3] == piece:
                return True
    return False



def is_valid_location(board, col):
    return board[0, col] == 0


def num_valid(board, col):
    j = 0
    for i in range(ROW_COUNT):
        if board[i, col] == 0:
            j= j+1
    return j

def get_next_open_row(board, col):
    for r in range(ROW_COUNT-1, -1, -1):
        if board[r, col] == 0:
            return r
    return None

def num_valid_moves(board):
    valid_moves = [num_valid(board, col) for col in range(COLUMN_COUNT)]
    valid_moves = [i for i in valid_moves if i >0]
    return len(valid_moves)

def valid_column(board):
    valid_cols = [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]
    return valid_cols
    


def simulate_game():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)
    states = []  # store board snapshots
    turn = 0  # 0 for player1 (piece = 1), 1 for player2 (piece = -1)
    game_over = False
    move_count = 0
    max_moves = ROW_COUNT * COLUMN_COUNT

    while not game_over and move_count < max_moves:
        valid_cols = [c for c in range(COLUMN_COUNT) if is_valid_location(board, c)]
        if not valid_cols:
            break  

        col = random.choice(valid_cols)
        row = get_next_open_row(board, col)
        if row is None:
            continue 

        piece = 1 if turn == 0 else -1
        board[row, col] = piece
        states.append(torch.tensor(board.copy(), dtype=torch.float32))
        move_count += 1

        if winning_move(board, piece):
            outcome = piece
            game_over = True
            break

        turn = (turn + 1) % 2
    if not game_over:
        outcome = 0  # draw
    return states, outcome


def simulate_move(board, piece):
    states = []
    valid_cols = valid_column(board)
    for col in valid_cols:
        row = get_next_open_row(board, col)
        if row is not None:
            temp_board = board.clone().detach()
            temp_board[row, col] = piece
            states.append(states.append(temp_board.clone().detach())) 
    
    return states


def is_terminal_node(board):
    return winning_move(board, 1) or winning_move(board, -1) or num_valid_moves(board) == 0


def is_fully_expanded(self):
        return len(self.children) == len(self.valid_actions)
    

# Masking Function
def invalid_moves(logits: torch.Tensor, board_tensor: torch.Tensor) -> torch.Tensor:
    board_tensor = board_tensor.to(logits.device)
    masked_logits = logits.clone()
    batch_size = logits.shape[0]

    for b in range(batch_size):
        for col in range(COLUMN_COUNT):
            if board_tensor[b, 0, 0, col] != 0:
                masked_logits[b, col] = -float('inf')
    return masked_logits





class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, input_channels=1, num_blocks=20, num_actions=COLUMN_COUNT):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.res_blocks = nn.ModuleList([ResidualBlock(64, 64) for _ in range(num_blocks)])

        # Policy Head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * ROW_COUNT * COLUMN_COUNT, num_actions)

        # Value Head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * ROW_COUNT * COLUMN_COUNT, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x_res = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x_res = block(x_res)

        # Policy Head: Move Probabilities
        policy_logits = self.policy_conv(x_res)
        policy_logits = F.relu(self.policy_bn(policy_logits))
        policy_logits = policy_logits.view(policy_logits.size(0), -1)
        policy_logits = self.policy_fc(policy_logits)
        masked_policy_logits = invalid_moves(policy_logits, x)
        policy = F.softmax(masked_policy_logits, dim=-1)


        # Value Head: Win Probability Estimate
        value = self.value_conv(x_res)
        value = F.relu(self.value_bn(value))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)) # Output between -1 and 1
        return policy, value





class MCTNode():
    def __init__(self, board, parent=None, action_taken=None):
        self.board = board.clone().detach() if isinstance(board, torch.Tensor) else torch.tensor(board, dtype=torch.float32)
        self.parent = parent
        self.action_taken = action_taken
        self.children = {}  # {action: MCTNode}
        self.visit_count = 0
        self.total_value = 0
        self.valid_actions = valid_column(self.board)

    def simulate_move(self, piece):
        states = []
        for col in self.valid_actions:
            row = get_next_open_row(self.board, col)
            if row is not None:
                temp_board = self.board.clone().detach()
                temp_board[row, col] = piece
                states.append(temp_board)
        return states

    def is_fully_expanded(self):
        return len(self.children) == len(self.valid_actions)

    def expand(self, turn):
        if is_terminal_node(self.board) or self.is_fully_expanded():
            return None

        for action in self.valid_actions:
            row = get_next_open_row(self.board, action)
            if row is not None:
                new_board = self.board.clone().detach()
                new_board[row, action] = turn  # Apply move
                self.children[action] = MCTNode(new_board, parent=self, action_taken=action)

    def best_child(self, c_puct=1.41):
        if not self.children:
            return None  
        best_action, best_node = max(self.children.items(), key=lambda item: self.uct_value(item[1], c_puct))
        return best_node

    def uct_value(self, child, c_puct):
        if child.visit_count == 0:
            return 10000  
        Q = child.total_value / (child.visit_count + 1e-6)  
        U = c_puct * np.sqrt(np.log(self.visit_count + 1e-6) / (child.visit_count + 1e-6))  # Exploration term
        return Q + U


    def backpropagate(self, value):
        self.visit_count += 1
        self.total_value += value
        if self.parent:
            self.parent.backpropagate(-value)  # Flip sign to account for both
            
            

"""
#Testing functions
def random_board():
    #Generates a valid random Connect 4 board
    board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)
    for col in range(COLUMN_COUNT):
        num_pieces = np.random.randint(0, ROW_COUNT + 1)
    if num_pieces > 0:
        pieces = np.random.choice([1, -1], size=num_pieces)
        board[ROW_COUNT - num_pieces:, col] = pieces  # for filling from bottom
        return board
    
board = random_board()
print(board)
board = torch.tensor(board, dtype=torch.float32)
valid_moves = num_valid_moves(board)
valid = [num_valid(board, col) for col in range(COLUMN_COUNT)]
print(valid)
print("\nNumber of valid moves:", valid_moves)
print(f"valid columns: {valid_column(board)}")
print(simulate_move(board,1))
print(is_terminal_node(board))
board = board.unsqueeze(0).unsqueeze(0) 
model = ResNet()
with torch.no_grad():
    policy, value = model(board)
print("Policy Output (Move Probabilities):", policy.numpy().flatten())
print("Value Output (Win Probability Estimate):", value.item())

"""

BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 10
CHECKPOINT_INTERVAL = 10  

model = ResNet()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn_policy = nn.CrossEntropyLoss()
loss_fn_value = nn.MSELoss()





  
def self_play(num_games=1000):
    training_data = []
    for game in tqdm(range(num_games), desc="Self-play Games"):
        board = torch.zeros((6, 7), dtype=torch.float32)
        root = MCTNode(board)
        game_states = []
        game_policies = []
        turn = 1

        ################################# WHILE LOOP #################################
        while not is_terminal_node(board):
            node = root


            for numnodes in range(100):
                # go through the tree using best_child() repeatedly.
                while True:
                    next_node = node.best_child()
                    if next_node is None:
                        break
                    node = next_node


                expanded_node = node.expand(turn)
                if expanded_node is not None:
                    # if expansion creates a new node move to that node
                    node = expanded_node
                    board_input = node.board.unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        _, value = model(board_input)
                    node.backpropagate(value.item())
                else:
                    # if no more expansion possible
                    board_input = node.board.unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        _, value = model(board_input)
                    node.backpropagate(value.item())


            chosen_child = root.best_child()
            if chosen_child is None:
                break
            chosen_action = chosen_child.action_taken

            # Record training data
            game_states.append(root.board.clone().detach())
            game_policies.append(chosen_action)

            row = get_next_open_row(board, chosen_action)
            board[row, chosen_action] = turn
            
            
            root = root.children[chosen_action]
            # Switch turns between players.
            turn *= -1
            
        ################################# WHILE LOOP #################################
        
        outcome = 1 if winning_move(board, 1) else -1 if winning_move(board, -1) else 0
        for state, policy_target in zip(game_states, game_policies):
            training_data.append((state, policy_target, outcome))

        if (game + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save(model.state_dict(), f"checkpoint_game_{game+1}.pth")

    return training_data







def train_model(training_data):
    states, policies, values = zip(*training_data)
    states = torch.stack(states).unsqueeze(1)  # (N, 1, 6, 7)
    policies = torch.tensor(policies, dtype=torch.long) 
    values = torch.tensor(values, dtype=torch.float32).unsqueeze(1)  # (N, 1)
    dataset = torch.utils.data.TensorDataset(states, policies, values)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


    
    for epoch in range(EPOCHS):
        total_loss = 0
        # tqdm progress bar for epochs
        for batch_states, batch_policies, batch_values in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            pred_policies, pred_values = model(batch_states)
            loss_policy = loss_fn_policy(pred_policies, batch_policies)
            loss_value = loss_fn_value(pred_values, batch_values)
            loss = loss_policy + loss_value
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss}")
    torch.save(model.state_dict(), "finally.pth")

training_data = self_play(num_games=100)
train_model(training_data)
