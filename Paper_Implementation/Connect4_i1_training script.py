import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
    """Checks if the top of the column is empty."""
    return board[0, col] == 0


def num_valid(board, col):
    """Counts the number of empty spaces in a column."""
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
    """Returns the number of columns with at least one empty space."""
    valid_moves = [num_valid(board, col) for col in range(COLUMN_COUNT)]
    valid_moves = [i for i in valid_moves if i >0]
    return len(valid_moves)

def valid_column(board):
    """Returns the columns with at least one empty space.""" 
    valid_cols = [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]
    return valid_cols    

def simulate_game():
    """
    Simulate one game with random moves.
    Returns:
      states: list of torch tensors representing board states (shape: (6,7))
      outcome: final game outcome (1 for player1 win, -1 for player2 win, 0 for draw)
    """
    board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)
    states = []  # store board snapshots
    turn = 0  # 0 for player1 (piece = 1), 1 for player2 (piece = -1)
    game_over = False
    move_count = 0
    max_moves = ROW_COUNT * COLUMN_COUNT

    while not game_over and move_count < max_moves:
        valid_cols = [c for c in range(COLUMN_COUNT) if is_valid_location(board, c)]
        if not valid_cols:
            break  # draw

        col = random.choice(valid_cols)
        row = get_next_open_row(board, col)
        if row is None:
            continue 

        piece = 1 if turn == 0 else -1
        board[row, col] = piece
        # Save a copy of the board state as a torch tensor (float type for training)
        states.append(torch.tensor(board.copy(), dtype=torch.float32))
        move_count += 1

        if winning_move(board, piece):
            outcome = piece  # 1 for player1 win, -1 for player2 win
            game_over = True
            break

        turn = (turn + 1) % 2
    if not game_over:
        outcome = 0  # draw
    return states, outcome

def simulate_move(board):
    """
    Simulates all possible moves for the current board and returns the resulting states.
    Args:
        board (np.array): The current board state.
    Returns:
        List[torch.Tensor]: A list of possible board states after making one move.
    """
    states = []
    valid_cols = valid_column(board)
    for col in valid_cols:
        row = get_next_open_row(board, col)
        if row is not None:
            temp_board = board.clone.detach()
            temp_board[row, col] = 1
            states.append(torch.tensor(temp_board, dtype=torch.float32)) 
    
    return states
        


class ConnectFourCNN(nn.Module):
    def __init__(self):
        super(ConnectFourCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * ROW_COUNT * COLUMN_COUNT, 128)
        self.fc2 = nn.Linear(128, 1)  # output predicting game outcome (logit)
    
    def forward(self, x):
        x = x.unsqueeze(1) 
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Hyperparameters
num_games = 10000 
learning_rate = 0.009
num_epochs = 300

print("Simulating games")
board_states = []
outcomes = []
for _ in range(num_games):
    states, outcome = simulate_game()

    board_states.extend(states)
    outcomes.extend([outcome] * len(states))

X = torch.stack(board_states).to(device)  # shape: (num_samples, 6, 7)
y = torch.tensor(outcomes, dtype=torch.float32).view(-1, 1)  # shape: (num_samples, 1)

# Convert outcomes for BCE loss: (-1,0,1) -> (0,0.5,1) 
y = (y + 1) / 2
y = y.to(device)


model = ConnectFourCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs,y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'connect_four_model_300epochs.pt')
