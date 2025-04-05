import sys
import numpy as np
import pygame
import math
import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
move = int(input("ENTER TURN MODE(Turn=0: human moves first , Turn=1: AI moves first) : "))
if move == 0:
    move =-1
pygame.init()
row_count = 6
column_count = 7
ROW_COUNT = 6
COLUMN_COUNT = 7
square_size = 100
radius = int(square_size / 2 - 5)

blue = (0, 0, 255)
black = (0, 0, 0)
red = (255, 0, 0)       # human
yellow = (255, 255, 0)  # AI
white = (255, 255, 255)

# Board size
width = column_count * square_size + 200
height = (row_count + 1) * square_size
size = (width, height)

# Board Mapping
column_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
row_labels = ['1', '2', '3', '4', '5', '6']

def position_label(row, col):
    return column_labels[col] + row_labels[row]


###########################################################################################
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
    model_board = convert_board_for_model(board)
    model_board_tensor = torch.tensor(model_board, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logit = model(model_board_tensor)
        prob = torch.sigmoid(logit).item()
    if turn == 1:
        valid_cols = [col for col in valid_cols if prob < 0.8]
    elif turn == -1:
        valid_cols = [col for col in valid_cols if prob > 0.2]
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
###########################################################################################










# display
screen = pygame.display.set_mode(size)
pygame.display.set_caption("")
font = pygame.font.SysFont("monospace", 30)

class GameBoard:
    @staticmethod
    def create_board():
        return np.zeros((row_count, column_count))
    
    @staticmethod
    def drop_piece(board, row, col, piece):
        board[row][col] = piece
    
    @staticmethod
    def is_valid(board, col):
        return board[0][col] == 0
    
    @staticmethod
    def next(board, col):
        for r in range(row_count-1, -1, -1):
            if board[r][col] == 0:
                return r
        return None
    
    
    @staticmethod
    def win_move(board, piece):
        # Check horizontal locations for win
        for c in range(column_count - 3):
            for r in range(row_count):
                if all(board[r][c+i] == piece for i in range(4)):
                    return True
        # Check vertical locations for win
        for c in range(column_count):
            for r in range(row_count - 3):
                if all(board[r+i][c] == piece for i in range(4)):
                    return True
        # Check positively sloped diagonals
        for c in range(column_count - 3):
            for r in range(row_count - 3):
                if all(board[r+i][c+i] == piece for i in range(4)):
                    return True
        # Check negatively sloped diagonals
        for c in range(column_count - 3):
            for r in range(3, row_count):
                if all(board[r-i][c+i] == piece for i in range(4)):
                    return True
        return False
    
    @staticmethod
    def draw_board(board, move_list):
        flipped = np.flipud(board)
        screen.fill(black)
        
        # Draw board background and empty circles
        for r in range(row_count):
            for c in range(column_count):
                pygame.draw.rect(screen, blue, (c*square_size, r*square_size+square_size, square_size, square_size))
                pygame.draw.circle(screen, black, 
                                   (int(c*square_size+square_size/2), int(r*square_size+square_size+square_size/2)), 
                                   radius)
        # Draw pieces
        for r in range(row_count):
            for c in range(column_count):
                if flipped[r][c] == 1:
                    pygame.draw.circle(screen, red, 
                                       (int(c*square_size+square_size/2), height - int(r*square_size+square_size/2)), 
                                       radius)
                elif flipped[r][c] == 2:
                    pygame.draw.circle(screen, yellow, 
                                       (int(c*square_size+square_size/2), height - int(r*square_size+square_size/2)), 
                                       radius)
        # Draw move history on the right side
        pygame.draw.rect(screen, white, (column_count * square_size, 0, 200, height))
        move_font = pygame.font.SysFont("monospace", 25)
        y_offset = 50
        for i, move in enumerate(move_list):
            label = move_font.render(move, 1, red if i % 2 == 0 else black)
            screen.blit(label, (column_count * square_size + 20, y_offset))
            y_offset += 30
        
        pygame.display.update()

# Model Setup for AI 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define the CNN model
class ConnectFourCNN(nn.Module):
    def __init__(self):
        super(ConnectFourCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * row_count * column_count, 128)
        self.fc2 = nn.Linear(128, 1)  # Output is a logit for player1 win probability
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch, 6, 7)
        x = x.unsqueeze(1)  # (batch, 1, 6, 7)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = ConnectFourCNN().to(device)
model.load_state_dict(torch.load(r"C:\Users\saatw\Downloads\connect_four_epoch_1200.pt", map_location=device))
model.eval()

def convert_board_for_model(board):
    """
    Convert game board from our representation (0, 1, 2) to the model representation:
    - Human's piece (1) stays 1.
    - AI's piece (2) becomes -1.
    Accepts NumPy arrays or torch tensors.
    """
    if isinstance(board, np.ndarray):
        board = torch.from_numpy(board).float()
    else:
        board = board.clone().detach().to(dtype=torch.float32)

    board[board == 2] = -1
    return board





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
            


def mcts_search(root_board, turn, iterations=100):
    root = MCTNode(root_board)
    
    for _ in range(iterations):
        node = root
        turn = move

        while not is_terminal_node(node.board):
            expanded_node = node.expand(turn)
            if expanded_node is None:
                # If node is fully expanded, move to best child
                node = node.best_child(c_puct=1.41)
                if node is None:
                    break
                turn *= -1
            else:
                node = expanded_node
                turn *= -1
                break

        if node is None:
            continue

        # EVALUATION: Use model to evaluate leaf node
        model_board = convert_board_for_model(node.board.cpu().numpy())
        model_board_tensor = torch.tensor(model_board, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logit = model(model_board_tensor)
            prob = torch.sigmoid(logit).item()

        if prob < 0.1:
            value = 1
        elif prob > 0.9:
            value = -1 
        else:
            value = prob
        if winning_move(node.board, turn):
            value = turn
        node.backpropagate(value)

    print("AI's Turn: Possible actions with prob < 0.8")
    for action, child in root.children.items():
        model_board = convert_board_for_model(child.board.cpu().numpy())
        model_board_tensor = torch.tensor(model_board, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(model(model_board_tensor)).item()
        if prob < 0.8:
            print(f" - Column {action} -> Prob: {prob:.3f}")


    if root.children:
        best_action, _ = max(root.children.items(), key=lambda item: item[1].visit_count)
        return best_action
    else:
        return random.choice(valid_column(root.board))  # fallback


def ai_move(board):
    valid_moves = [c for c in range(column_count) if GameBoard.is_valid(board, c)]
    best_move = None
    best_value = float('inf')
    
    for col in valid_moves:
        row = GameBoard.next(board, col)
        if row is None:
            continue
        temp_board = board.copy()
        GameBoard.drop_piece(temp_board, row, col, 2)
        # Convert board to model representation (1 and -1)
        model_board = convert_board_for_model(temp_board)
        model_board_tensor = torch.tensor(model_board, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logit = model(model_board_tensor)
            # Convert logit to probability with sigmoid:
            prob = torch.sigmoid(logit).item()
        # For AI, a lower probability (closer to 0) is better
        if prob < best_value:
            best_value = prob
            best_move = col
    return best_move



board = GameBoard.create_board()
game_over = False
turn = move  # turn==0: human (player1), turn==1: AI (player2)
move_list = []  # Stores moves
myfont = pygame.font.SysFont("monospace", 75)

GameBoard.draw_board(board, move_list)

while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
            
        if turn == -1:
            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, black, (0, 0, column_count * square_size, square_size))
                posx = event.pos[0]
                if posx < column_count * square_size:
                    pygame.draw.circle(screen, red, (posx, int(square_size/2)), radius)
                pygame.display.update()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(screen, black, (0, 0, column_count * square_size, square_size))
                posx = event.pos[0]
                col = int(math.floor(posx / square_size))
                
                if col < column_count and GameBoard.is_valid(board, col):
                    row = GameBoard.next(board, col)
                    GameBoard.drop_piece(board, row, col, 1)  # human piece
                    move_notation = position_label(row, col)
                    move_list.append(f"P1: {move_notation}")
                    
                    if GameBoard.win_move(board, 1):
                        label = myfont.render("Player 1 wins!", 1, red)
                        screen.blit(label, (40, 10))
                        pygame.display.update()
                        pygame.time.wait(2000) 
                        game_over = True
                    GameBoard.draw_board(board, move_list)
                    turn = 1  # Switch to AI's turn
        
        # AI move (player2) when it's its turn
        if turn == 1 and not game_over:
            time.sleep(1)
            col = mcts_search(board, turn, 100)
            if col is not None and GameBoard.is_valid(board, col):
                row = GameBoard.next(board, col)
                GameBoard.drop_piece(board, row, col, 2)  # AI piece
                move_notation = position_label(row, col)
                move_list.append(f"P2: {move_notation}")
                
                if GameBoard.win_move(board, 2):
                    label = myfont.render("Player 2 wins!", 1, yellow)
                    screen.blit(label, (40, 10))
                    pygame.display.update()
                    pygame.time.wait(2000)                    
                    game_over = True
                GameBoard.draw_board(board, move_list)
                turn = -1  # Switch back to human
        
    if game_over:
        pygame.time.wait(2000)
        board = GameBoard.create_board()
        move_list = []
        GameBoard.draw_board(board, move_list)
        game_over = False
        turn = -1
