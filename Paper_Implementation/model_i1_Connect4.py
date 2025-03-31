import sys
import numpy as np
import pygame
import math
import torch
import torch.nn as nn
import torch.optim as optim
import time
pygame.init()
row_count = 6
column_count = 7
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
model.load_state_dict(torch.load(r"C:\Users\saatw\Downloads\connect_four_epoch_800.pt", map_location=device))
model.eval()

def convert_board_for_model(board):
    """
    Convert game board from our representation (0, 1, 2) to the model representation:
    - Human's piece (1) remains 1.
    - AI's piece (2) becomes -1.
    """
    converted = board.copy().astype(np.float32)
    converted[converted == 2] = -1
    return converted

def ai_move(board):
    """
    Given the current board (numpy array with values 0,1,2), choose the move for AI (player2)
    by evaluating all valid moves using the CNN model.
    The model predicts the likelihood of a player1 win (1 means certain win, 0 means win for AI).
    So AI will choose the move that minimizes this value.
    """
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
turn = 0  # turn==0: human (player1), turn==1: AI (player2)
move_list = []  # Stores moves
myfont = pygame.font.SysFont("monospace", 75)

GameBoard.draw_board(board, move_list)

while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
            
        if turn == 0:
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
            col = ai_move(board)
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
                turn = 0  # Switch back to human
        
    if game_over:
        pygame.time.wait(2000)
        board = GameBoard.create_board()
        move_list = []
        GameBoard.draw_board(board, move_list)
        game_over = False
        turn = 0
        
        
""" kya Crazy khela
"""
