import sys
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import math
from torch.utils.data import DataLoader

move = int(input("ENTER TURN MODE (Turn=0: human moves first, Turn=1: AI moves first): "))

def winning_move(board, piece):
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            if board[r, c] == piece and board[r, c+1] == piece and board[r, c+2] == piece and board[r, c+3] == piece:
                return True
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r, c] == piece and board[r+1, c] == piece and board[r+2, c] == piece and board[r+3, c] == piece:
                return True
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            if board[r, c] == piece and board[r+1, c+1] == piece and board[r+2, c+2] == piece and board[r+3, c+3] == piece:
                return True
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
            j = j + 1
    return j

def get_next_open_row(board, col):
    for r in range(ROW_COUNT-1, -1, -1):
        if board[r, col] == 0:
            return r
    return None

def num_valid_moves(board):
    valid_moves = [num_valid(board, col) for col in range(COLUMN_COUNT)]
    valid_moves = [i for i in valid_moves if i > 0]
    return len(valid_moves)

def valid_column(board):
    valid_cols = [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]
    return valid_cols

def simulate_game():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)
    states = []
    turn = 0
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
        outcome = 0
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

pygame.init()
row_count = 6
ROW_COUNT = 6
COLUMN_COUNT = 7
column_count = 7
square_size = 100
radius = int(square_size / 2 - 5)
blue = (0, 0, 255)
black = (0, 0, 0)
red = (255, 0, 0)
yellow = (255, 255, 0)
white = (255, 255, 255)
width = column_count * square_size + 200
height = (row_count + 1) * square_size
size = (width, height)
column_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
row_labels = ['1', '2', '3', '4', '5', '6']
def position_label(row, col):
    return column_labels[col] + row_labels[row]
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Connect 4")
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
        for c in range(column_count - 3):
            for r in range(row_count):
                if all(board[r][c+i] == piece for i in range(4)):
                    return True
        for c in range(column_count):
            for r in range(row_count - 3):
                if all(board[r+i][c] == piece for i in range(4)):
                    return True
        for c in range(column_count - 3):
            for r in range(row_count - 3):
                if all(board[r+i][c+i] == piece for i in range(4)):
                    return True
        for c in range(column_count - 3):
            for r in range(3, row_count):
                if all(board[r-i][c+i] == piece for i in range(4)):
                    return True
        return False
    @staticmethod
    def draw_board(board, move_list):
        flipped = np.flipud(board)
        screen.fill(black)
        for r in range(row_count):
            for c in range(column_count):
                pygame.draw.rect(screen, blue, (c*square_size, r*square_size+square_size, square_size, square_size))
                pygame.draw.circle(screen, black, (int(c*square_size+square_size/2), int(r*square_size+square_size+square_size/2)), radius)
        for r in range(row_count):
            for c in range(column_count):
                if flipped[r][c] == 1:
                    pygame.draw.circle(screen, red, (int(c*square_size+square_size/2), height - int(r*square_size+square_size/2)), radius)
                elif flipped[r][c] == 2:
                    pygame.draw.circle(screen, yellow, (int(c*square_size+square_size/2), height - int(r*square_size+square_size/2)), radius)
        pygame.draw.rect(screen, white, (column_count * square_size, 0, 200, height))
        move_font = pygame.font.SysFont("monospace", 25)
        y_offset = 50
        for i, move in enumerate(move_list):
            label = move_font.render(move, 1, red if i % 2 == 0 else black)
            screen.blit(label, (column_count * square_size + 20, y_offset))
            y_offset += 30
        pygame.display.update()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * ROW_COUNT * COLUMN_COUNT, COLUMN_COUNT)
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * ROW_COUNT * COLUMN_COUNT, 256)
        self.value_fc2 = nn.Linear(256, 1)
    def forward(self, x):
        x_res = F.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x_res = block(x_res)
        policy_logits = self.policy_conv(x_res)
        policy_logits = F.relu(self.policy_bn(policy_logits))
        policy_logits = policy_logits.view(policy_logits.size(0), -1)
        policy_logits = self.policy_fc(policy_logits)
        masked_policy_logits = invalid_moves(policy_logits, x)
        policy = F.softmax(masked_policy_logits, dim=-1)
        value = self.value_conv(x_res)
        value = F.relu(self.value_bn(value))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        return policy, value

class MCTNode():
    def __init__(self, board, parent=None, action_taken=None):
        self.board = board.clone().detach() if isinstance(board, torch.Tensor) else torch.tensor(board, dtype=torch.float32)
        self.parent = parent
        self.action_taken = action_taken
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        self.valid_actions = [col for col in range(COLUMN_COUNT) if GameBoard.is_valid(self.board.numpy(), col)]
    def is_fully_expanded(self):
        return len(self.children) == len(self.valid_actions)
    def expand(self, turn):
        if is_terminal_node(self.board) or self.is_fully_expanded():
            return None
        for action in self.valid_actions:
            if action not in self.children:
                row = get_next_open_row(self.board, action)
                if row is not None:
                    new_board = self.board.clone().detach()
                    new_board[row, action] = turn
                    self.children[action] = MCTNode(new_board, parent=self, action_taken=action)
                    return self.children[action]
        return None
    def best_child(self, c_puct=1.41):
        if not self.children:
            return None
        return max(self.children.values(), key=lambda child: self.uct_value(child, c_puct))
    def uct_value(self, child, c_puct):
        if child.visit_count == 0:
            return float('inf')
        Q = child.total_value / (child.visit_count + 1e-6)
        U = c_puct * np.sqrt(np.log(self.visit_count + 1e-6) / (child.visit_count + 1e-6))
        return Q + U
    def backpropagate(self, value):
        self.visit_count += 1
        self.total_value += value
        if self.parent:
            self.parent.backpropagate(-value)

model = ResNet()
checkpoint_path = r"C:\Users\saatw\Downloads\checkpoint_game_500.pth"
try:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
except Exception as e:
    print("Error loading checkpoint:", e)
    sys.exit(1)
model.to(device)
model.eval()
print("Model loaded successfully.")

def ai_move(board, turn, num_simulations=100):
    root = MCTNode(board)
    for _ in range(num_simulations):
        node = root
        while True:
            next_node = node.best_child()
            if next_node is None:
                break
            node = next_node
        expanded_node = node.expand(turn)
        if expanded_node is not None:
            node = expanded_node
            board_input = node.board.unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                _, value = model(board_input)
            node.backpropagate(value.item())
        else:
            board_input = node.board.unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                _, value = model(board_input)
            node.backpropagate(value.item())
    chosen_child = root.best_child()
    if chosen_child is None:
        return random.choice(valid_column(board.numpy()))
    return chosen_child.action_taken

board = GameBoard.create_board()
game_over = False
turn = move
move_list = []
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
                    GameBoard.drop_piece(board, row, col, 1)
                    move_notation = position_label(row, col)
                    move_list.append(f"P1: {move_notation}")
                    if GameBoard.win_move(board, 1):
                        label = myfont.render("Player 1 wins!", 1, red)
                        screen.blit(label, (40, 10))
                        pygame.display.update()
                        pygame.time.wait(2000)
                        game_over = True
                    GameBoard.draw_board(board, move_list)
                    turn = 1
        if turn == 1 and not game_over:
            time.sleep(1)
            col = ai_move(torch.tensor(board, dtype=torch.float32), 2, num_simulations=100)
            if col is not None and GameBoard.is_valid(board, col):
                row = GameBoard.next(board, col)
                GameBoard.drop_piece(board, row, col, 2)
                move_notation = position_label(row, col)
                move_list.append(f"P2: {move_notation}")
                if GameBoard.win_move(board, 2):
                    label = myfont.render("Player 2 wins!", 1, yellow)
                    screen.blit(label, (40, 10))
                    pygame.display.update()
                    pygame.time.wait(2000)
                    game_over = True
                GameBoard.draw_board(board, move_list)
                turn = 0
    if not game_over and len(valid_column(board)) == 0:
        print("Game is a draw!")
        game_over = True
if game_over:
    pygame.time.wait(2000)
