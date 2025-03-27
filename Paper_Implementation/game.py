import sys
import numpy as np
import pygame
import math
import torch

# Game constants
row_count = 6
column_count = 7
square_size = 100
radius = int(square_size / 2 - 5)

# Colors
blue = (0, 0, 255)
black = (0, 0, 0)
red = (255, 0, 0)
yellow = (255, 255, 0)
white = (255, 255, 255)

width = column_count * square_size + 200
height = (row_count + 1) * square_size 
size = (width, height)

# Board Mapping (Assigning A1, B2, etc.)
column_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
row_labels = ['1', '2', '3', '4', '5', '6']

def get_position_label(row, col):
    return column_labels[col] + row_labels[row]

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Connect Four with Move Tracking")
font = pygame.font.SysFont("monospace", 30)

class GameBoard:
    @staticmethod
    def create_board():
        return np.zeros((row_count, column_count))

    @staticmethod
    def drop_piece(board, row, col, piece):
        board[row][col] = piece

    @staticmethod
    def is_valid_location(board, col):
        return board[0][col] == 0

    @staticmethod
    def get_next_open_row(board, col):
        for r in range(row_count-1, -1, -1):
            if board[r][col] == 0:
                return r

    @staticmethod
    def winning_move(board, piece):
        # Check horizontal
        for c in range(column_count - 3):
            for r in range(row_count):
                if all(board[r][c+i] == piece for i in range(4)):
                    return True
        # Check vertical
        for c in range(column_count):
            for r in range(row_count - 3):
                if all(board[r+i][c] == piece for i in range(4)):
                    return True
        # Check diagonals
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
        flipped = np.flipud(board)  # Flip the rows for correct visualization
        screen.fill(black)  

        # Draw board background
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

        # Draw Move History on the right side
        pygame.draw.rect(screen, white, (column_count * square_size, 0, 200, height))
        move_font = pygame.font.SysFont("monospace", 25)
        y_offset = 50
        for i, move in enumerate(move_list):
            label = move_font.render(move, 1, red if i % 2 == 0 else black)
            screen.blit(label, (column_count * square_size + 20, y_offset))
            y_offset += 30

        pygame.display.update()


board = GameBoard.create_board()
game_over = False
turn = 0 
move_list = []  # Stores moves as A1, B2


GameBoard.draw_board(board, move_list)
myfont = pygame.font.SysFont("monospace", 75)

dataform = torch.zeros((6, 7))
    
# Main game loop
while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
            
        if event.type == pygame.MOUSEMOTION:
            pygame.draw.rect(screen, black, (0, 0, column_count * square_size, square_size))
            posx = event.pos[0]
            if posx < column_count * square_size:
                color = red if turn == 0 else yellow
                pygame.draw.circle(screen, color, (posx, int(square_size/2)), radius)
            pygame.display.update()
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            pygame.draw.rect(screen, black, (0, 0, column_count * square_size, square_size))
            posx = event.pos[0]
            col = int(math.floor(posx / square_size))

            if col < column_count and GameBoard.is_valid_location(board, col):
                row = GameBoard.get_next_open_row(board, col)
                GameBoard.drop_piece(board, row, col, turn + 1)

                move_notation = get_position_label(row, col)
                move_list.append(f"P{turn+1}: {move_notation}")
                for i,move in enumerate(move_list):
                    #print(move)
                    pass

                if GameBoard.winning_move(board, turn + 1):
                    label = myfont.render(f"Player {turn + 1} wins!", 1, red if turn == 0 else yellow)
                    screen.blit(label, (40, 10))
                    pygame.display.update()  
                    pygame.time.wait(1000)
                    game_over = True  
                    

                    
                                    
                if color == red:
                    dataform[row][col] = 1
                elif color == yellow:
                    dataform[row][col] = -1
                print(dataform)
                
                GameBoard.draw_board(board, move_list)
                turn = (turn + 1) % 2  # Switch turns

            if game_over:
                pygame.time.wait(2000)
                
                board = GameBoard.create_board()
                move_list = []
                GameBoard.draw_board(board, move_list)
