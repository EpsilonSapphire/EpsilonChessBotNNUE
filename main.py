import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
import numpy as np
import pygame
import os
import random
from stockfish import Stockfish

# Initialize Pygame for GUI
WIDTH, HEIGHT = 640, 640
SQUARE_SIZE = WIDTH // 8
WHITE, BLACK = (238, 238, 210), (118, 150, 86)
HIGHLIGHT_COLOR = (255, 255, 0)
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess with AI Training")

# Load chess piece images
pieces = {}
for piece in ["p", "r", "n", "b", "q", "k"]:
    pieces[piece] = pygame.image.load(f"images/{piece}.png")  # Black pieces
    pieces[piece.upper()] = pygame.image.load(f"images/{'m' + piece}.png")  # White pieces

# Initialize Stockfish engine
stockfish = Stockfish(path="stockfish", parameters={"Threads": 2, "Minimum Thinking Time": 30})
stockfish.set_skill_level(20)


def draw_board():
    """Draw the chessboard on the screen."""
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))


def draw_pieces(board):
    """Draw pieces on the board."""
    for row in range(8):
        for col in range(8):
            piece = board.piece_at(chess.square(col, 7 - row))
            if piece:
                piece_img = pieces[piece.symbol()]
                screen.blit(pygame.transform.scale(piece_img, (SQUARE_SIZE, SQUARE_SIZE)),
                            (col * SQUARE_SIZE, row * SQUARE_SIZE))


def encode_position(board):
    """Encode the chessboard into bitboards for NNUE input."""
    planes = np.zeros((16, 64), dtype=np.float32)
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color_offset = 6 if piece.color == chess.BLACK else 0
            planes[piece_map[piece.piece_type] + color_offset][square] = 1

    return planes.flatten()


def get_stockfish_evaluation(board, depth=12):
    """Get Stockfish evaluation for a given position."""
    stockfish.set_fen_position(board.fen())
    return stockfish.get_evaluation()["value"] / 100.0


class NNUE(nn.Module):
    def __init__(self):
        """Define the neural network architecture for NNUE."""
        super(NNUE, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.batch_norm1 = nn.BatchNorm1d(512)

    def forward(self, x):
        """Forward pass of the neural network."""
        x = self.leaky_relu(self.batch_norm1(self.fc1(x)))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_positions_from_folder(folder):
    """Load chess positions from PGN files and encode them for training."""
    dataset = []
    for file in os.listdir(folder):
        if file.endswith(".pgn"):
            with open(os.path.join(folder, file), encoding='utf-8') as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    board = game.board()
                    for move in game.mainline_moves():
                        board.push(move)
                        encoded = encode_position(board)
                        evaluation = get_stockfish_evaluation(board)
                        dataset.append((encoded, evaluation))
    return dataset


def train_nnue(model, dataset, epochs=10, lr=0.001):
    """Train the NNUE model using supervised learning."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0
        for position, eval in dataset:
            x = torch.tensor(position, dtype=torch.float32).unsqueeze(0)
            y = torch.tensor([eval], dtype=torch.float32)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


def mcts_search(board, model, simulations=100):
    """Apply MCTS to find the best move"""
    move_scores = {}
    for move in board.legal_moves:
        move_scores[move] = 0

    for _ in range(simulations):
        board_copy = board.copy()
        for move in board.legal_moves:
            board_copy.push(move)
            encoded = torch.tensor(encode_position(board_copy), dtype=torch.float32).unsqueeze(0)
            eval_score = model(encoded).item()
            move_scores[move] += eval_score
            board_copy.pop()

    return max(move_scores, key=move_scores.get)


# Initialize the chess game
board = chess.Board()
dataset = load_positions_from_folder("games")
nnue = NNUE()
train_nnue(nnue, dataset)
torch.save(nnue.state_dict(), "nnue_model.pth")
nnue.load_state_dict(torch.load("nnue_model.pth"))
nnue.eval()

running = True
selected_square = None
while running:
    draw_board()
    draw_pieces(board)
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            col = x // SQUARE_SIZE
            row = 7 - (y // SQUARE_SIZE)
            square = chess.square(col, row)
            if selected_square is None:
                selected_square = square
            else:
                move = chess.Move(selected_square, square)
                if move in board.legal_moves:
                    board.push(move)
                    ai_move = mcts_search(board, nnue)
                    if ai_move:
                        board.push(ai_move)
                selected_square = None









