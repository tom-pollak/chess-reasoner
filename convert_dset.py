import io
import random
import chess
import chess.pgn
from datasets import load_dataset

dataset = load_dataset("Icannos/lichess_games", streaming=True)

def print_random_position(row):
    pgn = io.StringIO(row['text'])
    game = chess.pgn.read_game(pgn)
    board = game.board() # type: ignore
    mainline_moves = list(game.mainline_moves())
    for move in mainline_moves[:random.randint(0, len(mainline_moves))]:
        board.push(move)
    return board.fen()

if __name__ == "__main__:
    r = next(iter(dataset['train']))
    print_random_position(game)
