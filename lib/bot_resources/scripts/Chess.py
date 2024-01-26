import numpy as np
import chess
from . import utils

class Chess():
    def __init__(self):
      self.rows = 8
      self.cols = 8
      self.action_space = 1968

    def valid_moves(self, board):
      move_array = np.zeros(self.action_space, dtype=np.float32)
      moves = board.legal_moves

      for move in moves:
        uci = chess.Move.uci(move)
        idx = utils.uciToIdx[uci]
        move_array[idx] = 1

      return move_array

    def value_and_terminal(self, board):
      outcome = board.outcome()

      if outcome != None:
        if outcome.winner == None:
          return 0, True
        return 1, True
      return 0, False

    def encode_state(self, board):
      b3d = np.zeros((14, 8, 8), dtype=np.float32)

      for p in chess.PIECE_TYPES:
          for sq in board.pieces(p, board.turn):
            idx = np.unravel_index(sq, (8, 8))
            b3d[p - 1][7 - idx[0]][idx[1]] = 1

          for sq in board.pieces(p, not board.turn):
            idx = np.unravel_index(sq, (8, 8))
            b3d[p + 5][7 - idx[0]][idx[1]] = 1

          for m in board.legal_moves:
            i, j = utils.sqToIdx(m.to_square)
            b3d[12][i][j] = 1

          board.turn = not board.turn

          for m in board.legal_moves:
            i, j = utils.sqToIdx(m.to_square)
            b3d[13][i][j] = 1

          board.turn = not board.turn

      return b3d