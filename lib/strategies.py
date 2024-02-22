from __future__ import annotations
import torch
import chess
from chess.engine import PlayResult, Limit
from .engine_wrapper import MinimalEngine
from typing import Any

from lib.bot_resources.scripts.InceptionNet import InceptionNet
from lib.bot_resources.scripts.MCTS import MCTS

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

class inception_bot(MinimalEngine):
  model = InceptionNet(14, 180, 180, 5)
  model.load_state_dict(torch.load("lib/bot_resources/pretrained/inception_net_pretrained.pt"))
  model.eval()

  mcts = MCTS(model)

  def search(self, board: chess.Board, time_limit: Limit, *args: Any) -> PlayResult:
    if board.turn == chess.WHITE:
      time = time_limit.white_clock
      inc = time_limit.white_inc
    else:
      time = time_limit.black_clock
      inc = time_limit.black_inc

    if inc >= 3:
      return PlayResult(self.mcts.search(board, 100), None)
    elif inc >= 2:
      if time >= 90:
        return PlayResult(self.mcts.search(board, 100), None)
      else:
        return PlayResult(self.mcts.search(board, 75), None)
    elif inc >= 1:
      if time >= 120:
        return PlayResult(self.mcts.search(board, 100), None)
      elif time >= 60:
        return PlayResult(self.mcts.search(board, 50), None)
      else:
        return PlayResult(self.mcts.search(board, 40), None)
    
    if time >= 75:
      return PlayResult(self.mcts.search(board, 50), None)
    
    return PlayResult(self.mcts.search(board, 25), None)