"""
Some example classes for people who want to create a homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""
import chess
from chess.engine import PlayResult, Limit
import logging
import torch
from typing import Any

from lib.engine_wrapper import MinimalEngine
from model.inception_net import InceptionNet
from model.mcts import MCTS


# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

class ExampleEngine(MinimalEngine):
  """An example engine that all homemade engines inherit."""

class inception_bot(MinimalEngine):
  model = InceptionNet(14, 180, 180, 5)
  model.load_state_dict(torch.load("model/inception_net_pretrained.pt"))
  model.eval()

  mcts = MCTS(model)

  def search(self, board: chess.Board, time_limit: Limit, *args: Any) -> PlayResult:
    if isinstance(time_limit.time, float):
      time = time_limit.time
      inc = 0
    elif board.turn == chess.WHITE:
      time = time_limit.white_clock if isinstance(time_limit.white_clock, float) else 0
      inc = time_limit.white_inc if isinstance(time_limit.white_inc, float) else 0
    else:
      time = time_limit.black_clock if isinstance(time_limit.black_clock, float) else 0
      inc = time_limit.black_inc if isinstance(time_limit.black_inc, float) else 0

    if inc >= 3:
      return PlayResult(self.mcts.search(board, 125), None)
    elif inc >= 2:
      if time >= 90:
        return PlayResult(self.mcts.search(board, 125), None)
      else:
        return PlayResult(self.mcts.search(board, 100), None)
    elif inc >= 1:
      if time >= 120:
        return PlayResult(self.mcts.search(board, 125), None)
      elif time >= 60:
        return PlayResult(self.mcts.search(board, 75), None)
      else:
        return PlayResult(self.mcts.search(board, 50), None)
    
    if time >= 60:
      return PlayResult(self.mcts.search(board, 75), None)
    
    return PlayResult(self.mcts.search(board, 25), None)