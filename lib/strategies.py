from __future__ import annotations
import torch
import chess
from chess.engine import PlayResult
from .engine_wrapper import MinimalEngine
from typing import Any

from lib.bot_resources.src.InceptionNet import InceptionNet
from lib.bot_resources.src.MCTS import MCTS

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

class inception_bot(MinimalEngine):
  model = InceptionNet(14, 180, 180, 5)
  model.load_state_dict(torch.load("lib/bot_resources/saved/inception_net.pt"))
  model.eval()

  mcts = MCTS(model)

  def search(self, board: chess.Board, *args: Any) -> PlayResult:
    return PlayResult(self.mcts.search(board, 50), None)