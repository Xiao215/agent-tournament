
from src.agent import Agent
from src.games.base import Game
from src.mechanisms.base import Mechanism

# [TBD]

class Mediation(Mechanism):
    """
    Repetition mechanism that allows for multiple rounds of the same game.
    """
    def __init__(
            self,
            base_game: Game,
        ) -> None:
        super().__init__(base_game)
