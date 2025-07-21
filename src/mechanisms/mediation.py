
from src.mechanisms.base import Mechanism
from src.games.base import Game
from src.agent import Agent


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


    def run(self, agents: list[Agent]) -> dict[str, float]:
        """Repeat the base game for a specified number of repetitions.

        Returns:
            final_score (dict[str, float]): A dictionary mapping player names to their final scores after all rounds.
        """
        NotImplementedError("Mediation mechanism is not implemented yet.")
