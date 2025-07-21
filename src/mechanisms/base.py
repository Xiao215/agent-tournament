from collections import defaultdict

from abc import ABC, abstractmethod
from src.agent import Agent
from src.games.base import Game

class Mechanism(ABC):
    def __init__(self, base_game: Game):
        self.base_game = base_game

    @abstractmethod
    def run(self, agents: list[Agent]) -> dict[str, float]:
        """Run the mechanism over the base game."""
        raise NotImplementedError

class NoMechanism(Mechanism):
    """A mechanism that does nothing."""
    def __init__(self, base_game: Game):
        super().__init__(base_game)

    def run(self, agents: list[Agent]) -> dict[str, float]:
        """Run the base game without any modifications."""
        # if self.logger:
        #     self.logger.info(
        #         f"{'='*5} NoMechanism @ {self.base_game.__class__.__name__} {'='*5}"
        #     )

        final_score = defaultdict(float)
        players_moves = self.base_game.play(additional_info="None.", agents=agents)
        for move in players_moves:
            final_score[move.name] = final_score[move.name] + move.points

        # if self.logger:
        #     self.logger.info(
        #         f"{'='*5} Final Score {'='*5}\n"
        #         + "\n".join(f"{name}: {score}" for name, score in final_score.items())
        #     )
        return final_score
