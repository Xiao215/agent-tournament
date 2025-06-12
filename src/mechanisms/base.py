from collections import defaultdict
from logging import Logger

from abc import ABC, abstractmethod
from src.games.base import Game
from src.utils import register_classes

mechanism_registry: dict[str, type] = {}
register_mechanism = register_classes(mechanism_registry)

class Mechanism(ABC):
    def __init__(self, base_game: Game, logger: Logger | None):
        self.base_game = base_game
        self.logger = logger

    @abstractmethod
    def run(self) -> dict[str, float]:
        """Run the mechanism over the base game."""
        raise NotImplementedError

@register_mechanism
class NoMechanism(Mechanism):
    """A mechanism that does nothing."""
    def __init__(self, base_game: Game, logger: Logger | None = None):
        super().__init__(base_game, logger)

    def run(self):
        """Run the base game without any modifications."""
        if self.logger:
            self.logger.info(
                f"{'='*5} NoMechanism @ {self.base_game.__class__.__name__} {'='*5}"
            )

        final_score = defaultdict(float)
        players_moves = self.base_game.play(additional_info="None.")
        for move in players_moves:
            final_score[move.name] = final_score[move.name] + move.points

        if self.logger:
            self.logger.info(
                f"{'='*5} Final Score {'='*5}\n"
                + "\n".join(f"{name}: {score}" for name, score in final_score.items())
            )
        return final_score