import os
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Sequence

from config import OUTPUTS_DIR
from src.agent import Agent
from src.games.base import Game
from src.logging_config import setup_logger

now = datetime.now()
log_dir = OUTPUTS_DIR / f"{now.year}" / f"{now.month:02}" / f"{now.day:02}"
os.makedirs(log_dir, exist_ok=True)

logger = setup_logger(
    name="tournament_logger",
    log_file=str(log_dir / f"{now.hour:02}_{now.minute:02}_tournament.log"),
)


class Mechanism(ABC):
    def __init__(self, base_game: Game):
        self.base_game = base_game

    @abstractmethod
    def run(self, agents: Sequence[Agent]) -> dict[str, float]:
        """Run the mechanism over the base game."""
        raise NotImplementedError


class NoMechanism(Mechanism):
    """A mechanism that does nothing."""
    def __init__(self, base_game: Game):
        super().__init__(base_game)

    def run(self, agents: Sequence[Agent]) -> dict[str, float]:
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
