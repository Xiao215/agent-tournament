import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Sequence

from src.agent import Agent
from src.games.base import Game


class Mechanism(ABC):
    def __init__(self, base_game: Game):
        self.base_game = base_game

    @abstractmethod
    def run(self, agents: Sequence[Agent]) -> Any:
        # TODO change this immediately once experiment could run and more consideration is taken to the function design
        """Run the mechanism over the base game."""
        raise NotImplementedError

    def post_tournament(self, match_records: list[list[dict]]) -> None:
        """Most mechanisms do not need to implement this, but Reputation needs it to update reputation scores."""
        pass


class NoMechanism(Mechanism):
    """A mechanism that does nothing."""

    def run(self, agents: Sequence[Agent]) -> dict[str, float]:
        """Run the base game without any modifications."""

        final_score = defaultdict(float)
        players_moves = self.base_game.play(additional_info="None.", agents=agents)
        for move in players_moves:
            final_score[move.name] = final_score[move.name] + move.points

        return final_score
