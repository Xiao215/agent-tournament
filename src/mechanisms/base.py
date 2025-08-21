import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Sequence

from src.agent import Agent
from src.games.base import Game
from src.evolution.population_payoffs import PopulationPayoffs


class Mechanism(ABC):
    def __init__(self, base_game: Game):
        self.base_game = base_game

    @abstractmethod
    def run(self, agents: Sequence[Agent]) -> PopulationPayoffs:
        # TODO change this immediately once experiment could run and more consideration is taken to the function design
        """Run the mechanism over the base game."""
        raise NotImplementedError


class NoMechanism(Mechanism):
    """A mechanism that does nothing."""

    def run(self, agents: Sequence[Agent]) -> PopulationPayoffs:
        """Run the base game without any modifications."""
        payoffs = PopulationPayoffs(agent_names=[agent.name for agent in agents])
        final_score = defaultdict(float)
        players_moves = self.base_game.play(additional_info="None.", player=agents)
        for move in players_moves:
            final_score[move.name] = final_score[move.name] + move.points

        return final_score
