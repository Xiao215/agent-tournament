from dataclasses import dataclass
from logging import Logger
from abc import ABC, abstractmethod
from typing import ClassVar

from src.utils import register_classes
from src.agent import Agent

game_registry: dict[str, type] = {}
register_game = register_classes(game_registry)

@dataclass(frozen=True)
class Move:
    """
    A record of one player's action in a single round.

    Attributes:
        name: The name of the player.
        value: The action taken by the player.
        points: The points scored by the player in this round.
    """
    name: str
    value: str
    points: float

class Game(ABC):
    num_players: ClassVar[int]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'num_players'):
            raise TypeError(f"{cls.__name__} must define num_players")

    def __init__(self, debugger: Logger, agents: list[Agent]):
        self.debugger = debugger
        self.agents = agents

        assert len(self.agents) == self.num_players, (
            f"Game {self.__class__.__name__} requires exactly {self.num_players} agents, "
            f"but got {len(self.agents)}."
        )

    @abstractmethod
    def play(self, additional_info: str) -> list[Move]:
        """Play the game."""
        raise NotImplementedError

    @abstractmethod
    def _parse_action(self, response: str):
        """Base on the game type, extract the decision made by the llm agent."""
        raise NotImplementedError
