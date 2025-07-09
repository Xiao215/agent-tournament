from dataclasses import dataclass
from logging import Logger
from enum import Enum
from abc import ABC, abstractmethod

from src.agent import Agent

class Game(ABC):
    """
    Base class for all games in the tournament.
    """

    @dataclass(frozen=True)
    class Move:
        """
        A record of one player's action in a single round.

        Attributes:
            name: The name of the player.
            action: The action taken by the player.
            points: The points scored by the player in this round.
        """
        name: str
        action: str
        points: float


    Action = Enum("Action", {})
    def __init__(self, debugger: Logger | None = None):
        self.debugger = debugger

    @abstractmethod
    def play(self, additional_info: str, agents: list[Agent]) -> list[Move]:
        """Play the game."""
        raise NotImplementedError

    @abstractmethod
    def _parse_action(self, response: str):
        """Base on the game type, extract the decision made by the llm agent."""
        raise NotImplementedError
