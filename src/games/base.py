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

    def __init__(
        self,
        prompt: str,
        num_players: int,
        debugger: Logger | None = None
    ) -> None:
        self.prompt = prompt
        self.num_players = num_players
        self.debugger = debugger

    @abstractmethod
    def play(
        self,
        additional_info: str,
        agents: list[Agent]
    ) -> list[Move]:
        """Play the game."""
        raise NotImplementedError

    def _chat_and_parse(
        self,
        agent: Agent,
        additional_info: str,
    ) -> tuple[str, Enum]:
        prompt = self.prompt.format(
            agent_name=agent.name,
            additional_info=additional_info,
        )
        resp = agent.chat(prompt)
        return resp, self._parse_action(agent, resp)

    @abstractmethod
    def _parse_action(self, agent: Agent, response: str) -> Enum:
        """Base on the game type, extract the decision made by the llm agent. If decision is not valid with regex, use the agent to generate a valid action."""
        raise NotImplementedError
