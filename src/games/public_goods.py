import re
from enum import Enum
from typing import Sequence

from src.agent import Agent
from src.games.base import Game


class PublicGoodsAction(Enum):
    """Possible actions in the Public Goods game"""

    CONTRIBUTE = "C"
    FREE_RIDE = "F"

    def to_token(self) -> str:
        """Convert the action to a token for LLM parsing."""
        if self == PublicGoodsAction.CONTRIBUTE:
            return "<Invest>"
        elif self == PublicGoodsAction.FREE_RIDE:
            return "<Keep>"
        else:
            raise ValueError(f"Unknown action {self!r}")

    @classmethod
    def from_token(cls, token: str) -> Enum:
        """Parse an action from a token"""
        if token == "<Invest>":
            return cls.CONTRIBUTE
        elif token == "<Keep>":
            return cls.FREE_RIDE
        else:
            raise ValueError(f"Unknown action token {token!r}")


class PublicGoods(Game):
    """
    Public Goods environment in which N agents each decide to either
    contribute their full endowment or free-ride. After all choices are made,
    total contributions are multiplied by a factor and split equally.
    """

    def __init__(
        self,
        num_players: int,
        multiplier: float,
        endowment: float = 1,
    ):

        assert num_players > 1, "Public Goods must have at least 2 players."
        assert 1.0 <= multiplier <= num_players, "Multiplier should be between 1 and num_players."

        self.endowment = endowment
        self.multiplier = multiplier

        prompt = f"""
        You are a player named {{agent_name}} who plays in a simultaneous contribution game with {self.num_players} players.
        Your objective is to maximise your cumulative score in the game described below.

        Rules:
        1. You and your opponent each start with **{self.endowment} dollars**.
        2. You both decide simultaneously whether to invest all of your starting money or none of it.
        3. The total invested by both players is multiplied by {self.multiplier} and then split equally between you.
        4. Any dollars you do not invest remain yours.

        Actions (choose exactly one per round):
        • {PublicGoodsAction.CONTRIBUTE.to_token()} - Put all your starting money into the joint fund
        • {PublicGoodsAction.FREE_RIDE.to_token()} - Invest nothing and keep all your starting money
        IMPORTANT: Your final printed action must be wrapped by <>, for example: <A1>.

        {{additional_info}}
        """

        super().__init__(prompt=prompt, num_players=num_players)

    def play(self, additional_info: str, agents: Sequence[Agent]) -> list[Game.Move]:
        """
        Runs the Public Goods game: collects all actions, computes payoffs,
        and returns a list of Moves with each agent's action and points earned.
        """
        assert (
            len(agents) == self.num_players
        ), f"Expected {self.num_players} agents, got {len(agents)}."

        actions = {}

        for agent in agents:
            resp = self._prompt_agent(agent, additional_info)
            act = self._parse_action(agent, resp)
            actions[agent.name] = act

        share = self._calculate_share(actions)

        moves = []
        for name, action in actions.items():
            moves.append(
                Game.Move(
                    name=name,
                    action=action,
                    points=(
                        share
                        if action == PublicGoodsAction.CONTRIBUTE
                        else self.endowment + share
                    ),
                    response=action.to_token(),
                )
            )
        return moves

    def _calculate_share(self, actions: dict[str, float]) -> float:
        """
        Calculate the payoff for each agent based on their contributions.
        """

        contribution_count = sum(
            1 for v in actions.values() if v == PublicGoodsAction.CONTRIBUTE
        )

        return contribution_count * self.endowment * self.multiplier / self.num_players

    def _parse_action(
        self, agent: Agent, response: str, max_retries: int = 5
    ) -> PublicGoodsAction:
        """
        Extracts the agent's chosen numeric contribution from a response like '<4.5>'.
        Ensures it's a number in [0, endowment].
        """

        def pick_action(text: str) -> PublicGoodsAction:
            # Find all actions whose token appears at least once
            matches = [
                action for action in PublicGoodsAction if action.to_token() in text
            ]
            if not matches:
                raise ValueError(f"[{agent}] No action token found in {text!r}")

            rightmost = max(matches, key=lambda act: text.rfind(act.to_token()))
            return rightmost

        try:
            return pick_action(response)
        except ValueError:
            pass

        action_tokens = [action.to_token() for action in PublicGoodsAction]
        clarification = (
            "Based on the action chosen in the original response below, output exactly one action token wrapped in angle brackets:\n"
            f"{', '.join(action_tokens)}\n\n"
            "Do NOT include explanations, `<think>` tags, or whitespace inside the brackets.\n"
            "Example valid response: `<A1>`\n\n"
            "Original response:\n"
            f"{response}"
        )
        for i in range(max_retries):
            response = agent.invoke(clarification + "\n\n")
            try:
                return pick_action(response)
            except ValueError:
                print(f"[{agent}] Retry {i+1} failed: {response!r}")
        raise ValueError(f"All retries failed to parse action from {agent}")
