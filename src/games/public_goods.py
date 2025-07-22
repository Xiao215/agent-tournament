import re
from typing import Sequence

from src.agent import Agent
from src.games.base import Game


class PublicGoods(Game):
    """
    Public Goods environment in which N agents each decide to either
    contribute their full endowment or free-ride. After all choices are made,
    total contributions are multiplied by a factor and split equally.
    """

    def __init__(
        self,
        num_players: int,
        endowment: float,
        multiplier: float,
    ):

        assert num_players > 1, "Public Goods must have at least 2 players."
        assert 1.0 <= multiplier <= num_players, "Multiplier should be between 1 and num_players."

        self.endowment = endowment
        self.multiplier = multiplier

        prompt = f"""
        You are an expert called '{{agent_name}}' playing Public Goods game with {self.num_players} players.

        Each player starts with an endowment of {self.endowment}.
        Choose a contribution from [0, {endowment}].
        Wrap it in <>, e.g. <0>.
        After all decisions, total contributions are multiplied by {self.multiplier}
        and split equally among all players.

        This means, while collectively contributing maximizes the group payoff,
        individually free-riding can yield a higher personal payoff for you.

        IMPORTANT: Your final printed action must be wrapped in <>,
        for example: <0> or <{self.endowment}>.

        Additional information:
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

        contributions: list[float] = []
        names: list[str] = []

        # TODO: Might not have enough GPU to run 2+ agents in parallel. Need to come up with a better way to handle this.
        for agent in agents:
            resp = agent.chat(
                self.prompt.format(
                    agent_name=str(agent),
                    additional_info=additional_info,
                )
            )
            c = self._parse_action(resp)
            names.append(str(agent))
            contributions.append(c)

            # if self.debugger:
            #     self.debugger.info(f"{str(agent)} responded '{resp}', parsed contribution = {c}")

        payoffs = self._calculate_payoff(contributions)

        moves: list[Game.Move] = []
        for name, contrib, payoff in zip(names, contributions, payoffs):
            moves.append(Game.Move(
                name=name,
                action=str(contrib),
                points=payoff,
            ))
        return moves

    def _calculate_payoff(self, contributions: list[float]) -> list[float]:
        """
        Calculate the payoff for each agent based on their contributions.
        """
        total_contribution = sum(contributions)
        public_pool = total_contribution * self.multiplier
        share = public_pool / self.num_players

        # self.debugger.info(
        #     f"Each player's share: {share}"
        # )

        payoffs = []
        for contribution in contributions:
            payoff = (self.endowment - contribution) + share
            payoffs.append(payoff)

        return payoffs

    def _parse_action(self, response: str) -> float:
        """
        Extracts the agent's chosen numeric contribution from a response like '<4.5>'.
        Ensures it's a number in [0, endowment].
        """

        # Regex: match '<number>' or '<number.number>'
        m = re.fullmatch(r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>", response.strip())
        if not m:
            raise ValueError(
                f"Invalid format '{response}'. "
                f"Expected a single number wrapped in <>, e.g. <3> or <2.5>."
            )

        value = float(m.group(1))

        if not 0.0 <= value <= self.endowment:
            raise ValueError(
                f"Contribution {value} out of bounds. "
                f"Must be between 0 and {self.endowment}."
                f"Above error comes from response: {response}"
            )

        return value
