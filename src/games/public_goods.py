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
            return "<A1>"
        elif self == PublicGoodsAction.FREE_RIDE:
            return "<A2>"
        else:
            raise ValueError(f"Unknown action {self!r}")

    @classmethod
    def from_token(cls, token: str) -> "PublicGoodsAction":
        """Parse an action from a token like "<A1>" or "<A2>"."""
        try:
            idx = int(token.strip("<>").lstrip("A"))
            action = list(cls)[idx]
        except Exception as exp:
            raise ValueError(f"Unknown action token {token!r}") from exp
        return action


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
    ):

        assert num_players > 1, "Public Goods must have at least 2 players."
        assert 1.0 <= multiplier <= num_players, "Multiplier should be between 1 and num_players."

        self.endowment = 1  # value of endowment does not matter
        self.multiplier = multiplier
        self.num_players = num_players

        prompt = f"""
        You are a player named {{player_name}} in a simultaneous contribution game with {self.num_players} players.
        Your objective is to maximise your cumulative wealth in the game described below.

        Game Rules
        1) You and your opponent each start with {self.endowment} dollars.
        2) You both decide simultaneously whether to invest all of your starting money or none of it.
        3) The total invested is multiplied by {self.multiplier} and split equally between you.
        4) Any dollars you do not invest remain yours.

        Actions (choose exactly one per round):
        - {PublicGoodsAction.CONTRIBUTE.to_token()} - Put all your starting money into the joint fund
        - {PublicGoodsAction.FREE_RIDE.to_token()} - Invest nothing and keep all your starting money

        Decision Procedure
        - Decide on a probability distribution over A1 and A2 each round.
        - Probabilities must be integers in [0, 100] that sum to 100.
        - If you are very certain, you may assign 100 to one action and 0 to the other.
        - Report your chosen probabilities in the exact format below.

        Output Specification
        - On the last line, print both probabilities in this exact format:
        <A1=probability>|<A2=probability>
        - Replace `probability` with integers between 0 and 100 that sum to 100.
        - Example placeholders only: <A1=XX>|<A2=YY>

        {{additional_info}}
        """

        super().__init__(prompt=prompt, num_players=num_players)

    def play(self, additional_info: str, players: Sequence[Agent]) -> list[Game.Move]:
        """
        Runs the Public Goods game: collects all actions, computes payoffs,
        and returns a list of Moves with each agent's action and points earned.
        """
        assert (
            len(players) == self.num_players
        ), f"Expected {self.num_players} agents, got {len(players)}."

        actions = {}
        responses = {}

        for player in players:
            resp = self._prompt_player(player, additional_info)
            prob_distribution = self._parse_mix_strategy(player, resp)
            act = PublicGoodsAction.from_token(
                self._choose_from_mix_strategy(prob_distribution)
            )
            actions[player.name] = act
            responses[player.name] = resp

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
                    response=responses[name],
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
