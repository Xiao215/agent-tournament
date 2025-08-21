from enum import Enum
from typing import Sequence

from src.agent import Agent
from src.games.base import Game


class PrisonersDilemmaAction(Enum):
    """Possible actions in the Prisoner's Dilemma"""

    COOPERATE = "C"
    DEFECT = "D"

    def to_token(self) -> str:
        """Convert the action to a token (eg, A1) for LLM parsing."""
        idx = list(type(self)).index(self)
        return f"A{idx}"

    @classmethod
    def from_token(cls, token: str) -> "PrisonersDilemmaAction":
        """Parse an action from a token like "A1" or "A2"."""
        try:
            idx = int(token.lstrip("A"))
            action = list(cls)[idx]
        except Exception as exp:
            raise ValueError(f"Unknown action token {token!r}") from exp
        return action


class PrisonersDilemma(Game):
    """
    Prisoner's Dilemma environment that allows for one rounds of interaction
    between two LLM agents.
    """

    def __init__(
        self,
        payoff_matrix: dict[str, list[float]],
    ) -> None:
        self.payoff_matrix = self._parse_payoff_matrix(payoff_matrix)

        self.action_tokens = [act.to_token() for act in PrisonersDilemmaAction]
        actions_block = "\n\t".join(
            [f"\t- {act_tok}" for act_tok in self.action_tokens]
        )

        prompt = f"""
        You are a player named {{player_name}} who plays in a decision-making game.
        Your objective is to maximise your cumulative score in the game described below.

        Rule:
        1. You and the opponent each choose exactly one action, simultaneously.
        2. After both commitments are locked in, the two actions are revealed.
        3. Both players receive the points specified in the payoff table below.

        Actions (choose exactly one per round):
        {actions_block}

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

        Payoff matrix:
        {self._payoff_description()}

        {{additional_info}}
        """

        super().__init__(
            prompt=prompt,
            num_players=2,
        )

    def _payoff_description(self) -> str:
        lines = []
        for (a, b), (pts_a, pts_b) in self.payoff_matrix.items():
            lines.append(
                f"  • If you choose {a.to_token()} and opponent chooses {b.to_token()}: "
                f"you get {pts_a} points, opponent gets {pts_b} points."
            )
        return "\n".join(lines)

    def play(self, additional_info: str, players: Sequence[Agent]) -> list[Game.Move]:
        assert len(players) == 2
        player1, player2 = players

        resp1 = self._prompt_player(player1, additional_info)
        resp2 = self._prompt_player(player2, additional_info)

        prob_distribution1 = self._parse_mix_strategy(player1, resp1)
        prob_distribution2 = self._parse_mix_strategy(player2, resp2)

        act1 = PrisonersDilemmaAction.from_token(
            self._choose_from_mix_strategy(prob_distribution1)
        )
        act2 = PrisonersDilemmaAction.from_token(
            self._choose_from_mix_strategy(prob_distribution2)
        )

        pts1, pts2 = self.payoff_matrix[(act1, act2)]
        return [
            Game.Move(name=player1.name, action=act1, points=pts1, response=resp1),
            Game.Move(name=player2.name, action=act2, points=pts2, response=resp2),
        ]

    @classmethod
    def _parse_payoff_matrix(
        cls,
        raw_payoff: dict[str, list[float]],
    ) -> dict[
        tuple[PrisonersDilemmaAction, PrisonersDilemmaAction], tuple[float, float]
    ]:
        """
        Convert a raw payoff matrix with string keys into typed action pairs.
        """
        payoffs = {}
        for key, (p1, p2) in raw_payoff.items():
            a1 = PrisonersDilemmaAction(key[0])
            a2 = PrisonersDilemmaAction(key[1])
            payoffs[(a1, a2)] = (p1, p2)
        return payoffs
