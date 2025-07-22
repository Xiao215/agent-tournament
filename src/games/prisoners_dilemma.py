from enum import Enum
from typing import Sequence

from src.agent import Agent
from src.games.base import Game


class PrisonersDilemma(Game):
    """
    Prisoner's Dilemma environment that allows for one rounds of interaction
    between two LLM agents.
    """

    class Action(Enum):
        """Possible actions in the Prisoner's Dilemma"""
        COOPERATE = "A1"
        DEFECT = "A2"

        def __str__(self) -> str:
            return self.value

        @property
        def token(self) -> str:
            """The exact string we expect the model to emit."""
            return f"<{self.value}>"

        @classmethod
        def from_token(cls, token: str) -> Enum:
            """Convert '<A1>' → Action.COOPERATE, '<A2>' → Action.DEFECT."""
            alias = token.strip("<>")
            try:
                return cls(alias)
            except ValueError as e:
                raise ValueError(f"Unknown action token {token!r}") from e

        @property
        def symbol(self) -> str:
            """The single character symbol for the action, e.g. "C" for COOPERATE, "D" for DEFECT."""
            return self.name[0]

        @classmethod
        def from_symbol(cls, sym: str) -> Enum:
            """Convert 'C' → Action.COOPERATE, 'D' → Action.DEFECT."""
            for member in cls:
                if member.symbol == sym:
                    return member
            raise KeyError(f"Unknown symbol {sym!r}")

    def __init__(
        self,
        payoff_matrix: dict[str, list[float]],
    ) -> None:
        self.payoff_matrix = self._parse_payoff_matrix(payoff_matrix)

        self.action_tokens = [act.token for act in self.Action]
        actions_block = "\n\t".join(
            [f"\t• {act_tok}" for act_tok in self.action_tokens]
        )

        prompt = f"""
        You are a player named {{agent_name}} who plays in a decision-making game.
        Your objective is to maximise your cumulative score in the game described below.

        Rule:
        1. You and the opponent each choose exactly one action, simultaneously.
        2. After both commitments are locked in, the two actions are revealed.
        3. Both players receive the points specified in the payoff table below.

        Actions (choose exactly one per round):
        {actions_block}
        IMPORTANT: Your final printed action must be wrapped by <>, for example: <A1>.

        Payoff matrix:
        {self._payoff_description()}

        Additional information:
        Note, any of the following information related to {{agent_name}} is refering to yourself.
        {{additional_info}}
        """

        super().__init__(
            prompt=prompt,
            num_players=2
        )

    def _payoff_description(self) -> str:
        lines = []
        for (a, b), (pts_a, pts_b) in self.payoff_matrix.items():
            lines.append(
                f"  • If you choose {a} and opponent chooses {b}: "
                f"you get {pts_a} points, opponent gets {pts_b} points."
            )
        return "\n".join(lines)

    def play(self, additional_info: str, agents: Sequence[Agent]) -> list[Game.Move]:
        assert len(agents) == 2
        agent1, agent2 = agents

        resp1, act1 = self._chat_and_parse(agent1, additional_info)
        resp2, act2 = self._chat_and_parse(agent2, additional_info)

        pts1, pts2 = self.payoff_matrix[(act1, act2)]
        return [
            Game.Move(
                name=str(agent1),
                action=str(act1),
                points=pts1,
                response=resp1
            ),
            Game.Move(
                name=str(agent2),
                action=str(act2),
                points=pts2,
                response=resp2
            ),
        ]

    def _parse_action(self, agent: Agent, response: str) -> Enum:
        """
        Extract the choice action made by the LLM.
        """
        def pick_action(text: str) -> Enum:
            matches = [action for action in self.Action if action.token in text]
            if len(matches) == 1:
                return matches[0]
            if not matches:
                raise ValueError(f"No action token found in {text!r}")
            raise ValueError(f"Multiple action tokens {matches!r} in {text!r}")

        try:
            # first regex match
            return pick_action(response)
        except ValueError:
            # build a clarifying prompt
            clarification = (
                "According to the original response below, what is the action chosen?\n"
                "Please ONLY reply with **exactly one** of the following actions, wrapped in angle brackets:\n"
                f"{', '.join(self.action_tokens)}\n"
                + "\nDo not include any other text."
            )
            clarified = agent.invoke(clarification + "\n\nOriginal response:\n" + response)
            return pick_action(clarified)

    @classmethod
    def _parse_payoff_matrix(
        cls,
        raw_payoff: dict[str, list[float]],
    ) -> dict[tuple[Enum, Enum], tuple[float, float]]:
        """
        Convert a raw payoff matrix with string keys into typed action pairs.
        """
        payoffs = {}
        for sym_pair, (p1, p2) in raw_payoff.items():
            # e.g. "C","D"
            a_sym, b_sym = sym_pair
            a = cls.Action.from_symbol(a_sym)
            b = cls.Action.from_symbol(b_sym)
            payoffs[(a, b)] = (p1, p2)
        return payoffs
