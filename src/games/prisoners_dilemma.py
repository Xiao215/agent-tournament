import sys
from enum import Enum
from typing import Sequence

from src.agent import Agent
from src.games.base import Game


class PrisonersDilemmaAction(Enum):
    """Possible actions in the Prisoner's Dilemma"""

    COOPERATE = "C"
    DEFECT = "D"

    def to_token(self) -> str:
        # list(Action) is ordered by definition
        idx = list(type(self)).index(self)
        return f"<A{idx}>"

    @classmethod
    def from_token(cls, token: str) -> Enum:
        """Parse an action from a token like "<A1>" or "<A2>"."""
        try:
            idx = int(token.strip("<>").lstrip("A"))
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
                f"  • If you choose {a.to_token()} and opponent chooses {b.to_token()}: "
                f"you get {pts_a} points, opponent gets {pts_b} points."
            )
        return "\n".join(lines)

    def play(self, additional_info: str, agents: Sequence[Agent]) -> list[Game.Move]:
        assert len(agents) == 2
        agent1, agent2 = agents

        resp1 = self._prompt_agent(agent1, additional_info)
        resp2 = self._prompt_agent(agent2, additional_info)

        act1 = self._parse_action(agent1, resp1)
        act2 = self._parse_action(agent2, resp2)

        pts1, pts2 = self.payoff_matrix[(act1, act2)]
        return [
            Game.Move(name=agent1.name, action=act1, points=pts1, response=resp1),
            Game.Move(name=agent2.name, action=act2, points=pts2, response=resp2),
        ]

    def _parse_action(
        self, agent: Agent, response: str, max_retries: int = 5
    ) -> PrisonersDilemmaAction:
        """
        Extract the chosen action from the LLM's response, retrying up to max_retries
        with a clarifying prompt if parsing fails.
        """
        def pick_action(text: str) -> PrisonersDilemmaAction:
            # Find all actions whose token appears at least once
            matches = [
                action for action in PrisonersDilemmaAction if action.to_token() in text
            ]
            if not matches:
                raise ValueError(f"[{agent}] No action token found in {text!r}")

            rightmost = max(matches, key=lambda act: text.rfind(act.to_token()))
            return rightmost

        try:
            return pick_action(response)
        except ValueError:
            pass

        clarification = (
            "Based on the action chosen in the original response below, output exactly one action token wrapped in angle brackets:\n"
            f"{', '.join(self.action_tokens)}\n\n"
            "Do NOT include explanations, `<think>` tags, or whitespace inside the brackets.\n"
            "Example valid response: `<A2>`\n\n"
            "Original response:\n"
            f"{response}"
        )
        for i in range(max_retries):
            response = agent.invoke(clarification + "\n\n")
            try:
                return pick_action(response)
            except ValueError:
                print(f"[{agent}] Retry {i+1} failed: {response!r}", file=sys.stderr)
                # feed back the new bad response for the next retry
                clarification = "The response still wasn't one of the exact tokens. Please output only `<A1>`, `<A2>`, etc., with no spaces."
        raise ValueError(f"All retries failed to parse action from {agent}")

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
