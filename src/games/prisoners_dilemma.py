from enum import Enum
from logging import Logger

from src.agent import Agent
from src.games.base import Game

class PrisonersDilemma(Game):
    """
    Prisoner's Dilemma environment that allows for one rounds of interaction
    between two LLM agents.
    """

    class Action(Enum):
        """Possible actions in the Prisoner's Dilemma"""
        COOPERATE = "C"
        DEFECT = "D"

        def __str__(self) -> str:
            return self.value

    def __init__(
        self,
        payoff_matrix: dict[str, list[float]],
        *,
        debugger: Logger | None = None,
    ):
        super().__init__(debugger=debugger)

        self.payoff_matrix = self._parse_payoff_matrix(payoff_matrix)
        self.num_players = 2  # Fixed for Prisoner's Dilemma

        self.prompt = f"""
        You are an expert called {{agent_name}} at playing Prisoner's Dilemma.

        Your goal is to maximize your score of the game.
        Both players choose simultaneously; choices are revealed after both commit.

        Actions (choose exactly one per round):
        • <Cooperate>
        • <Defect>
        IMPORTANT: Your final printed action must be wrapped by <>, for example: <Cooperate> or <Defect>.

        Payoff matrix:
        {self._payoff_description()}

        Additional information:
        Note, any of the following information related to {{agent_name}} is refering to yourself.
        {{additional_info}}
        """

    def _payoff_description(self) -> str:
        lines = []
        for (a, b), (pts_a, pts_b) in self.payoff_matrix.items():
            lines.append(
                f"  • If you={a} and opponent={b}: "
                f"you get {pts_a}, opponent gets {pts_b}."
            )
        return "\n".join(lines)

    def play(self, additional_info: str, agents: list[Agent]) -> list[Game.Move]:
        """
        Play the Iterated Prisoner's Dilemma for the specified number of rounds.
        """
        assert len(agents) == self.num_players, (
            f"Expected {self.num_players} agents, got {len(agents)}."
        )
        agent1, agent2 = agents

        response1 = agent1.chat(self.prompt.format(
            agent_name= agent1.name,
            additional_info=additional_info,
        ))
        action1 = self._parse_action(response1)

        response2 = agent2.chat(self.prompt.format(
            agent_name=agent2.name,
            additional_info=additional_info,
        ))
        action2 = self._parse_action(response2)

        pts1, pts2 = self.payoff_matrix[(action1, action2)]

        if self.debugger:
            self.debugger.info(
                "-" * 20 + "\n"
                + "Example prompt:\n"
                + self.prompt.format(
                    agent_name=agent1.name,
                    additional_info=additional_info,
                )
                + "\n" + " " * 5 + "+" * 10 + "\n"
            )
            self.debugger.info(
                f"{agent1.name} chose {action1}: {response1}\n"
                f"{agent2.name} chose {action2}: {response2}\n"
            )

        return [Game.Move(
            name=agent1.name,
            action=str(action1),
            points=pts1,
        ), Game.Move(
            name=agent2.name,
            action=str(action2),
            points=pts2,
        )]

    def _parse_action(self, response: str) -> Action:
        """
        Extract the choice action made by the LLM.
        """
        coop_token = "<Cooperate>"
        defect_token = "<Defect>"

        last_coop = response.rfind(coop_token)
        last_defect = response.rfind(defect_token)

        if last_coop == -1 and last_defect == -1:
            raise ValueError(
                "Could not find '<Cooperate>' or '<Defect>' in response:\n"
                f"{response}"
            )

        return (
            type(self).Action.COOPERATE
            if last_coop > last_defect
            else type(self).Action.DEFECT
        )

    @classmethod
    def _parse_payoff_matrix(
        cls,
        raw_payoff: dict[str, list[float]],
    ) -> dict[
        tuple[Action, Action],
        tuple[float, float]
    ]:
        """
        Convert a raw payoff matrix with string keys into typed action pairs.
        """
        payoff = {}
        for actions, reward in raw_payoff.items():
            assert len(actions) == 2, f"Invalid payoff actions: {actions}"
            a1 = cls.Action(actions[0])
            a2 = cls.Action(actions[1])
            payoff[(a1, a2)] = tuple(reward)
        return payoff
