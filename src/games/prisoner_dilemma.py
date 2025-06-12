from enum import Enum
from logging import Logger

from src.agent import Agent
from src.games.base import Game, Move, register_game

class PDAction(Enum):
    """Possible actions in the Prisoner's Dilemma"""
    COOPERATE = "C"
    DEFECT = "D"

    def __str__(self) -> str:
        return self.value

@register_game
class PrisonersDilemma(Game):
    num_players = 2

    """
    Prisoner's Dilemma environment that allows for one rounds of interaction
    between two LLM agents.
    """
    def __init__(
        self,
        payoff_matrix: dict[str, list[float]],
        agents: list[Agent],
        *,
        debugger: Logger | None = None,
    ):
        super().__init__(debugger=debugger, agents=agents)

        self.payoff_matrix = self._parse_payoff_matrix(payoff_matrix)

        self.prompt = f"""
        You are an expert called '{{agent_name}}' at playing Iterated Prisoner's Dilemma.

        Your goal is to maximize your score of the game.
        Both players choose simultaneously; choices are revealed after both commit.

        Actions (choose exactly one per round):
        • <Cooperate>
        • <Defect>
        IMPORTANT: Your final printed action must be wrapped by <>, for example: <Cooperate> or <Defect>.

        Payoff matrix:
        {self._payoff_description()}

        Additional information:
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

    def play(self, additional_info: str) -> list[Move]:
        """
        Play the Iterated Prisoner's Dilemma for the specified number of rounds.
        """
        agent1 = self.agents[0]
        agent2 = self.agents[1]

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
                f"{agent1.name} chose {action1}: {response1}\n"
                f"{agent2.name} chose {action2}: {response2}\n"
            )

        return [Move(
            name=agent1.name,
            value=action1,
            points=pts1,
        ), Move(
            name=agent2.name,
            value=action2,
            points=pts2,
        )]

    def _parse_action(self, response: str) -> PDAction:
        """
        Extract the choice action made by the LLM.
        """
        coop_token = "<Cooperate>"
        defect_token = "<Defect>"

        last_coop = response.rfind(coop_token)
        last_defect = response.rfind(defect_token)

        if last_coop == -1 and last_defect == -1:
            raise ValueError(f"Could not find '<Cooperate>' or '<Defect>' in response:\n{response}")

        return PDAction.COOPERATE if last_coop > last_defect else PDAction.DEFECT

    @staticmethod
    def _parse_payoff_matrix(
        raw_payoff: dict[str, list[float]],
    ) -> dict[tuple[PDAction, PDAction], tuple[float, float]]:
        payoff = {}
        for actions, reward in raw_payoff.items():
            assert len(actions) == 2, f"Invalid payoff actions: {actions}"
            a1 = PDAction(actions[0])
            a2 = PDAction(actions[1])
            payoff[(a1, a2)] = tuple(reward)
        return payoff
