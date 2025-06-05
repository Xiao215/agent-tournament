from enum import Enum
import logging

from tqdm import tqdm

from src.agent import Agent
from src.games.base import Game


class PDAction(Enum):
    """Possible actions in the Prisoner's Dilemma"""
    COOPERATE = "C"
    DEFECT = "D"

    def __str__(self) -> str:
        return self.value


class IteratedPrisonersDilemma(Game):
    """
    Iterated Prisoner's Dilemma environment that runs multiple rounds
    between two LLM agents.
    """
    def __init__(
        self,
        payoff_matrix: dict[str, list[float]],
        *,
        agent1: Agent,
        agent2: Agent,
        num_rounds: int,
        logger: str = logging.Logger | None,
    ):
        self.payoff_matrix = self._parse_payoff_matrix(payoff_matrix)
        self.agent1 = agent1
        self.agent2 = agent2
        self.num_rounds = num_rounds

        self.actions_p1 = []
        self.actions_p2 = []
        self.scores_p1 = []
        self.scores_p2 = []

        self.prompt = f"""
        You are an expert at playing Iterated Prisoner's Dilemma.

        Round {{current_round}} of {self.num_rounds}
        Your goal is to maximize your cumulative score over {self.num_rounds} rounds.
        Both players choose simultaneously; choices are revealed after both commit.

        Actions (choose exactly one per round):
        • <Cooperate>
        • <Defect>
        IMPORTANT: Your final printed action must be wrapped by <>, for example: <Cooperate> or <Defect>.

        Payoff matrix:
        {self._payoff_description()}

        Previous rounds:
        {{past_history}}
        """

        self.logger = logger
        if self.logger:
            self.logger.info(
                f"{'='*10} {self.num_rounds} rounds Iterated Prisoner's Dilemma {'='*10}\n"
                f"{'='*10} {agent1} vs. {agent2} {'='*10}\n\n"
            )

    def _payoff_description(self) -> str:
        lines = []
        for (a, b), (pts_a, pts_b) in self.payoff_matrix.items():
            lines.append(
                f"  • If you={a} and opponent={b}: "
                f"you get {pts_a}, opponent gets {pts_b}."
            )
        return "\n".join(lines)

    def _format_history(self, identity: Agent) -> str:
        if not self.actions_p1:
            return "  (no previous rounds)"
        lines = []
        if identity == self.agent1:
            my_actions, opp_actions = self.actions_p1, self.actions_p2
            my_scores,  opp_scores  = self.scores_p1,  self.scores_p2
        else:
            my_actions, opp_actions = self.actions_p2, self.actions_p1
            my_scores,  opp_scores  = self.scores_p2,  self.scores_p1

        lines = [
            f"  Round {i}: You={act}, Opponent={opp_act}"
            for i, (act, opp_act) in enumerate(zip(my_actions, opp_actions), start=1)
        ]
        lines.append(f"  Your total score: {sum(my_scores)}")
        lines.append(f"  Opponent's total score: {sum(opp_scores)}")

        return "\n".join(lines)

    def play(self):
        """
        Play the Iterated Prisoner's Dilemma for the specified number of rounds.
        """
        for round_num in tqdm(
            range(1, self.num_rounds + 1),
            desc="Playing Iterated Prisoner's Dilemma"
        ):
            response1 = self.agent1.chat(self.prompt.format(
                current_round=round_num,
                past_history=self._format_history(self.agent1),
            ))

            action1 = self._parse_action(response1)

            response2 = self.agent2.chat(self.prompt.format(
                current_round=round_num,
                past_history=self._format_history(self.agent2),
            ))
            action2 = self._parse_action(response2)

            pts1, pts2 = self.payoff_matrix[(action1, action2)]
            self.actions_p1.append(action1)
            self.actions_p2.append(action2)
            self.scores_p1.append(pts1)
            self.scores_p2.append(pts2)

            # logging the round details
            if self.logger:
                self.logger.info(f"{'-'*10} Round {round_num} {'-'*10}")

                hist1 = [str(a) for a in self.actions_p1[:-1]]
                self.logger.info(f"    Past history for {self.agent1}: {hist1}")
                hist2 = [str(a) for a in self.actions_p2[:-1]]
                self.logger.info(f"    Past history for {self.agent2}: {hist2}")

                self.logger.info(f"        {self.agent1}: {response1}")
                self.logger.info(f"        {self.agent2}: {response2}")
                self.logger.info("")

        if self.logger:
            self.logger.info(f"{'='*10} Final Scores {'='*10}")
            self.logger.info(f"    {self.agent1}: {sum(self.scores_p1)}")
            self.logger.info(f"    {self.agent2}: {sum(self.scores_p2)}")
            self.logger.info(f"{'='*30}\n")


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
