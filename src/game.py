import os
from abc import ABC, abstractmethod
from datetime import datetime

from tqdm import tqdm

from src.agent import Agent
from src.action import Action
from src.plot import plot_ipd_results
from config import OUTPUTS_DIR

class Game(ABC):
    @abstractmethod
    def reset(self):
        """Reset internal state to run again."""
        pass

class IteratedPrisonersDilemma(Game):
    def __init__(
        self,
        payoff_matrix: dict[tuple[Action, Action], tuple[float, float]],
        num_rounds: int,
        log: bool,
        plot: bool = True
    ):
        self.num_rounds = num_rounds
        self.payoff_matrix = payoff_matrix
        self.log = log
        self.plot = plot
        self.log_file = None

        # Since agents' initialization depends on env, we will use enroll_agents to initialize agents later.
        self.agent1 = None
        self.agent2 = None

        self.actions_p1 = []
        self.actions_p2 = []

        self.scores_p1 = []
        self.scores_p2 = []

    def enroll_agents(
        self,
        agent1: Agent,
        agent2: Agent
    ):
        self.reset()
        self.agent1 = agent1
        self.agent2 = agent2

        if self.log:
            now = datetime.now()
            date_path = now.strftime("%Y/%m/%d")
            time_stamp = now.strftime("%H-%M")
            dir_path = OUTPUTS_DIR / date_path
            os.makedirs(dir_path, exist_ok=True)
            self.log_file = open(dir_path / f"{self.agent1} vs {self.agent2} - {time_stamp}.txt", "w", encoding="utf-8")
            self.log_file.write(f"Game Log: {self.agent1} vs {self.agent2}\nStart Time: {now}\n\n")

    def play_PD(self):
        """Play one round of Prisoners Dilemma"""
        action1, response1 = self.agent1.play(self.actions_p1, self.actions_p2)
        action2, response2 = self.agent2.play(self.actions_p2, self.actions_p1)

        score1, score2 = self.payoff_matrix[(action1, action2)]

        self.actions_p1.append(action1.value)
        self.actions_p2.append(action2.value)

        self.scores_p1.append(score1)
        self.scores_p2.append(score2)

        if self.log and self.log_file:
            round_num = len(self.actions_p1)
            self.log_file.write(f"Round {round_num}:\n")
            self.log_file.write(f"\t{self.agent1} played {action1.value} with response: \n\t\t{response1}\n")
            self.log_file.write(f"\t{self.agent2} played {action2.value} with response: \n\t\t{response2}\n")
            self.log_file.write("\n")

    def get_rule(self) -> str:
        def format_key(key: tuple[Action, Action]) -> str:
            return f'If you choose {key[0].value} and your opponent chooses {key[1].value}'

        game_rules = """
        You are an AI assistant with expertise in strategic thinking. You will play against an opponent on Iterated Prisoner's Dilemma.

        Game Rules:
        - In each round, you and your opponent will each choose one action: C (Cooperate) or D (Defect).
        - The game is played over multiple rounds. You can base your decision on the full history of past actions.
        - The payoff matrix is as follows:
        """
        for key, (p1, p2) in self.payoff_matrix.items():
            game_rules += f'\n    - {format_key(key)}: you get {p1} point(s), your opponent gets {p2}.'

        game_rules += """
        - Your goal is to maximize your total points over all rounds.
        """
        return game_rules.strip()

    def simulate(self) -> tuple[float, float]:
        assert self.agent1 is not None or self.agent2 is not None, "Agents not initialized"

        for _ in tqdm(range(self.num_rounds), desc=f'{self.agent1}&{self.agent2}'):
            self.play_PD()

        final_score_p1 = sum(self.scores_p1)
        final_score_p2 = sum(self.scores_p2)

        if self.log and self.log_file:
            self.log_file.write("Final Score:\n")
            self.log_file.write(f"  {self.agent1}: {final_score_p1}\n")
            self.log_file.write(f"  {self.agent2}: {final_score_p2}\n")
            self.log_file.close()

        if self.plot:
            plot_ipd_results(self)

        return final_score_p1, final_score_p2


    def reset(self):
        self.agent1 = None
        self.agent2 = None

        self.actions_p1 = []
        self.actions_p2 = []

        self.scores_p1 = []
        self.scores_p2 = []
