from abc import ABC, abstractmethod

from src.agent import Agent
from src.action import Action

class Game(ABC):
    # @abstractmethod
    # def get_game(self):
    #     """Log the meta info of the game"""
    #     pass

    @abstractmethod
    def reset(self):
        """Reset internal state to run again."""
        pass

class IteratedPrisonersDilemma(Game):
    def __init__(
        self,
        payoff_matrix: dict[tuple[Action, Action], tuple[float, float]],
        num_rounds: int,
    ):
        self.num_rounds = num_rounds
        self.payoff_matrix = payoff_matrix

        # Since agents' initialization depends on env, we will use enroll_agents to initialize agents later.
        self.agent1 = None
        self.agent2 = None

        self.history_p1 = []
        self.history_p2 = []

        self.total_score_p1 = 0
        self.total_score_p2 = 0

    def enroll_agents(
        self,
        agent1: Agent,
        agent2: Agent
    ):
        self.reset()
        self.agent1 = agent1
        self.agent2 = agent2

    def play_PD(self):
        """Play one round of Prisoners Dilemma"""
        action1 = self.agent1.play(self.history_p1, self.history_p2, reason=False)
        action2 = self.agent2.play(self.history_p2, self.history_p1, reason=False)

        score1, score2 = self.payoff_matrix[(action1, action2)]

        self.history_p1.append(action1.value)
        self.history_p2.append(action2.value)

        self.total_score_p1 += score1
        self.total_score_p2 += score2

        # self.history.append(((action1, action2), (score1, score2)))

    def reset(self):
        self.agent1 = None
        self.agent2 = None

        self.history_p1 = []
        self.history_p2 = []

        self.total_score_p1 = 0
        self.total_score_p2 = 0

    def get_rule(self) -> str:
        def format_key(key: tuple[Action, Action]) -> str:
            return f'If you choose "{key[0].value}" and your opponent chooses "{key[1].value}"'

        game_rules = f"""
        You are an AI assistant with expertise in strategic thinking. You will play against an opponent on Iterated Prisoner's Dilemma.

        Game Rules:
        - In each round, you and your opponent will each choose one action: "C" (Cooperate) or "D" (Defect).
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

        for epoch in range(self.num_rounds):
            self.play_PD()

        return self.total_score_p1, self.total_score_p2