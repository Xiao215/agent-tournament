from typing import Sequence
import math
import itertools
import random

from tqdm import tqdm

from src.agent import Agent
from src.games.base import Game, Move
from src.mechanisms.base import Mechanism
from src.evolution.population_payoffs import PopulationPayoffs

random.seed(42)

class Repetition(Mechanism):
    """
    Repetition mechanism that allows for multiple rounds of the same game.
    """
    def __init__(
        self,
        base_game: Game,
        num_rounds: int,
        discount: float,
    ) -> None:
        super().__init__(base_game)
        self.num_rounds = num_rounds

    def _parse_history(self, history: list[tuple[Move]]) -> str:
        """Parse the history of past actions as the mechanism information."""
        if not history:
            return "History: None of the players have played yet, so there is no history."

        history_str = "History:\n"
        for i, (players_moves) in enumerate(history):
            history_str = f"  Round {i + 1}: "
            for move in players_moves:
                history_str += f"{move.name}: {move.action.token}, "
            history_str = history_str[:-2]
            history_str += "\n"

        return (
            history_str.strip()
            + "\n\nNote: This game is repetitive so your chosen action "
            "will be visible to your opponents in future rounds."
        )

    def run(self, agents: Sequence[Agent]) -> PopulationPayoffs:
        """Repeat the base game for a specified number of repetitions.

        Returns:
            final_score (dict[str, float]): A dictionary mapping player names to their final scores after all rounds.
        """
        payoffs = PopulationPayoffs(agent_names=[agent.name for agent in agents])

        k = self.base_game.num_players
        n = len(agents)
        total_matches = math.comb(n, k)
        combo_iter = list(itertools.combinations(agents, k))
        random.shuffle(combo_iter)  # The order does not matter, kept just in case

        inner_tqdm_bar = tqdm(
            combo_iter,
            desc="Tournaments",
            total=total_matches,
            leave=False,
            position=1,
        )

        for players in inner_tqdm_bar:
            history = []
            score = {p.name: 0.0 for p in players}
            for i in tqdm(
                range(self.num_rounds),
                desc=f"Running Repetition Mechanism for {self.base_game.__class__.__name__}",
            ):
                repetition_information = self._parse_history(history)
                players_moves = self.base_game.play(
                    additional_info=repetition_information, players=players
                )
                history.append(players_moves)
                for move in players_moves:
                    score[move.name] += move.points

                # TBD

        return payoffs
