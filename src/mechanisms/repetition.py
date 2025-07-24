from collections import defaultdict
from typing import Sequence

from tqdm import tqdm

from src.agent import Agent
from src.games.base import Game
from src.mechanisms.base import Mechanism, logger


class Repetition(Mechanism):
    """
    Repetition mechanism that allows for multiple rounds of the same game.
    """
    def __init__(
            self,
            base_game: Game,
            num_rounds: int,
        ) -> None:
        super().__init__(base_game)
        self.num_rounds = num_rounds
        self.history = []

    def _parse_history(self)-> str:
        """Parse the history of past actions as the mechanism information."""
        if not self.history:
            return "History: None of the players have played yet, so there is no history."

        history_str = "History:\n"
        for i, (players_moves) in enumerate(self.history):
            history_str = f"  Round {i + 1}: "
            for move in players_moves:
                history_str += f"{move.name}: {move.action}, "
            history_str = history_str[:-2]
            history_str += "\n"

        return (
            history_str.strip()
            + "\n\nNote: This game is repetitive so your chosen action "
            "will be visible to your opponents in future rounds."
        )

    def run(self, agents: Sequence[Agent]) -> dict[str, float]:
        """Repeat the base game for a specified number of repetitions.

        Returns:
            final_score (dict[str, float]): A dictionary mapping player names to their final scores after all rounds.
        """
        final_score = defaultdict(float)
        for _ in tqdm(
            range(self.num_rounds),
            desc=f"Running Repetition Mechanism for {self.base_game.__class__.__name__}"
        ):
            repetition_information = self._parse_history()
            players_moves = self.base_game.play(
                additional_info=repetition_information, agents=agents
            )
            self.history.append(players_moves)
            for move in players_moves:
                final_score[move.name] = final_score[move.name] + move.points

            logger.info(
                "%s Repetition Info %s\n%s\n\nMoves:\n%s\n",
                "=" * 10,
                "=" * 10,
                repetition_information.strip(),
                "\n\n".join(
                    f"\t{move.name} → Action: {move.action}, Points: {move.points}\n\t\t"
                    + move.response.replace("\n", "\n\t\t")
                    for move in players_moves
                ),
            )

        return final_score
