from logging import Logger
from collections import defaultdict
from tqdm import tqdm

from src.mechanisms.base import Mechanism, register_mechanism
from src.games.base import Game

@register_mechanism
class Repetition(Mechanism):
    """"""
    def __init__(
            self,
            base_game: Game,
            num_rounds: int,
            logger: Logger | None = None
        ) -> None:
        super().__init__(base_game, logger)
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
                history_str += f"{move.name}: {move.value}, "
            history_str = history_str[:-2]
            history_str += "\n"

        return history_str.strip()


    def run(self) -> dict[str, float]:
        """Repeat the base game for a specified number of repetitions.

        Returns:
            final_score (dict[str, float]): A dictionary mapping player names to their final scores after all rounds.
        """
        if self.logger:
            self.logger.info(
                f"{'='*5} Repetition ({self.num_rounds}) @ {self.base_game.__class__.__name__} {'='*5}"
            )

        final_score = defaultdict(float)
        for _ in tqdm(
            range(self.num_rounds),
            desc=f"Running Repetition Mechanism for {self.base_game.__class__.__name__}"
        ):
            players_moves = self.base_game.play(
                additional_info=self._parse_history()
            )
            self.history.append(players_moves)
            for move in players_moves:
                final_score[move.name] = final_score[move.name] + move.points

        if self.logger:
            self.logger.info(
                f"{'='*5} Final Score {'='*5}\n"
                + "\n".join(f"{name}: {score}" for name, score in final_score.items())
            )

        return final_score