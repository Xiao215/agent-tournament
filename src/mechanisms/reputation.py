from logging import Logger
from collections import defaultdict

from src.mechanisms.base import Mechanism, register_mechanism
from src.games.base import Game

@register_mechanism
class Reputation(Mechanism):
    """"""
    def __init__(
            self,
            base_game: Game,
            cooperation_rate: dict[str, float],
            betrayal_rate: dict[str, float],
            logger: Logger | None = None,
        ):
        super().__init__(base_game, logger)

        # TODO: Think about how to feed the reputation easier.
        self.cooperation_rate = cooperation_rate
        # betrayal_rate represents the probability of a player defecting while opponent cooperates.
        self.betrayal_rate = betrayal_rate

    def _parse_reputation(self) -> str:
        reputation_str = (
            "Reputation:\n"
            + "\n".join(
                f"{name}: Cooperated in {rate * 100:.2f}% of the rounds, "
                f"When the opponent cooperated, "
                f"they defect in {self.betrayal_rate[name] * 100:.2f}% of the rounds."
                for name, rate in self.cooperation_rate.items()
            )
        )
        return reputation_str

    def run(self):
        """Repeat the base game for a specified number of repetitions."""
        if self.logger:
            self.logger.info(
                f"{'='*5} Reputation @ {self.base_game.__class__.__name__} {'='*5}"
            )

        final_score = defaultdict(float)
        players_moves = self.base_game.play(
            additional_info=self._parse_reputation()
        )
        for move in players_moves:
            final_score[move.name] += move.points

        if self.logger:
            self.logger.info(
                f"{'='*5} Final Score {'='*5}\n"
                + "\n".join(f"{name}: {score}" for name, score in final_score.items())
            )
        return final_score