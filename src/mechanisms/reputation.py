from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from src.agent import Agent
from src.games.base import Game
from src.games.prisoners_dilemma import (PrisonersDilemma,
                                         PrisonersDilemmaAction)
from src.games.public_goods import PublicGoods
from src.mechanisms.base import Mechanism


class ReputationStat:
    def __init__(self):
        # reputation[metric] = (positive_count, total_count)
        self.scores: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))

    def record(self, metric: str, success: bool) -> None:
        """Record one trial for `metric`."""
        pos, tot = self.scores[metric]
        tot += 1
        pos += int(success)
        self.scores[metric] = (pos, tot)

    def rate(self, metric: str) -> float | None:
        """Return success rate or None if no trials."""
        pos, tot = self.scores.get(metric, (0, 0))
        return (pos / tot) if tot else None

    def stat(self, metric: str) -> tuple[int, int]:
        """Return the raw counts for the metric."""
        return self.scores[metric]

    def all_rates(self) -> dict[str, float | None]:
        """Return all rates under the agent."""
        return {m: (p / t if t else None) for m, (p, t) in self.scores.items()}


@dataclass(frozen=True)
class Record:
    """A record of a player's move in a game with reputation."""

    name: str
    action: Enum
    points: float
    response: str
    reputation: ReputationStat

    def to_dict(self) -> dict:
        """
        Convert the record to a dictionary for serialization.
        """
        return {
            "name": self.name,
            "action": self.action.name,
            "points": self.points,
            "response": self.response,
            "reputation": self.reputation.all_rates(),
        }


class Reputation(Mechanism, ABC):
    """
    Reputation mechanism that makes each players' reputation visible to all players.
    """

    def __init__(self, base_game: Game):
        super().__init__(base_game)
        self.reputation: dict[str, ReputationStat] = defaultdict(ReputationStat)

    def run(self, agents: Sequence[Agent]):
        """Repeat the base game for a specified number of repetitions."""
        reputation_information = self._format_reputation(agents)
        print(reputation_information)
        players_moves = self.base_game.play(
            additional_info=reputation_information, agents=agents
        )

        match_record = [
            Record(
                name=move.name,
                action=move.action,
                points=move.points,
                response=move.response,
                reputation=self.reputation[move.name],
            ).to_dict()
            for move in players_moves
        ]

        return match_record

    @abstractmethod
    def _format_reputation(self, agents: Sequence[Agent]) -> str:
        """Format the reputation information into a string."""
        raise NotImplementedError(
            "`_format_reputation` should be implemented in subclasses."
        )


class ReputationPrisonersDilemma(Reputation):
    """
    Reputation mechanism for the Prisoner's Dilemma game.
    This mechanism tracks the cooperation rates of players.
    """
    def __init__(
        self,
        base_game: PrisonersDilemma,
    ):
        super().__init__(base_game)
        if not isinstance(self.base_game, PrisonersDilemma):
            raise TypeError(
                f"ReputationPrisonersDilemma can only be used with Prisoner's Dilemma games, "
                f"but got {self.base_game.__class__.__name__}."
            )

    def _format_reputation(self, agents: Sequence[Agent]) -> str:
        """Format the reputation information of the given agents into a string."""
        lines = []
        coop_tok = PrisonersDilemmaAction.COOPERATE.to_token()

        for agent in agents:
            name = agent.name
            agent_reputation = self.reputation[name]

            coop_rate = agent_reputation.rate("cooperation_rate")

            # Initial reputation information at game start
            if coop_rate is None:
                lines.append(f"{name}: No reputation data available.")
                continue
            else:
                coop_count, total_count = agent_reputation.stat("cooperation_rate")
                coop_pct = coop_rate
                lines.append(
                    f"They played {coop_tok} in {coop_count}/{total_count} rounds ({coop_pct:.2%})"
                )
        lines = [f"\n\t{line}" for line in lines]
        return (
            "\nReputation:"
            + "".join(lines)
            + "\n\tNote: Your chosen action will affect your reputation score."
        )

    def post_tournament(self, match_records: list[list[dict]]) -> None:
        """Update the global reputation rates based on the live rates."""
        for record in match_records:
            # For prisoner dilemma, only two players are involved in each match.
            for player in record:
                player_name = player["name"]

                player_stat = self.reputation[player_name]
                player_stat.record(
                    "cooperation_rate",
                    player["action"] == PrisonersDilemmaAction.COOPERATE.name,
                )


class ReputationPublicGoods(Reputation):
    """
    Reputation mechanism for the Public Goods game.
    This mechanism tracks the contribution rates of players.
    """

    def __init__(self, base_game: Game):
        super().__init__(base_game)
        if not isinstance(self.base_game, PublicGoods):
            raise TypeError(
                f"ReputationPublicGoods can only be used with PublicGoodsGame, "
                f"but got {self.base_game.__class__.__name__}"
            )
