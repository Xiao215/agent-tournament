from logging import Logger
from abc import ABC, abstractmethod
from collections import defaultdict

from src.mechanisms.base import Mechanism
from src.games.base import Game
from src.games.prisoners_dilemma import PrisonersDilemma
from src.agent import Agent

class Reputation(Mechanism, ABC):
    """
    Reputation mechanism that makes each players' reputation visible to all players.
    """
    def __init__(
            self,
            base_game: Game,
            logger: Logger | None = None,
        ):
        super().__init__(base_game, logger)

    def run(self, agents: list[Agent]) -> dict[str, float]:
        """Repeat the base game for a specified number of repetitions."""
        if self.logger:
            self.logger.info(
                f"{'='*5} Reputation @ {self.base_game.__class__.__name__} {'='*5}"
            )

        players_moves = self.base_game.play(
            additional_info=self._parse_reputation(agents),
            agents=agents
        )
        self._update_reputation(players_moves)

        final_score = defaultdict(float)
        for move in players_moves:
            final_score[move.name] += move.points

        if self.logger:
            self.logger.info(
                f"{'='*5} Final Score {'='*5}\n"
                + "\n".join(f"{name}: {score}" for name, score in final_score.items())
            )
        return final_score

    @abstractmethod
    def _update_reputation(self, players_moves: list[Game.Move]):
        """Update the reputation based on players' moves."""
        raise NotImplementedError("`_update_reputation` should be implemented in subclasses.")

    @abstractmethod
    def _parse_reputation(self, agents: list[Agent]) -> str:
        """Parse the reputation information into a string."""
        raise NotImplementedError("`_parse_reputation` should be implemented in subclasses.")

class ReputationPrisonersDilemma(Reputation):
    """
    Reputation mechanism for the Prisoner's Dilemma game.
    This mechanism tracks the cooperation and betrayal rates of players.
    """
    # A global static variable to store cooperation and betrayal rates
    # Mapping player names to tuples of (event occur count, total event count)
    cooperation_rate: defaultdict[str, list[int, int]] = defaultdict(lambda: [0, 0])
    betrayal_rate: defaultdict[str, list[int, int]] = defaultdict(lambda: [0, 0])

    def __init__(
            self,
            base_game: Game,
            logger: Logger | None = None,
        ):
        super().__init__(base_game, logger)
        if not isinstance(self.base_game, PrisonersDilemma):
            raise TypeError(
                f"ReputationPrisonersDilemma can only be used with Prisoner's Dilemma games, "
                f"but got {self.base_game.__class__.__name__}."
            )

    def _parse_reputation(self, agents: list[Agent]) -> str:
        """Parse the reputation information of the given agents into a string."""
        lines = []
        for agent in agents:
            name = agent.name

            # Look up coop history
            coop_entry = type(self).cooperation_rate.get(name)
            betray_entry = type(self).betrayal_rate.get(name)

            if coop_entry is None and betray_entry is None:
                lines.append(f"{name}: no reputation yet")
                continue

            parts = []

            if coop_entry is not None:
                coop_count, total_count = coop_entry
                coop_pct = coop_count / total_count * 100
                parts.append(f"Cooperated in {coop_count}/{total_count} rounds ({coop_pct:.2f}%)")

            if betray_entry is not None:
                betray_count, opp_coop_count = betray_entry
                betray_pct = betray_count / opp_coop_count * 100
                parts.append(
                    f"When opponent cooperated, betrayed in "
                    f"{betray_count}/{opp_coop_count} rounds ({betray_pct:.2f}%)"
                )

            lines.append(f"{name}: " + "; ".join(parts))

        return "Reputation:\n" + "\n".join(lines)

    def _update_reputation(self, players_moves: list[Game.Move]):
        for i, move in enumerate(players_moves):
            name = move.name
            opp = players_moves[1 - i]

            # 1) Update cooperation counts
            coop_count, total_count = type(self).cooperation_rate[name]
            total_count += 1
            if move.action == self.base_game.Action.COOPERATE.value:
                coop_count += 1
            type(self).cooperation_rate[name] = [coop_count, total_count]

            if opp.action == self.base_game.Action.COOPERATE.value:
                betray_count, opp_coop_count = type(self).betrayal_rate[name]
                opp_coop_count += 1
                if move.action == self.base_game.Action.DEFECT.value:
                    betray_count += 1
                type(self).betrayal_rate[name] = [betray_count, opp_coop_count]

class ReputationPublicGoods(Reputation):
    """
    Reputation mechanism for the Prisoner's Dilemma game.
    This mechanism tracks the cooperation and betrayal rates of players.
    """

    # Mapping player names to tuples of (average normalized contribution, total game played
    # contributes a consistent fraction of what they could.
    # normalized_contribution: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    # #
    # generosity: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0])


    # def _parse_reputation(self) -> str:
    #     lines = []
    #     for name, (coop_count, total_count) in type(self).cooperation_rate.items():
    #         coop_pct = (coop_count / total_count * 100) if total_count else 0.0

    #         betray_count, opp_coop_count = type(self).betrayal_rate[name]
    #         betray_pct = (betray_count / opp_coop_count * 100) if opp_coop_count else 0.0

    #         lines.append(
    #             f"{name}: Cooperated in {coop_pct:.2f}% of rounds; "
    #             f"when opponent cooperated, betrayed in {betray_pct:.2f}% of those rounds"
    #         )

    #     return "Reputation:\n" + "\n".join(lines)

    # def _update_reputation(self, players_moves: list[Game.Move]):
    #     for i, move in enumerate(players_moves):
    #         name = move.name
    #         opp = players_moves[1 - i]

    #         # 1) Update cooperation counts
    #         coop_count, total_count = type(self).cooperation_rate[name]
    #         total_count += 1
    #         if move.action == self.base_game.Action.COOPERATE:
    #             coop_count += 1
    #         type(self).cooperation_rate[name] = [coop_count, total_count]

    #         # 2) If opponent cooperated, update betrayal counts
    #         if opp.action == self.base_game.Action.COOPERATE:
    #             betray_count, opp_coop_count = type(self).betrayal_rate[name]
    #             opp_coop_count += 1
    #             if move.action == self.base_game.Action.DEFECT:
    #                 betray_count += 1
    #             type(self).betrayal_rate[name] = [betray_count, opp_coop_count]