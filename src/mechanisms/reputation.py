from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Sequence

from src.agent import Agent
from src.games.base import Game
from src.games.prisoners_dilemma import PrisonersDilemma
from src.mechanisms.base import Mechanism, logger


class Reputation(Mechanism, ABC):
    """
    Reputation mechanism that makes each players' reputation visible to all players.
    """

    def run(self, agents: Sequence[Agent]) -> dict[str, float]:
        """Repeat the base game for a specified number of repetitions."""
        reputation_information = self._parse_reputation(agents)
        players_moves = self.base_game.play(
            additional_info=reputation_information,
            agents=agents
        )
        self._update_reputation(players_moves)

        final_score = defaultdict(float)
        for move in players_moves:
            final_score[move.name] += move.points

        logger.info(
            "%s Reputation Info %s\n%s\n\nMoves:\n%s\n",
            "=" * 10,
            "=" * 10,
            reputation_information.strip(),
            "\n\n".join(
                f"\t{move.name} → Action: {move.action}, Points: {move.points}\n\t\t"
                + move.response.replace("\n", "\n\t\t")
                for move in players_moves
            ),
        )

        return final_score

    @abstractmethod
    def _update_reputation(self, players_moves: list[Game.Move]):
        """Update the reputation based on players' moves."""
        raise NotImplementedError("`_update_reputation` should be implemented in subclasses.")

    @abstractmethod
    def _parse_reputation(self, agents: Sequence[Agent]) -> str:
        """Parse the reputation information into a string."""
        raise NotImplementedError("`_parse_reputation` should be implemented in subclasses.")


class ReputationPrisonersDilemma(Reputation):
    """
    Reputation mechanism for the Prisoner's Dilemma game.
    This mechanism tracks the cooperation and betrayal rates of players.
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

        # Mapping player names to tuples of (event occur count, total event count)
        self.coop_rate: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.betray_rate: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])

        self.coop_rate_live: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.betray_rate_live: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])

    def _parse_reputation(self, agents: Sequence[Agent]) -> str:
        """Parse the reputation information of the given agents into a string."""
        lines = []
        for agent in agents:
            name = str(agent)

            coop_entry = self.coop_rate.get(name)
            betray_entry = self.betray_rate.get(name)

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

                coop_tok = PrisonersDilemma.Action.COOPERATE.token
                defect_tok = PrisonersDilemma.Action.DEFECT.token
                parts.append(
                    f"Of the times opponent played {coop_tok}, they replied with "
                    f"{defect_tok} in {betray_count}/{opp_coop_count} "
                    f"rounds ({betray_pct:.2f}%)."
                )

            lines.append(f"{name}: " + "; ".join(parts))
        lines = [f"\n\t{line}" for line in lines]
        return (
            "\nReputation:"
            + "".join(lines)
            + "\n\tNote: Your chosen action will affect your reputation score."
        )

    def _update_reputation(self, players_moves: list[Game.Move]) -> None:
        for i, move in enumerate(players_moves):
            name = move.name
            opp = players_moves[1 - i]

            coop_count, total_count = self.coop_rate_live[name]
            total_count += 1
            if move.action == PrisonersDilemma.Action.COOPERATE.value:
                coop_count += 1
            self.coop_rate_live[name] = [coop_count, total_count]

            if opp.action == PrisonersDilemma.Action.COOPERATE.value:
                betray_count, opp_coop_count = self.betray_rate_live[name]
                opp_coop_count += 1
                if move.action == PrisonersDilemma.Action.DEFECT.value:
                    betray_count += 1
                self.betray_rate_live[name] = [betray_count, opp_coop_count]

    def post_tournament(self, history) -> None:
        """Update the global reputation rates based on the live rates."""
        self.coop_rate = self.coop_rate_live.copy()
        self.betray_rate = self.betray_rate_live.copy()


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
