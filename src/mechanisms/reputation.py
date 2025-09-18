import itertools
import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence

from tqdm import tqdm

from src.agents.agent_manager import Agent
from src.evolution.population_payoffs import PopulationPayoffs
from src.games.base import Game
from src.games.prisoners_dilemma import PrisonersDilemma, PrisonersDilemmaAction
from src.games.public_goods import PublicGoods, PublicGoodsAction
from src.logger_manager import LOGGER
from src.mechanisms.base import RepetitiveMechanism

random.seed(42)


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


class Reputation(RepetitiveMechanism, ABC):
    """
    Reputation mechanism that makes each players' reputation visible to all players.
    """

    def __init__(self, base_game: Game, num_rounds: int, discount: float):
        super().__init__(base_game, num_rounds, discount)
        self.reputation: dict[str, ReputationStat] = defaultdict(ReputationStat)
        self.matchup_workers = 1

    @abstractmethod
    def _format_reputation(self, agents: Sequence[Agent]) -> str:
        """Format the reputation information into a string."""
        raise NotImplementedError(
            "`_format_reputation` should be implemented in subclasses."
        )

    def run_tournament(self, agents: Sequence[Agent]) -> PopulationPayoffs:
        """Run the mechanism over the base game across all players."""
        payoffs = self._build_payoffs(agent_names=[agent.name for agent in agents])

        for _ in tqdm(
            range(self.num_rounds),
            desc=f"Running Reputation Mechanism for {self.base_game.__class__.__name__}",
        ):
            self._play_matchup(agents, payoffs)
        return payoffs

    def _play_matchup(
        self, players: Sequence[Agent], payoffs: PopulationPayoffs
    ) -> None:
        k = self.base_game.num_players
        n = len(players)
        total_matches = math.comb(n, k)
        combo_iter = list(itertools.combinations_with_replacement(players, k))
        random.shuffle(combo_iter)

        def play_one(lineup: Sequence[Agent]) -> tuple[list[dict], PopulationPayoffs]:
            local_payoffs = self._build_payoffs(
                agent_names=[agent.name for agent in players]
            )
            moves = self.base_game.play(
                additional_info=self._format_reputation(lineup), players=lineup
            )
            local_payoffs.add_profile(moves)
            return [move.to_dict() for move in moves], local_payoffs

        moves_per_round: list[list[dict]] = []

        if self.matchup_workers <= 1 or len(combo_iter) <= 1:
            for lineup in tqdm(
                combo_iter,
                total=total_matches,
                leave=False,
                position=1,
                desc="Reputation matches",
            ):
                moves_dicts, local = play_one(lineup)
                moves_per_round.append(moves_dicts)
                payoffs.merge_from(local)
        else:
            with ThreadPoolExecutor(max_workers=self.matchup_workers) as ex:
                futures = {
                    ex.submit(play_one, lineup): lineup for lineup in combo_iter
                }
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    leave=False,
                    position=1,
                    desc="Reputation matches",
                ):
                    moves_dicts, local = fut.result()
                    moves_per_round.append(moves_dicts)
                    payoffs.merge_from(local)

        LOGGER.log_record(record=moves_per_round, file_name=self.record_file)

        # Update reputation score at the end of each round
        for players_moves in moves_per_round:
            for move in players_moves:
                self._update_reputation(move["name"], move["action"])

    @abstractmethod
    def _update_reputation(self, name: str, action: str) -> None:
        """Update the reputation of a player based on their action."""
        raise NotImplementedError


class ReputationPrisonersDilemma(Reputation):
    """
    Reputation mechanism for the Prisoner's Dilemma game.
    This mechanism tracks the cooperation rates of players.
    """

    def __init__(self, base_game: PrisonersDilemma, num_rounds: int, discount: float):
        super().__init__(base_game, num_rounds, discount)
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
                    f"{name} played {coop_tok} in {coop_count}/{total_count} rounds ({coop_pct:.2%})"
                )
        lines = [f"\n\t{line}" for line in lines]
        return (
            "\nReputation:"
            + "".join(lines)
            + "\n\tNote: Your chosen action will affect your reputation score."
        )

    def _update_reputation(self, name: str, action: str) -> None:
        self.reputation[name].record(
            "cooperation_rate",
            action == PrisonersDilemmaAction.COOPERATE.value,
        )


class ReputationPublicGoods(Reputation):
    """
    Reputation mechanism for the Public Goods game.
    This mechanism tracks the contribution rates of players.
    """

    def __init__(self, base_game: Game, num_rounds: int, discount: float):
        super().__init__(base_game, num_rounds=num_rounds, discount=discount)
        if not isinstance(self.base_game, PublicGoods):
            raise TypeError(
                f"ReputationPublicGoods can only be used with PublicGoodsGame, "
                f"but got {self.base_game.__class__.__name__}"
            )

    def _format_reputation(self, agents: Sequence[Agent]) -> str:
        """Format the reputation information of the given agents into a string."""
        lines = []
        coop_tok = PrisonersDilemmaAction.COOPERATE.to_token()

        for agent in agents:
            name = agent.name
            agent_reputation = self.reputation[name]

            coop_rate = agent_reputation.rate("contribution_rate")

            # Initial reputation information at game start
            if coop_rate is None:
                lines.append(f"{name}: No reputation data available.")
                continue
            else:
                coop_count, total_count = agent_reputation.stat("contribution_rate")
                coop_pct = coop_rate
                lines.append(
                    f"They chose {coop_tok} in {coop_count}/{total_count} rounds ({coop_pct:.2%})"
                )
        lines = [f"\n\t\t{line}" for line in lines]
        return (
            "\n\tReputation:"
            + "".join(lines)
            + "\n\t\tNote: Your chosen action will affect your reputation score."
        )

    def _update_reputation(self, name: str, action: str) -> None:
        self.reputation[name].record(
            "contribution_rate",
            action == PublicGoodsAction.CONTRIBUTE.value,
        )
