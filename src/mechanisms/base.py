import itertools
import math
import random
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Any, Sequence

from tqdm import tqdm

from src.agents.agent_manager import Agent
from src.evolution.population_payoffs import PopulationPayoffs
from src.games.base import Game


class Mechanism(ABC):

    def __init__(self, base_game: Game):
        self.base_game = base_game

        self.record_file = (
            f"{self.__class__.__name__}_{self.base_game.__class__.__name__}.jsonl"
        )
        self.matchup_workers = 1

    def _build_payoffs(self, agent_names: list[str]) -> PopulationPayoffs:
        return PopulationPayoffs(agent_names=agent_names)

    def run_tournament(self, agents: Sequence[Agent]) -> PopulationPayoffs:
        """Run the mechanism over the base game across all players."""
        payoffs = self._build_payoffs(agent_names=[agent.name for agent in agents])

        k = self.base_game.num_players
        combo_iter = list(itertools.combinations_with_replacement(agents, k))
        random.shuffle(combo_iter)  # The order does not matter, kept just in case

        if self.matchup_workers <= 1:
            first_duration = None
            with tqdm(
                total=len(combo_iter),
                desc="Tournaments",
                leave=True,
                dynamic_ncols=True,
            ) as pbar:
                for players in combo_iter:
                    matchup = " vs ".join(agent.name for agent in players)
                    pbar.set_postfix_str(matchup, refresh=False)
                    t0 = time.perf_counter()
                    self._play_matchup(players, payoffs)
                    dt = time.perf_counter() - t0
                    if first_duration is None:
                        first_duration = dt
                        # Rough ETA: match-count * per-match duration
                        est_total = dt * len(combo_iter)
                        print(
                            f"[ETA] ~{est_total/60:.1f} min for {len(combo_iter)} matchups (sequential)."
                        )
                    pbar.update(1)
            return payoffs

        # Parallel branch: run independent matchups concurrently
        def run_one(players: Sequence[Agent]) -> tuple[PopulationPayoffs, float]:
            local = self._build_payoffs(agent_names=[agent.name for agent in agents])
            t0 = time.perf_counter()
            self._play_matchup(players, local)
            dt = time.perf_counter() - t0
            return local, dt

        merged = payoffs
        with ThreadPoolExecutor(max_workers=self.matchup_workers) as ex:
            future_map = {
                ex.submit(run_one, players): players for players in combo_iter
            }
            first_dt = None
            with tqdm(
                total=len(future_map),
                desc="Tournaments",
                leave=True,
                dynamic_ncols=True,
            ) as pbar:
                for fut in as_completed(future_map):
                    local, dt = fut.result()
                    if first_dt is None:
                        first_dt = dt
                        waves = (len(future_map) + self.matchup_workers - 1) // self.matchup_workers
                        est_total = first_dt * waves
                        print(
                            f"[ETA] ~{est_total/60:.1f} min for {len(future_map)} matchups with {self.matchup_workers} workers."
                        )
                    merged.merge_from(local)
                    pbar.update(1)
        return merged

    @abstractmethod
    def _play_matchup(
        self, players: Sequence[Agent], payoffs: PopulationPayoffs
    ) -> None:
        """Play match(es) between the given players."""
        raise NotImplementedError


class RepetitiveMechanism(Mechanism):
    """A mechanism that repeats the game multiple times."""

    def __init__(self, base_game: Game, num_rounds: int, discount: float) -> None:
        super().__init__(base_game)
        self.num_rounds = num_rounds
        self.discount = discount

    def _build_payoffs(self, agent_names: list[str]) -> PopulationPayoffs:
        return PopulationPayoffs(agent_names=agent_names, discount=self.discount)


class NoMechanism(Mechanism):
    """A mechanism that does nothing."""

    def _play_matchup(
        self, players: Sequence[Agent], payoffs: PopulationPayoffs
    ) -> Any:
        """Run the base game without any modifications."""
        moves = self.base_game.play(additional_info="None.", players=players)
        payoffs.add_profile(moves)
        return moves
