import math
from typing import Any, Sequence

from src.games.base import Move

import numpy as np


class PopulationPayoffs:
    """A class to manage and compute payoffs for a population of agents."""

    def __init__(
        self, agent_names: Sequence[str], *, discount: float | None = None
    ) -> None:
        self.agent_names = list(agent_names)
        self.k = len(self.agent_names)
        self._idx = {t: i for i, t in enumerate(self.agent_names)}
        self.discount = discount if discount is not None else 1.0

        # Store list of payoff vectors
        self._table: dict[tuple[int, ...], np.ndarray] = {}

    def _players_to_counts(self, players: Sequence[str]) -> tuple[int, ...]:
        """Convert player names to counts (tuple of count)."""
        counts = np.zeros(self.k, dtype=int)
        for name in players:
            counts[self._idx[name]] += 1
        aaa = tuple(int(c) for c in counts)
        return aaa

    def _counts_to_players(self, counts: Sequence[int]) -> list[str]:
        """Convert counts (list of count) back to player names.
        List of player names are sorted by the order in self.agent_names.
        """
        out = []
        for i, c in enumerate(counts):
            if c > 0:
                out.extend([self.agent_names[i]] * int(c))
        return out

    def _discounted_average(self, payoffs: np.ndarray) -> np.ndarray:
        """
        Apply geometric discounting over the payoff sequence.
        Return the discounted average payoff vector for all players.
        """
        n = len(payoffs)
        cumsum_ave_payoffs = np.cumsum(payoffs, axis=0) / np.arange(1, n + 1)[:, None]

        x = self.discount
        if x < 1.0:
            weights = np.array([(1 - x) * (x**i) for i in range(n)])
            weights[-1] = x ** (n)  # last one special
        else:
            weights = np.ones(n) / n
        # weights already sum to 1, so no need to normalize
        return np.sum(weights[:, None] * cumsum_ave_payoffs, axis=0)

    def reset(self) -> None:
        """Reset the payoff table to be empty."""
        self._table.clear()

    def merge_from(self, other: "PopulationPayoffs") -> None:
        """
        Merge another PopulationPayoffs into this one by concatenating rounds per lineup key.
        Assumes same agent_names ordering and discount.
        """
        if self.agent_names != other.agent_names:
            raise ValueError("Cannot merge payoffs with different agent sets/order")
        if not math.isclose(self.discount, other.discount):
            raise ValueError("Cannot merge payoffs with different discounts")
        for key, arr in other._table.items():
            if key not in self._table:
                self._table[key] = arr.copy()
            else:
                self._table[key] = np.vstack([self._table[key], arr])

    def add_profile(self, moves: list[Move]) -> None:
        """
        Add one observed round for a lineup that may contain duplicates, e.g. ["A","A","B"].

        `payoffs` is per-type (interpreted per seat for that type). Any types not present
        in `payoffs` are assumed to have payoff 0 for the round.
        """
        key = self._players_to_counts([m.name for m in moves])

        vec = np.zeros(self.k, dtype=float)
        for m in moves:
            vec[self._idx[m.name]] = float(m.points)

        if key not in self._table:
            self._table[key] = vec[None, :]  # (1, k)
        else:
            # Stack against the 1st dimension for multiple rounds profiles
            self._table[key] = np.vstack([self._table[key], vec[None, :]])

    def fitness(self, population: np.ndarray) -> np.ndarray:
        """
        Expected per-type payoff with weight = ∏ p_i^{count_i} for each stored counts key.
        Note: This is *not* multiplied by any multinomial coefficient.
        """
        if population.shape != (self.k,):
            raise ValueError(
                f"population must be shape ({self.k},), got {population.shape}"
            )
        s = population.sum()
        if not np.isclose(s, 1.0):
            raise ValueError(f"Total population must sum to 1.0, got {s}")

        fitness = np.zeros(self.k, dtype=float)
        for counts, rounds in self._table.items():
            counts_arr = np.asarray(counts, dtype=int)
            if np.any((population == 0) & (counts_arr > 0)):
                weight = 0.0
            else:
                logprod = np.sum(
                    counts_arr * np.log(np.where(population > 0, population, 1.0))
                )
                comb_factor = math.factorial(sum(counts)) / np.prod(
                    [math.factorial(c) for c in counts_arr]
                )
                weight = comb_factor * math.exp(logprod)

            if weight == 0.0:
                continue

            avg_profile = self._discounted_average(rounds)  # (k,)
            fitness += weight * avg_profile
        return fitness

    def to_record(self) -> dict[str, Any]:
        """
        JSON-serializable snapshot.

        - expected_payoff: dict[name -> payoff] computed as fitness(population=all ones).
        """
        profiles = []
        for counts, arr in self._table.items():
            players_list = self._counts_to_players(counts)
            present_idxs = [
                self._idx[name]
                for name in sorted(set(players_list), key=lambda n: self._idx[n])
            ]

            profiles.append(
                {
                    "players": players_list,
                    "rounds": arr[:, present_idxs].tolist(),
                    "discounted_average": {
                        self.agent_names[i]: float(self._discounted_average(arr)[i])
                        for i in present_idxs
                    },
                }
            )

        expected_payoff = self.fitness(np.ones(self.k, dtype=float) / self.k)

        return {
            "discount": float(self.discount),
            "profiles": profiles,
            "expected_payoff": {
                name: float(v)
                for name, v in zip(self.agent_names, expected_payoff.tolist())
            },
        }
