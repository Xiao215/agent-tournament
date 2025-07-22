from typing import Sequence

import numpy as np


class PopulationPayoffs:
    """
    Stores payoffs compactly by aggregating over *unordered* strategy
    profiles (multisets).  A key like ('A','A','B') represents every
    seating permutation of two A-players and one B-player.
    """

    def __init__(self, agent_types: list[str]) -> None:
        self.agent_types = list(agent_types)
        self.k = len(agent_types)
        # key -> ( totals[k] , counts[k] )  both float/ints
        self._table: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}

    def reset(self) -> None:
        self._table.clear()

    def _normalize(self, names: list[str]) -> tuple[str, ...]:
        return tuple(sorted(names))

    def add_profile_payoffs(
        self,
        scores: dict[str, float],
    ) -> None:
        names = list(scores.keys())
        payoffs = scores.values()

        idx_of = {n: i for i, n in enumerate(self.agent_types)}

        key = self._normalize(names)

        totals = np.zeros(self.k, float)
        counts = np.zeros(self.k, int)

        for name, p in zip(names, payoffs):
            try:
                i = idx_of[name]
            except KeyError:
                raise KeyError(f"Unknown agent type: {name}")
            totals[i] += p
            counts[i] += 1

        # accumulate into storage
        if key in self._table:
            old_tot, old_cnt = self._table[key]
            totals += old_tot
            counts += old_cnt

        self._table[key] = (totals, counts)

    def expected_payoffs(self, population: Sequence[float] | np.ndarray) -> np.ndarray:
        """
        Expected fitness f_i(x) under random matching of n players
        (n = len(key)) and a population state x.
        """
        x = np.asarray(population, float)
        if x.shape != (self.k,):
            raise ValueError("population vector has wrong length")

        expected = np.zeros(self.k)

        for key, (totals, counts) in self._table.items():
            # probability mass of *this* multiset being drawn
            #   Pr(key) =  n! / (m1! m2! …) ·  Π_j x_j^m_j
            # We can ignore the multinomial front-factor because it
            # cancels when we divide by m_i below (see derivation).
            #
            # m_j = multiplicity of type j in the multiset
            m = np.zeros(self.k, int)
            for name in key:
                m[self.agent_types.index(name)] += 1
            prob_multiset = np.prod(np.where(m, x**m, 1.0))

            # convert aggregated totals into *per-individual* payoff
            with np.errstate(divide="ignore", invalid="ignore"):
                per_capita = np.where(m, totals / m, 0.0)

            expected += prob_multiset * per_capita

        return expected
