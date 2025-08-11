from typing import Sequence

import numpy as np


class PopulationPayoffs:
    """
    Simplified: each profile is a set of distinct types.
    _table[key] = (totals, calls)
      - totals: sum of payoffs per type over all observations of this profile
      - calls:  number of times this exact profile was added
    """

    def __init__(self, agent_names: Sequence[str]) -> None:
        self.agent_names = list(agent_names)
        self.k = len(self.agent_names)
        # index mapping name to index in the population vector
        self._idx = {t: i for i, t in enumerate(self.agent_names)}

        # mapping agent names to the payoff vector as well as the number of calls
        # Note, in most cases, if the same agent match up do not repeat again until reset,
        # The value of count is 1.
        self._table: dict[tuple[str, ...], tuple[np.ndarray, int]] = {}

    def reset(self) -> None:
        """Reset the internal state of the payoffs table."""
        self._table.clear()

    def add_profile(
        self,
        agent_names: Sequence[str],
        payoffs: Sequence[float],
    ) -> None:
        """Add a profile of agent names and their corresponding payoffs."""
        if len(agent_names) != len(payoffs):
            raise ValueError("agent_names and payoffs must align")

        # sort so e.g. ("B","A") and ("A","B") map to the same key
        key = tuple(sorted(agent_names))

        vec = np.zeros(self.k, float)
        for t, p in zip(agent_names, payoffs):
            vec[self._idx[t]] = p

        if key in self._table:
            # accumulate payoff if same profile is added again
            totals, calls = self._table[key]
            self._table[key] = (totals + vec, calls + 1)
        else:
            self._table[key] = (vec.copy(), 1)

    def fitness(self, population: np.ndarray) -> np.ndarray:
        """
        Weighted average of payoffs for the given population vector.
        If one match up have multiple profiles, the average is taken.
        """
        fitness = np.zeros(self.k, float)
        for key, (totals, calls) in self._table.items():
            avg_profile = totals / calls
            prob = 1.0
            for t in key:
                prob *= population[self._idx[t]]
            fitness += prob * avg_profile

        return fitness
