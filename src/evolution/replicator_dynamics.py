import itertools
from typing import Any, Sequence
import math

import numpy as np
from tqdm import tqdm

from src.mechanisms.base import Mechanism
from src.registry import create_agent



# Let us require that the payoff tensor is symmetric, that is, payoff_pl_i[k-th agent_type for player 1, ..., l-th agent_type for player i, ...] = payoff_pl_1[l-th agent_type for player 1, ..., k-th agent_type for player i, ...]. Hence, we only need to keep track of one tensor (of player 1).
class PopulationPayoffs:
    """
    n-player game payoffs.
    """

    def __init__(self, agent_types: list[str]) -> None:
        self.agent_types = list(agent_types)
        # profile_key: sorted tuple of length n -> np.ndarray of length k
        self._table: dict[tuple[str, ...], np.ndarray] = {}

    def reset(self) -> None:
        """Clear all stored payoffs."""
        self._table.clear()

    def _normalize(self, profile: list[str]) -> tuple[str, ...]:
        """
        Sorts the profile to make order irrelevant.
        """
        return tuple(sorted(profile))

    def add_profile_payoffs(self, payoffs: dict[str, float]) -> None:
        """
        Add the payoffs from one combination of agent game results.
        """
        profile = list(payoffs.keys())
        key = self._normalize(profile)

        # build payoff vector aligned with agent_types
        k = len(self.agent_types)
        vector = np.zeros(k, dtype=float)
        for name, value in payoffs.items():
            try:
                idx = self.agent_types.index(name)
            except ValueError as exc:
                raise KeyError(f"Unknown agent type: {name}") from exc
            vector[idx] = value

        self._table[key] = vector

    def expected_payoffs(self, population: Sequence[float]) -> np.ndarray:
        k = len(self.agent_types)
        print(self._table)

        # build a quick name→index map for lookups
        idx_map = {name: i for i, name in enumerate(self.agent_types)}

        expected = np.zeros(k, dtype=float)
        for profile_key, vector in self._table.items():
            # profile_key is a tuple of n *distinct* names
            idxs = [idx_map[name] for name in profile_key]
            # just multiply the probabilities for those names
            prob = np.prod(population[idxs])
            # accumulate the weighted payoff vector
            expected += prob * vector

        return expected


class DiscreteReplicatorDynamics:
    """
    Discrete-time replicator dynamics using exponential weight updates.

    This implements the update rule:
    x_i(t+1) = x_i(t) * exp(η * (f_i - f_avg)) / Z(t)

    where η is the learning rate and Z(t) is the normalization constant. For learning rate going to zero, this approaches the continuous-time replicator dynamics.
    """
    def __init__(
        self,
        agent_cfgs: list[dict[str, Any]],
        mechanism: Mechanism,
        population_payoffs: PopulationPayoffs | None = None,
        payoffs_updating: bool =False
    ) -> None:
        self.agent_cfgs = agent_cfgs
        self.mechanism = mechanism

        self.population_payoffs = (
            population_payoffs
            if population_payoffs is not None
            else PopulationPayoffs(agent_types=[config['llm']['model'] for config in agent_cfgs])
        )

        if payoffs_updating:
            raise NotImplementedError("Payoff updates in between the dynamics is not implemented yet!")

    def population_update(
        self,
        current_pop: np.ndarray,
        expected_payoffs: np.ndarray,
        weighted_mean_payoff: float,
        lr: float
    ) -> np.ndarray:
        """
        Args:
            current_dist: numpy array, current probability distribution over agent types
            fitness: numpy array, fitness of each agent type against current distribution
            avg_fitness: float, current average performance
            t: int, current time step (must be > 0)

        Returns:
            numpy array, next step's probability distribution over agent types
        """
        weights = current_pop * np.exp(lr * (expected_payoffs - weighted_mean_payoff))
        next_pop = weights / np.sum(weights)

        return next_pop

    def run_dynamics(
        self,
        initial_population: np.ndarray | str = "uniform",
        steps: int = 1000,
        tol: float = 1e-6,
        learning_rate: dict[str, float | str] | None = None
    ):
        """
        Run the multiplicative weights dynamics for a specified number of steps.
        """
        if learning_rate is None:
            learning_rate = {"method": "constant", "nu": 0.1}

        # Initialize learning rate function
        if learning_rate["method"] == "constant":
            lr_fct = lambda t: learning_rate["nu"]
        elif learning_rate["method"] == "sqrt":
            lr_fct = lambda t: learning_rate["nu"] / np.sqrt(t)
        else:
            raise ValueError("learning_rate method must be 'constant' or 'sqrt'")

        # Initialize population distribution
        if isinstance(initial_population, np.ndarray):
            assert len(initial_population) == len(self.population_payoffs.agent_types), "Initial population distribution must match number of agent types"
            assert np.all(initial_population >= 0), "Initial population distribution must be non-negative"
            population = initial_population
        elif initial_population == "random":
            population = np.random.exponential(scale=1.0, size=len(self.population_payoffs.agent_types))
        elif initial_population == "uniform":
            population = np.ones(len(self.population_payoffs.agent_types))
        else:
            raise ValueError("initial_population must be a numpy array or 'uniform'")

        # Normalize to ensure it is a probability distribution
        population /= population.sum()

        # Run the dynamics
        population_history = [population.copy()]
        payoff_history = []

        n = len(self.agent_cfgs)
        k = self.mechanism.base_game.num_players
        total_matches = math.comb(n, k)
        for _ in tqdm(range(steps), desc="Evolution Steps"):
            agents = [create_agent(cfg) for cfg in self.agent_cfgs]
            combo_iter = itertools.combinations(agents, k)
            inner_tqdm_bar = tqdm(
                combo_iter,
                desc="Tournaments",
                total=total_matches,
                leave=False,
                position=1,
            )
            for agents in combo_iter:
                names = [agent.name for agent in agents]
                inner_tqdm_bar.set_postfix(match=" vs ".join(names))

                tournament_payoffs = self.mechanism.run(
                    agents=agents,
                )
                self.population_payoffs.add_profile_payoffs(tournament_payoffs)

            expected_payoffs = self.population_payoffs.expected_payoffs(population)
            weighted_mean_payoff = np.dot(population, expected_payoffs)
            payoff_history.append(weighted_mean_payoff)

            if self.mechanism.logger:
                self.mechanism.logger.info(
                    f"Population: {population}, "
                    f"Expected Payoffs: {expected_payoffs}, "
                    f"Weighted Mean Payoff: {weighted_mean_payoff}"
                )

            if np.max(np.abs(expected_payoffs - weighted_mean_payoff)) < tol:
                print("Converged: approximate equilibrium reached")
                status = "converged: approximate equilibrium reached"
                return population_history, payoff_history, status

            population = self.population_update(
                current_pop=population,
                expected_payoffs=expected_payoffs,
                weighted_mean_payoff=weighted_mean_payoff,
                lr=lr_fct(len(population_history) + 1)
            )
            population_history.append(population.copy())
            self.population_payoffs.reset()

        # TODO, improve the logger so it is not so hardcoded
        if self.mechanism.logger:
            self.mechanism.logger.info(
                '-' * 50 + '\n' +
                f"Step {len(population_history)}: "
                f"Population: {population}, "
                f"Expected Payoffs: {expected_payoffs}, "
                f"Weighted Mean Payoff: {weighted_mean_payoff}"
                + '\n' +
                '-' * 50
            )

        status = "steps limit reached"
        print("Steps limit reached")
        # TODO: improve this return statements
        return population_history, payoff_history, status
