import itertools
from typing import Any, Sequence
import math
import random

import numpy as np
from tqdm import tqdm

from src.mechanisms.base import Mechanism
from src.registry import create_agent

class PopulationPayoffs:
    """
    Stores payoffs compactly by aggregating over *unordered* strategy
    profiles (multisets).  A key like ('A','A','B') represents every
    seating permutation of two A-players and one B-player.
    """

    def __init__(self, agent_types: list[str]) -> None:
        self.agent_types = list(agent_types)
        k = len(agent_types)
        # key -> ( totals[k] , counts[k] )  both float/ints
        self._table: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}

    def reset(self) -> None:
        self._table.clear()

    def _normalize(self, names: Sequence[str]) -> tuple[str, ...]:
        return tuple(sorted(names))

    def add_profile_payoffs(
        self,
        scores: dict[str, float],
    ) -> None:
        names = scores.keys()
        payoffs = scores.values()

        k = len(self.agent_types)
        idx_of = {n: i for i, n in enumerate(self.agent_types)}

        key = self._normalize(names)

        totals = np.zeros(k, float)
        counts = np.zeros(k, int)

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

    def expected_payoffs(self, population: Sequence[float]) -> np.ndarray:
        """
        Expected fitness f_i(x) under random matching of n players
        (n = len(key)) and a population state x.
        """
        k = len(self.agent_types)
        x = np.asarray(population, float)
        if x.shape != (k,):
            raise ValueError("population vector has wrong length")

        expected = np.zeros(k)

        for key, (totals, counts) in self._table.items():
            # probability mass of *this* multiset being drawn
            #   Pr(key) =  n! / (m1! m2! …) ·  Π_j x_j^m_j
            # We can ignore the multinomial front-factor because it
            # cancels when we divide by m_i below (see derivation).
            #
            # m_j = multiplicity of type j in the multiset
            m = np.zeros(k, int)
            for name in key:
                m[self.agent_types.index(name)] += 1
            prob_multiset = np.prod(np.where(m, x**m, 1.0))

            # convert aggregated totals into *per-individual* payoff
            with np.errstate(divide="ignore", invalid="ignore"):
                per_capita = np.where(m, totals / m, 0.0)

            expected += prob_multiset * per_capita

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
        self.mechanism = mechanism
        self.agents = [create_agent(cfg) for cfg in agent_cfgs]
        self.population_payoffs = (
            population_payoffs
            if population_payoffs is not None
            else PopulationPayoffs(agent_types=[str(agent) for agent in self.agents])
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

        n = len(self.agents)
        k = self.mechanism.base_game.num_players
        total_matches = math.comb(n, k)
        for _ in tqdm(range(steps), desc="Evolution Steps"):
            combo_iter = list(itertools.combinations(self.agents, k))
            random.shuffle(combo_iter)
            inner_tqdm_bar = tqdm(
                combo_iter,
                desc="Tournaments",
                total=total_matches,
                leave=False,
                position=1,
            )

            for agents in combo_iter:
                names = [str(agent) for agent in agents]
                inner_tqdm_bar.set_postfix(match=" vs ".join(names))

                tournament_payoffs = self.mechanism.run(
                    agents=agents,
                )
                self.population_payoffs.add_profile_payoffs(tournament_payoffs)

            expected_payoffs = self.population_payoffs.expected_payoffs(population)
            weighted_mean_payoff = np.dot(population, expected_payoffs)
            payoff_history.append(weighted_mean_payoff)

            # if self.mechanism.logger:
            #     self.mechanism.logger.info(
            #         f"Population: {population}, "
            #         f"Expected Payoffs: {expected_payoffs}, "
            #         f"Weighted Mean Payoff: {weighted_mean_payoff}"
            #     )

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
            print(self.population_payoffs._table)
            self.population_payoffs.reset()

        # TODO, improve the logger so it is not so hardcoded
        # if self.mechanism.logger:
        #     self.mechanism.logger.info(
        #         '-' * 50 + '\n' +
        #         f"Step {len(population_history)}: "
        #         f"Population: {population}, "
        #         f"Expected Payoffs: {expected_payoffs}, "
        #         f"Weighted Mean Payoff: {weighted_mean_payoff}"
        #         + '\n' +
        #         '-' * 50
        #     )

        status = "steps limit reached"
        print("Steps limit reached")
        # TODO: improve this return statements
        return population_history, payoff_history, status
