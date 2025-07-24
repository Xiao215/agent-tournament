import itertools
import math
import os
import random
from datetime import datetime
from typing import Literal

import numpy as np
from tqdm import tqdm

from config import OUTPUTS_DIR
from src.agent import Agent
from src.evolution.population_payoffs import PopulationPayoffs
from src.logging_config import setup_logger
from src.mechanisms.base import Mechanism

now = datetime.now()
log_dir = OUTPUTS_DIR / f"{now.year}" / f"{now.month:02}" / f"{now.day:02}"
os.makedirs(log_dir, exist_ok=True)

logger = setup_logger(
    name="evolution_logger",
    log_file=str(log_dir / f"{now.hour:02}_{now.minute:02}_evolution.log"),
)


class DiscreteReplicatorDynamics:
    """
    Discrete-time replicator dynamics using exponential weight updates.

    This implements the update rule:
    x_i(t+1) = x_i(t) * exp(η * (f_i - f_avg)) / Z(t)

    where η is the learning rate and Z(t) is the normalization constant.
    For learning rate going to zero, this approaches the continuous-time replicator dynamics.
    """
    def __init__(
        self,
        agents: list[Agent],
        mechanism: Mechanism,
        population_payoffs: PopulationPayoffs | None = None,
    ) -> None:
        self.mechanism = mechanism
        self.agents = agents
        self.population_payoffs = (
            population_payoffs
            if population_payoffs is not None
            else PopulationPayoffs(agent_types=[str(agent) for agent in self.agents])
        )

        # Log the initial pipeline setup information
        mech_name = type(self.mechanism).__name__
        base_game_name = type(self.mechanism.base_game).__name__
        header_lines = [
            f"Mechanism: {mech_name}",
            f"Base game: {base_game_name}",
            "Agents:",
        ]
        agent_lines = [f"  - {agent}" for agent in self.agents]
        all_lines = header_lines + agent_lines
        width = max(len(line) for line in all_lines)
        sep = "+" + "-" * (width + 2) + "+"
        content = [f"| {line.ljust(width)} |" for line in all_lines]
        info_box = "\n".join([sep] + content + [sep])
        logger.info("%s\n", info_box)

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
        *,
        lr_method: Literal["constant", "sqrt"] = "constant",
        lr_nu: float = 0.1,
    ):
        """
        Run the multiplicative weights dynamics for a specified number of steps.
        """

        if lr_method == "constant":
            lr_fct = lambda t: lr_nu
        elif lr_method == "sqrt":
            lr_fct = lambda t: lr_nu / np.sqrt(t)
        else:
            raise ValueError("learning_rate method must be 'constant' or 'sqrt'")

        # Initialize population distribution
        if isinstance(initial_population, np.ndarray):
            assert len(initial_population) == len(self.population_payoffs.agent_types), "Initial population distribution must match number of agent types"
            assert np.all(initial_population >= 0), "Initial population distribution must be non-negative"
            population = initial_population
        elif initial_population == "random":
            population = np.random.exponential(
                scale=1.0, size=len(self.population_payoffs.agent_types)
            )
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
        for step in tqdm(range(1, steps + 1), desc="Evolution Steps"):
            logger.info(
                "%s Evolution Step %d/%d %s", "=" * 10, step, steps + 1, "=" * 10
            )
            combo_iter = list(itertools.combinations(self.agents, k))
            random.shuffle(combo_iter)
            inner_tqdm_bar = tqdm(
                combo_iter,
                desc="Tournaments",
                total=total_matches,
                leave=False,
                position=1,
            )

            for agents in inner_tqdm_bar:
                names = [str(agent) for agent in agents]
                match_label = " vs ".join(names)
                inner_tqdm_bar.set_postfix(match=match_label)

                tournament_payoffs = self.mechanism.run(
                    agents=agents,
                )
                self.population_payoffs.add_profile_payoffs(tournament_payoffs)

                payoff_lines = [
                    f"\t\t{name}: {score:.2f}"
                    for name, score in tournament_payoffs.items()
                ]
                payoff_block = "\n".join(payoff_lines)
                logger.info(
                    "\t-> %s:\n%s",
                    match_label,
                    payoff_block,
                )

            expected_payoffs = self.population_payoffs.expected_payoffs(population)
            weighted_mean_payoff = np.dot(population, expected_payoffs)
            payoff_history.append(weighted_mean_payoff)

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
            pop_str = ", ".join(
                f"{str(agent)}: {population[i]:.4f}"
                for i, agent in enumerate(self.agents)
            )

            logger.info("\n\tPopulation distribution at step %d: %s", step, pop_str)
            self.population_payoffs.reset()
            self.mechanism.post_tournament()

        status = "steps limit reached"
        print("Steps limit reached")
        # TODO: improve this return statements
        return population_history, payoff_history, status
