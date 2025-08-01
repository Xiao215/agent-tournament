import itertools
import json
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
from src.mechanisms.base import Mechanism

now = datetime.now()
log_dir = OUTPUTS_DIR / f"{now.year}" / f"{now.month:02}" / f"{now.day:02}"
os.makedirs(log_dir, exist_ok=True)

evolution_json = open(
    log_dir / f"{now.hour:02}{now.minute:02}_evolution.json",
    mode="a",
    encoding="utf-8",
)

random.seed(42)

def log_evolution_record(record: dict) -> None:
    """Log the evolution record to a JSON file."""
    json.dump(record, evolution_json)
    evolution_json.write("\n")
    evolution_json.flush()


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
            else PopulationPayoffs(agent_names=[agent.name for agent in self.agents])
        )

    def population_update(
        self,
        current_pop: np.ndarray,
        fitness: np.ndarray,
        ave_population_fitness: float,
        lr: float,
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
        weights = current_pop * np.exp(lr * (fitness - ave_population_fitness))
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
    ) -> list[np.ndarray]:
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
            assert len(initial_population) == len(
                self.agents
            ), "Initial population distribution must match number of agent types"
            assert np.all(initial_population >= 0), "Initial population distribution must be non-negative"
            population = initial_population
        elif initial_population == "random":
            population = np.random.exponential(scale=1.0, size=len(self.agents))
        elif initial_population == "uniform":
            population = np.ones(len(self.agents))
        else:
            raise ValueError("initial_population must be a numpy array or 'uniform'")

        # Normalize to ensure it is a probability distribution
        population /= population.sum()
        population_history = [population.copy()]

        n = len(self.agents)
        k = self.mechanism.base_game.num_players
        total_matches = math.comb(n, k)
        for step in tqdm(range(1, steps + 1), desc="Evolution Steps"):
            combo_iter = list(itertools.combinations(self.agents, k))
            random.shuffle(combo_iter)
            inner_tqdm_bar = tqdm(
                combo_iter,
                desc="Tournaments",
                total=total_matches,
                leave=False,
                position=1,
            )

            match_records = []
            for agents in inner_tqdm_bar:
                match_label = " vs ".join(agent.name for agent in agents)
                inner_tqdm_bar.set_postfix(match=match_label)

                match_record = self.mechanism.run(
                    agents=agents,
                )

                self.population_payoffs.add_profile(
                    agent_names=[r["name"] for r in match_record],
                    payoffs=[r["points"] for r in match_record],
                )
                match_records.append(match_record)
            inner_tqdm_bar.close()

            fitness = self.population_payoffs.fitness(population)
            # average population fitness is the society's average performance
            ave_population_fitness = np.dot(population, fitness)

            evolution_record = {
                "step": step,
                "stats": [
                    {
                        "name": agent.name,
                        "fitness": fitness[i],
                        "population_fraction": population[i],
                    }
                    for i, agent in enumerate(self.agents)
                ],
                "average_population_fitness": ave_population_fitness,
                "match_records": match_records,
            }
            log_evolution_record(evolution_record)

            if np.max(np.abs(fitness - ave_population_fitness)) < tol:
                print("Converged: approximate equilibrium reached")
                return population_history

            population = self.population_update(
                current_pop=population,
                fitness=fitness,
                ave_population_fitness=ave_population_fitness,
                lr=lr_fct(len(population_history) + 1),
            )
            population_history.append(population.copy())

            self.population_payoffs.reset()
            self.mechanism.post_tournament(match_records)

        print("Steps limit reached")
        return population_history
