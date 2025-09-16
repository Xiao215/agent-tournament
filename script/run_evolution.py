import argparse
from pathlib import Path
import torch
import numpy as np
import random

import yaml

from config import CONFIG_DIR
from src.evolution.replicator_dynamics import DiscreteReplicatorDynamics
from src.plot import plot_probability_evolution
from src.registry.game_registry import GAME_REGISTRY
from src.registry.agent_registry import create_agent
from src.registry.mechanism_registry import MECHANISM_REGISTRY
from src.logger_manager import WandBLogger, LOGGER


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(filename: str) -> dict:
    """
    Load and parse a YAML configuration file.
    """
    config_path = Path(CONFIG_DIR) / filename
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found.")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    """
    Build agents and run pairwise IPD matches.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    # parser.add_argument('--log', action='store_true', help='Enable logging')
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases figure saving"
    )

    args = parser.parse_args()

    config = load_config(filename=args.config)

    game_class = GAME_REGISTRY[config["game"]["type"]]
    mechanism_class = MECHANISM_REGISTRY[config["mechanism"]["type"]]

    game = game_class(**config["game"].get("kwargs", {}))
    # Extract mechanism kwargs and handle matchup_workers separately to avoid ctor error
    mech_kwargs = (config["mechanism"].get("kwargs", {}) or {}).copy()
    workers = mech_kwargs.pop("matchup_workers", None)
    mechanism = mechanism_class(base_game=game, **mech_kwargs)
    # Optional parallelism across matchups
    if isinstance(workers, int):
        mechanism.matchup_workers = max(1, workers)

    agents = [create_agent(agent_cfg) for agent_cfg in config["agents"]]

    # Record the configuration as JSON
    for i, agent in enumerate(agents):
        # Create the name field to make frontend easier.
        config["agents"][i]["name"] = agent.name
    LOGGER.log_record(config, "config.json")

    print(
        f"Running {config['game']['type']} with mechanism {config['mechanism']['type']}.\n"
        f"Players: {', '.join(agent.name for agent in agents)}"
    )

    replicator_dynamics = DiscreteReplicatorDynamics(
        agents=agents,
        mechanism=mechanism,
    )

    # TODO: currently initial_population can only be a string, rather than a dynamic population
    population_history = replicator_dynamics.run_dynamics(
        initial_population=config["evolution"]["initial_population"],
        steps=config["evolution"]["steps"],
    )

    wb = None
    if args.wandb:
        wb = WandBLogger(project="llm-evolution", config=config)

    plot_probability_evolution(
        trajectory=population_history,
        labels=[agent.name for agent in agents],
        wb=wb,
        save_local=True,
    )


if __name__ == "__main__":
    set_seed()
    main()
