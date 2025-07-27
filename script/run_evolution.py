import argparse
from pathlib import Path

import yaml

from config import CONFIG_DIR
from src.evolution.replicator_dynamics import DiscreteReplicatorDynamics
from src.plot import plot_probability_evolution
from src.registry import GAME_REGISTRY, MECHANISM_REGISTRY, create_agent


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
    parser.add_argument('--config', type=str)
    # parser.add_argument('--log', action='store_true', help='Enable logging')

    args = parser.parse_args()

    config = load_config(filename=args.config)

    game_class = GAME_REGISTRY[config["game"]["type"]]
    mechanism_class = MECHANISM_REGISTRY[config["mechanism"]["type"]]

    game = game_class(
        **config["game"].get("kwargs", {})
    )
    mechanism = mechanism_class(
        base_game=game,
        **config["mechanism"].get("kwargs", {})
    )

    agents = [create_agent(agent_cfg) for agent_cfg in config["agents"]]

    replicator_dynamics = DiscreteReplicatorDynamics(
        agents=agents,
        mechanism=mechanism,
    )

    # TODO: currently initial_population can only be a string, rather than a dynamic population
    population_history, _, _ = replicator_dynamics.run_dynamics(
        initial_population=config["evolution"]["initial_population"],
        steps=config["evolution"]["steps"]
    )

    plot_probability_evolution(
        trajectory=population_history,
        labels=[agent.name for agent in agents],
    )

if __name__ == "__main__":
    main()
