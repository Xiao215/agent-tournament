import argparse
from pathlib import Path
from datetime import datetime

import yaml

from config import CONFIG_DIR, OUTPUTS_DIR

from src.evolution.replicator_dynamics import DiscreteReplicatorDynamics
from src.registry import GAME_REGISTRY, MECHANISM_REGISTRY
from src.plot import plot_probability_evolution

def load_config(path: str) -> dict:
    """
    Load and parse a YAML configuration file.
    """
    path = Path(CONFIG_DIR / path)
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} not found.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    """
    Build agents and run pairwise IPD matches.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    # parser.add_argument('--log', action='store_true', help='Enable logging')

    args = parser.parse_args()

    config = load_config(args.config)


    game_class = GAME_REGISTRY.get(config["game"]["type"])
    mechanism_class = MECHANISM_REGISTRY.get(config["mechanism"]["type"])

    game = game_class(
        **config["game"].get("kwargs", {})
    )
    mechanism = mechanism_class(
        base_game=game,
        **config["mechanism"].get("kwargs", {})
    )

    replicator_dynamics = DiscreteReplicatorDynamics(
        agent_cfgs=config["agents"],
        mechanism=mechanism,
    )

    # TODO: currently initial_population can only be a string, rather than a dynamic population
    population_history, _, _ = replicator_dynamics.run_dynamics(
        initial_population=config["evolution"]["initial_population"],
        steps=config["evolution"]["steps"]
    )

    plot_probability_evolution(
        trajectory=population_history,
        labels=[agent['llm']['model'] for agent in config['agents']],
    )

if __name__ == "__main__":
    main()
