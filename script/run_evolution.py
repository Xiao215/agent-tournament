import argparse
import logging
from pathlib import Path
from datetime import datetime

import yaml

from config import CONFIG_DIR, OUTPUTS_DIR

from src import plot
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
    parser.add_argument('--log', action='store_true', help='Enable logging')

    args = parser.parse_args()

    config = load_config(args.config)

    # Set up logging
    logger = None
    debugger = None

    game_class = GAME_REGISTRY.get(config["game"]["type"])
    mechanism_class = MECHANISM_REGISTRY.get(config["mechanism"]["type"])

    if args.log:
        now = datetime.now()
        date_path = now.strftime("%Y/%m/%d")
        time_stamp = now.strftime("%H-%M")
        logger_path = Path(OUTPUTS_DIR / date_path / f"{time_stamp}_result.log")
        debugger_path = Path(OUTPUTS_DIR / date_path / f"{time_stamp}_debug.log")
        logger_path.parent.mkdir(parents=True, exist_ok=True)

        # Logger is used for dense game outcomes while
        # debugger is used for logging all the game actions, reasonings and etc.
        logger = logging.getLogger("game_results")
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(logger_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)


        debugger = logging.getLogger("game_log")
        debugger.setLevel(logging.INFO)
        debugger_handler = logging.FileHandler(debugger_path, encoding="utf-8")
        debugger_handler.setFormatter(logging.Formatter("%(message)s"))
        debugger.addHandler(debugger_handler)

    game = game_class(
        debugger=debugger,
        **config["game"].get("kwargs", {})
    )
    mechanism = mechanism_class(
        base_game=game,
        logger=logger,
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
        trajectory=population_history
    )



    if logger:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()
