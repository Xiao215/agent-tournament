import gc
import argparse
import itertools
import logging
from typing import Any
from pathlib import Path
from datetime import datetime

import torch
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import yaml

from src.games.base import game_registry
from src.agent import agent_registry, Agent
from src.mechanisms.base import mechanism_registry, NoMechanism
from config import MODEL_WEIGHTS_DIR, CONFIG_DIR, OUTPUTS_DIR


# Temporary fix for registry issue
from src.mechanisms.repetition import Repetition
from src.games.prisoner_dilemma import PrisonersDilemma
from src.agent import IOAgent, CoTAgent
game_registry = {"PrisonersDilemma": PrisonersDilemma}
mechanism_registry = {"Repetition": Repetition, "NoMechanism": NoMechanism}
agent_registry = {
    "IOAgent": IOAgent,
    "CoTAgent": CoTAgent,
}

def load_config(path: str) -> dict:
    """
    Load and parse a YAML configuration file.
    """
    path = Path(CONFIG_DIR / path)
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} not found.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_huggingface_agent(
    agent_config: dict[str, Any],
) -> list[Agent]:
    """
    Instantiate an LLM-based Agent using HuggingFace pipeline.
    """
    model_path = MODEL_WEIGHTS_DIR / agent_config['llm']['model']

    llm_kwargs = agent_config['llm'].get("kwargs", {})

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        **llm_kwargs
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    chat_model = ChatHuggingFace(llm=llm, model_id=str(model_path))



    agent_class = agent_registry.get(agent_config['type'])

    if agent_class is None:
        raise ValueError(f"Unknown agent type: {agent_config['type']}")

    agent = agent_class(name=agent_config['llm']['model'], llm=chat_model)

    return agent

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

    game = game_registry.get(config["game"]["type"])
    mechanism = mechanism_registry.get(config["mechanism"]["type"], NoMechanism)

    if args.log:
        now = datetime.now()
        date_path = now.strftime("%Y/%m/%d")
        time_stamp = now.strftime("%H-%M")
        logger_path = Path(OUTPUTS_DIR / game.__class__.__name__ / date_path / f"{time_stamp}_result.log")
        debugger_path = Path(OUTPUTS_DIR / game.__class__.__name__ / date_path / f"{time_stamp}_debug.log")
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


    for _, agents_config in enumerate(
        itertools.combinations(config["agents"], game.num_players)
    ):
        agents = [build_huggingface_agent(
            config,
        ) for config in agents_config]

        # TODO: seperate agents from game
        game_instance = game(debugger=debugger, agents=agents, **config["game"]["kwargs"])
        game_with_mechanism = mechanism(
            base_game=game_instance,
            logger=logger,
            **config["mechanism"]["kwargs"]
        )

        game_with_mechanism.run()

        # Free GPU memory
        for agent in agents:
            del agent

        torch.cuda.empty_cache()
        gc.collect()


    if logger:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()
