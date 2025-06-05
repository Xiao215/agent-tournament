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

from src.agent import Agent, IOAgent, CoTAgent
from src.games.prisoner_dilemma import IteratedPrisonersDilemma

from config import MODEL_WEIGHTS_DIR, CONFIG_DIR, OUTPUTS_DIR

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


    agent_classes = {
        "IOAgent": IOAgent,
        "CoTAgent": CoTAgent,
    }
    agent_class = agent_classes.get(agent_config['type'])

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
    if args.log:
        now = datetime.now()
        date_path = now.strftime("%Y/%m/%d")
        time_stamp = now.strftime("%H-%M")
        log_path = Path(OUTPUTS_DIR / "IPD" / date_path / f"{time_stamp}.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger("IPD")
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)


    for _, (agent1_config, agent2_config) in enumerate(
        itertools.combinations(config["agents"], 2)
    ):
        agent1 = build_huggingface_agent(
            agent1_config,
        )
        agent2 = build_huggingface_agent(
            agent2_config,
        )

        env = IteratedPrisonersDilemma(
            agent1=agent1,
            agent2=agent2,
            logger=logger,
            **config["game"]
        )

        # Run the game
        env.play()

        # Free GPU memory
        del agent1
        del agent2
        del env.agent1
        del env.agent2

        torch.cuda.empty_cache()
        gc.collect()


    if logger:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()
