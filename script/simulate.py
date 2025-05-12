import gc
import argparse
import itertools
from typing import Any
from pathlib import Path

import torch
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import yaml

from src.agent import Agent, BaseAgent, CodeStrategyAgent
from src.game import IteratedPrisonersDilemma
from src.action import Action

from config import MODEL_WEIGHTS_DIR

def load_config(path: str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} not found.")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_huggingface_agent(
    agent_config: dict[str, Any],
    agent_type: str,
    rule: str,
    gpu_device_num: int,
) -> list[Agent]:
    model_path = MODEL_WEIGHTS_DIR / agent_config['model']

    kwargs = agent_config.get("kwargs", {})

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=gpu_device_num,
        **kwargs
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    chat_model = ChatHuggingFace(llm=llm, model_id=str(model_path))


    agent = None
    if agent_type == "BaseAgent":
        agent = BaseAgent(chat_model, agent_config['model'], rule)
    elif agent_type == "CodeStrategyAgent":
        agent = CodeStrategyAgent(chat_model, agent_config['model'], rule)
    else:
        raise ValueError(f'Agent type name `{agent_type}` is not allowed')

    return agent

def build_prisoners_dilemma(
    raw_payoff: dict[str, float],
    num_rounds: int,
    log: bool
) -> IteratedPrisonersDilemma:
    payoff = {}
    for actions, reward in raw_payoff.items():
        assert len(actions) == 2, f"Invalid payoff actions: {actions}"
        a1 = Action(actions[0])
        a2 = Action(actions[1])
        payoff[(a1, a2)] = tuple(reward)

    prisoners_dilemma_env = IteratedPrisonersDilemma(payoff, num_rounds, log)

    return prisoners_dilemma_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/toy.yaml')
    parser.add_argument('--log', action='store_true', help='Enable logging')

    args = parser.parse_args()

    config = load_config(args.config)
    env = build_prisoners_dilemma(
        raw_payoff=config["payoff_matrix"],
        num_rounds=config["simulation"]["num_rounds"],
        log=args.log
    )

    for i, (agent1_config, agent2_config) in enumerate(
        itertools.combinations(config["llm"]["agents"], 2)
    ):
        num_gpus = torch.cuda.device_count()
        gpu_num1 = 0
        gpu_num2 = 0 if num_gpus == 1 else 1
        agent1 = build_huggingface_agent(
            agent1_config,
            config["simulation"]["agent_type"],
            env.get_rule(),
            gpu_num1
        )
        agent2 = build_huggingface_agent(
            agent2_config,
            config["simulation"]["agent_type"],
            env.get_rule(),
            gpu_num2
        )

        env.enroll_agents(agent1, agent2)
        reward1, reward2 = env.simulate()
        print(f'{agent1} vs {agent2} — score: {reward1}, {reward2}')

        # Free GPU memory
        del agent1
        del agent2
        del env.agent1
        del env.agent2

        torch.cuda.empty_cache()
        gc.collect()