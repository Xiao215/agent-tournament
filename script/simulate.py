import gc
import argparse
import itertools
from typing import Any
from pathlib import Path
from collections import defaultdict

import torch
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import yaml

from src.agent import Agent, BaseAgent, CodeStrategyAgent
from src.game import IteratedPrisonersDilemma
from src.action import Action

from config import MODEL_WEIGHTS_DIR, CONFIG_DIR
from src.plot import plot_model_scores

def load_config(path: str) -> dict:

    path = Path(CONFIG_DIR / path)
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} not found.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_huggingface_agent(
    llm_config: dict[str, Any],
    agent_config: dict[str, Any],
    rule: str,
) -> list[Agent]:
    model_path = MODEL_WEIGHTS_DIR / llm_config['model']

    llm_kwargs = llm_config.get("kwargs", {})

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


    AGENT_CLASSES = {
        "BaseAgent": BaseAgent,
        "CodeStrategyAgent": CodeStrategyAgent,
    }
    agent_class = AGENT_CLASSES.get(agent_config['type'])

    if agent_class is None:
        raise ValueError(f"Unknown agent type: {agent_config['type']}")

    agent = agent_class(
        chat_model,
        name=llm_config['model'],
        rule=rule,
        **(agent_config.get("kwargs") or {})
    )

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
    parser.add_argument('--config', type=str)
    parser.add_argument('--log', action='store_true', help='Enable logging')

    args = parser.parse_args()

    config = load_config(args.config)
    env = build_prisoners_dilemma(
        raw_payoff=config["payoff_matrix"],
        num_rounds=config["simulation"]["num_rounds"],
        log=args.log
    )

    model_total_score = defaultdict(int)
    for i, (llm1_config, llm2_config) in enumerate(
        itertools.combinations(config["llm"], 2)
    ):
        agent1 = build_huggingface_agent(
            llm1_config,
            config["simulation"]["agent"],
            env.get_rule(),
        )
        agent2 = build_huggingface_agent(
            llm2_config,
            config["simulation"]["agent"],
            env.get_rule(),
        )

        env.enroll_agents(agent1, agent2)
        reward1, reward2 = env.simulate()
        model_total_score[str(agent1)] += reward1
        model_total_score[str(agent2)] += reward2

        print(f'{agent1} vs {agent2} — score: {reward1}, {reward2}')

        # Free GPU memory
        del agent1
        del agent2
        del env.agent1
        del env.agent2

        torch.cuda.empty_cache()
        gc.collect()

    plot_model_scores(model_total_score)