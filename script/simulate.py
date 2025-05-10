import argparse
import itertools
from typing import Any
from pathlib import Path

from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import pipeline
import yaml

from src.agent import Agent, BaseAgent, CodeStrategyAgent
from src.game import IteratedPrisonersDilemma
from src.action import Action

from huggingface_hub import login

# login()


def load_config(path: str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} not found.")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_huggingface_agent(
    agent_config: dict[str, Any],
    agent_type: str,
    rule: str
) -> list[Agent]:
    cache_dir = Path(__file__).resolve().parent.parent / 'cache'

    kwargs = agent_config.get("kwargs", {})
    pipe = pipeline(
        "text-generation",
        model=agent_config['model'],
        tokenizer=agent_config['model'],
        device=0,
        model_kwargs={"cache_dir": cache_dir},
        **kwargs
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    chat_model = ChatHuggingFace(llm=llm)


    agent = None
    if agent_type == "BaseAgent":
        agent = BaseAgent(chat_model, rule)
    elif agent_type == "CodeStrategyAgent":
        agent = CodeStrategyAgent(chat_model, rule)
    else:
        raise ValueError(f'Agent type name `{agent_type}` is not allowed')

    return agent

def build_prisoners_dilemma(
    raw_payoff: dict[str, float],
    num_rounds: int
) -> IteratedPrisonersDilemma:
    payoff = {}
    for actions, reward in raw_payoff.items():
        assert len(actions) == 2, f"Invalid payoff actions: {actions}"
        a1 = Action(actions[0])
        a2 = Action(actions[1])
        payoff[(a1, a2)] = tuple(reward)

    prisoners_dilemma_env = IteratedPrisonersDilemma(payoff, num_rounds)

    return prisoners_dilemma_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    env = build_prisoners_dilemma(
        raw_payoff=config["payoff_matrix"],
        num_rounds=config["simulation"]["num_rounds"],
    )

    for i, (agent1_config, agent2_config) in enumerate(
        itertools.combinations(config["llm"]["agents"], 2)
    ):
        agent1 = build_huggingface_agent(agent1_config, config["simulation"]["agent_type"], env.get_rule())
        agent2 = build_huggingface_agent(agent2_config, config["simulation"]["agent_type"], env.get_rule())

        env.enroll_agents(agent1, agent2)
        reward1, reward2 = env.simulate()
        print(f'{agent1.llm.llm.repo_id} vs {agent2.llm.llm.repo_id} — score: {reward1}, {reward2}')