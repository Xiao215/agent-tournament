from src.agent import Agent, CoTAgent, IOAgent
from src.games.prisoners_dilemma import PrisonersDilemma
from src.games.public_goods import PublicGoods
from src.mechanisms.base import NoMechanism
from src.mechanisms.repetition import Repetition
from src.mechanisms.reputation import ReputationPrisonersDilemma, ReputationPublicGoods

GAME_REGISTRY = {"PrisonersDilemma": PrisonersDilemma, "PublicGoods": PublicGoods}

MECHANISM_REGISTRY = {
    "ReputationPrisonersDilemma": ReputationPrisonersDilemma,
    "NoMechanism": NoMechanism,
    "ReputationPublicGoods": ReputationPublicGoods,
    "Repetition": Repetition,
}

AGENT_REGISTRY = {
    "IOAgent": IOAgent,
    "CoTAgent": CoTAgent,
}

def create_agent(agent_config: dict) -> Agent:
    agent_class = AGENT_REGISTRY.get(agent_config['type'])

    if agent_class is None:
        raise ValueError(f"Unknown agent type: {agent_config['type']}")

    agent = agent_class(llm_config=agent_config['llm'])

    return agent
