from src.mechanisms.repetition import Repetition
from src.mechanisms.base import NoMechanism
from src.mechanisms.reputation import ReputationPrisonersDilemma
from src.games.prisoners_dilemma import PrisonersDilemma
from src.agent import IOAgent, CoTAgent, Agent


GAME_REGISTRY = {
    "PrisonersDilemma": PrisonersDilemma
}

MECHANISM_REGISTRY = {
    "Repetition": Repetition,
    "ReputationPrisonersDilemma": ReputationPrisonersDilemma,
    "NoMechanism": NoMechanism
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