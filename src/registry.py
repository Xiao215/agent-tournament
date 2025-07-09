from src.mechanisms.repetition import Repetition
from src.mechanisms.base import NoMechanism
from src.mechanisms.reputation import ReputationPrisonersDilemma
from src.games.prisoners_dilemma import PrisonersDilemma
from src.agent import IOAgent, CoTAgent


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
