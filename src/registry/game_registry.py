from src.games.prisoners_dilemma import PrisonersDilemma
from src.games.public_goods import PublicGoods
from src.games.travellers_dilemma import TravellersDilemma
from src.games.trust_game import TrustGame

GAME_REGISTRY = {
    "PrisonersDilemma": PrisonersDilemma,
    "PublicGoods": PublicGoods,
    "TravellersDilemma": TravellersDilemma,
    "TrustGame": TrustGame,
}
