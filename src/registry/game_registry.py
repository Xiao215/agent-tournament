from src.games.prisoners_dilemma import PrisonersDilemma
from src.games.public_goods import PublicGoods
from src.games.travellers_dilemma import TravellersDilemma

GAME_REGISTRY = {
    "PrisonersDilemma": PrisonersDilemma,
    "PublicGoods": PublicGoods,
    "TravellersDilemma": TravellersDilemma,
}
