from abc import ABC, abstractmethod

class Game(ABC):
    # def __init__(self, log: bool = False):
    #     self.log = log

    @abstractmethod
    def play(self):
        """Play the game."""
        raise NotImplementedError

    @abstractmethod
    def _parse_action(self, response: str):
        """Base on the game type, extract the decision made by the llm agent."""
        raise NotImplementedError
