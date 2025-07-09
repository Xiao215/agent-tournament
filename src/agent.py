from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel

class Agent(ABC):
    """
    Abstract base class for an LLM-based agent.
    """

    def __init__(self, llm: BaseChatModel, name: str):
        self.llm = llm
        self.name = name

    @abstractmethod
    def chat(self,
        messages: str,
    ) -> str:
        """Chat with the agent using the provided messages."""
        raise NotImplementedError

    def __str__(self):
        return f"{self.name}({self.__class__.__name__})"

class IOAgent(Agent):
    """Input/Output Agent.
    This agent is designed to be the most basic llm agent. Given a message, answer it.
    """

    def chat(self,
        messages: str,
    ) -> str:
        """Chat with the agent using the provided messages."""
        messages += (
            "\nPlease ONLY provide the action you want to take. "
            "DO NOT provide any additional text or explanation.\n"
            "Action:"
        )
        response = self.llm.invoke(messages)
        return response.content.strip()

class CoTAgent(Agent):
    """Chain-of-Thought Agent.

    This agent wraps the prompt to ask the LLM to think step-by-step.
    """

    def chat(self,
        messages: str,
    ) -> str:
        """Chat with the agent using the provided messages."""
        messages += """
        First reason about the strategy you are taking **step by step, then you must choose one action from legal actions.

        Your output must be in the following format strictly:

        Thought:
        Your thought.

        Action:
        Your action wrapped in angles brackets, for example: <Action>
        """
        response = self.llm.invoke(messages)
        return response.content.strip()
