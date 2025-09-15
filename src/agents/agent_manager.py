from abc import ABC, abstractmethod

from src.agents.base import LLM
from src.agents.client_api_llm import ClientAPILLM
from src.agents.hf_llm import HFInstance


class LLMManager:
    """A class to manage a Hugging Face LLM pipeline that can be moved between CPU and GPU."""

    def __init__(self) -> None:
        self.llms = dict()

    def get_llm(self, model_name: str, provider: str) -> LLM:
        """Get an LLM instance for the given model name."""
        if model_name not in self.llms:
            if provider == "HFInstance":
                self.llms[model_name] = HFInstance(model_name)
            else:
                # Default to OpenAI API based models
                self.llms[model_name] = ClientAPILLM(model_name, provider)
        return self.llms[model_name]


class Agent(ABC):
    """
    Abstract base class for an LLM-based agent.
    """

    llm_manager = LLMManager()

    def __init__(self, llm_config: dict) -> None:
        self.model_type = llm_config["model"]
        self.kwargs = llm_config.get("kwargs", {})
        self.pipeline = type(self).llm_manager.get_llm(
            self.model_type, llm_config["provider"]
        )

    @abstractmethod
    def chat(
        self,
        messages: str,
    ) -> str:
        """Chat with the agent using the provided messages."""
        raise NotImplementedError

    def invoke(self, messages: str) -> str:
        """Invoke the agent using the provided messages. No prompting added."""
        response = self.pipeline.invoke(messages, **self.kwargs)
        return response

    def __str__(self):
        return self.name

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the agent."""
        raise NotImplementedError


class IOAgent(Agent):
    """Input/Output Agent.
    This agent is designed to be the most basic llm agent. Given a message, answer it.
    """

    def chat(
        self,
        messages: str,
    ) -> str:
        """Chat with the agent using the provided messages."""
        messages += (
            "\nPlease ONLY provide the output to the above question."
            "DO NOT provide any additional text or explanation.\n"
        )
        response = self.pipeline.invoke(messages, **self.kwargs)
        return response

    @property
    def name(self) -> str:
        """Return the name of the agent."""
        return f"{self.model_type}(IO)"


class CoTAgent(Agent):
    """Chain-of-Thought Agent.

    This agent wraps the prompt to ask the LLM to think step-by-step.
    """

    def chat(
        self,
        messages: str,
    ) -> str:
        """Chat with the agent using the provided messages."""
        messages += """
        Think about the question step by step, break it down into small steps, explain your reasoning, and then provide the final answer.
        """
        response = self.pipeline.invoke(messages, **self.kwargs)
        return response

    @property
    def name(self) -> str:
        """Return the name of the agent."""
        return f"{self.model_type}(CoT)"
