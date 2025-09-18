"""Abstract interfaces shared by concrete LLM backends."""

from typing import Any

from abc import ABC, abstractmethod

class LLM(ABC):
    """Abstract base class for an LLM pipeline that can be moved between CPU and GPU."""

    @abstractmethod
    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Invoke the model with the given prompt."""
        raise NotImplementedError
