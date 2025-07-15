from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

from config import MODEL_WEIGHTS_DIR

# Suppress HF warnings and progress bars
hf_logging.set_verbosity_error()
# hf_logging.disable_progress_bar()


class LLMManager():
    """A class to manage a Hugging Face LLM pipeline that can be moved between CPU and GPU."""
    def __init__(self, model_name: str) -> None:
        # self.name = agent_config['llm']['name']
        # self.model = None
        # self.tokenizer = None

        model_path = MODEL_WEIGHTS_DIR /  model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,      # use float16 for better performance on GPU
            device_map="auto",              # automatically shards layers across GPU/CPU
            offload_folder="hf_offload",
            offload_state_dict=True
        )

        """Use the following for models that should live in GPU memory.
        dispatch_model(
            model,
            device_map="auto",         # now re-shard layers to maximize GPU use
            offload_folder=None,       # disable further offload
            offload_state_dict=False
        )
        """

    def invoke(self, prompts: str | list[str]) -> str | list[str]:
        """Invoke the LLM with the given prompts."""
        single = isinstance(prompts, str)
        batch   = [prompts] if single else prompts

        inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs)

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded[0] if single else decoded


class Agent(ABC):
    """
    Abstract base class for an LLM-based agent.
    """

    def __init__(self, llm_config: dict) -> None:
        self.name = llm_config['model']
        self.llm = LLMManager(self.name)

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
        return response

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
        return response
